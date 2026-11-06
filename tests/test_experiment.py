"""Checks for the illustrative robust CBF experiment."""

import numpy as np
import pytest

import experiment as exp


def test_model_error_bound_on_sampled_domain():
    for concentration in np.linspace(0.0, 2.0, 7):
        for temperature in np.linspace(60.0, 95.0, 9):
            x = np.array([concentration, temperature])
            for u in (0.0, 2.0, 4.0):
                error = np.linalg.norm(
                    exp.true_dynamics(x, u) - exp.surrogate_dynamics(x, u)
                )
                assert error <= 0.75 + exp.TOL


@pytest.mark.parametrize('initial', [[1.2, 93.5], [1.8, 95.0], [0.8, 90.0]])
def test_robust_qp_matches_analytic_projection(initial):
    x = np.array(initial)
    # Independent scalar solution: project the desired input onto the
    # interval allowed by the input limits and robust temperature constraint.
    concentration, temperature = x
    reaction = 0.52 * np.exp(0.038 * (temperature - 75.0)) * concentration
    temperature_without_control = temperature + 0.10 * (
        8.2 * reaction - 0.42 * (temperature - 60.0)
    )
    allowed_temperature = 95.0 - 0.4 * (95.0 - temperature) - 0.75
    lower_input = max(0.0, (temperature_without_control - allowed_temperature) / 0.275)
    desired = np.clip(
        0.10 + 0.0025 * (temperature - 75.0) - 0.025 * (concentration - 1.0),
        0.0, 4.0,
    )
    expected = np.clip(desired, lower_input, 4.0)
    actual = exp.solve_robust_surrogate_cbf_qp(x)
    assert actual == pytest.approx(expected, abs=1e-7)


def test_robust_trajectory_satisfies_constraints():
    result = exp.simulate('robust_cbf', np.array([1.8, 95.0]))
    assert result['status'] == 'completed'
    assert result['steps_completed'] == exp.N_SIM
    assert len(result['x']) == len(result['u']) + 1
    assert np.isfinite(result['x']).all()
    assert np.all(result['h'] >= -exp.TOL)
    assert np.all(result['u'] >= exp.U_MIN)
    assert np.all(result['u'] <= exp.U_MAX)
    assert all(exp.in_omega(x) for x in result['x'])
    for key in ('model_bound_margin', 'robust_constraint_margin', 'true_cbf_margin'):
        assert np.isfinite(result[key]).all()
        assert np.all(result[key] >= -exp.TOL)


def test_infeasible_control_stops_without_fabricated_step():
    result = exp.simulate('robust_cbf', np.array([2.0, 95.0]))
    assert result['status'] == 'infeasible'
    assert result['steps_completed'] == 0
    assert result['u'].size == 0
    np.testing.assert_array_equal(result['x'], [[2.0, 95.0]])
    metrics = exp.compute_metrics(result)
    assert metrics['status'] == 'infeasible'
    assert metrics['steps_completed'] < metrics['requested_steps']


@pytest.mark.parametrize('controller', ['no_filter', 'nominal_cbf'])
def test_unsafe_trajectory_stops_and_keeps_violating_state(controller):
    result = exp.simulate(controller, np.array([1.8, 95.0]))
    assert result['status'] == 'safety_violation'
    assert result['steps_completed'] == 1
    assert result['x'].shape == (2, 2)
    assert result['h'][-1] < -exp.TOL
    metrics = exp.compute_metrics(result)
    assert metrics['violation_samples'] == 1
    assert metrics['max_T'] > exp.T_MAX
