import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# ============================================================
# Illustrative discrete-time robust CBF simulation
# Example: simplified CSTR-like thermal control with a
# robust surrogate CBF design. Not an industrially validated plant model.
#
# State x = [C, T]
#   C : reactant concentration
#   T : reactor temperature
#
# Control u:
#   cooling-jacket actuation
#
# Safety set:
#   h(x) = T_max - T >= 0
#
# Strict theorem-oriented setup:
#   true dynamics     F(x,u)   = Fhat(x,u) + d(x)
#   surrogate model   Fhat(x,u)
#   disturbance d(x) has a rigorous known bound
#   rho(x,u) = L_h * Delta_bar(x,u)
# ============================================================


# ============================================================
# 1. Global configuration
# ============================================================


DT = 0.10
N_SIM = 80

U_MIN = 0.0
U_MAX = 4.0

# Safety threshold
T_MAX = 95.0

# Compact domain Omega
C_MIN_OMEGA = 0.0
C_MAX_OMEGA = 2.0
T_MIN_OMEGA = 60.0
T_MAX_OMEGA = 95.0

# Barrier Lipschitz constant
# h(x)=T_max-T => L_h = 1
L_H = 1.0

# alpha(r) = gamma r
ALPHA_GAMMA = 0.6

# Numerical tolerance
TOL = 1e-8

# Grid resolution for heatmaps / multi-trajectory analysis
N_C_GRID = 21
N_T_GRID = 21

# Plot style




# ============================================================
# 2. Barrier and class-K function
# ============================================================
def h_of_x(x: np.ndarray) -> float:
    return T_MAX - x[1]


def alpha_of_h(h_val: float) -> float:
    """
    Strict theorem-consistent alpha, used only when h >= 0.
    """
    return ALPHA_GAMMA * h_val


def in_safe_set(x: np.ndarray) -> bool:
    return h_of_x(x) >= -TOL


def in_omega(x: np.ndarray) -> bool:
    return (
        C_MIN_OMEGA - TOL <= x[0] <= C_MAX_OMEGA + TOL
        and T_MIN_OMEGA - TOL <= x[1] <= T_MAX_OMEGA + TOL
    )


# ============================================================
# 3. Surrogate dynamics Fhat
# ============================================================
def surrogate_dynamics(x: np.ndarray, u: float) -> np.ndarray:
    """
    Nominal surrogate model of a simplified CSTR.
    """
    C, T = x

    # Nominal Arrhenius-like reaction
    reaction_nom = 0.52 * np.exp(0.038 * (T - 75.0)) * C

    # Concentration balance
    C_next = C + DT * (0.48 * (1.45 - C) - reaction_nom)

    # Temperature balance
    T_next = T + DT * (
        8.2 * reaction_nom
        - 0.42 * (T - 60.0)
        - 2.75 * u
    )

    return np.array([C_next, T_next], dtype=float)


# ============================================================
# 4. True dynamics F = Fhat + d
# ============================================================
def disturbance(x: np.ndarray) -> np.ndarray:
    """
    Known bounded model mismatch:
        d(x) = [0, d_T(x)]

    Choose
        d_T(x) = 0.35 + 0.20 * sin^2(0.09*T) + 0.10 * C

    On Omega:
        C in [0,2], T in [60,95]
    so:
        0.35 <= d_T(x) <= 0.35 + 0.20 + 0.20 = 0.75

    Hence:
        ||d(x)|| <= 0.75
    """
    C, T = x
    d_T = 0.35 + 0.20 * np.sin(0.09 * T) ** 2 + 0.10 * C
    return np.array([0.0, d_T], dtype=float)


def true_dynamics(x: np.ndarray, u: float) -> np.ndarray:
    return surrogate_dynamics(x, u) + disturbance(x)


# ============================================================
# 5. Rigorous theorem-consistent model-error bound
# ============================================================
def delta_bar(x: np.ndarray, u: float) -> float:
    """
    A rigorous upper bound satisfying
        ||F(x,u)-Fhat(x,u)|| <= Delta_bar(x,u)
    for all (x,u) in Omega x U.

    Here, since the disturbance bound is explicit, we use:
        Delta_bar(x,u) = 0.75
    """
    return 0.75


def rho_theorem(x: np.ndarray, u: float) -> float:
    return L_H * delta_bar(x, u)


# ============================================================
# 6. Desired controller
# ============================================================
def desired_control(x: np.ndarray) -> float:
    """
    Production-oriented nominal control:
    too weak to guarantee safety.
    """
    C, T = x
    u_des = 0.10 + 0.0025 * (T - 75.0) - 0.025 * (C - 1.0)
    return float(np.clip(u_des, U_MIN, U_MAX))


# ============================================================
# 7. Affine temperature prediction under surrogate model
# ============================================================
def surrogate_temperature_affine(x: np.ndarray):
    """
    T_hat_next = a_T(x) + b_T * u
    """
    C, T = x
    reaction_nom = 0.52 * np.exp(0.038 * (T - 75.0)) * C

    a_T = T + DT * (
        8.2 * reaction_nom
        - 0.42 * (T - 60.0)
    )
    b_T = DT * (-2.75)

    return float(a_T), float(b_T)


# ============================================================
# 8. Controllers
# ============================================================
class QPFailure(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def solve_cbf_qp(x, robust=False):
    if not in_safe_set(x) or not in_omega(x):
        raise ValueError("CBF controller requires a state in the safe model domain.")
    a_T, b_T = surrogate_temperature_affine(x)
    margin = rho_theorem(x, 0.0) if robust else 0.0
    required_h = (1.0 - ALPHA_GAMMA) * h_of_x(x) + margin
    u = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.square(u - desired_control(x))), [
        u >= U_MIN, u <= U_MAX, T_MAX - (a_T + b_T * u) >= required_h,
    ])
    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                      eps_abs=1e-9, eps_rel=1e-9)
    except cp.error.SolverError as exc:
        raise QPFailure("solver_error", str(exc)) from exc
    if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        raise QPFailure("infeasible", "CBF constraint cannot be met within input limits.")
    if problem.status != cp.OPTIMAL or u.value is None:
        raise QPFailure("solver_failure", f"QP status: {problem.status}")
    value = float(u.value)
    if not np.isfinite(value) or value < U_MIN - TOL or value > U_MAX + TOL:
        raise QPFailure("numerical_failure", "Invalid QP control value.")
    value = float(np.clip(value, U_MIN, U_MAX))
    if T_MAX - (a_T + b_T * value) < required_h - TOL:
        raise QPFailure("numerical_failure", "Returned control violates the CBF constraint.")
    return value


def solve_nominal_surrogate_cbf_qp(x):
    return solve_cbf_qp(x, robust=False)



def solve_robust_surrogate_cbf_qp(x):
    return solve_cbf_qp(x, robust=True)



# ============================================================
# 9. Inequality checks
# ============================================================
def check_model_error_bound(x: np.ndarray, u: float):
    err = np.linalg.norm(true_dynamics(x, u) - surrogate_dynamics(x, u))
    bound = delta_bar(x, u)
    return err <= bound + TOL, err, bound, bound - err


def check_robust_surrogate_constraint(x: np.ndarray, u: float):
    lhs = h_of_x(surrogate_dynamics(x, u)) - h_of_x(x)
    rhs = -alpha_of_h(h_of_x(x)) + rho_theorem(x, u)
    return lhs >= rhs - TOL, lhs, rhs, lhs - rhs


def check_true_cbf_condition(x: np.ndarray, u: float):
    lhs = h_of_x(true_dynamics(x, u)) - h_of_x(x)
    rhs = -alpha_of_h(h_of_x(x))
    return lhs >= rhs - TOL, lhs, rhs, lhs - rhs


# ============================================================
# 10. Closed-loop simulation
# ============================================================
def simulate(controller_name: str, x0: np.ndarray, stop_if_leave_safe=True):
    """Stop at a safety/domain exit or failed QP; never invent a fallback.

    stop_if_leave_safe is retained for compatibility. Exits always stop a run.
    The violating next state is included in the returned trajectory.
    """
    controllers = {
        "no_filter": desired_control,
        "nominal_cbf": solve_nominal_surrogate_cbf_qp,
        "robust_cbf": solve_robust_surrogate_cbf_qp,
    }
    if controller_name not in controllers:
        raise ValueError("Unknown controller.")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (2,) or not np.isfinite(x0).all() or not in_omega(x0):
        raise ValueError("Initial state must be a finite two-component state in Omega.")
    xs, us = [x0.copy()], []
    model_margins, robust_margins, true_margins = [], [], []
    status, message = "completed", "Full simulation horizon completed."
    for _ in range(N_SIM):
        x = xs[-1]
        try:
            u = controllers[controller_name](x)
        except QPFailure as exc:
            status, message = exc.status, str(exc)
            break
        model_margins.append(check_model_error_bound(x, u)[3])
        robust_margins.append(
            check_robust_surrogate_constraint(x, u)[3]
            if controller_name == "robust_cbf" else np.nan
        )
        true_margins.append(check_true_cbf_condition(x, u)[3])
        us.append(u)
        xs.append(true_dynamics(x, u))
        if not np.isfinite(xs[-1]).all():
            status, message = "nonfinite_state", "Simulation produced a nonfinite state."
            break
        if not in_safe_set(xs[-1]):
            status, message = "safety_violation", "Temperature limit exceeded; run stopped."
            break
        if not in_omega(xs[-1]):
            status, message = "domain_exit", "State left the model-bound domain; run stopped."
            break
    states = np.asarray(xs)
    return {
        "name": controller_name, "x": states, "u": np.asarray(us),
        "h": T_MAX - states[:, 1], "rho": np.full(len(us), 0.75),
        "model_bound_margin": np.asarray(model_margins),
        "robust_constraint_margin": np.asarray(robust_margins),
        "true_cbf_margin": np.asarray(true_margins),
        "status": status, "message": message, "steps_completed": len(us),
    }



# ============================================================
# 11. Metrics
# ============================================================
def compute_metrics(res: dict):
    """Metrics cover recorded samples only; early stops are explicit."""
    temperatures = res["x"][:, 1]
    violation = np.maximum(temperatures - T_MAX, 0.0)
    return {
        "status": res["status"],
        "steps_completed": res["steps_completed"],
        "requested_steps": N_SIM,
        "min_h": float(np.min(res["h"])),
        "max_T": float(np.max(temperatures)),
        "violation_samples": int(np.sum(temperatures > T_MAX + TOL)),
        "temperature_excess_sample_sum": float(np.sum(violation)),
        "control_squared_sum": float(np.sum(res["u"] ** 2)),
        "control_squared_integral": float(DT * np.sum(res["u"] ** 2)),
    }



def print_metrics_table(results):
    print("\nController comparison (runs stop on safety/domain exit or QP failure)")
    for res in results:
        m = compute_metrics(res)
        print(f"{res['name']:<14} status={m['status']:<18} "
              f"steps={m['steps_completed']:2d}/{N_SIM} "
              f"max_T={m['max_T']:.6f} min_h={m['min_h']:.6f}")



# ============================================================
# 12. Grid-based analysis for richer figures
# ============================================================
def evaluate_grid():
    C_vals = np.linspace(0.4, 1.8, N_C_GRID)
    T_vals = np.linspace(85.0, 95.0, N_T_GRID)

    min_h_nominal = np.full((N_T_GRID, N_C_GRID), np.nan)
    min_h_robust = np.full((N_T_GRID, N_C_GRID), np.nan)
    true_cbf_margin_grid = np.full((N_T_GRID, N_C_GRID), np.nan)
    intervention_grid = np.full((N_T_GRID, N_C_GRID), np.nan)

    for i, T0 in enumerate(T_vals):
        for j, C0 in enumerate(C_vals):
            x0 = np.array([C0, T0])

            if not in_safe_set(x0) or not in_omega(x0):
                continue

            # Single-step theorem margin at this state under robust control
            try:
                u_rob = solve_robust_surrogate_cbf_qp(x0)
            except QPFailure:
                continue
            _, _, _, m_true = check_true_cbf_condition(x0, u_rob)
            true_cbf_margin_grid[i, j] = m_true
            intervention_grid[i, j] = abs(u_rob - desired_control(x0))

            # Closed-loop minimal h over horizon
            res_nom = simulate("nominal_cbf", x0, stop_if_leave_safe=False)
            res_rob = simulate("robust_cbf", x0, stop_if_leave_safe=False)

            min_h_nominal[i, j] = np.min(res_nom["h"])
            if res_rob["status"] == "completed":
                min_h_robust[i, j] = np.min(res_rob["h"])

    return C_vals, T_vals, min_h_nominal, min_h_robust, true_cbf_margin_grid, intervention_grid


# ============================================================
# 13. Multi-trajectory bundle
# ============================================================
def simulate_bundle(controller_name: str):
    init_list = [
        np.array([0.6, 90.0]),
        np.array([0.8, 91.5]),
        np.array([1.0, 92.0]),
        np.array([1.2, 93.0]),
        np.array([1.4, 94.0]),
        np.array([1.6, 94.5]),
    ]
    results = []
    for x0 in init_list:
        if in_safe_set(x0) and in_omega(x0):
            results.append(simulate(controller_name, x0, stop_if_leave_safe=False))
    return results


# ============================================================
# 14. Plotting
# ============================================================
def plot_main_results(res_no, res_nom, res_rob):
    t_no = np.arange(len(res_no["h"]))
    t_nom = np.arange(len(res_nom["h"]))
    t_rob = np.arange(len(res_rob["h"]))

    # Temperature trajectories
    plt.figure()
    plt.plot(t_no, res_no["x"][:, 1], label="No safety filter")
    plt.plot(t_nom, res_nom["x"][:, 1], label="Nominal surrogate CBF")
    plt.plot(t_rob, res_rob["x"][:, 1], label="Robust surrogate CBF")
    plt.axhline(T_MAX, linestyle="--", label=r"$T_{\max}$")
    plt.xlabel("Time step")
    plt.ylabel("Temperature")
    plt.title("True reactor temperature trajectories")
    plt.legend()
    plt.grid(True)

    # Barrier trajectories
    plt.figure()
    plt.plot(t_no, res_no["h"], label="No safety filter")
    plt.plot(t_nom, res_nom["h"], label="Nominal surrogate CBF")
    plt.plot(t_rob, res_rob["h"], label="Robust surrogate CBF")
    plt.axhline(0.0, linestyle="--", label=r"$h=0$")
    plt.xlabel("Time step")
    plt.ylabel(r"$h(x_k)=T_{\max}-T_k$")
    plt.title("Barrier value trajectories")
    plt.legend()
    plt.grid(True)

    # Control inputs
    plt.figure()
    plt.plot(np.arange(len(res_no["u"])), res_no["u"], label="No safety filter")
    plt.plot(np.arange(len(res_nom["u"])), res_nom["u"], label="Nominal surrogate CBF")
    plt.plot(np.arange(len(res_rob["u"])), res_rob["u"], label="Robust surrogate CBF")
    plt.xlabel("Time step")
    plt.ylabel("Cooling input")
    plt.title("Control inputs")
    plt.legend()
    plt.grid(True)

    # True-CBF verification margin
    plt.figure()
    valid_margin = res_rob["true_cbf_margin"][~np.isnan(res_rob["true_cbf_margin"])]
    plt.plot(np.arange(len(valid_margin)), valid_margin, label="True-CBF inequality margin")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Margin")
    plt.title("True-CBF inequality margin under robust controller")
    plt.legend()
    plt.grid(True)


def plot_bundle(bundle_nom, bundle_rob):
    plt.figure()

    for res in bundle_nom:
        plt.plot(res["x"][:, 0], res["x"][:, 1], linestyle="--", alpha=0.8)
    for res in bundle_rob:
        plt.plot(res["x"][:, 0], res["x"][:, 1], linewidth=2)

    plt.axhline(T_MAX, linestyle="--", label=r"Safety boundary $T=T_{\max}$")
    plt.xlabel("Concentration C")
    plt.ylabel("Temperature T")
    plt.title("Bundle trajectories: nominal (dashed) vs robust (solid)")
    plt.grid(True)


def plot_heatmaps(C_vals, T_vals, min_h_nominal, min_h_robust, true_cbf_margin_grid, intervention_grid):
    extent = [C_vals[0], C_vals[-1], T_vals[0], T_vals[-1]]

    # Min h heatmap for nominal
    plt.figure()
    plt.imshow(min_h_nominal, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label=r"Minimum barrier value")
    plt.contour(C_vals, T_vals, min_h_nominal, levels=[0.0], colors="white", linewidths=2)
    plt.xlabel("Initial concentration C")
    plt.ylabel("Initial temperature T")
    plt.title("Nominal controller: minimum barrier over trajectory")

    # Min h heatmap for robust
    plt.figure()
    plt.imshow(min_h_robust, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label=r"Minimum barrier value")
    plt.contour(C_vals, T_vals, min_h_robust, levels=[0.0], colors="white", linewidths=2)
    plt.xlabel("Initial concentration C")
    plt.ylabel("Initial temperature T")
    plt.title("Robust controller: minimum barrier over trajectory")

    # True-CBF margin heatmap
    plt.figure()
    plt.imshow(true_cbf_margin_grid, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="True-CBF inequality margin")
    plt.contour(C_vals, T_vals, true_cbf_margin_grid, levels=[0.0], colors="white", linewidths=2)
    plt.xlabel("State concentration C")
    plt.ylabel("State temperature T")
    plt.title("Single-step true-CBF verification margin under robust controller")

    # Intervention heatmap
    plt.figure()
    plt.imshow(intervention_grid, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label=r"$|u_{\mathrm{rob}}-u_{\mathrm{des}}|$")
    plt.xlabel("State concentration C")
    plt.ylabel("State temperature T")
    plt.title("Robust controller intervention intensity")


# ============================================================
# 15. Main
# ============================================================
def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Illustrative robust discrete-time CBF demo")
    parser.add_argument("--grid", action="store_true", help="Also run the slower grid analysis")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.figsize": (8.8, 5.5), "font.size": 11})
    x0 = np.array([1.8, 95.0])
    results = [simulate(name, x0) for name in ("no_filter", "nominal_cbf", "robust_cbf")]
    print_metrics_table(results)
    probe = simulate("robust_cbf", np.array([2.0, 95.0]))
    print(f"\nBoundary feasibility probe [2.0, 95.0]: {probe['status']}")
    report = {
        "description": "Illustrative CSTR-like discrete-time model; not a validated reactor model.",
        "config": {"dt": DT, "steps": N_SIM, "initial_state": x0.tolist(),
                   "temperature_limit": T_MAX, "input_bounds": [U_MIN, U_MAX],
                   "model_error_bound_on_omega": 0.75, "alpha_gain": ALPHA_GAMMA,
                   "omega": [[C_MIN_OMEGA, C_MAX_OMEGA], [T_MIN_OMEGA, T_MAX_OMEGA]]},
        "termination_policy": "Stop on safety/domain exit or QP failure; no fallback.",
        "controllers": {r["name"]: {**compute_metrics(r), "message": r["message"]} for r in results},
        "boundary_probe": {"initial_state": [2.0, 95.0], **compute_metrics(probe), "message": probe["message"]},
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    for res in results:
        np.savez_compressed(output / f"{res['name']}_trajectory.npz",
                            x=res["x"], u=res["u"], h=res["h"],
                            model_bound_margin=res["model_bound_margin"],
                            robust_constraint_margin=res["robust_constraint_margin"],
                            true_cbf_margin=res["true_cbf_margin"])
    plot_main_results(*results)
    names = ["temperature", "barrier", "control", "true_cbf_margin"]
    for number, name in zip(plt.get_fignums(), names):
        fig = plt.figure(number)
        fig.suptitle("Runs stop on safety/domain exit or QP failure", fontsize=10)
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    if args.grid:
        grid = evaluate_grid()
        plot_heatmaps(*grid)
        np.savez_compressed(output / "grid.npz", C=grid[0], T=grid[1],
                            min_h_nominal=grid[2], min_h_robust=grid[3],
                            true_cbf_margin=grid[4], intervention=grid[5])
    print(f"\nResults saved to: {output}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()



if __name__ == "__main__":
    main()