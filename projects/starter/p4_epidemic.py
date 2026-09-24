"""P4 starter: SEIR simulation and calibration to (synthetic) case counts.

Run: python p4_epidemic.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

N = 1e6


def seir_rhs(t, x, beta, sigma, gamma):
    S, E, I, R = x
    inf = beta * S * I / N
    return [-inf, inf - sigma * E, sigma * E - gamma * I, gamma * I]


def incidence(theta, days, x0):
    """Daily new cases = sigma * E integrated over each day (approximated at day ends)."""
    beta, sigma, gamma = theta
    sol = solve_ivp(seir_rhs, (0, days[-1]), x0, t_eval=days, args=(beta, sigma, gamma), rtol=1e-8, atol=1e-6)
    cum = N - sol.y[0] - sol.y[1]  # everyone who has left E... proxy for cumulative infections
    return np.diff(cum, prepend=cum[0]) + 1e-9


def sensitivities(theta, days, x0):
    """TODO: integrate the variational equations s' = J_f s + df/dtheta alongside the ODE
    and return d(incidence)/d(theta); compare with finite differences."""
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true = np.array([0.5, 1 / 5.2, 1 / 7.0])
    x0 = [N - 20, 10, 10, 0]
    days = np.arange(0, 121.0)
    lam = incidence(true, days, x0)
    cases = rng.poisson(lam)
    print(f"true R0 = {true[0] / true[2]:.2f}, peak day {days[np.argmax(lam)]:.0f}")
    sigma_fixed = true[1]  # latent period assumed known (identifiability!)
    for ndays in (30, 50, 80, 120):
        d = days[: ndays + 1]
        resid = lambda p: np.log1p(incidence([p[0], sigma_fixed, p[1]], d, x0)) - np.log1p(cases[: ndays + 1])
        fit = least_squares(resid, x0=[0.3, 0.2], bounds=([0.01, 0.01], [3, 2]))
        J = fit.jac; s2 = 2 * fit.cost / max(1, len(d) - 2)
        se = np.sqrt(np.diag(np.linalg.inv(J.T @ J) * s2))
        b, g = fit.x
        print(f"first {ndays:3d} days: beta {b:.3f}+-{se[0]:.3f}  gamma {g:.3f}+-{se[1]:.3f}  R0 {b / g:.2f}  "
              f"growth rate beta-gamma (approx) {b - g:.3f}  corr {np.linalg.inv(J.T @ J)[0, 1] / np.sqrt(np.prod(np.diag(np.linalg.inv(J.T @ J)))):.3f}")
    # TODO: own RK45, sensitivity equations, profile likelihood for R0, real data with change point
