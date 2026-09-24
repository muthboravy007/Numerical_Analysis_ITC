"""Chapter 10 (B&F) — Numerical solution of nonlinear systems F(x) = 0.

Library equivalents: ``scipy.optimize.root`` (hybr, lm, broyden1),
``scipy.optimize.fsolve``.
"""

import numpy as np

from .roots import ConvergenceError, newton_system  # noqa: F401  (re-export)


def fixed_point_system(G, x0, tol=1e-10, max_iter=500):
    """x_{k+1} = G(x_k) for G: R^n -> R^n (B&F 10.1)."""
    x = np.asarray(x0, dtype=float)
    history = [x.copy()]
    for _ in range(max_iter):
        x_new = np.asarray(G(x), dtype=float)
        history.append(x_new.copy())
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, history
        x = x_new
    raise ConvergenceError("fixed-point system did not converge")


def jacobian_fd(F, x, h=1e-7):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(F(x))
    J = np.zeros((len(f0), len(x)))
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = h * max(1.0, abs(x[j]))
        J[:, j] = (np.asarray(F(x + e)) - f0) / e[j]
    return J


def broyden(F, x0, J0=None, tol=1e-10, max_iter=100):
    """Broyden's method with Sherman-Morrison update of A^{-1} (B&F Alg. 10.2)."""
    x = np.asarray(x0, dtype=float)
    A_inv = np.linalg.inv(jacobian_fd(F, x) if J0 is None else np.asarray(J0, dtype=float))
    v = np.asarray(F(x), dtype=float)
    s = -A_inv @ v
    x = x + s
    history = [np.asarray(x0, dtype=float), x.copy()]
    for _ in range(max_iter):
        w = v
        v = np.asarray(F(x), dtype=float)
        y = v - w
        z = -A_inv @ y
        p = -s @ z
        A_inv = A_inv + np.outer(s + z, s @ A_inv) / p
        s = -A_inv @ v
        x = x + s
        history.append(x.copy())
        if np.linalg.norm(s) < tol * max(1.0, np.linalg.norm(x)):
            return x, history
    raise ConvergenceError("Broyden did not converge")


def steepest_descent_system(F, J, x0, tol=1e-5, max_iter=1000):
    """Minimise g(x) = sum F_i(x)^2 by steepest descent with the B&F quadratic
    line search (Alg. 10.3). Good for finding starting values for Newton."""
    x = np.asarray(x0, dtype=float)
    g = lambda x: float(np.sum(np.asarray(F(x)) ** 2))
    history = [x.copy()]
    for _ in range(max_iter):
        g1 = g(x)
        z = 2 * np.asarray(J(x)).T @ np.asarray(F(x))
        z0 = np.linalg.norm(z)
        if z0 == 0:
            return x, history
        z = z / z0
        a3 = 1.0
        g3 = g(x - a3 * z)
        while g3 >= g1:
            a3 /= 2
            g3 = g(x - a3 * z)
            if a3 < tol / 2:
                return x, history
        a2 = a3 / 2
        g2 = g(x - a2 * z)
        h1 = (g2 - g1) / a2
        h2 = (g3 - g2) / (a3 - a2)
        h3 = (h2 - h1) / a3
        a0 = 0.5 * (a2 - h1 / h3) if h3 != 0 else a2
        g0 = g(x - a0 * z)
        a, gnew = (a0, g0) if g0 < g3 else (a3, g3)
        x = x - a * z
        history.append(x.copy())
        if abs(gnew - g1) < tol:
            return x, history
    return x, history


def homotopy(F, J, x0, N=10):
    """Continuation method (B&F Alg. 10.4): follow H(x, l) = F(x) + (l-1)F(x0)
    from l = 0 to 1 by RK4 on x'(l) = -J(x)^{-1} F(x0)."""
    x = np.asarray(x0, dtype=float)
    b = -np.asarray(F(x), dtype=float)
    h = 1.0 / N
    path = [x.copy()]
    for _ in range(N):
        k1 = h * np.linalg.solve(J(x), b)
        k2 = h * np.linalg.solve(J(x + k1 / 2), b)
        k3 = h * np.linalg.solve(J(x + k2 / 2), b)
        k4 = h * np.linalg.solve(J(x + k3), b)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        path.append(x.copy())
    return x, path
