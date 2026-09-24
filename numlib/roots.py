"""Chapter 2 — Solving nonlinear equations f(x) = 0.

Library equivalents: ``scipy.optimize.brentq``, ``scipy.optimize.newton``,
``scipy.optimize.root_scalar``, ``scipy.optimize.fsolve``.

Every solver returns ``(x, history)`` where ``history`` is the list of
iterates, so students can study convergence rates.
"""

import numpy as np


class ConvergenceError(RuntimeError):
    pass


def bisection(f, a, b, tol=1e-10, max_iter=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    history = []
    for _ in range(max_iter):
        m = a + (b - a) / 2  # safer than (a+b)/2
        fm = f(m)
        history.append(m)
        if fm == 0 or (b - a) / 2 < tol:
            return m, history
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return m, history


def bisection_iterations(a, b, tol):
    """Number of steps guaranteeing |x_n - r| <= tol."""
    return int(np.ceil(np.log2((b - a) / tol)))


def fixed_point(g, x0, tol=1e-10, max_iter=500):
    x = x0
    history = [x]
    for _ in range(max_iter):
        x_new = g(x)
        history.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, history
        x = x_new
    raise ConvergenceError("fixed-point iteration did not converge")


def newton(f, df, x0, tol=1e-12, max_iter=100):
    x = x0
    history = [x]
    for _ in range(max_iter):
        d = df(x)
        if d == 0:
            raise ConvergenceError("zero derivative encountered")
        x_new = x - f(x) / d
        history.append(x_new)
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return x_new, history
        x = x_new
    raise ConvergenceError("Newton's method did not converge")


def secant(f, x0, x1, tol=1e-12, max_iter=100):
    history = [x0, x1]
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if f1 == f0:
            raise ConvergenceError("zero secant slope")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        if abs(x2 - x1) < tol * max(1.0, abs(x2)):
            return x2, history
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
    raise ConvergenceError("secant method did not converge")


def regula_falsi(f, a, b, tol=1e-12, max_iter=500):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    history = []
    c_old = a
    for _ in range(max_iter):
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        history.append(c)
        if abs(c - c_old) < tol or fc == 0:
            return c, history
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_old = c
    return c, history


def safeguarded_newton(f, df, a, b, tol=1e-12, max_iter=200):
    """Newton step when it stays inside the bracket, bisection otherwise.
    A simplified version of the idea behind Brent's method."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("root not bracketed")
    x = a + (b - a) / 2
    history = [x]
    for _ in range(max_iter):
        fx, dfx = f(x), df(x)
        if fx == 0:
            return x, history
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_new = x - fx / dfx if dfx != 0 else None
        newton_step = x_new is not None and a < x_new < b
        if not newton_step:
            x_new = a + (b - a) / 2
        history.append(x_new)
        # declare convergence only on a Newton step (or a collapsed bracket)
        if (newton_step and abs(x_new - x) < tol * max(1.0, abs(x_new))) or b - a < tol:
            return x_new, history
        x = x_new
    return x, history


def newton_system(F, J, x0, tol=1e-12, max_iter=50):
    """Newton's method for F(x) = 0 with F: R^n -> R^n and Jacobian J."""
    x = np.asarray(x0, dtype=float)
    history = [x.copy()]
    for _ in range(max_iter):
        dx = np.linalg.solve(J(x), -F(x))
        x = x + dx
        history.append(x.copy())
        if np.linalg.norm(dx) < tol * max(1.0, np.linalg.norm(x)):
            return x, history
    raise ConvergenceError("Newton system did not converge")


def estimate_order(history, root):
    """Estimate convergence order p from e_{k+1} ~ C e_k^p."""
    e = np.abs(np.asarray(history, dtype=float) - root)
    e = e[e > 0]
    if len(e) < 3:
        return float("nan")
    p = np.log(e[2:] / e[1:-1]) / np.log(e[1:-1] / e[:-2])
    return p
