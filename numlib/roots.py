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


# ---------------- B&F 2.5: accelerating convergence ----------------

def aitken(seq):
    """Aitken's Delta^2: p_hat_n = p_n - (p_{n+1}-p_n)^2 / (p_{n+2}-2p_{n+1}+p_n)."""
    p = np.asarray(seq, dtype=float)
    d1 = p[1:-1] - p[:-2]
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:-2] - d1 ** 2 / d2


def steffensen(g, p0, tol=1e-12, max_iter=100):
    """Steffensen's method: Aitken applied to fixed-point iteration every 3 steps.
    Quadratically convergent when g'(p) != 1."""
    history = [p0]
    for _ in range(max_iter):
        p1 = g(p0)
        p2 = g(p1)
        denom = p2 - 2 * p1 + p0
        if denom == 0:
            return p2, history
        p = p0 - (p1 - p0) ** 2 / denom
        history.append(p)
        if abs(p - p0) < tol:
            return p, history
        p0 = p
    raise ConvergenceError("Steffensen did not converge")


# ---------------- B&F 2.6: zeros of polynomials ----------------

def horner_eval(coef, x0):
    """coef = [a_n, ..., a_0] (highest degree first).
    Returns (P(x0), P'(x0), quotient coefficients of P(x)/(x-x0))."""
    b = coef[0]
    d = coef[0]
    q = [b]
    for a in coef[1:-1]:
        b = a + b * x0
        d = b + d * x0
        q.append(b)
    b = coef[-1] + b * x0
    return b, d if len(coef) > 1 else 0.0, q[:len(coef) - 1]


def newton_horner(coef, x0, tol=1e-12, max_iter=100):
    """Newton's method for a polynomial using Horner's scheme for P and P'."""
    x = x0
    history = [x]
    for _ in range(max_iter):
        p, dp, _ = horner_eval(coef, x)
        x_new = x - p / dp
        history.append(x_new)
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return x_new, history
        x = x_new
    raise ConvergenceError("Newton-Horner did not converge")


def muller(f, p0, p1, p2, tol=1e-12, max_iter=100):
    """Muller's method: fits a parabola through three points; finds complex roots."""
    p0, p1, p2 = complex(p0), complex(p1), complex(p2)
    history = [p0, p1, p2]
    for _ in range(max_iter):
        h1, h2 = p1 - p0, p2 - p1
        d1 = (f(p1) - f(p0)) / h1
        d2 = (f(p2) - f(p1)) / h2
        a = (d2 - d1) / (h2 + h1)
        b = d2 + h2 * a
        D = np.sqrt(b * b - 4 * f(p2) * a + 0j)
        E = b + D if abs(b - D) < abs(b + D) else b - D
        h = -2 * f(p2) / E
        p = p2 + h
        history.append(p)
        if abs(h) < tol:
            return p, history
        p0, p1, p2 = p1, p2, p
    raise ConvergenceError("Muller did not converge")
