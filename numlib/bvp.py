"""Chapter 11 (B&F) — Boundary-value problems for ODEs.

y'' = f(x, y, y'),  a <= x <= b,  y(a) = alpha,  y(b) = beta.

Library equivalent: ``scipy.integrate.solve_bvp``.
"""

import numpy as np

from .linalg_direct import tridiagonal_solve
from .ode import rk4


def linear_shooting(p, q, r, a, b, alpha, beta, N):
    """y'' = p(x) y' + q(x) y + r(x) (B&F Alg. 11.1).
    Solves two IVPs with RK4 and superposes: y = y1 + (beta - y1(b))/y2(b) * y2."""
    h = (b - a) / N
    f1 = lambda x, u: np.array([u[1], p(x) * u[1] + q(x) * u[0] + r(x)])
    f2 = lambda x, u: np.array([u[1], p(x) * u[1] + q(x) * u[0]])
    x, U1 = rk4(f1, (a, b), [alpha, 0.0], h)
    _, U2 = rk4(f2, (a, b), [0.0, 1.0], h)
    c = (beta - U1[-1, 0]) / U2[-1, 0]
    return x, U1[:, 0] + c * U2[:, 0], U1[:, 1] + c * U2[:, 1]


def nonlinear_shooting(f, fy, fyp, a, b, alpha, beta, N, t0=None, tol=1e-10, max_iter=25):
    """Nonlinear shooting with Newton's method on the initial slope t
    (B&F Alg. 11.2). fy, fyp: partial derivatives of f w.r.t. y and y'."""
    h = (b - a) / N
    t = (beta - alpha) / (b - a) if t0 is None else t0
    slopes = [t]
    for _ in range(max_iter):
        # augmented system: (y, y', z, z') where z = dy/dt
        def F(x, u):
            y, yp, z, zp = u
            return np.array([yp, f(x, y, yp), zp, fy(x, y, yp) * z + fyp(x, y, yp) * zp])
        x, U = rk4(F, (a, b), [alpha, t, 0.0, 1.0], h)
        dt = (U[-1, 0] - beta) / U[-1, 2]
        t -= dt
        slopes.append(t)
        if abs(dt) < tol:
            break
    x, U = rk4(lambda x, u: np.array([u[1], f(x, u[0], u[1])]), (a, b), [alpha, t], h)
    return x, U[:, 0], slopes


def linear_fd(p, q, r, a, b, alpha, beta, N):
    """Finite differences for y'' = p y' + q y + r with N+1 subintervals
    (B&F Alg. 11.3, O(h^2)). Returns grid x_0..x_{N+1} and w."""
    h = (b - a) / (N + 1)
    x = a + h * np.arange(1, N + 1)
    diag = 2 + h * h * q(x)
    lower = -1 - h / 2 * p(x[1:])
    upper = -1 + h / 2 * p(x[:-1])
    rhs = -h * h * r(x)
    rhs[0] += (1 + h / 2 * p(x[0])) * alpha
    rhs[-1] += (1 - h / 2 * p(x[-1])) * beta
    w = tridiagonal_solve(lower, diag, upper, rhs)
    return np.concatenate([[a], x, [b]]), np.concatenate([[alpha], w, [beta]])


def nonlinear_fd(f, fy, fyp, a, b, alpha, beta, N, tol=1e-10, max_iter=50):
    """Nonlinear finite differences with Newton's method (B&F Alg. 11.4)."""
    h = (b - a) / (N + 1)
    x = a + h * np.arange(1, N + 1)
    w = alpha + (beta - alpha) / (b - a) * (x - a)  # linear initial guess
    for it in range(max_iter):
        wl = np.concatenate([[alpha], w[:-1]])
        wr = np.concatenate([w[1:], [beta]])
        yp = (wr - wl) / (2 * h)
        F = -wl + 2 * w - wr + h * h * f(x, w, yp)
        diag = 2 + h * h * fy(x, w, yp)
        upper = -1 + h / 2 * fyp(x[:-1], w[:-1], yp[:-1])
        lower = -1 - h / 2 * fyp(x[1:], w[1:], yp[1:])
        v = tridiagonal_solve(lower, diag, upper, -F)
        w = w + v
        if np.max(np.abs(v)) < tol:
            break
    return np.concatenate([[a], x, [b]]), np.concatenate([[alpha], w, [beta]]), it + 1
