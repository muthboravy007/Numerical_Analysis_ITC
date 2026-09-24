"""Chapter 7 — Interpolation and approximation.

Library equivalents: ``scipy.interpolate.lagrange``, ``BarycentricInterpolator``,
``CubicSpline``, ``interp1d``, ``np.polyfit``, ``numpy.polynomial.chebyshev``.
"""

import numpy as np


def vandermonde_interp(x, y):
    """Monomial coefficients c with p(t) = sum c_k t^k. Ill-conditioned!"""
    V = np.vander(np.asarray(x, dtype=float), increasing=True)
    return np.linalg.solve(V, y)


def lagrange_eval(x, y, t):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    p = np.zeros_like(t)
    for k in range(len(x)):
        Lk = np.ones_like(t)
        for j in range(len(x)):
            if j != k:
                Lk *= (t - x[j]) / (x[k] - x[j])
        p += y[k] * Lk
    return p


def barycentric_weights(x):
    x = np.asarray(x, dtype=float)
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    return 1.0 / np.prod(diff, axis=1)


def barycentric_eval(x, y, t, w=None):
    """Second (true) barycentric formula: O(n) per point, stable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = barycentric_weights(x) if w is None else w
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        d = ti - x
        hit = np.where(d == 0)[0]
        if hit.size:
            out[i] = y[hit[0]]
        else:
            q = w / d
            out[i] = (q @ y) / q.sum()
    return out


def divided_differences(x, y):
    """Newton coefficients f[x0], f[x0,x1], ..., f[x0..xn]."""
    x = np.asarray(x, dtype=float)
    c = np.array(y, dtype=float)
    n = len(x)
    for j in range(1, n):
        c[j:] = (c[j:] - c[j - 1:-1]) / (x[j:] - x[:-j])
    return c


def divided_difference_table(x, y):
    x = np.asarray(x, dtype=float)
    n = len(x)
    T = np.full((n, n), np.nan)
    T[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            T[i, j] = (T[i + 1, j - 1] - T[i, j - 1]) / (x[i + j] - x[i])
    return T


def newton_eval(x, coef, t):
    """Horner-like nested evaluation of the Newton form."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    p = np.full_like(t, coef[-1], dtype=float)
    for k in range(len(coef) - 2, -1, -1):
        p = p * (t - x[k]) + coef[k]
    return p


def horner(coef, t):
    """Evaluate sum coef[k] t^k with n multiplications."""
    p = np.zeros_like(np.asarray(t, dtype=float)) + coef[-1]
    for c in coef[-2::-1]:
        p = p * t + c
    return p


def chebyshev_nodes(n, a=-1.0, b=1.0):
    """n Chebyshev points of the first kind on [a, b]."""
    k = np.arange(n)
    x = np.cos((2 * k + 1) * np.pi / (2 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def runge(x):
    return 1.0 / (1.0 + 25.0 * np.asarray(x) ** 2)


def linear_spline(x, y, t):
    return np.interp(t, x, y)


def natural_cubic_spline(x, y):
    """Return the second derivatives M_i of the natural cubic spline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    A[0, 0] = A[n, n] = 1.0  # natural: M_0 = M_n = 0
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return np.linalg.solve(A, rhs)


def cubic_spline_eval(x, y, M, t):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    i = np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    a = x[i + 1] - t
    b = t - x[i]
    return (M[i] * a ** 3 / (6 * h) + M[i + 1] * b ** 3 / (6 * h)
            + (y[i] / h - M[i] * h / 6) * a + (y[i + 1] / h - M[i + 1] * h / 6) * b)


def neville(x, y, t):
    """Neville's iterated interpolation. Returns (value, table Q)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    Q = np.full((n, n), np.nan)
    Q[:, 0] = y
    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i, j] = ((t - x[i - j]) * Q[i, j - 1] - (t - x[i]) * Q[i - 1, j - 1]) / (x[i] - x[i - j])
    return Q[n - 1, n - 1], Q


def hermite_coefficients(x, f, df):
    """Hermite interpolation via divided differences with doubled nodes.
    Returns (z, coef) so that H(t) = newton_eval(z, coef, t)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    z = np.repeat(x, 2)
    Q = np.zeros((2 * n, 2 * n))
    for i in range(n):
        Q[2 * i, 0] = Q[2 * i + 1, 0] = f[i]
        Q[2 * i + 1, 1] = df[i]
        if i > 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])
    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])
    return z, np.diag(Q).copy()


def clamped_cubic_spline(x, y, fp0, fpn):
    """Second derivatives M_i of the clamped spline with S'(x0)=fp0, S'(xn)=fpn."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    A[0, 0], A[0, 1] = 2 * h[0], h[0]
    rhs[0] = 6 * ((y[1] - y[0]) / h[0] - fp0)
    A[n, n - 1], A[n, n] = h[n - 1], 2 * h[n - 1]
    rhs[n] = 6 * (fpn - (y[n] - y[n - 1]) / h[n - 1])
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return np.linalg.solve(A, rhs)


def spline_coefficients(x, y, M):
    """Convert (y_i, M_i) to B&F form S_j(t) = a_j + b_j(t-x_j) + c_j(t-x_j)^2 + d_j(t-x_j)^3."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.diff(x)
    a = y[:-1]
    c = M[:-1] / 2
    d = (M[1:] - M[:-1]) / (6 * h)
    b = (y[1:] - y[:-1]) / h - h * (2 * M[:-1] + M[1:]) / 6
    return a, b, c, d
