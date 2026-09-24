"""Chapter 8 — Numerical differentiation and integration.

Library equivalents: ``np.gradient``, ``scipy.integrate.quad``,
``scipy.integrate.trapezoid`` / ``simpson``, ``np.polynomial.legendre.leggauss``,
automatic differentiation (JAX, PyTorch).
"""

import numpy as np


# ---------- differentiation ----------

def forward_diff(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h


def backward_diff(f, x, h=1e-6):
    return (f(x) - f(x - h)) / h


def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def second_diff(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


def complex_step(f, x, h=1e-20):
    """f'(x) ~ Im f(x + ih) / h — no subtractive cancellation."""
    return np.imag(f(x + 1j * h)) / h


def richardson(D, h, p=2, levels=4):
    """Richardson extrapolation of a difference formula D(h) = D + O(h^p)
    with error expansion in powers h^p, h^{2p}, ... (p=2 for central)."""
    T = np.zeros((levels, levels))
    for i in range(levels):
        T[i, 0] = D(h / 2 ** i)
        for j in range(1, i + 1):
            f = 2 ** (p * j)
            T[i, j] = (f * T[i, j - 1] - T[i - 1, j - 1]) / (f - 1)
    return T[levels - 1, levels - 1], T


def gradient_fd(f, x, h=1e-6):
    """Central-difference gradient of f: R^n -> R (gradient checking)."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


# ---------- integration ----------

def midpoint(f, a, b, n):
    h = (b - a) / n
    x = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(x))


def trapezoid(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])


def simpson(f, a, b, n):
    if n % 2:
        raise ValueError("Simpson's rule needs an even number of subintervals")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


def trapezoid_data(x, y):
    """Trapezoid rule for tabulated (possibly non-uniform) data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2)


def romberg(f, a, b, levels=6):
    R = np.zeros((levels, levels))
    h = b - a
    R[0, 0] = h / 2 * (f(a) + f(b))
    for i in range(1, levels):
        h /= 2
        new_x = a + h * np.arange(1, 2 ** i, 2)
        R[i, 0] = R[i - 1, 0] / 2 + h * np.sum(f(new_x))
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4 ** j - 1)
    return R[-1, -1], R


def adaptive_simpson(f, a, b, tol=1e-10, max_depth=50):
    def S(a, b, fa, fm, fb):
        return (b - a) / 6 * (fa + 4 * fm + fb)

    evals = [0]

    def rec(a, b, fa, fm, fb, whole, tol, depth):
        m = (a + b) / 2
        lm, rm = (a + m) / 2, (m + b) / 2
        flm, frm = f(lm), f(rm)
        evals[0] += 2
        left = S(a, m, fa, flm, fm)
        right = S(m, b, fm, frm, fb)
        if depth <= 0 or abs(left + right - whole) <= 15 * tol:
            return left + right + (left + right - whole) / 15
        return (rec(a, m, fa, flm, fm, left, tol / 2, depth - 1)
                + rec(m, b, fm, frm, fb, right, tol / 2, depth - 1))

    fa, fb, fm = f(a), f(b), f((a + b) / 2)
    evals[0] += 3
    val = rec(a, b, fa, fm, fb, S(a, b, fa, fm, fb), tol, max_depth)
    return val, evals[0]


def gauss_legendre(f, a, b, n):
    """n-point Gauss-Legendre: exact for polynomials of degree <= 2n-1."""
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(w * f(t))


def monte_carlo(f, a, b, n, rng=None):
    """Plain Monte Carlo on [a,b]. Returns (estimate, standard_error)."""
    rng = rng or np.random.default_rng()
    x = rng.uniform(a, b, n)
    y = (b - a) * f(x)
    return y.mean(), y.std(ddof=1) / np.sqrt(n)


def monte_carlo_nd(f, lows, highs, n, rng=None):
    """Monte Carlo over a box in R^d; f takes an (n, d) array."""
    rng = rng or np.random.default_rng()
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    vol = np.prod(highs - lows)
    X = rng.uniform(lows, highs, size=(n, len(lows)))
    y = vol * f(X)
    return y.mean(), y.std(ddof=1) / np.sqrt(n)
