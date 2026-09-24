"""Chapter 8 (B&F) — Approximation theory: discrete and continuous least squares,
orthogonal polynomials, Chebyshev economization, Padé, trigonometric
approximation and the FFT.

Library equivalents: ``np.polyfit``, ``numpy.polynomial.{legendre,chebyshev}``,
``scipy.interpolate.pade``, ``np.fft.fft`` / ``rfft``.
"""

import numpy as np
from scipy import integrate


# ---------------- discrete least squares ----------------

def linear_fit(x, y):
    """Least-squares line y = a0 + a1 x from the 2x2 normal equations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(x)
    Sx, Sy, Sxx, Sxy = x.sum(), y.sum(), (x * x).sum(), (x * y).sum()
    det = m * Sxx - Sx ** 2
    a0 = (Sxx * Sy - Sxy * Sx) / det
    a1 = (m * Sxy - Sx * Sy) / det
    return a0, a1


def poly_fit(x, y, n):
    """Degree-n least-squares polynomial via normal equations (B&F 8.1).
    Returns coefficients a_0..a_n (increasing degree) and the error E."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vander(x, n + 1, increasing=True)
    a = np.linalg.solve(A.T @ A, A.T @ y)
    E = np.sum((y - A @ a) ** 2)
    return a, E


def exp_fit(x, y):
    """Fit y = b e^{a x} by linearising ln y = ln b + a x. Returns (b, a)."""
    c0, c1 = linear_fit(x, np.log(y))
    return np.exp(c0), c1


def power_fit(x, y):
    """Fit y = b x^a by linearising ln y = ln b + a ln x. Returns (b, a)."""
    c0, c1 = linear_fit(np.log(x), np.log(y))
    return np.exp(c0), c1


# ---------------- continuous least squares ----------------

def continuous_ls_monomial(f, a, b, n):
    """Continuous LS polynomial on [a,b] using the (Hilbert-like) monomial
    normal equations. Returns coefficients and the Gram matrix."""
    G = np.array([[(b ** (i + j + 1) - a ** (i + j + 1)) / (i + j + 1)
                   for j in range(n + 1)] for i in range(n + 1)])
    rhs = np.array([integrate.quad(lambda x, k=k: x ** k * f(x), a, b)[0] for k in range(n + 1)])
    return np.linalg.solve(G, rhs), G


def legendre_ls(f, n):
    """Continuous LS on [-1,1] in the Legendre basis: c_k = (2k+1)/2 * <f, P_k>."""
    from numpy.polynomial import legendre as L

    c = np.zeros(n + 1)
    for k in range(n + 1):
        Pk = L.Legendre.basis(k)
        c[k] = (2 * k + 1) / 2 * integrate.quad(lambda x: f(x) * Pk(x), -1, 1)[0]
    return L.Legendre(c)


def gram_schmidt_polys(n, a, b, w=lambda x: 1.0):
    """Monic orthogonal polynomials phi_0..phi_n on [a,b] w.r.t. weight w,
    by the three-term recurrence (B&F Theorem 8.7). Returns list of np.poly1d."""
    ip = lambda p, q: integrate.quad(lambda x: w(x) * p(x) * q(x), a, b)[0]
    x = np.poly1d([1, 0])
    phis = [np.poly1d([1.0])]
    B1 = ip(x * phis[0], phis[0]) / ip(phis[0], phis[0])
    phis.append(x - B1)
    for k in range(2, n + 1):
        pk1, pk2 = phis[-1], phis[-2]
        Bk = ip(x * pk1, pk1) / ip(pk1, pk1)
        Ck = ip(x * pk1, pk2) / ip(pk2, pk2)
        phis.append((x - Bk) * pk1 - Ck * pk2)
    return phis[: n + 1]


def chebyshev_T(n):
    """Chebyshev polynomial T_n as np.poly1d via T_{n+1} = 2x T_n - T_{n-1}."""
    T0, T1 = np.poly1d([1.0]), np.poly1d([1.0, 0.0])
    if n == 0:
        return T0
    for _ in range(n - 1):
        T0, T1 = T1, 2 * np.poly1d([1.0, 0.0]) * T1 - T0
    return T1


def economize(p, target_degree):
    """Chebyshev economization on [-1,1]: repeatedly subtract a_n * T_n / 2^{n-1}.
    Returns (economized poly1d, max error bound added)."""
    p = np.poly1d(p)
    bound = 0.0
    while p.order > target_degree:
        n = p.order
        an = p.coeffs[0]
        p = p - an * chebyshev_T(n) / 2 ** (n - 1)
        p = np.poly1d(p.coeffs[1:]) if abs(p.coeffs[0]) < 1e-14 else p
        bound += abs(an) / 2 ** (n - 1)
    return p, bound


def pade(taylor_coeffs, n, m):
    """Padé approximant r = p/q with deg p = n, deg q = m, q_0 = 1, from the
    Maclaurin coefficients a_0..a_{n+m}. Returns (p, q) increasing-order arrays."""
    a = np.asarray(taylor_coeffs, dtype=float)
    N = n + m
    # unknowns: q_1..q_m, p_0..p_n ; equations for k = 0..N:
    # sum_{i=0}^{k} a_i q_{k-i} - p_k = 0  (q_0 = 1, p_k = 0 for k > n, q_j = 0 for j > m)
    M = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)
    for k in range(N + 1):
        rhs[k] = -a[k]
        for j in range(1, m + 1):
            if k - j >= 0:
                M[k, j - 1] = a[k - j]
        if k <= n:
            M[k, m + k] = -1.0
    sol = np.linalg.solve(M, rhs)
    q = np.concatenate([[1.0], sol[:m]])
    p = sol[m:]
    return p, q


# ---------------- trigonometric approximation / FFT ----------------

def trig_ls_coeffs(y, n):
    """Discrete LS trigonometric polynomial S_n on 2m equally spaced points
    x_j = -pi + j pi / m (B&F 8.5). Returns (a_0..a_n, b_1..b_n)."""
    y = np.asarray(y, dtype=float)
    two_m = len(y)
    m = two_m // 2
    x = -np.pi + np.arange(two_m) * np.pi / m
    a = np.array([np.sum(y * np.cos(k * x)) / m for k in range(n + 1)])
    b = np.array([np.sum(y * np.sin(k * x)) / m for k in range(1, n + 1)])
    return a, b


def trig_eval(a, b, t):
    t = np.asarray(t, dtype=float)
    s = a[0] / 2 + sum(a[k] * np.cos(k * t) for k in range(1, len(a)))
    return s + sum(b[k - 1] * np.sin(k * t) for k in range(1, len(b) + 1))


def fft_radix2(x):
    """Recursive Cooley-Tukey FFT: X_k = sum_j x_j e^{-2 pi i jk/N}, N a power of 2."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N == 1:
        return x
    if N % 2:
        raise ValueError("length must be a power of 2")
    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    tw = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + tw, even - tw])


def dft_naive(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(k, k) / N) @ x
