"""Chapter 12 (B&F) — Finite-difference methods for PDEs.

Elliptic: Poisson  u_xx + u_yy = f on a rectangle (Dirichlet data g).
Parabolic: heat    u_t = alpha^2 u_xx, u(0,t) = u(l,t) = 0.
Hyperbolic: wave   u_tt = alpha^2 u_xx, u(0,t) = u(l,t) = 0.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .linalg_direct import tridiagonal_solve


def poisson_rect(f, g, a, b, c, d, n, m):
    """5-point scheme on [a,b]x[c,d] with n, m subintervals (B&F Alg. 12.1 but
    solved directly with a sparse LU). Returns x, y, W with W[i, j] ~ u(x_i, y_j)."""
    h = (b - a) / n
    k = (d - c) / m
    x = np.linspace(a, b, n + 1)
    y = np.linspace(c, d, m + 1)
    W = np.zeros((n + 1, m + 1))
    X, Y = np.meshgrid(x, y, indexing="ij")
    W[0, :], W[-1, :] = g(x[0], y), g(x[-1], y)
    W[:, 0], W[:, -1] = g(x, y[0]), g(x, y[-1])
    ni, mi = n - 1, m - 1
    lam = (h / k) ** 2
    idx = lambda i, j: (i - 1) * mi + (j - 1)
    A = sp.lil_matrix((ni * mi, ni * mi))
    rhs = np.zeros(ni * mi)
    for i in range(1, n):
        for j in range(1, m):
            p = idx(i, j)
            A[p, p] = 2 * (1 + lam)
            rhs[p] = -h * h * f(x[i], y[j])
            for (ii, jj, coef) in ((i - 1, j, 1.0), (i + 1, j, 1.0), (i, j - 1, lam), (i, j + 1, lam)):
                if 1 <= ii <= n - 1 and 1 <= jj <= m - 1:
                    A[p, idx(ii, jj)] = -coef
                else:
                    rhs[p] += coef * W[ii, jj]
    sol = spla.spsolve(A.tocsr(), rhs)
    W[1:n, 1:m] = sol.reshape(ni, mi)
    return x, y, W


def heat_ftcs(u0, alpha, l, T, m, N):
    """Forward difference (explicit) method; stable iff lam = alpha^2 k/h^2 <= 1/2."""
    h, k = l / m, T / N
    lam = alpha ** 2 * k / h ** 2
    x = np.linspace(0, l, m + 1)
    w = u0(x).astype(float)
    w[0] = w[-1] = 0.0
    for _ in range(N):
        w[1:-1] = (1 - 2 * lam) * w[1:-1] + lam * (w[:-2] + w[2:])
    return x, w, lam


def heat_btcs(u0, alpha, l, T, m, N):
    """Backward difference (implicit) method; unconditionally stable, O(k + h^2)."""
    h, k = l / m, T / N
    lam = alpha ** 2 * k / h ** 2
    x = np.linspace(0, l, m + 1)
    w = u0(x[1:-1]).astype(float)
    n = m - 1
    off = -lam * np.ones(n - 1)
    diag = (1 + 2 * lam) * np.ones(n)
    for _ in range(N):
        w = tridiagonal_solve(off, diag, off, w)
    return x, np.concatenate([[0.0], w, [0.0]]), lam


def heat_crank_nicolson(u0, alpha, l, T, m, N):
    """Crank-Nicolson; unconditionally stable, O(k^2 + h^2) (B&F Alg. 12.3)."""
    h, k = l / m, T / N
    lam = alpha ** 2 * k / h ** 2
    x = np.linspace(0, l, m + 1)
    w = u0(x[1:-1]).astype(float)
    n = m - 1
    off = -lam / 2 * np.ones(n - 1)
    diag = (1 + lam) * np.ones(n)
    for _ in range(N):
        rhs = (1 - lam) * w
        rhs[1:] += lam / 2 * w[:-1]
        rhs[:-1] += lam / 2 * w[1:]
        w = tridiagonal_solve(off, diag, off, rhs)
    return x, np.concatenate([[0.0], w, [0.0]]), lam


def wave_explicit(f, g, alpha, l, T, m, N):
    """Explicit scheme for u_tt = alpha^2 u_xx with u(x,0)=f, u_t(x,0)=g.
    Stable iff lam = alpha k / h <= 1 (CFL). Returns x and W at t = T."""
    h, k = l / m, T / N
    lam = alpha * k / h
    x = np.linspace(0, l, m + 1)
    w0 = f(x).astype(float)
    w0[0] = w0[-1] = 0.0
    w1 = w0.copy()
    w1[1:-1] = (1 - lam ** 2) * w0[1:-1] + lam ** 2 / 2 * (w0[2:] + w0[:-2]) + k * g(x[1:-1])
    for _ in range(N - 1):
        w2 = np.zeros_like(w1)
        w2[1:-1] = 2 * (1 - lam ** 2) * w1[1:-1] + lam ** 2 * (w1[2:] + w1[:-2]) - w0[1:-1]
        w0, w1 = w1, w2
    return x, w1, lam
