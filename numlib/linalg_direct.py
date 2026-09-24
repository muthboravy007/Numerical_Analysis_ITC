"""Chapter 3 — Direct methods for linear systems Ax = b.

Library equivalents: ``np.linalg.solve``, ``scipy.linalg.lu_factor`` /
``lu_solve``, ``scipy.linalg.cho_factor`` / ``cho_solve``,
``scipy.linalg.solve_triangular``, ``np.linalg.cond``.
"""

import numpy as np


def forward_substitution(L, b):
    """Solve Lx = b, L lower triangular. Cost: n^2 flops."""
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def back_substitution(U, b):
    """Solve Ux = b, U upper triangular. Cost: n^2 flops."""
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def gaussian_elimination(A, b, pivot=True):
    """Solve Ax = b by elimination on the augmented matrix."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    for k in range(n - 1):
        if pivot:
            p = k + np.argmax(np.abs(A[k:, k]))
            if p != k:
                A[[k, p]] = A[[p, k]]
                b[[k, p]] = b[[p, k]]
        if A[k, k] == 0:
            raise ZeroDivisionError("zero pivot; use pivot=True")
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
    return back_substitution(A, b)


def lu(A):
    """Doolittle LU without pivoting: A = L U."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        if U[k, k] == 0:
            raise ZeroDivisionError("zero pivot; use lu_partial_pivot")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return L, U


def lu_partial_pivot(A):
    """PA = LU with partial (row) pivoting. Returns P, L, U."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()
    for k in range(n - 1):
        p = k + np.argmax(np.abs(U[k:, k]))
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
        if U[k, k] == 0:
            continue  # singular column; nothing to eliminate
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    L += np.eye(n)
    return P, L, U


def lu_solve(P, L, U, b):
    y = forward_substitution(L, P @ np.asarray(b, dtype=float))
    return back_substitution(U, y)


def cholesky(A):
    """A = L L^T for symmetric positive definite A. Cost: n^3/3 flops."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0:
            raise np.linalg.LinAlgError("matrix is not positive definite")
        L[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L


def cholesky_solve(L, b):
    return back_substitution(L.T, forward_substitution(L, b))


def determinant_via_lu(A):
    P, L, U = lu_partial_pivot(A)
    sign = np.linalg.det(P)  # +-1; permutation parity
    return sign * np.prod(np.diag(U))


def inverse_via_lu(A):
    P, L, U = lu_partial_pivot(A)
    n = A.shape[0]
    return np.column_stack([lu_solve(P, L, U, e) for e in np.eye(n)])


def condition_number(A, p=2):
    A = np.asarray(A, dtype=float)
    if p == 2:
        s = np.linalg.svd(A, compute_uv=False)
        return s[0] / s[-1]
    return np.linalg.norm(A, p) * np.linalg.norm(np.linalg.inv(A), p)


def hilbert(n):
    i = np.arange(1, n + 1)
    return 1.0 / (i[:, None] + i[None, :] - 1)


def tridiagonal_solve(a, b, c, d):
    """Thomas algorithm. a: sub-diagonal (n-1), b: diagonal (n),
    c: super-diagonal (n-1), d: rhs (n). Cost O(n)."""
    n = len(b)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
