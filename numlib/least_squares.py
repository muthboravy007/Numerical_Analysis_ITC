"""Chapter 5 — Least squares, QR and the SVD.

Library equivalents: ``np.linalg.lstsq``, ``np.linalg.qr``,
``np.linalg.svd``, ``scipy.linalg.lstsq``, ``sklearn.linear_model.Ridge``.
"""

import numpy as np

from .linalg_direct import back_substitution, cholesky, cholesky_solve


def normal_equations(A, b):
    """Solve min ||Ax - b|| via A^T A x = A^T b (Cholesky).
    Fast but squares the condition number."""
    A = np.asarray(A, dtype=float)
    L = cholesky(A.T @ A)
    return cholesky_solve(L, A.T @ b)


def gram_schmidt(A, modified=True):
    """Thin QR by (modified) Gram-Schmidt."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    for j in range(n):
        if not modified:
            for i in range(j):
                R[i, j] = Q[:, i] @ A[:, j]
                V[:, j] -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        if modified:
            for k in range(j + 1, n):
                R[j, k] = Q[:, j] @ V[:, k]
                V[:, k] -= R[j, k] * Q[:, j]
    return Q, R


def householder_qr(A):
    """Full QR by Householder reflections: A = Q R, Q is m x m orthogonal."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        normx = np.linalg.norm(x)
        if normx == 0:
            continue
        v = x.copy()
        v[0] += np.copysign(normx, x[0])  # sign choice avoids cancellation
        v /= np.linalg.norm(v)
        R[k:, :] -= 2.0 * np.outer(v, v @ R[k:, :])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return Q, R


def qr_least_squares(A, b):
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    Q, R = householder_qr(A)
    return back_substitution(R[:n, :n], (Q.T @ b)[:n])


def svd_least_squares(A, b, rcond=1e-12):
    """Minimum-norm least-squares solution via the pseudoinverse."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > rcond * s[0]
    return Vt[keep].T @ ((U[:, keep].T @ b) / s[keep])


def ridge(A, b, lam):
    """Tikhonov / ridge regression via the SVD: filter factors s/(s^2+lam)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt.T @ ((s / (s ** 2 + lam)) * (U.T @ b))


def truncated_svd(A, k):
    """Best rank-k approximation (Eckart-Young)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] * s[:k] @ Vt[:k]


def polyfit_design(x, degree):
    """Vandermonde design matrix [1, x, x^2, ...]."""
    return np.vander(np.asarray(x, dtype=float), degree + 1, increasing=True)


def r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot
