"""Chapter 4 — Iterative methods for linear systems.

Library equivalents: ``scipy.sparse.linalg.cg``, ``gmres``, ``bicgstab``,
``scipy.sparse`` matrix formats.

Each solver returns ``(x, residual_norms)``.
"""

import numpy as np


def _check(A, b, x0):
    A = np.asarray(A, dtype=float) if not hasattr(A, "tocsr") else A
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    return A, b, x


def jacobi(A, b, x0=None, tol=1e-10, max_iter=10_000):
    A, b, x = _check(A, b, x0)
    D = np.diag(A)
    R = A - np.diag(D)
    bnorm = np.linalg.norm(b) or 1.0
    res = [np.linalg.norm(b - A @ x) / bnorm]
    for _ in range(max_iter):
        x = (b - R @ x) / D
        res.append(np.linalg.norm(b - A @ x) / bnorm)
        if res[-1] < tol:
            break
    return x, res


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=10_000):
    return sor(A, b, omega=1.0, x0=x0, tol=tol, max_iter=max_iter)


def sor(A, b, omega=1.5, x0=None, tol=1e-10, max_iter=10_000):
    """Successive over-relaxation. omega = 1 gives Gauss-Seidel."""
    A, b, x = _check(A, b, x0)
    n = len(b)
    bnorm = np.linalg.norm(b) or 1.0
    res = [np.linalg.norm(b - A @ x) / bnorm]
    for _ in range(max_iter):
        for i in range(n):
            sigma = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        res.append(np.linalg.norm(b - A @ x) / bnorm)
        if res[-1] < tol:
            break
    return x, res


def iteration_matrix(A, method="jacobi", omega=1.0):
    """Return T with x_{k+1} = T x_k + c for the stationary methods."""
    A = np.asarray(A, dtype=float)
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    if method == "jacobi":
        return np.linalg.solve(D, L + U)
    if method == "gauss_seidel":
        return np.linalg.solve(D - L, U)
    if method == "sor":
        return np.linalg.solve(D - omega * L, (1 - omega) * D + omega * U)
    raise ValueError(method)


def spectral_radius(T):
    return np.max(np.abs(np.linalg.eigvals(T)))


def optimal_sor_omega(rho_jacobi):
    """Young's formula for consistently ordered matrices."""
    return 2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi ** 2))


def steepest_descent(A, b, x0=None, tol=1e-10, max_iter=10_000):
    A, b, x = _check(A, b, x0)
    r = b - A @ x
    bnorm = np.linalg.norm(b) or 1.0
    res = [np.linalg.norm(r) / bnorm]
    for _ in range(max_iter):
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        res.append(np.linalg.norm(r) / bnorm)
        if res[-1] < tol:
            break
    return x, res


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None, M_inv=None):
    """(Preconditioned) CG for symmetric positive definite A.

    M_inv: callable applying the inverse preconditioner, e.g.
    ``lambda r: r / np.diag(A)`` for Jacobi preconditioning.
    """
    A, b, x = _check(A, b, x0)
    n = len(b)
    max_iter = max_iter or 10 * n
    apply_M = M_inv or (lambda r: r)
    r = b - A @ x
    z = apply_M(r)
    p = z.copy()
    rz = r @ z
    bnorm = np.linalg.norm(b) or 1.0
    res = [np.linalg.norm(r) / bnorm]
    for _ in range(max_iter):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        res.append(np.linalg.norm(r) / bnorm)
        if res[-1] < tol:
            break
        z = apply_M(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, res


def poisson_1d(n):
    """The n x n matrix tridiag(-1, 2, -1) — the classic SPD test matrix."""
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def poisson_2d_sparse(m):
    """Sparse 5-point Laplacian on an m x m interior grid (size m^2)."""
    import scipy.sparse as sp

    T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(m, m))
    I = sp.identity(m)
    return (sp.kron(I, T) + sp.kron(T, I)).tocsr()
