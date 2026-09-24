"""Chapter 6 — Eigenvalue problems.

Library equivalents: ``np.linalg.eig`` / ``eigh``, ``scipy.sparse.linalg.eigsh``,
``sklearn.decomposition.PCA``, ``networkx.pagerank``.
"""

import numpy as np


def power_iteration(A, x0=None, tol=1e-10, max_iter=10_000):
    """Dominant eigenpair. Returns (lambda, v, history_of_lambda)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) if x0 is None else np.array(x0, dtype=float)
    x /= np.linalg.norm(x)
    lam = x @ A @ x
    hist = [lam]
    for _ in range(max_iter):
        y = A @ x
        x_new = y / np.linalg.norm(y)
        lam_new = x_new @ A @ x_new  # Rayleigh quotient
        hist.append(lam_new)
        if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
            return lam_new, x_new, hist
        x, lam = x_new, lam_new
    return lam, x, hist


def inverse_iteration(A, shift=0.0, x0=None, tol=1e-12, max_iter=1000):
    """Eigenpair with eigenvalue closest to `shift`."""
    import scipy.linalg as sla

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    lu = sla.lu_factor(A - shift * np.eye(n))
    x = np.ones(n) if x0 is None else np.array(x0, dtype=float)
    x /= np.linalg.norm(x)
    lam = x @ A @ x
    hist = [lam]
    for _ in range(max_iter):
        y = sla.lu_solve(lu, x)
        x = y / np.linalg.norm(y)
        lam_new = x @ A @ x
        hist.append(lam_new)
        if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
            return lam_new, x, hist
        lam = lam_new
    return lam, x, hist


def rayleigh_quotient_iteration(A, x0, tol=1e-12, max_iter=50):
    """Cubically convergent for symmetric A."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = np.array(x0, dtype=float)
    x /= np.linalg.norm(x)
    lam = x @ A @ x
    hist = [lam]
    for _ in range(max_iter):
        try:
            y = np.linalg.solve(A - lam * np.eye(n), x)
        except np.linalg.LinAlgError:
            break  # shift hit an eigenvalue exactly
        x = y / np.linalg.norm(y)
        lam_new = x @ A @ x
        hist.append(lam_new)
        if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
            return lam_new, x, hist
        lam = lam_new
    return lam, x, hist


def qr_algorithm(A, tol=1e-12, max_iter=10_000, shift=True):
    """Unsymmetric-safe basic QR iteration (for real eigenvalues).
    Returns approximate eigenvalues (diagonal of the limit)."""
    Ak = np.array(A, dtype=float)
    n = Ak.shape[0]
    for _ in range(max_iter):
        mu = Ak[-1, -1] if shift else 0.0
        Q, R = np.linalg.qr(Ak - mu * np.eye(n))
        Ak = R @ Q + mu * np.eye(n)
        if np.max(np.abs(np.tril(Ak, -1))) < tol:
            break
    return np.sort(np.diag(Ak))[::-1]


def deflate(A, lam, v):
    """Hotelling deflation for symmetric A: A - lam v v^T."""
    v = v / np.linalg.norm(v)
    return A - lam * np.outer(v, v)


def gershgorin_discs(A):
    A = np.asarray(A)
    centers = np.diag(A)
    radii = np.sum(np.abs(A), axis=1) - np.abs(centers)
    return list(zip(centers, radii))


def pca(X, k):
    """PCA via SVD of the centred data matrix.

    Returns (scores, components, explained_variance_ratio).
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2 / (X.shape[0] - 1)
    return Xc @ Vt[:k].T, Vt[:k], var[:k] / var.sum()


def pagerank(adj, damping=0.85, tol=1e-12, max_iter=1000):
    """PageRank by power iteration. adj[i, j] = 1 if page i links to page j."""
    adj = np.asarray(adj, dtype=float)
    n = adj.shape[0]
    out = adj.sum(axis=1)
    # column-stochastic transition matrix; dangling nodes link everywhere
    P = np.where(out[:, None] > 0, adj / np.where(out == 0, 1, out)[:, None], 1.0 / n).T
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = damping * P @ r + (1 - damping) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def symmetric_power_method(A, x0, tol=1e-10, max_iter=1000):
    """B&F Alg. 9.2: 2-norm scaling, Rayleigh quotient estimate (error ~ ratio^{2k})."""
    A = np.asarray(A, dtype=float)
    x = np.array(x0, dtype=float)
    x /= np.linalg.norm(x)
    hist = []
    for _ in range(max_iter):
        y = A @ x
        mu = x @ y
        hist.append(mu)
        ny = np.linalg.norm(y)
        if ny == 0:
            return 0.0, x, hist
        err = np.linalg.norm(x - y / ny)
        x = y / ny
        if err < tol:
            break
    return mu, x, hist


def power_method_inf(A, x0, max_iter=100, tol=1e-10):
    """B&F Alg. 9.1: infinity-norm scaling; mu = y_p where |x_p| = 1."""
    A = np.asarray(A, dtype=float)
    x = np.array(x0, dtype=float)
    p = np.argmax(np.abs(x))
    x /= x[p]
    hist = []
    for _ in range(max_iter):
        y = A @ x
        mu = y[p]
        hist.append(mu)
        p = np.argmax(np.abs(y))
        if y[p] == 0:
            return 0.0, x, hist
        err = np.max(np.abs(x - y / y[p]))
        x = y / y[p]
        if err < tol:
            break
    return mu, x, hist


def householder_tridiagonal(A):
    """Reduce symmetric A to tridiagonal T = Q^T A Q by Householder (B&F Alg. 9.5)."""
    T = np.array(A, dtype=float)
    n = T.shape[0]
    Q = np.eye(n)
    for k in range(n - 2):
        x = T[k + 1:, k]
        alpha = -np.copysign(np.linalg.norm(x), x[0] if x[0] != 0 else 1.0)
        v = x.copy()
        v[0] -= alpha
        nv = np.linalg.norm(v)
        if nv < 1e-15:
            continue
        v /= nv
        H = np.eye(n)
        H[k + 1:, k + 1:] -= 2 * np.outer(v, v)
        T = H @ T @ H
        Q = Q @ H
    return T, Q


def wielandt_deflation(A, lam, v):
    """B&F 9.3: Wielandt deflation with x = (row i of A) / (lam v_i), i = argmax |v_i|.
    Returns B = A - lam v x^T (with row i zero); its other eigenvalues match A's."""
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.argmax(np.abs(v))
    x = A[i, :] / (lam * v[i])
    return A - lam * np.outer(v, x)


def svd_via_eig(A):
    """Teaching-only SVD from the eigen-decomposition of A^T A (squares kappa!)."""
    A = np.asarray(A, dtype=float)
    w, V = np.linalg.eigh(A.T @ A)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    s = np.sqrt(np.clip(w, 0, None))
    r = int(np.sum(s > 1e-12 * s[0]))
    U = A @ V[:, :r] / s[:r]
    return U, s, V.T
