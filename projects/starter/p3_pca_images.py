"""P3 starter: PCA of image data and a reference randomised SVD.

Run: python p3_pca_images.py
"""
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def pca_svd(X, k):
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, s, Vt[:k]


def randomized_svd(A, k, p=10, q=1, seed=0):
    """Halko-Martinsson-Tropp range finder with q power iterations."""
    rng = np.random.default_rng(seed)
    Y = A @ rng.standard_normal((A.shape[1], k + p))
    for _ in range(q):
        Y, _ = np.linalg.qr(Y)
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    Ub, s, Vt = np.linalg.svd(Q.T @ A, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def subspace_iteration(A, k, iters=50, seed=0):
    """TODO: block power (subspace) iteration for the top-k right singular vectors.
    Track the convergence of the subspace angle against (sigma_{k+1}/sigma_k)^(2 it)."""
    raise NotImplementedError


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    mu, s, V = pca_svd(X, 20)
    var = s ** 2 / np.sum(s ** 2)
    print("digits:", X.shape, " explained variance of first 10 PCs:", var[:10].round(3))
    print("components for 90% variance:", int(np.searchsorted(np.cumsum(var), 0.9) + 1))
    Xc = X - mu
    t0 = time.perf_counter(); Ur, sr, Vr = randomized_svd(Xc, 20); t1 = time.perf_counter()
    print(f"randomised SVD (k=20): max rel. error of sigma {np.max(np.abs(sr - s[:20]) / s[:20]):.2e}, {t1 - t0:.4f}s")
    for k in (2, 5, 10, 20, 40):
        _, _, Vk = pca_svd(X, k)
        acc = cross_val_score(LogisticRegression(max_iter=3000), (X - mu) @ Vk.T, y, cv=5).mean()
        print(f"logistic regression on {k:2d} PCs: 5-fold CV accuracy {acc:.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.semilogy(np.arange(1, len(s) + 1), s ** 2 / (len(X) - 1), "o-")
        plt.xlabel("component"); plt.ylabel("variance"); plt.title("Scree plot (digits)")
        plt.savefig("p3_scree.png", dpi=100)
        print("saved p3_scree.png")
    except ImportError:
        pass
    # TODO: subspace iteration, Olivetti faces (fetch_olivetti_faces), scaling study, bootstrap stability
