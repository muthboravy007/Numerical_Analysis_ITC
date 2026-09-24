"""P6 starter: PageRank and spectral bisection on Zachary's karate club.

Run: python p6_graph_analytics.py
"""
import numpy as np
import scipy.sparse as sp

EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19),
         (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 8), (2, 9),
         (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16), (6, 16),
         (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33),
         (20, 32), (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27),
         (24, 31), (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33),
         (31, 32), (31, 33), (32, 33)]
# Club split after the conflict: 1 = followed the instructor (node 0), 0 = followed the administrator (node 33)
FACTION = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def adjacency(edges, n):
    r, c = zip(*edges)
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return (A + A.T).tocsr()


def pagerank_power(A, alpha=0.85, tol=1e-12, max_iter=1000):
    n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    P_T = (sp.diags(1 / np.where(deg > 0, deg, 1)) @ A).T.tocsr()
    dangling = deg == 0
    x = np.full(n, 1 / n)
    for k in range(max_iter):
        xn = alpha * (P_T @ x + x[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(xn - x).sum() < tol:
            return xn, k + 1
        x = xn
    return x, max_iter


def pagerank_gauss_seidel(A, alpha=0.85):
    """TODO: solve (I - alpha P^T) x = (1 - alpha)/n 1 by Gauss-Seidel and compare iteration counts."""
    raise NotImplementedError


if __name__ == "__main__":
    n = 34
    A = adjacency(EDGES, n)
    for alpha in (0.5, 0.85, 0.99):
        x, its = pagerank_power(A, alpha)
        print(f"alpha {alpha}: {its} power iterations, top-3 nodes {np.argsort(-x)[:3]}")
    d = np.asarray(A.sum(1)).ravel()
    L = np.diag(d) - A.toarray()
    Lsym = np.eye(n) - A.toarray() / np.sqrt(np.outer(d, d))
    for name, M in (("L", L), ("L_sym", Lsym)):
        w, V = np.linalg.eigh(M)
        split = (V[:, 1] > 0).astype(int)
        acc = max(np.mean(split == FACTION), np.mean(split != FACTION))
        print(f"{name}: lambda_2 = {w[1]:.4f}; Fiedler split agrees with the real faction split for {acc:.0%} of members")
    # TODO: large SNAP graphs, Gauss-Seidel/GMRES, SBM phase transition, label propagation with CG
