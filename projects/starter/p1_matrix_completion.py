"""P1 starter: matrix completion for recommender systems.

Run:  python p1_matrix_completion.py            (synthetic data, offline)
      python p1_matrix_completion.py u.data     (MovieLens 100K ratings file)
"""
import sys

import numpy as np
import scipy.sparse as sp


def load_movielens(path):
    raw = np.loadtxt(path, dtype=int)  # user item rating timestamp
    return raw[:, 0] - 1, raw[:, 1] - 1, raw[:, 2].astype(float)


def synthetic(m=500, n=300, k=5, frac=0.1, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    U, V = rng.standard_normal((m, k)), rng.standard_normal((n, k))
    mask = rng.random((m, n)) < frac
    rows, cols = np.nonzero(mask)
    vals = (U @ V.T)[rows, cols] / np.sqrt(k) + 3.5 + noise * rng.standard_normal(len(rows))
    return rows, cols, vals


def split(rows, cols, vals, seed=0, frac=(0.8, 0.1)):
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(vals))
    a, b = int(frac[0] * len(p)), int((frac[0] + frac[1]) * len(p))
    return [(rows[i], cols[i], vals[i]) for i in (p[:a], p[a:b], p[b:])]


def rmse(pred, vals):
    return float(np.sqrt(np.mean((pred - vals) ** 2)))


def bias_baseline(train, m, n, lam=5.0, iters=10):
    """mu + b_i + c_j by alternating regularised least squares."""
    r, c, v = train
    mu = v.mean()
    b, cc = np.zeros(m), np.zeros(n)
    for _ in range(iters):
        b = np.bincount(r, v - mu - cc[c], m) / (np.bincount(r, minlength=m) + lam)
        cc = np.bincount(c, v - mu - b[r], n) / (np.bincount(c, minlength=n) + lam)
    return mu, b, cc


def als(train, m, n, k=10, lam=0.1, iters=15, seed=0, verbose=True):
    """Alternating least squares on the residual after global mean removal.
    Each row update solves a k x k ridge system (see brief)."""
    r, c, v = train
    mu = v.mean()
    rng = np.random.default_rng(seed)
    U = 0.1 * rng.standard_normal((m, k))
    V = 0.1 * rng.standard_normal((n, k))
    R = sp.csr_matrix((v - mu, (r, c)), shape=(m, n))
    Rt = R.T.tocsr()
    for it in range(iters):
        for i in range(m):
            idx = R.indices[R.indptr[i]:R.indptr[i + 1]]
            if len(idx):
                Vi = V[idx]
                U[i] = np.linalg.solve(Vi.T @ Vi + lam * len(idx) * np.eye(k), Vi.T @ R.data[R.indptr[i]:R.indptr[i + 1]])
        for j in range(n):
            idx = Rt.indices[Rt.indptr[j]:Rt.indptr[j + 1]]
            if len(idx):
                Uj = U[idx]
                V[j] = np.linalg.solve(Uj.T @ Uj + lam * len(idx) * np.eye(k), Uj.T @ Rt.data[Rt.indptr[j]:Rt.indptr[j + 1]])
        if verbose:
            pred = mu + np.sum(U[r] * V[c], axis=1)
            print(f"  ALS iter {it + 1:2d}: train RMSE {rmse(pred, v):.4f}")
    return mu, U, V


def sgd(train, m, n, k=10, lam=0.05, lr=0.01, epochs=20, seed=0):
    """TODO: implement SGD for the factorisation model (shuffle each epoch,
    update u_i and v_j with the gradient of one observed entry)."""
    raise NotImplementedError


def soft_impute(train, m, n, lam, iters=100):
    """TODO: implement soft-impute (singular-value soft-thresholding).
    Hint: Z_{t+1} = S_lam(P_Omega(M) + P_Omega^c(Z_t)); use a warm-started lambda path."""
    raise NotImplementedError


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rows, cols, vals = load_movielens(sys.argv[1])
    else:
        print("No MovieLens file given: using synthetic rank-5 data (10% observed).")
        rows, cols, vals = synthetic()
    m, n = rows.max() + 1, cols.max() + 1
    train, valid, test = split(rows, cols, vals)
    print(f"{m} users, {n} items, {len(vals)} ratings")
    print("global mean RMSE (test):", round(rmse(np.full(len(test[2]), train[2].mean()), test[2]), 4))
    mu, b, cc = bias_baseline(train, m, n)
    print("bias baseline RMSE (test):", round(rmse(mu + b[test[0]] + cc[test[1]], test[2]), 4))
    for k in (2, 5, 10):
        muA, U, V = als(train, m, n, k=k, lam=0.1, iters=10, verbose=False)
        pv = muA + np.sum(U[valid[0]] * V[valid[1]], axis=1)
        pt = muA + np.sum(U[test[0]] * V[test[1]], axis=1)
        print(f"ALS k={k:2d}: valid RMSE {rmse(pv, valid[2]):.4f}  test RMSE {rmse(pt, test[2]):.4f}")
    # TODO: SGD, soft-impute, time-vs-RMSE plot, phase diagram on synthetic data
