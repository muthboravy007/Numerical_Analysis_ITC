"""Reproduces every numerical example in Chapter 14 (Symmetric Matrices, SVD and PCA).
Run from the repository root:  python lectures/code/ch14.py"""

import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
from sklearn.datasets import load_digits, load_iris, load_sample_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import eigen as eg, least_squares as lsq  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=120)
rng = np.random.default_rng(0)

print("=== 14.1 Spectral theorem ===")
A = np.array([[6.0, 2, 1], [2, 3, 1], [1, 1, 1]]); w, Q = np.linalg.eigh(A)
print("Ex 14.1.1 eig", w, "Q\n", Q, "\n Q^TQ\n", Q.T @ Q)
print("Ex 14.1.2 spectral sum reconstruction error", np.linalg.norm(sum(w[i] * np.outer(Q[:, i], Q[:, i]) for i in range(3)) - A), "rank-1 terms norms", w)
S = Q @ np.diag(np.sqrt(w)) @ Q.T
print("Ex 14.1.3 sqrt(A)\n", S, "\n S^2 - A", np.linalg.norm(S @ S - A), " exp(A) via eig vs expm diff", np.linalg.norm(Q @ np.diag(np.exp(w)) @ Q.T - sla.expm(A)))
xs = rng.standard_normal((100000, 3)); rq = np.einsum("ij,jk,ik->i", xs, A, xs) / np.einsum("ij,ij->i", xs, xs)
print("Ex 14.1.4 Rayleigh quotient range over random x:", rq.min(), rq.max(), "eig min/max", w.min(), w.max())
v1 = Q[:, -1]
P = np.eye(3) - np.outer(v1, v1); Bm = P @ A @ P; ws = np.linalg.eigvalsh(Bm)
print("Ex 14.1.5 max RQ orthogonal to v1 = lambda2:", ws.max(), w[1])
Aper = A + 0.01 * np.array([[1.0, -1, 0.5], [-1, 0.3, 0.2], [0.5, 0.2, -0.4]])
print("Ex 14.1.6 Weyl: eig shift", np.linalg.eigvalsh(Aper) - w, "||E||_2", 0.01 * np.linalg.norm(np.array([[1.0, -1, 0.5], [-1, 0.3, 0.2], [0.5, 0.2, -0.4]]), 2))

print("\n=== 14.2 Positive definite matrices and quadratic forms ===")
M = np.array([[2.0, -1, 0], [-1, 2, -1], [0, -1, 2]])
print("Ex 14.2.1 eig", np.linalg.eigvalsh(M), "minors", [np.linalg.det(M[:k, :k]) for k in (1, 2, 3)], "chol\n", np.linalg.cholesky(M))
Xd = rng.standard_normal((50, 4)); G = Xd.T @ Xd
Z35 = rng.standard_normal((3, 5))
print("Ex 14.2.2 Gram eig", np.linalg.eigvalsh(G), " rank-deficient (n=3<p=5) Gram eig", np.linalg.eigvalsh(Z35.T @ Z35))
Kq = np.array([[5.0, 2], [2, 2]]); wk, Vk = np.linalg.eigh(Kq)
print("Ex 14.2.3 ellipse x^TKx=1: eig", wk, "axes dirs\n", Vk, "\n semi-axes", 1 / np.sqrt(wk))
Hs = lambda x, y: np.array([[12 * x ** 2 - 4, 0.0], [0.0, 2.0]])
for pt in ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)):
    print(f"Ex 14.2.4 f=x^4-2x^2+y^2 Hessian eig at {pt}: {np.linalg.eigvalsh(Hs(*pt))}")
Sg = np.array([[4.0, 1.2, 0.8], [1.2, 1.0, 0.3], [0.8, 0.3, 2.0]]); mu = np.array([1.0, 0, 2])
S11 = Sg[:1, :1]; S12 = Sg[:1, 1:]; S22 = Sg[1:, 1:]; x2 = np.array([0.5, 2.5])
cm = mu[0] + S12 @ np.linalg.solve(S22, x2 - mu[1:]); cv = S11 - S12 @ np.linalg.solve(S22, S12.T)
print("Ex 14.2.5 conditional mean", cm, "conditional var (Schur complement)", cv, "vs marginal var", S11)
Kr = np.exp(-0.5 * np.subtract.outer(np.linspace(0, 1, 30), np.linspace(0, 1, 30)) ** 2 / 0.3 ** 2)
wr = np.linalg.eigvalsh(Kr)
print("Ex 14.2.6 RBF kernel 30 pts: min eig", wr.min(), "max", wr.max(), "cond", wr.max() / max(wr.min(), 1e-300), " +1e-6 jitter cond", (wr.max() + 1e-6) / (wr.min() + 1e-6))

print("\n=== 14.3 SVD ===")
B = np.array([[1.0, 1], [1, -1], [2, 0]]); U, s, Vt = np.linalg.svd(B)
print("Ex 14.3.1 B^TB", B.T @ B, "s", s, "U\n", U, "\n Vt\n", Vt)
C = np.array([[1.0, 2, 3], [2, 4, 6], [1, 0, 1]]); U, s, Vt = np.linalg.svd(C)
print("Ex 14.3.2 s", s, "rank", np.sum(s > 1e-10), "null space v3", Vt[2], "C v3", C @ Vt[2], "left null u3", U[:, 2], "u3^T C", U[:, 2] @ C)
D = np.array([[3.0, 1], [1, 3]]); U, s, Vt = np.linalg.svd(D)
print("Ex 14.3.3 norms: ||D||_2", s[0], "||D||_F", np.linalg.norm(D, 'fro'), np.sqrt(np.sum(s ** 2)), "cond", s[0] / s[1], "nuclear", s.sum())
Sy = np.array([[2.0, 3], [3, -2]]); print("Ex 14.3.4 symmetric indefinite eig", np.linalg.eigvalsh(Sy), "singular values", np.linalg.svd(Sy, compute_uv=False))
E = 1e-3 * rng.standard_normal((3, 3)); print("Ex 14.3.5 singular value perturbation", np.linalg.svd(C + E, compute_uv=False) - np.linalg.svd(C, compute_uv=False), "||E||_2", np.linalg.norm(E, 2))
H = eg.householder_tridiagonal  # noqa
Ub, sb, Vbt = eg.svd_via_eig(np.vander(np.linspace(0, 1, 20), 8))
print("Ex 14.3.6 smallest sv of Vandermonde(20x8): via eig(A^TA)", sb[-1], "via svd", np.linalg.svd(np.vander(np.linspace(0, 1, 20), 8), compute_uv=False)[-1])

print("\n=== 14.4 Low-rank approximation ===")
img = load_sample_image("china.jpg").mean(axis=2) / 255.0
U, s, Vt = np.linalg.svd(img, full_matrices=False); m, n = img.shape
print("Ex 14.4.1 image", img.shape, "top singular values", s[:5])
for k in (5, 20, 50, 100):
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]; rel = np.linalg.norm(img - Ak) / np.linalg.norm(img)
    print(f"   k={k}: rel Frobenius err {rel:.4f} = sqrt(tail) {np.sqrt(np.sum(s[k:] ** 2)) / np.sqrt(np.sum(s ** 2)):.4f}; storage {k * (m + n + 1) / (m * n):.3f}")
energy = np.cumsum(s ** 2) / np.sum(s ** 2)
print("Ex 14.4.2 rank for 90/99/99.9% energy:", [int(np.searchsorted(energy, q) + 1) for q in (0.9, 0.99, 0.999)])
L = rng.standard_normal((200, 3)) @ rng.standard_normal((3, 100)); Nn = 0.5 * rng.standard_normal((200, 100))
U, s, Vt = np.linalg.svd(L + Nn, full_matrices=False)
print("Ex 14.4.3 noisy rank-3: first 6 sv", s[:6], "MP noise edge ~", 0.5 * (math.sqrt(200) + math.sqrt(100)))
Lh = (U[:, :3] * s[:3]) @ Vt[:3]
print("   denoise rel err: noisy", np.linalg.norm(Nn) / np.linalg.norm(L), "rank-3", np.linalg.norm(Lh - L) / np.linalg.norm(L))
def rsvd(A, k, p=10, q=1, seed=0):
    Om = np.random.default_rng(seed).standard_normal((A.shape[1], k + p)); Y = A @ Om
    for _ in range(q):
        Y = A @ (A.T @ Y)
    Qr, _ = np.linalg.qr(Y); Bq = Qr.T @ A; Ub, sb, Vb = np.linalg.svd(Bq, full_matrices=False)
    return (Qr @ Ub)[:, :k], sb[:k], Vb[:k]
Abig = rng.standard_normal((2000, 50)) @ np.diag(0.8 ** np.arange(50)) @ rng.standard_normal((50, 1500))
t0 = time.perf_counter(); s_full = np.linalg.svd(Abig, compute_uv=False); t1 = time.perf_counter(); Ur, sr, Vr = rsvd(Abig, 10); t2 = time.perf_counter()
print(f"Ex 14.4.4 randomized SVD k=10: rel err of s {np.max(np.abs(sr - s_full[:10]) / s_full[:10]):.2e}; time full {t1 - t0:.3f}s rsvd {t2 - t1:.3f}s")
Ak = (Ur * sr) @ Vr; print("   ||A - A_10||_2 rsvd", np.linalg.norm(Abig - Ak, 2), "optimal s11", s_full[10])
Mm = np.array([[5.0, 5, 0, 1], [4, 5, 1, 0], [1, 0, 5, 4], [0, 1, 4, 5], [5, 4, 1, 1]])
U, s, Vt = np.linalg.svd(Mm); print("Ex 14.4.5 scree of ratings", s, "energy", np.cumsum(s ** 2) / np.sum(s ** 2))

print("\n=== 14.5 PCA ===")
iris = load_iris(); X = iris.data; y = iris.target
Xc = X - X.mean(0); Cv = Xc.T @ Xc / (len(X) - 1); w, V = np.linalg.eigh(Cv); w, V = w[::-1], V[:, ::-1]
print("Ex 14.5.1 means", X.mean(0), "\n cov\n", Cv, "\n eig", w, "\n explained", w / w.sum(), "\n PC1", V[:, 0])
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print("Ex 14.5.2 s^2/(n-1)", s ** 2 / (len(X) - 1), "|Vt[0]| vs |V[:,0]|", np.abs(Vt[0]), "scores first 3 rows\n", (U * s)[:3, :2])
Z = (X - X.mean(0)) / X.std(0, ddof=1); wz = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
print("Ex 14.5.3 standardized PCA eig", wz, "explained", wz / wz.sum())
Xscale = X.copy(); Xscale[:, 0] *= 10
wsc = np.linalg.eigvalsh(np.cov(Xscale.T))[::-1]; print("   sepal length in mm instead of cm: explained", wsc / wsc.sum())
T2 = Xc @ V[:, :2]
for k in range(3):
    print(f"Ex 14.5.4 class {iris.target_names[k]}: PC1 mean {T2[y == k, 0].mean():.3f} PC2 mean {T2[y == k, 1].mean():.3f}")
for k in (1, 2, 3, 4):
    Xr = Xc @ V[:, :k] @ V[:, :k].T
    print(f"Ex 14.5.5 k={k}: reconstruction SSE/(n-1) {np.sum((Xc - Xr) ** 2) / (len(X) - 1):.4f} = sum of dropped eig {w[k:].sum():.4f}")
Wh = Xc @ V / np.sqrt(w); print("Ex 14.5.6 whitened covariance\n", np.cov(Wh.T))
digits = load_digits(); Xd = digits.data; yd = digits.target
Xdc = Xd - Xd.mean(0); Ud, sd, Vdt = np.linalg.svd(Xdc, full_matrices=False); ev = sd ** 2 / np.sum(sd ** 2)
print("Ex 14.5.7 digits 64-d: components for 80/90/95%:", [int(np.searchsorted(np.cumsum(ev), q) + 1) for q in (0.8, 0.9, 0.95)])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
for k in (2, 10, 20, 64):
    Zk = Xdc @ Vdt[:k].T
    acc = cross_val_score(LogisticRegression(max_iter=5000), Zk, yd, cv=5).mean()
    print(f"   PCA({k}) + logistic regression 5-fold accuracy {acc:.4f}")

print("\n=== 14.6 Applications ===")
terms = ["data", "model", "learning", "matrix", "vector", "football", "goal", "team"]
Tdm = np.array([[2, 1, 3, 0, 0, 0], [1, 2, 2, 0, 1, 0], [1, 3, 2, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 3, 2, 1], [0, 0, 0, 2, 3, 1], [0, 0, 0, 1, 2, 2]], dtype=float)
U, s, Vt = np.linalg.svd(Tdm, full_matrices=False); D2 = (np.diag(s[:2]) @ Vt[:2]).T
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print("Ex 14.6.1 LSA s", s[:4], "doc coords\n", D2, "\n cos(d1,d3) raw", cos(Tdm[:, 0], Tdm[:, 2]), "LSA", cos(D2[0], D2[2]), " cos(d1,d4) raw", cos(Tdm[:, 0], Tdm[:, 3]), "LSA", cos(D2[0], D2[3]))
q = np.zeros(8); q[terms.index("matrix")] = 1; qhat = q @ U[:, :2] / s[:2]
print("   query 'matrix' in LSA space vs docs cos:", [round(cos(qhat, d_ / s[:2] * s[:2]), 3) for d_ in (Vt[:2].T)])
R = np.array([[5, 4, np.nan, 1], [4, np.nan, 1, 1], [1, 1, np.nan, 5], [np.nan, 1, 5, 4], [5, 4, 2, np.nan]])
mask = ~np.isnan(R); k = 2; P = 0.1 * rng.standard_normal((5, k)) + 1; Qm = 0.1 * rng.standard_normal((4, k)) + 1; lam = 0.1
for it in range(200):
    for u in range(5):
        idx = mask[u]; P[u] = np.linalg.solve(Qm[idx].T @ Qm[idx] + lam * np.eye(k), Qm[idx].T @ R[u, idx])
    for i in range(4):
        idx = mask[:, i]; Qm[i] = np.linalg.solve(P[idx].T @ P[idx] + lam * np.eye(k), P[idx].T @ R[idx, i])
pred = P @ Qm.T; rmse = np.sqrt(np.mean((pred[mask] - R[mask]) ** 2))
print("Ex 14.6.2 ALS completion train RMSE", rmse, "\n predictions for missing:", [(int(a), int(b), round(pred[a, b], 2)) for a, b in zip(*np.where(~mask))])
faces = Xd[yd == 3]; mean3 = faces.mean(0); U3, s3, V3t = np.linalg.svd(faces - mean3, full_matrices=False)
test = Xd[yd == 3][0]; other = Xd[yd == 8][0]
def recon_err(v, k=10):
    c = (v - mean3) @ V3t[:k].T; return np.linalg.norm(v - mean3 - c @ V3t[:k])
print("Ex 14.6.3 'eigen-threes' reconstruction error: a 3", recon_err(test), "an 8", recon_err(other))
cities = np.array([[0, 0], [3, 0], [3, 4], [0, 4], [1.5, 2.0]]); Dm = np.linalg.norm(cities[:, None] - cities[None], axis=2)
n = 5; J = np.eye(n) - np.ones((n, n)) / n; Bc = -0.5 * J @ (Dm ** 2) @ J; wb, Vb = np.linalg.eigh(Bc); wb, Vb = wb[::-1], Vb[:, ::-1]
Y = Vb[:, :2] * np.sqrt(wb[:2]); Dy = np.linalg.norm(Y[:, None] - Y[None], axis=2)
print("Ex 14.6.4 classical MDS eig", wb, "max distance error", np.max(np.abs(Dy - Dm)))
xa = rng.standard_normal(500); xb = 0.8 * xa + 0.6 * rng.standard_normal(500); X2 = np.column_stack([xa, xb]); X2 = X2 - X2.mean(0)
w2, V2 = np.linalg.eigh(np.cov(X2.T)); ang = math.degrees(math.atan2(V2[1, -1], V2[0, -1]))
b_ols = np.polyfit(xa, xb, 1)[0]
print("Ex 14.6.5 PC1 (total least squares) slope", V2[1, -1] / V2[0, -1], "angle", ang, "vs OLS slope y~x", b_ols, "vs 1/(OLS x~y)", 1 / np.polyfit(xb, xa, 1)[0])
