"""Numerical answers for exercises/ch14_exercises.md."""
import os
import sys
import time

import numpy as np
from scipy.linalg import sqrtm, orthogonal_procrustes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

np.set_printoptions(precision=6, suppress=True)

# ---------- B ----------
S = np.array([[4.0, 1, 1], [1, 4, 1], [1, 1, 4]])
w, Q = np.linalg.eigh(S)
print("B1 eigenvalues", w, "\n Q\n", Q, "\n  Q^T S Q\n", Q.T @ S @ Q)
P6 = np.ones((3, 3)) / 3; P3 = np.eye(3) - P6
print("   spectral decomposition 6 P1 + 3 P2 == S:", np.allclose(6 * P6 + 3 * P3, S), "\n   S^{1/2}\n", np.real(sqrtm(S)), "\n   via projectors sqrt6 P1 + sqrt3 P2\n", np.sqrt(6) * P6 + np.sqrt(3) * P3,
      "\n   S^{-1} = P1/6 + P2/3\n", P6 / 6 + P3 / 3, "\n   check", np.linalg.inv(S))
for name, M in (("3x^2+2xy+3y^2", np.array([[3.0, 1], [1, 3]])), ("x^2+4xy+y^2", np.array([[1.0, 2], [2, 1]]))):
    ev, V = np.linalg.eigh(M)
    print(f"B2 {name}: eigenvalues {ev}, axes {V.T}")
A2 = np.array([[4.0, 2, -2], [2, 5, 1], [-2, 1, 6]])
print("   leading minors", [round(np.linalg.det(A2[:k, :k]), 6) for k in (1, 2, 3)], " eig", np.linalg.eigvalsh(A2), " Cholesky\n", np.linalg.cholesky(A2))
A3 = np.array([[3.0, 0], [4, 5]])
U, s, Vt = np.linalg.svd(A3)
print("B3 A^T A", A3.T @ A3, " sigma", s, " sigma^2", s**2, " 3sqrt5, sqrt5 =", 3 * 5**0.5, 5**0.5, "\n U\n", U, "\n V\n", Vt.T, "\n cond", s[0] / s[1], " pinv = inv\n", np.linalg.pinv(A3))
a = np.array([[1.0, 2, 2]]); b = np.array([9.0])
print("B4 min-norm solution", np.linalg.pinv(a) @ b, " norm", np.linalg.norm(np.linalg.pinv(a) @ b), " other solution (9,0,0) norm 9")
X5 = np.array([[2.0, 1], [3, 4], [5, 5], [7, 8], [8, 7]])
mu = X5.mean(0); Xc = X5 - mu; C5 = Xc.T @ Xc / (len(X5) - 1)
ev, V = np.linalg.eigh(C5); ev, V = ev[::-1], V[:, ::-1]
print("B5 mean", mu, " cov\n", C5, "\n eig", ev, " explained", ev / ev.sum(), "\n PC1", V[:, 0], " PC2", V[:, 1], "\n scores PC1", Xc @ V[:, 0], "\n rank-1 reconstruction\n", mu + np.outer(Xc @ V[:, 0], V[:, 0]),
      "\n reconstruction SSE", np.sum((Xc - np.outer(Xc @ V[:, 0], V[:, 0]))**2), " = (n-1) lambda2", (len(X5) - 1) * ev[1])
Sig = np.array([[4.0, 2], [2, 3]]); Si = np.linalg.inv(Sig)
for p in (np.array([2.0, 2]), np.array([2.0, -2])):
    print(f"B6 point {p}: Euclidean {np.linalg.norm(p):.4f}  Mahalanobis {np.sqrt(p @ Si @ p):.4f}")
L = np.linalg.cholesky(Sig); Wsym = np.real(sqrtm(Si))
print("   Sigma^{-1}\n", Si, "\n   L\n", L, "\n   L^{-1}\n", np.linalg.inv(L), "\n   Sigma^{-1/2}\n", Wsym, "\n   whitened covariance (Cholesky)", np.linalg.inv(L) @ Sig @ np.linalg.inv(L).T, " (ZCA)", Wsym @ Sig @ Wsym)
xv = np.array([1.0, 0, 0]); xv2 = np.array([1.0, 1, 1])
print("B7 Rayleigh quotients", xv @ S @ xv / (xv @ xv), xv2 @ S @ xv2 / (xv2 @ xv2), (np.array([1.0, -1, 0]) @ S @ np.array([1.0, -1, 0])) / 2)

# ---------- C ----------
print("C1 randomized SVD")
rng = np.random.default_rng(0)
m, n = 2000, 500
Uq, _ = np.linalg.qr(rng.standard_normal((m, n))); Vq, _ = np.linalg.qr(rng.standard_normal((n, n)))
sv = 1.0 / (1 + np.arange(n))  # slowly decaying spectrum sigma_i = 1/i
A = (Uq * sv) @ Vq.T
k = 20
t0 = time.perf_counter(); Uf, sf, Vtf = np.linalg.svd(A, full_matrices=False); t1 = time.perf_counter()
best = sf[k]
def rsvd(A, k, p=10, q=0, rng=rng):
    Om = rng.standard_normal((A.shape[1], k + p)); Y = A @ Om
    for _ in range(q):
        Y, _ = np.linalg.qr(Y); Y = A @ (A.T @ Y)
    Qr, _ = np.linalg.qr(Y); B = Qr.T @ A
    Ub, sb, Vtb = np.linalg.svd(B, full_matrices=False)
    return (Qr @ Ub)[:, :k], sb[:k], Vtb[:k]
print(f"   full SVD {t1 - t0:.3f}s; optimal rank-{k} error sigma_{k+1} = {best:.5f}")
for p, q in ((0, 0), (10, 0), (10, 1), (10, 2)):
    t2 = time.perf_counter(); Ur, sr, Vtr = rsvd(A, k, p, q); t3 = time.perf_counter()
    err = np.linalg.norm(A - (Ur * sr) @ Vtr, 2)
    print(f"   oversample p={p}, power its q={q}: ||A - A_k||_2 = {err:.5f} (ratio to optimal {err / best:.3f}), time {t3 - t2:.4f}s, max rel err in top-{k} sigma {np.max(np.abs(sr - sf[:k]) / sf[:k]):.1e}")

print("C2 eigenvector sensitivity vs singular-value stability")
for gap in (1e-1, 1e-3, 1e-5):
    A0 = np.diag([1.0, 1.0 + gap, 3.0]); E = 1e-4 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    w0, V0 = np.linalg.eigh(A0); w1, V1 = np.linalg.eigh(A0 + E)
    angle = np.degrees(np.arccos(min(1.0, abs(V0[:, 0] @ V1[:, 0]))))
    print(f"   gap {gap:.0e}, ||E|| = 1e-4: eigenvalue change {np.abs(w1 - w0).max():.1e} (<= ||E||), eigenvector angle {angle:.3f} deg (Davis-Kahan ~ ||E||/gap = {1e-4 / gap:.1e} rad)")

print("C3 orthogonal Procrustes")
rng = np.random.default_rng(1)
P = rng.standard_normal((50, 3))
th = np.radians(40); Rtrue = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
Rtrue = Rtrue @ np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
Bp = P @ Rtrue + 0.05 * rng.standard_normal(P.shape)
U, s, Vt = np.linalg.svd(P.T @ Bp); Rhat = U @ Vt
print("   ||Rhat - Rtrue||_F", np.linalg.norm(Rhat - Rtrue), " det", np.linalg.det(Rhat), " scipy agrees", np.allclose(Rhat, orthogonal_procrustes(P, Bp)[0]),
      "\n   residual ||P R - B||_F: Procrustes", np.linalg.norm(P @ Rhat - Bp), " unconstrained LS", np.linalg.norm(P @ np.linalg.lstsq(P, Bp, rcond=None)[0] - Bp))

print("C4 total least squares vs OLS (errors in both variables)")
rng = np.random.default_rng(2)
n = 500; xt = rng.uniform(-3, 3, n); yt = 2 * xt + 1
xo = xt + 0.8 * rng.standard_normal(n); yo = yt + 0.8 * rng.standard_normal(n)
b_ols = np.polyfit(xo, yo, 1)
Z = np.column_stack([xo - xo.mean(), yo - yo.mean()]); _, _, Vt = np.linalg.svd(Z, full_matrices=False)
nvec = Vt[-1]; slope_tls = -nvec[0] / nvec[1]; icpt_tls = yo.mean() - slope_tls * xo.mean()
print(f"   true slope 2; OLS {b_ols[0]:.4f} (attenuation factor var(x)/(var(x)+0.64) = {3 / (3 + 0.64):.4f} -> {2 * 3 / 3.64:.4f}); TLS {slope_tls:.4f}, intercept {icpt_tls:.4f}")

# ---------- D ----------
print("D1 SVD image compression")
from sklearn.datasets import load_sample_image
img = load_sample_image("china.jpg").astype(float).mean(axis=2)
U, s, Vt = np.linalg.svd(img, full_matrices=False)
mimg, nimg = img.shape
for k in (5, 20, 50, 100):
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    mse = np.mean((img - Ak)**2)
    print(f"   rank {k}: storage {k * (mimg + nimg + 1) / (mimg * nimg):.3f} of original, rel Frobenius err {np.linalg.norm(img - Ak) / np.linalg.norm(img):.4f} (= sqrt(sum tail sigma^2)/||A|| {np.sqrt(np.sum(s[k:]**2) / np.sum(s**2)):.4f}), PSNR {10 * np.log10(255**2 / mse):.2f} dB")
print("   energy in top 1/10/50 singular values:", [round(float(np.sum(s[:k]**2) / np.sum(s**2)), 5) for k in (1, 10, 50)])

print("D2 latent semantic analysis")
terms = ["matrix", "eigenvalue", "vector", "algebra", "football", "goal", "match", "team", "score"]
docs = {"d1": "matrix eigenvalue vector", "d2": "matrix algebra vector vector", "d3": "eigenvalue algebra matrix", "d4": "vector algebra",
        "d5": "football goal match", "d6": "team match goal goal", "d7": "football team score", "d8": "score goal", "d9": "matrix score team"}
Aterm = np.array([[d.split().count(t) for d in docs.values()] for t in terms], float)
U, s, Vt = np.linalg.svd(Aterm, full_matrices=False)
print("   singular values", s.round(4))
k = 2; Dk = (np.diag(s[:k]) @ Vt[:k]).T
q = np.array([1.0 if t in ("algebra",) else 0.0 for t in terms])
qk = q @ U[:, :k]
cos = lambda u, v: u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300)
raw = [cos(q, Aterm[:, j]) for j in range(Aterm.shape[1])]; lat = [cos(qk, Dk[j]) for j in range(Dk.shape[0])]
print("   query 'algebra' raw cosine:   ", dict(zip(docs, np.round(raw, 3))))
print("   query 'algebra' latent cosine:", dict(zip(docs, np.round(lat, 3))))
print("   doc coordinates in 2-D latent space:", {d: tuple(np.round(Dk[j], 3)) for j, d in enumerate(docs)})

print("D3 PCA of simulated yield curves (level / slope / curvature)")
rng = np.random.default_rng(4)
mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0]); lam_ns = 0.6
f1 = np.ones_like(mats); f2 = (1 - np.exp(-lam_ns * mats)) / (lam_ns * mats); f3 = f2 - np.exp(-lam_ns * mats)
Tn = 1000
beta = np.cumsum(rng.standard_normal((Tn, 3)) * np.array([0.05, 0.08, 0.10]), axis=0) + np.array([4.0, -1.5, 1.0])
Y = beta @ np.vstack([f1, f2, f3]) + 0.02 * rng.standard_normal((Tn, len(mats)))
dY = np.diff(Y, axis=0); dYc = dY - dY.mean(0)
U, s, Vt = np.linalg.svd(dYc, full_matrices=False); var = s**2 / np.sum(s**2)
print("   explained variance of daily changes, first 5 PCs:", var[:5].round(4), " cumulative 3:", var[:3].sum().round(4))
for i in range(3):
    v = Vt[i] * np.sign(Vt[i][-1] if i == 0 else Vt[i][-1] - Vt[i][0])
    print(f"   PC{i+1} loadings:", v.round(3))

print("D4 anomaly detection by PCA reconstruction error (breast cancer: train on benign)")
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
bc = load_breast_cancer(); Xb = bc.data; yb = (bc.target == 0).astype(int)  # 1 = malignant (anomaly)
rng = np.random.default_rng(5)
ben = np.nonzero(yb == 0)[0]; rng.shuffle(ben); tr = ben[:200]; te = np.setdiff1d(np.arange(len(yb)), tr)
mu_, sd_ = Xb[tr].mean(0), Xb[tr].std(0); Ztr = (Xb[tr] - mu_) / sd_; Zte = (Xb[te] - mu_) / sd_
_, s_, Vt_ = np.linalg.svd(Ztr, full_matrices=False)
for k in (1, 3, 5, 10, 20):
    Vk = Vt_[:k].T; rec = Zte - Zte @ Vk @ Vk.T; score = np.sum(rec**2, axis=1)
    lam_k = s_[:k]**2 / (len(tr) - 1); t2 = np.sum((Zte @ Vk)**2 / lam_k, axis=1)
    print(f"   k={k}: explained {np.sum(s_[:k]**2) / np.sum(s_**2):.3f}  AUC (SPE / Q-statistic) {roc_auc_score(yb[te], score):.4f}  AUC (Hotelling T^2) {roc_auc_score(yb[te], t2):.4f}")
