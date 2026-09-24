"""Chapter 14 worked examples — run:  python examples/ch14_examples.py"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

print("Example 14.1 — spectral decomposition of a stock correlation matrix")
R = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.7], [0.5, 0.7, 1.0]])
w, Q = np.linalg.eigh(R); w, Q = w[::-1], Q[:, ::-1]; Q[:, 0] *= np.sign(Q[0, 0])
print("  eigenvalues", w, " share of variance", (w / w.sum()).round(4))
print("  eigenvectors (columns)\n", Q)
print("  reconstruction sum lambda_i q_i q_i^T == R:", np.allclose(sum(w[i] * np.outer(Q[:, i], Q[:, i]) for i in range(3)), R))
wp = np.ones(3) / 3; print(f"  equal-weight portfolio variance (unit vols) {wp @ R @ wp:.4f}; min over unit-norm portfolios = lambda_min = {w[-1]:.4f} along {Q[:, -1].round(3)}")

print("Example 14.2 — 95% confidence ellipse of a bivariate normal")
S = np.array([[4.0, 1.5], [1.5, 1.0]]); c = stats.chi2.ppf(0.95, 2)
w2, V2 = np.linalg.eigh(S); V2 *= np.sign(V2[0])
print(f"  chi2_2(0.95) = {c:.4f}; eigenvalues {w2}; semi-axes sqrt(c lambda) = {np.sqrt(c * w2).round(4)}; major axis direction {V2[:, 1].round(4)} (angle {np.degrees(np.arctan2(V2[1, 1], V2[0, 1])):.2f} deg)")
Z = rng.multivariate_normal([0, 0], S, 100000); d2 = np.einsum("ij,jk,ik->i", Z, np.linalg.inv(S), Z)
print(f"  simulated coverage of the ellipse: {np.mean(d2 <= c):.4f}; area pi*sqrt(det)*c = {np.pi * np.sqrt(np.linalg.det(S)) * c:.4f}")

print("Example 14.3 — SVD of a tiny user-movie matrix")
M = np.array([[5.0, 5, 0], [4, 5, 1], [1, 0, 5], [0, 1, 4]])  # rows users, cols: action1, action2, romance
U, s, Vt = np.linalg.svd(M, full_matrices=False)
sgn = np.sign(Vt[:, 0]); Vt *= sgn[:, None]; U *= sgn[None, :]
print("  singular values", s, " energy share", (s ** 2 / np.sum(s ** 2)).round(4))
print("  V^T (movie factors)\n", Vt, "\n  U (user factors)\n", U)
M2 = (U[:, :2] * s[:2]) @ Vt[:2]
print("  rank-2 approximation\n", M2, "\n  error ||M - M2||_F =", np.linalg.norm(M - M2).round(4), "= sigma_3 =", s[2].round(4))

print("Example 14.4 — condition number and sensitivity of least squares")
t = np.linspace(0, 1, 20); X = np.column_stack([np.ones(20), t, t ** 2, t ** 3, t ** 4, t ** 5])
sX = np.linalg.svd(X, compute_uv=False); beta = np.ones(6); y = X @ beta
dy = 1e-6 * rng.standard_normal(20)
b1 = np.linalg.lstsq(X, y + dy, rcond=None)[0]
print(f"  singular values {sX.round(6)}; cond {sX[0] / sX[-1]:.3e}")
print(f"  relative perturbation of y {np.linalg.norm(dy) / np.linalg.norm(y):.1e} -> relative change of beta {np.linalg.norm(b1 - beta) / np.linalg.norm(beta):.1e} (amplification {np.linalg.norm(b1 - beta) / np.linalg.norm(beta) / (np.linalg.norm(dy) / np.linalg.norm(y)):.2e}); fitted values change {np.linalg.norm(X @ b1 - y) / np.linalg.norm(y):.1e}")

print("Example 14.5 — PCA depends on units: standardise!")
n = 300
height = rng.normal(170, 10, n); weight = 0.9 * (height - 170) + rng.normal(70, 8, n); income = rng.normal(40000, 15000, n) + 100 * (height - 170)
D = np.column_stack([height, weight, income])
for name, Dm in (("raw units", D - D.mean(0)), ("standardised", (D - D.mean(0)) / D.std(0))):
    Uu, ss, VV = np.linalg.svd(Dm, full_matrices=False)
    print(f"  {name}: variance share {(ss ** 2 / np.sum(ss ** 2)).round(4)}; PC1 loadings {(VV[0] * np.sign(VV[0, np.argmax(abs(VV[0]))])).round(4)}")

print("Example 14.6 — more features than samples: minimum-norm solutions and ridge")
n, p = 20, 100
Xw = rng.standard_normal((n, p)); btrue = np.zeros(p); btrue[:5] = [3, -2, 2, 1, -1]
yw = Xw @ btrue + 0.1 * rng.standard_normal(n)
bmn = np.linalg.pinv(Xw) @ yw
print(f"  rank {np.linalg.matrix_rank(Xw)}; min-norm solution: training residual {np.linalg.norm(Xw @ bmn - yw):.1e}, ||b|| {np.linalg.norm(bmn):.3f} vs ||b_true|| {np.linalg.norm(btrue):.3f}")
Uw, sw, Vw = np.linalg.svd(Xw, full_matrices=False)
for lam in (1e-1, 1e-3, 1e-6):
    br = Vw.T @ ((sw / (sw ** 2 + lam)) * (Uw.T @ yw))
    print(f"  ridge lambda {lam:.0e}: ||b_ridge - b_minnorm|| {np.linalg.norm(br - bmn):.2e}")
Xt = rng.standard_normal((1000, p)); yt = Xt @ btrue
print(f"  test RMSE: min-norm {np.sqrt(np.mean((Xt @ bmn - yt) ** 2)):.3f}  predict zero {np.sqrt(np.mean(yt ** 2)):.3f}; correlation of estimated and true coefficients {np.corrcoef(bmn, btrue)[0, 1]:.3f}")
