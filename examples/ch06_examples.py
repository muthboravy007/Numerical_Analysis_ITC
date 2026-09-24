"""Chapter 6 worked examples — run:  python examples/ch06_examples.py"""
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import linalg_direct as ld  # noqa: E402

np.set_printoptions(precision=4, suppress=True)

print("Example 6.1 — Leontief input-output model")
A = np.array([[0.2, 0.3, 0.1], [0.1, 0.1, 0.3], [0.2, 0.2, 0.1]])  # inputs per unit output (agri, manuf, services)
d = np.array([50.0, 80.0, 120.0])
M = np.eye(3) - A
x = ld.gaussian_elimination(M, d)
Linv = np.linalg.inv(M)
print("  I - A =\n", M, "\n  output x =", x, "\n  Leontief inverse (I-A)^-1 =\n", Linv, "\n  column sums (output multipliers):", Linv.sum(0))
print("  extra output needed for +10 demand in services:", Linv @ np.array([0, 0, 10.0]))

print("Example 6.2 — factor once, solve many right-hand sides")
rng = np.random.default_rng(0)
n, k = 1000, 500
B = rng.standard_normal((n, n)) + n * np.eye(n) * 0.05; R = rng.standard_normal((n, k))
t0 = time.perf_counter(); lu = sla.lu_factor(B); X1 = sla.lu_solve(lu, R); t1 = time.perf_counter()
X2 = np.empty_like(R)
t2 = time.perf_counter()
for j in range(50):
    X2[:, j] = np.linalg.solve(B, R[:, j])
t3 = time.perf_counter()
t4 = time.perf_counter(); X3 = np.linalg.inv(B) @ R; t5 = time.perf_counter()
print(f"  LU once + {k} solves: {t1 - t0:.3f}s | re-solving from scratch: {(t3 - t2) / 50 * k:.3f}s (extrapolated from 50) | explicit inverse: {t5 - t4:.3f}s")
print(f"  residual LU {np.linalg.norm(B @ X1 - R) / np.linalg.norm(R):.1e}   residual inverse {np.linalg.norm(B @ X3 - R) / np.linalg.norm(R):.1e}")
print(f"  flop counts: LU 2n^3/3 = {2 * n ** 3 / 3:.2e}, each solve 2n^2 = {2 * n ** 2:.2e}")

print("Example 6.3 — is this 'correlation matrix' valid? (Cholesky as a PD test)")
C = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.3], [0.7, 0.3, 1.0]])
try:
    ld.cholesky(C); print("  Cholesky succeeded")
except Exception as e:
    print("  numlib Cholesky failed:", type(e).__name__)
try:
    np.linalg.cholesky(C)
except np.linalg.LinAlgError as e:
    print("  numpy Cholesky failed:", e)
w, V = np.linalg.eigh(C); print("  eigenvalues", w.round(5), " det", np.linalg.det(C).round(5))
w2 = np.maximum(w, 1e-6); C2 = V @ np.diag(w2) @ V.T; Dm = np.diag(1 / np.sqrt(np.diag(C2))); C2 = Dm @ C2 @ Dm
print("  repaired by eigenvalue clipping + rescaling:\n", C2.round(4), "\n  new eigenvalues", np.linalg.eigvalsh(C2).round(5), " Cholesky L\n", np.linalg.cholesky(C2).round(4))

print("Example 6.4 — Thomas algorithm scaling")
for n in (10 ** 3, 10 ** 5, 10 ** 6):
    a = -np.ones(n - 1); b = 2.0 * np.ones(n) + 0.01; c = -np.ones(n - 1); dd = np.ones(n)
    t0 = time.perf_counter(); xs = ld.tridiagonal_solve(a, b, c, dd); t1 = time.perf_counter()
    ab = np.vstack([np.r_[0, c], b, np.r_[a, 0]]); t2 = time.perf_counter(); xb = sla.solve_banded((1, 1), ab, dd); t3 = time.perf_counter()
    print(f"  n={n:>8d}: numlib Thomas {t1 - t0:.4f}s  LAPACK banded {t3 - t2:.4f}s  max diff {np.abs(xs - xb).max():.1e}  dense would need {8 * n * n / 1e9:.3g} GB")

print("Example 6.5 — multivariate normal log-likelihood via Cholesky")
p = 600
Qm, _ = np.linalg.qr(rng.standard_normal((p, p))); ev = np.linspace(0.01, 0.5, p); S = (Qm * ev) @ Qm.T
xv = rng.multivariate_normal(np.zeros(p), S)
with np.errstate(all="ignore"):
    det = np.linalg.det(S)
L = np.linalg.cholesky(S); logdet = 2 * np.sum(np.log(np.diag(L))); z = sla.solve_triangular(L, xv, lower=True)
ll = -0.5 * (p * np.log(2 * np.pi) + logdet + z @ z)
from scipy.stats import multivariate_normal
print(f"  det(S) = {det} (underflows); log det via Cholesky = {logdet:.4f} (slogdet {np.linalg.slogdet(S)[1]:.4f}); log-likelihood {ll:.4f} (scipy {multivariate_normal(np.zeros(p), S).logpdf(xv):.4f})")

print("Example 6.6 — fill-in and ordering (arrow matrix)")
n = 2000
Aarr = sp.lil_matrix((n, n)); Aarr.setdiag(4.0); Aarr[0, :] = 1.0; Aarr[:, 0] = 1.0; Aarr[0, 0] = n
Aarr = Aarr.tocsc()
for order, name in (("NATURAL", "arrow pointing up-left (dense row/col first)"), ("COLAMD", "COLAMD reordering")):
    luo = spla.splu(Aarr, permc_spec=order, diag_pivot_thresh=0)
    print(f"  {name}: nnz(A) {Aarr.nnz}, nnz(L+U) {luo.L.nnz + luo.U.nnz}")
perm = np.r_[np.arange(1, n), 0]; Arev = Aarr[perm][:, perm].tocsc()
luo = spla.splu(Arev, permc_spec="NATURAL", diag_pivot_thresh=0)
print(f"  manual reversal (dense row/col last): nnz(L+U) {luo.L.nnz + luo.U.nnz}")
