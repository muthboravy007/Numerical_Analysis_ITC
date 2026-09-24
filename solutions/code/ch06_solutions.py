"""Numerical answers for exercises/ch06_exercises.md."""
import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import linalg_direct as ld  # noqa: E402
from numlib.floating import fl  # noqa: E402

src = open(os.path.join(os.path.dirname(__file__), "..", "..", "lectures", "code", "ch06.py")).read()
exec(src[src.index("def ge_digits"):src.index('print("=== 6.1')])

A = np.array([[1.0, 1, 1], [2, 3, 1], [3, 1, 4.0]]); b = np.array([6.0, 11, 17])
M = np.column_stack([A, b])
M[1] -= 2 * M[0]; M[2] -= 3 * M[0]; print("B1 step1\n", M); M[2] -= (M[2, 1] / M[1, 1]) * M[1]; print(" step2\n", M, "\n x", ld.back_substitution(M[:, :3], M[:, 3]))
A2 = [[0.003, 59.14], [5.291, -6.130]]
A2 = [[0.0012, 3.47], [2.14, -1.78]]; x2 = np.array([2.0, 1.0]); b2 = list(np.array(A2) @ x2)
print("B2 b", b2, "4-digit none", ge_digits(A2, b2, 4), "partial", ge_digits(A2, b2, 4, "partial"), "exact", x2)
A3 = [[2.0, 1000, 500], [1, 2, 3], [4, 1, -1.0]]; x3 = np.array([1.0, -1, 2]); b3 = list(np.array(A3) @ x3)
print("B3 b", b3, "scale", np.max(np.abs(np.array(A3)), 1), "ratios", np.abs(np.array(A3)[:, 0]) / np.max(np.abs(np.array(A3)), 1),
      "\n  3-digit none", ge_digits(A3, b3, 3), "partial", ge_digits(A3, b3, 3, "partial"), "scaled", ge_digits(A3, b3, 3, "scaled"))
L, U = ld.lu(A); print("B4 L\n", L, "\n U\n", U, "\n det", np.prod(np.diag(U)), np.linalg.det(A))
for rhs in ([6.0, 11, 17], [1.0, 0, 0]):
    print("   solve", rhs, "->", ld.back_substitution(U, ld.forward_substitution(L, np.array(rhs))))
P, L2, U2 = ld.lu_partial_pivot(A); print("B5 P\n", P, "\n L\n", L2, "\n U\n", U2)
S = np.array([[9.0, 3, -3], [3, 5, 1], [-3, 1, 11]]); Lc = ld.cholesky(S); Ll, d = ld.ldlt(S)
print("B6 Cholesky\n", Lc, "\n LDLT L\n", Ll, "d", d, "\n solve S x=(3,9,9):", ld.cholesky_solve(Lc, np.array([3.0, 9, 9])))
for k in (1.0, 2.0, 2.5, 3.0):
    T = np.array([[2.0, -1, 0], [-1, 2, -1], [0, -1, k]])
    print(f"B7 k={k}: minors {[round(np.linalg.det(T[:j, :j]), 4) for j in (1, 2, 3)]} eig {np.round(np.linalg.eigvalsh(T), 4)}")
print("   det T = 3k - 2 at k = 2/3, 0.5, 0.7:", [round(np.linalg.det(np.array([[2.0, -1, 0], [-1, 2, -1], [0, -1, kk]])), 4) for kk in (2 / 3, 0.5, 0.7)])
a = np.array([1.0, 1, 1]); dd = np.array([4.0, 4, 4, 4]); c = np.array([1.0, 1, 1]); r = np.array([5.0, 6, 6, 5])
print("B8 Thomas:", ld.tridiagonal_solve(a, dd, c, r))
Ai = np.array([[2.0, 1, 0], [1, 3, 1], [0, 1, 2]]); print("B9 inverse\n", np.linalg.inv(Ai) * 8, "/8")
for n in (100, 200, 400, 800):
    R = np.random.default_rng(n).standard_normal((n, n)); bb = np.ones(n)
    t0 = time.perf_counter(); ld.gaussian_elimination(R, bb); t1 = time.perf_counter(); np.linalg.solve(R, bb); t2 = time.perf_counter()
    print(f"C1 n={n}: numlib GE {t1 - t0:.3f}s  LAPACK {t2 - t1:.4f}s")
for n in (4, 6, 8, 10, 12, 14):
    H = ld.hilbert(n); xt = np.ones(n); bh = H @ xt; xs = np.linalg.solve(H, bh)
    print(f"C2 n={n}: cond {np.linalg.cond(H):.2e} rel err {np.linalg.norm(xs - xt) / np.linalg.norm(xt):.2e} residual {np.linalg.norm(H @ xs - bh) / np.linalg.norm(bh):.2e} log10 cond*eps {math.log10(np.linalg.cond(H) * 2.2e-16):.1f}")
rng = np.random.default_rng(0)
g = []
for n in (10, 50, 100, 200):
    gs = []
    for _ in range(50):
        R = rng.standard_normal((n, n)); P, L, U = ld.lu_partial_pivot(R); gs.append(np.max(np.abs(U)) / np.max(np.abs(R)))
    print(f"C3 n={n}: growth mean {np.mean(gs):.2f} max {np.max(gs):.2f}")
Sig = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]]); Lr = np.linalg.cholesky(Sig)
Z = rng.standard_normal((200000, 3)) @ Lr.T; print("D1 sample corr\n", np.corrcoef(Z.T))
ret = Z @ np.array([0.5, 0.3, 0.2]) * 0.02; print("   portfolio daily vol", ret.std(), "theory", 0.02 * math.sqrt(np.array([0.5, 0.3, 0.2]) @ Sig @ np.array([0.5, 0.3, 0.2])), "VaR99", -np.quantile(ret, 0.01))
Xg = np.linspace(0, 5, 6); yg = np.cos(Xg); K = np.exp(-0.5 * (Xg[:, None] - Xg[None]) ** 2) + 1e-4 * np.eye(6)
Lk = np.linalg.cholesky(K); alpha = ld.cholesky_solve(Lk, yg)
for xs_ in (2.5, 4.2, 7.0):
    ks = np.exp(-0.5 * (xs_ - Xg) ** 2); v = ld.forward_substitution(Lk, ks)
    print(f"D2 GP x*={xs_}: mean {ks @ alpha:.4f} (cos {math.cos(xs_):.4f}) sd {math.sqrt(max(1 - v @ v, 0)):.4f}")
print("   log marginal likelihood", -0.5 * yg @ alpha - np.sum(np.log(np.diag(Lk))) - 3 * math.log(2 * math.pi))
Q = np.array([[0.6, 0.3, 0.0], [0.2, 0.5, 0.2], [0.0, 0.3, 0.5]]); Rm = np.array([[0.1, 0.0], [0.05, 0.05], [0.0, 0.2]])
N = np.linalg.inv(np.eye(3) - Q); print("D3 N\n", N, "\n B=NR\n", N @ Rm, "\n expected steps", N.sum(1), "rows of B sum", (N @ Rm).sum(1))
