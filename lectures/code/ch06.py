"""Reproduces every numerical example in Chapter 6 (Direct Methods for Linear Systems).
Run from the repository root:  python lectures/code/ch06.py"""

import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import linalg_direct as ld  # noqa: E402
from numlib.floating import fl  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)


def ge_digits(A, b, k, pivot="none"):
    """Gaussian elimination in k-digit rounding arithmetic (every operation rounded)."""
    A = [[fl(v, k) for v in row] for row in A]
    b = [fl(v, k) for v in b]
    n = len(b)
    s = [max(abs(v) for v in row) for row in A]
    for i in range(n - 1):
        if pivot == "partial":
            p = max(range(i, n), key=lambda r: abs(A[r][i]))
        elif pivot == "scaled":
            p = max(range(i, n), key=lambda r: abs(A[r][i]) / s[r])
        else:
            p = i
        A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]; s[i], s[p] = s[p], s[i]
        for j in range(i + 1, n):
            m = fl(A[j][i] / A[i][i], k)
            for c in range(i, n):
                A[j][c] = fl(A[j][c] - fl(m * A[i][c], k), k)
            b[j] = fl(b[j] - fl(m * b[i], k), k)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = b[i]
        for c in range(i + 1, n):
            acc = fl(acc - fl(A[i][c] * x[c], k), k)
        x[i] = fl(acc / A[i][i], k)
    return x


print("=== 6.1 Gaussian elimination ===")
A = np.array([[2.0, 1, 1], [4, -6, 0], [-2, 7, 2]]); b = np.array([5.0, -2, 9])
M = np.column_stack([A, b])
M[1] -= 2 * M[0]; M[2] += M[0]; print("Ex 6.1.1 after step 1:\n", M)
M[2] += M[1]; print("   after step 2:\n", M, "\n   x =", ld.back_substitution(M[:, :3], M[:, 3]))
A4 = np.array([[4.0, -1, 1, 2], [2, 5, -1, 1], [1, 1, 6, -2], [-1, 2, 1, 7]]); x4 = np.array([1.0, -1, 2, 1])
b4 = A4 @ x4
print("Ex 6.1.2 b =", b4, "solution", ld.gaussian_elimination(A4, b4, pivot=False))
M = np.column_stack([A4, b4]).astype(float)
for k in range(3):
    for i in range(k + 1, 4):
        m = M[i, k] / M[k, k]; M[i] -= m * M[k]
    print(f"   after step {k + 1}:\n{M}")
Z4 = np.array([[1.0, 2, -1, 1], [2, 1, 1, -1], [1, -1, 2, 1], [3, 1, -1, 2]])
Mz = np.column_stack([Z4, Z4 @ x4])
for i in range(1, 4):
    Mz[i] -= Mz[i, 0] / Mz[0, 0] * Mz[0]
print("Ex 6.1.5 after step 1 (pivot a22 and a32 equal):\n", Mz)
Mz[2] -= Mz[2, 1] / Mz[1, 1] * Mz[1]; Mz[3] -= Mz[3, 1] / Mz[1, 1] * Mz[1]
print("   after step 2 (zero pivot in row 3):\n", Mz, "\n   pivoted solution", ld.gaussian_elimination(Z4, Z4 @ x4))
S = np.array([[1.0, 1, 1], [2, 2, 1], [3, 3, 2]])
print("Ex 6.1.3 rank", np.linalg.matrix_rank(S), "det", np.linalg.det(S))
for n in (3, 10, 100, 1000):
    md = n ** 3 / 3 + n ** 2 - n / 3; asub = n ** 3 / 3 + n ** 2 / 2 - 5 * n / 6
    print(f"Ex 6.1.4 n={n}: mult/div {md:.4g}  add/sub {asub:.4g}")

print("\n=== 6.2 Pivoting ===")
A = [[0.0004, 1.402], [0.4003, -1.502]]; b = [1.406, 2.501]
print("Ex 6.2.1 exact x = (10, 1); 4-digit no pivot:", ge_digits(A, b, 4), "partial:", ge_digits(A, b, 4, "partial"))
m = fl(0.4003 / 0.0004, 4); print("   multiplier", m, "a22", fl(-1.502 - fl(m * 1.402, 4), 4), "b2", fl(2.501 - fl(m * 1.406, 4), 4))
A2 = [[4.0, 14020.0], [0.4003, -1.502]]; b2 = [14060.0, 2.501]
print("Ex 6.2.2 scaled row: partial:", ge_digits(A2, b2, 4, "partial"), "scaled:", ge_digits(A2, b2, 4, "scaled"),
      "ratios", 4 / 14020, 0.4003 / 1.502)
A3 = [[30.0, 59140, 1200], [5.291, -6.13, 1.5], [2.2, 3.1, -4.4]]; b3 = [121910.0, -2.469, -4.8]
xt = np.linalg.solve(np.array(A3), np.array(b3))
print("Ex 6.2.3 3x3 exact", xt, "\n   4-digit none", ge_digits(A3, b3, 4), "\n   partial", ge_digits(A3, b3, 4, "partial"), "\n   scaled", ge_digits(A3, b3, 4, "scaled"))
s = np.max(np.abs(np.array(A3)), axis=1); print("   scale factors", s, "ratios col1", np.abs(np.array(A3)[:, 0]) / s)
C = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 10]]); bc = np.array([1.0, 1, 1])
print("Ex 6.2.4 complete pivot first pivot = max |a_ij| =", np.max(np.abs(C)), "at", np.unravel_index(np.argmax(np.abs(C)), C.shape), "solution", np.linalg.solve(C, bc))
for n in (5, 10, 20, 40):
    W = np.eye(n) - np.tril(np.ones((n, n)), -1); W[:, -1] = 1
    P, L, U = ld.lu_partial_pivot(W)
    print(f"Ex 6.2.5 Wilkinson n={n}: growth max|U|/max|A| = {np.max(np.abs(U)) / np.max(np.abs(W)):.3g}")
    if n == 40:
        xw = np.ones(n); bw = W @ xw
        print("   n=40 solve error:", np.max(np.abs(ld.lu_solve(P, L, U, bw) - xw)))
rng = np.random.default_rng(0)
g = []
for _ in range(200):
    R = rng.standard_normal((50, 50)); P, L, U = ld.lu_partial_pivot(R); g.append(np.max(np.abs(U)) / np.max(np.abs(R)))
print("   random 50x50 growth: median", np.median(g), "max", np.max(g))

print("\n=== 6.3 Linear algebra and inversion ===")
A = np.array([[1.0, 2], [3, 4]]); B = np.array([[0.0, 1], [1, 0]])
print("Ex 6.3.1 AB\n", A @ B, "\n BA\n", B @ A)
G = np.array([[2.0, 1, 0], [1, 2, 1], [0, 1, 2]])
Mg = np.column_stack([G, np.eye(3)])
for k in range(3):
    Mg[k] /= Mg[k, k]
    for i in range(3):
        if i != k:
            Mg[i] -= Mg[i, k] * Mg[k]
print("Ex 6.3.2 Gauss-Jordan inverse:\n", Mg[:, 3:], "\n  times 4:\n", 4 * Mg[:, 3:])
n = 500
R = rng.standard_normal((n, n)); bb = rng.standard_normal(n)
t0 = time.perf_counter(); x1 = np.linalg.solve(R, bb); t1 = time.perf_counter(); x2 = np.linalg.inv(R) @ bb; t2 = time.perf_counter()
print(f"Ex 6.3.3 n=500 solve {t1 - t0:.4f}s inv {t2 - t1:.4f}s; residuals {np.linalg.norm(R @ x1 - bb):.2e} {np.linalg.norm(R @ x2 - bb):.2e}")
H = ld.hilbert(10); xe = np.ones(10); bh = H @ xe
print("   Hilbert(10): solve err", np.max(np.abs(np.linalg.solve(H, bh) - xe)), "inv err", np.max(np.abs(np.linalg.inv(H) @ bh - xe)),
      "residuals", np.linalg.norm(H @ np.linalg.solve(H, bh) - bh), np.linalg.norm(H @ (np.linalg.inv(H) @ bh) - bh))
A = np.array([[1.0, 2], [3, 4]]); B = np.array([[2.0, 0], [1, 1]])
print("Ex 6.3.4 (AB)^-1 == B^-1 A^-1:", np.allclose(np.linalg.inv(A @ B), np.linalg.inv(B) @ np.linalg.inv(A)), np.linalg.inv(A @ B))
Bm = np.column_stack([[5.0, -2, 9], [1, 0, 0], [0, 1, 0]])
A = np.array([[2.0, 1, 1], [4, -6, 0], [-2, 7, 2]])
print("Ex 6.3.5 AX=B:\n", np.linalg.solve(A, Bm))

print("\n=== 6.4 Determinants ===")
print("Ex 6.4.1 det A (pivots 2,-8,1):", 2 * -8 * 1, np.linalg.det(A))
print("Ex 6.4.2 det A4:", np.linalg.det(A4))
P, L, U = ld.lu_partial_pivot(A4); print("   LU pivots", np.diag(U), "perm sign", round(np.linalg.det(P)))
print("Ex 6.4.3 swap rows ->", np.linalg.det(A[[1, 0, 2]]), "; scale row by 3 ->", np.linalg.det(np.diag([3, 1, 1]) @ A))
for n in (5, 10, 20, 25):
    print(f"Ex 6.4.4 n={n}: cofactor ~ n! = {math.factorial(n):.3e}  elimination ~ n^3/3 = {n ** 3 / 3:.3e}")
X = rng.standard_normal((400, 200)); Sig = X.T @ X / 400 * 0.5
print("Ex 6.4.5 det of 200x200 covariance:", np.linalg.det(Sig), "| slogdet:", np.linalg.slogdet(Sig), "| 2 sum log diag chol:", 2 * np.sum(np.log(np.diag(np.linalg.cholesky(Sig)))))

print("\n=== 6.5 Factorizations ===")
L, U = ld.lu(A); print("Ex 6.5.1 L\n", L, "\n U\n", U)
b0 = np.array([5.0, -2, 9]); y = ld.forward_substitution(L, b0); print("   y", y, "x", ld.back_substitution(U, y))
P, L, U = ld.lu_partial_pivot(A); print("Ex 6.5.2 P\n", P, "\n L\n", L, "\n U\n", U)
L4, U4 = ld.lu(A4); print("Ex 6.5.3 L4\n", L4, "\n U4\n", U4)
for rhs in (b4, np.array([1.0, 0, 0, 0])):
    print("   solve:", ld.back_substitution(U4, ld.forward_substitution(L4, rhs)))
try:
    ld.lu(np.array([[0.0, 1], [1, 1]]))
except ZeroDivisionError as e:
    print("Ex 6.5.4 [[0,1],[1,1]]:", e)
P, L, U = ld.lu_partial_pivot(np.array([[0.0, 1], [1, 1]])); print("   with pivoting P\n", P, "L\n", L, "U\n", U)
n, k = 1000, 50
print("Ex 6.5.5 cost: factor once + 50 solves", 2 / 3 * n ** 3 + k * 2 * n ** 2, "vs 50 eliminations", k * 2 / 3 * n ** 3)

print("\n=== 6.6 Special matrices ===")
D1 = np.array([[7.0, 2, 0], [3, 5, -1], [0, 5, -6]]); D2 = np.array([[6.0, 4, -3], [4, -2, 0], [-3, 0, 1]])
print("Ex 6.6.1 SDD?", ld.is_strictly_diagonally_dominant(D1), ld.is_strictly_diagonally_dominant(D2), "| D2^T SDD?", ld.is_strictly_diagonally_dominant(D2.T))
S = np.array([[4.0, 12, -16], [12, 37, -43], [-16, -43, 98]])
print("Ex 6.6.2 leading minors", [np.linalg.det(S[:k, :k]) for k in (1, 2, 3)], "\n   Cholesky L\n", ld.cholesky(S))
Lc = ld.cholesky(S); print("   solve Sx=(1,2,3):", ld.cholesky_solve(Lc, np.array([1.0, 2, 3])), "y =", ld.forward_substitution(Lc, np.array([1.0, 2, 3])))
Ld, d = ld.ldlt(S); print("Ex 6.6.3 LDL^T: L\n", Ld, "\n d", d)
a = -np.ones(3); c = -np.ones(3); dd = 2 * np.ones(4); rhs = np.array([1.0, 0, 0, 1])
print("Ex 6.6.4 Thomas:", ld.tridiagonal_solve(a, dd, c, rhs))
l = np.zeros(4); u = np.zeros(3); z = np.zeros(4)
l[0] = 2; u[0] = -1 / 2; z[0] = 1 / 2
for i in range(1, 4):
    l[i] = 2 - (-1) * u[i - 1]
    if i < 3:
        u[i] = -1 / l[i]
    z[i] = (rhs[i] - (-1) * z[i - 1]) / l[i]
print("   Crout l", l, "u", u, "z", z)
try:
    ld.cholesky(np.array([[1.0, 2], [2, 1]]))
except np.linalg.LinAlgError as e:
    print("Ex 6.6.5 [[1,2],[2,1]] Cholesky:", e, "eigs", np.linalg.eigvalsh(np.array([[1.0, 2], [2, 1]])))
for n in (1000, 10 ** 5, 10 ** 6):
    print(f"Ex 6.6.6 n={n}: dense LU {2 / 3 * n ** 3:.2e} flops, tridiagonal {8 * n:.2e}")

print("\n=== 6.7 DS applications ===")
Sigma = np.array([[4.0, 2.4], [2.4, 9.0]]); mu = np.array([1.0, -2.0])
Ls = np.linalg.cholesky(Sigma); print("Ex 6.7.1 L =", Ls)
Z = np.random.default_rng(1).standard_normal((100000, 2)); Xs = mu + Z @ Ls.T
print("   sample mean", Xs.mean(0), "sample cov\n", np.cov(Xs.T))
xq = np.array([3.0, 1.0]); zq = ld.forward_substitution(Ls, xq - mu)
print("Ex 6.7.2 Mahalanobis^2", zq @ zq, "check", (xq - mu) @ np.linalg.solve(Sigma, xq - mu), "logdet", 2 * np.sum(np.log(np.diag(Ls))), np.log(np.linalg.det(Sigma)),
      "loglik", -0.5 * (zq @ zq) - np.sum(np.log(np.diag(Ls))) - math.log(2 * math.pi))
from scipy.stats import multivariate_normal
print("   scipy logpdf", multivariate_normal(mu, Sigma).logpdf(xq))
Xt = np.array([0.0, 1, 2, 3]); yt = np.sin(Xt); ell, sn = 1.0, 0.1
K = np.exp(-0.5 * (Xt[:, None] - Xt[None, :]) ** 2 / ell ** 2)
Lk = np.linalg.cholesky(K + sn ** 2 * np.eye(4)); alpha = ld.cholesky_solve(Lk, yt)
xs = 1.5; ks = np.exp(-0.5 * (xs - Xt) ** 2 / ell ** 2)
v = ld.forward_substitution(Lk, ks)
print("Ex 6.7.3 GP: alpha", alpha, "mean at 1.5", ks @ alpha, "sin(1.5)", math.sin(1.5), "sd", math.sqrt(1 - v @ v))
Qm = np.array([[0.0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]]); Rm = np.array([[0.5, 0], [0, 0], [0, 0.5]])
N = np.linalg.inv(np.eye(3) - Qm); print("Ex 6.7.4 fundamental matrix\n", N, "\n absorption B = NR\n", N @ Rm, "\n expected steps", N.sum(1))
xd = np.array([1.0, 2, 3, 4, 5]); yd = np.array([2.2, 2.8, 3.6, 4.5, 5.1])
Xd = np.column_stack([np.ones(5), xd]); G = Xd.T @ Xd; r = Xd.T @ yd
Lg = ld.cholesky(G); print("Ex 6.7.5 X^TX", G, "X^Ty", r, "L", Lg, "beta", ld.cholesky_solve(Lg, r))
