"""Reproduces every numerical example in Chapter 9 (Approximating Eigenvalues).
Run from the repository root:  python lectures/code/ch09.py"""

import os
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import eigen as eg  # noqa: E402
from numlib import roots  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("=== 9.1 Linear algebra and eigenvalues ===")
A = np.array([[5.0, 1, 0], [1, -2, 1], [0.5, 0.5, 8]])
print("Ex 9.1.1 discs", eg.gershgorin_discs(A), "eigs", np.linalg.eigvals(A))
print("Ex 9.1.2 column discs", list(zip(np.diag(A), np.abs(A).sum(0) - np.abs(np.diag(A)))))
B = np.array([[2.0, 1, 0], [0, 3, 1], [0, 0, 5]]); w, V = np.linalg.eig(B)
print("Ex 9.1.3 eig", w, "V\n", V, "rank V", np.linalg.matrix_rank(V))
vs = np.array([[1.0, 1, 0], [1, 0, 1], [0, 1, 1]])
Q = np.zeros((3, 3))
for k in range(3):
    v = vs[k] - sum((vs[k] @ Q[j]) * Q[j] for j in range(k)); Q[k] = v / np.linalg.norm(v)
print("Ex 9.1.4 orthonormal set\n", Q, "\n Q Q^T\n", Q @ Q.T)
Sg = np.array([[4.0, 0.8, -0.3], [0.8, 2.0, 0.5], [-0.3, 0.5, 1.5]])
print("Ex 9.1.5 covariance discs", eg.gershgorin_discs(Sg), "eigs", np.linalg.eigvalsh(Sg))

print("\n=== 9.2 Orthogonal matrices and similarity ===")
S = np.array([[2.0, 1, 0], [1, 2, 1], [0, 1, 2]]); w, Qs = np.linalg.eigh(S)
print("Ex 9.2.1 eigs", w, "Q\n", Qs, "\n Q^T S Q\n", Qs.T @ S @ Qs)
J = np.array([[2.0, 1], [0, 2]]); w, V = np.linalg.eig(J); print("Ex 9.2.2 defective eig", w, "V\n", V, "rank", np.linalg.matrix_rank(V, tol=1e-8))
M = np.array([[1.0, 2], [3, 0]]); P = np.array([[1.0, 1], [0, 1]])
print("Ex 9.2.3 eig M", np.linalg.eigvals(M), "eig P^-1 M P", np.linalg.eigvals(np.linalg.inv(P) @ M @ P), "P^-1MP\n", np.linalg.inv(P) @ M @ P)
N = np.array([[1.0, 2, 3], [0, 4, 5], [1, 0, 6]]); T, Z = sla.schur(N)
print("Ex 9.2.4 Schur T\n", T, "\n eig", np.linalg.eigvals(N))
th = np.pi / 6; R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
print("Ex 9.2.5 R^T R\n", R.T @ R, "eig R", np.linalg.eigvals(R), "|x| vs |Rx|", np.linalg.norm([3, 4]), np.linalg.norm(R @ [3, 4]))
K = np.array([[2.0, -1], [-1, 2]]); print("   quadratic form eigs", np.linalg.eigvalsh(K), "x^TKx for (1,1),(1,-1):", np.array([1, 1]) @ K @ [1, 1], np.array([1, -1]) @ K @ [1, -1])

print("\n=== 9.3 Power method ===")
A = np.array([[4.0, -1, 1], [-1, 3, -2], [1, -2, 3]]); print("eigs", np.linalg.eigvalsh(A))
x = np.array([1.0, 0, 0]); p = 0; mus = []
for k in range(1, 11):
    y = A @ x; mu = y[p]; p = np.argmax(np.abs(y)); x = y / y[p]; mus.append(mu)
    print(f"Ex 9.3.1 k={k}: x={x} mu={mu:.7f}")
acc = roots.aitken(mus); print("Ex 9.3.2 Aitken of mu:", acc[:6])
mu, v, hist = eg.symmetric_power_method(A, [1.0, 0, 0], tol=1e-12)
print("Ex 9.3.3 symmetric power Rayleigh:", np.array(hist[:8]), "errors", np.abs(np.array(hist[:8]) - 6))
print("   inf-norm errors", np.abs(np.array(mus[:8]) - 6))
for q in (0.8, 2.9):
    lam, vv, h = eg.inverse_iteration(A, shift=q, x0=[1.0, 1, 1])
    print(f"Ex 9.3.4 inverse power q={q}: {np.array(h[:6])} -> {lam}")
mu1, v1, _ = eg.power_method_inf(A, [1.0, 0, 0], max_iter=500, tol=1e-14)
Bd = eg.wielandt_deflation(A, mu1, v1); print("Ex 9.3.5 v1", v1, "deflated B\n", Bd, "\n eig B", np.linalg.eigvals(Bd))
B2 = Bd[1:, 1:] if np.argmax(np.abs(v1)) == 0 else None
mu2, _, _ = eg.power_method_inf(B2, [1.0, 0], max_iter=200, tol=1e-12); print("   power on 2x2 block ->", mu2)
for r in (0.5, 0.9, 0.99):
    print(f"Ex 9.3.6 ratio {r}: iterations for 1e-8 ~ {np.log(1e-8) / np.log(r):.0f}")

print("\n=== 9.4 Householder ===")
A4 = np.array([[4.0, 2, 2, 1], [2, -3, 1, 1], [2, 1, 3, 1], [1, 1, 1, 2]])
T, Q = eg.householder_tridiagonal(A4); print("Ex 9.4.1 T\n", T, "\n eig A", np.linalg.eigvalsh(A4), "eig T", np.linalg.eigvalsh(T))
x = np.array([3.0, 4, 0]); alpha = -np.sign(x[0]) * np.linalg.norm(x); v = x.copy(); v[0] -= alpha; v /= np.linalg.norm(v)
H = np.eye(3) - 2 * np.outer(v, v); print("Ex 9.4.2 v", v, "H\n", H, "\n Hx", H @ x)
N4 = np.array([[4.0, 1, -2, 2], [1, 2, 0, 1], [2, 1, 3, -2], [1, 3, -1, 5]]); Hs, Qh = sla.hessenberg(N4, calc_q=True)
print("Ex 9.4.3 Hessenberg\n", Hs, "\n eig", np.sort_complex(np.linalg.eigvals(N4)), np.sort_complex(np.linalg.eigvals(Hs)))
for n in (10, 100, 1000):
    print(f"Ex 9.4.4 n={n}: tridiagonalization ~(2/3)n^3={2 / 3 * n ** 3:.2e}, QR step on tridiagonal ~ {14 * n:.1e} vs dense {4 / 3 * n ** 3:.1e}")
print("Ex 9.4.5 orthogonality ||Q^TQ-I||", np.linalg.norm(Q.T @ Q - np.eye(4)), "||Q^T A Q - T||", np.linalg.norm(Q.T @ A4 @ Q - T))

print("\n=== 9.5 QR algorithm ===")
T3 = np.array([[3.0, 1, 0], [1, 3, 1], [0, 1, 3]])
Ak = T3.copy()
for k in range(1, 13):
    Qk, Rk = np.linalg.qr(Ak); Ak = Rk @ Qk
    if k in (1, 2, 3, 5, 8, 12):
        print(f"Ex 9.5.1 k={k}: diag {np.diag(Ak)} subdiag {np.diag(Ak, -1)}")
print("   eigs", np.linalg.eigvalsh(T3), "ratios", (3 - np.sqrt(2) * 0 - 0) and [3 / (3 + np.sqrt(2)), (3 - np.sqrt(2)) / 3])
Ak = T3.copy(); n = 3; hist = []
for k in range(1, 8):
    a, b, c = Ak[n - 2, n - 2], Ak[n - 1, n - 1], Ak[n - 1, n - 2]
    d = (a - b) / 2; mu = b - np.sign(d if d != 0 else 1) * c ** 2 / (abs(d) + np.sqrt(d * d + c * c))
    Qk, Rk = np.linalg.qr(Ak - mu * np.eye(n)); Ak = Rk @ Qk + mu * np.eye(n)
    hist.append(abs(Ak[n - 1, n - 2]))
print("Ex 9.5.2 Wilkinson-shift |a_32| per step:", hist)
Nq = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 2.0]]) + 0.1
Ak = Nq.copy()
for k in range(50):
    Qk, Rk = np.linalg.qr(Ak); Ak = Rk @ Qk
print("Ex 9.5.3 after 50 QR steps\n", Ak, "\n eig", np.linalg.eigvals(Nq), "eig of 2x2 block", np.linalg.eigvals(Ak[1:, 1:]) if abs(Ak[1, 0]) < 1e-6 else np.linalg.eigvals(Ak[:2, :2]))
Cm = np.array([[6.0, -11, 6], [1, 0, 0], [0, 1, 0]])
print("Ex 9.5.4 companion of x^3-6x^2+11x-6 eig", eg.qr_algorithm(Cm), np.roots([1, -6, 11, -6]))
Ak = np.array([[4.0, 1, 0, 0], [1, 3, 1, 0], [0, 1, 2, 1], [0, 0, 1, 1]])
steps = 0
while abs(Ak[3, 2]) > 1e-12 and steps < 200:
    mu = Ak[3, 3]; Qk, Rk = np.linalg.qr(Ak - mu * np.eye(4)); Ak = Rk @ Qk + mu * np.eye(4); steps += 1
print("Ex 9.5.5 deflation after", steps, "shifted steps: lambda4 =", Ak[3, 3], "remaining block eig", np.linalg.eigvalsh(Ak[:3, :3]), "all", np.linalg.eigvalsh(np.array([[4.0, 1, 0, 0], [1, 3, 1, 0], [0, 1, 2, 1], [0, 0, 1, 1]])))

print("\n=== 9.6 SVD ===")
A = np.array([[3.0, 2, 2], [2, 3, -2]]); U, s, Vt = np.linalg.svd(A)
print("Ex 9.6.1 A A^T", A @ A.T, "eig", np.linalg.eigvalsh(A @ A.T), "\n U\n", U, "\n s", s, "\n Vt\n", Vt)
M = np.array([[1.0, 1], [0, 1], [1, 0]]); Uv, sv, Vv = eg.svd_via_eig(M)
print("Ex 9.6.2 A^TA", M.T @ M, "eig", np.linalg.eigvalsh(M.T @ M), "s", sv, "np", np.linalg.svd(M, compute_uv=False))
X = np.array([[1.0, 2, 3], [1, 2, 3.0], [2, 4, 6.0], [1, 0, 1]]); y = np.array([1.0, 2, 3, 1])
print("Ex 9.6.3 rank", np.linalg.matrix_rank(X), "sv", np.linalg.svd(X, compute_uv=False), "min-norm solution", np.linalg.pinv(X) @ y, "lstsq", np.linalg.lstsq(X, y, rcond=None)[0])
Mm = np.array([[5.0, 5, 0, 1], [4, 5, 1, 0], [1, 0, 5, 4], [0, 1, 4, 5], [5, 4, 1, 1]])
U, s, Vt = np.linalg.svd(Mm, full_matrices=False); M1 = s[0] * np.outer(U[:, 0], Vt[0]); M2 = M1 + s[1] * np.outer(U[:, 1], Vt[1])
print("Ex 9.6.4 ratings sv", s, "\n rank-2 approx\n", M2, "\n errors", np.linalg.norm(Mm - M1, 2), s[1], np.linalg.norm(Mm - M2, "fro"), np.sqrt(np.sum(s[2:] ** 2)))
A = np.array([[2.0, 1], [1, 1.2]]); U, s, Vt = np.linalg.svd(A); print("Ex 9.6.5 s", s, "cond", s[0] / s[1], np.linalg.cond(A), "v1", Vt[0], "A v1", A @ Vt[0], "s1 u1", s[0] * U[:, 0])

print("\n=== 9.7 DS ===")
Pw = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
pi = np.array([1.0, 0, 0])
for k in range(1, 21):
    pi = pi @ Pw
    if k in (1, 2, 5, 10, 20):
        print(f"Ex 9.7.1 k={k}: {pi}")
w, V = np.linalg.eig(Pw.T); st = np.real(V[:, np.argmax(np.real(w))]); print("   stationary", st / st.sum(), "|lambda2|", sorted(np.abs(w))[-2])
adj = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 0]])
print("Ex 9.7.2 PageRank with dangling node 5:", eg.pagerank(adj))
W = np.zeros((8, 8))
for i, j in [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (4, 6), (5, 6), (6, 7), (5, 7)]:
    W[i, j] = W[j, i] = 1
L = np.diag(W.sum(1)) - W; w, V = np.linalg.eigh(L)
print("Ex 9.7.3 Laplacian eigs", w, "\n Fiedler", V[:, 1], "clusters", (V[:, 1] > 0).astype(int))
rng = np.random.default_rng(0)
X = rng.multivariate_normal([0, 0], [[3, 1.5], [1.5, 1]], 500); Cv = np.cov(X.T); w, V = np.linalg.eigh(Cv)
print("Ex 9.7.4 covariance", Cv, "eig", w, "PC1", V[:, -1], "explained", w[-1] / w.sum())
Adj = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
U, s, Vt = np.linalg.svd(Adj)
print("Ex 9.7.5 HITS hubs", np.abs(U[:, 0]) / np.abs(U[:, 0]).sum(), "authorities", np.abs(Vt[0]) / np.abs(Vt[0]).sum(), "s1", s[0])
