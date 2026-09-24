"""Reproduces every numerical example in Chapter 7 (Iterative Techniques in Matrix Algebra).
Run from the repository root:  python lectures/code/ch07.py"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import linalg_direct as ld, linalg_iterative as li  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("=== 7.1 Norms ===")
x = np.array([-1.0, 1, -2])
print("Ex 7.1.1 norms", np.linalg.norm(x, 1), np.linalg.norm(x), np.linalg.norm(x, np.inf))
for k in (1, 10, 100, 1000):
    v = np.array([1, 2 + 1 / k, 3 / k ** 2, np.exp(-k) * np.sin(k)])
    print(f"Ex 7.1.2 k={k}: dist_inf to (1,2,0,0) = {np.max(np.abs(v - [1, 2, 0, 0])):.3e}")
A = np.array([[1.0, 2, -1], [0, 3, -1], [5, -1, 1]])
print("Ex 7.1.3 ||A||_inf", np.linalg.norm(A, np.inf), "||A||_1", np.linalg.norm(A, 1), "||A||_2", np.linalg.norm(A, 2), "||A||_F", np.linalg.norm(A, "fro"))
rng = np.random.default_rng(0)
V = rng.standard_normal((5, 4))
for v in V[:3]:
    print(f"Ex 7.1.4 inf {np.linalg.norm(v, np.inf):.4f} <= 2 {np.linalg.norm(v):.4f} <= sqrt(n) inf {2 * np.linalg.norm(v, np.inf):.4f}; |x.y| {abs(v @ V[4]):.4f} <= {np.linalg.norm(v) * np.linalg.norm(V[4]):.4f}")
xt = np.array([1.0, 1, 1]); xa = np.array([1.2001, 0.99991, 0.92538])
print("Ex 7.1.5 errors", np.linalg.norm(xa - xt, np.inf), np.linalg.norm(xa - xt))
B = np.array([[3.0, -1], [2, 4]])
print("Ex 7.1.6 induced vs Frobenius:", np.linalg.norm(B, 2), np.linalg.norm(B, "fro"))

print("\n=== 7.2 Eigenvalues ===")
A = np.array([[4.0, 1], [2, 3]]); w, Vv = np.linalg.eig(A)
print("Ex 7.2.1 eig", w, "\n", Vv / Vv[0])
B = np.array([[1.0, 2], [0, 2]])
print("Ex 7.2.2 A^T A eig", np.linalg.eigvalsh(B.T @ B), "||B||_2", np.sqrt(np.max(np.linalg.eigvalsh(B.T @ B))), np.linalg.norm(B, 2), "rho(B)", max(abs(np.linalg.eigvals(B))))
C = np.array([[0.5, 1.0], [0.0, 0.5]])
for k in (1, 2, 5, 10, 20, 40):
    print(f"Ex 7.2.3 C^{k} =", np.linalg.matrix_power(C, k).ravel())
print("   ||C||_inf", np.linalg.norm(C, np.inf), "rho", 0.5)
M = np.array([[1.0, 2, 0], [0, -1, 1], [1, 0, 2]])
print("Ex 7.2.4 rho", max(abs(np.linalg.eigvals(M))), "norms", np.linalg.norm(M, 1), np.linalg.norm(M, 2), np.linalg.norm(M, np.inf))
Acons = np.array([[0.2, 0.3, 0.1], [0.4, 0.1, 0.3], [0.1, 0.2, 0.2]]); d = np.array([100.0, 200, 150])
print("Ex 7.2.5 Leontief rho", max(abs(np.linalg.eigvals(Acons))), "x =", np.linalg.solve(np.eye(3) - Acons, d))
S = np.zeros(3); P = np.eye(3)
for k in range(60):
    S = S + P @ d; P = P @ Acons
    if k in (0, 1, 4, 9, 19, 59):
        print(f"   Neumann partial sum k={k}: {S}")

print("\n=== 7.3 Jacobi & Gauss-Seidel ===")
A = np.array([[5.0, -1, 1, 0], [-1, 6, -2, 1], [1, -2, 7, -1], [0, 1, -1, 4]]); xs = np.array([1.0, 2, -1, 1]); b = A @ xs
print("Ex 7.3.1 b", b)
xj = np.zeros(4); D = np.diag(A); R = A - np.diag(D)
for k in range(1, 4):
    xj = (b - R @ xj) / D; print(f"   Jacobi k={k}: {xj}")
xg = np.zeros(4)
for k in range(1, 4):
    for i in range(4):
        xg[i] = (b[i] - A[i, :i] @ xg[:i] - A[i, i + 1:] @ xg[i + 1:]) / A[i, i]
    print(f"Ex 7.3.2 GS k={k}: {xg}")
nJ = len(li.jacobi(A, b, tol=1e-8)[1]) - 1; nG = len(li.gauss_seidel(A, b, tol=1e-8)[1]) - 1
Tj = li.iteration_matrix(A); Tg = li.iteration_matrix(A, "gauss_seidel")
print("   iterations to 1e-8: Jacobi", nJ, "GS", nG)
print("Ex 7.3.3 rho(Tj)", li.spectral_radius(Tj), "rho(Tg)", li.spectral_radius(Tg), "||Tj||_inf", np.linalg.norm(Tj, np.inf))
E = np.array([[1.0, 2, -2], [1, 1, 1], [2, 2, 1]])
print("Ex 7.3.4 rho J", li.spectral_radius(li.iteration_matrix(E)), "rho GS", li.spectral_radius(li.iteration_matrix(E, "gauss_seidel")))
be = E @ np.array([1.0, 1, 1])
print("   Jacobi iterations:", len(li.jacobi(E, be, tol=1e-10, max_iter=50)[1]) - 1, li.jacobi(E, be, tol=1e-10, max_iter=50)[0],
      "GS after 20:", li.gauss_seidel(E, be, max_iter=20)[0])
E2 = np.array([[2.0, -1, 1], [2, 2, 2], [-1, -1, 2]])
print("   second matrix rho J", li.spectral_radius(li.iteration_matrix(E2)), "rho GS", li.spectral_radius(li.iteration_matrix(E2, "gauss_seidel")))
x0 = np.zeros(4); x1 = (b - R @ x0) / D
nt = np.linalg.norm(Tj, np.inf)
kreq = np.log(1e-6 * (1 - nt) / np.linalg.norm(x1 - x0, np.inf)) / np.log(nt)
print("Ex 7.3.5 ||Tj||inf", nt, "||x1-x0||", np.linalg.norm(x1 - x0, np.inf), "k needed for 1e-6:", kreq)
for n in (10, 20, 40, 80):
    Ap = li.poisson_1d(n); bp = np.ones(n)
    print(f"Ex 7.3.6 Poisson n={n}: rho_J={np.cos(np.pi / (n + 1)):.6f} Jacobi its {len(li.jacobi(Ap, bp, tol=1e-6, max_iter=100000)[1]) - 1}  GS its {len(li.gauss_seidel(Ap, bp, tol=1e-6, max_iter=100000)[1]) - 1}")

print("\n=== 7.4 SOR ===")
A3 = np.array([[4.0, -1, 0], [-1, 4, -1], [0, -1, 4]]); b3 = np.array([2.0, 4, 10])
print("Ex 7.4.1 exact", np.linalg.solve(A3, b3))
for w in (1.0, 1.1):
    x = np.zeros(3)
    for k in range(1, 4):
        for i in range(3):
            sig = A3[i, :i] @ x[:i] + A3[i, i + 1:] @ x[i + 1:]
            x[i] = (1 - w) * x[i] + w * (b3[i] - sig) / A3[i, i]
        print(f"   omega={w} k={k}: {x}")
    print(f"   omega={w}: iterations to 1e-10: {len(li.sor(A3, b3, omega=w, tol=1e-10)[1]) - 1}")
rJ = li.spectral_radius(li.iteration_matrix(A3)); wopt = li.optimal_sor_omega(rJ)
print("Ex 7.4.2 rho_J", rJ, "omega*", wopt, "rho SOR", li.spectral_radius(li.iteration_matrix(A3, "sor", wopt)), "omega*-1", wopt - 1,
      "rho GS", li.spectral_radius(li.iteration_matrix(A3, "gauss_seidel")))
n = 50; Ap = li.poisson_1d(n); bp = np.ones(n)
wopt50 = li.optimal_sor_omega(np.cos(np.pi / (n + 1)))
for w in (1.0, 1.5, 1.8, 1.9, round(wopt50, 4), 1.95, 1.99):
    its = len(li.sor(Ap, bp, omega=w, tol=1e-8, max_iter=100000)[1]) - 1
    print(f"Ex 7.4.3 n=50 omega={w}: rho={li.spectral_radius(li.iteration_matrix(Ap, 'sor', w)):.5f} iterations {its}")
for w in (2.0, 2.1):
    print(f"Ex 7.4.4 omega={w}: rho={li.spectral_radius(li.iteration_matrix(A3, 'sor', w)):.4f}")
m = 20
A2 = li.poisson_2d_sparse(m).toarray(); b2 = np.ones(m * m)
w2 = 2 / (1 + np.sin(np.pi / (m + 1)))
print("Ex 7.4.5 2D Poisson m=20: omega*", w2, "GS its", len(li.gauss_seidel(A2, b2, tol=1e-6, max_iter=100000)[1]) - 1,
      "SOR its", len(li.sor(A2, b2, omega=w2, tol=1e-6, max_iter=100000)[1]) - 1)

print("\n=== 7.5 Error bounds and iterative refinement ===")
A = np.array([[1.0, 1], [1, 1.0001]]); b = np.array([2.0, 2.0001]); xt = np.array([2.0, 0.0])
r = b - A @ xt
print("Ex 7.5.1 residual", r, "||r||inf", np.linalg.norm(r, np.inf), "error", np.linalg.norm(xt - [1, 1], np.inf), "kappa", np.linalg.cond(A, np.inf),
      "bound", np.linalg.cond(A, np.inf) * np.linalg.norm(r, np.inf) / np.linalg.norm(b, np.inf) * 1)
A = np.array([[1.0, 2, -1], [0, 3, -1], [5, -1, 1]])
print("Ex 7.5.2 A^-1\n", np.linalg.inv(A), "\n ||A||inf", np.linalg.norm(A, np.inf), "||A^-1||inf", np.linalg.norm(np.linalg.inv(A), np.inf), "kappa", np.linalg.cond(A, np.inf))
A = np.array([[1.0, 1], [1, 1.0001]]); dA = np.array([[0, 0], [0, 0.0001]])
print("Ex 7.5.3 perturbed A: x =", np.linalg.solve(A + dA, [2, 2.0001]), "rel change A", np.linalg.norm(dA, np.inf) / np.linalg.norm(A, np.inf))
H6 = ld.hilbert(6); xe = np.ones(6); bh = H6 @ xe
x1, hist = ld.iterative_refinement(H6, bh, iters=4)
print("Ex 7.5.4/5 Hilbert(6) cond", np.linalg.cond(H6, np.inf))
for k, xk in enumerate(hist):
    print(f"   refinement step {k}: max error {np.max(np.abs(xk - xe)):.3e}")
print("   kappa estimate ||y||/||x~|| 10^t with t=7:", np.linalg.norm(hist[1] - hist[0], np.inf) / np.linalg.norm(hist[0], np.inf) * 1e7)
for n in (3, 5, 8, 10, 12):
    H = ld.hilbert(n)
    print(f"Ex 7.5.6 Hilbert n={n}: cond_inf {np.linalg.cond(H, np.inf):.3e}  error solving Hx=H1: {np.max(np.abs(np.linalg.solve(H, H @ np.ones(n)) - 1)):.2e}")

print("\n=== 7.6 Conjugate gradient ===")
A = np.array([[4.0, 1], [1, 3]]); b = np.array([1.0, 2]); x = np.zeros(2); r = b - A @ x; v = r.copy()
for k in range(2):
    Av = A @ v; t = (r @ r) / (v @ Av); x = x + t * v; rn = r - t * Av; s = (rn @ rn) / (r @ r)
    print(f"Ex 7.6.1 k={k + 1}: t={t:.6f} x={x} r={rn} s={s:.6f}"); v = rn + s * v; r = rn
print("   exact", np.linalg.solve(A, b))
A = np.array([[4.0, -1, 1], [-1, 4, -2], [1, -2, 4]]); b = A @ np.array([1.0, 2, 3])
x, res = li.conjugate_gradient(A, b, tol=1e-14)
print("Ex 7.6.2 b", b, "3x3 CG iterations", len(res) - 1, "x", x, "residual history", res)
for n in (50, 100, 200, 400):
    Ap = li.poisson_1d(n); bp = np.ones(n)
    print(f"Ex 7.6.3 n={n}: kappa {np.linalg.cond(Ap):.3e}  CG its {len(li.conjugate_gradient(Ap, bp, tol=1e-8)[1]) - 1}  SD its {len(li.steepest_descent(Ap, bp, tol=1e-8, max_iter=10**6)[1]) - 1}")
rng = np.random.default_rng(3)
n = 200
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
A0 = Q @ np.diag(np.linspace(1, 10, n)) @ Q.T
Dsc = np.diag(10.0 ** rng.uniform(-3, 3, n))
As = Dsc @ A0 @ Dsc; bs = rng.standard_normal(n)
_, r0 = li.conjugate_gradient(As, bs, tol=1e-8, max_iter=20000)
_, r1 = li.conjugate_gradient(As, bs, tol=1e-8, M_inv=lambda r: r / np.diag(As))
print(f"Ex 7.6.4 badly scaled SPD: kappa {np.linalg.cond(As):.2e}, CG its {len(r0) - 1}, Jacobi-PCG its {len(r1) - 1}, kappa(D^-1/2 A D^-1/2) {np.linalg.cond(np.diag(np.diag(As) ** -0.5) @ As @ np.diag(np.diag(As) ** -0.5)):.2f}")
for m in (3, 5, 10):
    ev = np.repeat(np.arange(1, m + 1, dtype=float), n // m + 1)[:n]
    Am = Q @ np.diag(ev) @ Q.T
    print(f"Ex 7.6.5 {m} distinct eigenvalues: CG its {len(li.conjugate_gradient(Am, bs, tol=1e-10)[1]) - 1}")

print("\n=== 7.7 DS ===")
Aс = sp.csr_matrix(np.array([[4, 0, 1, 0], [0, 3, 0, 0], [1, 0, 2, 5], [0, 0, 5, 1]]))
print("Ex 7.7.1 CSR data", Aс.data, "indices", Aс.indices, "indptr", Aс.indptr)
adj = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
out = adj.sum(1); Pm = (adj / out[:, None]).T; dmp = 0.85; nn = 4
r = np.linalg.solve(np.eye(nn) - dmp * Pm, (1 - dmp) / nn * np.ones(nn)); print("Ex 7.7.2 PageRank solve:", r, "sum", r.sum())
rj = np.full(nn, 1 / nn)
for k in range(1, 61):
    rj = dmp * Pm @ rj + (1 - dmp) / nn
    if k in (1, 5, 10, 20, 60):
        print(f"   Jacobi/power k={k}: {rj}  err {np.max(np.abs(rj - r)):.2e}")
W = np.zeros((6, 6))
for i, j in [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)]:
    W[i, j] = W[j, i] = 1
Lg = np.diag(W.sum(1)) - W
yl = np.array([1.0, 0, 0, 0, 0, -1]); lam = 1.0; Lm = np.diag([lam, 0, 0, 0, 0, lam])
f, res = li.conjugate_gradient(Lg + Lm + 1e-9 * np.eye(6), Lm @ yl, tol=1e-12)
print("Ex 7.7.3 label propagation scores", f, "CG its", len(res) - 1)
rng = np.random.default_rng(4)
Xs = sp.random(20000, 500, density=0.01, random_state=5, format="csr"); ys = rng.standard_normal(20000); lam = 1.0
op = spla.LinearOperator((500, 500), matvec=lambda v: Xs.T @ (Xs @ v) + lam * v)
it = [0]
beta, info = spla.cg(op, Xs.T @ ys, rtol=1e-10, callback=lambda xk: it.__setitem__(0, it[0] + 1))
ref = np.linalg.solve((Xs.T @ Xs).toarray() + lam * np.eye(500), Xs.T @ ys)
print("Ex 7.7.4 sparse ridge 20000x500 nnz", Xs.nnz, "CG its", it[0], "max diff vs dense", np.max(np.abs(beta - ref)))
Xr = rng.standard_normal((1000, 3)) * np.array([1.0, 100.0, 0.01]); yr = rng.standard_normal(1000)
for name, X in (("raw", Xr), ("standardized", (Xr - Xr.mean(0)) / Xr.std(0))):
    G = X.T @ X
    print(f"Ex 7.7.5 {name}: cond(X^TX) {np.linalg.cond(G):.2e} CG its {len(li.conjugate_gradient(G, X.T @ yr, tol=1e-10, max_iter=1000)[1]) - 1}  SD its {len(li.steepest_descent(G, X.T @ yr, tol=1e-6, max_iter=200000)[1]) - 1}")
