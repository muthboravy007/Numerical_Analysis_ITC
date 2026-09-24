"""Numerical answers for exercises/ch07_exercises.md."""
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import linalg_iterative as it  # noqa: E402

np.set_printoptions(precision=5, suppress=True)

# ---------- B ----------
x = np.array([3.0, -4, 0, 12])
print("B1 x norms 1,2,inf:", np.linalg.norm(x, 1), np.linalg.norm(x), np.linalg.norm(x, np.inf))
A = np.array([[1.0, -2], [3, 4]])
print("B1 A norms 1,inf,2:", np.linalg.norm(A, 1), np.linalg.norm(A, np.inf), np.linalg.norm(A, 2),
      " eig(A^T A)", np.linalg.eigvalsh(A.T @ A), " rho(A)", it.spectral_radius(A), np.linalg.eigvals(A))
B = np.array([[2.0, 1, 0], [1, 2, 1], [0, 1, 2]])
print("B2 eig", np.linalg.eigvalsh(B), "2 +- sqrt2 =", 2 - 2**0.5, 2 + 2**0.5, " norm2", np.linalg.norm(B, 2),
      " norm inf", np.linalg.norm(B, np.inf))
print("   eigvec", np.linalg.eigh(B)[1].T)

A3 = np.array([[8.0, 1, -2], [1, 7, 2], [-1, 2, 9]]); xs = np.array([1.0, -1, 2]); b3 = A3 @ xs
print("B3 b =", b3)
xj = np.zeros(3); xg = np.zeros(3)
for k in range(1, 4):
    xj = (b3 - (A3 - np.diag(np.diag(A3))) @ xj) / np.diag(A3)
    for i in range(3):
        xg[i] = (b3[i] - A3[i, :i] @ xg[:i] - A3[i, i + 1:] @ xg[i + 1:]) / A3[i, i]
    print(f"   k={k} Jacobi {xj}  GS {xg}  err_inf J {np.abs(xj - xs).max():.2e} GS {np.abs(xg - xs).max():.2e}")
print("   iterations to 1e-8: Jacobi", len(it.jacobi(A3, b3, tol=1e-8)[1]) - 1, "GS", len(it.gauss_seidel(A3, b3, tol=1e-8)[1]) - 1)
Tj = it.iteration_matrix(A3, "jacobi"); Tg = it.iteration_matrix(A3, "gauss_seidel")
print("B4 Tj\n", Tj, "\n  ||Tj||inf", np.linalg.norm(Tj, np.inf), "rho", it.spectral_radius(Tj),
      "\n Tg\n", Tg, "\n  ||Tg||inf", np.linalg.norm(Tg, np.inf), "rho", it.spectral_radius(Tg))

A5 = np.array([[4.0, -1, 0], [-1, 4, -1], [0, -1, 4]]); b5 = A5 @ np.array([1.0, 2, 3])
print("B5 b", b5)
for w in (1.0, 1.1):
    xw = np.zeros(3)
    for k in range(1, 3):
        for i in range(3):
            s = A5[i, :i] @ xw[:i] + A5[i, i + 1:] @ xw[i + 1:]
            xw[i] = (1 - w) * xw[i] + w * (b5[i] - s) / A5[i, i]
        print(f"   omega={w} k={k}: {xw}")
rj = it.spectral_radius(it.iteration_matrix(A5, "jacobi")); wopt = it.optimal_sor_omega(rj)
print(f"   rho_J {rj:.6f} (sqrt2/4={2**0.5/4:.6f}) omega* {wopt:.6f} rho(T_omega*) {it.spectral_radius(it.iteration_matrix(A5, 'sor', wopt)):.6f} = omega*-1 {wopt-1:.6f}"
      f" rho_GS {it.spectral_radius(it.iteration_matrix(A5, 'gauss_seidel')):.6f}")
for w in (1.0, wopt, 1.1):
    print(f"   omega {w:.4f}: iterations to 1e-10 {len(it.sor(A5, b5, omega=w)[1]) - 1}")

A6 = np.array([[1.0, 2], [1.0001, 2]]); b6 = np.array([3.0, 3.0001]); b6p = np.array([3.0, 3.0003])
Ai = np.linalg.inv(A6)
x6 = np.linalg.solve(A6, b6); x6p = np.linalg.solve(A6, b6p)
k6 = np.linalg.norm(A6, np.inf) * np.linalg.norm(Ai, np.inf)
print("B6 A^-1\n", Ai, "\n kappa_inf", k6, "x", x6, "x perturbed", x6p,
      "\n rel change b", np.linalg.norm(b6p - b6, np.inf) / np.linalg.norm(b6, np.inf),
      "rel change x", np.linalg.norm(x6p - x6, np.inf) / np.linalg.norm(x6, np.inf),
      "bound", k6 * np.linalg.norm(b6p - b6, np.inf) / np.linalg.norm(b6, np.inf))
xt = np.array([1.0, 1.0]); xbad = np.array([3.0, 0.0])
print("   residual of (3,0):", b6 - A6 @ xbad, " error", xbad - xt)

A7 = np.array([[3.0, 1], [1, 2]]); b7 = np.array([5.0, 5])
x = np.zeros(2); r = b7 - A7 @ x; p = r.copy()
print("B7 CG, x* =", np.linalg.solve(A7, b7))
for k in range(1, 3):
    Ap = A7 @ p; a = (r @ r) / (p @ Ap); x = x + a * p; rn = r - a * Ap; beta = (rn @ rn) / (r @ r)
    print(f"   k={k}: alpha {a:.6f} x {x} r {rn} beta {beta:.6f} ")
    pn = rn + beta * p; print("        p_new", pn, " p_old^T A p_new =", p @ A7 @ pn, " r_new . r_old", rn @ r)
    r, p = rn, pn
x = np.zeros(2); r = b7.copy()
ev = np.linalg.eigvalsh(A7); kap = ev[-1] / ev[0]
print("B8 steepest descent; eig", ev, "kappa", kap, "rate (k-1)/(k+1)", (kap - 1) / (kap + 1))
xstar = np.linalg.solve(A7, b7)
def anorm(e):
    return np.sqrt(e @ A7 @ e)
e0 = anorm(x - xstar)
for k in range(1, 6):
    a = (r @ r) / (r @ A7 @ r); x = x + a * r; r = b7 - A7 @ x
    print(f"   k={k}: alpha {a:.5f} x {x}  ||e||_A ratio {anorm(x - xstar) / e0:.5f}")
    e0 = anorm(x - xstar)

# ---------- C ----------
m = 20; n = m * m
P = it.poisson_2d_sparse(m).toarray(); bp = np.ones(n)
rhoJ = np.cos(np.pi / (m + 1)); wth = 2 / (1 + np.sin(np.pi / (m + 1)))
print(f"C1 m={m}: rho_J theory {rhoJ:.6f} numeric {it.spectral_radius(it.iteration_matrix(P, 'jacobi')):.6f} omega* {wth:.4f}")
print("   Jacobi iters to 1e-6:", len(it.jacobi(P, bp, tol=1e-6)[1]) - 1)
for w in (1.0, 1.2, 1.4, 1.6, 1.7, 1.75, wth, 1.8, 1.9, 1.95):
    print(f"   SOR omega {w:.4f}: iters {len(it.sor(P, bp, omega=w, tol=1e-6)[1]) - 1}")

print("C2 2-D Poisson, iterations to 1e-8")
for m in (16, 32, 64, 128):
    A = it.poisson_2d_sparse(m); b = np.ones(m * m)
    kap = (np.sin(np.pi * m / (2 * (m + 1))) / np.sin(np.pi / (2 * (m + 1)))) ** 2
    nsd = len(it.steepest_descent(A, b, tol=1e-8, max_iter=200000)[1]) - 1 if m <= 32 else None
    ncg = len(it.conjugate_gradient(A, b, tol=1e-8)[1]) - 1
    # no pivoting and natural ordering keep the incomplete factorisation symmetric (CG needs an SPD preconditioner)
    ilu = spla.spilu(A.tocsc(), drop_tol=1e-2, fill_factor=10, permc_spec="NATURAL", diag_pivot_thresh=0.0)
    npc = len(it.conjugate_gradient(A, b, tol=1e-8, M_inv=ilu.solve)[1]) - 1
    print(f"   m={m} n={m*m} kappa {kap:.1f} sqrt {np.sqrt(kap):.1f}  SD {nsd}  CG {ncg}  ILU-PCG {npc} (fill {ilu.nnz / A.nnz:.2f}x)")

print("C3 mixed-precision iterative refinement")
import scipy.linalg as sla
rng = np.random.default_rng(0)
for n, target in ((200, 1e3), (200, 1e6)):
    U, _ = np.linalg.qr(rng.standard_normal((n, n))); V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(target), n); A = (U * s) @ V.T
    xt = rng.standard_normal(n); b = A @ xt
    lu32 = sla.lu_factor(A.astype(np.float32))
    x = sla.lu_solve(lu32, b.astype(np.float32)).astype(np.float64)
    errs = [np.linalg.norm(x - xt) / np.linalg.norm(xt)]
    for k in range(6):
        r = b - A @ x
        d = sla.lu_solve(lu32, r.astype(np.float32)).astype(np.float64)
        x = x + d
        errs.append(np.linalg.norm(x - xt) / np.linalg.norm(xt))
    x64 = np.linalg.solve(A, b)
    print(f"   kappa {target:.0e}: errors", " ".join(f"{e:.1e}" for e in errs), f" float64 direct {np.linalg.norm(x64 - xt) / np.linalg.norm(xt):.1e}")

# ---------- D ----------
print("D1 ridge regression by CG (matrix-free)")
rng = np.random.default_rng(1)
N, p = 5000, 200
Z = rng.standard_normal((N, p)); X = Z @ np.diag(np.logspace(0, -2, p)) @ np.linalg.qr(rng.standard_normal((p, p)))[0]
beta = rng.standard_normal(p); y = X @ beta + 0.1 * rng.standard_normal(N)
for lam in (1e-4, 1e-2, 1.0, 100.0):
    op = spla.LinearOperator((p, p), matvec=lambda v, lam=lam: X.T @ (X @ v) + lam * v)
    bdir = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
    count = [0]
    def cb(xk):
        count[0] += 1
    bcg, info = spla.cg(op, X.T @ y, rtol=1e-10, maxiter=5000, callback=cb)
    ev = np.linalg.eigvalsh(X.T @ X)
    print(f"   lambda {lam:g}: kappa {(ev[-1] + lam) / (ev[0] + lam):.2e}  CG iters {count[0]}  rel diff vs direct {np.linalg.norm(bcg - bdir) / np.linalg.norm(bdir):.1e}")

print("D2 PageRank as a linear system")
rng = np.random.default_rng(2)
n = 2000; alpha = 0.85
out = rng.integers(1, 15, n)
rows = np.repeat(np.arange(n), out); cols = rng.integers(0, n, out.sum())
# preferential target: 30% of the links go to the first 50 "hub" pages
hub = rng.random(out.sum()) < 0.3; cols[hub] = rng.integers(0, 50, hub.sum())
Adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)); Adj.data[:] = 1.0
deg = np.asarray(Adj.sum(1)).ravel()
Pt = (sp.diags(1 / deg) @ Adj).T.tocsr()  # column-stochastic
v = np.ones(n) / n
x = v.copy(); k = 0
while True:
    xn = alpha * (Pt @ x) + (1 - alpha) * v; k += 1
    if np.abs(xn - x).sum() < 1e-10:
        x = xn; break
    x = xn
M = (sp.identity(n) - alpha * Pt).tocsr(); rhs = (1 - alpha) * v
xd = spla.spsolve(M.tocsc(), rhs)
# Gauss-Seidel on M x = rhs
xg = v.copy(); Mu = sp.triu(M, 1).tocsr(); L = sp.tril(M).tocsr(); kg = 0
while True:
    xn = spla.spsolve_triangular(L, rhs - Mu @ xg, lower=True); kg += 1
    if np.abs(xn - xg).sum() < 1e-10:
        xg = xn; break
    xg = xn
print(f"   power/Jacobi iterations {k}, Gauss-Seidel {kg}, |power-direct|_1 {np.abs(x - xd).sum():.1e}, |GS-direct|_1 {np.abs(xg - xd).sum():.1e}, sum {xd.sum():.6f}")
top = np.argsort(-xd)[:5]
print("   top pages", top, np.round(xd[top], 5), " all hubs (<50)?", bool((top < 50).all()))

print("D3 semi-supervised label propagation (harmonic solution) by CG")
from sklearn.datasets import make_moons
Xm, ym = make_moons(600, noise=0.08, random_state=0)
from scipy.spatial.distance import cdist
Dm = cdist(Xm, Xm); sig = 0.1
W = np.exp(-Dm**2 / (2 * sig**2)); np.fill_diagonal(W, 0); W[W < 1e-6] = 0
Lg = np.diag(W.sum(1)) - W
rng = np.random.default_rng(3)
lab = np.concatenate([rng.choice(np.where(ym == c)[0], 3, replace=False) for c in (0, 1)])
unl = np.setdiff1d(np.arange(len(ym)), lab)
Luu = sp.csr_matrix(Lg[np.ix_(unl, unl)]); Lul = Lg[np.ix_(unl, lab)]
f_l = ym[lab].astype(float)
count = [0]
def cb(xk):
    count[0] += 1
f_u, info = spla.cg(Luu, -Lul @ f_l, rtol=1e-8, maxiter=10000, callback=cb, M=sp.diags(1 / Luu.diagonal()))
acc = ((f_u > 0.5) == ym[unl]).mean()
print(f"   {len(lab)} labels, {len(unl)} unlabelled, nnz W {np.count_nonzero(W)}, Jacobi-PCG iters {count[0]} info {info}, accuracy {acc:.4f}")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(Xm[lab], ym[lab])
print(f"   supervised logistic regression on the 6 labels only: accuracy {(lr.predict(Xm[unl]) == ym[unl]).mean():.4f}")
