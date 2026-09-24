"""Chapter 7 worked examples — run:  python examples/ch07_examples.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import linalg_iterative as it  # noqa: E402

np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

print("Example 7.1 — a-posteriori error bound from the residual")
A = np.array([[4.0, 1, 0], [1, 3, 1], [0, 1, 2]]); x_true = np.array([1.0, -2, 3]); b = A @ x_true
x_approx = np.array([1.01, -2.02, 3.015])
r = b - A @ x_approx
kappa = np.linalg.cond(A, np.inf)
print(f"  residual {r}, ||r||/||b|| = {np.abs(r).max() / np.abs(b).max():.4e}, kappa_inf = {kappa:.4f}")
print(f"  bound on relative error: {kappa * np.abs(r).max() / np.abs(b).max():.4e}; actual {np.abs(x_approx - x_true).max() / np.abs(x_true).max():.4e}")

print("Example 7.2 — steady temperature in a plate: Jacobi vs Gauss-Seidel")
m = 15; P = it.poisson_2d_sparse(m).toarray(); bb = np.zeros(m * m); bb[-m:] = 100.0  # top edge at 100 degrees
for name, fn in (("Jacobi", it.jacobi), ("Gauss-Seidel", it.gauss_seidel)):
    x, res = fn(P, bb, tol=1e-6)
    rates = np.array(res[1:]) / np.array(res[:-1])
    print(f"  {name}: {len(res) - 1} iterations; asymptotic residual ratio {np.median(rates[-50:]):.5f}")
rj = np.cos(np.pi / (m + 1)); print(f"  theory: rho(T_J) = cos(pi/{m + 1}) = {rj:.5f}, rho(T_GS) = rho_J^2 = {rj ** 2:.5f};  centre temperature {x.reshape(m, m)[m // 2, m // 2]:.4f} (exact limit 25 by symmetry)")

print("Example 7.3 — choosing the SOR parameter")
n = 50; T1 = it.poisson_1d(n); bt = np.ones(n)
rho = np.cos(np.pi / (n + 1)); wopt = it.optimal_sor_omega(rho)
print(f"  rho_J {rho:.6f}, omega* = {wopt:.4f}, rho(T_omega*) = omega*-1 = {wopt - 1:.4f}")
for w in (1.0, 1.5, 1.8, 1.85, wopt, 1.95, 1.99):
    _, res = it.sor(T1, bt, omega=w, tol=1e-8, max_iter=100000)
    print(f"  omega {w:.4f}: {len(res) - 1} iterations; spectral radius {it.spectral_radius(it.iteration_matrix(T1, 'sor', w)):.5f}")

print("Example 7.4 — CG for a Gaussian-process kernel system")
xg = np.sort(rng.uniform(0, 10, 300)); K = np.exp(-(xg[:, None] - xg[None, :]) ** 2 / (2 * 1.0 ** 2))
yg = np.sin(xg) + 0.1 * rng.standard_normal(len(xg))
for jit in (1e-1, 1e-2, 1e-4, 1e-6):
    Kj = K + jit * np.eye(len(xg))
    _, res = it.conjugate_gradient(Kj, yg, tol=1e-8, max_iter=5000)
    print(f"  noise/jitter {jit:.0e}: cond {np.linalg.cond(Kj):.2e}, CG iterations {len(res) - 1}, sqrt(cond) {np.sqrt(np.linalg.cond(Kj)):.0f}")

print("Example 7.5 — Jacobi (diagonal) preconditioning of a badly scaled SPD matrix")
n = 200; Qm, _ = np.linalg.qr(rng.standard_normal((n, n))); Base = (Qm * np.linspace(1, 10, n)) @ Qm.T
Dsc = np.diag(np.logspace(0, 2, n)); As = Dsc @ Base @ Dsc; bs = rng.standard_normal(n)
_, r0 = it.conjugate_gradient(As, bs, tol=1e-10, max_iter=20000)
_, r1 = it.conjugate_gradient(As, bs, tol=1e-10, max_iter=20000, M_inv=lambda r: r / np.diag(As))
Dh = np.diag(1 / np.sqrt(np.diag(As)))
print(f"  cond(A) {np.linalg.cond(As):.2e}, cond(D^-1/2 A D^-1/2) {np.linalg.cond(Dh @ As @ Dh):.2e}; CG iterations {len(r0) - 1}, Jacobi-PCG {len(r1) - 1}")

print("Example 7.6 — CG as regularisation: semi-convergence in deblurring")
n = 200; s = np.linspace(0, 1, n)
Kb = np.exp(-(s[:, None] - s[None, :]) ** 2 / (2 * 0.03 ** 2)); Kb /= Kb.sum(1, keepdims=True)
xtrue = ((s > 0.2) & (s < 0.4)).astype(float) + np.exp(-((s - 0.7) / 0.05) ** 2)
yb = Kb @ xtrue + 0.01 * rng.standard_normal(n)
N = Kb.T @ Kb; rhs = Kb.T @ yb
x = np.zeros(n); r = rhs - N @ x; p = r.copy(); errs = []
for k in range(1, 201):
    Ap = N @ p; a = (r @ r) / (p @ Ap); x = x + a * p; rn = r - a * Ap; p = rn + (rn @ rn) / (r @ r) * p; r = rn
    errs.append(np.linalg.norm(x - xtrue) / np.linalg.norm(xtrue))
kbest = int(np.argmin(errs)) + 1
print(f"  cond(K) {np.linalg.cond(Kb):.2e}; best iteration {kbest} (error {errs[kbest - 1]:.3f})")
for k in (1, 5, 10, 20, 50, 100, 200):
    print(f"    after {k:3d} CG iterations: relative error {errs[k - 1]:.3f}")
xn = np.linalg.solve(N + 1e-12 * np.eye(n), rhs)
print(f"  'exact' solve of the normal equations: relative error {np.linalg.norm(xn - xtrue) / np.linalg.norm(xtrue):.2e}")
