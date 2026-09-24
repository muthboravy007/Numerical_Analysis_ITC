"""Chapter 15 worked examples — run:  python examples/ch15_examples.py"""
import os
import sys

import numpy as np
from scipy.optimize import nnls, brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import optimization as op  # noqa: E402

np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

print("Example 15.1 — Armijo backtracking by hand")
f = lambda v: v[0] ** 2 + 10 * v[1] ** 2; g = lambda v: np.array([2 * v[0], 20 * v[1]])
x = np.array([1.0, 1.0]); gx = g(x); p = -gx; a = 1.0; c1 = 1e-4
print(f"  f(x0) {f(x):.4f}, grad {gx}, slope g^T p = {gx @ p:.1f}")
while f(x + a * p) > f(x) + c1 * a * (gx @ p):
    print(f"    alpha {a:.5f}: f = {f(x + a * p):.4f} > {f(x) + c1 * a * (gx @ p):.4f} -> halve"); a /= 2
print(f"    alpha {a:.5f}: f = {f(x + a * p):.4f} accepted; exact line-search step would be {(gx @ gx) / (gx @ np.diag([2.0, 20]) @ gx):.5f}")

print("Example 15.2 — Newton vs gradient descent for a one-parameter logistic MLE")
xs = np.array([-2.0, -1, -0.5, 0.3, 0.8, 1.5, 2.0, 2.5]); ys = np.array([0, 0, 1, 0, 1, 1, 0, 1.0])
nll = lambda b: np.sum(np.logaddexp(0, b * xs) - ys * b * xs)
dn = lambda b: np.sum((1 / (1 + np.exp(-b * xs)) - ys) * xs)
d2 = lambda b: np.sum(xs ** 2 * np.exp(-b * xs) / (1 + np.exp(-b * xs)) ** 2)
b = 0.0; hist = [b]
for _ in range(6):
    b -= dn(b) / d2(b); hist.append(b)
bstar = hist[-1]
print("  Newton iterates", np.round(hist, 8), " errors", [f"{abs(h - bstar):.1e}" for h in hist[:-1]])
L = np.sum(xs ** 2) / 4; bg = 0.0; k = 0
while abs(bg - bstar) > 1e-8:
    bg -= dn(bg) / L; k += 1
print(f"  gradient descent with step 1/L (L = sum x^2/4 = {L:.3f}) needs {k} iterations; SE from Fisher information 1/sqrt(I) = {1 / np.sqrt(d2(bstar)):.4f}")

print("Example 15.3 — momentum on an ill-conditioned quadratic")
A = np.diag([1.0, 100.0]); gq = lambda v: A @ v
for name, fn in (("GD step 1/L", lambda: op.gradient_descent(None, gq, [1.0, 1.0], lr=0.01, tol=1e-8, max_iter=100000)),
                 ("GD step 2/(mu+L)", lambda: op.gradient_descent(None, gq, [1.0, 1.0], lr=2 / 101, tol=1e-8, max_iter=100000)),
                 ("heavy ball (optimal)", lambda: op.momentum_gd(gq, [1.0, 1.0], lr=4 / (1 + 10) ** 2, beta=((10 - 1) / (10 + 1)) ** 2, tol=1e-8)),
                 ("Nesterov (step 1/L, beta 0.8)", lambda: op.nesterov(gq, [1.0, 1.0], lr=0.01, beta=0.8, tol=1e-8))):
    x, h = fn(); print(f"  {name:30s}: {len(h) - 1:5d} iterations")
print(f"  theory (gradient norm must fall from ~100 to 1e-8, a factor 1e-10): GD rate {(100 - 1) / 101:.4f} -> ~{np.log(1e-10) / np.log(99 / 101):.0f} its; heavy ball rate {9 / 11:.4f} -> ~{np.log(1e-10) / np.log(9 / 11):.0f} its")

print("Example 15.4 — L-BFGS memory on the extended Rosenbrock function (n = 100)")
def fr(x):
    return np.sum(100 * (x[1::2] - x[::2] ** 2) ** 2 + (1 - x[::2]) ** 2)
def gr(x):
    g = np.zeros_like(x); t = x[1::2] - x[::2] ** 2
    g[::2] = -400 * x[::2] * t - 2 * (1 - x[::2]); g[1::2] = 200 * t; return g
x0 = np.tile([-1.2, 1.0], 50)
for m in (1, 3, 5, 10, 20):
    x, h = op.lbfgs(fr, gr, x0, m=m, tol=1e-6, max_iter=5000)
    print(f"  m = {m:2d}: {len(h) - 1:4d} iterations, final f {fr(x):.1e}, memory {2 * m * 100} numbers")
x, h = op.bfgs(fr, gr, x0, tol=1e-6); print(f"  full BFGS (H0 = I): {len(h) - 1} iterations, memory {100 * 100} numbers")
from scipy.optimize import minimize
print(f"  scipy BFGS (H0 = I): {minimize(fr, x0, jac=gr, method='BFGS', options={'gtol': 1e-6}).nit} iterations; scipy L-BFGS-B: {minimize(fr, x0, jac=gr, method='L-BFGS-B', options={'gtol': 1e-6}).nit}")

print("Example 15.5 — SGD for least squares: step size, batch size and the noise floor")
n, p = 2000, 10; X = rng.standard_normal((n, p)); wt = rng.standard_normal(p); y = X @ wt + 0.5 * rng.standard_normal(n)
wls = np.linalg.lstsq(X, y, rcond=None)[0]
gi = lambda w, idx: X[idx].T @ (X[idx] @ w - y[idx]) / len(idx)
for name, kw in (("const lr 0.05, b=1", dict(lr=0.05, batch_size=1)), ("const lr 0.01, b=1", dict(lr=0.01, batch_size=1)),
                 ("const lr 0.05, b=32", dict(lr=0.05, batch_size=32)), ("decaying 0.05/(1+t), b=1", dict(lr=0.05, batch_size=1, decay=1.0))):
    w, h = op.sgd(gi, np.zeros(p), n, epochs=30, rng=np.random.default_rng(1), **kw)
    print(f"  {name:26s}: ||w - w_LS|| after 5/30 epochs: {np.linalg.norm(h[5] - wls):.4f} / {np.linalg.norm(h[30] - wls):.4f}")

print("Example 15.6 — maximum entropy distribution on a die with a mean constraint")
k = np.arange(1, 7.0); target = 4.5
Z = lambda lam: np.sum(np.exp(lam * k)); mean = lambda lam: np.sum(k * np.exp(lam * k)) / Z(lam)
lam = 0.0; its = []
for _ in range(8):
    pk = np.exp(lam * k) / Z(lam); m1 = pk @ k; var = pk @ k ** 2 - m1 ** 2
    lam -= (m1 - target) / var; its.append(lam)
pk = np.exp(lam * k) / Z(lam)
print("  Newton on the dual (multiplier):", np.round(its, 10))
print(f"  lambda* = {lam:.8f}; p = {pk.round(5)}; mean {pk @ k:.6f}; entropy {-(pk @ np.log(pk)):.5f} (uniform: {np.log(6):.5f})")

print("Example 15.7 — non-negative least squares by projected gradient")
Xn = np.abs(rng.standard_normal((50, 8))); bt = np.array([2.0, 0, 1.5, 0, 0, 0.5, 0, 0]); yn = Xn @ bt + 0.1 * rng.standard_normal(50)
Ln = np.linalg.eigvalsh(Xn.T @ Xn).max()
bp, hist = op.projected_gradient(lambda b: Xn.T @ (Xn @ b - yn), lambda v: np.maximum(v, 0), np.zeros(8), lr=1 / Ln, tol=1e-12, max_iter=200000)
bs, _ = nnls(Xn, yn); bu = np.linalg.lstsq(Xn, yn, rcond=None)[0]
print(f"  projected gradient ({len(hist) - 1} its): {bp.round(4)}\n  scipy nnls: {bs.round(4)}\n  unconstrained LS: {bu.round(4)}")
print(f"  max |PG - nnls| {np.abs(bp - bs).max():.1e}; zeros found by PG at indices {np.nonzero(bp < 1e-10)[0]} (true zeros at {np.nonzero(bt == 0)[0]})")
