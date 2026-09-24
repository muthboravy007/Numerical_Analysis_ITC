"""Chapter 10 worked examples — run:  python examples/ch10_examples.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import nonlinear_systems as ns  # noqa: E402
from numlib import regression as rg  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(0)

print("Example 10.1 — trilateration: Newton with 2 beacons, Gauss-Newton with 4")
beacons = np.array([[0.0, 0], [10, 0], [0, 10], [10, 10]]); pos = np.array([3.0, 4.0])
dist = np.linalg.norm(beacons - pos, axis=1)
F = lambda p: np.array([np.sum((p - beacons[i]) ** 2) - dist[i] ** 2 for i in range(2)])
J = lambda p: np.array([2 * (p - beacons[i]) for i in range(2)])
x, hist = ns.newton_system(F, J, [5.0, 5.0])
print("  exact distances", dist.round(6), "\n  Newton (2 beacons) from (5,5):", [h.round(6) for h in hist])
x, hist = ns.newton_system(F, J, [5.0, -5.0]); print("  from (5,-5) it converges to the mirror solution", x.round(6))
noisy = dist + 0.1 * rng.standard_normal(4)
r = lambda p: np.linalg.norm(beacons - p, axis=1) - noisy
Jr = lambda p: (p - beacons) / np.linalg.norm(beacons - p, axis=1)[:, None]
pg, hg = rg.gauss_newton(r, Jr, [5.0, 5.0])
cov = np.linalg.inv(Jr(pg).T @ Jr(pg)) * 0.1 ** 2
print(f"  Gauss-Newton (4 noisy ranges, sd 0.1): {pg.round(4)} in {len(hg) - 1} iterations, SE {np.sqrt(np.diag(cov)).round(4)}")

print("Example 10.2 — endemic equilibrium of an SIR model with births")
mu, beta, gamma = 0.02, 0.5, 0.1
G = lambda z: np.array([mu - beta * z[0] * z[1] - mu * z[0], beta * z[0] * z[1] - (gamma + mu) * z[1]])
JG = lambda z: np.array([[-beta * z[1] - mu, -beta * z[0]], [beta * z[1], beta * z[0] - gamma - mu]])
z, hist = ns.newton_system(G, JG, [0.5, 0.05])
R0 = beta / (gamma + mu)
print(f"  Newton: (S*, I*) = {z.round(8)} in {len(hist) - 1} iterations; theory S* = 1/R0 = {1 / R0:.8f}, I* = mu(R0-1)/beta = {mu * (R0 - 1) / beta:.8f}")
print("  Jacobian eigenvalues at equilibrium:", np.linalg.eigvals(JG(z)).round(5), "(negative real parts -> stable, complex -> damped oscillations)")
z2, _ = ns.newton_system(G, JG, [0.9, 1e-4]); print("  from (0.9, 1e-4): converges to", z2.round(6), "(disease-free equilibrium, unstable when R0 > 1)")

print("Example 10.3 — Broyden vs Newton on a discretised nonlinear BVP (n = 50)")
n = 50; h = 1 / (n + 1)
def Fb(w):
    wl = np.r_[0, w[:-1]]; wr = np.r_[w[1:], 0]
    return -wl + 2 * w - wr - h * h * 2 * np.exp(w)
def Jb(w):
    return np.diag(2 - h * h * 2 * np.exp(w)) - np.eye(n, k=1) - np.eye(n, k=-1)
cnt = {"F": 0}
def Fc(w):
    cnt["F"] += 1; return Fb(w)
w0 = np.zeros(n)
wn, hn = ns.newton_system(Fc, Jb, w0, tol=1e-12); nF_newton = cnt["F"]
cnt["F"] = 0; wb, hb = ns.broyden(Fc, w0, J0=Jb(w0), tol=1e-12); nF_b = cnt["F"]
print(f"  Newton: {len(hn) - 1} iterations, {nF_newton} F-evals, {len(hn) - 1} Jacobians; Broyden: {len(hb) - 1} iterations, {nF_b} F-evals, 1 Jacobian; max diff {np.abs(wn - wb).max():.1e}; max w {wn.max():.6f}")

print("Example 10.4 — all four solutions of x^2 + y^2 = 4, xy = 1")
F4 = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 4, v[0] * v[1] - 1]); J4 = lambda v: np.array([[2 * v[0], 2 * v[1]], [v[1], v[0]]])
found = []
for s in [(2, 0.5), (0.5, 2), (-2, -0.5), (-0.5, -2), (1, 1.1)]:
    x, h = ns.newton_system(F4, J4, s); found.append(x.round(8)); print(f"  start {s}: -> {x.round(8)} ({len(h) - 1} its)")
a = np.sqrt(2 + np.sqrt(3)); b = np.sqrt(2 - np.sqrt(3)); print(f"  exact: (+-{a:.8f}, +-{b:.8f}) and (+-{b:.8f}, +-{a:.8f}); start (1,1.1) is near the singular line x=y")

print("Example 10.5 — best-response dynamics as a fixed-point iteration (Cournot duopoly)")
A_, c1, c2 = 100.0, 10.0, 16.0
BR = lambda q: np.array([(A_ - c1 - q[1]) / 2, (A_ - c2 - q[0]) / 2])
q, hist = ns.fixed_point_system(BR, [0.0, 0.0], tol=1e-10)
print(f"  equilibrium {q.round(6)} after {len(hist) - 1} iterations; theory ((A-2c1+c2)/3, (A-2c2+c1)/3) = ({(A_ - 2 * c1 + c2) / 3:.6f}, {(A_ - 2 * c2 + c1) / 3:.6f}); Jacobian norm 1/2 -> error halves each step")
print("  first iterates:", [hh.round(3) for hh in hist[:5]])

print("Example 10.6 — fitting a circle: algebraic (linear) vs geometric (Gauss-Newton)")
th = rng.uniform(0, np.pi / 2, 30); cx, cy, R = 2.0, -1.0, 3.0
pts = np.column_stack([cx + R * np.cos(th), cy + R * np.sin(th)]) + 0.05 * rng.standard_normal((30, 2))
A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(30)]); bvec = (pts ** 2).sum(1)
sol = np.linalg.lstsq(A, bvec, rcond=None)[0]; ca = sol[:2] / 2; Ra = np.sqrt(sol[2] + ca @ ca)
rr = lambda p: np.linalg.norm(pts - p[:2], axis=1) - p[2]
Jrr = lambda p: np.column_stack([-(pts - p[:2]) / np.linalg.norm(pts - p[:2], axis=1)[:, None], -np.ones(30)])
pgeo, hgeo = rg.levenberg_marquardt(rr, Jrr, np.r_[ca, Ra])
print(f"  algebraic (Kasa): centre {ca.round(4)}, radius {Ra:.4f};  geometric (LM from Kasa): centre {pgeo[:2].round(4)}, radius {pgeo[2]:.4f}; truth (2,-1), 3")
print(f"  RMS geometric residual: Kasa {np.sqrt(np.mean(rr(np.r_[ca, Ra]) ** 2)):.4f}  LM {np.sqrt(np.mean(rr(pgeo) ** 2)):.4f}")
