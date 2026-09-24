"""Chapter 9 worked examples — run:  python examples/ch09_examples.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import eigen as eg  # noqa: E402

np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

print("Example 9.1 — Gershgorin bounds for a covariance matrix before any computation")
S = np.array([[2.0, 0.3, 0.2, 0.1], [0.3, 1.5, 0.4, 0.0], [0.2, 0.4, 1.0, 0.2], [0.1, 0.0, 0.2, 0.5]])
for c, r in eg.gershgorin_discs(S):
    print(f"  disc centre {c}, radius {r:.1f} -> [{c - r:.1f}, {c + r:.1f}]")
print("  eigenvalues", np.linalg.eigvalsh(S), " PD guaranteed since all discs lie in (0, inf):", all(c - r > 0 for c, r in eg.gershgorin_discs(S)))

print("Example 9.2 — Leslie population model: dominant eigenvalue = long-run growth")
Lm = np.array([[0.0, 1.2, 1.1, 0.3], [0.8, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 0.5, 0]])
lam, v, hist = eg.power_iteration(Lm, np.ones(4), tol=1e-12)
v = np.abs(v) / np.abs(v).sum()
ev = np.linalg.eigvals(Lm); ev = ev[np.argsort(-abs(ev))]
print(f"  power method: lambda = {lam:.6f} after {len(hist) - 1} iterations; stable age distribution {v}")
print(f"  |lambda2/lambda1| = {abs(ev[1]) / abs(ev[0]):.4f} -> about {np.log(1e-12) / np.log(abs(ev[1]) / abs(ev[0])):.0f} iterations for 1e-12; eigenvalues {ev}")
print(f"  growth: {100 * (lam - 1):.2f}% per period")

print("Example 9.3 — shift-and-invert to target an interior eigenvalue")
n = 100; T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
exact = 2 - 2 * np.cos(np.pi * np.arange(1, n + 1) / (n + 1))
target = 1.0; idx = np.argmin(abs(exact - target))
lam, v, hist = eg.inverse_iteration(T, shift=target, x0=rng.standard_normal(n), tol=1e-14)
srt = np.sort(abs(exact - target))
print(f"  closest eigenvalue to 1: {exact[idx]:.10f}; inverse iteration: {lam:.10f} in {len(hist) - 1} iterations; ratio |l_k-q|/|l_next-q| = {srt[0] / srt[1]:.3f}")
lam2, v2, h2 = eg.rayleigh_quotient_iteration(T, v + 0.3 * rng.standard_normal(n) / 10)
print(f"  Rayleigh quotient iteration from a perturbed vector: {lam2:.12f} in {len(h2) - 1} iterations")

print("Example 9.4 — Wilkinson shift: QR steps per eigenvalue")
A = rng.standard_normal((6, 6)); A = (A + A.T) / 2
Tt, Q = eg.householder_tridiagonal(A)
print("  tridiagonal form:\n", Tt, "\n  eigenvalues preserved:", np.allclose(np.linalg.eigvalsh(Tt), np.linalg.eigvalsh(A)))
Ak = Tt.copy(); m = 6; steps = []
while m > 1:
    k = 0
    while abs(Ak[m - 1, m - 2]) > 1e-14 * (abs(Ak[m - 1, m - 1]) + abs(Ak[m - 2, m - 2])):
        a, b, c = Ak[m - 2, m - 2], Ak[m - 1, m - 2], Ak[m - 1, m - 1]
        d = (a - c) / 2; mu = c - np.copysign(b * b, d if d != 0 else 1.0) / (abs(d) + np.hypot(d, b))
        Qk, Rk = np.linalg.qr(Ak[:m, :m] - mu * np.eye(m)); Ak[:m, :m] = Rk @ Qk + mu * np.eye(m); k += 1
        if k <= 3 and m == 6:
            print(f"    step {k}: |subdiag| = {abs(Ak[m - 1, m - 2]):.2e}")
    steps.append(k); m -= 1
print("  QR steps per deflation:", steps, " eigenvalues:", np.sort(np.diag(Ak)), " vs eigvalsh:", np.linalg.eigvalsh(A))

print("Example 9.5 — Markov chain mixing: web-surfer model")
P = np.array([[0.1, 0.6, 0.3, 0.0], [0.4, 0.1, 0.4, 0.1], [0.3, 0.3, 0.2, 0.2], [0.5, 0.0, 0.0, 0.5]])
w, V = np.linalg.eig(P.T); i = np.argmin(abs(w - 1)); pi = np.real(V[:, i]); pi /= pi.sum()
x = np.array([0, 0, 0, 1.0]); tvs = []
for k in range(1, 21):
    x = x @ P; tvs.append(0.5 * np.abs(x - pi).sum())
mods = np.sort(abs(w))[::-1]
print("  stationary distribution", pi, " |lambda| sorted", mods)
print("  TV distance after 1,2,5,10,20 steps:", [float(round(tvs[k - 1], 6)) for k in (1, 2, 5, 10, 20)], " ratio ~ |lambda2| =", round(tvs[9] / tvs[8], 4))

print("Example 9.6 — SVD for denoising a low-rank data matrix")
m_, n_, r = 200, 50, 3
Ltrue = rng.standard_normal((m_, r)) @ rng.standard_normal((r, n_)); Xn = Ltrue + 0.5 * rng.standard_normal((m_, n_))
U, s, Vt = np.linalg.svd(Xn, full_matrices=False)
print("  top 6 singular values:", s[:6].round(2), " noise edge sigma(sqrt m + sqrt n) =", round(0.5 * (np.sqrt(m_) + np.sqrt(n_)), 2))
for k in (1, 3, 5, 10, 50):
    Xk = (U[:, :k] * s[:k]) @ Vt[:k]
    print(f"  rank {k:2d}: relative error vs truth {np.linalg.norm(Xk - Ltrue) / np.linalg.norm(Ltrue):.4f}")
