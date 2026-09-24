"""Chapter 12 worked examples — run:  python examples/ch12_examples.py"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import linalg_iterative as it  # noqa: E402
from numlib.linalg_direct import tridiagonal_solve  # noqa: E402

np.set_printoptions(precision=5, suppress=True)

print("Example 12.1 — room with a heater: Poisson with a source")
m = 49; h = 1 / (m + 1); xg = np.linspace(h, 1 - h, m); X, Y = np.meshgrid(xg, xg, indexing="ij")
src = np.where((abs(X - 0.3) < 0.1) & (abs(Y - 0.3) < 0.1), 500.0, 0.0)
A = it.poisson_2d_sparse(m).tocsc() / h ** 2; b = src.ravel()  # -Laplace u = f, u = 0 on walls
t0 = time.perf_counter(); u = spla.spsolve(A, b); t1 = time.perf_counter()
ucg, res = it.conjugate_gradient(A, b, tol=1e-10); t2 = time.perf_counter()
U = u.reshape(m, m); imax = np.unravel_index(np.argmax(U), U.shape)
print(f"  {m * m} unknowns: sparse direct {t1 - t0:.3f}s, CG {len(res) - 1} iterations {t2 - t1:.3f}s, max diff {np.abs(u - ucg).max():.1e}")
print(f"  hottest point ({xg[imax[0]]:.2f}, {xg[imax[1]]:.2f}) T = {U.max():.4f}; room centre T = {U[m // 2, m // 2]:.4f}; opposite corner (0.8,0.8) T = {U[np.argmin(abs(xg - 0.8)), np.argmin(abs(xg - 0.8))]:.4f}")

print("Example 12.2 — when does the centre of a rod cool below 10% of its initial value?")
L_ = 1.0; alpha2 = 1.0
def centre_time(method, m, k):
    x = np.linspace(0, 1, m + 1); w = np.sin(np.pi * x)[1:-1] + 0.3 * np.sin(3 * np.pi * x)[1:-1]
    lam = alpha2 * k / (1 / m) ** 2; n = m - 1; t = 0.0; c0 = w[n // 2]
    while True:
        prev = w[n // 2]
        if method == "ftcs":
            wn = w.copy(); wn[1:-1] = (1 - 2 * lam) * w[1:-1] + lam * (w[:-2] + w[2:]); wn[0] = (1 - 2 * lam) * w[0] + lam * w[1]; wn[-1] = (1 - 2 * lam) * w[-1] + lam * w[-2]; w = wn
        else:
            rhs = (1 - lam) * w; rhs[1:] += lam / 2 * w[:-1]; rhs[:-1] += lam / 2 * w[1:]
            w = tridiagonal_solve(-lam / 2 * np.ones(n - 1), (1 + lam) * np.ones(n), -lam / 2 * np.ones(n - 1), rhs)
        t += k
        if w[n // 2] < 0.1 * c0:
            return t - k + k * (prev - 0.1 * c0) / (prev - w[n // 2]), lam
from scipy.optimize import brentq
c = lambda t: np.exp(-np.pi ** 2 * t) - 0.3 * np.exp(-9 * np.pi ** 2 * t)
t_exact = brentq(lambda t: c(t) - 0.1 * c(0), 0, 1)
for method, m, k in (("ftcs", 20, 0.001), ("ftcs", 20, 0.0012), ("cn", 20, 0.01), ("cn", 20, 0.002)):
    with np.errstate(all="ignore"):
        tc, lam = centre_time(method, m, k)
    print(f"  {method.upper():4s} h=1/{m} k={k} (lambda={lam:.2f}): t* = {tc:.5f}")
print(f"  exact t* = {t_exact:.5f}")

print("Example 12.3 — advection-diffusion of a pollutant (Peclet number)")
Lx, v, D = 1.0, 1.0, 0.002; M = 100; h = Lx / M; x = np.linspace(0, Lx, M + 1)
c0 = np.exp(-((x - 0.2) / 0.05) ** 2); k = 0.4 * h / v; steps = int(round(0.5 / k))
Pe = v * h / (2 * D)
for scheme in ("central", "upwind"):
    cc = c0.copy()
    for _ in range(steps):
        adv = (cc[2:] - cc[:-2]) / (2 * h) if scheme == "central" else (cc[1:-1] - cc[:-2]) / h
        cc[1:-1] = cc[1:-1] + k * (-v * adv + D * (cc[2:] - 2 * cc[1:-1] + cc[:-2]) / h ** 2)
    sig2 = 0.05 ** 2 / 2 + 2 * D * 0.5; exact = np.sqrt(0.05 ** 2 / 2 / sig2) * np.exp(-(x - 0.7) ** 2 / (2 * sig2))
    print(f"  {scheme:7s}: cell Peclet {Pe:.2f}; peak {cc.max():.4f} at x={x[np.argmax(cc)]:.2f} (exact peak {exact.max():.4f} at 0.70); min {cc.min():.4f}; max error {np.abs(cc - exact).max():.4f}")

print("Example 12.4 — a plucked guitar string: harmonics via FFT")
Ls, cs = 1.0, 1.0; M = 200; h = Ls / M; lam = 1.0; k = lam * h / cs; x = np.linspace(0, Ls, M + 1)
f0 = np.where(x < 0.2, x / 0.2, (1 - x) / 0.8) * 0.01
w0 = f0.copy(); w1 = w0.copy(); w1[1:-1] = (1 - lam ** 2) * w0[1:-1] + lam ** 2 / 2 * (w0[2:] + w0[:-2])
probe = []; j = int(0.9 * M)
for n_ in range(4000):
    w2 = np.zeros_like(w1); w2[1:-1] = 2 * (1 - lam ** 2) * w1[1:-1] + lam ** 2 * (w1[2:] + w1[:-2]) - w0[1:-1]
    w0, w1 = w1, w2; probe.append(w1[j])
sig = np.array(probe); F = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))); fr = np.fft.rfftfreq(len(sig), k)
peaks = [fr[np.argmax(F * (abs(fr - n / 2) < 0.2))] for n in range(1, 7)]
amps = [F[np.argmin(abs(fr - n / 2))] for n in range(1, 7)]
print("  detected frequencies (theory n c / 2L = n/2):", np.round(peaks, 3))
print("  relative amplitudes of harmonics 1..6:", np.round(np.array(amps) / amps[0], 3), " theory |sin(n pi 0.2)| sin(n pi 0.9)/n^2 ratios:", np.round([abs(np.sin(n * np.pi * 0.2) * np.sin(n * np.pi * 0.9)) / n ** 2 / abs(np.sin(np.pi * 0.2) * np.sin(np.pi * 0.9)) for n in range(1, 7)], 3))

print("Example 12.5 — Black-Scholes via the heat equation in log-price")
K, r, sig_, T = 100.0, 0.03, 0.3, 0.5
def bs_call(S, tau):
    d1 = (np.log(S / K) + (r + sig_ ** 2 / 2) * tau) / (sig_ * np.sqrt(tau)); return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d1 - sig_ * np.sqrt(tau))
for M in (100, 200, 400):
    xmin, xmax = np.log(K) - 3, np.log(K) + 3; xs = np.linspace(xmin, xmax, M + 1); dx = xs[1] - xs[0]
    N = M // 2; dt = T / N
    V = np.maximum(np.exp(xs) - K, 0.0)
    a_ = sig_ ** 2 / 2; b_ = r - sig_ ** 2 / 2
    lo = a_ / dx ** 2 - b_ / (2 * dx); di = -2 * a_ / dx ** 2 - r; up = a_ / dx ** 2 + b_ / (2 * dx)
    n = M - 1
    Ai = sp.diags([lo * np.ones(n - 1), di * np.ones(n), up * np.ones(n - 1)], [-1, 0, 1], format="csc")
    Ip = sp.identity(n, format="csc"); lu = spla.splu(Ip - dt / 2 * Ai); B = Ip + dt / 2 * Ai
    for s_ in range(N):
        tau = (s_ + 1) * dt; bnd_old = np.exp(xmax) - K * np.exp(-r * (tau - dt)); bnd_new = np.exp(xmax) - K * np.exp(-r * tau)
        rhs = B @ V[1:-1]; rhs[-1] += dt / 2 * up * (bnd_old + bnd_new)
        V[1:-1] = lu.solve(rhs); V[0] = 0.0; V[-1] = bnd_new
    val = np.interp(np.log(100.0), xs, V)
    print(f"  M={M}: V(S=100) {val:.6f}  error {abs(val - bs_call(100.0, T)):.2e}")
print(f"  exact {bs_call(100.0, T):.6f}")

print("Example 12.6 — heat diffusion on a graph (label smoothing)")
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (5, 6), (6, 7), (7, 8), (6, 8)]
nG = 9; Adj = np.zeros((nG, nG))
for i, j in edges:
    Adj[i, j] = Adj[j, i] = 1
Lg = np.diag(Adj.sum(1)) - Adj
f0 = np.zeros(nG); f0[0] = 1.0
for t in (0.5, 2.0, 10.0):
    exact = expm(-t * Lg) @ f0
    ie = f0.copy(); steps = 20
    for _ in range(steps):
        ie = np.linalg.solve(np.eye(nG) + (t / steps) * Lg, ie)
    print(f"  t={t:4.1f}: heat kernel e^(-tL) f0 = {exact.round(3)}  implicit Euler (20 steps) max diff {np.abs(ie - exact).max():.1e}  total heat {exact.sum():.3f}")
