"""Chapter 8 worked examples — run:  python examples/ch08_examples.py"""
import math
import os
import sys
import time

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import approximation as ap  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(0)

print("Example 8.1 — Zipf's law by log-log least squares")
rank = np.arange(1, 51.0); freq = 0.08 * rank ** -1.05 * np.exp(0.08 * rng.standard_normal(50))
b, a = ap.power_fit(rank, freq)
(bn, an), _ = optimize.curve_fit(lambda r, b, a: b * r ** a, rank, freq, p0=(b, a))
print(f"  log-log fit: freq = {b:.4f} r^{a:.4f};  nonlinear LS on raw scale: {bn:.4f} r^{an:.4f}  (true 0.08 r^-1.05)")
print(f"  SSE (raw scale): loglog {np.sum((b * rank ** a - freq) ** 2):.3e}  nonlinear {np.sum((bn * rank ** an - freq) ** 2):.3e}")

print("Example 8.2 — Chebyshev approximation of the sigmoid on [-6, 6]")
sig = lambda x: 1 / (1 + np.exp(-x))
xx = np.linspace(-6, 6, 20001)
for n in (3, 5, 9, 15, 21):
    c = np.polynomial.chebyshev.chebinterpolate(lambda t: sig(6 * t), n)
    err = np.abs(np.polynomial.chebyshev.chebval(xx / 6, c) - sig(xx)).max()
    tay = np.abs(sum([0.5, 0.25, 0, -1 / 48, 0, 1 / 480][k] * xx ** k for k in range(6)) - sig(xx)).max() if n == 5 else None
    print(f"  degree {n:2d}: max error {err:.2e}" + (f"   (Taylor degree 5 about 0: max error {tay:.2e})" if tay is not None else ""))

print("Example 8.3 — discrete orthogonal polynomials make the normal equations diagonal")
xd = np.linspace(0, 1, 11); yd = np.exp(xd) * np.cos(3 * xd)
P = [np.ones_like(xd), xd - xd.mean()]
for k in range(2, 4):
    pk, pk1 = P[-1], P[-2]
    B = np.sum(xd * pk * pk) / np.sum(pk * pk); C = np.sum(xd * pk * pk1) / np.sum(pk1 * pk1)
    P.append((xd - B) * pk - C * pk1)
Pm = np.column_stack(P)
G = Pm.T @ Pm
coef = (Pm.T @ yd) / np.diag(G)
V = np.vander(xd, 4, increasing=True)
print("  Gram matrix of the recurrence basis (off-diagonals ~0):\n", G.round(8))
print("  coefficients", coef.round(6), "; same fit as monomial LS:", np.allclose(Pm @ coef, V @ np.linalg.lstsq(V, yd, rcond=None)[0]))
print(f"  cond(Gram) monomial {np.linalg.cond(V.T @ V):.2e} vs orthogonal {np.linalg.cond(G):.2e}")
print("  coefficients for degree 2 fit are the first three (unchanged):", ((Pm[:, :3].T @ yd) / np.diag(G)[:3]).round(6))

print("Example 8.4 — Pade approximation of ln(1+x)")
tay = [0, 1, -1 / 2, 1 / 3, -1 / 4]
p, q = ap.pade(tay, 2, 2)
print("  [2/2] p", p.round(6), " q", q.round(6))
for x in (0.5, 1.0, 2.0, 4.0):
    r = np.polyval(p[::-1], x) / np.polyval(q[::-1], x); t4 = sum(tay[k] * x ** k for k in range(5))
    print(f"  x={x}: ln(1+x) {np.log1p(x):.6f}  Pade {r:.6f} (err {abs(r - np.log1p(x)):.1e})  Taylor-4 {t4:.6f} (err {abs(t4 - np.log1p(x)):.1e})")

print("Example 8.5 — seasonal cycle of daily temperature")
days = np.arange(365); tt = 2 * np.pi * days / 365
temp = 28 + 3 * np.cos(tt - 2 * np.pi * 110 / 365) + 1.0 * np.cos(2 * tt - 0.5) + 0.8 * rng.standard_normal(365)
X = np.column_stack([np.ones(365), np.cos(tt), np.sin(tt), np.cos(2 * tt), np.sin(2 * tt)])
beta = np.linalg.lstsq(X, temp, rcond=None)[0]
amp1 = np.hypot(beta[1], beta[2]); phase = np.arctan2(beta[2], beta[1]); peak_day = (phase % (2 * np.pi)) / (2 * np.pi) * 365
print(f"  mean {beta[0]:.3f}, annual amplitude {amp1:.3f} (true 3), peak day {peak_day:.1f} (true 110), semi-annual amplitude {np.hypot(beta[3], beta[4]):.3f} (true 1)")
print("  X^T X / 365 (orthogonality on a full cycle):", np.round(np.diag(X.T @ X) / 365, 4), " max off-diagonal", np.abs(X.T @ X - np.diag(np.diag(X.T @ X))).max().round(10))

print("Example 8.6 — convolution by FFT (moving-average smoothing)")
for n, k in ((10 ** 5, 101), (10 ** 5, 10001), (10 ** 6, 10001)):
    sgn = rng.standard_normal(n); ker = np.ones(k) / k
    t0 = time.perf_counter(); d = np.convolve(sgn, ker, mode="full"); t1 = time.perf_counter()
    L = 1 << int(np.ceil(np.log2(n + k - 1)))
    t2 = time.perf_counter(); f = np.fft.irfft(np.fft.rfft(sgn, L) * np.fft.rfft(ker, L), L)[: n + k - 1]; t3 = time.perf_counter()
    print(f"  n={n:>7d}, kernel {k}: direct {t1 - t0:.4f}s  FFT {t3 - t2:.4f}s  max diff {np.abs(d - f).max():.1e}")

print("Example 8.7 — spectral leakage and windowing")
fs, Tsec = 100.0, 4.0; t = np.arange(0, Tsec, 1 / fs)
x = np.sin(2 * np.pi * 10.1 * t) + 0.01 * np.sin(2 * np.pi * 14.3 * t)
freqs = np.fft.rfftfreq(len(t), 1 / fs)
for name, w in (("rectangular", np.ones(len(t))), ("Hann", np.hanning(len(t)))):
    P = np.abs(np.fft.rfft(x * w)) ** 2
    Pdb = 10 * np.log10(P / P.max())
    i14 = np.argmin(abs(freqs - 14.25))
    print(f"  {name:11s}: power near 14.25 Hz {Pdb[i14]:.1f} dB, leakage floor at 20 Hz {Pdb[np.argmin(abs(freqs - 20))]:.1f} dB, weak tone visible? {Pdb[i14] > Pdb[np.argmin(abs(freqs - 17))] + 10}")
print(f"  frequency resolution 1/T = {1 / Tsec} Hz; 10.1 Hz is not on the grid (bins at multiples of 0.25 Hz)")
