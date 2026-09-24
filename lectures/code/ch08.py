"""Reproduces every numerical example in Chapter 8 (Approximation Theory).
Run from the repository root:  python lectures/code/ch08.py"""

import math
import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as C, legendre as Lg
from scipy import integrate
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import approximation as ap  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("=== 8.1 Discrete least squares ===")
x = np.arange(1, 9.0); y = np.array([1.3, 3.5, 4.2, 5.0, 7.0, 8.8, 10.1, 12.5])
print("Ex 8.1.1 sums: m", len(x), "Sx", x.sum(), "Sy", y.sum(), "Sxx", (x * x).sum(), "Sxy", (x * y).sum())
a0, a1 = ap.linear_fit(x, y); E = np.sum((y - a0 - a1 * x) ** 2)
print("   a0 a1", a0, a1, "E", E, "pred x=9:", a0 + 9 * a1)
x2 = np.array([0.0, 0.5, 1, 1.5, 2]); y2 = np.array([1.02, 1.28, 2.15, 3.33, 5.26])
V = np.vander(x2, 3, increasing=True)
print("Ex 8.1.2 normal matrix\n", V.T @ V, "\n rhs", V.T @ y2)
a, E2 = ap.poly_fit(x2, y2, 2); print("   coefficients", a, "E", E2, "residuals", y2 - V @ a)
t = np.array([0.0, 1, 2, 3, 4, 5]); N = np.array([102.0, 148, 225, 334, 490, 745])
b, c = ap.exp_fit(t, N); print("Ex 8.1.3 linearised exp fit b, a:", b, c, "doubling time", math.log(2) / c)
pn, _ = curve_fit(lambda t, b, a: b * np.exp(a * t), t, N, p0=[b, c]); print("   nonlinear LS:", pn, "SSE lin", np.sum((N - b * np.exp(c * t)) ** 2), "SSE nls", np.sum((N - pn[0] * np.exp(pn[1] * t)) ** 2))
T = np.array([0.241, 0.615, 1.0, 1.881, 11.86, 29.46]); Rd = np.array([0.387, 0.723, 1.0, 1.524, 5.203, 9.537])
bb, aa = ap.power_fit(Rd, T); print("Ex 8.1.4 Kepler T = b R^a:", bb, aa)
rng = np.random.default_rng(0)
xs = np.linspace(0, 1, 12); ys = np.sin(2 * np.pi * xs) + 0.15 * rng.standard_normal(12)
for n in range(0, 10):
    a, En = ap.poly_fit(xs, ys, n)
    print(f"Ex 8.1.5 degree {n}: E = {En:.5f}")
for n in (2, 4, 6, 8, 10):
    xx = np.linspace(0, 1, 50); Vn = np.vander(xx, n + 1, increasing=True)
    print(f"Ex 8.1.6 degree {n}: cond(A^T A) = {np.linalg.cond(Vn.T @ Vn):.3e}  cond(A) = {np.linalg.cond(Vn):.3e}")

print("\n=== 8.2 Orthogonal polynomials ===")
for n in (1, 2):
    coef, G = ap.continuous_ls_monomial(np.exp, 0, 1, n)
    tt = np.linspace(0, 1, 1001)
    print(f"Ex 8.2.1 e^x on [0,1] degree {n}: coef {coef}  max err {np.max(np.abs(np.polyval(coef[::-1], tt) - np.exp(tt))):.4e}\n   Gram (Hilbert) matrix\n{G}")
phis = ap.gram_schmidt_polys(4, -1, 1)
for k, p in enumerate(phis):
    print(f"Ex 8.2.2 phi_{k} =", p.coeffs)
tt = np.linspace(-1, 1, 2001)
for n in (1, 2, 3, 4):
    P = ap.legendre_ls(np.exp, n)
    print(f"Ex 8.2.3 e^x Legendre degree {n}: coef {P.coef}  max err {np.max(np.abs(P(tt) - np.exp(tt))):.3e}  L2 err {math.sqrt(integrate.quad(lambda x: (P(x) - math.exp(x)) ** 2, -1, 1)[0]):.3e}")
phisL = ap.gram_schmidt_polys(3, 0, 50, w=lambda x: math.exp(-x))
for k, p in enumerate(phisL):
    print(f"Ex 8.2.4 Laguerre-type monic phi_{k} =", np.round(p.coeffs, 4))
for n in (1, 2, 3):
    coef, _ = ap.continuous_ls_monomial(np.exp, -1, 1, n)
    print(f"Ex 8.2.5 monomial coefficients degree {n}: {coef}")

print("\n=== 8.3 Chebyshev ===")
for n in range(6):
    print(f"Ex 8.3.1 T_{n} =", ap.chebyshev_T(n).coeffs, " roots" if n else "", np.sort(np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))) if n else "")
tt = np.linspace(-1, 1, 20001)
for n in (2, 3, 4, 5):
    monic = ap.chebyshev_T(n) / 2 ** (n - 1)
    alt = np.poly1d(np.poly(np.linspace(-1, 1, n)))
    print(f"Ex 8.3.2 n={n}: max|T~_n| = {np.max(np.abs(monic(tt))):.5f} = 2^(1-n) = {2.0 ** (1 - n):.5f}; equispaced-root monic poly max {np.max(np.abs(alt(tt))):.5f}")
xc = ap.chebyshev_T(4).r
Pc = np.polyfit(xc, np.exp(xc), 3)
print("Ex 8.3.3 Chebyshev nodes", np.sort(xc), "max err", np.max(np.abs(np.polyval(Pc, tt) - np.exp(tt))), "bound e/(2^3 4!)", math.e / (8 * 24))
xe = np.linspace(-1, 1, 4); Pe = np.polyfit(xe, np.exp(xe), 3); print("   equispaced max err", np.max(np.abs(np.polyval(Pe, tt) - np.exp(tt))))
P4 = np.poly1d([1 / 24, 1 / 6, 1 / 2, 1, 1])
P3e, b3 = ap.economize(P4, 3); P2e, b2 = ap.economize(P4, 2)
print("Ex 8.3.4 P4 max err", np.max(np.abs(P4(tt) - np.exp(tt))), "\n   economized to 3:", P3e.coeffs, "added bound", b3, "max err", np.max(np.abs(P3e(tt) - np.exp(tt))),
      "\n   economized to 2:", P2e.coeffs, "added bound", b2, "max err", np.max(np.abs(P2e(tt) - np.exp(tt))),
      "\n   Taylor P3 max err", np.max(np.abs(np.poly1d([1 / 6, 1 / 2, 1, 1])(tt) - np.exp(tt))), "Taylor P2", np.max(np.abs(np.poly1d([1 / 2, 1, 1])(tt) - np.exp(tt))))
for f, name in ((np.exp, "e^x"), (np.abs, "|x|"), (lambda x: 1 / (1 + 25 * x ** 2), "Runge")):
    cc = C.chebinterpolate(f, 40)
    print(f"Ex 8.3.5 {name} Chebyshev coefficients |c_k| k=0,4,8,16,32:", ["%.1e" % abs(cc[k]) for k in (0, 4, 8, 16, 32)] if name != "|x|" else ["%.1e" % abs(cc[k]) for k in (0, 4, 8, 16, 32)])

print("\n=== 8.4 Rational / Pade ===")
p, q = ap.pade([1, 1, 1 / 2, 1 / 6, 1 / 24], 2, 2)
print("Ex 8.4.1 Pade[2/2] e^x p", p, "q", q)
for xv in (0.2, 0.4, 0.6, 0.8, 1.0):
    r = np.polyval(p[::-1], xv) / np.polyval(q[::-1], xv); t4 = sum(xv ** k / math.factorial(k) for k in range(5))
    print(f"   x={xv}: e^x {math.exp(xv):.7f} Taylor4 err {abs(t4 - math.exp(xv)):.2e}  Pade22 err {abs(r - math.exp(xv)):.2e}")
lc = [0, 1, -1 / 2, 1 / 3, -1 / 4]
p, q = ap.pade(lc, 2, 2); print("Ex 8.4.2 Pade[2/2] ln(1+x) p", p, "q", q)
for xv in (0.5, 1.0, 2.0):
    r = np.polyval(p[::-1], xv) / np.polyval(q[::-1], xv); t4 = sum(lc[k] * xv ** k for k in range(5))
    print(f"   x={xv}: ln {math.log1p(xv):.6f} Taylor4 {t4:.6f} Pade {r:.6f}")
atc = [0, 1, 0, -1 / 3, 0, 1 / 5]
p, q = ap.pade(atc, 3, 2); print("Ex 8.4.3 Pade[3/2] arctan p", p, "q", q)
for xv in (0.5, 1.0, 2.0):
    r = np.polyval(p[::-1], xv) / np.polyval(q[::-1], xv); t5 = sum(atc[k] * xv ** k for k in range(6))
    print(f"   x={xv}: atan {math.atan(xv):.6f} Taylor5 {t5:.6f} Pade {r:.6f}")
p, q = ap.pade([1, 1, 1 / 2], 1, 1); print("Ex 8.4.4 Pade[1/1] e^x", p, q, "value at -10:", (1 - 5) / (1 + 5), "Taylor1 at -10:", -9, "e^-10", math.exp(-10))
sq = [1, 0.5, -0.125, 0.0625, -0.0390625]
p, q = ap.pade(sq, 2, 2); print("Ex 8.4.5 Pade[2/2] sqrt(1+x)", p, q)
for xv in (1.0, 3.0, 8.0):
    r = np.polyval(p[::-1], xv) / np.polyval(q[::-1], xv); t4 = sum(sq[k] * xv ** k for k in range(5))
    print(f"   x={xv}: sqrt {math.sqrt(1 + xv):.6f} Taylor4 {t4:.6f} Pade {r:.6f}")

print("\n=== 8.5 Trigonometric approximation ===")
tt = np.linspace(-np.pi, np.pi, 4001)
for n in (1, 3, 5, 11):
    S = np.pi / 2 + sum(2 / (np.pi * k ** 2) * ((-1) ** k - 1) * np.cos(k * tt) for k in range(1, n + 1))
    print(f"Ex 8.5.1 |x| Fourier S_{n}: max err {np.max(np.abs(S - np.abs(tt))):.4f}")
m = 4; xj = -np.pi + np.arange(2 * m) * np.pi / m; yj = xj ** 2
a, b = ap.trig_ls_coeffs(yj, 2); print("Ex 8.5.2 x^2 at 8 points: a", a, "b", b)
for n in (5, 21, 101):
    S = sum(4 / (np.pi * k) * np.sin(k * tt) for k in range(1, n + 1, 2))
    print(f"Ex 8.5.3 square wave S_{n}: max {np.max(S):.4f}  overshoot {np.max(S) - 1:.4f}")
months = np.arange(12); temp = np.array([26.5, 27.9, 29.3, 30.1, 29.8, 28.9, 28.4, 28.3, 27.9, 27.6, 27.1, 26.4])
xm = -np.pi + months * np.pi / 6
a, b = ap.trig_ls_coeffs(temp, 2)
for n in (1, 2):
    fit = ap.trig_eval(a[: n + 1], b[:n], xm)
    print(f"Ex 8.5.4 temperature S_{n}: a {a[: n + 1]} b {b[:n]} RMSE {np.sqrt(np.mean((fit - temp) ** 2)):.3f}")
for name, f in (("smooth e^cos", lambda x: np.exp(np.cos(x))), ("kink |x|", np.abs), ("jump sign", np.sign)):
    ck = np.abs(np.fft.rfft(f(-np.pi + (np.arange(1024) + 0.5) * 2 * np.pi / 1024))) / 512
    print(f"Ex 8.5.5 {name}: |c_k| k=1,5,17,65:", ["%.1e" % ck[k] for k in (1, 5, 17, 65)])

print("\n=== 8.6 FFT ===")
xv = np.array([1.0, 2, 0, -1])
print("Ex 8.6.1 DFT", ap.dft_naive(xv), "np.fft", np.fft.fft(xv))
x8 = np.array([1.0, 1, 1, 1, 0, 0, 0, 0])
print("Ex 8.6.2 FFT 8-pt:", np.round(ap.fft_radix2(x8), 6))
print("   even-half FFT:", np.round(ap.fft_radix2(x8[0::2]), 6), "odd-half:", np.round(ap.fft_radix2(x8[1::2]), 6))
for N in (2 ** 10, 2 ** 16, 2 ** 20):
    print(f"Ex 8.6.3 N={N}: DFT N^2 = {N ** 2:.2e}  FFT (N/2)log2N mults = {N / 2 * math.log2(N):.2e}  ratio {N ** 2 / (N / 2 * math.log2(N)):.0f}")
rng = np.random.default_rng(1)
fs = 100; tsig = np.arange(0, 2, 1 / fs)
sig = np.sin(2 * np.pi * 5 * tsig) + 0.5 * np.sin(2 * np.pi * 12 * tsig) + 0.8 * rng.standard_normal(len(tsig))
F = np.abs(np.fft.rfft(sig)) / (len(sig) / 2); freqs = np.fft.rfftfreq(len(sig), 1 / fs)
top = np.argsort(F)[::-1][:3]; print("Ex 8.6.4 top frequencies", freqs[top], "amplitudes", F[top])
p1 = np.array([1.0, 2, 3]); p2 = np.array([4.0, 5]); Nn = 4
prod = np.real(np.fft.ifft(np.fft.fft(p1, Nn) * np.fft.fft(p2, Nn)))[:4]
print("Ex 8.6.5 (1+2x+3x^2)(4+5x) via FFT:", np.round(prod, 10), "direct", np.convolve(p1, p2))

print("\n=== 8.7 DS ===")
rng = np.random.default_rng(2)
xd = np.sort(rng.uniform(0, 1, 40)); yd = np.sin(2 * np.pi * xd) + 0.2 * rng.standard_normal(40)
folds = np.array_split(rng.permutation(40), 5)
for n in (1, 3, 5, 7, 9, 12):
    err = 0
    for f_ in folds:
        tr = np.setdiff1d(np.arange(40), f_)
        P = np.polynomial.Legendre.fit(xd[tr], yd[tr], n)
        err += np.sum((P(xd[f_]) - yd[f_]) ** 2)
    print(f"Ex 8.7.1 degree {n}: CV MSE {err / 40:.4f}  train MSE {np.mean((np.polynomial.Legendre.fit(xd, yd, n)(xd) - yd) ** 2):.4f}")
N = 512; tn = np.arange(N) / N
clean = np.sin(2 * np.pi * 3 * tn) + 0.5 * np.cos(2 * np.pi * 7 * tn); noisy = clean + 0.5 * rng.standard_normal(N)
Fn = np.fft.rfft(noisy); Fn[20:] = 0; den = np.fft.irfft(Fn, N)
print("Ex 8.7.2 denoise RMSE noisy", np.sqrt(np.mean((noisy - clean) ** 2)), "filtered", np.sqrt(np.mean((den - clean) ** 2)))
days = np.arange(3 * 365); sales = 100 + 0.05 * days + 10 * np.sin(2 * np.pi * days / 365.25) + 4 * np.cos(2 * np.pi * days / 7) + 3 * rng.standard_normal(len(days))
X = np.column_stack([np.ones_like(days), days, np.sin(2 * np.pi * days / 365.25), np.cos(2 * np.pi * days / 365.25), np.sin(2 * np.pi * days / 7), np.cos(2 * np.pi * days / 7)])
beta = np.linalg.lstsq(X, sales, rcond=None)[0]
print("Ex 8.7.3 Fourier-feature regression coefficients", beta, "residual sd", np.std(sales - X @ beta))
xx = np.linspace(0, 1, 30)
for n in (5, 10, 15):
    Vm = np.vander(xx, n + 1, increasing=True); Vc = C.chebvander(2 * xx - 1, n)
    print(f"Ex 8.7.4 degree {n}: cond monomial {np.linalg.cond(Vm):.2e}  cond Chebyshev {np.linalg.cond(Vc):.2e}")
ts = np.cumsum(rng.standard_normal(1000)) * 0 + np.sin(2 * np.pi * np.arange(1000) / 50) + 0.5 * rng.standard_normal(1000)
z = ts - ts.mean(); Fz = np.fft.fft(z, 2048); acf = np.real(np.fft.ifft(Fz * np.conj(Fz)))[:1000] / np.sum(z * z)
acf_direct = np.array([np.sum(z[:1000 - k] * z[k:]) for k in range(60)]) / np.sum(z * z)
print("Ex 8.7.5 ACF lags 0,25,50 via FFT", acf[[0, 25, 50]], "direct", acf_direct[[0, 25, 50]])
