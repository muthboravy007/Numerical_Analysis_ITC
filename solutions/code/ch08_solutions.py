"""Numerical answers for exercises/ch08_exercises.md."""
import os
import sys
import time

import numpy as np
from scipy import integrate, optimize
from scipy.fft import dct, idct, dctn, idctn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import approximation as ap  # noqa: E402

np.set_printoptions(precision=6, suppress=True)

# ---------- B ----------
x = np.arange(6.0); y = np.array([2.1, 2.9, 4.2, 4.8, 6.1, 7.2])
a0, a1 = ap.linear_fit(x, y); res = y - a0 - a1 * x
print("B1 sums", x.sum(), y.sum(), (x * x).sum(), (x * y).sum(), f" a0 {a0:.6f} a1 {a1:.6f} E {np.sum(res**2):.6f} R2 {1 - np.sum(res**2) / np.sum((y - y.mean())**2):.6f}")

x = np.arange(-2.0, 3); y = np.array([4.1, 1.2, 0.1, 0.9, 3.9])
a, E = ap.poly_fit(x, y, 2)
print("B2 sums x^k:", [np.sum(x**k) for k in range(5)], "x^k y:", [np.sum(x**k * y) for k in range(3)], " coeffs", a, "E", round(E, 6))

x = np.arange(5.0); y = np.array([3.0, 4.4, 6.7, 10.0, 14.9])
b_lin, a_lin = ap.exp_fit(x, y)
(b_nl, a_nl), _ = optimize.curve_fit(lambda t, b, a: b * np.exp(a * t), x, y, p0=(b_lin, a_lin))
E_lin = np.sum((y - b_lin * np.exp(a_lin * x))**2); E_nl = np.sum((y - b_nl * np.exp(a_nl * x))**2)
print(f"B3 linearised b {b_lin:.5f} a {a_lin:.5f} SSE {E_lin:.5f} | nonlinear b {b_nl:.5f} a {a_nl:.5f} SSE {E_nl:.5f}")

c, G = ap.continuous_ls_monomial(np.sqrt, 0, 1, 1)
err = integrate.quad(lambda t: (np.sqrt(t) - c[0] - c[1] * t)**2, 0, 1)[0]
print("B4a sqrt on [0,1] linear LS:", c, "(4/15, 4/5) =", 4 / 15, 4 / 5, " L2 error^2", err, "max err", max(abs(np.sqrt(t) - c[0] - c[1] * t) for t in np.linspace(0, 1, 100001)))
L = ap.legendre_ls(np.abs, 4)
print("B4b |x| Legendre coeffs up to 4:", L.coef, " as power series", L.convert(kind=np.polynomial.Polynomial).coef)
for n in (0, 2, 4):
    Ln = ap.legendre_ls(np.abs, n)
    print(f"    degree {n}: L2 err {np.sqrt(integrate.quad(lambda t: (abs(t) - Ln(t))**2, -1, 1, points=[0])[0]):.6f}  max err {max(abs(abs(t) - Ln(t)) for t in np.linspace(-1, 1, 20001)):.6f}")

ph = ap.gram_schmidt_polys(3, 0, 1)
print("B5 monic orthogonal on [0,1]:", [np.round(p.coeffs, 6) for p in ph])
ph = ap.gram_schmidt_polys(2, 0, 1, w=lambda t: t)
print("   weight x on [0,1]:", [np.round(p.coeffs, 6) for p in ph])

nodes = np.cos((2 * np.arange(1, 4) - 1) * np.pi / 6)
f = lambda t: np.log(t + 2)
P = np.polyfit(nodes, f(nodes), 2)
tt = np.linspace(-1, 1, 200001)
print("B6a Chebyshev nodes", nodes, "interp coeffs", P, "max err", np.abs(f(tt) - np.polyval(P, tt)).max(),
      "bound max|f'''|/(3! 2^2) =", 2 / 1**3 / (6 * 4))
eq = np.polyfit(np.linspace(-1, 1, 3), f(np.linspace(-1, 1, 3)), 2)
print("    equispaced max err", np.abs(f(tt) - np.polyval(eq, tt)).max())
p5 = np.poly1d([1 / 120, 0, -1 / 6, 0, 1, 0])
pe, bound = ap.economize(p5, 3)
print("B6b economised sin:", pe.coeffs, "added bound", bound, " max|sin - P5|", np.abs(np.sin(tt) - p5(tt)).max(),
      " max|sin - econ|", np.abs(np.sin(tt) - pe(tt)).max(), " max|sin - P3 Taylor|", np.abs(np.sin(tt) - (tt - tt**3 / 6)).max())

pc, qc = ap.pade([1, 0, -1 / 2, 0, 1 / 24], 2, 2)
r = lambda t: np.polyval(pc[::-1], t) / np.polyval(qc[::-1], t)
print("B7a cos [2/2]: p", pc, "q", qc, " r(1)", r(1.0), "cos1", np.cos(1), " err", abs(r(1.0) - np.cos(1)),
      " Taylor4 err", abs(1 - 0.5 + 1 / 24 - np.cos(1)))
pl, ql = ap.pade([0, 1, -1 / 2], 1, 1)
rl = lambda t: np.polyval(pl[::-1], t) / np.polyval(ql[::-1], t)
print("B7b ln(1+x) [1/1]: p", pl, "q", ql, " at x=1:", rl(1.0), np.log(2), " Taylor2 at 1:", 0.5, " at x=0.5:", rl(0.5), np.log(1.5), 0.5 - 0.125)

xs = -np.pi + np.arange(8) * np.pi / 4
aa, bb = ap.trig_ls_coeffs(xs, 3)
print("B8 f=x on 8 points: a", aa, "b", bb, " exact Fourier b_k = 2(-1)^{k+1}/k:", [2 * (-1)**(k + 1) / k for k in (1, 2, 3)])
print("   S_3 at x = pi/2:", ap.trig_eval(aa, bb, np.pi / 2))

for v in ([2, 1, 0, 1], [0, 1, 0, -1], [1, 2, 3, 4]):
    print("B9 DFT of", v, "=", np.round(np.fft.fft(v), 6), " numlib", np.round(ap.fft_radix2(v), 6))

# ---------- C ----------
print("C1 Runge function 1/(1+25x^2), 200 points on [-1,1]")
xg = np.linspace(-1, 1, 200); yg = 1 / (1 + 25 * xg**2); xf = np.linspace(-1, 1, 5001); yf = 1 / (1 + 25 * xf**2)
for n in (5, 10, 15, 20, 30):
    V = np.vander(xg, n + 1, increasing=True)
    a_ne = np.linalg.solve(V.T @ V, V.T @ yg)
    a_qr = np.linalg.lstsq(V, yg, rcond=None)[0]
    C = np.polynomial.chebyshev.chebvander(xg, n)
    c_ch = np.linalg.lstsq(C, yg, rcond=None)[0]
    Vf = np.vander(xf, n + 1, increasing=True)
    e_ne = np.abs(Vf @ a_ne - yf).max(); e_qr = np.abs(Vf @ a_qr - yf).max(); e_ch = np.abs(np.polynomial.chebyshev.chebval(xf, c_ch) - yf).max()
    print(f"   n={n}: cond(V) {np.linalg.cond(V):.1e} cond(V^TV) {np.linalg.cond(V.T @ V):.1e} cond(Cheb) {np.linalg.cond(C):.1f}  max err NE {e_ne:.2e} QR {e_qr:.2e} Cheb {e_ch:.2e}")

print("C2 FFT timings")
rng = np.random.default_rng(0)
for p in (8, 10, 12, 14, 16):
    N = 2**p; v = rng.standard_normal(N)
    t0 = time.perf_counter(); F1 = ap.fft_radix2(v); t1 = time.perf_counter()
    F2 = np.fft.fft(v); t2 = time.perf_counter()
    if p <= 12:
        F3 = ap.dft_naive(v); t3 = time.perf_counter(); tn = f"{t3 - t2:.4f}s"
    else:
        tn = "-"
    print(f"   N=2^{p}: radix2 {t1 - t0:.4f}s numpy {t2 - t1:.5f}s naive {tn}  max diff {np.abs(F1 - F2).max():.1e}")

print("C3 Chebyshev coefficient decay (DCT of values at Chebyshev extreme points)")
def cheb_coeffs(fun, N):
    k = np.arange(N + 1); xk = np.cos(np.pi * k / N)
    c = dct(fun(xk), type=1) / N; c[0] /= 2; c[-1] /= 2
    return c
for name, fun in (("exp", np.exp), ("runge", lambda t: 1 / (1 + 25 * t**2)), ("abs", np.abs), ("tanh(50x)", lambda t: np.tanh(50 * t))):
    c = cheb_coeffs(fun, 2048)
    nneed = np.max(np.nonzero(np.abs(c) > 1e-12 * np.abs(c).max())[0]) if np.any(np.abs(c) > 1e-12) else 0
    mk = lambda k: max(abs(c[k]), abs(c[k + 1]))  # even/odd functions have every other coefficient zero
    print(f"   {name}: max(|c_k|,|c_k+1|) at k=2,10,50,200: {mk(2):.1e} {mk(10):.1e} {mk(50):.1e} {mk(200):.1e}   last k with |c_k|>1e-12 max: {nneed}")
cr = cheb_coeffs(lambda t: 1 / (1 + 25 * t**2), 256)
print("   runge rate: ratio |c_{k+2}/c_k| ~", abs(cr[42] / cr[40]), " predicted rho^-2, rho = 0.2+sqrt(1.04) =", (0.2 + np.sqrt(1.04))**-2)

print("C4 Pade vs Taylor for e^x on [-1,1]")
from math import factorial
coef = [1 / factorial(k) for k in range(9)]
for (n, m) in ((2, 2), (3, 3), (4, 4)):
    p, q = ap.pade(coef, n, m)
    r = np.polyval(p[::-1], tt) / np.polyval(q[::-1], tt)
    tay = np.polyval(coef[:n + m + 1][::-1], tt)
    print(f"   [{n}/{m}]: max err Pade {np.abs(r - np.exp(tt)).max():.2e}  Taylor deg {n+m} {np.abs(tay - np.exp(tt)).max():.2e}  at x=-1 Pade {abs(r[0]-np.exp(-1)):.1e} at x=1 Pade {abs(r[-1]-np.e):.1e}")

# ---------- D ----------
print("D1 polynomial degree selection")
rng = np.random.default_rng(4)
n = 60; xd = np.sort(rng.uniform(0, 1, n)); ftrue = lambda t: np.exp(t) * np.sin(3 * np.pi * t); yd = ftrue(xd) + 0.3 * rng.standard_normal(n)
folds = np.array_split(rng.permutation(n), 5)
rows = []
for deg in range(0, 13):
    C = np.polynomial.chebyshev.chebvander(2 * xd - 1, deg)
    beta = np.linalg.lstsq(C, yd, rcond=None)[0]; rss = np.sum((yd - C @ beta)**2); k = deg + 1
    aic = n * np.log(rss / n) + 2 * k; bic = n * np.log(rss / n) + k * np.log(n)
    cv = 0
    for f_ in folds:
        tr = np.setdiff1d(np.arange(n), f_)
        bt = np.linalg.lstsq(C[tr], yd[tr], rcond=None)[0]; cv += np.sum((yd[f_] - C[f_] @ bt)**2)
    xx = np.linspace(0, 1, 2001); true_err = np.mean((np.polynomial.chebyshev.chebvander(2 * xx - 1, deg) @ beta - ftrue(xx))**2)
    rows.append((deg, rss / n, cv / n, aic, bic, true_err))
    print(f"   deg {deg:2d}: train MSE {rss / n:.4f} CV MSE {cv / n:.4f} AIC {aic:8.2f} BIC {bic:8.2f} true MSE {true_err:.4f}")
R = np.array(rows)
print("   argmin CV", int(R[np.argmin(R[:, 2]), 0]), "AIC", int(R[np.argmin(R[:, 3]), 0]), "BIC", int(R[np.argmin(R[:, 4]), 0]), "true", int(R[np.argmin(R[:, 5]), 0]))

print("D2 Mauna Loa CO2: periodogram + Fourier regression")
import statsmodels.api as sm
co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
yv = co2.values; tv = np.arange(len(yv)) / 12.0
print("   months", len(yv), co2.index[0].date(), co2.index[-1].date())
detr = yv - np.polyval(np.polyfit(tv, yv, 2), tv)
F = np.fft.rfft(detr - detr.mean()); freqs = np.fft.rfftfreq(len(detr), d=1 / 12)
top = np.argsort(-np.abs(F))[:3]
print("   top periodogram frequencies (cycles/yr):", np.round(freqs[top], 3), "periods (months)", np.round(12 / freqs[top], 2))
ntest = 60
for K in (0, 1, 2, 3, 4):
    cols = [np.ones_like(tv), tv, tv**2] + [f(2 * np.pi * k * tv) for k in range(1, K + 1) for f in (np.cos, np.sin)]
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X[:-ntest], yv[:-ntest], rcond=None)[0]
    rmse_tr = np.sqrt(np.mean((X[:-ntest] @ beta - yv[:-ntest])**2)); rmse_te = np.sqrt(np.mean((X[-ntest:] @ beta - yv[-ntest:])**2))
    extra = ""
    if K >= 1:
        extra = f" annual amplitude {np.hypot(beta[3], beta[4]):.3f} ppm"
    print(f"   K={K}: train RMSE {rmse_tr:.3f}  test RMSE (last 5 yrs) {rmse_te:.3f}{extra}")

print("D3 image compression with the 2-D DCT")
from sklearn.datasets import load_sample_image
img = load_sample_image("china.jpg").astype(float).mean(axis=2)
Cimg = dctn(img, norm="ortho")
flat = np.sort(np.abs(Cimg).ravel())[::-1]
energy = np.cumsum(flat**2) / np.sum(flat**2)
for frac in (0.01, 0.05, 0.10, 0.25):
    k = int(frac * Cimg.size); thr = flat[k - 1]
    rec = idctn(np.where(np.abs(Cimg) >= thr, Cimg, 0), norm="ortho")
    mse = np.mean((rec - img)**2); psnr = 10 * np.log10(255**2 / mse)
    print(f"   keep {frac:.0%}: energy kept {energy[k - 1]:.5f}  PSNR {psnr:.2f} dB")
print("   Parseval check:", np.sum(img**2), np.sum(Cimg**2))
