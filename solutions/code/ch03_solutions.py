"""Numerical answers for exercises/ch03_exercises.md."""
import math
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import interpolation as I  # noqa: E402

x = np.array([0.0, 1, 3]); y = np.array([1.0, 3, 2])
print("B1 P(2) =", I.lagrange_eval(x, y, 2.0), "coeffs", np.polyfit(x, y, 2))
xs = np.array([1.0, 4, 9, 16]); v, Q = I.neville(xs[[1, 2, 0, 3]], np.sqrt(xs[[1, 2, 0, 3]]), 5.0)
print("B2 Neville sqrt at 5 (nodes 4,9,1,16):\n", Q, "\n true", math.sqrt(5))
xd = np.array([-1.0, 0, 1, 2]); yd = xd ** 3 - xd ** 2 + 1
print("B3 f values", yd, "\n", I.divided_difference_table(xd, yd))
c = I.divided_differences(np.r_[xd, 3.0], np.r_[yd, 20.0]); print("   add (3,20): coef", c)
xl = np.array([1.0, 1.1, 1.2, 1.3]); fl = np.round(np.log(xl), 6)
D = np.zeros((4, 4)); D[:, 0] = fl
for j in range(1, 4):
    D[: 4 - j, j] = D[1: 5 - j, j - 1] - D[: 4 - j, j - 1]
s = 0.5
P = fl[0] + s * D[0, 1] + s * (s - 1) / 2 * D[0, 2] + s * (s - 1) * (s - 2) / 6 * D[0, 3]
print("B4 table", fl, "diffs", D[0], "P(1.05)", P, "ln1.05", math.log(1.05), "err", abs(P - math.log(1.05)))
z, cc = I.hermite_coefficients([1.0, 2.0], [1.0, 0.5], [-1.0, -0.25]); H = I.newton_eval(z, cc, 1.5)
print("B5 Hermite coef", cc, "H(1.5)", H, "1/1.5", 1 / 1.5, "err", abs(H - 2 / 3), "bound", 24 / 24 * (0.25 * 0.25), "(f4=24/x^5, max 24 on [1,2])")
xn = np.array([0.0, 1, 2]); yn = np.array([1.0, 2, 0]); M = I.natural_cubic_spline(xn, yn)
print("B6 natural M", M, "coeffs", I.spline_coefficients(xn, yn, M), "S(0.5),S(1.5)", I.cubic_spline_eval(xn, yn, M, [0.5, 1.5]))
Mc = I.clamped_cubic_spline(xn, yn, 0.0, 0.0)
print("B7 clamped M", Mc, "coeffs", I.spline_coefficients(xn, yn, Mc), "S(0.5),S(1.5)", I.cubic_spline_eval(xn, yn, Mc, [0.5, 1.5]))
Pc = np.array([[0.0, 0], [1, 3], [4, 3], [5, 0]]); t = 1 / 3
L1 = (1 - t) * Pc[:-1] + t * Pc[1:]; L2 = (1 - t) * L1[:-1] + t * L1[1:]; L3 = (1 - t) * L2[:-1] + t * L2[1:]
print("B8 de Casteljau", L1, L2, L3)
tt = np.linspace(-1, 1, 4001)
for n in (5, 10, 20, 40):
    xe = np.linspace(-1, 1, n + 1); xc = I.chebyshev_nodes(n + 1)
    print(f"C1 n={n}: equi {np.max(np.abs(I.barycentric_eval(xe, I.runge(xe), tt) - I.runge(tt))):.3e} cheb {np.max(np.abs(I.barycentric_eval(xc, I.runge(xc), tt) - I.runge(tt))):.3e}")
def leb(xn):
    n = len(xn); return np.max(np.sum(np.abs(np.array([I.lagrange_eval(xn, np.eye(n)[k], tt) for k in range(n)])), axis=0))
for n in (4, 8, 12, 16):
    print(f"C2 n={n}: Lebesgue equi {leb(np.linspace(-1, 1, n + 1)):.3f} cheb {leb(I.chebyshev_nodes(n + 1)):.3f} bound (2/pi)ln(n+1)+1 = {2 / math.pi * math.log(n + 1) + 1:.3f}")
for n in (8, 16, 32, 64):
    xs = np.linspace(0, 2 * np.pi, n + 1); Mx = I.clamped_cubic_spline(xs, np.cos(xs), 0.0, 0.0)
    tq = np.linspace(0, 2 * np.pi, 5001); e = np.max(np.abs(I.cubic_spline_eval(xs, np.cos(xs), Mx, tq) - np.cos(tq)))
    print(f"C3 clamped spline cos n={n}: err {e:.3e}; scipy clamped diff {np.max(np.abs(CubicSpline(xs, np.cos(xs), bc_type='clamped')(tq) - I.cubic_spline_eval(xs, np.cos(xs), Mx, tq))):.1e}")
rng = np.random.default_rng(0)
tday = np.arange(60.0); true = 50 + 10 * np.sin(2 * np.pi * tday / 30) + 0.2 * tday
miss = np.sort(rng.choice(np.arange(1, 59), 15, replace=False)); keep = np.setdiff1d(np.arange(60), miss)
for name, fn in (("linear", lambda: np.interp(miss, tday[keep], true[keep])), ("cubic spline", lambda: CubicSpline(tday[keep], true[keep])(miss)), ("PCHIP", lambda: PchipInterpolator(tday[keep], true[keep])(miss))):
    print(f"C4 {name}: RMSE {np.sqrt(np.mean((fn() - true[miss]) ** 2)):.4f}")
ti = np.array([0, 1.5, 2.0, 5.0, 5.5, 9.0, 10.0]); cum = np.array([0, 5, 30, 32, 60, 61, 90.0])
tg = np.linspace(0, 10, 1001)
cs = CubicSpline(ti, cum)(tg); pc = PchipInterpolator(ti, cum)(tg)
print("D1 cubic spline min slope", np.min(np.diff(cs)), "decreasing points", np.sum(np.diff(cs) < 0), "| PCHIP min slope", np.min(np.diff(pc)), "values at t=3", CubicSpline(ti, cum)(3.0), PchipInterpolator(ti, cum)(3.0))
img = np.array([[0, 0, 1, 1], [0, 1, 2, 1], [1, 2, 3, 2], [1, 1, 2, 2.0]])
from scipy.ndimage import zoom
bil = zoom(img, 2, order=1); nn = zoom(img, 2, order=0)
print("D2 bilinear 8x8 row 2:", np.round(bil[2], 3), "\n   nearest row 2:", nn[2])
acc = lambda lg: 0.9 - 0.02 * (lg + 2) ** 2 + 0.01 * np.sin(3 * lg)
nodes = I.chebyshev_nodes(9, -5, 1); vals = acc(nodes)
tq = np.linspace(-5, 1, 6001); surr = I.barycentric_eval(nodes, vals, tq)
print("D3 surrogate max at", tq[np.argmax(surr)], "value", surr.max(), "| true max at", tq[np.argmax(acc(tq))], acc(tq).max(), "| max surrogate error", np.max(np.abs(surr - acc(tq))))
