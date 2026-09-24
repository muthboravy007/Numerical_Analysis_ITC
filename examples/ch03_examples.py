"""Chapter 3 worked examples — run:  python examples/ch03_examples.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import interpolation as ip  # noqa: E402

np.set_printoptions(precision=6, suppress=True)

print("Example 3.1 — Lagrange interpolation of a yield curve")
T = np.array([1.0, 2, 5]); Y = np.array([3.10, 3.50, 4.00])
t = 3.0
L = [np.prod([(t - T[j]) / (T[i] - T[j]) for j in range(3) if j != i]) for i in range(3)]
print("  basis values L_i(3):", np.round(L, 6), " sum", sum(L), " P(3) =", np.dot(L, Y), " lagrange_eval:", ip.lagrange_eval(T, Y, np.array([t])))
print("  linear interpolation between 2y and 5y:", np.interp(3, T, Y))

print("Example 3.2 — inverse interpolation with Neville")
xs = np.array([0.70, 0.72, 0.74, 0.76]); gs = np.cos(xs) - xs
val, Q = ip.neville(gs, xs, 0.0)
print("  g values", gs.round(6), "\n  Neville table\n", Q, "\n  root estimate", val, " true", 0.7390851332151607, " error", abs(val - 0.7390851332151607))

print("Example 3.3 — US population (millions), Newton form, interpolation vs extrapolation")
yr = np.array([1950.0, 1960, 1970, 1980, 1990, 2000]); pop = np.array([151.3, 179.3, 203.3, 226.5, 248.7, 281.4])
s = (yr - 1950) / 10
coef = ip.divided_differences(s, pop)
print("  divided differences (in decades):", np.round(coef, 5))
for year in (1965, 1995, 2010, 2020):
    sv = (year - 1950) / 10
    p5 = ip.newton_eval(s, coef, np.array([sv]))[0]
    lin = np.polyval(np.polyfit(s, pop, 1), sv)
    print(f"  {year}: degree-5 interpolant {p5:.1f}   least-squares line {lin:.1f}")
print("  actual census: 2010 = 308.7, 2020 = 331.4")

print("Example 3.4 — cubic Hermite for a trajectory")
z, c = ip.hermite_coefficients(np.array([0.0, 1.0]), np.array([0.0, 10.0]), np.array([2.0, 8.0]))
Hc = lambda t: c[0] + c[1] * (t - z[0]) + c[2] * (t - z[0]) * (t - z[1]) + c[3] * (t - z[0]) * (t - z[1]) * (t - z[2])
print("  z", z, " coefficients", c, " H(0.5) =", Hc(0.5), " as polynomial", np.polyfit(np.linspace(0, 1, 5), Hc(np.linspace(0, 1, 5)), 3).round(6))

print("Example 3.5 — spline vs polynomial on dose-response data")
dose = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8]); resp = np.array([0.00, 0.01, 0.02, 0.10, 0.90, 0.98, 0.99, 1.00, 1.00])
M = ip.natural_cubic_spline(dose, resp)
tt = np.linspace(0, 8, 801)
sp = ip.cubic_spline_eval(dose, resp, M, tt)
poly = np.polyval(np.polyfit(dose, resp, 8), tt)
print("  spline second derivatives M:", M.round(5))
print(f"  on [0,8]: spline range [{sp.min():.4f}, {sp.max():.4f}]  degree-8 polynomial range [{poly.min():.4f}, {poly.max():.4f}]")
print(f"  at dose 7.5: spline {ip.cubic_spline_eval(dose, resp, M, np.array([7.5]))[0]:.4f}   polynomial {np.polyval(np.polyfit(dose, resp, 8), 7.5):.4f}")
Mc = ip.clamped_cubic_spline(dose, resp, 0.0, 0.0)
from scipy.interpolate import PchipInterpolator
pc = PchipInterpolator(dose, resp)(tt)
print(f"  monotone PCHIP range [{pc.min():.4f}, {pc.max():.4f}]; spline non-monotone? {bool(np.any(np.diff(sp) < -1e-12))}  PCHIP non-monotone? {bool(np.any(np.diff(pc) < -1e-12))}")
print(f"  clamped (slopes 0 at ends) at 0.5: {ip.cubic_spline_eval(dose, resp, Mc, np.array([0.5]))[0]:.4f}  natural: {ip.cubic_spline_eval(dose, resp, M, np.array([0.5]))[0]:.4f}")

print("Example 3.6 — Chebyshev vs equispaced nodes for a steep sigmoid")
f = lambda x: 1 / (1 + np.exp(-10 * x))
xx = np.linspace(-1, 1, 20001)
for n in (6, 10, 16, 24):
    xe = np.linspace(-1, 1, n + 1); xc = ip.chebyshev_nodes(n + 1)
    ee = np.abs(ip.barycentric_eval(xe, f(xe), xx) - f(xx)).max(); ec = np.abs(ip.barycentric_eval(xc, f(xc), xx) - f(xx)).max()
    print(f"  degree {n:2d}: equispaced max error {ee:.2e}   Chebyshev {ec:.2e}")
