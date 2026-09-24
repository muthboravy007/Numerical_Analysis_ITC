"""Reproduces every numerical example in Chapter 3 (Interpolation and Polynomial Approximation).
Run from the repository root:  python lectures/code/ch03.py"""

import math
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import interpolation as I  # noqa: E402

np.set_printoptions(precision=7, suppress=True, linewidth=120)

print("=== 3.1 Lagrange ===")
for n in range(0, 8):
    print(f"Ex 3.1.1 Taylor of 1/x about 1, n={n}: P_n(3) = {sum((-2) ** k for k in range(n + 1))}")
x = np.array([2.0, 5.0]); y = np.array([4.0, 1.0])
print("Ex 3.1.2 linear through (2,4),(5,1): P(x) = -x + 6, P(3) =", I.lagrange_eval(x, y, 3.0)[0])
x = np.array([2.0, 2.75, 4.0]); y = 1 / x
P3 = I.lagrange_eval(x, y, 3.0)[0]
print("Ex 3.1.3 1/x at 2,2.75,4: P(3) =", P3, "1/3 =", 1 / 3, "err", abs(P3 - 1 / 3))
c = np.polyfit(x, y, 2); print("   P coefficients (high->low):", c)
t = np.linspace(2, 4, 200001)
w = np.abs((t - 2) * (t - 2.75) * (t - 4))
print("Ex 3.1.4 max |(x-2)(x-2.75)(x-4)| on [2,4]:", w.max(), "at", t[w.argmax()], "| bound:", 6 / 2 ** 4 / 6 * w.max(),
      "| actual max err:", np.max(np.abs(I.lagrange_eval(x, y, t) - 1 / t)))
h = math.sqrt(8e-6 / math.e)
print("Ex 3.1.5 h for e^x linear interp error <= 1e-6:", h)
x = np.array([1.0, 2.0, 4.0])
print("Ex 3.1.6 ln3 from 1,2,4:", I.lagrange_eval(x, np.log(x), 3.0)[0], "true", math.log(3))
x = np.array([2.0, 2.5, 4.0])
print("   ln3 from 2,2.5,4:", I.lagrange_eval(x, np.log(x), 3.0)[0])

print("\n=== 3.2 Neville ===")
xs = np.array([2.0, 2.2, 2.4, 2.6, 2.8])
fs = np.round(np.log(xs), 7)
v, Q = I.neville(xs, fs, 2.5)
print("Ex 3.2.1 ln table:", fs); print(Q); print("   ln 2.5 =", math.log(2.5))
order = [2, 3, 1, 4, 0]
v2, Q2 = I.neville(xs[order], fs[order], 2.5)
print("Ex 3.2.2 nodes ordered by closeness:", xs[order]); print(Q2)
# Ex 3.2.3 missing value: P_{0,1,2}(1.5) with x=0,1,2 f0=1,f1=2, find f2 s.t. P(1.5)=3
# P(1.5) = f0 L0 + f1 L1 + f2 L2
x = np.array([0.0, 1.0, 2.0]); tt = 1.5
L = [np.prod([(tt - x[j]) / (x[k] - x[j]) for j in range(3) if j != k]) for k in range(3)]
f2 = (3 - 1 * L[0] - 2 * L[1]) / L[2]
print("Ex 3.2.3 Lagrange basis at 1.5:", L, "f2 =", f2, "check", I.lagrange_eval(x, [1, 2, f2], 1.5))
hrs = np.array([0.0, 3, 6, 9, 12]); temp = np.array([24.1, 22.8, 25.3, 30.2, 33.0])
v, Q = I.neville(hrs[[2, 3, 1, 4, 0]], temp[[2, 3, 1, 4, 0]], 7.5)
print("Ex 3.2.4 temperature at 7.5 h, Neville (nodes 6,9,3,12,0):"); print(Q)
xx = np.array([0.5, 0.55, 0.6, 0.65])
gg = xx - np.exp(-xx)
v, Q = I.neville(gg, xx, 0.0)
print("Ex 3.2.5 inverse interpolation: g values", gg); print(Q); print("   true root 0.5671432904")

print("\n=== 3.3 Divided differences ===")
print("Ex 3.3.1"); print(I.divided_difference_table([0, 1, 2, 3], [1, 2, 9, 28]))
xd = np.array([1.0, 2, 3, 4]); yd = np.array([2.0, 3, 5, 4])
print("Ex 3.3.2"); print(I.divided_difference_table(xd, yd)); print("   P(2.5)=", I.newton_eval(xd, I.divided_differences(xd, yd), 2.5))
xs = np.array([0.0, 0.2, 0.4, 0.6]); fs = np.sin(xs)
D = np.zeros((4, 4)); D[:, 0] = fs
for j in range(1, 4):
    D[: 4 - j, j] = D[1: 5 - j, j - 1] - D[: 4 - j, j - 1]
print("Ex 3.3.3 forward differences of sin:"); print(D)
s = (0.1 - 0) / 0.2
Pf = fs[0] + s * D[0, 1] + s * (s - 1) / 2 * D[0, 2] + s * (s - 1) * (s - 2) / 6 * D[0, 3]
print("   forward formula sin(0.1)=", Pf, "true", math.sin(0.1), "err", abs(Pf - math.sin(0.1)))
s = (0.55 - 0.6) / 0.2
Pb = fs[3] + s * D[2, 1] + s * (s + 1) / 2 * D[1, 2] + s * (s + 1) * (s + 2) / 6 * D[0, 3]
print("Ex 3.3.4 backward formula sin(0.55)=", Pb, "true", math.sin(0.55), "err", abs(Pb - math.sin(0.55)))
xa = np.array([1.0, 2, 3, 4, 5]); ya = np.array([2.0, 3, 5, 4, 6])
ca = I.divided_differences(xa, ya)
print("Ex 3.3.5 add point (5,6): coefficients", ca, "new term coef", ca[-1], "P4(2.5)=", I.newton_eval(xa, ca, 2.5))
xe = np.array([0.0, 0.1, 0.2, 0.3])
ce = I.divided_differences(xe, np.exp(xe))
print("Ex 3.3.6 f[x0..x3] for e^x on 0..0.3:", ce[-1], "e^xi/6 range", 1 / 6, math.exp(0.3) / 6, "xi =", math.log(6 * ce[-1]))

print("\n=== 3.4 Hermite ===")
z, c = I.hermite_coefficients([0.0, 1.0], [1.0, 2.0], [0.0, 3.0])
print("Ex 3.4.1 coefficients", c)
z, c = I.hermite_coefficients([0.0, math.pi / 2], [0.0, 1.0], [1.0, 0.0])
H = I.newton_eval(z, c, math.pi / 4)
print("Ex 3.4.2 sin Hermite coef", c, "H(pi/4)=", H, "true", math.sin(math.pi / 4), "err", abs(H - math.sin(math.pi / 4)))
tt = np.linspace(0, math.pi / 2, 100001)
print("   bound max|x^2(x-pi/2)^2|/24 =", np.max(tt ** 2 * (tt - math.pi / 2) ** 2) / 24, "= (pi/4)^4/24 =", (math.pi / 4) ** 4 / 24,
      "actual max err", np.max(np.abs(I.newton_eval(z, c, tt) - np.sin(tt))))
xs = np.array([1.0, 1.5, 2.0])
z, c = I.hermite_coefficients(xs, np.log(xs), 1 / xs)
H = I.newton_eval(z, c, 1.25)
Q = np.zeros((6, 6)); zz = np.repeat(xs, 2)
for i in range(3):
    Q[2 * i, 0] = Q[2 * i + 1, 0] = math.log(xs[i]); Q[2 * i + 1, 1] = 1 / xs[i]
    if i: Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / 0.5
for j in range(2, 6):
    for i in range(j, 6):
        Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (zz[i] - zz[i - j])
print("Ex 3.4.3 table:"); print(Q); print("   H5(1.25)=", H, "ln1.25=", math.log(1.25), "err", abs(H - math.log(1.25)))
print("   Lagrange 3-pt at 1.25:", I.lagrange_eval(xs, np.log(xs), 1.25)[0])
tt = np.linspace(1, 2, 100001)
print("Ex 3.4.4 bound for E3 at 1.25: f^(6)=-120/x^6 -> 120/720*|w|:", 120 / 720 * abs((0.25 * -0.25 * -0.75) ** 2))
tk = np.array([0.0, 5, 10]); dk = np.array([0.0, 90, 220]); vk = np.array([15.0, 20, 28])
z, c = I.hermite_coefficients(tk, dk, vk)
pos = I.newton_eval(z, c, 8.0)
eps = 1e-6
spd = (I.newton_eval(z, c, 8.0 + eps) - I.newton_eval(z, c, 8.0 - eps)) / (2 * eps)
print("Ex 3.4.5 car: position at t=8:", pos, "speed", spd)

print("\n=== 3.5 Cubic splines ===")
x = np.array([0.0, 1, 2, 3]); y = np.array([0.0, 1, 0, 1])
M = I.natural_cubic_spline(x, y); a, b, cc, d = I.spline_coefficients(x, y, M)
print("Ex 3.5.1 M:", M, "\n   a", a, "\n   b", b, "\n   c", cc, "\n   d", d, "\n   S(0.5), S(1.5):", I.cubic_spline_eval(x, y, M, [0.5, 1.5]))
xc = np.linspace(0, np.pi, 5); yc = np.cos(xc)
Mn = I.natural_cubic_spline(xc, yc); Mc = I.clamped_cubic_spline(xc, yc, 0.0, 0.0)
tt = np.linspace(0, np.pi, 10001)
en = np.max(np.abs(I.cubic_spline_eval(xc, yc, Mn, tt) - np.cos(tt)))
ec = np.max(np.abs(I.cubic_spline_eval(xc, yc, Mc, tt) - np.cos(tt)))
print("Ex 3.5.2 cos on [0,pi], 5 nodes: M natural", Mn, "M clamped", Mc, "maxerr nat", en, "clamped", ec)
for name, Mx in (("nat", Mn), ("clamped", Mc)):
    a, b, cc, d = I.spline_coefficients(xc, yc, Mx)
    h = np.diff(xc)
    integ = np.sum(a * h + b * h ** 2 / 2 + cc * h ** 3 / 3 + d * h ** 4 / 4)
    ai, bi, ci, di = I.spline_coefficients(xc, np.sin(xc), I.natural_cubic_spline(xc, np.sin(xc)) if name == "nat" else I.clamped_cubic_spline(xc, np.sin(xc), 1.0, -1.0))
    print(f"   {name}: spline(sin) integral {np.sum(ai*h+bi*h**2/2+ci*h**3/3+di*h**4/4):.8f} vs 2")
r = lambda x: 1 / (1 + x ** 2)
xr = np.linspace(-5, 5, 11); tt = np.linspace(-5, 5, 10001)
Mn = I.natural_cubic_spline(xr, r(xr)); Mc = I.clamped_cubic_spline(xr, r(xr), 10 / 26 ** 2, -10 / 26 ** 2)
print("Ex 3.5.3 1/(1+x^2) 11 nodes: poly err", np.max(np.abs(I.barycentric_eval(xr, r(xr), tt) - r(tt))),
      "natural spline err", np.max(np.abs(I.cubic_spline_eval(xr, r(xr), Mn, tt) - r(tt))),
      "clamped", np.max(np.abs(I.cubic_spline_eval(xr, r(xr), Mc, tt) - r(tt))),
      "linear", np.max(np.abs(np.interp(tt, xr, r(xr)) - r(tt))))
for n in (5, 10, 20, 40):
    xs = np.linspace(0, np.pi, n + 1)
    Mc = I.clamped_cubic_spline(xs, np.sin(xs), 1.0, -1.0)
    tq = np.linspace(0, np.pi, 20001)
    print(f"Ex 3.5.4 clamped spline sin, n={n}: max err {np.max(np.abs(I.cubic_spline_eval(xs, np.sin(xs), Mc, tq) - np.sin(tq))):.3e}")
days = np.arange(0, 15.0)
true = 20 + 5 * np.sin(2 * np.pi * days / 14) + 0.5 * days
miss = np.array([4, 5, 6, 10])
keep = np.setdiff1d(np.arange(15), miss)
lin = np.interp(miss, days[keep], true[keep])
cs = CubicSpline(days[keep], true[keep], bc_type="natural")(miss)
print("Ex 3.5.5 gap fill true", np.round(true[miss], 3), "linear", np.round(lin, 3), "spline", np.round(cs, 3),
      "RMSE lin", np.sqrt(np.mean((lin - true[miss]) ** 2)), "spline", np.sqrt(np.mean((cs - true[miss]) ** 2)))

print("\n=== 3.6 Parametric curves ===")
P = np.array([[0.0, 0], [1, 2], [3, 3], [4, 0]])
def bez(P, t):
    t = np.atleast_1d(t)[:, None]
    return ((1 - t) ** 3 * P[0] + 3 * t * (1 - t) ** 2 * P[1] + 3 * t ** 2 * (1 - t) * P[2] + t ** 3 * P[3])
print("Ex 3.6.1 Bezier(0.5) =", bez(P, 0.5), "B(0.25)=", bez(P, 0.25))
Q1 = (P[:-1] + P[1:]) / 2; Q2 = (Q1[:-1] + Q1[1:]) / 2; Q3 = (Q2[:-1] + Q2[1:]) / 2
print("Ex 3.6.2 de Casteljau levels", Q1, Q2, Q3)
print("Ex 3.6.3 x(t) coefficients", np.polyfit(np.linspace(0, 1, 50), bez(P, np.linspace(0, 1, 50))[:, 0], 3),
      "y(t)", np.polyfit(np.linspace(0, 1, 50), bez(P, np.linspace(0, 1, 50))[:, 1], 3))
th = np.linspace(0, 2 * np.pi, 5)
cx = CubicSpline(th, np.cos(th), bc_type="periodic"); cy = CubicSpline(th, np.sin(th), bc_type="periodic")
tt = np.linspace(0, 2 * np.pi, 2001)
print("Ex 3.6.4 periodic spline circle 5 pts: max radius err", np.max(np.abs(np.hypot(cx(tt), cy(tt)) - 1)))
pts = np.array([[0, 0], [1, 1.5], [2.5, 1.8], [3, 0.5], [4.2, 0.9]])
d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
print("Ex 3.6.5 chord-length parameters", d)
sx = CubicSpline(d, pts[:, 0], bc_type="natural"); sy = CubicSpline(d, pts[:, 1], bc_type="natural")
tt = np.linspace(0, d[-1], 20001)
L = np.sum(np.hypot(np.diff(sx(tt)), np.diff(sy(tt))))
print("   arc length of spline path", L, "polyline length", d[-1])

print("\n=== 3.7 DS extension ===")
for n in (5, 10, 15, 20):
    t = np.linspace(-1, 1, 2001)
    xe = np.linspace(-1, 1, n + 1); xc = I.chebyshev_nodes(n + 1)
    print(f"Ex 3.7.1 n={n}: equispaced {np.max(abs(I.barycentric_eval(xe, I.runge(xe), t) - I.runge(t))):.3g}  chebyshev {np.max(abs(I.barycentric_eval(xc, I.runge(xc), t) - I.runge(t))):.3g}")
t = np.linspace(-1, 1, 20001)
for n in (4, 8, 16):
    xe = np.linspace(-1, 1, n + 1); xc = I.chebyshev_nodes(n + 1)
    print(f"Ex 3.7.2 n={n}: max|w| equi {np.max(np.abs(np.prod(t[:, None] - xe, axis=1))):.3e}  cheb {np.max(np.abs(np.prod(t[:, None] - xc, axis=1))):.3e}  2^-n {2.0 ** -n:.3e}")
xb = np.array([0.0, 1, 3]); yb = np.array([1.0, 3, 2])
wb = I.barycentric_weights(xb)
print("Ex 3.7.3 barycentric weights", wb, "p(2)=", I.barycentric_eval(xb, yb, 2.0))
for n in (5, 10, 20):
    xe = np.linspace(-1, 1, n + 1); xc = I.chebyshev_nodes(n + 1)
    def leb(xn):
        return np.max(np.sum(np.abs(np.array([I.lagrange_eval(xn, np.eye(n + 1)[k], t) for k in range(n + 1)])), axis=0))
    print(f"Ex 3.7.4 Lebesgue n={n}: equi {leb(xe):.3g} cheb {leb(xc):.3g}")
rng = np.random.default_rng(1)
xn = np.linspace(0, 1, 11); yn = np.sin(2 * np.pi * xn) + 0.1 * rng.standard_normal(11)
t = np.linspace(0, 1, 2001)
pint = I.barycentric_eval(xn, yn, t)
cfit = np.polyfit(xn, yn, 3)
print("Ex 3.7.5 noisy data: interp maxdev from truth", np.max(np.abs(pint - np.sin(2 * np.pi * t))),
      "cubic LS maxdev", np.max(np.abs(np.polyval(cfit, t) - np.sin(2 * np.pi * t))))
f00, f10, f01, f11 = 10.0, 14.0, 12.0, 20.0
xq, yq = 0.3, 0.6
print("Ex 3.7.6 bilinear:", f00 * (1 - xq) * (1 - yq) + f10 * xq * (1 - yq) + f01 * (1 - xq) * yq + f11 * xq * yq)
