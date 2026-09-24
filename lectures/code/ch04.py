"""Reproduces every numerical example in Chapter 4 (Numerical Differentiation and Integration).
Run from the repository root:  python lectures/code/ch04.py"""

import math
import os
import sys

import numpy as np
from scipy import integrate as si
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import integration as Q  # noqa: E402

np.set_printoptions(precision=9, suppress=True, linewidth=120)

print("=== 4.1 Numerical differentiation ===")
f = math.log
for h in (0.1, 0.05, 0.01):
    fd = (f(2 + h) - f(2)) / h
    print(f"Ex 4.1.1 ln at 2, h={h}: forward {fd:.7f} err {abs(fd - 0.5):.2e} bound h/(2*4)={h / 8:.2e}")
fe = math.exp
x0, h = 1.0, 0.1
e3 = Q.three_point_endpoint(fe, x0, h); e3m = Q.central_diff(fe, x0, h); e5 = Q.five_point_midpoint(fe, x0, h)
print("Ex 4.1.2 e^x at 1, h=0.1: 2pt fwd", (fe(1.1) - fe(1)) / 0.1, "3pt endpoint", e3, "3pt mid", e3m, "5pt", e5, "true", math.e)
print("   errors:", abs((fe(1.1) - fe(1)) / 0.1 - math.e), abs(e3 - math.e), abs(e3m - math.e), abs(e5 - math.e))
xs = np.array([1.8, 1.9, 2.0, 2.1, 2.2]); ys = np.round(xs * np.exp(xs), 6)
print("Ex 4.1.3 table x e^x:", ys)
print("   3pt mid f'(2):", (ys[3] - ys[1]) / 0.2, "3pt end f'(2) h=0.1:", (-3 * ys[2] + 4 * ys[3] - ys[4]) / 0.2,
      "5pt f'(2):", (ys[0] - 8 * ys[1] + 8 * ys[3] - ys[4]) / 1.2, "true", 3 * math.exp(2))
print("   f''(2) 3pt:", (ys[1] - 2 * ys[2] + ys[3]) / 0.01, "true", 4 * math.exp(2))
for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    fd2 = (math.sin(1 + h) - 2 * math.sin(1) + math.sin(1 - h)) / h ** 2
    print(f"Ex 4.1.4 sin''(1) h={h:.0e}: {fd2:.10f} err {abs(fd2 + math.sin(1)):.2e}")
print("   optimal h ~ (48 eps)^(1/4) =", (48 * 2.2e-16 * math.sin(1) / math.sin(1)) ** 0.25)
cum = np.array([0, 12, 30, 51, 74, 98, 120, 138, 151])  # cumulative cases, days 0..8
rate = np.gradient(cum.astype(float))
print("Ex 4.1.5 daily rate np.gradient:", rate)
d = 7
print("   central rate day 4:", (cum[5] - cum[3]) / 2, "second diff (acceleration) day 4:", cum[5] - 2 * cum[4] + cum[3])

print("\n=== 4.2 Richardson extrapolation ===")
N1 = lambda h: (math.exp(1 + h) - math.exp(1 - h)) / (2 * h)
v, T = Q.richardson(N1, 0.2, p=2, levels=4)
print("Ex 4.2.1 central-diff Richardson table for e'(1):"); print(T); print("   errors diag", [abs(T[i, i] - math.e) for i in range(4)])
Nf = lambda h: (math.exp(1 + h) - math.exp(1)) / h
T = np.zeros((4, 4))
for i in range(4):
    T[i, 0] = Nf(0.2 / 2 ** i)
    for j in range(1, i + 1):
        T[i, j] = T[i, j - 1] + (T[i, j - 1] - T[i - 1, j - 1]) / (2 ** j - 1)
print("Ex 4.2.2 forward-diff Richardson (all powers of h):"); print(T); print("   err", abs(T[3, 3] - math.e))
Npi = lambda n: n * math.sin(math.pi / n)
a1, a2, a3 = Npi(6), Npi(12), Npi(24)
r1 = (4 * a2 - a1) / 3; r2 = (4 * a3 - a2) / 3; r3 = (16 * r2 - r1) / 15
print("Ex 4.2.3 Archimedes:", a1, a2, a3, "| R1:", r1, r2, "| R2:", r3, "pi", math.pi)
F = lambda h: (1 + h) ** (1 / h)
b1, b2, b3 = F(0.04), F(0.02), F(0.01)
c1 = 2 * b2 - b1; c2 = 2 * b3 - b2; c3 = (4 * c2 - c1) / 3
print("Ex 4.2.4 (1+h)^(1/h):", b1, b2, b3, "| R1", c1, c2, "| R2", c3, "e", math.e)
T8 = Q.trapezoid(np.sin, 0, np.pi, 8); T16 = Q.trapezoid(np.sin, 0, np.pi, 16)
print("Ex 4.2.5 T8 T16 extrap:", T8, T16, (4 * T16 - T8) / 3, "S16 =", Q.simpson(np.sin, 0, np.pi, 16))

print("\n=== 4.3 Elements of numerical integration ===")
g = lambda x: np.exp(-x ** 2)
I1 = math.sqrt(math.pi) / 2 * erf(1)
print("Ex 4.3.1 int_0^1 e^{-x^2}:", I1, "trap", 0.5 * (g(0) + g(1)), "simpson", 1 / 6 * (g(0) + 4 * g(0.5) + g(1)),
      "midpoint", g(0.5), "3/8", Q.simpson38(g, 0, 1))
for k in range(6):
    p = lambda x, k=k: x ** k
    ex = 2 ** (k + 1) / (k + 1)
    print(f"Ex 4.3.2 x^{k} on [0,2]: exact {ex:.4f} trap {Q.trapezoid(p, 0, 2, 1):.4f} simp {Q.simpson(p, 0, 2, 2):.4f} 3/8 {Q.simpson38(p, 0, 2):.4f}")
print("Ex 4.3.3 Simpson error bound e^{-x^2} on [0,1]: h=0.5, max|f4|=12 ->", 0.5 ** 5 / 90 * 12, "actual", abs(1 / 6 * (g(0) + 4 * g(0.5) + g(1)) - I1))
f1 = lambda x: 1 / (1 + x)
a, b = 0.0, 2.0
hh = (b - a) / 2; m0 = 2 * hh * f1(a + hh)
hh = (b - a) / 3; m1 = 3 * hh / 2 * (f1(a + hh) + f1(a + 2 * hh))
hh = (b - a) / 4; m2 = 4 * hh / 3 * (2 * f1(a + hh) - f1(a + 2 * hh) + 2 * f1(a + 3 * hh))
hh = (b - a) / 5; m3 = 5 * hh / 24 * (11 * f1(a + hh) + f1(a + 2 * hh) + f1(a + 3 * hh) + 11 * f1(a + 4 * hh))
c1 = (b - a) / 2 * (f1(a) + f1(b)); c2 = (b - a) / 6 * (f1(a) + 4 * f1(1) + f1(b)); c3 = Q.simpson38(f1, a, b)
hh = (b - a) / 4; c4 = 2 * hh / 45 * (7 * f1(0) + 32 * f1(0.5) + 12 * f1(1) + 32 * f1(1.5) + 7 * f1(2))
print("Ex 4.3.4 int_0^2 1/(1+x) = ln3 =", math.log(3))
print("   closed n=1..4:", c1, c2, c3, c4)
print("   open   n=0..3:", m0, m1, m2, m3)
for k in range(4):
    print(f"Ex 4.3.5 midpoint on x^{k} over [0,2]: {2 * 1.0 ** k:.4f} exact {2 ** (k + 1) / (k + 1):.4f}")
w = np.linalg.solve(np.array([[1, 1, 1], [0, 1, 3], [0, 1, 9]], dtype=float), np.array([3, 4.5, 9]))
print("Ex 4.3.6 weights for nodes 0,1,3 on [0,3]:", w, "check x^3:", w @ np.array([0, 1, 27]), "exact", 81 / 4)

print("\n=== 4.4 Composite rules ===")
for n in (2, 4, 8, 16, 32):
    Tn = Q.trapezoid(g, 0, 1, n); Mn = Q.midpoint(g, 0, 1, n); Sn = Q.simpson(g, 0, 1, n)
    print(f"Ex 4.4.1 n={n:2d}: M {Mn:.9f} ({abs(Mn - I1):.2e})  T {Tn:.9f} ({abs(Tn - I1):.2e})  S {Sn:.9f} ({abs(Sn - I1):.2e})")
nT = math.ceil(math.sqrt(2 / 12 / 1e-6)) ; nS = math.ceil((12 / 180 / 1e-6) ** 0.25)
print("Ex 4.4.2 n for 1e-6 on e^{-x^2}: trap (max|f''|=2)", math.sqrt(2 / (12e-6)), "simpson (max|f4|=12)", (12 / (180e-6)) ** 0.25)
print("   check: T_n err at n=409", abs(Q.trapezoid(g, 0, 1, 409) - I1), "S_n at n=18", abs(Q.simpson(g, 0, 1, 18) - I1))
fpi = lambda x: 4 / (1 + x ** 2)
print("Ex 4.4.3 pi via 4/(1+x^2): T8", Q.trapezoid(fpi, 0, 1, 8), "S8", Q.simpson(fpi, 0, 1, 8), "pi", math.pi)
for n in (4, 8, 16):
    per = Q.trapezoid(lambda t: np.exp(np.cos(t)), 0, 2 * math.pi, n)
    print(f"Ex 4.4.4 periodic e^cos, n={n}: T={per:.15f} err {abs(per - 2 * math.pi * 7.95492652101284e-1 * 0 - 2*math.pi*1.2660658777520084):.2e}")
fpr = np.array([0.0, 0.0, 0.1, 0.1, 0.3, 0.3, 0.6, 1.0]); tpr = np.array([0.0, 0.4, 0.4, 0.7, 0.7, 0.9, 1.0, 1.0])
print("Ex 4.4.5 ROC AUC trapezoid:", Q.trapezoid_data(fpr, tpr))
tt = np.array([0, 10, 20, 30, 40, 50, 60.0]); vv = np.array([0, 12, 21, 27, 30, 31, 31.5])
print("Ex 4.4.6 distance from speed data: trap", Q.trapezoid_data(tt, vv), "simpson", si.simpson(vv, x=tt))

print("\n=== 4.5 Romberg ===")
v, R = Q.romberg(g, 0, 1, 5)
print("Ex 4.5.1 Romberg e^{-x^2}:"); print(R); print("   err R44", abs(R[4, 4] - I1))
v, R = Q.romberg(np.sin, 0, math.pi, 5)
print("Ex 4.5.2 Romberg sin on [0,pi]:"); print(R); print("   err", abs(R[4, 4] - 2))
v, R = Q.romberg(np.sqrt, 0, 1, 6)
print("Ex 4.5.3 Romberg sqrt on [0,1] diag errors:", [abs(R[i, i] - 2 / 3) for i in range(6)], "col0 errors", [abs(R[i, 0] - 2 / 3) for i in range(6)])
v, R = Q.romberg(lambda x: 1 / x, 1, 3, 5)
print("Ex 4.5.4 Romberg 1/x on [1,3] diag:", np.diag(R), "ln3", math.log(3))
print("Ex 4.5.5 function evaluations for R_nn with n=5 levels:", 2 ** 4 + 1)

print("\n=== 4.6 Adaptive quadrature ===")
fa = lambda x: 100 / x ** 2 * np.sin(10 / x)
exact = si.quad(fa, 1, 3, epsabs=1e-14, limit=200)[0]
val, ne = Q.adaptive_simpson(fa, 1, 3, tol=1e-4)
print("Ex 4.6.1 100/x^2 sin(10/x) on [1,3]: exact", exact, "adaptive", val, "evals", ne, "err", abs(val - exact))
for n in (8, 16, 32, 64, 128):
    print(f"   composite Simpson n={n}: err {abs(Q.simpson(fa, 1, 3, n) - exact):.2e}")
S01 = 1 / 6 * (g(0) + 4 * g(0.5) + g(1)); S0 = 0.25 / 3 * (g(0) + 4 * g(0.25) + g(0.5)); S1 = 0.25 / 3 * (g(0.5) + 4 * g(0.75) + g(1))
print("Ex 4.6.2 one adaptive step: S(0,1)", S01, "S(0,.5)+S(.5,1)", S0 + S1, "est err |diff|/15", abs(S01 - S0 - S1) / 15, "true err", abs(S0 + S1 - I1))
fs = lambda x: np.sqrt(np.abs(x - 0.3))
exact_s = (2 / 3) * (0.3 ** 1.5 + 0.7 ** 1.5)
val, ne = Q.adaptive_simpson(fs, 0, 1, tol=1e-8)
print("Ex 4.6.3 sqrt|x-0.3|: adaptive", val, "err", abs(val - exact_s), "evals", ne, "| composite S n=512 err", abs(Q.simpson(fs, 0, 1, 512) - exact_s))
val, ne = Q.adaptive_simpson(lambda x: np.exp(-50 * (x - 0.5) ** 2), 0, 1, tol=1e-10)
exg = math.sqrt(math.pi / 50) * erf(math.sqrt(50) * 0.5)
print("Ex 4.6.4 peak exp(-50(x-.5)^2):", val, "err", abs(val - exg), "evals", ne)
res = si.quad(fa, 1, 3, full_output=1)
print("Ex 4.6.5 scipy quad:", res[0], "abserr est", res[1], "evals", res[2]["neval"])

print("\n=== 4.7 Gaussian quadrature ===")
for n in (1, 2, 3, 4, 5):
    x, w = np.polynomial.legendre.leggauss(n)
    print(f"Ex 4.7.1 n={n}: nodes {x} weights {w}")
for n in (2, 3, 4, 5):
    val = Q.gauss_legendre(g, 0, 1, n)
    print(f"Ex 4.7.2 e^-x^2 GL n={n}: {val:.12f} err {abs(val - I1):.2e}")
f3 = lambda x: x ** 3 + 2 * x ** 2 - 1
x2, w2 = np.polynomial.legendre.leggauss(2)
print("Ex 4.7.3 2-pt Gauss on x^3+2x^2-1 over [-1,1]:", np.sum(w2 * f3(x2)), "exact", 4 / 3 - 2, "| on x^4:", np.sum(w2 * x2 ** 4), "exact", 2 / 5)
val = Q.gauss_legendre(lambda x: x ** 2 * np.exp(-x), 0, 2, 3)
print("Ex 4.7.4 int_0^2 x^2 e^-x GL3:", val, "exact", 2 - 10 * math.exp(-2))
for n in (3, 5, 10):
    print(f"Ex 4.7.5 E[X^4] N(1,4) Gauss-Hermite n={n}:", Q.gauss_hermite_expectation(lambda x: x ** 4, 1.0, 2.0, n), "exact", 1 + 6 * 4 + 3 * 16,
          " E[sigmoid(X)]:", Q.gauss_hermite_expectation(lambda x: 1 / (1 + np.exp(-x)), 1.0, 2.0, n))
xl, wl = np.polynomial.laguerre.laggauss(4)
print("Ex 4.7.6 Gauss-Laguerre int_0^inf e^-x x^3 = 6:", np.sum(wl * xl ** 3), "| int_0^inf e^-x/(1+x):", np.sum(wl / (1 + xl)), "vs", si.quad(lambda x: np.exp(-x) / (1 + x), 0, np.inf)[0])

print("\n=== 4.8 Multiple integrals ===")
f2 = lambda x, y: np.log(1 + x + y)
ex2 = si.dblquad(lambda y, x: np.log(1 + x + y), 0.0, 1.0, 0.0, 2.0)[0]
print("Ex 4.8.1 ln(1+x+y) on [0,1]x[0,2]: exact", ex2, "Simpson n=m=2", Q.simpson_double(f2, 0, 1, 0, 2, 2, 2),
      "n=m=4", Q.simpson_double(f2, 0, 1, 0, 2, 4, 4), "GL 3x3", Q.gauss_legendre_2d(f2, 0, 1, 0, 2, 3))
fr = lambda x, y: x * np.exp(y)
exr = 1 - (math.e - 1) / 2
print("Ex 4.8.2 x e^y over x^2<=y<=x, 0<=x<=1: exact", exr, "Simpson n=m=2", Q.simpson_double(fr, 0, 1, lambda x: x ** 2, lambda x: x, 2, 2),
      "n=m=4", Q.simpson_double(fr, 0, 1, lambda x: x ** 2, lambda x: x, 4, 4), "n=m=10", Q.simpson_double(fr, 0, 1, lambda x: x ** 2, lambda x: x, 10, 10))
print("Ex 4.8.3 area of quarter disc via double Simpson:", Q.simpson_double(lambda x, y: np.ones_like(y), 0, 1, 0, lambda x: np.sqrt(1 - x * x), 20, 20), "pi/4", math.pi / 4)
pdf = lambda x, y: np.exp(-(x ** 2 - 2 * 0.5 * x * y + y ** 2) / (2 * (1 - 0.25))) / (2 * math.pi * math.sqrt(0.75))
print("Ex 4.8.4 P(X<=0,Y<=0) bivariate normal rho=.5 on [-6,0]^2 GL 20x20:", Q.gauss_legendre_2d(pdf, -6, 0, -6, 0, 20), "exact 1/4+asin(.5)/(2pi) =", 0.25 + math.asin(0.5) / (2 * math.pi))
for d in (1, 2, 3, 5, 10):
    print(f"Ex 4.8.5 tensor Simpson with 11 pts/axis in d={d}: evaluations {11 ** d:.3e}")

print("\n=== 4.9 Improper integrals ===")
# int_0^1 e^x / sqrt(x): subtract Taylor P4 and integrate remainder with Simpson
P4 = lambda x: 1 - x ** 2 / 2 + x ** 4 / 24
Ip = 4 / 3 - 0.5 / (11 / 4) + (1 / 24) / (19 / 4)
Gf = lambda x: np.where(x > 0, (np.cos(x) - P4(x)) / np.where(x > 0, x, 1) ** 0.25, 0.0)
Ig = Q.simpson(Gf, 0, 1, 4)
exi = si.quad(lambda x: np.cos(x) / x ** 0.25, 0, 1)[0]
print("Ex 4.9.1 cos(x)/x^(1/4): P4 part", Ip, "remainder Simpson n=4", Ig, "total", Ip + Ig, "exact", exi, "err", abs(Ip + Ig - exi))
G2 = lambda t: np.where(t > 0, t ** 2 * np.sin(1 / np.where(t > 0, t, 1)), 0.0)
print("Ex 4.9.2 int_1^inf x^-4 sin x via t=1/x -> int_0^1 t^2 sin(1/t): Simpson n=8,32,128",
      [Q.simpson(G2, 0, 1, n) for n in (8, 32, 128)], "adaptive", Q.adaptive_simpson(G2, 0, 1, 1e-8),
      "exact", si.quad(lambda x: x ** -4 * np.sin(x), 1, np.inf, limit=200)[0])
v = Q.simpson(lambda t: np.where(t > 0, np.exp(-1 / np.where(t > 0, t, 1) ** 2) / np.where(t > 0, t, 1) ** 2, 0.0), 0, 1, 16)
print("Ex 4.9.3 int_1^inf e^{-x^2}: t=1/x Simpson n=16", v, "exact", math.sqrt(math.pi) / 2 * (1 - erf(1)))
vt = Q.gauss_legendre(lambda t: np.exp(-np.tan(t) ** 2) / np.cos(t) ** 2, 0, math.pi / 2, 20)
print("Ex 4.9.4 int_0^inf e^{-x^2} via x=tan t, GL20:", vt, "exact", math.sqrt(math.pi) / 2)
lam = 0.5
el = si.quad(lambda x: x * lam * np.exp(-lam * x), 0, np.inf)[0]
xl, wl = np.polynomial.laguerre.laggauss(2)
print("Ex 4.9.5 E[X] exponential(0.5): quad", el, "Gauss-Laguerre 2pt after x=u/lam:", np.sum(wl * xl) / lam)
sq = Q.gauss_legendre(lambda x: 1 / np.sqrt(x), 0, 1, 10)
print("Ex 4.9.6 int_0^1 x^-1/2 with GL10 (singular):", sq, "exact 2; with x=u^2:", Q.gauss_legendre(lambda u: 2 * np.ones_like(u), 0, 1, 1))

print("\n=== 4.10 Monte Carlo ===")
rng = np.random.default_rng(0)
for n in (100, 10_000, 1_000_000):
    X = rng.uniform(-1, 1, (n, 2)); ins = (X ** 2).sum(1) <= 1
    print(f"Ex 4.10.1 pi n={n}: {4 * ins.mean():.5f} se {4 * ins.std(ddof=1) / math.sqrt(n):.5f}")
rng = np.random.default_rng(1)
for d in (2, 5, 10):
    n = 100_000
    X = rng.uniform(0, 1, (n, d)); y = np.prod(np.cos(X), axis=1)
    exact = math.sin(1) ** d
    print(f"Ex 4.10.2 int cos(x1)...cos(xd) d={d}: MC {y.mean():.6f} se {y.std(ddof=1) / math.sqrt(n):.1e} exact {exact:.6f}")
rng = np.random.default_rng(2)
n = 10_000
u = rng.uniform(0, 1, n)
plain = np.exp(u)
anti = 0.5 * (np.exp(u) + np.exp(1 - u))
cv = np.exp(u) - (math.e - 1) / 0.5 * 0 - 1.6903 * (u - 0.5)
print("Ex 4.10.3 int_0^1 e^x: plain", plain.mean(), "se", plain.std(ddof=1) / math.sqrt(n),
      "| antithetic", anti.mean(), "se", anti.std(ddof=1) / math.sqrt(n), "| control var", cv.mean(), "se", cv.std(ddof=1) / math.sqrt(n), "| exact", math.e - 1)
rng = np.random.default_rng(3)
n = 100_000
z = rng.standard_normal(n)
naive = (z > 4).mean()
y = rng.standard_normal(n) + 4
wts = np.exp(-4 * y + 8)
iss = np.mean((y > 4) * wts)
from scipy.stats import norm, qmc
print("Ex 4.10.4 P(Z>4): naive", naive, "IS", iss, "se", np.std((y > 4) * wts, ddof=1) / math.sqrt(n), "exact", norm.sf(4))
for m in (8, 10, 12, 14):
    n = 2 ** m
    eq = [4 * np.mean((qmc.Sobol(d=2, scramble=True, seed=s_).random(n) ** 2).sum(1) <= 1) - math.pi for s_ in range(20)]
    em = [4 * np.mean((np.random.default_rng(100 + s_).uniform(0, 1, (n, 2)) ** 2).sum(1) <= 1) - math.pi for s_ in range(20)]
    print(f"Ex 4.10.5 n=2^{m}: RMS error over 20 runs  QMC {np.sqrt(np.mean(np.square(eq))):.2e}  MC {np.sqrt(np.mean(np.square(em))):.2e}")
rng = np.random.default_rng(5)
theta = rng.normal(0.3, 0.1, 50_000)
vals = 1 - (1 - np.clip(theta, 0, 1)) ** 5
print("Ex 4.10.6 posterior predictive P(at least one success in 5):", vals.mean(), "se", vals.std(ddof=1) / math.sqrt(len(vals)),
      "| GH:", Q.gauss_hermite_expectation(lambda t: 1 - (1 - t) ** 5, 0.3, 0.1, 20))
