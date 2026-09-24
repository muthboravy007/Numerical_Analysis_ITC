"""Numerical answers for exercises/ch04_exercises.md."""
import math
import os
import sys

import numpy as np
from scipy import integrate as si
from scipy.special import erf
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import integration as Q  # noqa: E402

f = lambda x: x * np.exp(x)
for h in (0.1, 0.05):
    print(f"B1 h={h}: fwd {(f(1 + h) - f(1)) / h:.6f} central {(f(1 + h) - f(1 - h)) / (2 * h):.6f} 5pt {Q.five_point_midpoint(f, 1.0, h):.8f} true {2 * math.e:.8f}")
N = lambda h: (f(1 + h) - f(1 - h)) / (2 * h)
n1, n2 = N(0.2), N(0.1); print("B2 N(0.2), N(0.1), extrap", n1, n2, (4 * n2 - n1) / 3, "err", abs((4 * n2 - n1) / 3 - 2 * math.e))
g = lambda x: 1 / (1 + x ** 2)
print("B3 trap", 0.5 * (g(0) + g(1)), "simpson", (g(0) + 4 * g(0.5) + g(1)) / 6, "midpoint", g(0.5), "exact pi/4", math.pi / 4)
for n in (4, 8):
    print(f"B4 n={n}: comp trap {Q.trapezoid(g, 0, 1, n):.8f} comp simpson {Q.simpson(g, 0, 1, n):.8f} err T {abs(Q.trapezoid(g, 0, 1, n) - math.pi / 4):.2e} S {abs(Q.simpson(g, 0, 1, n) - math.pi / 4):.2e}")
fs = lambda x: np.sin(x); print("B5 n for trapezoid 1e-6 on sin [0,pi]: sqrt(pi^3/(12e-6))", math.sqrt(math.pi ** 3 / 12e-6), " simpson (pi^5/(180e-6))^(1/4)", (math.pi ** 5 / 180e-6) ** 0.25)
v, R = Q.romberg(lambda x: np.exp(-x) * np.cos(x), 0, math.pi / 2, 4)
print("B6 Romberg\n", R, "\n exact", 0.5 * (1 + math.exp(-math.pi / 2) * (math.sin(math.pi / 2) - math.cos(math.pi / 2))) )
ex_int = si.quad(lambda x: np.exp(-x) * np.cos(x), 0, math.pi / 2)[0]; print("   quad", ex_int)
x2, w2 = np.polynomial.legendre.leggauss(2); x3, w3 = np.polynomial.legendre.leggauss(3)
gl2 = 1.5 * np.sum(w2 * (lambda t: np.log(t))(1.5 * x2 + 2.5)); gl3 = 1.5 * np.sum(w3 * np.log(1.5 * x3 + 2.5))
print("B7 int_1^4 ln x: GL2", gl2, "GL3", gl3, "exact", 4 * math.log(4) - 3)
print("B8 double Simpson int_0^1 int_0^x (x+y) dy dx:", Q.simpson_double(lambda x, y: x + y, 0, 1, 0, lambda x: x, 2, 2), "exact 0.5")
Gf = lambda x: np.where(x > 0, (np.exp(-x) - (1 - x + x ** 2 / 2)) / np.sqrt(np.where(x > 0, x, 1)), 0.0)
Pp = 2 - 2 / 3 + 1 / 5
print("B9 int_0^1 e^-x/sqrt x: poly part", Pp, "remainder S4", Q.simpson(Gf, 0, 1, 4), "total", Pp + Q.simpson(Gf, 0, 1, 4), "exact", math.sqrt(math.pi) * erf(1))
for h in 10.0 ** -np.arange(1, 9):
    e2 = abs((np.log(2 + h) - 2 * np.log(2) + np.log(2 - h)) / h ** 2 + 0.25)
    print(f"C1 h={h:.0e} second-diff err {e2:.2e}")
gauss = lambda x: np.exp(-x ** 2)
for tol in (1e-4, 1e-8):
    v, ne = Q.adaptive_simpson(lambda x: np.sqrt(x) * np.log(np.where(x > 0, x, 1)), 0, 1, tol)
    print(f"C2 adaptive int sqrt(x) ln x tol {tol}: {v} evals {ne} exact {-4 / 9}")
for n in (2, 4, 8, 16):
    print(f"C3 GL n={n} for int_-1^1 |x| : {Q.gauss_legendre(np.abs, -1, 1, n):.6f} ; for e^x: err {abs(Q.gauss_legendre(np.exp, -1, 1, n) - (math.e - 1 / math.e)):.1e}")
rng = np.random.default_rng(0)
for d in (1, 3, 6, 10):
    n = 200000; X = rng.uniform(0, 1, (n, d)); y = np.exp(-np.sum(X ** 2, axis=1))
    ex = (math.sqrt(math.pi) / 2 * erf(1)) ** d
    print(f"C4 d={d}: MC {y.mean():.6f} +- {y.std() / math.sqrt(n):.1e} exact {ex:.6f}")
yt = np.array([0, 0.2, 0.2, 0.55, 0.55, 0.8, 1.0, 1.0]); xt = np.array([0, 0, 0.1, 0.1, 0.4, 0.4, 0.7, 1.0])
print("D1 AUC", Q.trapezoid_data(xt, yt))
t = np.linspace(0, 24, 13); cpk = 10 * (np.exp(-0.2 * t) - np.exp(-1.2 * t))
print("D2 AUC trap", Q.trapezoid_data(t, cpk), "simpson", si.simpson(cpk, x=t), "exact", 10 * ((1 - math.exp(-4.8)) / 0.2 - (1 - math.exp(-28.8)) / 1.2))
mu, sd = 0.5, 1.2
for n in (5, 10, 20):
    print(f"D3 E[max(X,0)] GH n={n}: {Q.gauss_hermite_expectation(lambda x: np.maximum(x, 0), mu, sd, n):.6f}  exact {mu * norm.cdf(mu / sd) + sd * norm.pdf(mu / sd):.6f}")
z = rng.standard_normal(100000)
print("   MC", np.mean(np.maximum(mu + sd * z, 0)), "+-", np.std(np.maximum(mu + sd * z, 0)) / math.sqrt(100000))
S0, K, r, sig, T = 100.0, 110.0, 0.03, 0.25, 1.0
zz = rng.standard_normal(200000); ST = S0 * np.exp((r - 0.5 * sig ** 2) * T + sig * math.sqrt(T) * zz); pay = math.exp(-r * T) * np.maximum(ST - K, 0)
STa = S0 * np.exp((r - 0.5 * sig ** 2) * T - sig * math.sqrt(T) * zz); paya = math.exp(-r * T) * np.maximum(STa - K, 0)
d1 = (math.log(S0 / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T)); bs = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - sig * math.sqrt(T))
print("D4 option MC", pay.mean(), "+-", pay.std() / math.sqrt(len(pay)), "antithetic", (0.5 * (pay + paya)).mean(), "+-", (0.5 * (pay + paya)).std() / math.sqrt(len(pay)), "BS", bs)
