"""Numerical answers for exercises/ch02_exercises.md. Run: python solutions/code/ch02_solutions.py"""
import math
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import roots  # noqa: E402

f = lambda x: x ** 3 + 2 * x ** 2 - 3 * x - 1
x, h = roots.bisection(f, 1, 2, 1e-12)
print("B1 f(1), f(2):", f(1), f(2), "first 4 midpoints", [(m, round(f(m), 6)) for m in h[:4]], "root", x, "n for 1e-4:", math.ceil(math.log2(1 / 1e-4)))
for name, g, dg in (("x^2-2", lambda x: x * x - 2, lambda x: 2 * x), ("sqrt(x+2)", lambda x: math.sqrt(x + 2), lambda x: 0.5 / math.sqrt(x + 2)), ("1+2/x", lambda x: 1 + 2 / x, lambda x: -2 / x ** 2)):
    seq = [2.5]
    for _ in range(6):
        try:
            seq.append(g(seq[-1]))
        except Exception:
            break
    print(f"B2 g={name}: |g'(2)|={abs(dg(2.0)):.3f} iterates {np.round(seq, 5)}")
x, h = roots.newton(lambda x: x ** 3 - 5, lambda x: 3 * x * x, 2.0)
print("B3 Newton cube root 5:", h, "5^(1/3)=", 5 ** (1 / 3))
g = lambda x: x * x - math.exp(-x)
x, h = roots.secant(g, 0.0, 1.0, tol=1e-12); print("B4 secant:", h)
x, h = roots.regula_falsi(g, 0.0, 1.0, tol=1e-12); print("B5 false position first 4:", h[:4], "iterations", len(h), "root", x)
seq = [2.5]
for _ in range(8):
    seq.append(math.sqrt(seq[-1] + 2))
print("B6 fixed point seq", np.round(seq, 8), "\n   Aitken", np.round(roots.aitken(seq), 10))
P = [1, -3, 1, 1, 1]
val, der, q = roots.horner_eval(P, 2.0); print("B7 Horner P(2)", val, "P'(2)", der, "quotient", q)
x, h = roots.newton_horner(P, 2.5); print("   Newton-Horner from 2.5:", h, "all roots", np.roots(P))
z, hz = roots.muller(lambda t: t ** 4 - 3 * t ** 3 + t ** 2 + t + 1, 0.0, 0.5, 1.0); print("   Muller from 0,0.5,1:", z, len(hz), "| from -0.5,-0.3,0:", roots.muller(lambda t: t ** 4 - 3 * t ** 3 + t ** 2 + t + 1, -0.5, -0.3, 0.0)[0])
fm = lambda x: (x - 1) ** 3; dfm = lambda x: 3 * (x - 1) ** 2
xs = [2.0]
for _ in range(6):
    xs.append(xs[-1] - fm(xs[-1]) / dfm(xs[-1]))
print("B8 Newton on (x-1)^3:", xs, "ratios", [(xs[i + 1] - 1) / (xs[i] - 1) for i in range(5)], "modified m=3 one step:", 2 - 3 * fm(2) / dfm(2))
F = lambda x: x - 2 * math.sin(x); dF = lambda x: 1 - 2 * math.cos(x)
xb, hb = roots.bisection(F, 1.5, 2.5, 1e-12); xn, hn = roots.newton(F, dF, 2.0); xs_, hs = roots.secant(F, 1.5, 2.5)
r = brentq(F, 1.5, 2.5, xtol=1e-15)
print("C1 root", r, "iterations bisection", len(hb), "newton", len(hn) - 1, "secant", len(hs) - 2)
print("C2 orders newton", roots.estimate_order(hn, r)[:4], "secant", roots.estimate_order(hs, r)[:6])
xg = np.linspace(-2, 2, 401); Z = xg[None, :] + 1j * xg[:, None]
with np.errstate(divide="ignore", invalid="ignore"):
    for _ in range(60):
        Z = Z - (Z ** 3 - 1) / (3 * Z ** 2)
rts = np.array([1, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)])
lab = np.argmin(np.abs(Z[..., None] - rts), axis=-1)
print("C3 basin fractions", [round(float(np.mean(lab == k)), 4) for k in range(3)])
x, h = roots.safeguarded_newton(lambda x: math.atan(x), lambda x: 1 / (1 + x * x), -2.0, 5.0); print("C4 safeguarded Newton arctan on [-2,5]:", x, "iterates", h)
w = np.poly(np.arange(1, 11))
print("C5 np.roots max err Wilkinson-10", np.max(np.abs(np.sort(np.roots(w).real) - np.arange(1, 11))))
coef = list(w); found = []
for _ in range(10):
    xr, _ = roots.newton_horner(coef, 0.0); found.append(xr); coef = roots.horner_eval(coef, xr)[2]
coef2 = list(w); f2 = []
for _ in range(10):
    xr, _ = roots.newton_horner(coef2, 12.0); f2.append(xr); coef2 = roots.horner_eval(coef2, xr)[2]
print("   deflation (largest first) max err", np.max(np.abs(np.sort(f2) - np.arange(1, 11))))
print("   deflation (smallest first) max err", np.max(np.abs(np.sort(found) - np.arange(1, 11))))
npv = lambda r: -100 + 230 / (1 + r) - 132 / (1 + r) ** 2
print("D1 NPV roots", brentq(npv, 0.0, 0.15), brentq(npv, 0.15, 0.5), "NPV(0)", npv(0), "NPV(0.15)", npv(0.15))
xbar = 1.8
lam, h = roots.newton(lambda l: l / (1 - math.exp(-l)) - xbar, lambda l: (1 - math.exp(-l) - l * math.exp(-l)) / (1 - math.exp(-l)) ** 2, xbar)
print("D2 zero-truncated Poisson MLE lambda", lam, "iterations", h)
q, h = roots.newton(lambda x: stats.gamma.cdf(x, 3, scale=2) - 0.95, lambda x: stats.gamma.pdf(x, 3, scale=2), 6.0)
print("D3 Gamma(3,2) 95% quantile", q, h, "scipy", stats.gamma.ppf(0.95, 3, scale=2))
d2 = np.array([1.0, 1.5, 2.0, 4.0, 5.0, 9.0])
def H(beta):
    p = np.exp(-beta * d2); p /= p.sum(); return -np.sum(p * np.log2(p))
target = math.log2(3.0)
b, hb = roots.bisection(lambda be: H(be) - target, 1e-3, 20, 1e-10)
pb = np.exp(-b * d2); pb /= pb.sum()
print("D4 perplexity 3: beta", b, "iterations", len(hb), "entropy", H(b), "perplexity", 2 ** H(b), "probs", np.round(pb, 4))
