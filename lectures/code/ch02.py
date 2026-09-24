"""Reproduces every numerical example in Chapter 2 (Solutions of Equations in One Variable).
Run from the repository root:  python lectures/code/ch02.py"""

import math
import os
import sys

import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import roots  # noqa: E402

f = lambda x: x ** 3 - 2 * x - 5
df = lambda x: 3 * x ** 2 - 2
r = 2.0945514815423265


def show(label, hist, fn=None, digits=10, every=1, limit=None):
    print(label)
    for k, v in enumerate(hist[:limit]):
        if k % every == 0:
            extra = f"  f={fn(v):+.3e}" if fn else ""
            print(f"   {k:2d}: {v:.{digits}f}{extra}")


print("=== 2.1 Bisection ===")
x, h = roots.bisection(f, 2, 3, 1e-6)
show("Ex 2.1.1 x^3-2x-5 on [2,3]", h, f, 6, limit=8)
print("   iterations:", len(h), "| bound formula:", roots.bisection_iterations(2, 3, 1e-6))
print("Ex 2.1.2 iterations for 1e-4 on [1,2]:", math.ceil(math.log2(1 / 1e-4)))
g = lambda x: math.exp(x) - 3 * x
x, h = roots.bisection(g, 0, 1, 1e-3)
show("Ex 2.1.3 e^x - 3x on [0,1]", h, g, 6)
x, h = roots.bisection(lambda x: x * x - 3, 1, 2, 1e-4)
print("Ex 2.1.4 sqrt3:", x, "iterations", len(h), "error", abs(x - math.sqrt(3)))
x, h = roots.bisection(lambda x: 1 / (x - 1), 0, 2.5, 1e-8)
print("Ex 2.1.5 1/(x-1) on [0,2.5] ->", x, "f there:", 1 / (x - 1))
p = 1.05
print("Ex 2.1.6 (x-1)^10 at 1.05:", (p - 1) ** 10, "| at 1.2:", 0.2 ** 10)

print("\n=== 2.2 Fixed-point iteration ===")
gA = lambda x: (x * x + 2) / 5
x, h = roots.fixed_point(gA, 0.0, 1e-8)
show("Ex 2.2.1 g=(x^2+2)/5, p=(5-sqrt17)/2=" + str((5 - math.sqrt(17)) / 2), h, None, 8, limit=8)
print("   iterations:", len(h) - 1)
x, h = roots.fixed_point(lambda x: 3 ** (-x), 0.5, 1e-8)
print("Ex 2.2.2 g=3^-x: p =", x, "iterations", len(h) - 1, "g'(0)=", -math.log(3), "g'(p)=", -math.log(3) * 3 ** (-x))
k = 0.4
n = math.ceil(math.log(1e-5 * (1 - k) / abs(gA(0) - 0)) / math.log(k))
print("Ex 2.2.3 a-priori n for 1e-5 with k=0.4, |p1-p0|=0.4:", n, "actual error after n:",
      abs(roots.fixed_point(gA, 0.0, 1e-15)[1][n] - (5 - math.sqrt(17)) / 2))
x, h = roots.fixed_point(math.cos, 1.0, 1e-8)
print("Ex 2.2.4 cos: p =", x, "iter", len(h) - 1, "|g'(p)|=", math.sin(x))
e = [abs(v - x) for v in h]
print("   error ratios:", [round(e[i + 1] / e[i], 3) for i in range(5, 10)])
a = 2.0
seq = [1.0]
for _ in range(5):
    seq.append(a / seq[-1])
print("Ex 2.2.5 g=a/x:", seq)
x, h = roots.fixed_point(lambda x: 0.5 * (x + a / x), 1.0, 1e-14)
print("   g=(x+a/x)/2:", h)

print("\n=== 2.3 Newton, secant, false position ===")
x, h = roots.newton(f, df, 2.0)
show("Ex 2.3.1 Newton x^3-2x-5", h, None, 16)
q = lambda x: x - math.exp(-x)
dq = lambda x: 1 + math.exp(-x)
xn, hn = roots.newton(q, dq, 1.0, tol=1e-14)
xs, hs = roots.secant(q, 0.0, 1.0, tol=1e-14)
xf, hf = roots.regula_falsi(q, 0.0, 1.0, tol=1e-14)
show("Ex 2.3.2 Newton x=e^-x", hn, None, 15)
show("Ex 2.3.3 secant x=e^-x", hs, None, 15)
show("Ex 2.3.4 false position x=e^-x (first 8)", hf, None, 15, limit=8)
print("   false position iterations:", len(hf))
h = [0.0]
for _ in range(4):
    h.append(h[-1] - (h[-1] ** 3 - 2 * h[-1] + 2) / (3 * h[-1] ** 2 - 2))
print("Ex 2.3.5 cycling x^3-2x+2 from 0:", h)
h = [1.5]
for _ in range(4):
    h.append(h[-1] - math.atan(h[-1]) * (1 + h[-1] ** 2))
print("   arctan from 1.5:", h)
h = [1.3]
for _ in range(5):
    h.append(h[-1] - math.atan(h[-1]) * (1 + h[-1] ** 2))
print("   arctan from 1.3:", h)
aa = 7.0
h = [0.1]
for _ in range(6):
    h.append(h[-1] * (2 - aa * h[-1]))
print("Ex 2.3.6 reciprocal of 7:", h, 1 / 7)

print("\n=== 2.4 Error analysis ===")
lin = [0.5 ** n for n in range(8)]
quad = [0.5 ** (2 ** n) for n in range(8)]
print("Ex 2.4.1 linear:", lin)
print("   quadratic:", quad)
m1 = lambda x: math.exp(x) - x - 1
dm1 = lambda x: math.exp(x) - 1
d2m1 = lambda x: math.exp(x)
x, h = roots.newton(m1, dm1, 1.0, tol=1e-8, max_iter=200)
print("Ex 2.4.2 Newton on e^x-x-1 from 1: iterations", len(h) - 1, "first:", [round(v, 6) for v in h[:6]])
hh = [1.0]
for _ in range(5):
    xk = hh[-1]
    if m1(xk) == 0:
        break
    hh.append(xk - m1(xk) * dm1(xk) / (dm1(xk) ** 2 - m1(xk) * d2m1(xk)))
print("   modified mu=f/f':", hh)
hm = [1.0]
for _ in range(5):
    xk = hm[-1]
    if dm1(xk) == 0:
        break
    hm.append(xk - 2 * m1(xk) / dm1(xk))
print("Ex 2.4.3 m=2 Newton:", hm)
xs, hs = roots.secant(f, 2.0, 3.0)
es = [abs(v - r) for v in hs if abs(v - r) > 0]
print("Ex 2.4.4 secant errors:", ["%.2e" % v for v in es])
print("   order estimates:", [round(math.log(es[i + 1] / es[i]) / math.log(es[i] / es[i - 1]), 3) for i in range(1, len(es) - 1)])
x, h = roots.newton(lambda x: x * x - 5, lambda x: 2 * x, 2.0)
e = [abs(v - math.sqrt(5)) for v in h]
print("Ex 2.4.5 sqrt5 ratios e_{k+1}/e_k^2:", [e[i + 1] / e[i] ** 2 for i in range(3)], "theory 1/(2 sqrt5)=", 1 / (2 * math.sqrt(5)))

print("\n=== 2.5 Accelerating convergence ===")
seq = [1.0]
for _ in range(10):
    seq.append(math.cos(seq[-1]))
acc = roots.aitken(seq)
pstar = 0.7390851332151607
print("Ex 2.5.1")
for n in range(7):
    print(f"   {n}: p={seq[n]:.9f} err={abs(seq[n] - pstar):.2e}  aitken={acc[n]:.9f} err={abs(acc[n] - pstar):.2e}")
S = np.cumsum([(-1) ** k / (2 * k + 1) for k in range(10)])
A1 = roots.aitken(S)
A2 = roots.aitken(A1)
print("Ex 2.5.2 Leibniz partial sums x4:", 4 * S, "\n   Aitken x4:", 4 * A1, "\n   Aitken^2 x4:", 4 * A2, "\n   pi:", math.pi)
x, h = roots.steffensen(math.cos, 1.0)
print("Ex 2.5.3 Steffensen cos:", h)
x, h = roots.steffensen(lambda x: (2 * x + 5) ** (1 / 3), 2.0)
print("Ex 2.5.4 Steffensen cube root form:", h)
seq = [0.5]
for _ in range(8):
    seq.append(math.exp(-seq[-1]))
acc = roots.aitken(seq)
ps = 0.5671432904097838
print("Ex 2.5.5 e^-x: p errs", ["%.1e" % abs(v - ps) for v in seq[:7]], "aitken errs", ["%.1e" % abs(v - ps) for v in acc[:7]])

print("\n=== 2.6 Polynomials ===")
P = [1, -2, -5, 6, 3]
val, dval, quo = roots.horner_eval(P, 2.0)
print("Ex 2.6.1 Horner at 2: P=", val, "P'=", dval, "quotient", quo)
x, h = roots.newton_horner(P, 2.0)
print("Ex 2.6.2 Newton-Horner from 2:", h)
print("   all roots np.roots:", np.sort(np.roots(P)))
coef = P
found = []
x0s = [2.0, 1.0, -1.0, -2.0]
for x0 in x0s:
    if len(coef) < 2:
        break
    x, _ = roots.newton_horner(coef, x0)
    found.append(x)
    coef = roots.horner_eval(coef, x)[2]
print("Ex 2.6.3 deflation roots:", found, "last quotient", coef)
Pm = lambda x: x ** 3 + 2 * x ** 2 + 3 * x + 4
z, h = roots.muller(Pm, 0.5, 1.0, 1.5)
print("Ex 2.6.4 Muller from 0.5,1,1.5:", [complex(round(v.real, 8), round(v.imag, 8)) for v in h])
print("   np.roots:", np.roots([1, 2, 3, 4]))
C = np.array([[-2.0, -3, -4], [1, 0, 0], [0, 1, 0]])
print("Ex 2.6.5 companion eigenvalues:", np.linalg.eigvals(C))
print("   Cauchy bound 1+max|a_i/a_n| =", 1 + 4)

print("\n=== 2.7 Data-science applications ===")
x, h = roots.newton(lambda z: norm.cdf(z) - 0.975, norm.pdf, 2.0)
print("Ex 2.7.1 z_0.975 Newton:", h)
data = np.array([2.3, 1.1, 4.5, 3.2, 0.8, 2.9, 5.1, 1.7, 2.2, 3.6])
s = np.log(data.mean()) - np.log(data).mean()
a0 = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
a, h = roots.newton(lambda a: np.log(a) - digamma(a) - s, lambda a: 1 / a - polygamma(1, a), a0)
print("Ex 2.7.2 Gamma MLE: s", s, "a0", a0, "iter", [float(v) for v in h], "theta", data.mean() / a)
npv = lambda r: -1000 + 300 / (1 + r) + 400 / (1 + r) ** 2 + 500 / (1 + r) ** 3
x, h = roots.secant(npv, 0.0, 0.2)
print("Ex 2.7.3 IRR secant:", h, "NPV(0.2)=", npv(0.2))
cd = np.array([-1.2, 0.3, 0.9, 1.4, 1.8, 2.1, 9.5])
score = lambda t: np.sum(2 * (cd - t) / (1 + (cd - t) ** 2))
dscore = lambda t: np.sum((2 * (cd - t) ** 2 - 2) / (1 + (cd - t) ** 2) ** 2)
for t0 in (np.median(cd), np.mean(cd), 6.0):
    try:
        x, h = roots.newton(score, dscore, float(t0))
        print(f"Ex 2.7.4 Cauchy from {t0:.4f}: root {x:.6f} in {len(h) - 1} its; loglik'' sign {np.sign(dscore(x))}")
    except Exception as ex:
        print("Ex 2.7.4 from", t0, "failed:", ex)
rng = np.random.default_rng(0)
logits = rng.normal(-1.0, 1.5, 2000)
sig = lambda z: 1 / (1 + np.exp(-z))
F = lambda t: np.mean(sig(logits + t)) - 0.30
dF = lambda t: np.mean(sig(logits + t) * (1 - sig(logits + t)))
print("Ex 2.7.5 base rate:", np.mean(sig(logits)))
x, h = roots.newton(F, dF, 0.0)
print("   shift t:", h, "new rate:", np.mean(sig(logits + x)))
