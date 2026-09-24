"""Numerical answers for exercises/ch01_exercises.md. Run: python solutions/code/ch01_solutions.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib.floating import fl, ieee_double_decode, kahan_sum, logsumexp, naive_sum, pairwise_sum, softmax, variance_textbook, variance_two_pass, variance_welford  # noqa

print("A5 unstable recurrence y_n = 1/n - 5 y_{n-1}:")
y = [math.log(6 / 5)]
for n in range(1, 31):
    y.append(1 / n - 5 * y[-1])
yb = [0.0] * 41
for n in range(40, 0, -1):
    yb[n - 1] = (1 / n - yb[n]) / 5
from scipy import integrate
for n in (5, 10, 15, 20, 25, 30):
    print(f"  n={n}: forward {y[n]: .6e} backward {yb[n]:.10f} quad {integrate.quad(lambda x: x ** n / (x + 5), 0, 1)[0]:.10f}")
print("B1 P3(0.5) of e^-x:", 1 - 0.5 + 0.125 - 0.5 ** 3 / 6, "true", math.exp(-0.5), "err", abs(1 - 0.5 + 0.125 - 0.5 ** 3 / 6 - math.exp(-0.5)), "bound", 0.5 ** 4 / 24)
print("B2 Leibniz terms: need 1/(2n+3) < 1e-3 -> n+1 >= ", math.ceil((1000 - 1) / 2), "terms (n=499 -> error bound 1/1001)")
x, yv, z = 0.54617, 0.54601, 0.001234
k = 4
d = fl(fl(x, k) - fl(yv, k), k); q = fl(d / fl(z, k), k)
print("B3 fl(x), fl(y)", fl(x, k), fl(yv, k), "(x-y)/z 4-digit round:", d, q, "exact", x - yv, (x - yv) / z, "rel err", abs(q - (x - yv) / z) / ((x - yv) / z))
xs, ys = 0.3721, 0.3720
print("   x*y - x*x 4-digit round:", fl(fl(xs * ys, 4) - fl(xs * xs, 4), 4), "vs x*(y-x):", fl(xs * fl(ys - xs, 4), 4), "exact", xs * ys - xs * xs)
xx = 1e-8
print("B4 (1-cos x)/sin x naive", (1 - math.cos(xx)) / math.sin(xx), "stable tan(x/2)", math.tan(xx / 2))
xl = 1e8
print("   ln(x - sqrt(x^2-1)) naive", math.log(xl - math.sqrt(xl * xl - 1)) if xl - math.sqrt(xl * xl - 1) > 0 else "-inf/log(0)", "stable -ln(x+sqrt(x^2-1))", -math.log(xl + math.sqrt(xl * xl - 1)))
xe = 1e-10
print("   e^x - e^-x naive", math.exp(xe) - math.exp(-xe), "stable 2 sinh", 2 * math.sinh(xe))
xsn = 1e-5
tay = xsn ** 3 / 6 - xsn ** 5 / 120
print("   x - sin x naive", xsn - math.sin(xsn), "series", tay)
P = lambda t: 1.01 * t ** 3 - 4.62 * t ** 2 - 3.11 * t + 12.2
t = 1.72; k = 3
t2 = fl(t * t, k); t3 = fl(t2 * t, k)
dir_ = fl(fl(fl(fl(1.01 * t3, k) - fl(4.62 * t2, k), k) - fl(3.11 * t, k), k) + 12.2, k)
nes = fl(fl(fl(fl(fl(fl(1.01 * t, k) - 4.62, k) * t, k) - 3.11, k) * t, k) + 12.2, k)
print("B5 exact", P(t), "direct 3-digit", dir_, "nested 3-digit", nes)
for name, kap in (("x^n at any x (n=5)", 5), ("1/x", 1), ("sqrt(1+x)-1 at 1e-3", abs(1e-3 * 0.5 / math.sqrt(1 + 1e-3) / (math.sqrt(1 + 1e-3) - 1))), ("e^x at x=50", 50), ("cos x at x=pi/2-1e-3", abs((math.pi / 2 - 1e-3) * math.tan(math.pi / 2 - 1e-3)))):
    print(f"B6 kappa {name}: {kap:.4g}")
print("B7 0.625 = 0.101_2; 0.1 bits:", format(int(0.1 * 2 ** 12), "012b"), "decode:", ieee_double_decode("1 10000000001 1010" + "0" * 48))
for dt in (np.float16, np.float32, np.float64):
    one = dt(1); eps = dt(1)
    while one + eps / dt(2) > one:
        eps = eps / dt(2)
    print("C1", dt.__name__, eps, np.finfo(dt).eps)
for n in (10 ** 6, 10 ** 7):
    xs = (1.0 / np.arange(1, n + 1)).astype(np.float32)
    exact = math.fsum(1.0 / k for k in range(1, n + 1))
    naive = float(np.cumsum(xs, dtype=np.float32)[-1])
    print(f"C2 n={n}: exact {exact:.10f} naive float32 {naive:.7f} (err {abs(naive - exact):.2e})  np.sum float32 {float(np.sum(xs)):.7f} (err {abs(float(np.sum(xs)) - exact):.2e})  kahan32 ", end="")
    s = np.float32(0); c = np.float32(0)
    for v in xs:
        yk = v - c; tk = s + yk; c = (tk - s) - yk; s = tk
    print(f"{float(s):.7f} (err {abs(float(s) - exact):.2e})")
hs = 10.0 ** -np.arange(1, 16)
err = np.abs((np.sin(1 + hs) - np.sin(1)) / hs - np.cos(1))
print("C3 forward diff errors:", ["%.1e" % e for e in err], "best h", hs[np.argmin(err)])
print("C4 logsumexp([-1000,-1001]) =", logsumexp([-1000, -1001]), "softmax([1e4, 1e4+1])", softmax([1e4, 1e4 + 1]))
dat = 1e6 + np.random.default_rng(0).standard_normal(1000)
for dt in (np.float32, np.float64):
    d = dat.astype(dt)
    print("C5", dt.__name__, "textbook", float(variance_textbook(d)), "two-pass", float(variance_two_pass(d)), "welford", float(variance_welford(d)))
p = np.full(100000, 0.7); yb_ = (np.random.default_rng(1).random(100000) < 0.7)
lik = np.prod(np.where(yb_, p, 1 - p)); ll = np.sum(np.where(yb_, np.log(p), np.log(1 - p)))
print("D1 likelihood", lik, "log-likelihood", ll, "per-observation", ll / 1e5, "entropy-like", 0.7 * math.log(0.7) + 0.3 * math.log(0.3))
g = np.float16(1e-8); print("D2 float16(1e-8) =", g, "; scaled by 1024 then cast:", np.float16(1e-8 * 1024), "-> unscale in float32:", np.float32(np.float16(1e-8 * 1024)) / 1024, "; smallest float16 subnormal", np.finfo(np.float16).smallest_subnormal)
