"""Reproduces every numerical example in Chapter 1 of the lecture notes.
Run from the repository root:  python lectures/code/ch01.py"""

import math
import os
import sys

import numpy as np
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib.floating import (fl, ieee_double_decode, kahan_sum, logsumexp,  # noqa: E402
                             naive_sum, quadratic_roots_naive, quadratic_roots_stable,
                             variance_textbook, variance_two_pass, variance_welford)

R = lambda x, k=5: fl(x, k)

print("=== 1.1 Review of calculus ===")
f = lambda x: x ** 5 - 2 * x ** 3 + 3 * x ** 2 - 1
print("Ex 1.1.1 f(0), f(1), f(0.5), f(0.75):", f(0), f(1), f(0.5), f(0.75))
print("Ex 1.1.2 MVT c = 2/sqrt(3) =", 2 / math.sqrt(3))
g = lambda x: 2 - math.exp(x) + 2 * x
print("Ex 1.1.3 g(0), g(ln2), g(1):", g(0), g(math.log(2)), g(1))
print("Ex 1.1.4 cos(0.01) - P2(0.01) =", math.cos(0.01) - (1 - 0.01 ** 2 / 2), "bound", 0.01 ** 4 / 24)
print("Ex 1.1.5 sin(0.1) =", math.sin(0.1), "approx", 0.1 - 0.1 ** 3 / 6,
      "err", math.sin(0.1) - (0.1 - 0.1 ** 3 / 6), "bound", 0.1 ** 5 / 120)
for n in range(6, 11):
    approx = sum(1 / math.factorial(k) for k in range(n + 1))
    print(f"Ex 1.1.6 n={n}: bound 3/(n+1)! = {3 / math.factorial(n + 1):.3e}, actual err {math.e - approx:.3e}")
h = lambda x: math.exp(-x * x)
print("Ex 1.1.7 weighted MVT: int_0^1 x e^{-x^2} =", integrate.quad(lambda x: x * h(x), 0, 1)[0],
      "=(1-e^-1)/2 =", (1 - math.exp(-1)) / 2)

print("\n=== 1.2 Round-off errors ===")
print("Ex 1.2.1 decode:", ieee_double_decode("0 10000000010 0110" + "0" * 48),
      "next:", ieee_double_decode("0 10000000010 0110" + "0" * 47 + "1") - 11)
for mode in ("chop", "round"):
    v = fl(math.pi, 5, mode)
    print(f"Ex 1.2.2 pi {mode}: {v} abs {abs(math.pi - v):.4e} rel {abs(math.pi - v) / math.pi:.4e}")
x, y = 2 / 3, 3 / 11
fx, fy = R(x), R(y)
print("Ex 1.2.3 fl(x), fl(y):", fx, fy)
for name, ex, ap in (("x+y", x + y, R(fx + fy)), ("x-y", x - y, R(fx - fy)),
                     ("x*y", x * y, R(fx * fy)), ("x/y", x / y, R(fx / fy))):
    print(f"   {name}: exact {ex:.10f} 5-digit {ap} abs {abs(ex - ap):.3e} rel {abs(ex - ap) / abs(ex):.3e}")
u = 0.66666
print("   x-u exact", x - u, "5-digit", R(fx - u), "rel", abs((x - u) - R(fx - u)) / abs(x - u))
b = 97.43
d = R(math.sqrt(R(R(b * b, 4) - 4, 4)), 4)
ex1 = (-b + math.sqrt(b * b - 4)) / 2
ex2 = (-b - math.sqrt(b * b - 4)) / 2
n1 = R(R(-b + d, 4) / 2, 4)
alt = R(-2 / R(b + d, 4), 4)
print("Ex 1.2.4 b^2 ->", R(b * b, 4), "sqrt ->", d, "exact roots", ex1, ex2)
print("   naive x1", n1, "rel", abs(n1 - ex1) / abs(ex1), "| rationalised", alt, "rel", abs(alt - ex1) / abs(ex1))
F = lambda x: x ** 3 - 5.2 * x ** 2 + 2.7 * x + 1.4
x0, k = 3.83, 3
x2 = R(x0 * x0, k); x3 = R(x2 * x0, k); t1 = R(5.2 * x2, k); t2 = R(2.7 * x0, k)
direct = R(R(R(x3 - t1, k) + t2, k) + 1.4, k)
h1 = R(x0 - 5.2, k); h2 = R(h1 * x0, k); h3 = R(h2 + 2.7, k); h4 = R(h3 * x0, k); nested = R(h4 + 1.4, k)
print("Ex 1.2.5 exact", F(x0), "| direct", x2, x3, t1, t2, direct, "| nested", h1, h2, h3, h4, nested)
print("   rel errors: direct", abs(direct - F(x0)) / abs(F(x0)), "nested", abs(nested - F(x0)) / abs(F(x0)))
print("Ex 1.2.7 quadratic x^2+1e8 x+1: naive", quadratic_roots_naive(1, 1e8, 1),
      "stable", quadratic_roots_stable(1, 1e8, 1))

print("\n=== 1.3 Algorithms and convergence ===")
p = [1.0, R(1 / 3)]
for n in range(2, 13):
    p.append(R(R(R(7 / 3) * p[-1]) - R(R(2 / 3) * p[-2])))
for n in (0, 2, 4, 6, 8, 10, 12):
    print(f"Ex 1.3.1 n={n}: exact {(1 / 3) ** n:.5e} computed {p[n]:.5e} err {abs(p[n] - (1 / 3) ** n):.3e}")
q = [1.0, R(1 / 3)]
for n in range(2, 13):
    q.append(R(R(2 * q[-1]) - q[-2]))
for n in (2, 4, 6, 8, 10, 12):
    print(f"Ex 1.3.2 n={n}: exact {1 - 2 * n / 3:.5f} computed {q[n]} err {abs(q[n] - (1 - 2 * n / 3)):.2e}")
I = [1 - math.exp(-1)]
for n in range(1, 21):
    I.append(1 - n * I[-1])
J = [0.0] * 31
for n in range(30, 0, -1):
    J[n - 1] = (1 - J[n]) / n
for n in (0, 5, 10, 15, 17, 18, 19, 20):
    print(f"Ex 1.3.3 I_{n}: forward {I[n]: .10f} backward {J[n]:.10f}")
for hh in (0.1, 0.01, 0.001):
    print(f"Ex 1.3.4 h={hh}: cos h + h^2/2 - 1 = {math.cos(hh) + hh * hh / 2 - 1:.4e}  h^4/24 = {hh ** 4 / 24:.4e}")
for n in (10, 100, 1000):
    print(f"Ex 1.3.5 n={n}: (2n+1)/n^2 = {(2 * n + 1) / n ** 2:.6f}  (n+2)/n^3 = {(n + 2) / n ** 3:.3e}")
for hh in (0.1, 0.01, 0.001):
    print(f"Ex 1.3.6 h={hh}: (e^h-1)/h - 1 = {(math.exp(hh) - 1) / hh - 1:.4e}")

print("\n=== 1.4 Conditioning ===")
print("Ex 1.4.2 kappa ln at 1.001 =", 1 / abs(math.log(1.001)))
print("Ex 1.4.3 kappa tan at 1.57 =", abs(1.57 / (math.cos(1.57) ** 2 * math.tan(1.57))),
      "tan(1.57)=", math.tan(1.57), "tan(1.5701)=", math.tan(1.5701))
A = np.array([[1, 1], [1, 1.0001]])
print("Ex 1.4.4 cond_inf =", np.linalg.cond(A, np.inf), np.linalg.solve(A, [2, 2.0001]), np.linalg.solve(A, [2, 2.0002]))
w = np.poly(np.arange(1, 21))
w2 = w.copy(); w2[1] += -2 ** -23
r = np.sort_complex(np.roots(w2))
print("Ex 1.4.5 Wilkinson perturbed roots (subset):", np.round(r[10:16], 3))
y = 1.4142
print("Ex 1.4.6 forward", abs(y - math.sqrt(2)), "backward", abs(2 - y * y))

print("\n=== 1.5 Floating point in data science ===")
xs = [0.1] * 1_000_000
e = math.fsum(xs)
print("Ex 1.5.1 naive err", naive_sum(xs) - e, "kahan err", kahan_sum(xs) - e, "np.sum err", np.sum(xs) - e)
d = 1e9 + np.array([4.0, 7, 13, 16])
print("Ex 1.5.2 variances:", variance_textbook(d), variance_two_pass(d), variance_welford(d))
print("Ex 1.5.3 logsumexp(1000,1001,1002) =", logsumexp([1000, 1001, 1002]))
pr = np.full(1000, 0.01)
print("Ex 1.5.4 prod =", np.prod(pr), "sum log =", np.sum(np.log(pr)))
acc = np.float32(0)
for _ in range(10_000):
    acc += np.float32(0.1)
print("Ex 1.5.5 float32 accumulate =", acc, "| float32(2**24)+1 =", np.float32(2 ** 24) + np.float32(1))
z = np.array([-800.0, 0.0])
with np.errstate(over="ignore"):
    naive_sig = 1 / (1 + np.exp(800.0))
print("Ex 1.5.6 sigmoid(-800) naive:", naive_sig, " log(sigmoid(-800)) stable:", -np.logaddexp(0, 800.0))
