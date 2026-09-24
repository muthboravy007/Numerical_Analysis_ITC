"""Chapter 1 worked examples — run:  python examples/ch01_examples.py"""

import math
import os
import sys
from decimal import Decimal

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import floating  # noqa: E402

print("Example 1.1")
for approx in (22 / 7, 355 / 113):
    print(f"  {approx:.10f}  abs={abs(approx - math.pi):.3e}  "
          f"rel={floating.relative_error(approx, math.pi):.3e}  "
          f"digits={floating.significant_digits(approx, math.pi):.1f}")

print("Example 1.2")
print("  0.1 + 0.2 =", 0.1 + 0.2, "| stored 0.1 =", Decimal(0.1))
print("  isclose:", math.isclose(0.1 + 0.2, 0.3))

print("Example 1.3")
print("  1e16 + 1 - 1e16 =", 1e16 + 1 - 1e16)
print("  float32(2**24) + 1 =", np.float32(2 ** 24) + np.float32(1))
acc = np.float32(0)
for _ in range(10_000):
    acc += np.float32(0.1)
print("  float32 sum of 0.1 x 10000 =", acc)

print("Example 1.4")
x = 1e8
print("  naive  :", math.sqrt(x + 1) - math.sqrt(x))
print("  stable :", 1 / (math.sqrt(x + 1) + math.sqrt(x)))

print("Example 1.5")
print("  kappa of ln at 1.001 =", 1 / abs(math.log(1.001)))

print("Example 1.6")
xs = 1e9 + np.array([4.0, 7, 13, 16])
print("  textbook :", floating.variance_textbook(xs))
print("  two-pass :", floating.variance_two_pass(xs))
print("  Welford  :", floating.variance_welford(xs))

print("Example 1.7")
z = [1000, 1001, 1002]
print("  logsumexp =", floating.logsumexp(z))
print("  softmax   =", floating.softmax(z))

print("Example 1.8")
E = np.array([4e-3, 1e-3, 2.5e-4])
h = np.array([0.1, 0.05, 0.025])
print("  observed orders:", np.log(E[:-1] / E[1:]) / np.log(h[:-1] / h[1:]))

print("Example 1.9")
y = 1.4142
print(f"  forward={abs(y - math.sqrt(2)):.3e}  backward={abs(2 - y * y):.3e}")
