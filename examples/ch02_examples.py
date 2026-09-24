"""Chapter 2 worked examples — run:  python examples/ch02_examples.py"""

import math
import os
import sys

import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import roots  # noqa: E402

f = lambda x: x ** 3 - 2 * x - 5
df = lambda x: 3 * x ** 2 - 2
r = 2.0945514815423265

print("Example 2.1 — bisection")
x, hist = roots.bisection(f, 2, 3, tol=1e-6)
for k, m in enumerate(hist[:6], 1):
    print(f"  {k}: m={m:.6f} f(m)={f(m):+.6f}")
print("  iterations:", len(hist), "predicted:", roots.bisection_iterations(2, 3, 1e-6))

print("Example 2.2 — Newton")
x, hist = roots.newton(f, df, 2.0)
e = [abs(v - r) for v in hist]
for k in range(len(hist) - 1):
    ratio = e[k] / e[k - 1] ** 2 if k and e[k - 1] else float("nan")
    print(f"  {k}: {hist[k]:.16f} err={e[k]:.2e} ratio={ratio:.3f}")
print("  theory f''/(2f') =", 6 * r / (2 * df(r)))

print("Example 2.3 — fixed point forms")
for name, gp in (("a", lambda x: 1.5 * x ** 2),
                 ("b", lambda x: 2 / 3 * (2 * x + 5) ** (-2 / 3)),
                 ("c", lambda x: -10 * x / (x ** 2 - 2) ** 2)):
    print(f"  |g_{name}'(r)| = {abs(gp(r)):.3f}")
x, hist = roots.fixed_point(lambda x: (2 * x + 5) ** (1 / 3), 2.0, tol=1e-8)
print("  (b) iterates:", [round(v, 5) for v in hist[:6]], "... n =", len(hist) - 1)

print("Example 2.4 — IRR by secant")
npv = lambda r: -1000 + 300 / (1 + r) + 400 / (1 + r) ** 2 + 500 / (1 + r) ** 3
x, hist = roots.secant(npv, 0.0, 0.2)
print("  secant:", [f"{v:.8f}" for v in hist], "IRR =", f"{x:.4%}")
print("  bisection iterations:", len(roots.bisection(npv, 0, 0.2, 1e-6)[1]))

print("Example 2.5 — normal quantile")
x, hist = roots.newton(lambda z: norm.cdf(z) - 0.95, norm.pdf, 1.5)
print("  ", [f"{v:.7f}" for v in hist], "scipy:", norm.ppf(0.95))

print("Example 2.6 — Gamma MLE")
data = np.array([2.3, 1.1, 4.5, 3.2, 0.8, 2.9, 5.1, 1.7, 2.2, 3.6])
s = np.log(data.mean()) - np.log(data).mean()
a0 = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
a, hist = roots.newton(lambda a: np.log(a) - digamma(a) - s,
                       lambda a: 1 / a - polygamma(1, a), a0)
print(f"  s={s:.6f} a0={a0:.5f} iterates={[round(float(v), 7) for v in hist]}")
print(f"  alpha={a:.4f} theta={data.mean() / a:.4f}")

print("Example 2.7 — double root")
x, hist = roots.newton(lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1), 2.0)
print("  plain Newton iterations:", len(hist) - 1)
x1 = 2.0 - 2 * (2.0 - 1) ** 2 / (2 * (2.0 - 1))
print("  modified (m=2) after one step:", x1)

print("Example 2.8 — Newton system")
F = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 4, v[0] * v[1] - 1])
J = lambda v: np.array([[2 * v[0], 2 * v[1]], [v[1], v[0]]])
x, hist = roots.newton_system(F, J, [2.0, 0.5])
for v in hist:
    print("  ", np.round(v, 6))
print("  exact:", math.sqrt(2 + math.sqrt(3)), math.sqrt(2 - math.sqrt(3)))
