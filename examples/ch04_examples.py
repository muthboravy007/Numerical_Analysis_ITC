"""Chapter 4 worked examples — run:  python examples/ch04_examples.py"""
import math
import os
import sys

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import integration as ig  # noqa: E402

np.set_printoptions(precision=6, suppress=True)

print("Example 4.1 — acceleration from tabulated speeds")
t = np.arange(0, 9.0); v = np.array([0.0, 3.1, 6.4, 9.6, 12.5, 15.1, 17.2, 18.9, 20.1])  # m/s
h = 1.0
a_c = (v[2:] - v[:-2]) / (2 * h)
a0 = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * h); a8 = (3 * v[8] - 4 * v[7] + v[6]) / (2 * h)
a4_5 = (v[2] - 8 * v[3] + 8 * v[5] - v[6]) / (12 * h)
print("  central a(1..7):", a_c.round(3), " endpoint a(0)", a0, " a(8)", a8, " five-point a(4)", round(a4_5, 4))
print("Example 4.2 — distance from speed data")
T = ig.trapezoid_data(t, v); S = h / 3 * (v[0] + 4 * v[1:-1:2].sum() + 2 * v[2:-1:2].sum() + v[-1])
print(f"  trapezoid {T:.4f} m   Simpson {S:.4f} m   difference {S - T:.4f}")
print("Example 4.3 — Romberg for Phi(1.5)")
f = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
val, R = ig.romberg(f, 0, 1.5, levels=5)
print("  Romberg table\n", R, "\n  0.5 + R[4,4] =", 0.5 + val, " exact", stats.norm.cdf(1.5), " err", abs(0.5 + val - stats.norm.cdf(1.5)))
print("Example 4.4 — Gauss-Legendre for E[ln(1+X)], X ~ U(0,2)")
exact = (3 * np.log(3) - 2) / 2
for n in (1, 2, 3, 4, 5):
    val = ig.gauss_legendre(lambda x: np.log1p(x) / 2, 0, 2, n)
    sim = ig.simpson(lambda x: np.log1p(x) / 2, 0, 2, 2 * ((n + 1) // 2)) if n >= 2 else float("nan")
    print(f"  n={n}: Gauss {val:.10f} err {abs(val - exact):.2e}   (Simpson with {2 * ((n + 1) // 2)} intervals: err {abs(sim - exact):.2e})")
print("  exact", exact)
print("Example 4.5 — adaptive Simpson on a sharp peak")
g = lambda x: 1 / (1e-4 + (x - 0.3) ** 2)
ex = (np.arctan(0.7 / 0.01) + np.arctan(0.3 / 0.01)) / 0.01
for tol in (1e-4, 1e-8):
    val, nev = ig.adaptive_simpson(g, 0, 1, tol=tol)
    print(f"  tol {tol:.0e}: value {val:.8f} error {abs(val - ex):.2e} evaluations {nev}")
for n in (100, 1000, 10000):
    print(f"  composite Simpson n={n}: error {abs(ig.simpson(g, 0, 1, n) - ex):.2e}")
print("  exact", ex)
print("Example 4.6 — bivariate normal probability over a square")
rho = 0.5
pdf = lambda x, y: np.exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho ** 2))) / (2 * np.pi * np.sqrt(1 - rho ** 2))
ref = stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([1, 1]) - 2 * stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([-1, 1]) + stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([-1, -1])
for n in (2, 4, 8):
    print(f"  Gauss-Legendre {n}x{n}: {ig.gauss_legendre_2d(pdf, -1, 1, -1, 1, n):.8f}")
print(f"  Simpson 8x8: {ig.simpson_double(pdf, -1, 1, -1, 1, 8, 8):.8f}")
rng = np.random.default_rng(0); Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 10 ** 5)
p = np.mean((np.abs(Z) <= 1).all(1)); print(f"  Monte Carlo 1e5: {p:.4f} +- {np.sqrt(p * (1 - p) / 1e5):.4f}   reference (scipy CDF) {ref:.8f}")
print("Example 4.7 — Gauss-Hermite for a lognormal mean and a call payoff")
mu, s = 0.05, 0.2
for n in (2, 3, 5, 10):
    m = ig.gauss_hermite_expectation(np.exp, mu, s, n)
    c = ig.gauss_hermite_expectation(lambda z: np.maximum(np.exp(z) - 1.0, 0), mu, s, n)
    print(f"  n={n:2d}: E[e^X] {m:.10f} (err {abs(m - np.exp(mu + s * s / 2)):.1e})   E[(e^X-1)+] {c:.6f}")
d1 = (mu + s * s) / s; d2 = mu / s
print("  exact E[e^X] =", np.exp(mu + s * s / 2), " exact E[(e^X-1)+] =", np.exp(mu + s * s / 2) * stats.norm.cdf(d1) - stats.norm.cdf(d2))
