"""Chapter 13 worked examples — run:  python examples/ch13_examples.py"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import regression as rg  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

print("Example 13.1 — advertising and sales: a complete simple-regression analysis")
tv = np.array([230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8, 66.1, 214.7, 23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3])
sales = np.array([22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6, 8.6, 17.4, 9.2, 9.7, 19.0, 22.4, 12.5, 24.4, 11.3, 14.6])
s = rg.simple_regression(tv, sales); n = len(tv)
tq = stats.t.ppf(0.975, n - 2)
print(f"  b0 {s['b0']:.4f} b1 {s['b1']:.5f} (se {s['se_b1']:.5f}, t {s['b1'] / s['se_b1']:.2f}), R2 {s['R2']:.4f}, s {s['s']:.4f}")
print(f"  95% CI slope [{s['b1'] - tq * s['se_b1']:.5f}, {s['b1'] + tq * s['se_b1']:.5f}] -> each extra 100 units of TV spend adds {100 * s['b1']:.2f} units of sales")
x0 = 100.0; y0 = s['b0'] + s['b1'] * x0; xb = tv.mean()
se_m = s['s'] * np.sqrt(1 / n + (x0 - xb) ** 2 / s['Sxx']); se_p = s['s'] * np.sqrt(1 + 1 / n + (x0 - xb) ** 2 / s['Sxx'])
print(f"  at TV=100: mean {y0:.3f} CI [{y0 - tq * se_m:.3f}, {y0 + tq * se_m:.3f}], prediction interval [{y0 - tq * se_p:.3f}, {y0 + tq * se_p:.3f}]")
res = sales - (s['b0'] + s['b1'] * tv)
print(f"  heteroscedasticity check: corr(|residual|, TV) = {np.corrcoef(np.abs(res), tv)[0, 1]:.3f}")

print("Example 13.2 — omitted-variable bias")
n = 500; ability = rng.standard_normal(n); educ = 12 + 2 * ability + rng.standard_normal(n)
wage = 5 + 1.0 * educ + 3.0 * ability + rng.standard_normal(n)
short = rg.ols(educ, wage); full = rg.ols(np.column_stack([educ, ability]), wage)
aux = rg.ols(educ, ability)
print(f"  wage ~ educ: slope {short.beta[1]:.4f} (se {short.se[1]:.4f});  wage ~ educ + ability: {full.beta[1:].round(4)} (true 1, 3)")
print(f"  bias formula: beta_ability * cov(educ, ability)/var(educ) = 3 x {aux.beta[1]:.4f} = {3 * aux.beta[1]:.4f}; observed bias {short.beta[1] - 1:.4f}")

print("Example 13.3 — polynomial regression and the bias-variance trade-off")
f = lambda x: np.sin(2 * np.pi * x)
xt = np.linspace(0, 1, 200)
for deg in (1, 3, 5, 7, 9):
    preds = []
    for rep in range(200):
        x = rng.uniform(0, 1, 25); y = f(x) + 0.3 * rng.standard_normal(25)
        C = np.polynomial.legendre.legvander(2 * x - 1, deg); b = np.linalg.lstsq(C, y, rcond=None)[0]
        preds.append(np.polynomial.legendre.legvander(2 * xt - 1, deg) @ b)
    P = np.array(preds); bias2 = np.mean((P.mean(0) - f(xt)) ** 2); var = np.mean(P.var(0))
    print(f"  degree {deg:2d}: bias^2 {bias2:.4f}  variance {var:.4f}  expected test MSE {bias2 + var + 0.09:.4f}")

print("Example 13.4 — ridge path and effective degrees of freedom")
n, p = 50, 10
Z = rng.standard_normal((n, p)); Z[:, 1] = Z[:, 0] + 0.05 * rng.standard_normal(n)
beta = np.array([1.0, 1.0, 0.5, 0, 0, 0, 0, 0, -0.5, 0]); y = Z @ beta + 0.5 * rng.standard_normal(n)
U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
for lam in (0.0, 0.1, 1.0, 10.0, 100.0):
    b = Vt.T @ ((sv / (sv ** 2 + lam)) * (U.T @ y)); df = np.sum(sv ** 2 / (sv ** 2 + lam))
    print(f"  lambda {lam:6.1f}: df {df:.3f}  b1,b2 = {b[:2].round(3)}  ||b|| {np.linalg.norm(b):.3f}")
print(f"  singular values {sv.round(3)}  -> cond {sv[0] / sv[-1]:.1f}")

print("Example 13.5 — weighted least squares for grouped (averaged) data")
groups = np.arange(1, 9.0); sizes = np.array([5, 50, 5, 50, 5, 50, 5, 50])
means = []
for g, m_ in zip(groups, sizes):
    means.append(np.mean(2 + 0.5 * g + 1.0 * rng.standard_normal(m_)))
means = np.array(means)
b_ols = rg.ols(groups, means).beta; b_wls = rg.wls(groups, means, sizes)
mc_o, mc_w = [], []
for _ in range(2000):
    mm = np.array([np.mean(2 + 0.5 * g + rng.standard_normal(m_)) for g, m_ in zip(groups, sizes)])
    mc_o.append(np.polyfit(groups, mm, 1)[0]); mc_w.append(rg.wls(groups, mm, sizes)[1])
print(f"  OLS {b_ols.round(4)}  WLS (weights = group sizes) {b_wls.round(4)}; slope SD over 2000 repetitions: OLS {np.std(mc_o):.4f}, WLS {np.std(mc_w):.4f}")

print("Example 13.6 — Michaelis-Menten enzyme kinetics: linearisation vs nonlinear LS")
Sx = np.array([0.5, 1, 2, 4, 8, 16, 32.0]); Vmax, Km = 10.0, 3.0
v = Vmax * Sx / (Km + Sx) * (1 + 0.05 * rng.standard_normal(7))
lb = np.polyfit(1 / Sx, 1 / v, 1); Vlb, Klb = 1 / lb[1], lb[0] / lb[1]
r = lambda p: p[0] * Sx / (p[1] + Sx) - v
J = lambda p: np.column_stack([Sx / (p[1] + Sx), -p[0] * Sx / (p[1] + Sx) ** 2])
pn, hn = rg.gauss_newton(r, J, [Vlb, Klb])
print(f"  Lineweaver-Burk (1/v vs 1/S): Vmax {Vlb:.3f}, Km {Klb:.3f};  Gauss-Newton ({len(hn) - 1} its): Vmax {pn[0]:.3f}, Km {pn[1]:.3f};  truth 10, 3")
print(f"  SSE: Lineweaver-Burk {np.sum(r([Vlb, Klb]) ** 2):.4f}  nonlinear {np.sum(r(pn) ** 2):.4f}")

print("Example 13.7 — Poisson regression for count data (IRLS)")
n = 300; x = rng.uniform(0, 2, n); lam_true = np.exp(0.5 + 0.8 * x); yc = rng.poisson(lam_true)
X = np.column_stack([np.ones(n), x]); bp, cov, hist = rg.glm_irls(X, yc, "poisson")
print(f"  IRLS iterates: {[h.round(4) for h in hist[:6]]} ... total {len(hist) - 1} iterations")
print(f"  beta {bp.round(4)} se {np.sqrt(np.diag(cov)).round(4)} (true 0.5, 0.8); rate ratio per unit x: {np.exp(bp[1]):.4f}")
print(f"  check: mean of y {yc.mean():.4f} = mean of fitted {np.exp(X @ bp).mean():.4f} (score equation with intercept)")
