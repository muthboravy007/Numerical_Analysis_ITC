"""Numerical answers for exercises/ch13_exercises.md."""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import regression as rg  # noqa: E402

np.set_printoptions(precision=6, suppress=True)

# ---------- B ----------
x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8]); y = np.array([3.1, 4.9, 6.2, 7.8, 9.1, 11.2, 12.4, 14.3])
s = rg.simple_regression(x, y); n = len(x)
tq = stats.t.ppf(0.975, n - 2)
x0 = 5.5; yhat0 = s["b0"] + s["b1"] * x0
se_mean = s["s"] * np.sqrt(1 / n + (x0 - x.mean())**2 / s["Sxx"]); se_pred = s["s"] * np.sqrt(1 + 1 / n + (x0 - x.mean())**2 / s["Sxx"])
print("B1 xbar", x.mean(), "ybar", y.mean(), {k: round(float(v), 6) for k, v in s.items()})
print(f"   t(b1) {s['b1'] / s['se_b1']:.3f}  95% CI b1 [{s['b1'] - tq * s['se_b1']:.4f}, {s['b1'] + tq * s['se_b1']:.4f}]  t quantile {tq:.4f}")
print(f"   at x0=5.5: yhat {yhat0:.4f}  CI mean [{yhat0 - tq * se_mean:.4f}, {yhat0 + tq * se_mean:.4f}]  PI [{yhat0 - tq * se_pred:.4f}, {yhat0 + tq * se_pred:.4f}]")

X2 = np.array([[1.0, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5]]); y2 = np.array([5.1, 6.2, 10.8, 12.1, 17.2, 17.9])
A = np.column_stack([np.ones(6), X2])
print("B2 X^T X\n", A.T @ A, "\n X^T y", A.T @ y2, "\n beta (normal eq)", np.linalg.solve(A.T @ A, A.T @ y2))
res = rg.ols(X2, y2)
print("   OLS via QR:", res.beta, " residuals", res.residuals, " R2", res.r2, " sigma2", res.sigma2, " se", res.se)
Q, R = np.linalg.qr(A)
print("   R\n", R)

x3 = np.array([1.0, 2, 3, 4, 5, 1, 2, 3, 4, 5]); d3 = np.array([0.0] * 5 + [1.0] * 5)
y3 = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 4.0, 7.1, 9.8, 13.2, 16.0])
r3 = rg.ols(np.column_stack([x3, d3, x3 * d3]), y3)
print("B3 interaction model beta", r3.beta, " se", r3.se, " t", r3.t, " p", r3.p)
print("   group 0 line: b0, b1 =", r3.beta[0], r3.beta[1], " group 1 line:", r3.beta[0] + r3.beta[2], r3.beta[1] + r3.beta[3])

x4 = np.array([1.0, 2, 3, 4, 5, 6, 7, 20]); y4 = np.array([2.2, 4.1, 5.8, 8.3, 9.9, 12.1, 13.8, 22.0])
r4 = rg.ols(x4, y4); p4 = 2
cook = r4.residuals**2 / (p4 * r4.sigma2) * r4.leverage / (1 - r4.leverage)**2
stud = r4.residuals / np.sqrt(r4.sigma2 * (1 - r4.leverage))
print("B4 fit with point 8:", r4.beta, " leverage", r4.leverage.round(4), "\n   residuals", r4.residuals.round(4), "\n   internally studentised", stud.round(3), "\n   Cook's D", cook.round(4))
r4b = rg.ols(x4[:-1], y4[:-1])
print("   fit without point 8:", r4b.beta, "  prediction at x=20 from the reduced fit", r4b.beta[0] + 20 * r4b.beta[1])
print("   LOOCV via PRESS", np.mean((r4.residuals / (1 - r4.leverage))**2))

rho = 0.95
C = np.array([[1, rho], [rho, 1.0]])
print("B5 VIF = 1/(1-rho^2) =", 1 / (1 - rho**2), " eig of correlation", np.linalg.eigvalsh(C), " cond", np.linalg.cond(C))
bhat = np.array([2.0, 0.5])  # suppose X^T y / n gives these OLS coefficients
Xty = C @ bhat
for lam in (0.0, 0.1, 1.0, 10.0):
    br = np.linalg.solve(C + lam * np.eye(2), Xty)
    print(f"   lambda {lam}: ridge beta {br.round(4)}  norm {np.linalg.norm(br):.4f}  shrink factors d/(d+lam) {(np.linalg.eigvalsh(C) / (np.linalg.eigvalsh(C) + lam)).round(4)}")

bols = np.array([3.0, -1.2, 0.4, 0.05])
for lam in (0.1, 0.5, 1.5, 3.5):
    print(f"B6 lambda {lam}: lasso {rg.soft_threshold(bols, lam)}  ridge (orthonormal) {(bols / (1 + lam)).round(4)}")

x7 = np.array([0.0, 1, 2, 3]); y7 = np.array([2.0, 3.0, 5.2, 8.1])
r7 = lambda b: b[0] * np.exp(b[1] * x7) - y7
J7 = lambda b: np.column_stack([np.exp(b[1] * x7), b[0] * x7 * np.exp(b[1] * x7)])
b0 = np.array([2.0, 0.4])
Jb = J7(b0); rb = r7(b0); step = np.linalg.solve(Jb.T @ Jb, -Jb.T @ rb)
print("B7 J(b0)\n", Jb, "\n r(b0)", rb, " step", step, " b1", b0 + step, " SSE", np.sum(rb**2), "->", np.sum(r7(b0 + step)**2))
bg, hist = rg.gauss_newton(r7, J7, b0)
print("   converged", bg, "iterations", len(hist) - 1, "SSE", np.sum(r7(bg)**2))
ll = np.polyfit(x7, np.log(y7), 1)
print("   linearised fit a, b =", np.exp(ll[1]), ll[0], " SSE", np.sum((np.exp(ll[1]) * np.exp(ll[0] * x7) - y7)**2))

x8 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]); y8 = np.array([0.0, 0, 1, 0, 1, 1])
X8 = np.column_stack([np.ones(6), x8])
b = np.zeros(2); mu = 1 / (1 + np.exp(-X8 @ b)); w = mu * (1 - mu); z = X8 @ b + (y8 - mu) / w
print("B8 step 1: mu", mu, " w", w, " z", z, "\n   X^T W X", X8.T @ (w[:, None] * X8), " X^T W z", X8.T @ (w * z))
b1 = np.linalg.solve(X8.T @ (w[:, None] * X8), X8.T @ (w * z))
print("   beta1", b1)
bf, cov, hist = rg.glm_irls(X8, y8, "logistic")
print("   IRLS iterates", [np.round(h, 6) for h in hist], "\n   final", bf, " se", np.sqrt(np.diag(cov)), " odds ratio per unit x", np.exp(bf[1]), " P(y=1|x=1.75)", 1 / (1 + np.exp(-(bf[0] + 1.75 * bf[1]))))

# ---------- C ----------
print("C1 Longley data vs NIST certified values")
import statsmodels.api as sm
lg = sm.datasets.longley.load_pandas()
XL = lg.exog.values; yL = lg.endog.values
cert = np.array([-3482258.63459582, 15.0618722713733, -0.358191792925910e-1, -2.02022980381683, -1.03322686717359, -0.511041056535807e-1, 1829.15146461355])
AL = np.column_stack([np.ones(len(yL)), XL])
print("   cond(X) =", f"{np.linalg.cond(AL):.2e}", " cond(X^T X) =", f"{np.linalg.cond(AL.T @ AL):.2e}")
b_ne = np.linalg.solve(AL.T @ AL, AL.T @ yL)
b_qr = rg.ols(XL, yL).beta
b_svd = np.linalg.lstsq(AL, yL, rcond=None)[0]
b_ne32 = np.linalg.solve((AL.T @ AL).astype(np.float32), (AL.T @ yL).astype(np.float32)).astype(float)
Q32, R32 = np.linalg.qr(AL.astype(np.float32)); b_qr32 = np.linalg.solve(R32, Q32.T @ yL.astype(np.float32)).astype(float)
lre = lambda b: -np.log10(np.abs(b - cert) / np.abs(cert))
for name, b in (("normal eq", b_ne), ("QR", b_qr), ("SVD", b_svd), ("normal eq float32", b_ne32), ("QR float32", b_qr32)):
    print(f"   {name:18s}: min correct digits (LRE) {lre(b).min():.1f}  per coefficient {np.round(lre(b), 1)}")
Xs = (XL - XL.mean(0)) / XL.std(0)
print("   after standardising: cond", f"{np.linalg.cond(np.column_stack([np.ones(16), Xs])):.2e}", " VIFs", rg.vif(XL).round(1))

print("C2 lasso path by coordinate descent vs sklearn (diabetes)")
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
Xd, yd = load_diabetes(return_X_y=True)
names = load_diabetes().feature_names
Xs = (Xd - Xd.mean(0)) / Xd.std(0); yc = yd - yd.mean()
for lam in (20.0, 5.0, 1.0, 0.1):
    bcd, its = rg.lasso_cd(Xs, yc, lam)
    bsk = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100000).fit(Xs, yc).coef_
    print(f"   lambda {lam}: nonzero {int(np.sum(bcd != 0))}, CD sweeps {its}, max |ours - sklearn| {np.abs(bcd - bsk).max():.1e}, active {[names[j] for j in np.nonzero(bcd)[0]]}")
lam_max = np.abs(Xs.T @ yc).max() / len(yc)
print(f"   lambda_max = max|X^T y|/n = {lam_max:.4f} (all coefficients zero above this)")

print("C3 heteroscedasticity: classical vs HC3 vs bootstrap SE")
rng = np.random.default_rng(0)
n = 200; xh = rng.uniform(0, 10, n); yh = 1 + 0.5 * xh + (0.2 + 0.3 * xh) * rng.standard_normal(n)
Xh = np.column_stack([np.ones(n), xh]); fit = rg.ols(xh, yh)
XtXi = np.linalg.inv(Xh.T @ Xh); e = fit.residuals; hlev = fit.leverage
hc3 = XtXi @ (Xh.T * (e**2 / (1 - hlev)**2)) @ Xh @ XtXi
boot = []
for _ in range(2000):
    idx = rng.integers(0, n, n); boot.append(np.linalg.lstsq(Xh[idx], yh[idx], rcond=None)[0])
boot = np.array(boot)
mc = []
for _ in range(2000):
    yy = 1 + 0.5 * xh + (0.2 + 0.3 * xh) * rng.standard_normal(n); mc.append(np.linalg.lstsq(Xh, yy, rcond=None)[0])
print("   slope SE: classical", fit.se[1].round(5), " HC3", np.sqrt(hc3[1, 1]).round(5), " pairs bootstrap", boot[:, 1].std().round(5), " true (Monte Carlo over new noise)", np.array(mc)[:, 1].std().round(5))
print("   intercept SE: classical", fit.se[0].round(5), " HC3", np.sqrt(hc3[0, 0]).round(5), " bootstrap", boot[:, 0].std().round(5), " true", np.array(mc)[:, 0].std().round(5))
wts = 1 / (0.2 + 0.3 * xh)**2
bw = rg.wls(xh, yh, wts)
mcw = []
for _ in range(2000):
    yy = 1 + 0.5 * xh + (0.2 + 0.3 * xh) * rng.standard_normal(n); mcw.append(rg.wls(xh, yy, wts))
print("   WLS with known weights: beta", bw.round(4), " true SE of slope", np.array(mcw)[:, 1].std().round(5), "(efficiency gain vs OLS", (np.array(mc)[:, 1].var() / np.array(mcw)[:, 1].var()).round(2), "x)")

print("C4 Gauss-Newton vs Levenberg-Marquardt on a sum of two exponentials")
t = np.linspace(0, 5, 40); rng = np.random.default_rng(1)
ytrue = 3 * np.exp(-0.4 * t) + 2 * np.exp(-2.5 * t); yo = ytrue + 0.02 * rng.standard_normal(len(t))
rr = lambda b: b[0] * np.exp(-b[1] * t) + b[2] * np.exp(-b[3] * t) - yo
JJ = lambda b: np.column_stack([np.exp(-b[1] * t), -b[0] * t * np.exp(-b[1] * t), np.exp(-b[3] * t), -b[2] * t * np.exp(-b[3] * t)])
for start in ([1.0, 1.0, 1.0, 3.0], [1.0, 0.1, 1.0, 0.2], [5.0, 1.0, 0.5, 1.1]):
    bgn = np.array(start, float); hg = [bgn]
    with np.errstate(all="ignore"):
        for _ in range(100):  # plain Gauss-Newton, stopped as soon as it blows up
            Jb_, rb_ = JJ(bgn), rr(bgn)
            if not (np.all(np.isfinite(Jb_)) and np.all(np.isfinite(rb_))) or np.abs(bgn).max() > 1e6:
                break
            dp = np.linalg.lstsq(Jb_, -rb_, rcond=None)[0]; bgn = bgn + dp; hg.append(bgn)
            if np.linalg.norm(dp) < 1e-10 * max(1, np.linalg.norm(bgn)):
                break
        sse_gn = np.sum(rr(bgn)**2)
    blm, hl = rg.levenberg_marquardt(rr, JJ, start)
    print(f"   start {start}: GN {len(hg)-1} its -> {np.round(bgn, 4)} SSE {sse_gn:.3g} | LM {len(hl)-1} accepted steps -> {np.round(blm, 4)} SSE {np.sum(rr(blm)**2):.5f}")

# ---------- D ----------
print("D1 diabetes: OLS vs ridge vs lasso (train/test split)")
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=0)
mu_, sd_ = Xtr.mean(0), Xtr.std(0); Ztr = (Xtr - mu_) / sd_; Zte = (Xte - mu_) / sd_
kf = KFold(5, shuffle=True, random_state=0)
ols_m = LinearRegression().fit(Ztr, ytr)
rid = RidgeCV(alphas=np.logspace(-3, 3, 61), cv=kf).fit(Ztr, ytr)
las = LassoCV(cv=kf, random_state=0, max_iter=100000).fit(Ztr, ytr)
for name, m in (("OLS", ols_m), ("ridge", rid), ("lasso", las)):
    mse = np.mean((m.predict(Zte) - yte)**2)
    extra = f" alpha {m.alpha_:.3g}" if hasattr(m, "alpha_") else ""
    print(f"   {name}: test MSE {mse:.1f}  R2 {1 - mse / np.var(yte):.4f}  nonzero {int(np.sum(np.abs(m.coef_) > 1e-8))}{extra}")
print("   lasso coefficients:", dict(zip(names, np.round(las.coef_, 2))))
print("   OLS coefficients:  ", dict(zip(names, np.round(ols_m.coef_, 2))))
print("   VIF (train):", dict(zip(names, rg.vif(Ztr).round(1))))

print("D2 Poisson GLM with exposure (insurance claims)")
rng = np.random.default_rng(2)
n = 5000; age = rng.uniform(18, 80, n); urban = rng.integers(0, 2, n); expo = rng.uniform(0.2, 1.0, n)
eta = -2.0 + 0.02 * (age - 50) ** 2 / 100 + 0.4 * urban
claims = rng.poisson(expo * np.exp(eta))
Xp = np.column_stack([np.ones(n), (age - 50)**2 / 100, urban])
bp, covp, hp = rg.glm_irls(Xp, claims, "poisson", offset=np.log(expo))
smf = sm.GLM(claims, Xp, family=sm.families.Poisson(), offset=np.log(expo)).fit()
print("   IRLS beta", bp.round(5), "se", np.sqrt(np.diag(covp)).round(5), "iterations", len(hp) - 1, "\n   statsmodels", smf.params.round(5), smf.bse.round(5))
print("   rate ratio urban/rural", np.exp(bp[2]).round(4), "95% CI", np.exp(bp[2] + np.array([-1.96, 1.96]) * np.sqrt(covp[2, 2])).round(4))
bnaive, _, _ = rg.glm_irls(Xp, claims, "poisson")
print("   ignoring exposure: intercept", bnaive[0].round(4), "vs", bp[0].round(4), " (mean exposure", expo.mean().round(3), ", log", np.log(expo.mean()).round(4), ")")
pearson = np.sum((claims - expo * np.exp(Xp @ bp))**2 / (expo * np.exp(Xp @ bp))) / (n - 3)
print("   Pearson dispersion", round(pearson, 4))

print("D3 logistic regression on breast cancer (3 features)")
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
bc = load_breast_cancer(); feats = ["mean radius", "mean texture", "mean smoothness"]
cols = [list(bc.feature_names).index(f) for f in feats]
Xb = bc.data[:, cols]; yb = (bc.target == 0).astype(float)  # 1 = malignant
Zb = (Xb - Xb.mean(0)) / Xb.std(0); Xb1 = np.column_stack([np.ones(len(yb)), Zb])
bb, covb, hb = rg.glm_irls(Xb1, yb, "logistic")
pb = 1 / (1 + np.exp(-Xb1 @ bb))
sml = sm.Logit(yb, Xb1).fit(disp=0)
print("   IRLS its", len(hb) - 1, "beta", bb.round(4), "se", np.sqrt(np.diag(covb)).round(4), " statsmodels", sml.params.round(4))
print("   odds ratios per SD:", dict(zip(feats, np.exp(bb[1:]).round(3))), " AUC (in-sample)", round(roc_auc_score(yb, pb), 4), " accuracy", round(np.mean((pb > 0.5) == yb), 4))
bins = np.linspace(0, 1, 6); idx = np.digitize(pb, bins[1:-1])
print("   calibration (mean predicted vs observed per quintile of p):", [(round(pb[idx == k].mean(), 3), round(yb[idx == k].mean(), 3)) for k in range(5)])

print("D4 robust regression with contamination")
rng = np.random.default_rng(3)
n = 100; xr = rng.uniform(0, 10, n); yr = 2 + 1.5 * xr + rng.standard_normal(n)
for frac in (0.0, 0.05, 0.10, 0.20):
    yc_ = yr.copy(); k = int(frac * n); bad = rng.choice(n, k, replace=False) if k else np.array([], int)
    yc_[bad] += 30  # gross outliers
    bo = rg.ols(xr, yc_).beta; bh, wh, ith = rg.huber_irls(xr, yc_)
    print(f"   {frac:.0%} outliers: OLS {bo.round(3)}  Huber {bh.round(3)} ({ith} IRLS its, min weight {wh.min():.3f})")
yc_ = yr.copy(); lev = np.argsort(xr)[-10:]; yc_[lev] -= 25
print("   10 high-leverage outliers at the largest x: OLS", rg.ols(xr, yc_).beta.round(3), " Huber", rg.huber_irls(xr, yc_)[0].round(3), " (true 2, 1.5)")
