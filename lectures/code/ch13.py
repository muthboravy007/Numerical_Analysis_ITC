"""Reproduces every numerical example in Chapter 13 (Regression and Least Squares for Data Science).
Run from the repository root:  python lectures/code/ch13.py"""

import math
import os
import sys

import numpy as np
from scipy import stats
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import least_squares as lsq, regression as rg  # noqa: E402

np.set_printoptions(precision=5, suppress=True, linewidth=120)

print("=== 13.1 Simple linear regression ===")
x = np.arange(1, 11.0); y = np.array([3.2, 4.8, 5.1, 7.4, 8.0, 9.9, 10.1, 12.6, 13.0, 14.8])
s = rg.simple_regression(x, y)
print("Ex 13.1.1 xbar", x.mean(), "ybar", y.mean(), "Sxx", s["Sxx"], "Sxy", s["Sxy"], "Syy", s["Syy"], "b1", s["b1"], "b0", s["b0"])
res = y - s["b0"] - s["b1"] * x
print("Ex 13.1.2 residuals", res, "SSE", s["SSE"], "s^2", s["s"] ** 2, "s", s["s"], "R2", s["R2"], "r", s["r"])
t = s["b1"] / s["se_b1"]; q = stats.t.ppf(0.975, 8)
print("Ex 13.1.3 se_b1", s["se_b1"], "t", t, "p", 2 * stats.t.sf(abs(t), 8), "t_crit", q, "CI", s["b1"] - q * s["se_b1"], s["b1"] + q * s["se_b1"], "se_b0", s["se_b0"])
x0 = 12.0; yhat = s["b0"] + s["b1"] * x0
se_mean = s["s"] * math.sqrt(1 / 10 + (x0 - x.mean()) ** 2 / s["Sxx"]); se_pred = s["s"] * math.sqrt(1 + 1 / 10 + (x0 - x.mean()) ** 2 / s["Sxx"])
print("Ex 13.1.4 x0=12 yhat", yhat, "CI mean", yhat - q * se_mean, yhat + q * se_mean, "PI", yhat - q * se_pred, yhat + q * se_pred)
ax = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5.0])
ays = [np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
       np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
       np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])]
x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8.0]); y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])
for k, (xa, ya) in enumerate([(ax, ays[0]), (ax, ays[1]), (ax, ays[2]), (x4, y4)], 1):
    sa = rg.simple_regression(xa, ya); ra = ya - sa["b0"] - sa["b1"] * xa
    print(f"Ex 13.1.5 Anscombe {k}: b0={sa['b0']:.3f} b1={sa['b1']:.3f} R2={sa['R2']:.3f} max|resid|={np.max(np.abs(ra)):.2f} lag-corr of sorted resid={np.corrcoef(ra[np.argsort(xa)][:-1], ra[np.argsort(xa)][1:])[0, 1]:.2f}")
xo = np.r_[x, 25.0]; yo = np.r_[y, 12.0]
so = rg.simple_regression(xo, yo); fo = rg.ols(xo, yo)
print("Ex 13.1.6 with high-leverage point (25,12): b1", so["b1"], "b0", so["b0"], "leverage of new point", fo.leverage[-1], "mean leverage", 2 / 11)

print("\n=== 13.2 Multiple regression ===")
size = np.array([1.2, 1.5, 1.7, 2.0, 2.2, 2.5, 2.8, 3.0]); age = np.array([20, 15, 18, 10, 8, 12, 5, 3.0])
price = np.array([12.5, 15.8, 16.2, 21.0, 23.1, 24.9, 29.8, 32.0])
X = np.column_stack([np.ones(8), size, age])
print("Ex 13.2.1 X^TX\n", X.T @ X, "\n X^Ty", X.T @ price, "\n beta", np.linalg.solve(X.T @ X, X.T @ price))
f = rg.ols(np.column_stack([size, age]), price)
print("Ex 13.2.2 leverages", f.leverage, "sum", f.leverage.sum())
print("Ex 13.2.3 se", f.se, "t", f.t, "p", f.p, "sigma", math.sqrt(f.sigma2), "R2", f.r2, "adjR2", f.r2_adj)
SSR = np.sum((f.fitted - price.mean()) ** 2); SSE = np.sum(f.residuals ** 2); F = (SSR / 2) / (SSE / 5)
print("   F", F, "p", stats.f.sf(F, 2, 5))
print("   simple regression on age only: b_age", rg.simple_regression(age, price)["b1"], " corr(size, age)", np.corrcoef(size, age)[0, 1])
rng = np.random.default_rng(0)
noise = rng.standard_normal(8)
f2 = rg.ols(np.column_stack([size, age, noise]), price)
print("Ex 13.2.5 add noise feature: R2", f.r2, "->", f2.r2, "adjR2", f.r2_adj, "->", f2.r2_adj)
Xs = np.column_stack([np.ones(30), np.linspace(0, 1, 30), np.linspace(0, 1, 30) ** 2]); btrue = np.array([1.0, 2.0, -1.0]); sg = 0.5
B = np.array([np.linalg.lstsq(Xs, Xs @ btrue + sg * rng.standard_normal(30), rcond=None)[0] for _ in range(20000)])
print("Ex 13.2.6 Gauss-Markov sim mean", B.mean(0), "sim var", B.var(0), "theory var", np.diag(sg ** 2 * np.linalg.inv(Xs.T @ Xs)))

print("\n=== 13.3 Numerical methods for LS ===")
bNE = lsq.normal_equations(X, price); bQR = lsq.qr_least_squares(X, price); bSVD = lsq.svd_least_squares(X, price)
print("Ex 13.3.1", bNE, bQR, bSVD, "cond X", np.linalg.cond(X), "cond XtX", np.linalg.cond(X.T @ X))
tt = np.linspace(0, 1, 40); Vd = np.vander(tt, 11, increasing=True); ctrue = np.ones(11); yy = Vd @ ctrue
for name, sol in (("normal eq (np.solve)", np.linalg.solve(Vd.T @ Vd, Vd.T @ yy)), ("Householder QR", lsq.qr_least_squares(Vd, yy)), ("SVD", lsq.svd_least_squares(Vd, yy, rcond=1e-15))):
    print(f"Ex 13.3.2 degree 10: {name}: coefficient error {np.max(np.abs(sol - ctrue)):.2e}")
print("   cond(A)", np.linalg.cond(Vd), "cond(A^TA)", np.linalg.cond(Vd.T @ Vd))
A = np.array([[3.0, -6], [4, -8], [0, 1]]); bb = np.array([-1.0, 7, 2])
Q, R = lsq.gram_schmidt(A, modified=False); print("Ex 13.3.3 GS Q\n", Q, "\n R\n", R, "\n Q^T b", Q.T @ bb, "x", np.linalg.solve(R, Q.T @ bb))
Qh, Rh = lsq.householder_qr(A); print("   Householder R\n", Rh, "\n Q^T b", Qh.T @ bb, "residual norm", abs((Qh.T @ bb)[2]))
x1 = rng.normal(0, 1, 50); x2 = x1 + 0.05 * rng.normal(0, 1, 50); x3 = rng.normal(0, 1, 50)
print("Ex 13.3.4 VIF", rg.vif(np.column_stack([x1, x2, x3])))
yc = 1 + x1 + x3 + 0.5 * rng.standard_normal(50); fc = rg.ols(np.column_stack([x1, x2, x3]), yc)
print("   coefficients", fc.beta, "se", fc.se)
Xraw = np.column_stack([np.ones(8), size * 100, age * 365])
Xstd = np.column_stack([np.ones(8), (size - size.mean()) / size.std(), (age - age.mean()) / age.std()])
print("Ex 13.3.5 cond raw (m^2, days)", np.linalg.cond(Xraw), "standardized", np.linalg.cond(Xstd))
grp = np.array([0, 0, 1, 1, 2, 2, 2, 1]); D = np.eye(3)[grp]; Xd = np.column_stack([np.ones(8), D])
print("Ex 13.3.6 dummy trap rank", np.linalg.matrix_rank(Xd), "of", Xd.shape[1], "sv", np.linalg.svd(Xd, compute_uv=False),
      "min-norm", np.linalg.lstsq(Xd, price, rcond=None)[0], "drop one", np.linalg.lstsq(Xd[:, :3], price, rcond=None)[0])

print("\n=== 13.4 Polynomial and basis regression ===")
xs = np.sort(rng.uniform(-1, 1, 60)); ys = np.exp(xs) * np.sin(3 * xs) + 0.2 * rng.standard_normal(60)
n = len(xs)
for d in (1, 2, 3, 4, 5, 6, 8, 10, 12):
    P = np.polynomial.Legendre.fit(xs, ys, d); rss = np.sum((P(xs) - ys) ** 2)
    aic = n * math.log(rss / n) + 2 * (d + 1); bic = n * math.log(rss / n) + math.log(n) * (d + 1)
    cv = rg.kfold_cv_mse(lambda a, b, c: np.polynomial.Legendre.fit(a, b, d)(c), xs, ys, k=10)
    print(f"Ex 13.4.1 degree {d:2d}: RSS {rss:7.3f} AIC {aic:8.2f} BIC {bic:8.2f} 10-fold CV {cv:.4f}")
for d in (5, 10, 15):
    print(f"Ex 13.4.2 degree {d}: cond monomial {np.linalg.cond(np.vander(xs, d + 1)):.2e} cond Legendre {np.linalg.cond(np.polynomial.legendre.legvander(xs, d)):.2e}")
knots = np.array([-0.5, 0.0, 0.5])
Bs = np.column_stack([np.ones(n), xs, xs ** 2, xs ** 3] + [np.maximum(xs - kk, 0) ** 3 for kk in knots])
bs_ = np.linalg.lstsq(Bs, ys, rcond=None)[0]
tt = np.linspace(-1, 1, 401); Bt = np.column_stack([np.ones_like(tt), tt, tt ** 2, tt ** 3] + [np.maximum(tt - kk, 0) ** 3 for kk in knots])
truth = np.exp(tt) * np.sin(3 * tt)
print("Ex 13.4.3 cubic regression spline (3 knots, 7 params) max dev from truth", np.max(np.abs(Bt @ bs_ - truth)),
      "| degree-6 poly", np.max(np.abs(np.polynomial.Legendre.fit(xs, ys, 6)(tt) - truth)), "| cond truncated-power basis", np.linalg.cond(Bs))
edu = np.array([12, 16, 12, 18, 14, 16, 12, 20.0]); female = np.array([0, 1, 1, 0, 1, 0, 0, 1.0]); wage = np.array([15, 24, 13, 30, 18, 25, 16, 31.0])
Xi = np.column_stack([np.ones(8), edu, female, edu * female]); bi = np.linalg.lstsq(Xi, wage, rcond=None)[0]
print("Ex 13.4.4 interaction model coefficients", bi, "slope men", bi[1], "slope women", bi[1] + bi[3])
for d in (3, 12):
    P = np.polynomial.Legendre.fit(xs, ys, d, domain=[-1, 1])
    print(f"Ex 13.4.5 degree {d}: prediction at x=1.2: {P(1.2):.3f}  x=1.5: {P(1.5):.3f}  truth {math.exp(1.2) * math.sin(3.6):.3f}, {math.exp(1.5) * math.sin(4.5):.3f}")

print("\n=== 13.5 Regularisation ===")
Xr = np.column_stack([x1, x2, x3]); Xr = (Xr - Xr.mean(0)) / Xr.std(0); yr = yc - yc.mean()
U, sv, Vt = np.linalg.svd(Xr, full_matrices=False); print("Ex 13.5.1 singular values", sv)
for lam in (0.0, 0.1, 1.0, 10.0, 100.0):
    print(f"   lambda={lam}: filter {np.round(sv ** 2 / (sv ** 2 + lam), 4)} beta {lsq.ridge(Xr, yr, lam) if lam else lsq.svd_least_squares(Xr, yr)}")
bt = np.array([1.0, 0.0, 1.0]); reps = []
for lam in (0.0, 1.0, 5.0, 25.0):
    ests = []
    for r_ in range(2000):
        ysim = Xr @ bt + 0.5 * rng.standard_normal(50); ests.append(lsq.ridge(Xr, ysim, lam) if lam else np.linalg.lstsq(Xr, ysim, rcond=None)[0])
    ests = np.array(ests); bias2 = np.sum((ests.mean(0) - bt) ** 2); var = np.sum(ests.var(0))
    print(f"Ex 13.5.2 lambda={lam}: bias^2 {bias2:.4f} variance {var:.4f} MSE {bias2 + var:.4f}")
for lam in (0.01, 0.1, 1.0, 3.0, 10.0, 30.0):
    H = Xr @ np.linalg.solve(Xr.T @ Xr + lam * np.eye(3), Xr.T); r_ = yr - H @ yr
    loo = np.mean((r_ / (1 - np.diag(H))) ** 2); gcv = np.mean(r_ ** 2) / (1 - np.trace(H) / 50) ** 2
    print(f"Ex 13.5.3 lambda={lam}: df {np.trace(H):.3f} LOOCV {loo:.4f} GCV {gcv:.4f}")
Xl = rng.standard_normal((100, 8)); bl = np.array([3, -2, 0, 0, 1.5, 0, 0, 0.0]); yl = Xl @ bl + 0.5 * rng.standard_normal(100)
Xl = (Xl - Xl.mean(0)) / Xl.std(0); yl = yl - yl.mean()
for lam in (0.01, 0.1, 0.5, 1.0, 2.0):
    b_, its = rg.lasso_cd(Xl, yl, lam)
    print(f"Ex 13.5.4 lasso lambda={lam}: {np.round(b_, 3)} nonzeros {np.sum(np.abs(b_) > 1e-10)} iters {its}")
from sklearn.linear_model import Lasso
print("   sklearn check lambda=0.1:", np.round(Lasso(alpha=0.1, fit_intercept=False, tol=1e-12, max_iter=100000).fit(Xl, yl).coef_, 3))
for lam in (1.0, 100.0, 1000.0):
    print(f"Ex 13.5.5 ridge lambda={lam}: {np.round(lsq.ridge(Xl, yl, lam), 3)}")
print("   soft-threshold S(2.5,1), S(0.4,1), S(-1.7,1):", rg.soft_threshold(2.5, 1), rg.soft_threshold(0.4, 1), rg.soft_threshold(-1.7, 1))

print("\n=== 13.6 Weighted, generalised, robust ===")
xh = np.linspace(1, 10, 40); yh = 2 + 3 * xh + xh * rng.standard_normal(40) * 0.8
fo = rg.ols(xh, yh); bw = rg.wls(xh, yh, 1 / xh ** 2)
Xh = np.column_stack([np.ones(40), xh]); Wm = np.diag(1 / xh ** 2)
covw = np.linalg.inv(Xh.T @ Wm @ Xh) * (np.sum((yh - Xh @ bw) ** 2 / xh ** 2) / 38)
print("Ex 13.6.1 OLS", fo.beta, "se", fo.se, "| WLS", bw, "se", np.sqrt(np.diag(covw)))
nT = 100; phi = 0.8; e = np.zeros(nT); eps = rng.standard_normal(nT)
for t_ in range(1, nT):
    e[t_] = phi * e[t_ - 1] + eps[t_]
xt = np.linspace(0, 10, nT); yt = 1 + 0.5 * xt + e
Sig = phi ** np.abs(np.subtract.outer(np.arange(nT), np.arange(nT))) / (1 - phi ** 2)
Lc = np.linalg.cholesky(Sig); Xt_ = np.column_stack([np.ones(nT), xt])
Xw = np.linalg.solve(Lc, Xt_); yw = np.linalg.solve(Lc, yt)
bg = np.linalg.lstsq(Xw, yw, rcond=None)[0]; covg = np.linalg.inv(Xw.T @ Xw)
fols = rg.ols(xt, yt)
covols_true = np.linalg.inv(Xt_.T @ Xt_) @ Xt_.T @ Sig @ Xt_ @ np.linalg.inv(Xt_.T @ Xt_)
print("Ex 13.6.2 GLS", bg, "se", np.sqrt(np.diag(covg)), "| OLS", fols.beta, "naive se", fols.se, "true OLS se", np.sqrt(np.diag(covols_true)))
xo2 = np.linspace(0, 10, 30); yo2 = 1 + 2 * xo2 + 0.5 * rng.standard_normal(30); yo2[[5, 20, 27]] += [15, -20, 25]
bo = rg.ols(xo2, yo2).beta; bhub, wts, its = rg.huber_irls(xo2, yo2)
print("Ex 13.6.3 outliers: OLS", bo, "Huber", bhub, "iterations", its, "weights of outliers", wts[[5, 20, 27]])
means = np.array([10.2, 12.1, 13.8, 16.3]); cnt = np.array([50, 5, 20, 2.0]); xg = np.array([1.0, 2, 3, 4])
print("Ex 13.6.4 group means: unweighted", np.polyfit(xg, means, 1)[::-1], "weighted by n", rg.wls(xg, means, cnt))
bl1 = np.linalg.lstsq(np.column_stack([np.ones(30), xo2]), yo2, rcond=None)[0]
for _ in range(100):
    r_ = yo2 - np.column_stack([np.ones(30), xo2]) @ bl1; w_ = 1 / np.maximum(np.abs(r_), 1e-8)
    bl1 = rg.wls(xo2, yo2, w_)
print("Ex 13.6.5 LAD via IRLS", bl1)

print("\n=== 13.7 Nonlinear least squares ===")
td = np.array([0, 1, 2, 3, 4, 5, 6.0]); cd = np.array([10.1, 6.2, 3.6, 2.3, 1.4, 0.8, 0.5])
r = lambda b: b[0] * np.exp(-b[1] * td) - cd
J = lambda b: np.column_stack([np.exp(-b[1] * td), -b[0] * td * np.exp(-b[1] * td)])
bgn, hist = rg.gauss_newton(r, J, [8.0, 0.3])
for k, v in enumerate(hist[:7]):
    print(f"Ex 13.7.1 GN k={k}: {v} SSE {np.sum(r(v) ** 2):.6f}")
S = np.array([0.5, 1, 2, 4, 8, 16.0]); Vv = np.array([1.2, 1.9, 2.7, 3.3, 3.8, 4.0])
rm = lambda b: b[0] * S / (b[1] + S) - Vv
Jm = lambda b: np.column_stack([S / (b[1] + S), -b[0] * S / (b[1] + S) ** 2])
bmm, hm = rg.levenberg_marquardt(rm, Jm, [1.0, 1.0])
print("Ex 13.7.2 Michaelis-Menten LM: Vmax, Km", bmm, "iterations", len(hm) - 1, "SSE", np.sum(rm(bmm) ** 2))
tg = np.arange(0, 15.0); yg = np.array([2.1, 3.0, 4.4, 6.9, 9.8, 14.1, 19.2, 25.9, 31.8, 37.5, 41.8, 45.0, 47.1, 48.6, 49.2])
rl = lambda b: b[0] / (1 + np.exp(-b[1] * (tg - b[2]))) - yg
Jl = lambda b: np.column_stack([1 / (1 + np.exp(-b[1] * (tg - b[2]))),
                                b[0] * (tg - b[2]) * np.exp(-b[1] * (tg - b[2])) / (1 + np.exp(-b[1] * (tg - b[2]))) ** 2,
                                -b[0] * b[1] * np.exp(-b[1] * (tg - b[2])) / (1 + np.exp(-b[1] * (tg - b[2]))) ** 2])
bl3, hl = rg.levenberg_marquardt(rl, Jl, [40.0, 0.5, 5.0])
print("Ex 13.7.3 logistic growth K, r, t0:", bl3, "iterations", len(hl) - 1)
for start in ([10.0, 0.1, 1.0], [100.0, 2.0, 20.0]):
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            bgn2, h2 = rg.gauss_newton(rl, Jl, start, max_iter=50)
        print(f"Ex 13.7.4 GN from {start}: {bgn2} SSE {np.sum(rl(bgn2) ** 2):.3g}")
    except Exception as ex:
        print(f"Ex 13.7.4 GN from {start}: failed ({type(ex).__name__})")
    with np.errstate(over="ignore", invalid="ignore"):
        blm, hlm = rg.levenberg_marquardt(rl, Jl, start, max_iter=500)
    print(f"   LM from {start}: {blm} SSE {np.sum(rl(blm) ** 2):.3g}")
Jf = J(bgn); s2 = np.sum(r(bgn) ** 2) / (7 - 2); covn = s2 * np.linalg.inv(Jf.T @ Jf)
print("Ex 13.7.5 decay fit se", np.sqrt(np.diag(covn)), "95% CI k", bgn[1] - stats.t.ppf(0.975, 5) * math.sqrt(covn[1, 1]), bgn[1] + stats.t.ppf(0.975, 5) * math.sqrt(covn[1, 1]), "half-life", math.log(2) / bgn[1])

print("\n=== 13.8 GLMs ===")
hrs = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
pas = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1.0])
Xl2 = np.column_stack([np.ones(20), hrs]); bl2, covl, hl2 = rg.glm_irls(Xl2, pas, "logistic")
print("Ex 13.8.1 logistic IRLS:", [np.round(v, 5) for v in hl2[:7]], "\n   beta", bl2, "se", np.sqrt(np.diag(covl)), "odds ratio per hour", math.exp(bl2[1]), "P(pass|2h)", expit(bl2[0] + 2 * bl2[1]), "50% hours", -bl2[0] / bl2[1])
cnts = np.array([2, 3, 6, 7, 8, 9, 10, 12, 15.0]); expo = np.array([10, 12, 15, 14, 13, 12, 11, 12, 13.0]); xp = np.arange(9.0)
bp, covp, hp = rg.glm_irls(np.column_stack([np.ones(9), xp]), cnts, "poisson", offset=np.log(expo))
print("Ex 13.8.2 Poisson IRLS beta", bp, "se", np.sqrt(np.diag(covp)), "rate ratio", math.exp(bp[1]), "iterations", len(hp) - 1)
xsep = np.array([1, 2, 3, 4, 5, 6.0]); ysep = np.array([0, 0, 0, 1, 1, 1.0])
with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    bsep, _, hsep = rg.glm_irls(np.column_stack([np.ones(6), xsep]), ysep, "logistic", max_iter=25)
    nlls = [float(np.sum(np.logaddexp(0, v[0] + v[1] * xsep) - ysep * (v[0] + v[1] * xsep))) for v in hsep[:12]]
print("Ex 13.8.3 separable data slope by iteration:", [round(float(v[1]), 3) for v in hsep[:12]], "\n   NLL:", np.round(nlls, 6))
import statsmodels.api as sm
res_sm = sm.GLM(pas, Xl2, family=sm.families.Binomial()).fit()
print("Ex 13.8.4 statsmodels", res_sm.params, res_sm.bse)
from scipy.optimize import minimize
nll = lambda b: np.sum(np.logaddexp(0, Xl2 @ b) - pas * (Xl2 @ b))
print("Ex 13.8.5 BFGS on NLL", minimize(nll, [0, 0], method="BFGS").x, "IRLS iterations", len(hl2) - 1)
