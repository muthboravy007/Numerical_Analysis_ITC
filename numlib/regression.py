"""Chapter 13 — Regression from the numerical-analysis viewpoint.

Simple, multiple, polynomial, weighted and nonlinear least squares with the
statistics a data scientist needs (standard errors, t-values, R^2, confidence
intervals, leverage).

Library equivalents: ``statsmodels.api.OLS`` / ``WLS``,
``sklearn.linear_model.LinearRegression``, ``scipy.optimize.least_squares``,
``scipy.optimize.curve_fit``.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    sigma2: float
    r2: float
    r2_adj: float
    residuals: np.ndarray
    fitted: np.ndarray
    leverage: np.ndarray
    df: int

    def conf_int(self, level=0.95):
        q = stats.t.ppf(0.5 + level / 2, self.df)
        return np.column_stack([self.beta - q * self.se, self.beta + q * self.se])


def ols(X, y, add_intercept=True):
    """Ordinary least squares via thin QR (never the normal equations)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    if add_intercept:
        X = np.column_stack([np.ones(len(y)), X])
    n, p = X.shape
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    fitted = X @ beta
    res = y - fitted
    df = n - p
    sigma2 = res @ res / df
    Rinv = np.linalg.inv(R)
    cov = sigma2 * Rinv @ Rinv.T  # sigma^2 (X^T X)^{-1}
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), df)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - res @ res / sst
    r2_adj = 1 - (1 - r2) * (n - 1) / df
    leverage = np.sum(Q ** 2, axis=1)  # diag of hat matrix H = Q Q^T
    return OLSResult(beta, se, t, pval, sigma2, r2, r2_adj, res, fitted, leverage, df)


def simple_regression(x, y):
    """Closed-form simple linear regression y = b0 + b1 x.
    Returns dict with b0, b1, Sxx, Sxy, r, R2, s, se_b0, se_b1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    xb, yb = x.mean(), y.mean()
    Sxx = np.sum((x - xb) ** 2)
    Syy = np.sum((y - yb) ** 2)
    Sxy = np.sum((x - xb) * (y - yb))
    b1 = Sxy / Sxx
    b0 = yb - b1 * xb
    sse = Syy - b1 * Sxy
    s2 = sse / (n - 2)
    return dict(b0=b0, b1=b1, Sxx=Sxx, Syy=Syy, Sxy=Sxy, r=Sxy / np.sqrt(Sxx * Syy),
                R2=Sxy ** 2 / (Sxx * Syy), SSE=sse, s=np.sqrt(s2),
                se_b1=np.sqrt(s2 / Sxx), se_b0=np.sqrt(s2 * (1 / n + xb ** 2 / Sxx)))


def polynomial_regression(x, y, degree, orthogonal=True):
    """Polynomial LS. With orthogonal=True the design uses Legendre polynomials on
    the scaled variable (well conditioned); coefficients are returned in the
    monomial basis of the original x for readability."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not orthogonal:
        A = np.vander(x, degree + 1, increasing=True)
        return np.linalg.lstsq(A, y, rcond=None)[0], np.linalg.cond(A)
    from numpy.polynomial import Legendre

    P = Legendre.fit(x, y, degree)
    A = np.polynomial.legendre.legvander(P.mapparms()[0] + P.mapparms()[1] * x, degree)
    return P.convert(kind=np.polynomial.Polynomial).coef, np.linalg.cond(A)


def wls(X, y, w, add_intercept=True):
    """Weighted least squares: minimise sum w_i (y_i - x_i^T b)^2."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_intercept:
        X = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(np.asarray(w, dtype=float))
    return np.linalg.lstsq(X * sw[:, None], np.asarray(y) * sw, rcond=None)[0]


def gauss_newton(r, J, beta0, tol=1e-10, max_iter=100):
    """Nonlinear least squares min ||r(beta)||^2: solve J dp = -r by QR each step."""
    beta = np.asarray(beta0, dtype=float)
    history = [beta.copy()]
    for _ in range(max_iter):
        dp = np.linalg.lstsq(J(beta), -r(beta), rcond=None)[0]
        beta = beta + dp
        history.append(beta.copy())
        if np.linalg.norm(dp) < tol * max(1.0, np.linalg.norm(beta)):
            break
    return beta, history


def levenberg_marquardt(r, J, beta0, lam=1e-2, tol=1e-10, max_iter=200):
    """Levenberg-Marquardt: (J^T J + lam D) dp = -J^T r with D = diag(J^T J)
    (floored to stay positive) and adaptive lam."""
    beta = np.asarray(beta0, dtype=float)
    cost = np.sum(r(beta) ** 2)
    history = [beta.copy()]
    for _ in range(max_iter):
        Jb, rb = J(beta), r(beta)
        A = Jb.T @ Jb
        g = Jb.T @ rb
        D = np.diag(np.maximum(np.diag(A), 1e-12 * max(1.0, np.max(np.diag(A)))))
        try:
            dp = np.linalg.solve(A + lam * D, -g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        new = beta + dp
        with np.errstate(over="ignore", invalid="ignore"):
            new_cost = np.sum(r(new) ** 2)
        if np.isfinite(new_cost) and new_cost < cost:
            beta, cost, lam = new, new_cost, max(lam / 10, 1e-12)
            history.append(beta.copy())
            if np.linalg.norm(dp) < tol * max(1.0, np.linalg.norm(beta)):
                break
        else:
            lam *= 10
            if lam > 1e16:
                break
    return beta, history


def vif(X):
    """Variance inflation factors VIF_j = 1/(1 - R_j^2) (columns of X, no intercept)."""
    X = np.asarray(X, dtype=float)
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        out.append(1 / (1 - ols(others, X[:, j]).r2))
    return np.array(out)


def soft_threshold(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def lasso_cd(X, y, lam, max_iter=10_000, tol=1e-10):
    """Lasso min (1/2n)||y - Xb||^2 + lam ||b||_1 by cyclic coordinate descent.
    X should be centred/standardised and y centred (no intercept)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    b = np.zeros(p)
    col_sq = (X ** 2).sum(axis=0) / n
    r = y - X @ b
    for it in range(max_iter):
        b_old = b.copy()
        for j in range(p):
            r += X[:, j] * b[j]
            b[j] = soft_threshold(X[:, j] @ r / n, lam) / col_sq[j]
            r -= X[:, j] * b[j]
        if np.max(np.abs(b - b_old)) < tol:
            break
    return b, it + 1


def huber_irls(X, y, c=1.345, max_iter=100, tol=1e-10, add_intercept=True):
    """Robust regression with Huber's loss by iteratively reweighted least squares."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_intercept:
        X = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for it in range(max_iter):
        r = y - X @ beta
        s = np.median(np.abs(r - np.median(r))) / 0.6745 or 1.0
        u = np.abs(r / s)
        w = np.where(u <= c, 1.0, c / u)
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta, w, it + 1


def glm_irls(X, y, family="logistic", offset=None, max_iter=50, tol=1e-10):
    """Fit a GLM with canonical link by IRLS (= Newton's method).
    family: 'logistic' (y in {0,1}) or 'poisson' (counts). Returns (beta, cov, history)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(X.shape[1])
    history = [beta.copy()]
    for _ in range(max_iter):
        eta = X @ beta + off
        if family == "logistic":
            mu = 1 / (1 + np.exp(-eta))
            w = mu * (1 - mu)
        elif family == "poisson":
            mu = np.exp(eta)
            w = mu
        else:
            raise ValueError(family)
        z = eta - off + (y - mu) / w          # working response
        WX = X * w[:, None]
        new = np.linalg.solve(X.T @ WX, WX.T @ z)
        history.append(new.copy())
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    eta = X @ beta + off
    mu = 1 / (1 + np.exp(-eta)) if family == "logistic" else np.exp(eta)
    w = mu * (1 - mu) if family == "logistic" else mu
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    return beta, cov, history


def kfold_cv_mse(fit_predict, X, y, k=5, seed=0):
    """Generic K-fold cross-validation. fit_predict(Xtr, ytr, Xte) -> predictions."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    err = 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        pred = fit_predict(X[tr], y[tr], X[f])
        err += np.sum((pred - y[f]) ** 2)
    return err / len(y)
