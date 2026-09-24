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
    """Levenberg-Marquardt: (J^T J + lam diag(J^T J)) dp = -J^T r with adaptive lam."""
    beta = np.asarray(beta0, dtype=float)
    cost = np.sum(r(beta) ** 2)
    history = [beta.copy()]
    for _ in range(max_iter):
        Jb, rb = J(beta), r(beta)
        A = Jb.T @ Jb
        g = Jb.T @ rb
        dp = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
        new = beta + dp
        new_cost = np.sum(r(new) ** 2)
        if new_cost < cost:
            beta, cost, lam = new, new_cost, lam / 10
            history.append(beta.copy())
            if np.linalg.norm(dp) < tol * max(1.0, np.linalg.norm(beta)):
                break
        else:
            lam *= 10
    return beta, history


def vif(X):
    """Variance inflation factors VIF_j = 1/(1 - R_j^2) (columns of X, no intercept)."""
    X = np.asarray(X, dtype=float)
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        out.append(1 / (1 - ols(others, X[:, j]).r2))
    return np.array(out)
