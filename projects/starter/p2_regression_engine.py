"""P2 starter: a numerically careful OLS engine.

Run: python p2_regression_engine.py
"""
import numpy as np
from scipy import stats


class OLS:
    """OLS with an inference table. method: 'normal', 'qr' (numpy), 'householder' (TODO)."""

    def __init__(self, method="qr"):
        self.method = method

    def fit(self, X, y, names=None):
        X = np.column_stack([np.ones(len(y)), X])
        n, p = X.shape
        if self.method == "normal":
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            R = np.linalg.cholesky(X.T @ X).T
        elif self.method == "qr":
            Q, R = np.linalg.qr(X)
            beta = np.linalg.solve(R, Q.T @ y)
        elif self.method == "householder":
            Q, R = householder_qr(X)
            beta = np.linalg.solve(R, Q.T @ y)
        else:
            raise ValueError(self.method)
        res = y - X @ beta
        self.df = n - p
        self.sigma2 = res @ res / self.df
        Rinv = np.linalg.inv(R)
        self.cov = self.sigma2 * Rinv @ Rinv.T
        self.beta, self.se = beta, np.sqrt(np.diag(self.cov))
        self.r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
        self.names = ["const"] + (list(names) if names is not None else [f"x{j}" for j in range(1, p)])
        return self

    def summary(self):
        t = self.beta / self.se
        pval = 2 * stats.t.sf(np.abs(t), self.df)
        lines = [f"{'':10s}{'coef':>14s}{'se':>12s}{'t':>9s}{'p':>9s}"]
        for nm, b, s, tt, pp in zip(self.names, self.beta, self.se, t, pval):
            lines.append(f"{nm:10s}{b:14.6g}{s:12.4g}{tt:9.3f}{pp:9.4f}")
        lines.append(f"R^2 = {self.r2:.4f}, sigma = {np.sqrt(self.sigma2):.4g}, df = {self.df}")
        return "\n".join(lines)


def householder_qr(A):
    """TODO: implement Householder QR (thin Q, R) yourself.
    Test: ||Q^T Q - I|| ~ 1e-16 and ||QR - A|| / ||A|| ~ 1e-16."""
    raise NotImplementedError


LONGLEY_CERTIFIED = np.array([-3482258.63459582, 15.0618722713733, -0.358191792925910e-1,
                              -2.02022980381683, -1.03322686717359, -0.511041056535807e-1, 1829.15146461355])

if __name__ == "__main__":
    import statsmodels.api as sm
    d = sm.datasets.longley.load_pandas()
    X, y = d.exog.values, d.endog.values
    for method in ("normal", "qr"):
        fit = OLS(method).fit(X, y, d.exog.columns)
        lre = -np.log10(np.abs(fit.beta - LONGLEY_CERTIFIED) / np.abs(LONGLEY_CERTIFIED))
        print(f"Longley, method={method}: correct digits (LRE) min {lre.min():.1f}")
    print(OLS("qr").fit(X, y, d.exog.columns).summary())
    # TODO: NIST StRD suite (Filip, Wampler), diagnostics, robust SE, ridge/lasso, real data
