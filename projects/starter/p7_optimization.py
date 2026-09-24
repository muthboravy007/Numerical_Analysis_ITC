"""P7 starter: logistic regression trained by gradient descent vs L-BFGS.

Run: python p7_optimization.py
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer


def loss(w, X, y, lam):
    z = X @ w
    return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w


def grad(w, X, y, lam):
    p = 0.5 * (1 + np.tanh(0.5 * (X @ w)))  # stable sigmoid
    return X.T @ (p - y) / len(y) + lam * w


def gradient_descent(X, y, lam, lr, iters):
    w = np.zeros(X.shape[1]); hist = []
    for _ in range(iters):
        w -= lr * grad(w, X, y, lam)
        hist.append(loss(w, X, y, lam))
    return w, np.array(hist)


def nesterov(X, y, lam, lr, iters):
    """TODO: Nesterov accelerated gradient; compare the slope of log(f - f*) with GD."""
    raise NotImplementedError


class Var:
    """TODO: build a reverse-mode autodiff scalar (see Chapter 15, C4) and use it for an MLP."""


if __name__ == "__main__":
    Xr, y = load_breast_cancer(return_X_y=True)
    lam = 1e-3
    for scaled in (False, True):
        X = (Xr - Xr.mean(0)) / Xr.std(0) if scaled else Xr / Xr.max()
        X = np.column_stack([np.ones(len(y)), X])
        L = np.linalg.eigvalsh(X.T @ X / len(y)).max() / 4 + lam
        ref = minimize(loss, np.zeros(X.shape[1]), args=(X, y, lam), jac=grad, method="L-BFGS-B", options={"gtol": 1e-12, "maxiter": 10000})
        fstar = ref.fun
        _, h = gradient_descent(X, y, lam, 1 / L, 2000)
        kappa = L / lam
        print(f"{'standardised' if scaled else 'max-scaled':12s}: L = {L:.3g}, kappa <= {kappa:.3g}; L-BFGS {ref.nit} its; "
              f"GD gap after 100/1000/2000 its: {h[99] - fstar:.2e} {h[999] - fstar:.2e} {h[1999] - fstar:.2e}")
    # TODO: Newton, BFGS with Wolfe, SGD, SVRG, Adam, autodiff MLP, multi-seed studies
