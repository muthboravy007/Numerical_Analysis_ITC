"""Chapter 9 — Numerical optimization.

Library equivalents: ``scipy.optimize.minimize`` (BFGS, L-BFGS-B, Newton-CG),
``scipy.optimize.minimize_scalar``, ``torch.optim.SGD`` / ``Adam``.

Each optimizer returns ``(x, history)`` with history the list of iterates.
"""

import numpy as np


def golden_section(f, a, b, tol=1e-8):
    """Minimise a unimodal f on [a, b]."""
    invphi = (np.sqrt(5) - 1) / 2  # 0.618...
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    history = []
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        history.append((a + b) / 2)
    return (a + b) / 2, history


def backtracking(f, grad_x, x, p, alpha=1.0, rho=0.5, c=1e-4):
    """Armijo backtracking line search along descent direction p."""
    fx = f(x)
    slope = grad_x @ p
    while f(x + alpha * p) > fx + c * alpha * slope:
        alpha *= rho
        if alpha < 1e-16:
            break
    return alpha


def gradient_descent(f, grad, x0, lr=None, tol=1e-8, max_iter=10_000):
    """Fixed step if lr is given, otherwise Armijo backtracking."""
    x = np.asarray(x0, dtype=float)
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        step = lr if lr is not None else backtracking(f, g, x, -g)
        x = x - step * g
        history.append(x.copy())
    return x, history


def momentum_gd(grad, x0, lr, beta=0.9, tol=1e-8, max_iter=10_000):
    """Heavy-ball momentum."""
    x = np.asarray(x0, dtype=float)
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        v = beta * v - lr * g
        x = x + v
        history.append(x.copy())
    return x, history


def newton_opt(grad, hess, x0, tol=1e-10, max_iter=100):
    x = np.asarray(x0, dtype=float)
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - np.linalg.solve(hess(x), g)
        history.append(x.copy())
    return x, history


def bfgs(f, grad, x0, tol=1e-8, max_iter=1000):
    x = np.asarray(x0, dtype=float)
    n = len(x)
    H = np.eye(n)  # inverse-Hessian approximation
    g = grad(x)
    history = [x.copy()]
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        p = -H @ g
        alpha = backtracking(f, g, x, p)
        s = alpha * p
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = s @ y
        if sy > 1e-12:  # curvature condition keeps H positive definite
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, g = x_new, g_new
        history.append(x.copy())
    return x, history


def sgd(grad_i, x0, n_samples, lr=0.01, epochs=50, batch_size=1, rng=None,
        decay=0.0):
    """Mini-batch SGD. grad_i(x, idx) returns the mean gradient over rows idx.
    Learning rate at epoch t is lr / (1 + decay * t)."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x0, dtype=float)
    history = [x.copy()]
    for epoch in range(epochs):
        eta = lr / (1 + decay * epoch)
        perm = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            x = x - eta * grad_i(x, idx)
        history.append(x.copy())
    return x, history


def adam(grad_i, x0, n_samples, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
         epochs=50, batch_size=32, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    history = [x.copy()]
    for _ in range(epochs):
        perm = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            g = grad_i(x, idx)
            t += 1
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(x.copy())
    return x, history


def rosenbrock(x, a=1.0, b=100.0):
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x, a=1.0, b=100.0):
    return np.array([
        -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
        2 * b * (x[1] - x[0] ** 2),
    ])


def rosenbrock_hess(x, a=1.0, b=100.0):
    return np.array([
        [2 - 4 * b * (x[1] - 3 * x[0] ** 2), -4 * b * x[0]],
        [-4 * b * x[0], 2 * b],
    ])


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def logistic_loss(w, X, y, lam=0.0):
    """Mean cross-entropy for labels y in {0,1}; stable via logaddexp."""
    z = X @ w
    return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w


def logistic_grad(w, X, y, lam=0.0):
    return X.T @ (sigmoid(X @ w) - y) / len(y) + lam * w


def logistic_hess(w, X, y, lam=0.0):
    p = sigmoid(X @ w)
    return (X.T * (p * (1 - p))) @ X / len(y) + lam * np.eye(len(w))
