"""Chapter 10 — Ordinary differential equations y' = f(t, y).

Library equivalents: ``scipy.integrate.solve_ivp`` (RK45, DOP853, Radau, BDF,
LSODA), ``scipy.integrate.solve_bvp``.

All steppers accept scalar or vector y and return ``(t, Y)`` where ``Y[k]``
approximates y(t[k]).
"""

import numpy as np


def _integrate(step, f, t_span, y0, h):
    t0, tf = t_span
    n = int(np.ceil((tf - t0) / h - 1e-12))
    h = (tf - t0) / n
    t = t0 + h * np.arange(n + 1)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    Y = np.zeros((n + 1, len(y0)))
    Y[0] = y0
    for k in range(n):
        Y[k + 1] = step(f, t[k], Y[k], h)
    return t, Y


def euler_step(f, t, y, h):
    return y + h * f(t, y)


def heun_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + h / 2 * (k1 + k2)


def midpoint_step(f, t, y, h):
    k1 = f(t, y)
    return y + h * f(t + h / 2, y + h / 2 * k1)


def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler(f, t_span, y0, h):
    return _integrate(euler_step, f, t_span, y0, h)


def heun(f, t_span, y0, h):
    return _integrate(heun_step, f, t_span, y0, h)


def midpoint(f, t_span, y0, h):
    return _integrate(midpoint_step, f, t_span, y0, h)


def rk4(f, t_span, y0, h):
    return _integrate(rk4_step, f, t_span, y0, h)


def implicit_euler(f, t_span, y0, h, jac=None, newton_tol=1e-10, newton_iter=20):
    """Backward Euler: y_{k+1} = y_k + h f(t_{k+1}, y_{k+1}), solved by Newton.
    If jac is None a finite-difference Jacobian is used."""

    def J_fd(t, y):
        n = len(y)
        J = np.zeros((n, n))
        f0 = f(t, y)
        eps = 1e-7
        for j in range(n):
            e = np.zeros(n)
            e[j] = eps * max(1.0, abs(y[j]))
            J[:, j] = (f(t, y + e) - f0) / e[j]
        return J

    jac = jac or J_fd

    def step(f, t, y, h):
        z = y + h * f(t, y)  # explicit Euler predictor
        n = len(y)
        for _ in range(newton_iter):
            G = z - y - h * f(t + h, z)
            dz = np.linalg.solve(np.eye(n) - h * jac(t + h, z), -G)
            z = z + dz
            if np.linalg.norm(dz) < newton_tol * max(1.0, np.linalg.norm(z)):
                break
        return z

    return _integrate(step, f, t_span, y0, h)


def trapezoidal_linear(A, t_span, y0, h):
    """Crank-Nicolson for linear y' = A y (A-stable, 2nd order)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    I = np.eye(n)
    M = np.linalg.solve(I - h / 2 * A, I + h / 2 * A)
    return _integrate(lambda f, t, y, h: M @ y, None, t_span, y0, h)


# Dormand-Prince 5(4) coefficients
_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200,
                187 / 2100, 1 / 40])


def rk45(f, t_span, y0, rtol=1e-6, atol=1e-9, h0=None, max_steps=100_000):
    """Adaptive Dormand-Prince RK5(4) with standard step-size control."""
    t0, tf = t_span
    y = np.atleast_1d(np.asarray(y0, dtype=float))
    t = t0
    h = h0 or (tf - t0) / 100
    ts, Ys = [t], [y.copy()]
    n_rejected = 0
    for _ in range(max_steps):
        if t >= tf:
            break
        h = min(h, tf - t)
        K = np.zeros((7, len(y)))
        for i in range(7):
            yi = y + h * sum(a * K[j] for j, a in enumerate(_A[i]))
            K[i] = f(t + _C[i] * h, yi)
        y5 = y + h * (_B5 @ K)
        y4 = y + h * (_B4 @ K)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err = np.sqrt(np.mean(((y5 - y4) / scale) ** 2))
        if err <= 1.0:
            t, y = t + h, y5
            ts.append(t)
            Ys.append(y.copy())
        else:
            n_rejected += 1
        h *= min(5.0, max(0.2, 0.9 * (err + 1e-16) ** (-1 / 5)))
    return np.array(ts), np.array(Ys), n_rejected


def convergence_order(errors, hs):
    """Observed order from successive (h, error) pairs."""
    errors = np.asarray(errors)
    hs = np.asarray(hs)
    return np.log(errors[1:] / errors[:-1]) / np.log(hs[1:] / hs[:-1])


# ----- model problems used in lectures -----

def logistic_growth(r=1.0, K=1.0):
    return lambda t, y: r * y * (1 - y / K)


def sir(beta=0.3, gamma=0.1):
    """SIR epidemic model with S, I, R as population fractions."""
    def f(t, y):
        S, I, R = y
        return np.array([-beta * S * I, beta * S * I - gamma * I, gamma * I])
    return f


def lotka_volterra(a=1.0, b=0.1, c=1.5, d=0.075):
    def f(t, y):
        x, z = y
        return np.array([a * x - b * x * z, -c * z + d * x * z])
    return f


# ---------------- B&F 5.3: Taylor methods ----------------

def taylor2(f, ft, fy, t_span, y0, h):
    """Taylor method of order 2 for scalar y' = f(t,y):
    y_{k+1} = y_k + h f + h^2/2 (f_t + f_y f)."""
    def step(_, t, y, h):
        fv = f(t, y)
        return y + h * fv + h * h / 2 * (ft(t, y) + fy(t, y) * fv)
    return _integrate(step, f, t_span, y0, h)


def modified_euler(f, t_span, y0, h):
    """B&F's name for Heun's method (trapezoid predictor-corrector)."""
    return heun(f, t_span, y0, h)


# ---------------- B&F 5.5: Runge-Kutta-Fehlberg ----------------

def rkf45(f, t_span, y0, tol=1e-6, hmax=0.25, hmin=1e-8):
    """Runge-Kutta-Fehlberg (B&F Algorithm 5.3) for scalar or vector IVPs.
    Returns (t, Y, h_used)."""
    t0, tf = t_span
    t = t0
    y = np.atleast_1d(np.asarray(y0, dtype=float))
    h = hmax
    ts, Ys, hs = [t], [y.copy()], []
    while t < tf - 1e-14:
        h = min(h, tf - t)
        k1 = h * f(t, y)
        k2 = h * f(t + h / 4, y + k1 / 4)
        k3 = h * f(t + 3 * h / 8, y + 3 / 32 * k1 + 9 / 32 * k2)
        k4 = h * f(t + 12 * h / 13, y + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3)
        k5 = h * f(t + h, y + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4)
        k6 = h * f(t + h / 2, y - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5)
        R = np.max(np.abs(k1 / 360 - 128 / 4275 * k3 - 2197 / 75240 * k4 + k5 / 50 + 2 / 55 * k6)) / h
        if R <= tol:
            t = t + h
            y = y + 25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - k5 / 5
            ts.append(t)
            Ys.append(y.copy())
            hs.append(h)
        delta = 0.84 * (tol / R) ** 0.25 if R > 0 else 4.0
        h = h * min(max(delta, 0.1), 4.0)
        h = min(h, hmax)
        if h < hmin:
            raise RuntimeError("minimum h exceeded")
    return np.array(ts), np.array(Ys), np.array(hs)


# ---------------- B&F 5.6: multistep methods ----------------

def adams_bashforth4(f, t_span, y0, h):
    """Explicit 4-step Adams-Bashforth, started with RK4."""
    t0, tf = t_span
    n = int(round((tf - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    Y = np.zeros((n + 1, len(y0)))
    Y[0] = y0
    for k in range(min(3, n)):
        Y[k + 1] = rk4_step(f, t[k], Y[k], h)
    F = [f(t[k], Y[k]) for k in range(min(4, n + 1))]
    for k in range(3, n):
        Y[k + 1] = Y[k] + h / 24 * (55 * F[k] - 59 * F[k - 1] + 37 * F[k - 2] - 9 * F[k - 3])
        F.append(f(t[k + 1], Y[k + 1]))
    return t, Y


def adams_pc4(f, t_span, y0, h):
    """Adams fourth-order predictor-corrector (AB4 predictor, AM3 corrector),
    B&F Algorithm 5.4."""
    t0, tf = t_span
    n = int(round((tf - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    Y = np.zeros((n + 1, len(y0)))
    Y[0] = y0
    for k in range(min(3, n)):
        Y[k + 1] = rk4_step(f, t[k], Y[k], h)
    F = [f(t[k], Y[k]) for k in range(min(4, n + 1))]
    for k in range(3, n):
        yp = Y[k] + h / 24 * (55 * F[k] - 59 * F[k - 1] + 37 * F[k - 2] - 9 * F[k - 3])
        fp = f(t[k + 1], yp)
        Y[k + 1] = Y[k] + h / 24 * (9 * fp + 19 * F[k] - 5 * F[k - 1] + F[k - 2])
        F.append(f(t[k + 1], Y[k + 1]))
    return t, Y


def heun3_step(f, t, y, h):
    """B&F's third-order Heun method."""
    k1 = f(t, y)
    k2 = f(t + h / 3, y + h / 3 * k1)
    k3 = f(t + 2 * h / 3, y + 2 * h / 3 * k2)
    return y + h / 4 * (k1 + 3 * k3)


def heun3(f, t_span, y0, h):
    return _integrate(heun3_step, f, t_span, y0, h)


def taylor4_linear_example(t_span, y0, h):
    """Taylor method of order 4 for the model problem y' = y - t^2 + 1 (B&F 5.3)."""
    def step(_, t, y, h):
        f1 = y - t ** 2 + 1
        f2 = y - t ** 2 + 1 - 2 * t
        f3 = y - t ** 2 - 2 * t - 1
        return y + h * f1 + h ** 2 / 2 * f2 + h ** 3 / 6 * f3 + h ** 4 / 24 * f3
    return _integrate(step, None, t_span, y0, h)
