"""Reproduces every numerical example in Chapter 11 (Boundary-Value Problems for ODEs).
Run from the repository root:  python lectures/code/ch11.py"""

import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq, least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import bvp, ode, roots  # noqa: E402
from numlib.linalg_direct import tridiagonal_solve  # noqa: E402

np.set_printoptions(precision=7, suppress=True, linewidth=120)
p0 = lambda x: 0 * x
q1 = lambda x: 1 + 0 * x
rx = lambda x: x
ex = lambda x: 2 * np.sinh(x) / math.sinh(1) - x

print("=== 11.1 Linear shooting ===")
x, w, wp = bvp.linear_shooting(p0, q1, rx, 0, 1, 0, 1, 10)
for xi, wi in list(zip(x, w))[::2]:
    print(f"Ex 11.1.1 x={xi:.1f} w={wi:.8f} y={ex(xi):.8f} err={abs(wi - ex(xi)):.1e}")
f1 = lambda x, u: np.array([u[1], u[0] + x]); f2 = lambda x, u: np.array([u[1], u[0]])
_, U1 = ode.rk4(f1, (0, 1), [0.0, 0.0], 0.1); _, U2 = ode.rk4(f2, (0, 1), [0.0, 1.0], 0.1)
print("Ex 11.1.2 y1(1)", U1[-1, 0], "y2(1)", U2[-1, 0], "c", (1 - U1[-1, 0]) / U2[-1, 0], "exact y1(1)=sinh1-1", math.sinh(1) - 1, "y2(1)=sinh1", math.sinh(1))
for N in (5, 10, 20, 40):
    x, w, _ = bvp.linear_shooting(p0, q1, rx, 0, 1, 0, 1, N)
    print(f"Ex 11.1.3 N={N}: max err {np.max(np.abs(w - ex(x))):.3e}")
exs = lambda x: np.exp(-20 * x)
for N in (20, 50, 100):
    x, w, _ = bvp.linear_shooting(p0, lambda x: 400 + 0 * x, p0, 0, 1, 1.0, math.exp(-20), N)
    print(f"Ex 11.1.4 y''=400y N={N}: max err {np.max(np.abs(w - exs(x))):.3e}  w(0.5)={w[N // 2]:.3e} exact {exs(0.5):.3e}")
hc, Ta = 0.01, 20.0
x, w, _ = bvp.linear_shooting(p0, lambda x: hc + 0 * x, lambda x: -hc * Ta + 0 * x, 0, 10, 40.0, 200.0, 20)
lam = math.sqrt(hc); A_ = np.array([[1, 1], [math.exp(10 * lam), math.exp(-10 * lam)]]); c_rod = np.linalg.solve(A_, [40 - Ta, 200 - Ta])
Tex = lambda x: Ta + c_rod[0] * np.exp(lam * x) + c_rod[1] * np.exp(-lam * x)
print("Ex 11.1.5 rod T(5) shooting", w[10], "exact", Tex(5.0), "T at 2.5, 7.5:", w[5], w[15])

print("\n=== 11.2 Nonlinear shooting ===")
fn = lambda x, y, yp: 2 * y ** 3
fy = lambda x, y, yp: 6 * y ** 2
fyp = lambda x, y, yp: 0.0
exn = lambda x: 1 / (x + 1)
x, w, slopes = bvp.nonlinear_shooting(fn, fy, fyp, 1, 2, 0.5, 1 / 3, 10, t0=0.0)
print("Ex 11.2.1 Newton slopes", slopes, "exact -0.25; max err", np.max(np.abs(w - exn(x))))
for xi, wi in list(zip(x, w))[::2]:
    print(f"   x={xi:.1f} w={wi:.8f} y={exn(xi):.8f}")
def endval(t):
    _, U = ode.rk4(lambda x, u: np.array([u[1], 2 * u[0] ** 3]), (1, 2), [0.5, t], 0.1); return U[-1, 0] - 1 / 3
t, h = roots.secant(endval, 0.0, -0.5, tol=1e-12)
print("Ex 11.2.2 secant slopes", h)
_, U = ode.rk4(lambda x, u: np.array([u[1], 2 * u[0] ** 3]), (1, 2), [0.5, 0.0], 0.1)
print("Ex 11.2.3 t0=0: y(2)=", U[-1, 0], "target", 1 / 3)
bratu = lambda x, u: np.array([u[1], -math.exp(u[0])])
def bend(t):
    _, U = ode.rk4(bratu, (0, 1), [0.0, t], 0.01); return U[-1, 0]
ts = [brentq(bend, 0.1, 1.0), brentq(bend, 5.0, 12.0)]
for t in ts:
    _, U = ode.rk4(bratu, (0, 1), [0.0, t], 0.01); print(f"Ex 11.2.4 Bratu lambda=1: slope {t:.6f} max y {U[:, 0].max():.6f}")
g, cdrag, T = 9.81, 0.02, 3.0
def hend(v0):
    _, U = ode.rk4(lambda t, u: np.array([u[1], -g - cdrag * u[1] * abs(u[1])]), (0, T), [0.0, v0], 0.01); return U[-1, 0]
v0 = brentq(hend, 5, 40)
_, U = ode.rk4(lambda t, u: np.array([u[1], -g - cdrag * u[1] * abs(u[1])]), (0, T), [0.0, v0], 0.01)
print("Ex 11.2.5 launch speed for 3 s flight with drag:", v0, "vacuum:", g * T / 2, "max height", U[:, 0].max())

print("\n=== 11.3 Finite differences (linear) ===")
x, w = bvp.linear_fd(p0, q1, rx, 0, 1, 0, 1, 4)
print("Ex 11.3.1 N=4 (h=0.2):", w, "\n   exact", ex(x), "\n   errors", np.abs(w - ex(x)))
h = 0.2; print("   matrix diag", 2 + h * h, "offdiag -1, rhs", -h * h * x[1:-1] + np.array([0, 0, 0, 1.0]))
errs = []
for N in (4, 9, 19, 39):
    x, w = bvp.linear_fd(p0, q1, rx, 0, 1, 0, 1, N); errs.append(abs(w[(N + 1) // 2] - ex(0.5)) if (N + 1) % 2 == 0 else np.nan)
    print(f"Ex 11.3.2 N={N} h={1 / (N + 1)}: max err {np.max(np.abs(w - ex(x))):.3e}")
wh = {}
for N in (1, 3, 7):
    x, w = bvp.linear_fd(p0, q1, rx, 0, 1, 0, 1, N); wh[N] = w[(N + 1) // 2]
e1 = (4 * wh[3] - wh[1]) / 3; e2 = (4 * wh[7] - wh[3]) / 3; e3 = (16 * e2 - e1) / 15
print("   Richardson at x=0.5: w(h=.5,.25,.125)", wh, "Ext1", e1, e2, "Ext2", e3, "exact", ex(0.5), "err", abs(e3 - ex(0.5)))
x, w = bvp.linear_fd(p0, q1, rx, 0, 1, 0, 1, 3)
A = np.diag([2 + 1 / 16] * 3) - np.eye(3, k=1) - np.eye(3, k=-1)
print("Ex 11.3.3 N=3 matrix\n", A, "\n rhs", -x[1:-1] / 16 + np.array([0, 0, 1.0]), "sol", w[1:-1])
x, w = bvp.linear_fd(p0, lambda x: hc + 0 * x, lambda x: -hc * Ta + 0 * x, 0, 10, 40.0, 200.0, 9)
print("Ex 11.3.4 rod FD h=1: T(5)=", w[5], "exact", Tex(5.0), "err", abs(w[5] - Tex(5.0)))
for N in (9, 99, 999):
    h = 1 / (N + 1); Am = np.diag([2 + h * h] * N) - np.eye(N, k=1) - np.eye(N, k=-1)
    print(f"Ex 11.3.5 N={N}: cond {np.linalg.cond(Am):.3e}  ~ 4/(pi h)^2 = {4 / (math.pi * h) ** 2:.3e}")

print("\n=== 11.4 Finite differences (nonlinear) ===")
fnv = lambda x, y, yp: 2 * y ** 3
fyv = lambda x, y, yp: 6 * y ** 2
fypv = lambda x, y, yp: 0 * y
x, w, its = bvp.nonlinear_fd(fnv, fyv, fypv, 1, 2, 0.5, 1 / 3, 9)
print("Ex 11.4.1 N=9 Newton iterations", its, "max err", np.max(np.abs(w - exn(x))), "\n  ", w)
fb = lambda x, y, yp: -np.exp(y); fby = lambda x, y, yp: -np.exp(y)
def bratu_fd(guess_amp, N=49):
    h = 1 / (N + 1); xg = h * np.arange(1, N + 1); wv = guess_amp * np.sin(np.pi * xg)
    for it in range(50):
        wl = np.r_[0, wv[:-1]]; wr = np.r_[wv[1:], 0]
        F = -wl + 2 * wv - wr + h * h * fb(xg, wv, 0); d = 2 + h * h * fby(xg, wv, 0)
        dv = tridiagonal_solve(-np.ones(N - 1), d, -np.ones(N - 1), -F); wv = wv + dv
        if np.max(np.abs(dv)) < 1e-12:
            break
    return wv.max(), it + 1
print("Ex 11.4.2 Bratu FD: guess 0.1 ->", bratu_fd(0.1), " guess 4 ->", bratu_fd(4.0))
for N in (4, 9, 19, 39):
    x, w, _ = bvp.nonlinear_fd(fnv, fyv, fypv, 1, 2, 0.5, 1 / 3, N)
    print(f"Ex 11.4.3 N={N}: max err {np.max(np.abs(w - exn(x))):.3e}")
print("Ex 11.4.4 Jacobian at initial guess (N=4): diag", 2 + 0.04 * 6 * (0.5 + (1 / 3 - 0.5) * (1 + 0.2 * np.arange(1, 5) - 1)) ** 2)
x, w, _ = bvp.nonlinear_shooting(fn, fy, fyp, 1, 2, 0.5, 1 / 3, 10, t0=0.0)
x2, w2, _ = bvp.nonlinear_fd(fnv, fyv, fypv, 1, 2, 0.5, 1 / 3, 9)
print("Ex 11.4.5 shooting (RK4) max err", np.max(np.abs(w - exn(x))), "FD max err", np.max(np.abs(w2 - exn(x2))))

print("\n=== 11.5 Rayleigh-Ritz / FEM ===")
def fem(n, qv, f):
    h = 1 / (n + 1); xs = h * np.arange(n + 2)
    main = 2 / h + qv * 2 * h / 3; off = -1 / h + qv * h / 6
    K = np.diag([main] * n) + np.diag([off] * (n - 1), 1) + np.diag([off] * (n - 1), -1)
    from scipy import integrate
    b = np.array([integrate.quad(lambda t: f(t) * (1 - abs(t - xs[i]) / h), xs[i] - h, xs[i] + h)[0] for i in range(1, n + 1)])
    c = np.linalg.solve(K, b)
    return xs, np.r_[0, c, 0], K, b
xs, c, K, b = fem(3, 0.0, lambda t: math.pi ** 2 * math.sin(math.pi * t))
print("Ex 11.5.1 -y''=pi^2 sin(pi x), n=3: K\n", K, "\n b", b, "\n c", c[1:-1], "exact", np.sin(math.pi * xs[1:-1]))
exf = lambda x: x - np.sinh(x) / math.sinh(1)
for n in (3, 7, 15):
    xs, c, K, b = fem(n, 1.0, lambda t: t)
    print(f"Ex 11.5.2 -y''+y=x n={n}: nodal max err {np.max(np.abs(c - exf(xs))):.3e}")
xs, c, K, b = fem(3, 0.0, lambda t: math.pi ** 2 * math.sin(math.pi * t))
Jf = lambda v: 0.5 * v @ K @ v - b @ v
print("Ex 11.5.3 energy at FE solution", Jf(c[1:-1]), "at exact nodal values", Jf(np.sin(math.pi * xs[1:-1])), "perturbed", Jf(c[1:-1] + 0.01))
for n in (3, 7, 15, 31):
    xs, c, K, b = fem(n, 0.0, lambda t: math.pi ** 2 * math.sin(math.pi * t))
    tt = np.linspace(0, 1, 2001); interp = np.interp(tt, xs, c)
    print(f"Ex 11.5.4/5 n={n}: nodal err {np.max(np.abs(c - np.sin(math.pi * xs))):.1e}  global max err {np.max(np.abs(interp - np.sin(math.pi * tt))):.3e}")

print("\n=== 11.6 DS ===")
rng = np.random.default_rng(0)
n = 200; t = np.linspace(0, 1, n); truth = np.sin(2 * np.pi * t) + 0.5 * t; yv = truth + 0.3 * rng.standard_normal(n)
D = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
for lamb in (1.0, 100.0, 1e4, 1e6):
    z = spla.spsolve((sp.identity(n) + lamb * D.T @ D).tocsc(), yv)
    print(f"Ex 11.6.1 Whittaker lambda={lamb:g}: RMSE vs truth {np.sqrt(np.mean((z - truth) ** 2)):.4f}")
T = 120; tt = np.arange(T); trend = 100 + 0.3 * tt + 5 * np.sin(2 * np.pi * tt / 60); cyc = 2 * np.sin(2 * np.pi * tt / 12)
series = trend + cyc + 0.5 * rng.standard_normal(T)
D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(T - 2, T))
tau = spla.spsolve((sp.identity(T) + 1600 * D2.T @ D2).tocsc(), series)
print("Ex 11.6.2 HP filter: trend RMSE", np.sqrt(np.mean((tau - trend) ** 2)), "cycle corr", np.corrcoef(series - tau, cyc)[0, 1])
N = 9; L = np.diag([2.0] * N) - np.eye(N, k=1) - np.eye(N, k=-1)
print("Ex 11.6.3 harmonic interpolation between 0 and 1:", tridiagonal_solve(-np.ones(N - 1), 2 * np.ones(N), -np.ones(N - 1), np.r_[np.zeros(N - 1), 1.0]))
xm = np.array([2.0, 4, 6, 8]); Tm = np.round(Tex(xm) + np.array([0.4, -0.3, 0.5, -0.2]), 1); print("   measured T", Tm)
def res(p):
    xg, wg, _ = bvp.linear_shooting(p0, lambda x: p[0] + 0 * x, lambda x: -p[0] * Ta + 0 * x, 0, 10, 40.0, 200.0, 20)
    return np.interp(xm, xg, wg) - Tm
fit = least_squares(res, [0.005], bounds=([1e-5], [1.0]))
print("Ex 11.6.4 fitted heat-loss coefficient h'", fit.x, "residuals", fit.fun)
for N in (9, 49, 199):
    h = 1 / (N + 1); Lm = (np.diag([2.0] * N) - np.eye(N, k=1) - np.eye(N, k=-1)) / h ** 2
    ev = np.sort(np.linalg.eigvalsh(Lm))[:3]
    print(f"Ex 11.6.5 N={N}: lowest eigenvalues {ev} vs (k pi)^2 = {[(k * math.pi) ** 2 for k in (1, 2, 3)]}")
