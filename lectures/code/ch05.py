"""Reproduces every numerical example in Chapter 5 (Initial-Value Problems for ODEs).
Run from the repository root:  python lectures/code/ch05.py"""

import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import ode  # noqa: E402

np.set_printoptions(precision=7, suppress=True, linewidth=130)
f = lambda t, y: y - t ** 2 + 1
ex = lambda t: (t + 1) ** 2 - 0.5 * np.exp(t)

print("=== 5.1 Theory ===")
print("Ex 5.1.3 two solutions of y'=y^(1/3): y=0 and y=(2t/3)^1.5 at t=1:", (2 / 3) ** 1.5)
print("Ex 5.1.4 y'=y^2, y(0)=1: y=1/(1-t); at t=0.9, 0.99:", 1 / 0.1, 1 / 0.01)
for eps in (1e-3, 1e-2):
    print(f"Ex 5.1.5 perturb y0 by {eps}: change at t=2 = {eps * math.exp(2):.5f}")

print("\n=== 5.2 Euler ===")
t, Y = ode.euler(f, (0, 2), 0.5, 0.2)
M = abs(0.5 * math.exp(2) - 2)
for ti, wi in zip(t, Y[:, 0]):
    print(f"Ex 5.2.1 t={ti:.1f} w={wi:.7f} y={ex(ti):.7f} err={abs(wi - ex(ti)):.7f} bound={0.2 * M / 2 * (math.exp(ti) - 1):.7f}")
g = lambda t, y: -2 * t * y
for h in (0.1, 0.05, 0.025):
    t, Y = ode.euler(g, (0, 1), 1.0, h)
    print(f"Ex 5.2.2/3 y'=-2ty h={h}: w(1)={Y[-1, 0]:.7f} err={abs(Y[-1, 0] - math.exp(-1)):.2e}")
delta = 5e-7
print("Ex 5.2.4 optimal h = sqrt(2 delta / M):", math.sqrt(2 * delta / M), "M =", M)
lg = ode.logistic_growth(0.8, 100.0)
exl = lambda t: 100 / (1 + 99 * np.exp(-0.8 * t))
for h in (1.0, 0.5, 0.1):
    t, Y = ode.euler(lg, (0, 10), 1.0, h)
    print(f"Ex 5.2.5 logistic Euler h={h}: y(5)={Y[int(round(5 / h)), 0]:.4f} (exact {exl(5):.4f}) y(10)={Y[-1, 0]:.4f} (exact {exl(10):.4f})")
for alpha in (0.1, 0.5, 0.9, 1.1):
    w = [0.0]
    for _ in range(6):
        w.append(w[-1] - alpha * 2 * (w[-1] - 3))
    print(f"Ex 5.2.6 GD alpha={alpha}:", np.round(w, 4))

print("\n=== 5.3 Taylor methods ===")
t, Y2 = ode.taylor2(f, lambda t, y: -2 * t, lambda t, y: 1.0, (0, 2), 0.5, 0.2)
t, Y4 = ode.taylor4_linear_example((0, 2), 0.5, 0.2)
for ti, a, b in zip(t, Y2[:, 0], Y4[:, 0]):
    print(f"Ex 5.3.1/2 t={ti:.1f} T2={a:.7f} err {abs(a - ex(ti)):.2e}  T4={b:.7f} err {abs(b - ex(ti)):.2e}")
t, Yg = ode.taylor2(g, lambda t, y: -2 * y, lambda t, y: -2 * t, (0, 1), 1.0, 0.1)
print("Ex 5.3.3 Taylor2 y'=-2ty h=0.1: w(1)=", Yg[-1, 0], "err", abs(Yg[-1, 0] - math.exp(-1)))
for h in (0.2, 0.1, 0.05):
    e2 = abs(ode.taylor2(f, lambda t, y: -2 * t, lambda t, y: 1.0, (0, 2), 0.5, h)[1][-1, 0] - ex(2))
    e4 = abs(ode.taylor4_linear_example((0, 2), 0.5, h)[1][-1, 0] - ex(2))
    print(f"Ex 5.3.4 h={h}: T2 err {e2:.3e}  T4 err {e4:.3e}")
w0, h = 0.5, 0.2
w1 = ode.taylor4_linear_example((0, 0.2), 0.5, 0.2)[1][-1, 0]
print("Ex 5.3.5 one T4 step from exact y(0): local error", abs(w1 - ex(0.2)))

print("\n=== 5.4 Runge-Kutta ===")
for name, m in (("midpoint", ode.midpoint), ("modified Euler", ode.heun), ("Heun3", ode.heun3), ("RK4", ode.rk4)):
    t, Y = m(f, (0, 2), 0.5, 0.2)
    print(f"Ex 5.4.1 {name:14s} h=0.2: w(1)={Y[5, 0]:.7f} err {abs(Y[5, 0] - ex(1)):.2e}  w(2)={Y[-1, 0]:.7f} err {abs(Y[-1, 0] - ex(2)):.2e}")
t, Y = ode.rk4(f, (0, 2), 0.5, 0.2)
for ti, wi in zip(t, Y[:, 0]):
    print(f"Ex 5.4.2 RK4 t={ti:.1f} w={wi:.7f} err {abs(wi - ex(ti)):.2e}")
k1 = f(0, 0.5); k2 = f(0.1, 0.5 + 0.1 * k1); k3 = f(0.1, 0.5 + 0.1 * k2); k4 = f(0.2, 0.5 + 0.2 * k3)
print("   first step k's:", k1, k2, k3, k4, "w1=", 0.5 + 0.2 / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
for name, m, h in (("Euler", ode.euler, 0.025), ("modified Euler", ode.heun, 0.05), ("RK4", ode.rk4, 0.1)):
    t, Y = m(f, (0, 2), 0.5, h)
    print(f"Ex 5.4.3 equal work (80 evals): {name} h={h}: err at 2 = {abs(Y[-1, 0] - ex(2)):.2e}")
for h in (0.2, 0.1, 0.05):
    e = abs(ode.rk4(g, (0, 1), 1.0, h)[1][-1, 0] - math.exp(-1))
    print(f"Ex 5.4.4 RK4 y'=-2ty h={h}: err {e:.3e}")

print("\n=== 5.5 RKF45 ===")
t, Y, hs = ode.rkf45(f, (0, 2), 0.5, tol=1e-6, hmax=0.5, hmin=1e-4)
for ti, wi, hi in zip(t[1:], Y[1:, 0], hs):
    print(f"Ex 5.5.1 t={ti:.7f} h={hi:.7f} w={wi:.7f} err={abs(wi - ex(ti)):.1e}")
stf = lambda t, y: 10 * y * (1 - y)
t, Y, hs = ode.rkf45(stf, (0, 2), 1e-3, tol=1e-6, hmax=0.2, hmin=1e-6)
print("Ex 5.5.2 steep logistic: steps", len(hs), "min h", hs.min(), "at t~", t[1:][hs.argmin()], "max h", hs.max(),
      "final", Y[-1, 0], "exact", 1 / (1 + 999 * math.exp(-20)))
exs = lambda t: 1 / (1 + 999 * np.exp(-10 * t))
print("   max error", np.max(np.abs(Y[:, 0] - exs(t))))
sol = solve_ivp(stf, (0, 2), [1e-3], method="RK45", rtol=1e-6, atol=1e-9)
print("Ex 5.5.4 scipy RK45 nfev", sol.nfev, "steps", len(sol.t) - 1, "final", sol.y[0, -1])
h = 0.25
k1 = h * f(0, 0.5); k2 = h * f(h / 4, 0.5 + k1 / 4); k3 = h * f(3 * h / 8, 0.5 + 3 / 32 * k1 + 9 / 32 * k2)
k4 = h * f(12 * h / 13, 0.5 + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3)
k5 = h * f(h, 0.5 + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4)
k6 = h * f(h / 2, 0.5 - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5)
w4 = 0.5 + 25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - k5 / 5
w5 = 0.5 + 16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6
R = abs(w5 - w4) / h
q = 0.84 * (1e-6 / R) ** 0.25
print("Ex 5.5.3/5 one RKF step h=0.25: w4", w4, "w5", w5, "R", R, "true local err of w4", abs(w4 - ex(0.25)), "q", q, "new h", q * h)

print("\n=== 5.6 Multistep ===")
t, Yab = ode.adams_bashforth4(f, (0, 2), 0.5, 0.2)
t, Ypc = ode.adams_pc4(f, (0, 2), 0.5, 0.2)
for ti, a, b in zip(t, Yab[:, 0], Ypc[:, 0]):
    print(f"Ex 5.6.1/2 t={ti:.1f} AB4={a:.7f} err {abs(a - ex(ti)):.2e}  PC4={b:.7f} err {abs(b - ex(ti)):.2e}")
w = [0.5, ex(0.2)]
for i in range(1, 10):
    ti = 0.2 * i
    w.append(w[i] + 0.1 * (3 * f(ti, w[i]) - f(ti - 0.2, w[i - 1])))
print("Ex 5.6.3 AB2 h=0.2 (w1 exact): w(2)=", w[-1], "err", abs(w[-1] - ex(2)))
lam, h = -2.0, 0.1
wt = [1.0]
for i in range(10):
    wt.append(wt[-1] * (1 + h * lam / 2) / (1 - h * lam / 2))
print("Ex 5.6.4 AM2/trapezoid on y'=-2y h=0.1: w(1)=", wt[-1], "exact", math.exp(-2), "err", abs(wt[-1] - math.exp(-2)))
for h in (0.2, 0.1, 0.05):
    e = abs(ode.adams_pc4(f, (0, 2), 0.5, h)[1][-1, 0] - ex(2))
    e2 = abs(ode.adams_bashforth4(f, (0, 2), 0.5, h)[1][-1, 0] - ex(2))
    print(f"Ex 5.6.5 h={h}: AB4 err {e2:.3e}  PC4 err {e:.3e}")

print("\n=== 5.7 Systems ===")
osc = lambda t, u: np.array([u[1], -4 * u[0] - 0.5 * u[1]])
t, U = ode.rk4(osc, (0, 5), [1.0, 0.0], 0.1)
om = math.sqrt(4 - 0.0625)
exo = lambda t: np.exp(-0.25 * t) * (np.cos(om * t) + 0.25 / om * np.sin(om * t))
print("Ex 5.7.1 damped oscillator RK4 h=0.1: y(5)=", U[-1, 0], "exact", exo(5), "err", abs(U[-1, 0] - exo(5)))
t, U = ode.rk4(ode.lotka_volterra(1.0, 0.1, 1.5, 0.075), (0, 15), [10.0, 5.0], 0.01)
V = lambda x, z: 0.075 * x - 1.5 * math.log(x) + 0.1 * z - 1.0 * math.log(z)
print("Ex 5.7.2 Lotka-Volterra: prey range", U[:, 0].min(), U[:, 0].max(), "pred range", U[:, 1].min(), U[:, 1].max(),
      "invariant drift", abs(V(*U[-1]) - V(*U[0])))
t, U = ode.rk4(ode.sir(0.3, 0.1), (0, 160), [0.99, 0.01, 0.0], 0.5)
i = np.argmax(U[:, 1])
print("Ex 5.7.3 SIR: peak day", t[i], "peak I", U[i, 1], "final S", U[-1, 0], "final R", U[-1, 2])
from scipy.optimize import brentq
Sinf = brentq(lambda s: s - 0.99 * math.exp(-3 * (1 - s)), 1e-6, 0.5)
print("   final-size equation S_inf =", Sinf)
ho = lambda t, u: np.array([u[1], -u[0]])
for name, m in (("Euler", ode.euler), ("RK4", ode.rk4)):
    t, U = m(ho, (0, 20 * math.pi), [1.0, 0.0], 0.1)
    E = 0.5 * (U[:, 0] ** 2 + U[:, 1] ** 2)
    print(f"Ex 5.7.4 harmonic oscillator {name} h=0.1: energy at t=20pi {E[-1]:.6f} (start 0.5)")
th = lambda t, u: np.array([u[1], u[2], -2 * u[2] + u[1] + 2 * u[0] + np.exp(t) * 0])
t, U = ode.rk4(lambda t, u: np.array([u[1], u[2], 6 - 0 * u[0]]), (0, 1), [1.0, 0.0, 0.0], 0.1)
print("Ex 5.7.5 y'''=6 (y=1+t^3): y(1)=", U[-1, 0])

print("\n=== 5.8 Stability ===")
print("Ex 5.8.2 AB4 characteristic roots:", np.roots([1, -1, 0, 0, 0]))
h = 0.1
w = [1.0, math.exp(-h)]
for i in range(1, 30):
    w.append(-4 * w[i] + 5 * w[i - 1] + h * (4 * -w[i] + 2 * -w[i - 1]))
print("Ex 5.8.3 unstable 2-step on y'=-y: w at t=0.5,1,2,3:", w[5], w[10], w[20], w[30], "exact", math.exp(-0.5), math.exp(-1), math.exp(-2), math.exp(-3))
print("   roots of l^2+4l-5:", np.roots([1, 4, -5]))
w = [1.0, math.exp(-h)]
for i in range(1, 200):
    w.append(w[i - 1] + 2 * h * -w[i])
print("Ex 5.8.4 leapfrog on y'=-y h=0.1: w(5), w(10), w(20):", w[50], w[100], w[200], "exact", math.exp(-5), math.exp(-10), math.exp(-20))
print("   roots of l^2+2hl-1:", np.roots([1, 2 * h, -1]))
from scipy.optimize import brentq as bq
rk4R = lambda z: 1 + z + z * z / 2 + z ** 3 / 6 + z ** 4 / 24
print("Ex 5.8.5 RK4 real stability limit:", bq(lambda z: abs(rk4R(z)) - 1, -3, -2), "Euler -2, Heun:", bq(lambda z: abs(1 + z + z * z / 2) - 1, -3, -1))

print("\n=== 5.9 Stiff ===")
t, Y = ode.euler(lambda t, y: -50 * y, (0, 1), 1.0, 0.05)
t2, Y2 = ode.implicit_euler(lambda t, y: -50 * y, (0, 1), 1.0, 0.05)
print("Ex 5.9.1 Euler:", Y[:6, 0], "\n   implicit:", Y2[:6, 0])
sf = lambda t, y: -1000 * (y - np.cos(t)) - np.sin(t)
for h in (0.0025, 0.003, 0.01):
    t, Y = ode.rk4(sf, (0, 0.2), 1.0, h)
    print(f"Ex 5.9.2 RK4 h={h}: y(0.2)={Y[-1, 0]:.6g} exact {math.cos(0.2):.6f}")
for h in (0.01, 0.1):
    t, Y = ode.implicit_euler(sf, (0, 1), 1.0, h)
    t3, Y3 = ode.trapezoidal_linear(np.array([[-1000.0]]), (0, 1), 1.0, h)
    print(f"   implicit Euler h={h}: y(1)={Y[-1, 0]:.6f} err {abs(Y[-1, 0] - math.cos(1)):.2e}")
def rober(t, y):
    y1, y2, y3 = y
    return [-0.04 * y1 + 1e4 * y2 * y3, 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 ** 2, 3e7 * y2 ** 2]
for meth in ("RK45", "BDF", "Radau"):
    s = solve_ivp(rober, (0, 100), [1, 0, 0], method=meth, rtol=1e-6, atol=1e-10)
    print(f"Ex 5.9.3 Robertson {meth}: nfev {s.nfev}, steps {len(s.t) - 1}, y1(100)={s.y[0, -1]:.6f}")
A = np.array([[-1.0, 0], [0, -1000]])
for h in (0.1, 0.5):
    t, Y = ode.trapezoidal_linear(A, (0, 2), [1.0, 1.0], h)
    t, Ye = ode.euler(lambda t, y: A @ y, (0, 2), [1.0, 1.0], h)
    print(f"Ex 5.9.4 h={h}: trapezoid y(2)={Y[-1]} (exact {math.exp(-2):.6f}, ~0)  Euler y(2)={Ye[-1]}")
print("Ex 5.9.5 stiffness ratio 1000/1 =", 1000, "| Euler needs h <", 2 / 1000)

print("\n=== 5.10 DS applications ===")
rng = np.random.default_rng(0)
tt = np.arange(0, 60, 3.0)
true = solve_ivp(ode.sir(0.35, 0.12), (0, 60), [0.995, 0.005, 0], t_eval=tt, rtol=1e-8).y[1]
obs = true * (1 + 0.05 * rng.standard_normal(len(tt)))
def resid(p):
    s = solve_ivp(ode.sir(p[0], p[1]), (0, 60), [0.995, 0.005, 0], t_eval=tt, rtol=1e-8)
    return s.y[1] - obs
fit = least_squares(resid, [0.2, 0.2], bounds=([0, 0], [2, 2]))
print("Ex 5.10.1 SIR fit: beta,gamma =", fit.x, "R0 =", fit.x[0] / fit.x[1], "(true 0.35, 0.12, R0", 0.35 / 0.12, ")")
L = 10.0
for alpha in (0.05, 0.19, 0.21):
    w = np.array([1.0, 1.0])
    for _ in range(50):
        w = w - alpha * np.array([1.0, L]) * w
    print(f"Ex 5.10.2 GD on 0.5(w1^2+10 w2^2) alpha={alpha}: |w| after 50 = {np.linalg.norm(w):.3e}")
t = np.arange(0, 13.0)
adopt = 1000 / (1 + 49 * np.exp(-0.6 * t)) * (1 + 0.03 * rng.standard_normal(len(t)))
def r2(p):
    return p[0] / (1 + (p[0] / 20 - 1) * np.exp(-p[1] * t)) - adopt
fp = least_squares(r2, [500, 0.3]).x
print("Ex 5.10.3 logistic adoption fit K, r:", fp, "forecast t=20:", fp[0] / (1 + (fp[0] / 20 - 1) * math.exp(-fp[1] * 20)))
hres = 0.1
x = np.array([1.0, 0.0])
Wm = np.array([[0.0, -1.0], [1.0, 0.0]])
for _ in range(10):
    x = x + hres * np.tanh(Wm @ x)
print("Ex 5.10.4 10 residual blocks h=0.1:", x, "| RK4 of x'=tanh(Wx) to t=1:", ode.rk4(lambda t, u: np.tanh(Wm @ u), (0, 1), [1.0, 0.0], 0.1)[1][-1])
beta_m, alpha_m = 0.9, 0.02
w, v = np.array([1.0, 1.0]), np.zeros(2)
for _ in range(100):
    v = beta_m * v - alpha_m * np.array([1.0, 10.0]) * w
    w = w + v
wg = np.array([1.0, 1.0])
for _ in range(100):
    wg = wg - alpha_m * np.array([1.0, 10.0]) * wg
print("Ex 5.10.5 momentum |w| after 100:", np.linalg.norm(w), " plain GD:", np.linalg.norm(wg))
