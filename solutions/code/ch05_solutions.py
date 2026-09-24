"""Numerical answers for exercises/ch05_exercises.md."""
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import ode  # noqa: E402

f = lambda t, y: t * np.exp(-t) - y
ex = lambda t: 0.5 * t ** 2 * np.exp(-t)
t, Y = ode.euler(f, (0, 1), 0.0, 0.5); print("B1 Euler h=0.5:", Y.ravel(), "exact", ex(t))
t, Y = ode.euler(f, (0, 1), 0.0, 0.25); print("   h=0.25:", Y.ravel(), "exact", ex(t))
ypp = lambda t: np.exp(-t) * (1 - 2 * t + 0.5 * t ** 2)
tt = np.linspace(0, 1, 100001); M = np.max(np.abs(ypp(tt)))
print("B2 L=1, M=max|y''| =", M, "bound at t=1, h=0.25:", 0.25 * M / 2 * (math.e - 1), "actual Euler err", abs(ode.euler(f, (0, 1), 0.0, 0.25)[1][-1, 0] - ex(1.0)))
g = lambda t, y: y / t + t ** 2
exg = lambda t: t ** 3 / 2 + t / 2
t, Y = ode.taylor2(g, lambda t, y: -y / t ** 2 + 2 * t, lambda t, y: 1 / t, (1, 2), 1.0, 0.25)
print("B3 Taylor2:", Y.ravel(), "exact", exg(t))
h = 0.25; y0 = 1.0; t0 = 1.0
k1 = h * g(t0, y0); k2 = h * g(t0 + h / 2, y0 + k1 / 2); k3 = h * g(t0 + h / 2, y0 + k2 / 2); k4 = h * g(t0 + h, y0 + k3)
print("B4 RK4 one step k:", k1, k2, k3, k4, "y1", y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6, "exact", exg(1.25))
t, Y = ode.rk4(g, (1, 2), 1.0, 0.25); print("   RK4 full", Y.ravel(), "errors", np.abs(Y.ravel() - exg(t)))
t, Ym = ode.midpoint(g, (1, 2), 1.0, 0.25); t, Yh = ode.heun(g, (1, 2), 1.0, 0.25)
print("B5 midpoint y(2)", Ym[-1, 0], "modified Euler", Yh[-1, 0], "exact", exg(2.0))
t, Yab = ode.adams_bashforth4(g, (1, 2), 1.0, 0.2); t, Ypc = ode.adams_pc4(g, (1, 2), 1.0, 0.2)
print("B6 AB4", Yab.ravel(), "PC4", Ypc.ravel(), "exact", exg(t))
lam = -20.0
for h in (0.05, 0.1, 0.15):
    print(f"B7 y'=-20y h={h}: Euler factor {1 + h * lam:.2f}, RK4 factor {1 + h * lam + (h * lam) ** 2 / 2 + (h * lam) ** 3 / 6 + (h * lam) ** 4 / 24:.4f}, backward Euler factor {1 / (1 - h * lam):.4f}")
print("B8 root condition for w_{i+1} = w_{i-1} + 2h f_i: roots", np.roots([1, 0, -1]), "; for w_{i+1}=(4w_i - w_{i-1})/3 + 2h/3 f_{i+1} (BDF2):", np.roots([1, -4 / 3, 1 / 3]))
for name, m in (("Euler", ode.euler), ("Heun", ode.heun), ("RK4", ode.rk4)):
    errs = [abs(m(f, (0, 1), 0.0, h)[1][-1, 0] - ex(1.0)) for h in (0.1, 0.05, 0.025, 0.0125)]
    print(f"C1 {name}: errors {['%.2e' % e for e in errs]} orders {np.round(np.log2(np.array(errs[:-1]) / np.array(errs[1:])), 2)}")
t, Y, hs = ode.rkf45(f, (0, 1), 0.0, tol=1e-6, hmax=0.25, hmin=1e-6)
print("C2 RKF45 steps", len(hs), "h range", hs.min(), hs.max(), "max err", np.max(np.abs(Y[:, 0] - ex(t))))
pend = lambda t, u: np.array([u[1], -9.81 * np.sin(u[0])])
for th0 in (0.1, 1.0, 2.5):
    s = solve_ivp(pend, (0, 20), [th0, 0], rtol=1e-10, atol=1e-12, dense_output=True, events=lambda t, u: u[1])
    tz = s.t_events[0]; period = 2 * np.mean(np.diff(tz[tz > 0.01])) if len(tz) > 2 else np.nan
    print(f"C3 pendulum theta0={th0}: period {period:.5f} small-angle {2 * math.pi / math.sqrt(9.81):.5f}")
rob = lambda t, y: [-0.04 * y[0] + 1e4 * y[1] * y[2], 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2, 3e7 * y[1] ** 2]
for meth in ("RK45", "LSODA", "BDF"):
    s = solve_ivp(rob, (0, 40), [1, 0, 0], method=meth, rtol=1e-6, atol=1e-10)
    print(f"C4 Robertson to t=40 {meth}: nfev {s.nfev} y1 {s.y[0, -1]:.6f}")
seir = lambda b, s_, g: (lambda t, y: np.array([-b * y[0] * y[2], b * y[0] * y[2] - s_ * y[1], s_ * y[1] - g * y[2], g * y[2]]))
s = solve_ivp(seir(0.5, 0.2, 0.1), (0, 300), [0.999, 0, 0.001, 0], rtol=1e-8, dense_output=True)
tt = np.linspace(0, 300, 30001); I_ = s.sol(tt)[2]
print("D1 SEIR peak day", tt[np.argmax(I_)], "peak I", I_.max(), "final R", s.y[3, -1], "R0", 0.5 / 0.1)
for red in (0.0, 0.3, 0.5):
    s2 = solve_ivp(seir(0.5 * (1 - red), 0.2, 0.1), (0, 600), [0.999, 0, 0.001, 0], rtol=1e-8, dense_output=True)
    tt2 = np.linspace(0, 600, 60001); I2 = s2.sol(tt2)[2]
    print(f"   contact reduction {red}: peak day {tt2[np.argmax(I2)]:.1f} peak I {I2.max():.4f} attack rate {s2.y[3, -1]:.4f}")
rng = np.random.default_rng(0)
td = np.arange(0, 25.0, 2); ktrue, Vd = 0.25, 10.0
conc = 100 / Vd * np.exp(-ktrue * td) * (1 + 0.05 * rng.standard_normal(len(td)))
res = least_squares(lambda p: solve_ivp(lambda t, c: -p[0] * c, (0, 24), [100 / p[1]], t_eval=td, rtol=1e-10).y[0] - conc, [0.1, 5.0])
print("D2 PK fit k, V:", res.x, "half-life", math.log(2) / res.x[0])
F = lambda w: 0.5 * (w[0] ** 2 + 25 * w[1] ** 2)
for alpha in (0.01, 0.07, 0.079, 0.081):
    w = np.array([1.0, 1.0])
    for _ in range(200):
        w = w - alpha * np.array([w[0], 25 * w[1]])
    print(f"D3 GD alpha={alpha}: |w| after 200 = {np.linalg.norm(w):.3e}  (stability limit 2/25 = 0.08)")
def gd_flow(t, w): return -np.array([w[0], 25 * w[1]])
sflow = solve_ivp(gd_flow, (0, 2), [1.0, 1.0], rtol=1e-10, atol=1e-30, method="Radau"); print("   gradient flow w(2)", sflow.y[:, -1], "exact", [math.exp(-2), math.exp(-50)])
