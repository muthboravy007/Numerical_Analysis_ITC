"""Chapter 11 worked examples — run:  python examples/ch11_examples.py"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import bvp  # noqa: E402
from numlib.linalg_direct import tridiagonal_solve  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
zero = lambda x: 0.0 * np.asarray(x, dtype=float)

print("Example 11.1 — hanging cable (catenary) by nonlinear shooting")
a = 2.0
f = lambda x, y, yp: np.sqrt(1 + yp ** 2) / a
fy = lambda x, y, yp: 0 * yp
fyp = lambda x, y, yp: yp / (a * np.sqrt(1 + yp ** 2))
x, w, slopes = bvp.nonlinear_shooting(f, fy, fyp, -1, 1, 3.0, 3.0, 40, t0=0.0)
exact = lambda x: a * np.cosh(x / a) + (3 - a * np.cosh(1 / a))
print("  slopes", np.round(slopes, 10), " sag at centre:", 3 - w[20], " exact", 3 - exact(0.0), " max error", np.abs(w - exact(x)).max())

print("Example 11.2 — catalyst pellet: y'' = phi^2 y, y'(0) = 0, y(1) = 1 (ghost-point FD)")
for phi in (1.0, 5.0):
    for N in (10, 20, 40):
        h = 1 / N; n = N  # unknowns y_0..y_{N-1}
        diag = np.full(n, 2 + (h * phi) ** 2); lower = -np.ones(n - 1); upper = -np.ones(n - 1); upper[0] = -2.0
        rhs = np.zeros(n); rhs[-1] = 1.0
        y = tridiagonal_solve(lower, diag, upper, rhs)
        eta = (3 * 1.0 - 4 * y[-1] + y[-2]) / (2 * h) / phi ** 2  # effectiveness = y'(1)/phi^2 with a 3-point one-sided difference
        print(f"  phi={phi}, h=1/{N}: centre y(0) {y[0]:.6f} (exact {1 / np.cosh(phi):.6f}), effectiveness {eta:.6f} (exact tanh(phi)/phi = {np.tanh(phi) / phi:.6f})")

print("Example 11.3 — optimal control as a two-point BVP")
Tf = 2.0
# minimise int_0^T (x^2 + u^2) dt, x' = u, x(0) = 1 -> x'' = x, x(0) = 1, x'(T) = 0
N = 200; h = Tf / N
n = N  # unknowns x_1..x_N, Neumann at T via ghost point x_{N+1} = x_{N-1}
diag = np.full(n, 2 + h * h); lower = -np.ones(n - 1); upper = -np.ones(n - 1); lower[-1] = -2.0
rhs = np.zeros(n); rhs[0] = 1.0
xs = np.r_[1.0, tridiagonal_solve(lower, diag, upper, rhs)]; tg = np.linspace(0, Tf, N + 1)
xe = np.cosh(Tf - tg) / np.cosh(Tf); u = np.gradient(xs, h)
J = np.trapezoid(xs ** 2 + u ** 2, tg)
print(f"  max |x - exact| {np.abs(xs - xe).max():.2e}; optimal cost {J:.6f} (exact tanh(T) = {np.tanh(Tf):.6f}); u(0) = {u[0]:.4f} (exact {-np.tanh(Tf):.4f})")
print(f"  compare do-nothing u=0: cost {Tf:.1f}; naive linear decay to 0 at T: cost {np.trapezoid((1 - tg / Tf) ** 2 + (1 / Tf) ** 2, tg):.4f}")

print("Example 11.4 — composite wall: arithmetic vs harmonic averaging of conductivity")
k = lambda x: np.where(x < 0.5, 1.0, 10.0)
for N in (10, 20, 40):  # N+1 intervals (odd), so the interface x = 0.5 lies midway between two nodes
    h = 1 / (N + 1); xg = np.linspace(0, 1, N + 2)
    kn = k(xg)  # conductivity known only at the nodes
    ka = (kn[:-1] + kn[1:]) / 2
    kh = 2 * kn[:-1] * kn[1:] / (kn[:-1] + kn[1:])
    Tex = np.where(xg <= 0.5, 100 - 100 * xg / 0.55, (100 / 0.55) * 0.1 * (1 - xg))
    for name, kk in (("arithmetic mean", ka), ("harmonic mean", kh)):
        diag = kk[:-1] + kk[1:]; off = -kk[1:-1]; rhs = np.zeros(N); rhs[0] = kk[0] * 100.0
        T = np.r_[100.0, tridiagonal_solve(off, diag, off, rhs), 0.0]
        flux = kk[0] * (T[0] - T[1]) / h
        print(f"  N={N}, {name:15s}: max error {np.abs(T - Tex).max():.2e}, heat flux {flux:.4f} (exact {100 / 0.55:.4f})")

print("Example 11.5 — vibrating string: FD eigenvalues of -y'' = lambda y")
for N in (10, 20, 40, 80):
    h = 1 / (N + 1); ev = np.sort(2 * (1 - np.cos(np.pi * np.arange(1, N + 1) / (N + 1))) / h ** 2)
    evn = np.linalg.eigvalsh((2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) / h ** 2)
    ex = (np.pi * np.arange(1, 4)) ** 2
    print(f"  N={N}: lambda_1..3 {evn[:3].round(4)}  rel. errors {((ex - evn[:3]) / ex).round(5)}  (highest mode rel. err {((np.pi * N) ** 2 - evn[-1]) / (np.pi * N) ** 2:.3f})")
print("  exact pi^2 k^2:", ((np.pi * np.arange(1, 4)) ** 2).round(4))
