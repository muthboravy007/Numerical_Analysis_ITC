"""Chapter 5 worked examples — run:  python examples/ch05_examples.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numlib import ode  # noqa: E402

np.set_printoptions(precision=6, suppress=True)

print("Example 5.1 — logistic user growth")
r, K, y0 = 0.8, 100.0, 1.0
f = lambda t, y: r * y * (1 - y / K)
exact = lambda t: K / (1 + (K / y0 - 1) * np.exp(-r * t))
for h in (1.0, 0.5, 0.25):
    t, Ye = ode.euler(f, (0, 10), [y0], h); _, Yr = ode.rk4(f, (0, 10), [y0], h)
    print(f"  h={h}: Euler y(10)={Ye[-1, 0]:.4f} err {abs(Ye[-1, 0] - exact(10)):.2e}   RK4 y(10)={Yr[-1, 0]:.6f} err {abs(Yr[-1, 0] - exact(10)):.2e}")
print("  exact y(10) =", exact(10), " inflection at t =", np.log(K / y0 - 1) / r)

print("Example 5.2 — repeated dosing (one-compartment model)")
k, V, D, tau = 0.2, 20.0, 100.0, 8.0
c = 0.0; hh = 0.1; troughs = []; peaks = []
fc = lambda t, y: -k * y
for dose in range(8):
    c += D / V; peaks.append(c)
    _, Yc = ode.rk4(fc, (0, tau), [c], hh); c = Yc[-1, 0]; troughs.append(c)
Css_peak = D / V / (1 - np.exp(-k * tau)); Css_trough = Css_peak * np.exp(-k * tau)
print("  peaks", np.round(peaks, 4), "\n  troughs", np.round(troughs, 4), f"\n  steady state theory: peak {Css_peak:.4f} trough {Css_trough:.4f}; accumulation factor {1 / (1 - np.exp(-k * tau)):.4f}")

print("Example 5.3 — adaptive RKF45 on a fast transient")
g = lambda t, y: -50 * (y - np.cos(t))
exact3 = lambda t: (2500 * np.cos(t) + 50 * np.sin(t)) / 2501 - (2500 / 2501) * np.exp(-50 * t)
t, Y, hs = ode.rkf45(g, (0, 2), [0.0], tol=1e-6, hmax=0.5)
hs = np.array(hs)
print(f"  accepted steps {len(hs)}; h on [0,0.1]: min {hs[np.array(t[:-1]) < 0.1].min():.4f} max {hs[np.array(t[:-1]) < 0.1].max():.4f}; h on [1,2]: mean {hs[np.array(t[:-1]) > 1].mean():.4f}; max error {np.abs(Y[:, 0] - exact3(np.array(t))).max():.2e}")
tr, Yr = ode.rk4(g, (0, 2), [0.0], 2 / len(hs))
print(f"  fixed-step RK4 with the same number of steps (h={2 / len(hs):.4f}): max error {np.abs(Yr[:, 0] - exact3(tr)).max():.2e}")

print("Example 5.4 — predator-prey: conserved quantity")
a, b, cc, d = 1.0, 0.1, 1.5, 0.075
lv = lambda t, z: np.array([a * z[0] - b * z[0] * z[1], -cc * z[1] + d * z[0] * z[1]])
Hq = lambda z: d * z[0] - cc * np.log(z[0]) + b * z[1] - a * np.log(z[1])
z0 = [10.0, 5.0]
for name, meth in (("Euler", ode.euler), ("Heun", ode.heun), ("RK4", ode.rk4)):
    t, Z = meth(lv, (0, 50), z0, 0.01)
    Hv = np.array([Hq(z) for z in Z])
    print(f"  {name}: relative drift of H over t in [0,50]: {abs(Hv[-1] - Hv[0]) / abs(Hv[0]):.2e}; final (prey, predator) {Z[-1].round(3)}")
t, Z = ode.rk4(lv, (0, 50), z0, 0.01)
i = np.argmax(Z[:, 0]); print(f"  prey peak {Z[:, 0].max():.2f} at t={t[i]:.2f}; equilibrium (c/d, a/b) = ({cc / d}, {a / b})")

print("Example 5.5 — stiffness: explicit vs implicit")
s = lambda t, y: -1000 * (y - np.sin(t)) + np.cos(t)
ex5 = lambda t: np.sin(t)
for h in (0.001, 0.0021, 0.01, 0.1):
    with np.errstate(all="ignore"):
        te, Ye = ode.euler(s, (0, 1), [1.0], h)
    ti, Yi = ode.implicit_euler(s, (0, 1), [1.0], h, jac=lambda t, y: np.array([[-1000.0]]))
    print(f"  h={h}: explicit Euler y(1) {Ye[-1, 0]:.6g}  implicit Euler y(1) {Yi[-1, 0]:.6f}  (exact sin 1 = {np.sin(1):.6f}; stability limit 2/1000 = 0.002)")

print("Example 5.6 — damped oscillator as a system; Adams predictor-corrector")
om, zeta = 2.0, 0.1
osc = lambda t, u: np.array([u[1], -2 * zeta * om * u[1] - om * om * u[0]])
wd = om * np.sqrt(1 - zeta ** 2)
exact6 = lambda t: np.exp(-zeta * om * t) * (np.cos(wd * t) + zeta * om / wd * np.sin(wd * t))
for h in (0.1, 0.05, 0.025):
    t, U = ode.adams_pc4(osc, (0, 10), [1.0, 0.0], h)
    _, U4 = ode.rk4(osc, (0, 10), [1.0, 0.0], h)
    print(f"  h={h}: Adams PC max error {np.abs(U[:, 0] - exact6(t)).max():.2e} (f-evals ~{2 * len(t)})   RK4 max error {np.abs(U4[:, 0] - exact6(t)).max():.2e} (f-evals {4 * (len(t) - 1)})")
