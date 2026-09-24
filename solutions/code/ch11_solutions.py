"""Numerical answers for exercises/ch11_exercises.md."""
import os
import sys

import numpy as np
from scipy import integrate, optimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import bvp  # noqa: E402
from numlib.linalg_direct import tridiagonal_solve  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
zero = lambda x: 0.0 * np.asarray(x, dtype=float)

# ---------- B ----------
# B1: y'' = 4y, y(0) = 1, y(1) = 2
c = (2 - np.cosh(2)) / (np.sinh(2) / 2)
print("B1 y1(1)=cosh2", np.cosh(2), " y2(1)=sinh2/2", np.sinh(2) / 2, " c", c, " y'(0) = c =", c)
xs, w, wp = bvp.linear_shooting(zero, lambda x: 4.0 + zero(x), zero, 0, 1, 1, 2, 4)
ex = lambda x: np.cosh(2 * x) + c * np.sinh(2 * x) / 2
print("   RK4 h=0.25:", w, " exact", ex(xs), " max err", np.abs(w - ex(xs)).max())

# B2: y'' = -y + x, y(0)=0, y(1)=2, exact x + sin x / sin 1
ex2 = lambda x: x + np.sin(x) / np.sin(1)
x, w = bvp.linear_fd(zero, lambda x: -1.0 + zero(x), lambda x: np.asarray(x, float), 0, 1, 0, 2, 3)
print("B2 FD h=0.25 matrix diag", 2 - 0.0625, " w", w, " exact", ex2(x), " errors", np.abs(w - ex2(x)))
for N in (3, 7, 15, 31):
    x, w = bvp.linear_fd(zero, lambda x: -1.0 + zero(x), lambda x: np.asarray(x, float), 0, 1, 0, 2, N)
    print(f"   h=1/{N+1}: max err {np.abs(w - ex2(x)).max():.3e}")

# B3: y'' = -2y', y(0)=0, y(1)=1
ex3 = lambda x: (1 - np.exp(-2 * x)) / (1 - np.exp(-2))
x, w = bvp.linear_fd(lambda x: -2.0 + zero(x), zero, zero, 0, 1, 0, 1, 3)
print("B3 FD h=0.25: w", w, " exact", ex3(x), " sub/super diag", -1 - 0.125 * -2, -1 + 0.125 * -2)

# B4: convection-dominated y'' = 50 y'
ex4 = lambda x: np.expm1(50 * x) / np.expm1(50)
for N in (9, 49):
    x, w = bvp.linear_fd(lambda x: 50.0 + zero(x), zero, zero, 0, 1, 0, 1, N)
    h = 1 / (N + 1)
    print(f"B4 central FD h={h}: h|p|/2 = {h*25:.2f}  min w {w.min():.4f}  w at last 4 nodes {w[-5:-1]}  exact {ex4(x[-5:-1])}  sign changes {int(np.sum(np.diff(np.sign(w[1:-1])) != 0))}")
    # upwind: -y'' + p y' = 0 with p > 0 is convection to the right, so y' ~ backward difference
    n = N; hh = h; p = 50.0
    diag = np.full(n, 2 + hh * p); lower = np.full(n - 1, -1 - hh * p); upper = np.full(n - 1, -1.0)
    rhs = np.zeros(n); rhs[-1] = 1.0
    wu = tridiagonal_solve(lower, diag, upper, rhs)
    print(f"   upwind: min {wu.min():.4f}, last 4 nodes {wu[-4:]}, max err {np.abs(wu - ex4(x[1:-1])).max():.4f}, central max err {np.abs(w - ex4(x)).max():.4f}")

# B5: y'' = -(y')^2, y(0)=0, y(1)=ln2, exact ln(1+x)
f = lambda x, y, yp: -yp**2
fy = lambda x, y, yp: 0.0 * yp
fyp = lambda x, y, yp: -2 * yp
x, w, slopes = bvp.nonlinear_shooting(f, fy, fyp, 0, 1, 0, np.log(2), 10)
print("B5 Newton shooting slopes", np.round(slopes, 10), " exact slope 1; max err", np.abs(w - np.log1p(x)).max())
t = np.log(2)
sol = integrate.solve_ivp(lambda x, u: [u[1], -u[1]**2, u[3], -2 * u[1] * u[3]], (0, 1), [0, t, 0, 1], rtol=1e-12, atol=1e-12)
print("   first shot t0=ln2: y(1) =", sol.y[0, -1], " z(1) =", sol.y[2, -1], " t1 = t0 - (y(1)-ln2)/z(1) =", t - (sol.y[0, -1] - np.log(2)) / sol.y[2, -1],
      " analytic y(1;t) = ln(1+t):", np.log1p(t), " z = 1/(1+t):", 1 / (1 + t))

# B6: nonlinear FD same problem h = 0.25
x, w, its = bvp.nonlinear_fd(lambda x, y, yp: -yp**2, lambda x, y, yp: 0 * yp, lambda x, y, yp: -2 * yp, 0, 1, 0, np.log(2), 3)
print("B6 nonlinear FD h=0.25:", its, "Newton its, w", w, " exact", np.log1p(x), " max err", np.abs(w - np.log1p(x)).max())
# first Newton step by hand
h = 0.25; xi = np.array([0.25, 0.5, 0.75]); w0 = np.log(2) * xi
wl = np.r_[0, w0[:-1]]; wr = np.r_[w0[1:], np.log(2)]; yp = (wr - wl) / (2 * h)
F = -wl + 2 * w0 - wr + h * h * (-yp**2)
J = np.diag(np.full(3, 2.0)) + np.diag(-1 + h / 2 * (-2 * yp[:-1]), 1) + np.diag(-1 - h / 2 * (-2 * yp[1:]), -1)
print("   w0", w0, " F(w0)", F, "\n   J\n", J, "\n   w1", w0 + np.linalg.solve(J, -F))
for N in (3, 7, 15, 31):
    x, w, its = bvp.nonlinear_fd(lambda x, y, yp: -yp**2, lambda x, y, yp: 0 * yp, lambda x, y, yp: -2 * yp, 0, 1, 0, np.log(2), N)
    print(f"   h=1/{N+1}: max err {np.abs(w - np.log1p(x)).max():.3e} ({its} its)")

# B7: -y'' + y = 1, y(0)=y(1)=0, P1 FEM with 3 interior nodes
exact7 = lambda x: 1 - np.cosh(x - 0.5) / np.cosh(0.5)
for n in (3, 7, 15):
    h = 1 / (n + 1)
    K = (np.diag(np.full(n, 2.0)) - np.eye(n, k=1) - np.eye(n, k=-1)) / h + h / 6 * (np.diag(np.full(n, 4.0)) + np.eye(n, k=1) + np.eye(n, k=-1))
    b = np.full(n, h); cc = np.linalg.solve(K, b); xn = h * np.arange(1, n + 1)
    if n == 3:
        print("B7 stiffness+mass matrix\n", K, "\n   load", b, "\n   c", cc, " exact", exact7(xn))
    print(f"   n={n}: nodal max err {np.abs(cc - exact7(xn)).max():.3e}")

# ---------- C ----------
print("C1 convergence study for y'' = -y + x (B2)")
for N in (4, 8, 16, 32, 64):
    xs, w, _ = bvp.linear_shooting(zero, lambda x: -1.0 + zero(x), lambda x: np.asarray(x, float), 0, 1, 0, 2, N)
    x, wf = bvp.linear_fd(zero, lambda x: -1.0 + zero(x), lambda x: np.asarray(x, float), 0, 1, 0, 2, N - 1)
    print(f"   h=1/{N}: shooting(RK4) max err {np.abs(w - ex2(xs)).max():.2e}   FD max err {np.abs(wf - ex2(x)).max():.2e}")
vals = []
for N in (4, 8, 16):
    x, wf = bvp.linear_fd(zero, lambda x: -1.0 + zero(x), lambda x: np.asarray(x, float), 0, 1, 0, 2, N - 1)
    vals.append(wf[np.argmin(abs(x - 0.5))])
R1 = [(4 * vals[i + 1] - vals[i]) / 3 for i in range(2)]; R2 = (16 * R1[1] - R1[0]) / 15
print("   Richardson at x=0.5:", np.round(vals, 10), np.round(R1, 10), round(R2, 12), " exact", ex2(0.5), " errors", [f"{abs(v - ex2(0.5)):.1e}" for v in vals + R1 + [R2]])

print("C2 Bratu problem y'' + lam e^y = 0")
# analytic: y = -2 ln( cosh((x-1/2) th/2) / cosh(th/4) ),  th = sqrt(2 lam) cosh(th/4)
lam_of = lambda th: 2 * (th / np.cosh(th / 4))**2 / 4  # from th = sqrt(2 lam) cosh(th/4)
res = optimize.minimize_scalar(lambda th: -lam_of(th), bounds=(0.1, 10), method="bounded")
print(f"   critical lambda {lam_of(res.x):.9f} at theta {res.x:.6f}, max y there {-2*np.log(1/np.cosh(res.x/4)):.6f}")
for lam in (1.0, 2.0, 3.0, 3.5):
    ths = [optimize.brentq(lambda th: lam_of(th) - lam, a, b_) for a, b_ in ((1e-6, res.x), (res.x, 20))]
    ymax = [-2 * np.log(1 / np.cosh(th / 4)) for th in ths]
    out = []
    for guess_amp in (0.1, 6.0):
        N = 199; h = 1 / (N + 1); xi = h * np.arange(1, N + 1); wv = guess_amp * np.sin(np.pi * xi)
        for k in range(50):
            wl = np.r_[0, wv[:-1]]; wr = np.r_[wv[1:], 0]
            F = -wl + 2 * wv - wr - h * h * lam * np.exp(wv)
            v = tridiagonal_solve(-np.ones(N - 1), 2 - h * h * lam * np.exp(wv), -np.ones(N - 1), -F); wv += v
            if np.abs(v).max() < 1e-12:
                break
        out.append((wv.max(), k + 1))
    print(f"   lambda {lam}: analytic max y lower {ymax[0]:.6f} upper {ymax[1]:.6f} | FD Newton from 0.1 sin: {out[0][0]:.6f} ({out[0][1]} its), from 6 sin: {out[1][0]:.6f} ({out[1][1]} its)")

print("C3 P1 FEM for -((1+x^2) y')' + y = f with y = sin(pi x)")
pfun = lambda x: 1 + x**2
ffun = lambda x: -(2 * x * np.pi * np.cos(np.pi * x) - (1 + x**2) * np.pi**2 * np.sin(np.pi * x)) + np.sin(np.pi * x)
gx, gw = np.polynomial.legendre.leggauss(3)
prev = None
for n in (7, 15, 31, 63, 127):
    h = 1 / (n + 1); nodes = np.linspace(0, 1, n + 2)
    K = np.zeros((n + 2, n + 2)); b = np.zeros(n + 2)
    for e in range(n + 1):
        xl, xr = nodes[e], nodes[e + 1]; xq = (xl + xr) / 2 + h / 2 * gx; wq = h / 2 * gw
        phi = np.array([(xr - xq) / h, (xq - xl) / h]); dphi = np.array([-1 / h, 1 / h])
        for i in range(2):
            b[e + i] += np.sum(wq * ffun(xq) * phi[i])
            for j in range(2):
                K[e + i, e + j] += np.sum(wq * (pfun(xq) * dphi[i] * dphi[j] + phi[i] * phi[j]))
    cvec = np.zeros(n + 2); cvec[1:-1] = np.linalg.solve(K[1:-1, 1:-1], b[1:-1])
    # errors with fine quadrature
    l2 = h1 = 0.0
    for e in range(n + 1):
        xl, xr = nodes[e], nodes[e + 1]; xq = (xl + xr) / 2 + h / 2 * np.polynomial.legendre.leggauss(6)[0]; wq = h / 2 * np.polynomial.legendre.leggauss(6)[1]
        uh = cvec[e] + (cvec[e + 1] - cvec[e]) * (xq - xl) / h; duh = (cvec[e + 1] - cvec[e]) / h
        l2 += np.sum(wq * (np.sin(np.pi * xq) - uh)**2); h1 += np.sum(wq * (np.pi * np.cos(np.pi * xq) - duh)**2)
    errs = (np.abs(cvec - np.sin(np.pi * nodes)).max(), np.sqrt(l2), np.sqrt(h1))
    rates = "" if prev is None else "  rates " + " ".join(f"{np.log2(p_ / e_):.2f}" for p_, e_ in zip(prev, errs))
    print(f"   h=1/{n+1}: nodal max {errs[0]:.2e}  L2 {errs[1]:.2e}  H1-semi {errs[2]:.2e}{rates}")
    prev = errs

print("C4 shooting vs FD for y'' = k^2 y, y(0)=1, y(1)=e^{-k}")
for k in (10, 20, 40):
    exk = lambda x: np.exp(-k * x)
    xs, w, _ = bvp.linear_shooting(zero, lambda x: k * k + zero(x), zero, 0, 1, 1, np.exp(-k), 1000)
    x, wf = bvp.linear_fd(zero, lambda x: k * k + zero(x), zero, 0, 1, 1, np.exp(-k), 999)
    print(f"   k={k}: y2(1)=sinh(k)/k = {np.sinh(k)/k:.2e}; shooting max err {np.abs(w - exk(xs)).max():.2e}, rel err at x=1 {abs(w[-1]-exk(1))/exk(1):.1e}; FD max err {np.abs(wf - exk(x)).max():.2e}")

# ---------- D ----------
print("D1 Whittaker smoother with GCV")
rng = np.random.default_rng(0)
n = 300; tt = np.linspace(0, 1, n); truth = np.sin(4 * np.pi * tt) * np.exp(-2 * tt) + tt
y = truth + 0.25 * rng.standard_normal(n)
D2 = np.diff(np.eye(n), 2, axis=0)
best = None
for lam in 10.0 ** np.arange(-1, 7):
    S = np.linalg.inv(np.eye(n) + lam * D2.T @ D2); z = S @ y
    tr = np.trace(S); gcv = n * np.sum((y - z)**2) / (n - tr)**2; rmse = np.sqrt(np.mean((z - truth)**2))
    print(f"   lambda {lam:.0e}: edf {tr:6.2f}  GCV {gcv:.5f}  RMSE vs truth {rmse:.4f}")
    if best is None or gcv < best[1]:
        best = (lam, gcv, rmse)
print("   GCV choice", best)

print("D2 identifiability in a reaction-diffusion calibration: -D c'' + k c = 0, c(0)=1, c(1)=0")
xm = np.linspace(0.1, 0.9, 9)
def model(D, k, x):
    m = np.sqrt(k / D); return np.sinh(m * (1 - x)) / np.sinh(m)
def model_fd(D, k, x, N=199):
    xg, w = bvp.linear_fd(zero, lambda s: k / D + zero(s), zero, 0, 1, 1, 0, N)
    return np.interp(x, xg, w)
rng = np.random.default_rng(1)
data = model(0.5, 8.0, xm) + 0.005 * rng.standard_normal(len(xm))
fit = optimize.least_squares(lambda p: model_fd(np.exp(p[0]), np.exp(p[1]), xm) - data, x0=[0.0, 0.0])
Jm = fit.jac
print(f"   true (D,k) = (0.5, 8), sqrt(k/D) = 4.0; fit D {np.exp(fit.x[0]):.4g} k {np.exp(fit.x[1]):.4g}", " ratio sqrt(k/D)", np.sqrt(np.exp(fit.x[1] - fit.x[0])).round(4),
      "\n   singular values of the Jacobian (log-params)", np.linalg.svd(Jm, compute_uv=False), " -> rank-deficient: only k/D identifiable")
fit2 = optimize.least_squares(lambda p: model_fd(np.exp(p[0]), np.exp(p[1]), xm) - data, x0=[1.0, 2.0])
print(f"   another start gives D {np.exp(fit2.x[0]):.4g} k {np.exp(fit2.x[1]):.4g}", " ratio", np.sqrt(np.exp(fit2.x[1] - fit2.x[0])).round(4), " cost", fit.cost, fit2.cost)
fit3 = optimize.least_squares(lambda p: model_fd(1.0, p[0]**2, xm) - data, x0=[1.0])
print("   reparametrised m = sqrt(k/D):", fit3.x.round(4), " SE", np.sqrt(np.linalg.inv(fit3.jac.T @ fit3.jac)[0, 0] * 2 * fit3.cost / (len(xm) - 1)).round(4))

print("D3 drifted Brownian motion: ruin probability and expected exit time")
mu, sig, L = 0.3, 1.0, 5.0
N = 499; h = L / (N + 1); xg = h * np.arange(N + 2)
p_ = 2 * mu / sig**2
xr, u = bvp.linear_fd(lambda x: -p_ + zero(x), zero, zero, 0, L, 1.0, 0.0, N)
_, T = bvp.linear_fd(lambda x: -p_ + zero(x), zero, lambda x: -2 / sig**2 + zero(x), 0, L, 0.0, 0.0, N)
u_ex = lambda x: (np.exp(-p_ * x) - np.exp(-p_ * L)) / (1 - np.exp(-p_ * L))
T_ex = lambda x: (L * (1 - np.exp(-p_ * x)) / (1 - np.exp(-p_ * L)) - x) / mu
x0 = 2.0; i0 = np.argmin(abs(xr - x0))
print(f"   FD: ruin prob u(2) {u[i0]:.6f} (exact {u_ex(x0):.6f}), E[T](2) {T[i0]:.6f} (exact {T_ex(x0):.6f})")
rng = np.random.default_rng(2)
def exit_mc(M, dt, bridge):
    pos = np.full(M, x0); tex = np.zeros(M); alive = np.ones(M, bool); ruin = np.zeros(M, bool); tcur = 0.0
    while alive.any():
        idx = np.nonzero(alive)[0]; old = pos[idx]
        new = old + mu * dt + sig * np.sqrt(dt) * rng.standard_normal(len(idx)); tcur += dt
        h0 = new <= 0; hL = new >= L
        if bridge:  # Brownian-bridge probability of an unobserved crossing between the two grid times
            inside = ~(h0 | hL); u_ = rng.random(len(idx))
            p0 = np.exp(-2 * old * np.maximum(new, 0) / (sig**2 * dt)); pL = np.exp(-2 * (L - old) * np.maximum(L - new, 0) / (sig**2 * dt))
            h0 |= inside & (u_ < p0); hL |= inside & ~h0 & (u_ >= p0) & (u_ < p0 + pL)
        pos[idx] = new
        ruin[idx[h0]] = True; tex[idx[h0 | hL]] = tcur; alive[idx[h0 | hL]] = False
    return ruin, tex
M = 20000
for dt, bridge in ((1e-2, False), (1e-3, False), (1e-2, True)):
    ruin, tex = exit_mc(M, dt, bridge)
    print(f"   Monte Carlo ({M} paths, dt={dt}, bridge={bridge}): ruin {ruin.mean():.4f} +- {np.sqrt(ruin.mean()*(1-ruin.mean())/M):.4f}, E[T] {tex.mean():.3f} +- {tex.std()/np.sqrt(M):.3f}")
