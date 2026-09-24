"""Numerical answers for exercises/ch12_exercises.md."""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import pde  # noqa: E402
from numlib.linalg_direct import tridiagonal_solve  # noqa: E402

np.set_printoptions(precision=6, suppress=True)

# ---------- A1 classification ----------
for name, (A, B, C) in {"u_xx+4u_xy+3u_yy": (1, 4, 3), "u_xx+2u_xy+u_yy": (1, 2, 1), "u_xx+u_xy+u_yy": (1, 1, 1)}.items():
    disc = B * B - 4 * A * C
    print(f"A1 {name}: B^2-4AC = {disc} -> {'hyperbolic' if disc > 0 else 'parabolic' if disc == 0 else 'elliptic'}")

# ---------- B ----------
# B1: Laplacian u = 4 on unit square, u = x^2 + y^2 on the boundary, h = 1/3
x, y, W = pde.poisson_rect(lambda x, y: 4.0, lambda x, y: x**2 + y**2, 0, 1, 0, 1, 3, 3)
X, Y = np.meshgrid(x, y, indexing="ij")
print("B1 interior W\n", W[1:3, 1:3], "\n exact\n", (X**2 + Y**2)[1:3, 1:3])

# B2: Laplace, u(x,1) = sin(pi x), other sides 0, h = 1/4
ex2 = lambda x, y: np.sinh(np.pi * y) * np.sin(np.pi * x) / np.sinh(np.pi)
g2 = lambda x, y: np.where(np.isclose(y, 1.0), np.sin(np.pi * np.asarray(x, float)), 0.0) + 0 * np.asarray(x, float)
for n in (4, 8, 16, 32):
    x, y, W = pde.poisson_rect(lambda x, y: 0.0, g2, 0, 1, 0, 1, n, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    if n == 4:
        print("B2 h=1/4 interior (rows x=0.25,0.5,0.75; cols y=0.25,0.5,0.75)\n", W[1:4, 1:4], "\n exact\n", ex2(X, Y)[1:4, 1:4])
    print(f"   h=1/{n}: max err {np.abs(W - ex2(X, Y)).max():.3e}   u(0.5,0.5) {W[n // 2, n // 2]:.6f} exact {ex2(0.5, 0.5):.6f}")

# B3: FTCS by hand
u0 = lambda x: np.sin(np.pi * x)
for k, N in ((0.025, 2), (0.05, 2), (0.05, 20)):
    x, w, lam = pde.heat_ftcs(u0, 1.0, 1.0, k * N, 4, N)
    print(f"B3 FTCS h=0.25 k={k} (lambda={lam}) after {N} steps t={k*N:.3f}: {w[1:4]}  exact {np.exp(-np.pi**2 * k * N) * np.sin(np.pi * x[1:4])}")
# growth of the sawtooth mode for lambda = 0.8
x = np.linspace(0, 1, 5); w = np.sin(np.pi * x) + 0.001 * np.array([0, 1, -1, 1, 0])
lam = 0.8
for s in range(20):
    w[1:-1] = (1 - 2 * lam) * w[1:-1] + lam * (w[:-2] + w[2:])
print("   lambda=0.8 with 0.001 sawtooth perturbation after 20 steps:", w[1:4], " amplification of highest mode per step: 1-4*lam*sin^2(3pi/8) =", 1 - 4 * lam * np.sin(3 * np.pi / 8)**2)

# B4: BTCS and CN one step
for meth in (pde.heat_btcs, pde.heat_crank_nicolson):
    x, w, lam = meth(u0, 1.0, 1.0, 0.05, 4, 1)
    print(f"B4 {meth.__name__} one step k=0.05 lambda={lam}: {w[1:4]} exact {np.exp(-np.pi**2 * 0.05) * np.sin(np.pi * x[1:4])}")
print("   amplification of mode sin(pi x): FTCS", 1 - 4 * 0.8 * np.sin(np.pi / 8)**2, " BTCS", 1 / (1 + 4 * 0.8 * np.sin(np.pi / 8)**2),
      " CN", (1 - 2 * 0.8 * np.sin(np.pi / 8)**2) / (1 + 2 * 0.8 * np.sin(np.pi / 8)**2), " exact e^{-pi^2 k}", np.exp(-np.pi**2 * 0.05))

# B5: wave u_tt = 4 u_xx, h = 0.25, k = 0.1 (lambda = 0.8)
x = np.linspace(0, 1, 5)
w0 = np.sin(np.pi * x); lam = 2 * 0.1 / 0.25
w1 = w0.copy(); w1[1:-1] = (1 - lam**2) * w0[1:-1] + lam**2 / 2 * (w0[2:] + w0[:-2])
w2 = w1.copy(); w2[1:-1] = 2 * (1 - lam**2) * w1[1:-1] + lam**2 * (w1[2:] + w1[:-2]) - w0[1:-1]
print("B5 lambda", lam, " w(t=0.1)", w1[1:4], "exact", np.sin(np.pi * x[1:4]) * np.cos(2 * np.pi * 0.1), " w(t=0.2)", w2[1:4], "exact", np.sin(np.pi * x[1:4]) * np.cos(2 * np.pi * 0.2))

# B6: amplification factors
print("B6 amplification factors g(theta) at theta = pi (highest mode), s = sin^2(theta/2) = 1:")
for lam in (0.4, 0.6, 5.0):
    print(f"   lambda {lam}: FTCS {1 - 4 * lam:.3f}  BTCS {1 / (1 + 4 * lam):.4f}  CN {(1 - 2 * lam) / (1 + 2 * lam):.4f}")

# ---------- C ----------
print("C1 Poisson on [0,2]x[0,1] with u = e^x sin(pi y) (h = 2/n, k = 1/m)")
f1 = lambda x, y: (1 - np.pi**2) * np.exp(x) * np.sin(np.pi * y)
g1 = lambda x, y: np.exp(x) * np.sin(np.pi * y)
prev = None
for n, m in ((8, 4), (16, 8), (32, 16), (64, 32), (128, 64)):
    x, y, W = pde.poisson_rect(f1, g1, 0, 2, 0, 1, n, m)
    X, Y = np.meshgrid(x, y, indexing="ij"); err = np.abs(W - g1(X, Y)).max()
    print(f"   n={n}, m={m}: unknowns {(n-1)*(m-1)}  max err {err:.3e}" + ("" if prev is None else f"  ratio {prev/err:.2f}"))
    prev = err

print("C2 heat equation convergence at t = 0.1, u0 = sin(pi x) + 0.5 sin(3 pi x)")
u0c = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
exc = lambda x, t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x) + 0.5 * np.exp(-9 * np.pi**2 * t) * np.sin(3 * np.pi * x)
T = 0.1
print("   (a) k = 0.4 h^2 (all three stable)")
for m in (10, 20, 40, 80):
    h = 1 / m; N = int(round(T / (0.4 * h * h)))
    errs = [np.abs(meth(u0c, 1.0, 1.0, T, m, N)[1] - exc(np.linspace(0, 1, m + 1), T)).max() for meth in (pde.heat_ftcs, pde.heat_btcs, pde.heat_crank_nicolson)]
    print(f"   m={m} N={N}: FTCS {errs[0]:.2e} BTCS {errs[1]:.2e} CN {errs[2]:.2e}")
print("   (b) h = 1/400 fixed, k varies (time error dominates)")
for N in (5, 10, 20, 40):
    errs = [np.abs(meth(u0c, 1.0, 1.0, T, 400, N)[1] - exc(np.linspace(0, 1, 401), T)).max() for meth in (pde.heat_btcs, pde.heat_crank_nicolson)]
    print(f"   k={T/N:.4f} (lambda={T/N*400**2:.0f}): BTCS {errs[0]:.2e} CN {errs[1]:.2e}")
print("   (c) rough data (step) with lambda = 50: CN vs Rannacher start")
m = 100; h = 1 / m; xg = np.linspace(0, 1, m + 1); step = (np.abs(xg - 0.5) < 0.25).astype(float)
exstep = lambda x, t: sum(2 / (np.pi * j) * (np.cos(np.pi * j / 4) - np.cos(3 * np.pi * j / 4)) * np.exp(-(np.pi * j)**2 * t) * np.sin(np.pi * j * x) for j in range(1, 2000))
k = 50 * h * h; n = m - 1
def cn_steps(w, nsteps, lam, rannacher=0):
    w = w.copy()
    for s in range(nsteps):
        if s < rannacher:  # two half-size backward Euler steps
            for _ in range(2):
                w = tridiagonal_solve(-lam / 2 * np.ones(n - 1), (1 + lam) * np.ones(n), -lam / 2 * np.ones(n - 1), w)
        else:
            rhs = (1 - lam) * w; rhs[1:] += lam / 2 * w[:-1]; rhs[:-1] += lam / 2 * w[1:]
            w = tridiagonal_solve(-lam / 2 * np.ones(n - 1), (1 + lam) * np.ones(n), -lam / 2 * np.ones(n - 1), rhs)
    return w
for nsteps in (1, 2, 5):
    ref = exstep(xg[1:-1], nsteps * k)
    wcn = cn_steps(step[1:-1], nsteps, 50.0); wra = cn_steps(step[1:-1], nsteps, 50.0, rannacher=2)
    print(f"   after {nsteps} steps (t={nsteps*k:.3f}): CN max err {np.abs(wcn - ref).max():.3e} (min value {wcn.min():.3f})  Rannacher {np.abs(wra - ref).max():.3e}")

print("C3 2-D heat equation u_t = Laplacian u on the unit square, m = 63 (3969 unknowns), to t = 0.05")
m = 63; h = 1 / (m + 1)
T1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m)); I = sp.identity(m)
L = (sp.kron(I, T1) + sp.kron(T1, I)).tocsc() / h**2
g = np.linspace(h, 1 - h, m); Xg, Yg = np.meshgrid(g, g, indexing="ij")
U0 = (np.sin(np.pi * Xg) * np.sin(2 * np.pi * Yg)).ravel(); Tend = 0.05
exact = np.exp(-5 * np.pi**2 * Tend) * U0
kexp = 0.25 * h**2; Nexp = int(np.ceil(Tend / kexp)); kexp = Tend / Nexp
t0 = time.perf_counter(); u = U0.copy()
for _ in range(Nexp):
    u = u - kexp * (L @ u)
t1 = time.perf_counter(); e_exp = np.abs(u - exact).max()
for Ncn in (10, 50):
    kc = Tend / Ncn; Ap = (sp.identity(m * m) + kc / 2 * L).tocsc(); Am = (sp.identity(m * m) - kc / 2 * L).tocsr()
    t2 = time.perf_counter(); lu = spla.splu(Ap); u = U0.copy()
    for _ in range(Ncn):
        u = lu.solve(Am @ u)
    t3 = time.perf_counter()
    print(f"   CN {Ncn} steps (k={kc}): max err {np.abs(u - exact).max():.2e} time {t3 - t2:.3f}s (one sparse LU, fill {lu.L.nnz + lu.U.nnz} nnz)")
print(f"   explicit: {Nexp} steps (k = {kexp:.2e} <= h^2/4): max err {e_exp:.2e} time {t1 - t0:.3f}s")

print("C4 linear advection u_t + u_x = 0, periodic, h = 0.01, CFL 0.8, one revolution")
M = 100; h = 1 / M; nu = 0.8; kk = nu * h; steps = int(round(1 / kk))
xg = np.arange(M) * h
for name, u0f in (("Gaussian", lambda x: np.exp(-200 * (x - 0.5)**2)), ("square", lambda x: ((x > 0.3) & (x < 0.7)).astype(float))):
    uu = u0f(xg); ul = uu.copy(); uw = uu.copy()
    for _ in range(steps):
        uw = uw - nu * (uw - np.roll(uw, 1))
        ul = ul - nu / 2 * (np.roll(ul, -1) - np.roll(ul, 1)) + nu**2 / 2 * (np.roll(ul, -1) - 2 * ul + np.roll(ul, 1))
    print(f"   {name}: upwind max err {np.abs(uw - uu).max():.3f} peak {uw.max():.3f} min {uw.min():.3f} | Lax-Wendroff max err {np.abs(ul - uu).max():.3f} peak {ul.max():.3f} min {ul.min():.3f}")

# ---------- D ----------
print("D1 Black-Scholes by Crank-Nicolson (European and American put)")
K, r, sig, Tm = 100.0, 0.05, 0.25, 1.0
def bs_put(S, tau):
    S = np.asarray(S, float)
    with np.errstate(divide="ignore"):
        d1 = (np.log(S / K) + (r + sig**2 / 2) * tau) / (sig * np.sqrt(tau)); d2 = d1 - sig * np.sqrt(tau)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
def cn_put(M, N, american=False, Smax=300.0):
    S = np.linspace(0, Smax, M + 1); dt = Tm / N; V = np.maximum(K - S, 0.0); i = np.arange(1, M)
    a = 0.5 * dt * (sig**2 * i**2 - r * i) / 2; b = -0.5 * dt * (sig**2 * i**2 + r) ; c = 0.5 * dt * (sig**2 * i**2 + r * i) / 2
    # (I - A/2) V^{n+1} = (I + A/2) V^n, with A the BS operator (time to maturity forward)
    lo, di, up = -a[1:], 1 - b, -c[:-1]
    for n_ in range(N):
        tau = (n_ + 1) * dt
        rhs = a * V[:-2] + (1 + b) * V[1:-1] + c * V[2:]
        V0 = K * np.exp(-r * tau) if not american else K
        rhs[0] += a[0] * V0
        Vin = tridiagonal_solve(lo, di, up, rhs)
        if american:
            Vin = np.maximum(Vin, K - S[1:-1])
        V = np.concatenate([[V0], Vin, [0.0]])
    return S, V
for M_, N_ in ((150, 50), (300, 100), (600, 200)):
    S, V = cn_put(M_, N_)
    i100 = np.argmin(abs(S - 100)); dS = S[1] - S[0]
    delta = (V[i100 + 1] - V[i100 - 1]) / (2 * dS); gamma = (V[i100 + 1] - 2 * V[i100] + V[i100 - 1]) / dS**2
    print(f"   European M={M_} N={N_}: V(100) {V[i100]:.5f} (exact {bs_put(100.0, Tm):.5f}) delta {delta:.5f} (exact {norm.cdf((np.log(1)+(r+sig**2/2))/sig) - 1:.5f}) gamma {gamma:.6f} (exact {norm.pdf((r+sig**2/2)/sig)/(100*sig):.6f})")
S, Va = cn_put(600, 200, american=True)
i100 = np.argmin(abs(S - 100))
# binomial tree for the American put
def crr_american(n):
    dt = Tm / n; u = np.exp(sig * np.sqrt(dt)); d = 1 / u; p = (np.exp(r * dt) - d) / (u - d); disc = np.exp(-r * dt)
    ST = 100 * u**np.arange(n, -1, -1) * d**np.arange(0, n + 1); V = np.maximum(K - ST, 0)
    for j in range(n - 1, -1, -1):
        ST = 100 * u**np.arange(j, -1, -1) * d**np.arange(0, j + 1)
        V = np.maximum(disc * (p * V[:-1] + (1 - p) * V[1:]), K - ST)
    return V[0]
ex_bd = S[np.nonzero(Va - np.maximum(K - S, 0) > 1e-8)[0][0]]
print(f"   American put CN+projection V(100) {Va[i100]:.5f}; CRR binomial (2000 steps) {crr_american(2000):.5f}; early-exercise premium {Va[i100] - bs_put(100.0, Tm):.5f}; exercise boundary at t=0 approx S* = {ex_bd:.1f}")

print("D2 image denoising: linear heat vs Perona-Malik")
rng = np.random.default_rng(0)
n = 128; img = np.zeros((n, n)); img[32:96, 32:96] = 1.0; img[48:80, 48:80] = 0.5
yy, xx = np.mgrid[0:n, 0:n]; img[(xx - 100)**2 + (yy - 28)**2 < 15**2] = 0.8
noisy = img + 0.2 * rng.standard_normal(img.shape)
psnr = lambda a: 10 * np.log10(1 / np.mean((a - img)**2))
def diffuse(u, steps, dt, K_=None):
    u = u.copy()
    for _ in range(steps):
        up = np.pad(u, 1, mode="edge")
        dN = up[:-2, 1:-1] - u; dS = up[2:, 1:-1] - u; dE = up[1:-1, 2:] - u; dW = up[1:-1, :-2] - u
        if K_ is None:
            u = u + dt * (dN + dS + dE + dW)
        else:
            gfun = lambda d: 1 / (1 + (d / K_)**2)
            u = u + dt * (gfun(dN) * dN + gfun(dS) * dS + gfun(dE) * dE + gfun(dW) * dW)
    return u
print(f"   noisy PSNR {psnr(noisy):.2f} dB")
for steps in (5, 10, 20, 40):
    print(f"   {steps} steps (dt=0.2): heat PSNR {psnr(diffuse(noisy, steps, 0.2)):.2f} dB   Perona-Malik (K=0.15) PSNR {psnr(diffuse(noisy, steps, 0.2, 0.15)):.2f} dB")
edge = lambda a: np.abs(np.diff(a[64, 20:44])).max()
print(f"   max jump across the edge at row 64 (true 1.0): heat(20) {edge(diffuse(noisy, 20, 0.2)):.3f}  PM(20) {edge(diffuse(noisy, 20, 0.2, 0.15)):.3f}")

print("D3 Fokker-Planck for the Ornstein-Uhlenbeck process")
theta, sg = 1.0, 0.8
Lx = 4.0; M = 400; xg = np.linspace(-Lx, Lx, M + 1); h = xg[1] - xg[0]
p = norm.pdf(xg, loc=2.0, scale=0.1)
# flux-conservative FD: p_t = d/dx( theta x p + (sg^2/2) p_x ), zero-flux boundaries, CN in time
D = sg**2 / 2
xm = (xg[:-1] + xg[1:]) / 2
A = np.zeros((M + 1, M + 1))
for i in range(M + 1):
    if i < M:  # flux at i+1/2 contributes
        A[i, i + 1] += D / h**2 + theta * xm[i] / (2 * h); A[i, i] += -D / h**2 + theta * xm[i] / (2 * h)
    if i > 0:
        A[i, i - 1] += D / h**2 - theta * xm[i - 1] / (2 * h); A[i, i] += -D / h**2 - theta * xm[i - 1] / (2 * h)
A = sp.csc_matrix(A); dt = 0.01; Id = sp.identity(M + 1, format="csc")
lu = spla.splu((Id - dt / 2 * A).tocsc()); B = (Id + dt / 2 * A).tocsr()
for tstop in (0.5, 1.0, 3.0):
    pp = p.copy()
    for _ in range(int(round(tstop / dt))):
        pp = lu.solve(B @ pp)
    mass = pp.sum() * h; mean = (xg * pp).sum() * h / mass; var = ((xg - mean)**2 * pp).sum() * h / mass
    print(f"   t={tstop}: mass {mass:.6f} mean {mean:.4f} (theory {2*np.exp(-theta*tstop):.4f}) var {var:.4f} (theory {0.01*np.exp(-2*theta*tstop) + sg**2/(2*theta)*(1-np.exp(-2*theta*tstop)):.4f})")
rng = np.random.default_rng(3); Xs = rng.normal(2.0, 0.1, 100000)
for _ in range(300):
    Xs = Xs - theta * Xs * 0.01 + sg * np.sqrt(0.01) * rng.standard_normal(Xs.size)
print(f"   Monte Carlo (Euler-Maruyama, 1e5 paths) at t=3: mean {Xs.mean():.4f} var {Xs.var():.4f}; stationary variance sigma^2/(2 theta) = {sg**2/(2*theta):.4f}")
