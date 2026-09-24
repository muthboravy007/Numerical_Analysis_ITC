"""Reproduces every numerical example in Chapter 12 (Numerical Solutions to PDEs).
Run from the repository root:  python lectures/code/ch12.py"""

import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import linalg_iterative as li, pde  # noqa: E402
from numlib.linalg_direct import tridiagonal_solve  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("=== 12.1 Elliptic ===")
A = np.array([[4.0, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 4, -1], [0, -1, -1, 4]])
b = np.array([100.0, 100, 0, 0])
print("Ex 12.1.1 plate 2x2 unknowns (top row first):", np.linalg.solve(A, b))
uex = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
f = lambda x, y: -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
g0 = lambda x, y: 0 * x * y
for n in (4, 8, 16, 32):
    x, y, W = pde.poisson_rect(f, g0, 0, 1, 0, 1, n, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    print(f"Ex 12.1.2/3 n={n} h={1 / n}: u(0.5,0.5)~{W[n // 2, n // 2]:.6f} max err {np.max(np.abs(W - uex(X, Y))):.3e}")
m = 20; A2 = li.poisson_2d_sparse(m); hh = 1 / (m + 1)
xs = hh * np.arange(1, m + 1); XX, YY = np.meshgrid(xs, xs, indexing="ij")
rhs = (hh ** 2 * np.ones_like(XX)).ravel()  # -Laplace u = 1 (generic right-hand side)
Ad = A2.toarray()
its_gs = len(li.gauss_seidel(Ad, rhs, tol=1e-8, max_iter=100000)[1]) - 1
wopt = 2 / (1 + math.sin(math.pi * hh))
its_sor = len(li.sor(Ad, rhs, omega=wopt, tol=1e-8, max_iter=100000)[1]) - 1
its_cg = len(li.conjugate_gradient(Ad, rhs, tol=1e-8)[1]) - 1
print(f"Ex 12.1.4 m=20 (400 unknowns): GS {its_gs}  SOR(w={wopt:.4f}) {its_sor}  CG {its_cg}")
for mm in (10, 100, 1000):
    Am = li.poisson_2d_sparse(mm) if mm <= 100 else None
    nnz = Am.nnz if Am is not None else 5 * mm * mm - 4 * mm
    print(f"Ex 12.1.5 m={mm}: unknowns {mm * mm}, nnz {nnz}, dense entries {(mm * mm) ** 2:.2e}, bandwidth {mm}")

print("\n=== 12.2 Parabolic ===")
u0 = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
uexh = lambda x, t: np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x) + 0.5 * np.exp(-9 * np.pi ** 2 * t) * np.sin(3 * np.pi * x)
tri0 = lambda x: 1 - np.abs(2 * x - 1)
def tri_exact(x, t):
    return sum(8 / (np.pi * kk) ** 2 * (-1) ** ((kk - 1) // 2) * np.sin(kk * np.pi * x) * np.exp(-(kk * np.pi) ** 2 * t) for kk in range(1, 400, 2))
T1 = 0.1
for k in (0.004, 0.005, 0.006):
    x, w, lam = pde.heat_ftcs(tri0, 1.0, 1.0, T1, 10, int(round(T1 / k)))
    print(f"Ex 12.2.1 FTCS tent h=0.1 k={k} lambda={lam:.2f} steps {int(round(T1 / k))}: w(0.5)={w[5]: .6f} exact {tri_exact(0.5, T1):.6f} max err {np.max(np.abs(w - tri_exact(x, T1))):.2e}")
    if lam > 0.5:
        print("   values:", np.round(w, 3))
T = 0.5
m = 10
for lam in (0.4, 0.5, 0.6):
    ev = 1 - 4 * lam * np.sin(np.arange(1, m) * np.pi / (2 * m)) ** 2
    print(f"Ex 12.2.2 lambda={lam}: eigenvalues of A range [{ev.min():.4f}, {ev.max():.4f}], rho={np.max(np.abs(ev)):.4f}")
for k in (0.01, 0.05):
    x, w, lam = pde.heat_btcs(u0, 1.0, 1.0, T, 10, int(round(T / k)))
    print(f"Ex 12.2.3 BTCS k={k} lambda={lam}: w(0.5)={w[5]:.6e} max err {np.max(np.abs(w - uexh(x, T))):.2e}")
for k in (0.01, 0.05):
    x, w, lam = pde.heat_crank_nicolson(u0, 1.0, 1.0, T, 10, int(round(T / k)))
    print(f"Ex 12.2.4 CN k={k} lambda={lam}: w(0.5)={w[5]:.6e} max err {np.max(np.abs(w - uexh(x, T))):.2e}")
print("Ex 12.2.5 refinement with k = h^2/ ... :")
for m in (10, 20, 40):
    h = 1 / m
    kf = 0.4 * h * h; kb = h / 5
    _, wf, _ = pde.heat_ftcs(u0, 1, 1, T, m, int(round(T / kf)))
    _, wb, _ = pde.heat_btcs(u0, 1, 1, T, m, int(round(T / kb)))
    _, wc, _ = pde.heat_crank_nicolson(u0, 1, 1, T, m, int(round(T / kb)))
    xx = np.linspace(0, 1, m + 1); ue = uexh(xx, T)
    print(f"   m={m}: FTCS(k=0.4h^2, {int(round(T / kf))} steps) err {np.max(np.abs(wf - ue)):.2e} | BTCS(k=h/5, {int(round(T / kb))} steps) err {np.max(np.abs(wb - ue)):.2e} | CN(k=h/5) err {np.max(np.abs(wc - ue)):.2e}")

print("\n=== 12.3 Hyperbolic ===")
fw = lambda x: np.sin(np.pi * x); gw = lambda x: 0 * x
for lamw, N in ((1.0, 10), (0.5, 20), (0.8, 13)):
    x, w, lam = pde.wave_explicit(fw, gw, 1.0, 1.0, 1.0, 10, N)
    ex = np.sin(np.pi * x) * np.cos(np.pi * 1.0)
    print(f"Ex 12.3.1 lambda={lam:.3f}: w(0.5, t=1)={w[5]:.6f} exact {ex[5]:.6f} max err {np.max(np.abs(w - ex)):.2e}")
for N in (10, 9, 8):
    x, w, lam = pde.wave_explicit(fw, gw, 1.0, 1.0, 5.0, 10, N * 5)
    print(f"Ex 12.3.2 lambda={lam:.3f}: max|w| at t=5 = {np.max(np.abs(w)):.3e}")
tri = lambda x: np.where(x < 0.5, 2 * x, 2 - 2 * x)
x, w, lam = pde.wave_explicit(tri, gw, 1.0, 1.0, 0.3, 10, 3)
def dalembert(x, t):
    def ext(s):
        s = np.mod(s, 2.0); return np.where(s <= 1, tri(s), -tri(2 - s))
    return 0.5 * (ext(x - t) + ext(x + t))
print("Ex 12.3.3 plucked string t=0.3 lambda=1:", w, "\n   d'Alembert:", dalembert(x, 0.3), "max diff", np.max(np.abs(w - dalembert(x, 0.3))))
m = 50; x, w, lam = pde.wave_explicit(fw, gw, 1.0, 1.0, 2.0, m, 100)
print("Ex 12.3.4 after one full period t=2 (lambda=1): max|w - u0|", np.max(np.abs(w - fw(x))))
c = 1.0; L = 1.0; m = 100; h = L / m; k = 0.5 * h; xa = np.linspace(0, L, m + 1)
ua = np.exp(-200 * (xa - 0.25) ** 2); u = ua.copy()
for _ in range(int(round(0.5 / k))):
    u[1:] = u[1:] - c * k / h * (u[1:] - u[:-1])
exa = np.exp(-200 * (xa - 0.75) ** 2)
print("Ex 12.3.5 upwind advection: peak", u.max(), "(exact 1) at x", xa[u.argmax()], "max err", np.max(np.abs(u - exa)))

print("\n=== 12.4 DS ===")
rng = np.random.default_rng(0)
n = 256; xs = np.linspace(0, 1, n); sig = np.sign(np.sin(2 * np.pi * 3 * xs)); noisy = sig + 0.3 * rng.standard_normal(n)
h = 1 / (n - 1)
for tfin in (1e-5, 3e-5, 1e-4, 1e-3):
    w = noisy.copy(); kk = 0.4 * h * h; steps = int(round(tfin / kk)); lam = kk / h ** 2
    for _ in range(steps):
        w[1:-1] = w[1:-1] + lam * (w[2:] - 2 * w[1:-1] + w[:-2])
    sigma_eq = math.sqrt(2 * tfin)
    print(f"Ex 12.4.1 heat smoothing t={tfin:g} (Gaussian sigma={sigma_eq:.4f}): RMSE {np.sqrt(np.mean((w - sig) ** 2)):.4f}  (noisy {np.sqrt(np.mean((noisy - sig) ** 2)):.4f})")
N = 64; img = np.zeros((N, N)); img[16:48, 16:48] = 1.0; img[24:40, 24:40] = 0.5
noisy_img = img + 0.2 * rng.standard_normal(img.shape)
def psnr(a, b):
    return 10 * np.log10(1 / np.mean((a - b) ** 2))
u = noisy_img.copy()
for it in range(1, 21):
    up = np.pad(u, 1, mode="edge")
    u = u + 0.2 * (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2] - 4 * u)
    if it in (2, 5, 10, 20):
        print(f"Ex 12.4.2 2-D heat denoising iter {it}: PSNR {psnr(u, img):.2f} dB (noisy {psnr(noisy_img, img):.2f})")
Kst, r, sgm, Tm = 100.0, 0.05, 0.2, 1.0
Smax = 300.0; M = 300; Nt = 200; dS = Smax / M; dt = Tm / Nt; S = np.linspace(0, Smax, M + 1)
V = np.maximum(S - Kst, 0)
i = np.arange(1, M)
a = 0.25 * dt * (sgm ** 2 * i ** 2 - r * i); bb = -0.5 * dt * (sgm ** 2 * i ** 2 + r); cc = 0.25 * dt * (sgm ** 2 * i ** 2 + r * i)
for n_ in range(Nt):
    tau = (n_ + 1) * dt
    rhs = a * V[:-2] + (1 + bb) * V[1:-1] + cc * V[2:]
    bc_hi = Smax - Kst * math.exp(-r * tau)
    rhs[-1] += cc[-1] * bc_hi
    V_new = tridiagonal_solve(-a[1:], 1 - bb, -cc[:-1], rhs)
    V = np.r_[0.0, V_new, bc_hi]
d1 = (math.log(100 / Kst) + (r + 0.5 * sgm ** 2) * Tm) / (sgm * math.sqrt(Tm)); d2 = d1 - sgm * math.sqrt(Tm)
bs = 100 * norm.cdf(d1) - Kst * math.exp(-r * Tm) * norm.cdf(d2)
print("Ex 12.4.3 Black-Scholes CN: V(S=100)", V[100], "closed form", bs, "err", abs(V[100] - bs))
W = np.zeros((6, 6))
for i_, j_ in [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)]:
    W[i_, j_] = W[j_, i_] = 1
Lg = np.diag(W.sum(1)) - W; wv, Vv = np.linalg.eigh(Lg)
f0 = np.array([1.0, 0.2, 0.9, -0.8, -1.1, -0.1])
for t in (0.1, 0.5, 2.0):
    ft = Vv @ (np.exp(-t * wv) * (Vv.T @ f0))
    print(f"Ex 12.4.4 graph heat kernel t={t}: {np.round(ft, 4)}  mean {ft.mean():.4f}")
img2 = img.copy(); mask = np.zeros_like(img, bool); mask[28:36, 10:54] = True
u = img2.copy(); u[mask] = 0.0
for it in range(3000):
    up = np.pad(u, 1, mode="edge"); avg = 0.25 * (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2])
    u[mask] = avg[mask]
print("Ex 12.4.5 harmonic inpainting: RMSE in hole", np.sqrt(np.mean((u[mask] - img[mask]) ** 2)), "vs zero-fill", np.sqrt(np.mean(img[mask] ** 2)))
