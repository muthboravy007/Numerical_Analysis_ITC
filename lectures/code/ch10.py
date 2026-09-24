"""Reproduces every numerical example in Chapter 10 (Nonlinear Systems of Equations).
Run from the repository root:  python lectures/code/ch10.py"""

import math
import os
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import nonlinear_systems as ns, roots  # noqa: E402

np.set_printoptions(precision=8, suppress=True, linewidth=120)

print("=== 10.1 Fixed points ===")
G = lambda v: np.array([0.5 * np.cos(v[1]) + 0.1, 0.3 * np.sin(v[0]) + 0.4])
x, h = ns.fixed_point_system(G, [0.0, 0.0], tol=1e-10)
for k in range(6):
    print(f"Ex 10.1.1 k={k}: {h[k]}")
print("   converged", x, "iterations", len(h) - 1, "residual", G(x) - x)
Jg = lambda v: np.array([[0, -0.5 * np.sin(v[1])], [0.3 * np.cos(v[0]), 0]])
print("Ex 10.1.2 max partials on D=[0,1]^2: 0.5, 0.3 -> K/n with n=2 gives K=", 2 * 0.5, "; ||J||_inf at x*", np.linalg.norm(Jg(x), np.inf))
xg = np.array([0.0, 0.0]); hs = [xg.copy()]
for k in range(30):
    xn = xg.copy(); xn[0] = 0.5 * np.cos(xn[1]) + 0.1; xn[1] = 0.3 * np.sin(xn[0]) + 0.4
    hs.append(xn.copy())
    if np.max(np.abs(xn - xg)) < 1e-10:
        break
    xg = xn
print("Ex 10.1.3 Gauss-Seidel style iterations", len(hs) - 1, "vs Jacobi style", len(h) - 1)
F = lambda v: np.array([v[0] ** 2 + v[1] - 11, v[0] + v[1] ** 2 - 7])
Gb = lambda v: np.array([11 - v[1], 7 - v[0]]) ** 0 * np.array([11 - v[1] - v[0] ** 2 + v[0], 7 - v[0] - v[1] ** 2 + v[1]])
it = [np.array([2.9, 2.1])]
for k in range(6):
    it.append(Gb(it[-1]))
print("Ex 10.1.4 naive rearrangement x=x-F(x) from (2.9,2.1):", [np.round(v, 3) for v in it])
print("   Jacobian of G at (3,2):", np.eye(2) - np.array([[6, 1], [1, 4]]), "rho", max(abs(np.linalg.eigvals(np.eye(2) - np.array([[6, 1], [1, 4.0]])))))
Gc = lambda v: np.array([math.sqrt(11 - v[1]), math.sqrt(7 - v[0])])
x, h = ns.fixed_point_system(Gc, [2.5, 2.5], tol=1e-10)
Jc = np.array([[0, -0.5 / math.sqrt(11 - 2)], [-0.5 / math.sqrt(7 - 3), 0]])
print("Ex 10.1.5 G=(sqrt(11-y), sqrt(7-x)) converged to", x, "in", len(h) - 1, "its; rho(J)", max(abs(np.linalg.eigvals(Jc))))
e = [np.max(np.abs(v - np.array([3.0, 2.0]))) for v in h]
print("   error ratios", [round(e[k + 1] / e[k], 3) for k in range(3, 9)])

print("\n=== 10.2 Newton ===")
Fc = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 4, v[0] * v[1] - 1])
Jc2 = lambda v: np.array([[2 * v[0], 2 * v[1]], [v[1], v[0]]])
x, h = roots.newton_system(Fc, Jc2, [2.0, 0.5])
for k, v in enumerate(h):
    print(f"Ex 10.2.1 k={k}: {v}  ||F||={np.linalg.norm(Fc(v)):.2e}")
F3 = lambda v: np.array([v[0] + v[1] + v[2] - 6, v[0] ** 2 + v[1] ** 2 + v[2] ** 2 - 14, v[0] * v[1] * v[2] - 6])
J3 = lambda v: np.array([[1, 1, 1], [2 * v[0], 2 * v[1], 2 * v[2]], [v[1] * v[2], v[0] * v[2], v[0] * v[1]]])
x, h = roots.newton_system(F3, J3, [0.5, 1.5, 3.5])
v0 = np.array([0.5, 1.5, 3.5]); print("Ex 10.2.2 F(x0)", F3(v0), "J(x0)\n", J3(v0), "\n y0", np.linalg.solve(J3(v0), -F3(v0)))
for k, v in enumerate(h):
    print(f"   k={k}: {v}  err={np.max(np.abs(v - [1, 2, 3])):.2e}")
Fh = lambda v: np.array([v[0] ** 2 + v[1] - 11, v[0] + v[1] ** 2 - 7])
Jh = lambda v: np.array([[2 * v[0], 1], [1, 2 * v[1]]])
x, h = roots.newton_system(Fh, Jh, [1.0, 1.0])
e = [np.max(np.abs(v - x)) for v in h]
print("Ex 10.2.3 Himmelblau from (1,1):", x, "errors", ["%.1e" % v for v in e], "ratios e_{k+1}/e_k^2", [round(e[k + 1] / e[k] ** 2, 3) for k in range(len(e) - 3)])
for s0 in ([-3.0, 3.0], [-3.0, -3.0], [4.0, -2.0], [0.0, 0.0]):
    try:
        x, h = roots.newton_system(Fh, Jh, s0); print(f"Ex 10.2.4 start {s0} -> {x} in {len(h) - 1} its")
    except Exception as ex:
        print("Ex 10.2.4", s0, "failed", ex)
Jfd = ns.jacobian_fd(F3, [0.5, 1.5, 3.5]); print("Ex 10.2.5 FD Jacobian\n", Jfd, "\n max diff", np.max(np.abs(Jfd - J3([0.5, 1.5, 3.5]))), "F-evals per Newton step n+1 =", 4)
Fa = lambda v: np.array([math.atan(v[0]) + 0.1 * v[1], v[1] - 0.5 * v[0]])
Ja = lambda v: np.array([[1 / (1 + v[0] ** 2), 0.1], [-0.5, 1]])
xx = np.array([3.0, 0.0]); pure = [xx.copy()]
for _ in range(5):
    xx = xx - np.linalg.solve(Ja(xx), Fa(xx)); pure.append(xx.copy())
print("Ex 10.2.6 pure Newton:", [np.round(v, 3) for v in pure])
xx = np.array([3.0, 0.0]); damp = [xx.copy()]
for _ in range(12):
    d = -np.linalg.solve(Ja(xx), Fa(xx)); t = 1.0
    while np.linalg.norm(Fa(xx + t * d)) >= (1 - 1e-4 * t) * np.linalg.norm(Fa(xx)) and t > 1e-8:
        t /= 2
    xx = xx + t * d; damp.append(xx.copy())
print("   damped Newton:", [np.round(v, 5) for v in damp[:8]], "... final", damp[-1])

print("\n=== 10.3 Broyden ===")
x, h = ns.broyden(Fc, [2.0, 0.5], J0=Jc2([2.0, 0.5]))
xs = np.array([math.sqrt(2 + math.sqrt(3)), math.sqrt(2 - math.sqrt(3))])
for k, v in enumerate(h):
    print(f"Ex 10.3.1 k={k}: {v} err {np.max(np.abs(v - xs)):.2e}")
A0 = Jc2([2.0, 0.5]); x0 = np.array([2.0, 0.5]); x1 = h[1]; s = x1 - x0; yv = Fc(x1) - Fc(x0)
A1 = A0 + np.outer(yv - A0 @ s, s) / (s @ s)
Ai = np.linalg.inv(A0); Ai1 = Ai + np.outer(s - Ai @ yv, s @ Ai) / (s @ Ai @ yv)
print("Ex 10.3.2 A1 =", A1, "\n inv(A1) =", np.linalg.inv(A1), "\n Sherman-Morrison =", Ai1, "\n secant check A1 s = y:", A1 @ s, yv)
e = [np.max(np.abs(v - xs)) for v in h if np.max(np.abs(v - xs)) > 0]
print("Ex 10.3.3 ratios e_{k+1}/e_k:", [round(e[k + 1] / e[k], 4) for k in range(len(e) - 1)])
cnt = [0]
def F3c(v):
    cnt[0] += 1; return F3(v)
x, h = ns.broyden(F3c, [0.5, 1.5, 3.5], J0=J3([0.5, 1.5, 3.5]))
print("Ex 10.3.4/5 Broyden 3x3:", x, "iterations", len(h) - 1, "F evals", cnt[0])
cnt2 = [0]
def F3d(v):
    cnt2[0] += 1; return F3(v)
xn = np.array([0.5, 1.5, 3.5]); k = 0
while True:
    Jn = ns.jacobian_fd(F3d, xn); d = np.linalg.solve(Jn, -F3d(xn)); xn = xn + d; k += 1
    if np.linalg.norm(d) < 1e-10 or k > 30:
        break
print("   Newton with FD Jacobian: iterations", k, "F evals", cnt2[0], xn)

print("\n=== 10.4 Steepest descent ===")
gF = lambda v: float(np.sum(Fh(v) ** 2))
x, h = ns.steepest_descent_system(Fh, Jh, [0.0, 0.0], tol=1e-4)
for k in range(min(6, len(h))):
    print(f"Ex 10.4.1 k={k}: {h[k]} g={gF(h[k]):.5f}")
print("   stopped after", len(h) - 1, "at", h[-1], "g", gF(h[-1]))
x2, h2 = roots.newton_system(Fh, Jh, h[-1]); print("Ex 10.4.2 Newton from SD point:", x2, "in", len(h2) - 1, "its")
xk = np.array([0.0, 0.0]); g1 = gF(xk); z = 2 * Jh(xk).T @ Fh(xk); z0 = np.linalg.norm(z); z = z / z0
a3 = 1.0; g3 = gF(xk - a3 * z)
while g3 >= g1:
    a3 /= 2; g3 = gF(xk - a3 * z)
a2 = a3 / 2; g2 = gF(xk - a2 * z); h1 = (g2 - g1) / a2; h2_ = (g3 - g2) / (a3 - a2); h3 = (h2_ - h1) / a3; a0 = 0.5 * (a2 - h1 / h3)
print("Ex 10.4.3 one step: grad", 2 * Jh(xk).T @ Fh(xk), "z", z, "g1", g1, "a3", a3, "g3", g3, "a2", a2, "g2", g2, "h1,h2,h3", h1, h2_, h3, "a0", a0, "g(a0)", gF(xk - a0 * z))
x, h = ns.steepest_descent_system(Fh, Jh, [2.5, 1.5], tol=1e-12, max_iter=200)
print("Ex 10.4.4 SD near root: iterations", len(h) - 1, "final g", gF(h[-1]), "vs Newton", len(roots.newton_system(Fh, Jh, [2.5, 1.5])[1]) - 1)
Fl = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 1, v[0] ** 2 + v[1] ** 2 - 4])  # inconsistent: no root
Jl = lambda v: np.array([[2 * v[0], 2 * v[1]], [2 * v[0], 2 * v[1]]])
x, h = ns.steepest_descent_system(Fl, Jl, [1.0, 0.5], tol=1e-12, max_iter=500)
print("Ex 10.4.5 system with no real root: SD ->", x, "r^2", x @ x, "g", float(np.sum(Fl(x) ** 2)), "grad", 2 * Jl(x).T @ Fl(x))

print("\n=== 10.5 Homotopy ===")
x, path = ns.homotopy(Fh, Jh, np.array([1.0, 1.0]), N=4)
for k, v in enumerate(path):
    print(f"Ex 10.5.1 lambda={k / 4:.2f}: {v}")
x2, _ = roots.newton_system(Fh, Jh, x); print("   Newton polish:", x2)
Fe = lambda v: np.array([np.exp(v[0]) - 2 + 0 * v[1], v[1] - v[0] ** 2])
Fq = lambda v: np.array([math.atan(v[0]) - 1.0, v[1] - v[0]])
Jq = lambda v: np.array([[1 / (1 + v[0] ** 2), 0], [-1, 1]])
xx = np.array([3.0, 0.0]); pn = [xx]
for _ in range(4):
    xx = xx - np.linalg.solve(Jq(xx), Fq(xx)); pn.append(xx)
print("Ex 10.5.2 Newton on atan(x)=1 from 3:", [np.round(v, 3) for v in pn])
x, path = ns.homotopy(Fq, Jq, np.array([3.0, 0.0]), N=10); print("   homotopy N=10:", x, "true", math.tan(1.0))
for N in (1, 2, 4, 8, 16):
    x, _ = ns.homotopy(Fh, Jh, np.array([1.0, 1.0]), N=N)
    print(f"Ex 10.5.3 N={N}: {x} err {np.max(np.abs(x - [3, 2])):.2e}")
lam_vals = np.linspace(0, 2, 9); xc = 0.0; out = []
for lam in lam_vals:
    xc, _ = roots.newton(lambda x: x ** 3 + x - lam, lambda x: 3 * x ** 2 + 1, xc); out.append(round(xc, 6))
print("Ex 10.5.4 natural continuation x^3+x=lambda:", list(zip(lam_vals, out)))
xc = -1.2; out = []
for lam in np.linspace(-1.0, 1.0, 11):
    try:
        xc, _ = roots.newton(lambda x: x ** 3 - x - lam, lambda x: 3 * x ** 2 - 1, xc, max_iter=50); out.append((round(lam, 1), round(xc, 4)))
    except Exception:
        out.append((round(lam, 1), "fail"))
print("Ex 10.5.5 continuation on x^3-x=lambda from branch x=-1.2:", out, "fold at lambda=", 2 / (3 * math.sqrt(3)))

print("\n=== 10.6 DS ===")
from scipy.special import digamma
rng = np.random.default_rng(0)
data = rng.weibull(1.8, 200) * 3.0
lx = np.log(data)
def score(p):
    k, lam = p
    z = (data / lam) ** k
    return np.array([len(data) / k + np.sum(lx) - len(data) * np.log(lam) - np.sum(z * np.log(data / lam)),
                     -len(data) * k / lam + k / lam * np.sum(z)])
p, h = roots.newton_system(score, lambda p: ns.jacobian_fd(score, p, 1e-6), [1.0, float(np.mean(data))])
print("Ex 10.6.1 Weibull MLE k, lambda:", p, "iterations", len(h) - 1, "(true 1.8, 3.0)")
X = np.column_stack([np.ones(8), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]]); yb = np.array([0, 0, 1, 0, 1, 0, 1, 1.0])
w = np.zeros(2); hist = [w.copy()]
for k in range(8):
    pr = expit(X @ w); gr = X.T @ (yb - pr); H = X.T @ (X * (pr * (1 - pr))[:, None]); w = w + np.linalg.solve(H, gr); hist.append(w.copy())
print("Ex 10.6.2 logistic Newton/IRLS:", [np.round(v, 6) for v in hist[:7]])
xm = np.concatenate([rng.normal(-2, 1, 300), rng.normal(3, 1.5, 200)])
pi_, m1, m2, s1, s2 = 0.5, -1.0, 1.0, 1.0, 1.0; trace = []
for it in range(200):
    from scipy.stats import norm
    r = pi_ * norm.pdf(xm, m1, s1); r = r / (r + (1 - pi_) * norm.pdf(xm, m2, s2))
    new = (r.mean(), np.sum(r * xm) / r.sum(), np.sum((1 - r) * xm) / (1 - r).sum())
    s1 = math.sqrt(np.sum(r * (xm - new[1]) ** 2) / r.sum()); s2 = math.sqrt(np.sum((1 - r) * (xm - new[2]) ** 2) / (1 - r).sum())
    trace.append(new[1]); pi_, m1, m2 = new
print("Ex 10.6.3 EM mixture: pi, mu1, mu2, s1, s2 =", pi_, m1, m2, s1, s2)
err = np.abs(np.array(trace) - trace[-1])
print("   |mu1_k - mu1*| k=5,10,20,40:", err[[5, 10, 20, 40]], "ratio", err[21] / err[20])
Fm = lambda p: np.array([100 - 2 * p[0] + p[1] - (10 + 3 * p[0]), 80 - 3 * p[1] + 0.5 * p[0] - (5 + 2 * p[1] ** 1.2)])
p, h = roots.newton_system(Fm, lambda p: ns.jacobian_fd(Fm, p), [10.0, 10.0])
print("Ex 10.6.4 market equilibrium prices:", p, "residual", Fm(p), "iterations", len(h) - 1)
pts = np.concatenate([rng.normal([0, 0], 0.5, (50, 2)), rng.normal([3, 3], 0.5, (50, 2)), rng.normal([0, 4], 0.5, (50, 2))])
for name, init in (("good init", [0, 50, 100]), ("bad init", [0, 1, 2])):
    c = pts[init].copy(); moves = []
    for it in range(50):
        lab = np.argmin(((pts[:, None, :] - c[None]) ** 2).sum(2), axis=1)
        cn = np.array([pts[lab == j].mean(0) for j in range(3)]); moves.append(np.max(np.abs(cn - c))); c = cn
        if moves[-1] == 0:
            break
    sse = np.sum((pts - c[lab]) ** 2)
    print(f"Ex 10.6.5 k-means ({name}): iterations {len(moves)} moves {np.round(moves, 4)} centres {np.round(c, 3).tolist()} SSE {sse:.2f}")
