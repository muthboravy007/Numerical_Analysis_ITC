"""Reproduces every numerical example in Chapter 15 (Numerical Optimization for Data Science).
Run from the repository root:  python lectures/code/ch15.py"""

import math
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog, minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import integration as I, optimization as op  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(0)

print("=== 15.1 Problems and optimality ===")
X = np.column_stack([np.ones(5), [1.0, 2, 3, 4, 5]]); y = np.array([2.2, 2.8, 3.6, 4.5, 5.1])
F = lambda w: np.mean((X @ w - y) ** 2); g = lambda w: 2 / 5 * X.T @ (X @ w - y); H = 2 / 5 * X.T @ X
print("Ex 15.1.1 grad at 0", g(np.zeros(2)), "Hessian", H, "eig", np.linalg.eigvalsh(H), "minimiser", np.linalg.solve(H, 2 / 5 * X.T @ y), "F*", F(np.linalg.solve(X.T @ X, X.T @ y)))
for pt in ((1.0, 0.0), (-1.0, 0.0)):
    Hh = np.array([[6 * pt[0], 0], [0, 2.0]]); print(f"Ex 15.1.2 f=x^3-3x+y^2 at {pt}: grad {(3 * pt[0] ** 2 - 3, 2 * pt[1])} Hessian eig {np.linalg.eigvalsh(Hh)}")
Xl = rng.standard_normal((30, 3)); wl = rng.standard_normal(3)
p = 1 / (1 + np.exp(-Xl @ wl)); Hl = Xl.T @ (Xl * (p * (1 - p))[:, None]) / 30
print("Ex 15.1.3 logistic Hessian eig at random w:", np.linalg.eigvalsh(Hl))
fq = lambda x: x ** 4 - 3 * x ** 2 + x
crit = np.sort(np.roots([4, 0, -6, 1]).real)
print("Ex 15.1.4 critical points", crit, "values", fq(crit), "second derivs", 12 * crit ** 2 - 6)
A = np.array([[10.0, 0], [0, 1.0]]); print("Ex 15.1.5 quadratic mu, L, kappa:", 1.0, 10.0, 10.0)

print("\n=== 15.2 One-dimensional ===")
f1 = lambda x: x ** 2 - np.sin(x)
a, b = 0.0, 2.0; phi = (math.sqrt(5) - 1) / 2; c = b - phi * (b - a); d = a + phi * (b - a)
rows = []
for k in range(8):
    rows.append((k, a, b, c, d, f1(c), f1(d)))
    if f1(c) < f1(d):
        b, d = d, c; c = b - phi * (b - a)
    else:
        a, c = c, d; d = a + phi * (b - a)
for r in rows[:6]:
    print("Ex 15.2.1 k=%d [%.6f, %.6f] c=%.6f d=%.6f f(c)=%.6f f(d)=%.6f" % r)
xg, hg = op.golden_section(f1, 0, 2, tol=1e-8); print("   minimiser", xg, "iterations", len(hg), "exact (root of 2x=cos x)", __import__("scipy.optimize", fromlist=["brentq"]).brentq(lambda x: 2 * x - math.cos(x), 0, 1))
xp = op.parabolic_step(f1, 0.0, 0.5, 1.0); print("Ex 15.2.2 parabolic step from 0,0.5,1:", xp, "second", op.parabolic_step(f1, 0.5, xp, 1.0))
fN = lambda x: x ** 4 - 3 * x ** 3 + 2; xn = 3.0; hs = [xn]
for _ in range(5):
    xn = xn - (4 * xn ** 3 - 9 * xn ** 2) / (12 * xn ** 2 - 18 * xn); hs.append(xn)
print("Ex 15.2.3 Newton 1-D:", hs)
Xr = rng.standard_normal((40, 6)); Xr[:, 1] = Xr[:, 0] + 0.05 * rng.standard_normal(40); yr = Xr @ np.array([1, 1, 0.5, 0, 0, -1.0]) + 0.5 * rng.standard_normal(40)
U, s, Vt = np.linalg.svd(Xr, full_matrices=False)
def gcv(loglam):
    lam = 10 ** loglam; fil = s ** 2 / (s ** 2 + lam); yhat = U @ (fil * (U.T @ yr)); return np.mean((yr - yhat) ** 2) / (1 - fil.sum() / 40) ** 2
xl, hl = op.golden_section(gcv, -4, 3, tol=1e-4)
print("Ex 15.2.4 GCV-optimal lambda", 10 ** xl, "GCV", gcv(xl), "evaluations", len(hl), "grid check", min((gcv(t), t) for t in np.linspace(-4, 3, 701)))
Aq = np.array([[1.0, 0], [0, 10.0]]); x0 = np.array([10.0, 1.0]); gq = Aq @ x0
print("Ex 15.2.5 exact step alpha = g^Tg/g^TAg =", gq @ gq / (gq @ Aq @ gq), "new point", x0 - gq @ gq / (gq @ Aq @ gq) * gq)

print("\n=== 15.3 Gradient descent ===")
x = x0.copy()
for k in range(4):
    gg = Aq @ x; al = gg @ gg / (gg @ Aq @ gg); x = x - al * gg; print(f"Ex 15.3.1 k={k + 1}: alpha={al:.6f} x={x} f={0.5 * x @ Aq @ x:.6f}")
for kap in (10, 100, 1000):
    Ak = np.diag([1.0, kap]); xk, hk = op.gradient_descent(lambda v: 0.5 * v @ Ak @ v, lambda v: Ak @ v, [1.0, 1.0], lr=1 / kap, tol=1e-6, max_iter=100000)
    print(f"Ex 15.3.2 kappa={kap}: GD with 1/L iterations {len(hk) - 1}  predicted ~ kappa*ln(1e6*sqrt2) = {kap * math.log(1e6 * math.sqrt(2)):.0f}")
x, hr = op.gradient_descent(op.rosenbrock, op.rosenbrock_grad, [-1.2, 1.0], max_iter=50000)
print("Ex 15.3.3 Rosenbrock GD+Armijo iterations", len(hr) - 1, "final", x, "f at 100/1000/10000:", [op.rosenbrock(hr[k]) for k in (100, 1000, 10000)])
Ak = np.diag([1.0, 100.0]); L_, mu_ = 100.0, 1.0
_, h_gd = op.gradient_descent(lambda v: 0.5 * v @ Ak @ v, lambda v: Ak @ v, [1.0, 1.0], lr=1 / L_, tol=1e-8, max_iter=100000)
beta_opt = ((math.sqrt(L_) - math.sqrt(mu_)) / (math.sqrt(L_) + math.sqrt(mu_))) ** 2; lr_hb = 4 / (math.sqrt(L_) + math.sqrt(mu_)) ** 2
_, h_hb = op.momentum_gd(lambda v: Ak @ v, [1.0, 1.0], lr=lr_hb, beta=beta_opt, tol=1e-8)
_, h_nag = op.nesterov(lambda v: Ak @ v, [1.0, 1.0], lr=1 / L_, beta=(math.sqrt(L_) - 1) / (math.sqrt(L_) + 1), tol=1e-8)
print("Ex 15.3.4 kappa=100: GD", len(h_gd) - 1, "heavy-ball", len(h_hb) - 1, "(beta", beta_opt, "lr", lr_hb, ") Nesterov", len(h_nag) - 1)
Xg = np.column_stack([rng.normal(0, 1, 200), rng.normal(0, 100, 200)]); yg = Xg @ np.array([2.0, 0.03]) + 0.1 * rng.standard_normal(200)
for name, Xm in (("raw", Xg), ("standardized", Xg / Xg.std(0))):
    Hm = Xm.T @ Xm / 200; Lm = np.max(np.linalg.eigvalsh(Hm))
    _, hh = op.gradient_descent(lambda w: np.mean((Xm @ w - yg) ** 2) / 2, lambda w: Xm.T @ (Xm @ w - yg) / 200, np.zeros(2), lr=1 / Lm, tol=1e-6, max_iter=500000)
    print(f"Ex 15.3.5 linear regression GD {name}: kappa {np.linalg.cond(Hm):.1f} iterations {len(hh) - 1}")

print("\n=== 15.4 Newton and quasi-Newton ===")
x, hn = op.newton_opt(op.rosenbrock_grad, op.rosenbrock_hess, [-1.2, 1.0])
for k, v in enumerate(hn):
    print(f"Ex 15.4.1 Newton k={k}: {v} f={op.rosenbrock(v):.3e} |grad|={np.linalg.norm(op.rosenbrock_grad(v)):.2e}")
x, hb = op.bfgs(op.rosenbrock, op.rosenbrock_grad, [-1.2, 1.0]); x2, hl2 = op.lbfgs(op.rosenbrock, op.rosenbrock_grad, [-1.2, 1.0], m=5)
sp = minimize(op.rosenbrock, [-1.2, 1.0], jac=op.rosenbrock_grad, method="BFGS")
print("Ex 15.4.2 BFGS its", len(hb) - 1, x, "L-BFGS(m=5) its", len(hl2) - 1, x2, "scipy BFGS its", sp.nit)
Xc = np.column_stack([np.ones(200), rng.standard_normal((200, 2))]); wt = np.array([-0.5, 2.0, -1.0])
yc = (rng.random(200) < 1 / (1 + np.exp(-Xc @ wt))).astype(float)
w, hN = op.newton_opt(lambda v: op.logistic_grad(v, Xc, yc), lambda v: op.logistic_hess(v, Xc, yc), np.zeros(3))
_, hG = op.gradient_descent(lambda v: op.logistic_loss(v, Xc, yc), lambda v: op.logistic_grad(v, Xc, yc), np.zeros(3), tol=1e-8, max_iter=100000)
print("Ex 15.4.3 logistic: Newton its", len(hN) - 1, w, "GD+Armijo its", len(hG) - 1)
fx = lambda v: v[0] ** 4 - 2 * v[0] ** 2 + v[1] ** 2
gx = lambda v: np.array([4 * v[0] ** 3 - 4 * v[0], 2 * v[1]]); Hx = lambda v: np.array([[12 * v[0] ** 2 - 4, 0], [0, 2.0]])
v = np.array([0.3, 0.5]); d = -np.linalg.solve(Hx(v), gx(v))
print("Ex 15.4.4 at (0.3,0.5): Hessian eig", np.linalg.eigvalsh(Hx(v)), "Newton dir", d, "g^T d", gx(v) @ d, "-> ascent in x")
muH = 1.0 - min(0, np.min(np.linalg.eigvalsh(Hx(v)))) + 0.5
xx = np.array([0.3, 0.5]); traj = [xx.copy()]
for _ in range(20):
    Hm = Hx(xx); lm = np.min(np.linalg.eigvalsh(Hm)); shift = max(0.0, -lm + 1.0)
    xx = xx - np.linalg.solve(Hm + shift * np.eye(2), gx(xx)); traj.append(xx.copy())
print("   g_x*d_x", gx(v)[0] * d[0], "pure Newton converges to", op.newton_opt(gx, Hx, [0.3, 0.5])[0], "(saddle!) ; modified Newton iterates", [np.round(t_, 4) for t_ in traj[:6]], "->", traj[-1])
n = 300; Mq = rng.standard_normal((n, n)); Aq2 = Mq.T @ Mq / n + np.eye(n); bq = rng.standard_normal(n)
fq2 = lambda v: 0.5 * v @ Aq2 @ v - bq @ v; gq2 = lambda v: Aq2 @ v - bq
t0 = time.perf_counter(); xN, hN2 = op.newton_opt(gq2, lambda v: Aq2, np.zeros(n)); t1 = time.perf_counter()
xB, hB2 = op.bfgs(fq2, gq2, np.zeros(n), tol=1e-8); t2 = time.perf_counter(); xL, hL2 = op.lbfgs(fq2, gq2, np.zeros(n), m=10, tol=1e-8); t3 = time.perf_counter()
print(f"Ex 15.4.5 n=300 quadratic (kappa {np.linalg.cond(Aq2):.1f}): Newton its {len(hN2) - 1} ({t1 - t0:.3f}s), BFGS its {len(hB2) - 1} ({t2 - t1:.3f}s), L-BFGS its {len(hL2) - 1} ({t3 - t2:.3f}s)")

print("\n=== 15.5 Stochastic gradient ===")
nS = 5000; Xs = np.column_stack([np.ones(nS), rng.standard_normal((nS, 4))]); ws = np.array([1.0, -2, 0.5, 3, 0]); ys = Xs @ ws + 0.5 * rng.standard_normal(nS)
wls = np.linalg.lstsq(Xs, ys, rcond=None)[0]
gi = lambda w, idx: Xs[idx].T @ (Xs[idx] @ w - ys[idx]) / len(idx)
for lr, dec in ((0.01, 0.0), (0.05, 0.0), (0.05, 0.5)):
    w_, hs_ = op.sgd(gi, np.zeros(5), nS, lr=lr, epochs=20, batch_size=1, decay=dec, rng=np.random.default_rng(1))
    print(f"Ex 15.5.1 SGD lr={lr} decay={dec}: ||w - w_LS|| after 1,5,20 epochs: {[round(float(np.linalg.norm(hs_[e] - wls)), 4) for e in (1, 5, 20)]}")
for bs in (1, 32, 512, 5000):
    w_, hs_ = op.sgd(gi, np.zeros(5), nS, lr=0.05 if bs < 5000 else 0.5, epochs=10, batch_size=bs, rng=np.random.default_rng(2))
    print(f"Ex 15.5.2 batch {bs}: updates/epoch {math.ceil(nS / bs)}, ||w - w_LS|| after 10 epochs {np.linalg.norm(w_ - wls):.4f}")
Xa = np.column_stack([np.ones(2000), rng.normal(0, 1, 2000), rng.normal(0, 50, 2000)]); wa = np.array([0.3, 1.5, 0.04])
ya = (rng.random(2000) < 1 / (1 + np.exp(-Xa @ wa))).astype(float)
ga = lambda w, idx: op.logistic_grad(w, Xa[idx], ya[idx])
wopt, _ = op.newton_opt(lambda v: op.logistic_grad(v, Xa, ya), lambda v: op.logistic_hess(v, Xa, ya), np.zeros(3))
for name, fn in (("SGD lr=1e-3", lambda: op.sgd(ga, np.zeros(3), 2000, lr=1e-3, epochs=30, batch_size=32)),
                 ("SGD lr=1e-2", lambda: op.sgd(ga, np.zeros(3), 2000, lr=1e-2, epochs=30, batch_size=32)),
                 ("Adam lr=1e-2", lambda: op.adam(ga, np.zeros(3), 2000, lr=1e-2, epochs=30, batch_size=32))):
    with np.errstate(over="ignore"):
        w_, _ = fn()
    print(f"Ex 15.5.3 {name}: w {np.round(w_, 4)}  loss {op.logistic_loss(w_, Xa, ya):.5f} (optimum {op.logistic_loss(wopt, Xa, ya):.5f}, w* {np.round(wopt, 4)})")
nB = 200000; XB = np.column_stack([np.ones(nB), rng.standard_normal((nB, 9))]); yB = XB @ rng.standard_normal(10) + rng.standard_normal(nB)
wB = np.linalg.lstsq(XB, yB, rcond=None)[0]; fB = lambda w: np.mean((XB @ w - yB) ** 2) / 2; fstar = fB(wB)
t0 = time.perf_counter(); w_, _ = op.sgd(lambda w, idx: XB[idx].T @ (XB[idx] @ w - yB[idx]) / len(idx), np.zeros(10), nB, lr=0.05, epochs=1, batch_size=64); t1 = time.perf_counter()
wg = np.zeros(10); k = 0
t2 = time.perf_counter()
while fB(wg) - fstar > fB(w_) - fstar and k < 1000:
    wg = wg - 0.5 * XB.T @ (XB @ wg - yB) / nB; k += 1
t3 = time.perf_counter()
print(f"Ex 15.5.4 n=200000: one SGD epoch (batch 64) gap {fB(w_) - fstar:.2e} in {t1 - t0:.2f}s; full GD needs {k} passes ({t3 - t2:.2f}s) for same gap")
for lr in (0.1, 0.5, 1.2, 2.5):
    with np.errstate(over="ignore", invalid="ignore"):
        w_, _ = op.sgd(gi, np.zeros(5), nS, lr=lr, epochs=2, batch_size=64, rng=np.random.default_rng(3))
    print(f"Ex 15.5.5 mini-batch lr={lr}: ||w - w_LS|| = {np.linalg.norm(w_ - wls):.3e}")

print("\n=== 15.6 Constrained ===")
print("Ex 15.6.1 max xy s.t. x+y=10: x=y=5, lambda=5, value 25")
print("Ex 15.6.2 min x^2+y^2 s.t. x+y>=1:", minimize(lambda v: v @ v, [2, 2], constraints=[{"type": "ineq", "fun": lambda v: v[0] + v[1] - 1}]).x)
Sig = np.array([[0.04, 0.006, 0.01], [0.006, 0.09, -0.012], [0.01, -0.012, 0.16]]); mu = np.array([0.06, 0.10, 0.14])
wp, hp = op.projected_gradient(lambda w: 2 * Sig @ w, op.project_simplex, np.array([1.0, 0, 0]), lr=2.0, tol=1e-12)
print("Ex 15.6.3 min variance long-only portfolio:", wp, "variance", wp @ Sig @ wp, "return", mu @ wp, "iterations", len(hp) - 1)
res = linprog(c=[-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], bounds=[(0, None), (0, None)])
print("Ex 15.6.4 LP (Wyndor-type) x", res.x, "profit", -res.fun, "duals", res.ineqlin.marginals)
r_target = 0.10; ones = np.ones(3)
K = np.block([[2 * Sig, mu[:, None], ones[:, None]], [mu[None, :], np.zeros((1, 2))], [ones[None, :], np.zeros((1, 2))]])
sol = np.linalg.solve(K, np.r_[np.zeros(3), r_target, 1.0])
print("Ex 15.6.5 Markowitz KKT weights", sol[:3], "variance", sol[:3] @ Sig @ sol[:3], "multipliers", sol[3:])
for rho in (1.0, 10.0, 100.0, 1000.0):
    xpen = minimize(lambda v: v @ v + rho * max(0, 1 - v[0] - v[1]) ** 2, [0.0, 0.0]).x
    print(f"Ex 15.6.6 penalty rho={rho}: x={xpen} constraint violation {1 - xpen.sum():.2e}")

print("\n=== 15.7 Automatic differentiation ===")
class Dual:
    def __init__(self, a, b=0.0): self.a, self.b = a, b
    def __add__(self, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(self.a + o.a, self.b + o.b)
    __radd__ = __add__
    def __mul__(self, o): o = o if isinstance(o, Dual) else Dual(o); return Dual(self.a * o.a, self.a * o.b + self.b * o.a)
    __rmul__ = __mul__
def dsin(u): return Dual(math.sin(u.a), math.cos(u.a) * u.b)
r = Dual(2.0, 1.0) * dsin(Dual(2.0, 1.0))
print("Ex 15.7.1 dual numbers f=x sin x at 2:", r.a, r.b, "exact", 2 * math.sin(2), math.sin(2) + 2 * math.cos(2))
xv, yv = 1.0, 2.0; a1 = xv * yv; a2 = math.sin(xv); a3 = a1 + a2; f = a3 ** 2
fb = 1.0; a3b = 2 * a3 * fb; a1b = a3b; a2b = a3b; xb = a1b * yv + a2b * math.cos(xv); yb = a1b * xv
print("Ex 15.7.2 forward: a1,a2,a3,f", a1, a2, a3, f, " reverse: a3bar", a3b, "xbar", xb, "ybar", yb, "check FD", I.gradient_fd(lambda v: (v[0] * v[1] + math.sin(v[0])) ** 2, [1.0, 2.0]))
wc = rng.standard_normal(3)
ga_ = op.logistic_grad(wc, Xc, yc, 0.1); gn = I.gradient_fd(lambda v: op.logistic_loss(v, Xc, yc, 0.1), wc, 1e-5)
print("Ex 15.7.3 gradient check rel err", np.linalg.norm(ga_ - gn) / np.linalg.norm(ga_ + gn), "| with a bug (missing 1/n):", np.linalg.norm(ga_ * 200 - gn) / np.linalg.norm(ga_ * 200 + gn))
from sklearn.datasets import make_moons
Xm, ym = make_moons(400, noise=0.2, random_state=0)
Hh = 16; W1 = rng.standard_normal((2, Hh)) * 0.5; b1 = np.zeros(Hh); W2 = rng.standard_normal(Hh) * 0.5; b2 = 0.0
def forward(W1, b1, W2, b2, X):
    Z = np.tanh(X @ W1 + b1); return Z, 1 / (1 + np.exp(-(Z @ W2 + b2)))
def loss_params(theta):
    W1_ = theta[:32].reshape(2, 16); b1_ = theta[32:48]; W2_ = theta[48:64]; b2_ = theta[64]
    _, pp = forward(W1_, b1_, W2_, b2_, Xm); return -np.mean(ym * np.log(pp + 1e-12) + (1 - ym) * np.log(1 - pp + 1e-12))
def backprop(W1, b1, W2, b2):
    Z, pp = forward(W1, b1, W2, b2, Xm); d2 = (pp - ym) / len(ym)
    gW2 = Z.T @ d2; gb2 = d2.sum(); dZ = np.outer(d2, W2) * (1 - Z ** 2); gW1 = Xm.T @ dZ; gb1 = dZ.sum(0)
    return gW1, gb1, gW2, gb2
theta = np.r_[W1.ravel(), b1, W2, b2]; g_bp = np.r_[[v.ravel() if np.ndim(v) else [v] for v in backprop(W1, b1, W2, b2)][0], backprop(W1, b1, W2, b2)[1], backprop(W1, b1, W2, b2)[2], backprop(W1, b1, W2, b2)[3]]
g_fd = I.gradient_fd(loss_params, theta, 1e-6)
print("Ex 15.7.4 backprop vs finite differences rel err", np.linalg.norm(g_bp - g_fd) / np.linalg.norm(g_bp + g_fd))
for it in range(3000):
    gW1, gb1, gW2, gb2 = backprop(W1, b1, W2, b2); W1 -= 0.5 * gW1; b1 -= 0.5 * gb1; W2 -= 0.5 * gW2; b2 -= 0.5 * gb2
    if it in (0, 99, 999, 2999):
        _, pp = forward(W1, b1, W2, b2, Xm); print(f"   epoch {it + 1}: loss {-np.mean(ym * np.log(pp) + (1 - ym) * np.log(1 - pp)):.4f} accuracy {np.mean((pp > 0.5) == ym):.3f}")
print("Ex 15.7.5 parameters", theta.size, "-> FD gradient needs", 2 * theta.size, "loss evaluations (central); backprop ~ 2-3 forward costs")
