"""Numerical answers for exercises/ch15_exercises.md."""
import math
import os
import sys
import time

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import optimization as op  # noqa: E402

np.set_printoptions(precision=6, suppress=True)

# ---------- B ----------
f1 = lambda x: x**2 - 2 * np.sin(x)
a, b = 0.0, 2.0; invphi = (np.sqrt(5) - 1) / 2
c, d = b - invphi * (b - a), a + invphi * (b - a)
print("B1 golden section on x^2 - 2 sin x, [0,2]")
for k in range(1, 6):
    if f1(c) < f1(d):
        b, d = d, c; c = b - invphi * (b - a)
    else:
        a, c = c, d; d = a + invphi * (b - a)
    print(f"   step {k}: [{a:.6f}, {b:.6f}] width {b - a:.6f}")
xs = optimize.brentq(lambda x: 2 * x - 2 * np.cos(x), 0, 2)
print("   true minimiser", xs, " f*", f1(xs), " parabolic step through 0,1,2:", op.parabolic_step(f1, 0.0, 1.0, 2.0), " golden iterations for 1e-8:", len(op.golden_section(f1, 0, 2, 1e-8)[1]))

A2 = np.diag([2.0, 10.0]); g2 = lambda v: A2 @ v; F2 = lambda v: 0.5 * v @ A2 @ v
x = np.array([5.0, 1.0])
print("B2 f = x^2 + 5y^2, fixed step 0.1:")
for k in range(1, 4):
    x = x - 0.1 * g2(x); print(f"   k={k}: {x}  f {F2(x):.6f}")
x = np.array([5.0, 1.0])
print("   exact line search:")
for k in range(1, 4):
    gk = g2(x); t = (gk @ gk) / (gk @ A2 @ gk); x = x - t * gk
    print(f"   k={k}: t {t:.6f} x {x}  f {F2(x):.6f}")
print("   optimal fixed step 2/(2+10) =", 2 / 12, " rate (k-1)/(k+1) with k=5:", 4 / 6, " step 0.1 gives factors |1-0.2|, |1-1| =", 0.8, 0.0)
x = np.array([5.0, 1.0])
for lr in (0.1, 1 / 6, 0.19, 0.21):
    with np.errstate(all="ignore"):
        xx, h = op.gradient_descent(F2, g2, [5.0, 1.0], lr=lr, tol=1e-8, max_iter=10000)
    print(f"   lr {lr:.4f}: iterations to ||g||<1e-8: {len(h) - 1 if np.linalg.norm(g2(xx)) < 1e-8 else 'diverged/not converged'}")

x0 = np.array([-1.2, 1.0])
g0 = op.rosenbrock_grad(x0); H0 = op.rosenbrock_hess(x0)
print("B3 Rosenbrock at (-1.2,1): f", op.rosenbrock(x0), " grad", g0, "\n   Hessian\n", H0, "\n   eig", np.linalg.eigvalsh(H0), " Newton step", -np.linalg.solve(H0, g0), " x1", x0 - np.linalg.solve(H0, g0), " f(x1)", op.rosenbrock(x0 - np.linalg.solve(H0, g0)))
xn, hn = op.newton_opt(op.rosenbrock_grad, op.rosenbrock_hess, x0)
print("   pure Newton iterations", len(hn) - 1, " f values", [f"{op.rosenbrock(h):.3g}" for h in hn])

A4 = np.diag([1.0, 5.0]); f4 = lambda v: 0.5 * v @ A4 @ v; g4 = lambda v: A4 @ v
x = np.array([1.0, 1.0]); H = np.eye(2)
for k in range(1, 3):
    g = g4(x); p = -H @ g; t = -(g @ p) / (p @ A4 @ p); s = t * p; xn_ = x + s; y = g4(xn_) - g
    rho = 1 / (s @ y); I = np.eye(2)
    Hn = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
    print(f"B4 BFGS step {k}: p {p} t {t:.6f} s {s} y {y} x {xn_}\n   H{k}\n", Hn, "\n   secant check H y - s", Hn @ y - s)
    x, H = xn_, Hn
print("   A^{-1} =", np.linalg.inv(A4).diagonal(), " (BFGS with exact line search on a quadratic terminates in n steps with H_n = A^{-1})")

res = optimize.minimize(lambda v: v @ v, [1.0, 1.0], constraints=[{"type": "ineq", "fun": lambda v: v[0] + 2 * v[1] - 4}])
print("B5a min x^2+y^2 s.t. x+2y>=4:", res.x, " multiplier mu = 2x/1 =", 2 * res.x[0])
res = optimize.minimize(lambda v: v[0] + v[1], [0.5, 0.5], constraints=[{"type": "ineq", "fun": lambda v: 2 - v @ v}])
print("B5b min x+y s.t. x^2+y^2<=2:", res.x, " mu from 1 + 2 mu x = 0:", -1 / (2 * res.x[0]))

v = np.array([0.8, 0.6, -0.2, 0.4])
u = np.sort(v)[::-1]; css = np.cumsum(u)
print("B6 sorted", u, " cumsum", css, " test u_j - (css_j - 1)/j:", u - (css - 1) / np.arange(1, 5), " projection", op.project_simplex(v), " sum", op.project_simplex(v).sum())

x1, x2 = 1.0, 2.0
v1 = x1 * x2; v2 = np.sin(x1); v3 = v1 + v2; v4 = np.exp(x2); fval = v3 * v4
bar_v3 = v4; bar_v4 = v3; bar_v1 = bar_v3; bar_v2 = bar_v3
gx1 = bar_v1 * x2 + bar_v2 * np.cos(x1); gx2 = bar_v1 * x1 + bar_v4 * v4
print(f"B7 f(1,2) = (x1 x2 + sin x1) e^x2 = {fval:.6f}; intermediates v1 {v1} v2 {v2:.6f} v3 {v3:.6f} v4 {v4:.6f}")
print(f"   reverse mode: vbar3 {bar_v3:.6f} vbar4 {bar_v4:.6f} -> df/dx1 {gx1:.6f}  df/dx2 {gx2:.6f}")
h = 1e-6
fd = [((lambda a, b: (a * b + np.sin(a)) * np.exp(b))(x1 + h, x2) - (lambda a, b: (a * b + np.sin(a)) * np.exp(b))(x1 - h, x2)) / (2 * h),
      ((lambda a, b: (a * b + np.sin(a)) * np.exp(b))(x1, x2 + h) - (lambda a, b: (a * b + np.sin(a)) * np.exp(b))(x1, x2 - h)) / (2 * h)]
print("   central differences", np.round(fd, 6), "  forward-mode seed e1: dv1 = x2 =", x2, ", dv2 = cos x1 =", round(np.cos(x1), 6), ", dv3 =", round(x2 + np.cos(x1), 6), ", df =", round((x2 + np.cos(x1)) * v4, 6))

for lam in (1.0, 10.0):
    for alpha, beta in ((0.1, 0.0), (0.1, 0.5), (0.1, 0.9), (0.15, 0.9)):
        roots = np.roots([1, -(1 + beta - alpha * lam), beta])
        print(f"B8 lambda {lam} alpha {alpha} beta {beta}: roots {np.round(roots, 4)} |r|max {np.abs(roots).max():.4f}")
kap = 10; print("   optimal heavy ball for spectrum [1,10]: alpha = 4/(sqrt(L)+sqrt(mu))^2 =", 4 / (np.sqrt(10) + 1)**2, " beta = ((sqrt k-1)/(sqrt k+1))^2 =", ((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1))**2,
              " rate", (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1), " vs GD rate", (kap - 1) / (kap + 1))

# ---------- C ----------
print("C1 optimizer comparison")
def counted(f, g):
    cnt = {"f": 0, "g": 0}
    def ff(x):
        cnt["f"] += 1; return f(x)
    def gg(x):
        cnt["g"] += 1; return g(x)
    return ff, gg, cnt
rng = np.random.default_rng(0)
n = 100; Qm, _ = np.linalg.qr(rng.standard_normal((n, n))); ev = np.logspace(0, 3, n); Aq = (Qm * ev) @ Qm.T; bq = rng.standard_normal(n)
fq = lambda x: 0.5 * x @ Aq @ x - bq @ x; gq = lambda x: Aq @ x - bq; xq = np.linalg.solve(Aq, bq)
problems = {"Rosenbrock": (op.rosenbrock, op.rosenbrock_grad, op.rosenbrock_hess, np.array([-1.2, 1.0]), np.array([1.0, 1.0]), 1 / 1000),
            "quadratic k=1e3": (fq, gq, lambda x: Aq, np.zeros(n), xq, 1 / 1000)}
for pname, (f, g, Hf, x0, xstar, lr) in problems.items():
    print(f"   {pname}:")
    for name in ("GD fixed step", "GD backtracking", "Nesterov", "Newton", "BFGS", "L-BFGS m=5"):
        ff, gg, cnt = counted(f, g)
        t0 = time.perf_counter()
        if name == "GD fixed step":
            x, h = op.gradient_descent(ff, gg, x0, lr=lr, tol=1e-6, max_iter=200000)
        elif name == "GD backtracking":
            x, h = op.gradient_descent(ff, gg, x0, tol=1e-6, max_iter=200000)
        elif name == "Nesterov":
            if pname == "Rosenbrock":  # momentum needs a smaller step than plain GD here (lr = 1e-3 diverges)
                x, h = op.nesterov(gg, x0, lr=5e-4, beta=0.95, tol=1e-6, max_iter=200000)
            else:
                x, h = op.nesterov(gg, x0, lr=lr, beta=(np.sqrt(1000) - 1) / (np.sqrt(1000) + 1), tol=1e-6, max_iter=200000)
        elif name == "Newton":
            x, h = op.newton_opt(gg, Hf, x0, tol=1e-6)
        elif name == "BFGS":
            x, h = op.bfgs(ff, gg, x0, tol=1e-6)
        else:
            x, h = op.lbfgs(ff, gg, x0, m=5, tol=1e-6, max_iter=5000)
        t1 = time.perf_counter()
        print(f"      {name:16s}: iterations {len(h) - 1:6d}  f-evals {cnt['f']:6d}  g-evals {cnt['g']:6d}  error {np.linalg.norm(x - xstar):.1e}  time {t1 - t0:.3f}s")

print("C2 iterations vs condition number (quadratic, n = 50, tol 1e-6 relative)")
for kap in (10, 100, 1000, 10000):
    ev = np.logspace(0, np.log10(kap), 50); Ak = (Qm[:50, :50] * 0 + np.linalg.qr(rng.standard_normal((50, 50)))[0]); Ak = (Ak * ev) @ Ak.T
    bk = np.ones(50); gk = lambda x, A=Ak: A @ x - bk
    _, h1 = op.gradient_descent(None, gk, np.zeros(50), lr=2 / (1 + kap), tol=1e-6 * np.linalg.norm(bk), max_iter=10**6)
    beta = ((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1))**2; alpha = 4 / (1 + np.sqrt(kap))**2
    _, h2 = op.momentum_gd(gk, np.zeros(50), lr=alpha, beta=beta, tol=1e-6 * np.linalg.norm(bk), max_iter=10**6)
    from numlib import linalg_iterative as it
    _, rcg = it.conjugate_gradient(Ak, bk, tol=1e-6)
    print(f"   kappa {kap:6d}: GD(optimal step) {len(h1) - 1:6d}  heavy ball {len(h2) - 1:5d}  CG {len(rcg) - 1:4d}   (kappa ln(1e6)/2 = {kap * np.log(1e6) / 2:.0f}, sqrt(kappa) ln(1e6)/2 = {np.sqrt(kap) * np.log(1e6) / 2:.0f})")

print("C3 gradient checking")
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer(); Xb = (bc.data - bc.data.mean(0)) / bc.data.std(0); Xb = np.column_stack([np.ones(len(Xb)), Xb]); yb = bc.target.astype(float)
w = 0.1 * np.random.default_rng(1).standard_normal(Xb.shape[1])
ga = op.logistic_grad(w, Xb, yb, 1e-2)
for hh in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    gfd = np.array([(op.logistic_loss(w + hh * e, Xb, yb, 1e-2) - op.logistic_loss(w - hh * e, Xb, yb, 1e-2)) / (2 * hh) for e in np.eye(len(w))])
    print(f"   h={hh:.0e}: relative error {np.linalg.norm(gfd - ga) / np.linalg.norm(gfd + ga):.2e}")
buggy = lambda w: Xb.T @ (op.sigmoid(Xb @ w) - yb) / len(yb) + 2e-2 * w  # planted bug: factor 2 in the ridge term
gfd = np.array([(op.logistic_loss(w + 1e-6 * e, Xb, yb, 1e-2) - op.logistic_loss(w - 1e-6 * e, Xb, yb, 1e-2)) / 2e-6 for e in np.eye(len(w))])
print(f"   planted bug (2*lam*w): relative error {np.linalg.norm(gfd - buggy(w)) / np.linalg.norm(gfd + buggy(w)):.2e}")

print("C4 a minimal reverse-mode autodiff")
class Var:
    def __init__(self, val, parents=()):
        self.val = val; self.parents = parents; self.grad = 0.0
    def __add__(self, o):
        o = o if isinstance(o, Var) else Var(o); return Var(self.val + o.val, ((self, 1.0), (o, 1.0)))
    __radd__ = __add__
    def __mul__(self, o):
        o = o if isinstance(o, Var) else Var(o); return Var(self.val * o.val, ((self, o.val), (o, self.val)))
    __rmul__ = __mul__
    def __sub__(self, o):
        return self + (-1.0) * o
    def tanh(self):
        t = math.tanh(self.val); return Var(t, ((self, 1 - t * t),))
    def log1pexp(self):
        return Var(math.log1p(math.exp(self.val)), ((self, 1 / (1 + math.exp(-self.val))),))
    def backward(self):
        order, seen = [], set()
        def topo(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p, _ in v.parents:
                    topo(p)
                order.append(v)
        topo(self); self.grad = 1.0
        for v in reversed(order):
            for p, local in v.parents:
                p.grad += local * v.grad
rng = np.random.default_rng(2)
W1 = rng.standard_normal((3, 2)); b1 = rng.standard_normal(3); W2 = rng.standard_normal(3); xin = np.array([0.5, -1.0]); ytrue = 1.0
def loss_np(W1, b1, W2):
    hdd = np.tanh(W1 @ xin + b1); z = W2 @ hdd; return np.log1p(np.exp(z)) - ytrue * z
P1 = [[Var(W1[i, j]) for j in range(2)] for i in range(3)]; Pb = [Var(b1[i]) for i in range(3)]; P2 = [Var(W2[i]) for i in range(3)]
hid = [(P1[i][0] * xin[0] + P1[i][1] * xin[1] + Pb[i]).tanh() for i in range(3)]
z = hid[0] * P2[0] + hid[1] * P2[1] + hid[2] * P2[2]
L = z.log1pexp() - ytrue * z; L.backward()
gad = np.array([P1[i][j].grad for i in range(3) for j in range(2)] + [p.grad for p in Pb] + [p.grad for p in P2])
theta = np.concatenate([W1.ravel(), b1, W2])
def loss_flat(th):
    return loss_np(th[:6].reshape(3, 2), th[6:9], th[9:])
gfd = np.array([(loss_flat(theta + 1e-6 * e) - loss_flat(theta - 1e-6 * e)) / 2e-6 for e in np.eye(12)])
print(f"   loss {L.val:.6f} (numpy {loss_np(W1, b1, W2):.6f}); max |AD - finite difference| over 12 parameters {np.abs(gad - gfd).max():.1e}")

# ---------- D ----------
print("D1 logistic regression (breast cancer, 31 params, lambda = 1e-3): optimizers to relative gap 1e-6")
lam = 1e-3
Fl = lambda w: op.logistic_loss(w, Xb, yb, lam); Gl = lambda w: op.logistic_grad(w, Xb, yb, lam); Hl = lambda w: op.logistic_hess(w, Xb, yb, lam)
wstar, _ = op.newton_opt(Gl, Hl, np.zeros(Xb.shape[1]), tol=1e-12); fstar = Fl(wstar)
Lsm = np.linalg.eigvalsh(Xb.T @ Xb / len(yb)).max() / 4 + lam
print(f"   f* = {fstar:.8f}, smoothness L = {Lsm:.3f}, strong convexity >= lambda = {lam}")
def gap_iters(hist):
    for k, w_ in enumerate(hist):
        if Fl(w_) - fstar <= 1e-6 * abs(fstar):
            return k
    return None
for name in ("GD 1/L", "Nesterov", "Newton", "BFGS", "L-BFGS"):
    t0 = time.perf_counter()
    if name == "GD 1/L":
        _, h = op.gradient_descent(Fl, Gl, np.zeros(31), lr=1 / Lsm, tol=1e-9, max_iter=50000)
    elif name == "Nesterov":
        _, h = op.nesterov(Gl, np.zeros(31), lr=1 / Lsm, beta=0.9, tol=1e-9, max_iter=50000)
    elif name == "Newton":
        _, h = op.newton_opt(Gl, Hl, np.zeros(31), tol=1e-9)
    elif name == "BFGS":
        _, h = op.bfgs(Fl, Gl, np.zeros(31), tol=1e-9)
    else:
        _, h = op.lbfgs(Fl, Gl, np.zeros(31), tol=1e-9)
    t1 = time.perf_counter()
    print(f"   {name:9s}: iterations to gap 1e-6: {gap_iters(h)}  (total {len(h) - 1}, {t1 - t0:.3f}s)")
gi = lambda w, idx: Xb[idx].T @ (op.sigmoid(Xb[idx] @ w) - yb[idx]) / len(idx) + lam * w
for name, fn in (("SGD b=32 lr=0.5/(1+0.1t)", lambda: op.sgd(gi, np.zeros(31), len(yb), lr=0.5, epochs=100, batch_size=32, decay=0.1)),
                 ("SGD b=32 lr=0.5 const", lambda: op.sgd(gi, np.zeros(31), len(yb), lr=0.5, epochs=100, batch_size=32)),
                 ("Adam b=32 lr=0.01", lambda: op.adam(gi, np.zeros(31), len(yb), lr=0.01, epochs=100, batch_size=32))):
    x, h = fn()
    gaps = [Fl(w_) - fstar for w_ in h]
    print(f"   {name:24s}: relative gap after 10/50/100 epochs {gaps[10] / fstar:.1e} {gaps[50] / fstar:.1e} {gaps[100] / fstar:.1e}")

print("D2 one-hidden-layer network on two moons (hand-coded backprop)")
from sklearn.datasets import make_moons
Xm, ym = make_moons(1000, noise=0.2, random_state=0); Xte_m, yte_m = make_moons(1000, noise=0.2, random_state=1)
def train(opt, lr, epochs=200, H=16, seed=0):
    r = np.random.default_rng(seed)
    params = {"W1": r.standard_normal((2, H)) * 1.0, "b1": np.zeros(H), "W2": r.standard_normal(H) / np.sqrt(H), "b2": 0.0}
    m = {k: np.zeros_like(v) for k, v in params.items()}; v2 = {k: np.zeros_like(v) for k, v in params.items()}; t = 0
    for ep in range(epochs):
        perm = r.permutation(len(ym))
        for s in range(0, len(ym), 32):
            idx = perm[s:s + 32]; X_ = Xm[idx]; y_ = ym[idx]
            A1 = np.tanh(X_ @ params["W1"] + params["b1"]); z = A1 @ params["W2"] + params["b2"]; p = op.sigmoid(z)
            dz = (p - y_) / len(idx)
            grads = {"W2": A1.T @ dz, "b2": dz.sum()}
            dA1 = np.outer(dz, params["W2"]) * (1 - A1**2)
            grads["W1"] = X_.T @ dA1; grads["b1"] = dA1.sum(0)
            t += 1
            for k in params:
                if opt == "sgd":
                    params[k] = params[k] - lr * grads[k]
                else:
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]; v2[k] = 0.999 * v2[k] + 0.001 * grads[k]**2
                    params[k] = params[k] - lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v2[k] / (1 - 0.999**t)) + 1e-8)
    pred = lambda X_: op.sigmoid(np.tanh(X_ @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]) > 0.5
    return (pred(Xm) == ym).mean(), (pred(Xte_m) == yte_m).mean()
for opt, lr in (("sgd", 0.01), ("sgd", 0.1), ("sgd", 1.0), ("adam", 0.001), ("adam", 0.01), ("adam", 0.1)):
    tr_acc, te_acc = train(opt, lr)
    print(f"   {opt} lr={lr}: train acc {tr_acc:.3f}  test acc {te_acc:.3f}")
from sklearn.linear_model import LogisticRegression
print("   linear logistic regression test acc", round(LogisticRegression().fit(Xm, ym).score(Xte_m, yte_m), 3))

print("D3 long-only minimum-variance portfolio: projected gradient onto the simplex")
rng = np.random.default_rng(18)
nA = 8; Fct = rng.standard_normal((nA, 2)) * 0.15; Sig = Fct @ Fct.T + np.diag(rng.uniform(0.005, 0.06, nA))
gP = lambda w: 2 * Sig @ w
Lp = 2 * np.linalg.eigvalsh(Sig).max()
wpg, hpg = op.projected_gradient(gP, op.project_simplex, np.ones(nA) / nA, lr=1 / Lp, tol=1e-12, max_iter=100000)
res = optimize.minimize(lambda w: w @ Sig @ w, np.ones(nA) / nA, method="SLSQP", bounds=[(0, None)] * nA, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-15})
w_unc = np.linalg.solve(Sig, np.ones(nA)); w_unc /= w_unc.sum()
print("   projected gradient", wpg.round(4), f"({len(hpg) - 1} its) var {wpg @ Sig @ wpg:.6f}")
print("   SLSQP             ", res.x.round(4), f"var {res.x @ Sig @ res.x:.6f}")
print("   unconstrained (shorting allowed)", w_unc.round(4), f"var {w_unc @ Sig @ w_unc:.6f}")
act = wpg > 1e-8; lagr = 2 * Sig @ wpg
print("   KKT check: gradient on active assets", lagr[act].round(6), " inactive (must be >=)", lagr[~act].round(6))

print("D4 lasso by ISTA vs FISTA vs coordinate descent (diabetes)")
from sklearn.datasets import load_diabetes
from numlib import regression as rg
Xd, yd = load_diabetes(return_X_y=True); Xs = (Xd - Xd.mean(0)) / Xd.std(0); yc = yd - yd.mean(); nd = len(yc); lamL = 1.0
obj = lambda bb: 0.5 / nd * np.sum((yc - Xs @ bb)**2) + lamL * np.abs(bb).sum()
bcd, _ = rg.lasso_cd(Xs, yc, lamL, tol=1e-14); fstar = obj(bcd)
Ld = np.linalg.eigvalsh(Xs.T @ Xs / nd).max()
def prox_grad(fista, iters):
    bb = np.zeros(10); z = bb.copy(); t = 1.0; gaps = []
    for k in range(iters):
        grad = -Xs.T @ (yc - Xs @ z) / nd
        bn = rg.soft_threshold(z - grad / Ld, lamL / Ld)
        if fista:
            tn = (1 + np.sqrt(1 + 4 * t * t)) / 2; z = bn + (t - 1) / tn * (bn - bb); t = tn
        else:
            z = bn
        bb = bn; gaps.append(obj(bb) - fstar)
    return np.array(gaps)
gi_, gf_ = prox_grad(False, 5000), prox_grad(True, 5000)
for k in (10, 100, 1000, 5000):
    print(f"   iteration {k}: ISTA gap {gi_[k - 1]:.2e}  FISTA gap {gf_[k - 1]:.2e}")
print(f"   L = {Ld:.3f}; coordinate descent optimum objective {fstar:.6f}, nonzeros {int(np.sum(bcd != 0))}")
