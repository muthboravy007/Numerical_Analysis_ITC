"""Numerical answers for exercises/ch10_exercises.md."""
import os
import sys
import time

import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import nonlinear_systems as ns  # noqa: E402

np.set_printoptions(precision=8, suppress=True)

# ---------- B ----------
G = lambda v: np.array([(1 + np.cos(v[1])) / 4, (1 + np.sin(v[0])) / 3])
xf, hist = ns.fixed_point_system(G, [0.0, 0.0], tol=1e-8)
print("B1 first iterates", [np.round(h, 6) for h in hist[:4]], " solution", xf, " iterations", len(hist) - 1)
x = np.zeros(2); k = 0
while True:
    xo = x.copy(); x[0] = (1 + np.cos(x[1])) / 4; x[1] = (1 + np.sin(x[0])) / 3; k += 1
    if np.abs(x - xo).max() < 1e-8:
        break
print("   Gauss-Seidel variant iterations", k, x)
Jg = np.array([[0, -np.sin(xf[1]) / 4], [np.cos(xf[0]) / 3, 0]])
print("   J_G at fixed point", Jg, " rho", max(abs(np.linalg.eigvals(Jg))), " ||J_G||_inf", np.abs(Jg).sum(1).max())

F2 = lambda v: np.array([v[0]**2 + v[1]**2 - 5, v[0] - v[1] + 1])
J2 = lambda v: np.array([[2 * v[0], 2 * v[1]], [1.0, -1.0]])
xs, hist = ns.newton_system(F2, J2, [2.0, 3.0])
print("B2a Newton from (2,3):", [np.round(h, 8) for h in hist], " errors", [f"{np.linalg.norm(h - [1, 2], np.inf):.2e}" for h in hist])
F3 = lambda v: np.array([v[0] + v[1] + v[2] - 3, v[0]**2 + v[1]**2 + v[2]**2 - 5, np.exp(v[0]) + v[1] * v[2] - 3])
J3 = lambda v: np.array([[1, 1, 1], [2 * v[0], 2 * v[1], 2 * v[2]], [np.exp(v[0]), v[2], v[1]]])
x0 = np.array([0.1, 1.2, 1.7])
print("B2b F(x0)", F3(x0), "\n J(x0)\n", J3(x0), "\n step", np.linalg.solve(J3(x0), -F3(x0)))
xs, hist = ns.newton_system(F3, J3, x0)
print("   iterates", [np.round(h, 8) for h in hist], "\n   errors", [f"{np.linalg.norm(h - [0, 1, 2], np.inf):.1e}" for h in hist])

xb, hist = ns.broyden(F2, [2.0, 3.0], J0=J2(np.array([2.0, 3.0])))
print("B3 Broyden from (2,3):", [np.round(h, 8) for h in hist[:4]], " errors", [f"{np.linalg.norm(h - [1, 2], np.inf):.1e}" for h in hist])
# first update by hand
x0 = np.array([2.0, 3.0]); A0 = J2(x0); s = np.linalg.solve(A0, -F2(x0)); x1 = x0 + s; y = F2(x1) - F2(x0)
A1 = A0 + np.outer(y - A0 @ s, s) / (s @ s)
print("   s1", s, " x1", x1, " y1", y, "\n   A1\n", A1, "\n   J(x1)\n", J2(x1), "\n   secant check A1 s - y", A1 @ s - y, " x2", x1 + np.linalg.solve(A1, -F2(x1)))

xsd, hsd = ns.steepest_descent_system(F2, J2, [0.0, 0.0], tol=1e-5)
g = lambda v: np.sum(F2(v)**2)
print("B4 steepest descent from (0,0): g0", g(np.zeros(2)), " grad", 2 * J2(np.zeros(2)).T @ F2(np.zeros(2)))
for i, h in enumerate(hsd[:4]):
    print(f"   k={i}: x {np.round(h, 6)} g {g(h):.6f}")
print("   iterations", len(hsd) - 1, "final", hsd[-1], g(hsd[-1]), " det J there", np.linalg.det(J2(hsd[-1])))
xsd2, hsd2 = ns.steepest_descent_system(F2, J2, [0.0, 1.0], tol=1e-5)
print("   from (0,1):", len(hsd2) - 1, "iterations ->", hsd2[-1], "g", g(hsd2[-1]), " Newton from there:", ns.newton_system(F2, J2, hsd2[-1])[0])

for N in (1, 2, 4, 8):
    xh, path = ns.homotopy(F2, J2, [1.0, 0.0], N=N)
    print(f"B5 homotopy from (1,0), N={N}: end {np.round(xh, 6)} err {np.linalg.norm(xh - [1, 2], np.inf):.1e}" + (f"  path {[np.round(p, 4) for p in path]}" if N == 2 else ""))

F6 = lambda v: np.array([v[0]**2 + v[1]**2 - 2, v[0] + v[1] - 2])
J6 = lambda v: np.array([[2 * v[0], 2 * v[1]], [1.0, 1.0]])
x = np.array([2.0, 0.5]); e = []
for k in range(12):
    try:
        x = x + np.linalg.solve(J6(x), -F6(x))
    except np.linalg.LinAlgError:
        break
    e.append(np.linalg.norm(x - [1, 1], np.inf))
print("B6 tangency root (1,1), Newton errors", [f"{v:.2e}" for v in e], "\n   ratios", np.round(np.array(e[1:]) / np.array(e[:-1]), 4))

# ---------- C ----------
print("C1 Broyden tridiagonal problem")
def Ftri(x):
    xm = np.concatenate([[0.0], x[:-1]]); xp = np.concatenate([x[1:], [0.0]])
    return (3 - 2 * x) * x - xm - 2 * xp + 1
def Jtri(x):
    n = len(x); return np.diag(3 - 4 * x) - np.eye(n, k=-1) - 2 * np.eye(n, k=1)
for n in (100, 1000):
    x0 = -np.ones(n)
    cnt = [0]
    def Fc(x):
        cnt[0] += 1; return Ftri(x)
    t0 = time.perf_counter(); xn, hn = ns.newton_system(Fc, Jtri, x0, tol=1e-10); t1 = time.perf_counter(); nf_n = cnt[0]
    cnt[0] = 0; t2 = time.perf_counter(); xb, hb = ns.broyden(Fc, x0, J0=Jtri(x0), tol=1e-10); t3 = time.perf_counter(); nf_b = cnt[0]
    cnt[0] = 0; t4 = time.perf_counter(); sol = optimize.root(Fc, x0, method="hybr", tol=1e-10); t5 = time.perf_counter(); nf_h = cnt[0]
    cnt[0] = 0; t6 = time.perf_counter(); solk = optimize.root(Fc, x0, method="krylov", tol=1e-10); t7 = time.perf_counter(); nf_k = cnt[0]
    print(f"   n={n}: Newton {len(hn)-1} its {nf_n} F-evals {t1-t0:.3f}s | Broyden {len(hb)-1} its {nf_b} F-evals {t3-t2:.3f}s | hybr {nf_h} F-evals {t5-t4:.3f}s | Newton-Krylov {nf_k} F-evals {t7-t6:.3f}s | max|F| {np.abs(Ftri(xn)).max():.1e} {np.abs(Ftri(xb)).max():.1e} {np.abs(Ftri(sol.x)).max():.1e} {np.abs(Ftri(solk.x)).max():.1e}")

print("C2 Newton basins for z^3 = 1 as a real 2x2 system")
roots = np.exp(2j * np.pi * np.arange(3) / 3)
g = np.linspace(-2, 2, 401); Z = g[None, :] + 1j * g[:, None]
Zk = Z.copy(); its = np.zeros(Z.shape, int); done = np.zeros(Z.shape, bool)
for k in range(60):
    with np.errstate(all="ignore"):
        Zk = np.where(done, Zk, Zk - (Zk**3 - 1) / (3 * Zk**2))
    d = np.min(np.abs(Zk[..., None] - roots), axis=-1) < 1e-10
    its[d & ~done] = k + 1; done |= d
which = np.argmin(np.abs(Zk[..., None] - roots), axis=-1)
print("   fraction to root 1, w, w^2:", [round(float(np.mean(done & (which == i))), 4) for i in range(3)], " not converged", round(float(np.mean(~done)), 5), " mean iterations", round(float(its[done].mean()), 2))
r = np.abs(Z)
for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 2.9)):
    msk = (r >= lo) & (r < hi) & done
    print(f"   |z0| in [{lo},{hi}): mean iterations {its[msk].mean():.2f}")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    plt.figure(figsize=(5, 5)); plt.imshow(np.where(done, which, -1), extent=(-2, 2, -2, 2), origin="lower", cmap="viridis")
    plt.title("Newton basins for $z^3=1$"); plt.savefig(os.path.join(os.path.dirname(__file__), "figures", "ch10_newton_basins.png"), dpi=100); plt.close()
except Exception as ex:  # pragma: no cover
    print("   (plot skipped)", ex)

print("C3 pure vs damped Newton on F = (arctan x + 0.1 y, y - 0.5 x)")
Fa = lambda v: np.array([np.arctan(v[0]) + 0.1 * v[1], v[1] - 0.5 * v[0]])
Ja = lambda v: np.array([[1 / (1 + v[0]**2), 0.1], [-0.5, 1.0]])
def newton_run(x0, damped, maxit=50):
    x = np.array(x0, float)
    for k in range(maxit):
        f = Fa(x); dx = np.linalg.solve(Ja(x), -f); t = 1.0
        if damped:
            while np.linalg.norm(Fa(x + t * dx)) > (1 - 1e-4 * t) * np.linalg.norm(f) and t > 1e-8:
                t /= 2
        x = x + t * dx
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e8:
            return None
        if np.linalg.norm(Fa(x)) < 1e-12:
            return k + 1
    return None
rng = np.random.default_rng(0)
starts = rng.uniform(-10, 10, (1000, 2))
pure = [newton_run(s, False) for s in starts]; damp = [newton_run(s, True) for s in starts]
print("   success rate pure", np.mean([p is not None for p in pure]), " damped", np.mean([p is not None for p in damp]),
      " mean its (successes) pure", round(np.mean([p for p in pure if p]), 2), " damped", round(np.mean([p for p in damp if p]), 2))
print("   pure Newton from (x0, 0):", end=" ")
for x0 in (1.5, 1.6, 1.7, 1.8, 1.9, 2.0):
    print(f"{x0}:{'ok' if newton_run([x0, 0.0], False) else 'fail'}", end=" ")
print()

print("C4 finite-difference Jacobian error vs h (3x3 system of B2 at x0)")
x0 = np.array([0.1, 1.2, 1.7]); Jex = J3(x0)
for h in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    Jf = np.column_stack([(F3(x0 + h * e) - F3(x0)) / h for e in np.eye(3)])
    Jc = np.column_stack([(F3(x0 + h * e) - F3(x0 - h * e)) / (2 * h) for e in np.eye(3)])
    print(f"   h={h:.0e}: forward err {np.abs(Jf - Jex).max():.1e}  central err {np.abs(Jc - Jex).max():.1e}")

# ---------- D ----------
print("D1 Gamma MLE")
rng = np.random.default_rng(7)
data = rng.gamma(shape=2.5, scale=1.6, size=400)
n = len(data); mean = data.mean(); mlog = np.log(data).mean()
def score(p):
    k, th = p
    return np.array([np.sum(np.log(data)) - n * np.log(th) - n * special.digamma(k), np.sum(data) / th**2 - n * k / th])
def hess(p):
    k, th = p
    return np.array([[-n * special.polygamma(1, k), -n / th], [-n / th, -2 * np.sum(data) / th**3 + n * k / th**2]])
k0 = mean**2 / data.var(); th0 = data.var() / mean
p, hist = ns.newton_system(score, hess, [k0, th0], tol=1e-12)
cov = np.linalg.inv(-hess(p)); se = np.sqrt(np.diag(cov))
kf, _, thf = stats.gamma.fit(data, floc=0)
print(f"   method of moments start k {k0:.5f} theta {th0:.5f}; Newton {len(hist)-1} its -> k {p[0]:.6f} theta {p[1]:.6f}; SE {se.round(4)}; scipy fit k {kf:.6f} theta {thf:.6f}")
print("   95% CI for k:", np.round([p[0] - 1.96 * se[0], p[0] + 1.96 * se[0]], 4), " corr(k,theta)", round(cov[0, 1] / (se[0] * se[1]), 4))

print("D2 Cournot oligopoly with nonlinear demand")
a, eta = 100.0, 1.5
c = np.array([10.0, 12.0, 15.0])
price = lambda Q: a * Q**(-1 / eta)
dprice = lambda Q: -(1 / eta) * a * Q**(-1 / eta - 1)
d2price = lambda Q: (1 / eta) * (1 / eta + 1) * a * Q**(-1 / eta - 2)
def Fc(q):
    Q = q.sum(); return price(Q) + q * dprice(Q) - c
def Jc(q):
    Q = q.sum(); n = len(q)
    J = np.outer(np.ones(n), np.ones(n)) * dprice(Q) + np.outer(q, np.ones(n)) * d2price(Q)
    J[np.diag_indices(n)] += dprice(Q)
    return J
q, hist = ns.newton_system(Fc, Jc, np.ones(3) * 5)
Q = q.sum()
print("   equilibrium q", q.round(5), " Q", round(Q, 5), " price", round(price(Q), 5), " markups (p-c)/p", ((price(Q) - c) / price(Q)).round(4),
      " shares", (q / Q).round(4), " Newton its", len(hist) - 1)
print("   check Lerner: (p-c)/p = s_i/eta:", (q / Q / eta).round(4))

print("D3 multinomial logistic regression by Newton (iris)")
from sklearn.datasets import load_iris
Xi, yi = load_iris(return_X_y=True)
Xi = (Xi - Xi.mean(0)) / Xi.std(0); X1 = np.column_stack([np.ones(len(Xi)), Xi]); K = 3; p_ = X1.shape[1]
Y = np.eye(K)[yi]; lam = 1.0
def probs(W):
    S = X1 @ np.column_stack([np.zeros(p_), W.reshape(K - 1, p_).T]); S -= S.max(1, keepdims=True)
    P = np.exp(S); return P / P.sum(1, keepdims=True)
def grad(W):
    P = probs(W); G = (X1.T @ (P - Y))[:, 1:].T.ravel(); Wm = W.reshape(K - 1, p_).copy(); Wm[:, 0] = 0
    return G + lam * Wm.ravel()
def hessian(W):
    P = probs(W)[:, 1:]; H = np.zeros(((K - 1) * p_, (K - 1) * p_))
    for i in range(K - 1):
        for j in range(K - 1):
            w = P[:, i] * ((i == j) - P[:, j])
            H[i * p_:(i + 1) * p_, j * p_:(j + 1) * p_] = X1.T @ (w[:, None] * X1)
    R = np.tile(np.r_[0.0, np.ones(p_ - 1)], K - 1)
    return H + lam * np.diag(R)
W, hist = ns.newton_system(grad, hessian, np.zeros((K - 1) * p_), tol=1e-12)
gn = [np.linalg.norm(grad(h)) for h in hist]
acc = (probs(W).argmax(1) == yi).mean()
def objective(W):
    P = probs(W); Wm = W.reshape(K - 1, p_).copy(); Wm[:, 0] = 0
    return -np.sum(Y * np.log(P)) + 0.5 * lam * np.sum(Wm**2)
cnt = [0]
res = optimize.minimize(objective, np.zeros((K - 1) * p_), jac=grad, method="BFGS", options={"gtol": 1e-8})
print("   Newton iterations", len(hist) - 1, " ||grad|| per iteration", [f"{v:.1e}" for v in gn], " train accuracy", round(acc, 4))
print("   BFGS on the same objective:", res.nit, "iterations, max |W_newton - W_bfgs|", f"{np.abs(res.x - W).max():.1e}", " objective", round(objective(W), 6))
