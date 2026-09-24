"""Verifies every numerical answer in the assessments/ exam solutions."""
import os
import sys

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib.floating import fl  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
print("=================== EXAM 1 (Semester 1 midterm) ===================")
x = 500.0
s1, s0 = fl(np.sqrt(fl(x + 1, 4, "chop")), 4, "chop"), fl(np.sqrt(x), 4, "chop")
naive = fl(x * fl(s1 - s0, 4, "chop"), 4, "chop")
den = fl(s1 + s0, 4, "chop"); ration = fl(x / den, 4, "chop")
true = x * (np.sqrt(x + 1) - np.sqrt(x))
print(f"Q1 sqrt(501)->{s1}, sqrt(500)->{s0}, naive {naive} (rel err {abs(naive - true) / true:.3e}), rationalised {ration} (rel err {abs(ration - true) / true:.3e}), true {true:.8f}")
f = lambda x: x**3 + 2 * x - 5
a, b = 1.0, 2.0
for k in range(3):
    p = (a + b) / 2
    print(f"Q2 bisection step {k + 1}: p={p}, f(p)={f(p):.6f}")
    a, b = (p, b) if f(a) * f(p) > 0 else (a, p)
x = 1.5
for k in range(2):
    x = x - f(x) / (3 * x * x + 2); print(f"Q2 Newton x{k + 1} = {x:.8f}")
root = np.roots([1, 0, 2, -5]); root = root[np.isreal(root)].real[0]
print(f"   root {root:.10f}; bisection steps for 1e-6 on [1,2]: n >= log2(1/1e-6) = {np.log2(1e6):.2f} -> 20")
for name, gd in (("g1=(5-x^3)/2", lambda p: -1.5 * p * p), ("g2=(5-2x)^(1/3)", lambda p: -2 / 3 * (5 - 2 * p)**(-2 / 3)), ("g3=5/(x^2+2)", lambda p: -10 * p / (p * p + 2)**2)):
    print(f"Q3 {name}: g'(p) = {gd(root):.6f}")
g2 = lambda x: (5 - 2 * x)**(1 / 3); x = 1.0
it = [x]
for _ in range(4):
    x = g2(x); it.append(x)
print("   g2 iterates from 1:", np.round(it, 6))
xs = np.array([0.0, 1, 2, 4]); ys = np.array([1.0, 2, 4, 16])  # samples of 2^x
D = np.zeros((4, 4)); D[:, 0] = ys
for j in range(1, 4):
    D[j:, j] = (D[j:, j - 1] - D[j - 1:-1, j - 1]) / (xs[j:] - xs[:-j])
print("Q4 divided differences\n", D)
P = lambda t: D[0, 0] + D[1, 1] * t + D[2, 2] * t * (t - 1) + D[3, 3] * t * (t - 1) * (t - 2)
print(f"   P(3) = {P(3.0):.6f} (true 2^3 = 8); P(x) coefficients (highest first)", np.polyfit(xs, ys, 3).round(6), " error-bound factor |3(3-1)(3-2)(3-4)| =", abs(3 * 2 * 1 * -1), " f^(4) = (ln2)^4 2^xi in", [np.log(2)**4, np.log(2)**4 * 16])
g = lambda t: np.exp(-t * t); ex = integrate.quad(g, 0, 1)[0]
h = 0.25; xv = np.linspace(0, 1, 5); gv = g(xv)
T4 = h * (gv[0] / 2 + gv[1:-1].sum() + gv[-1] / 2); S4 = h / 3 * (gv[0] + 4 * gv[1] + 2 * gv[2] + 4 * gv[3] + gv[4])
T2 = 0.5 * (gv[0] / 2 + gv[2] + gv[-1] / 2)
print(f"Q5 values {gv.round(6)}; T4 {T4:.6f} S4 {S4:.6f} T2 {T2:.6f} Richardson (4T4-T2)/3 {(4 * T4 - T2) / 3:.6f} exact {ex:.6f}")
print(f"   errors T4 {abs(T4 - ex):.2e} S4 {abs(S4 - ex):.2e}; trapezoid bound (b-a)h^2/12 max|f''| = {1 / 12 * h * h * 2:.5f} (max|f''| = 2 at 0)")
fy = lambda t, y: -2 * t * y; h = 0.2
eu = 1 + h * fy(0, 1)
k1 = fy(0, 1); k2 = fy(h, 1 + h * k1); heun = 1 + h / 2 * (k1 + k2)
K1 = h * fy(0, 1); K2 = h * fy(h / 2, 1 + K1 / 2); K3 = h * fy(h / 2, 1 + K2 / 2); K4 = h * fy(h, 1 + K3); rk4 = 1 + (K1 + 2 * K2 + 2 * K3 + K4) / 6
print(f"Q6 y(0.2) exact {np.exp(-0.04):.8f}: Euler {eu:.8f} Heun {heun:.8f} RK4 {rk4:.8f} (K's {K1:.6f} {K2:.6f} {K3:.6f} {K4:.6f})")
print("Q7 Euler stable for y'=-50y iff |1-50h|<=1 -> h <= 0.04")

print("=================== EXAM 2 (Semester 1 final) ===================")
A = np.array([[1.0, 2, -1], [4, 3, 1], [2, 2, 3]]); b = np.array([2.0, 12, 13])
import scipy.linalg as sla
P_, L_, U_ = sla.lu(A)
print("Q1 P^T (row order)\n", P_.T, "\n L\n", L_, "\n U\n", U_, "\n x", np.linalg.solve(A, b), " det", np.linalg.det(A))
S = np.array([[4.0, 2, 2], [2, 10, 4], [2, 4, 6]])
print("Q2 Cholesky\n", np.linalg.cholesky(S), " minors", [np.linalg.det(S[:k, :k]) for k in (1, 2, 3)], " solve S x = (8,16,12):", np.linalg.solve(S, [8.0, 16, 12]))
A3 = np.array([[5.0, -1, 2], [1, 6, -2], [2, -1, 7]]); b3 = np.array([6.0, 5, 8])
xs3 = np.linalg.solve(A3, b3); xj = np.zeros(3); xg = np.zeros(3)
for k in range(2):
    xj = (b3 - (A3 - np.diag(np.diag(A3))) @ xj) / np.diag(A3)
    for i in range(3):
        xg[i] = (b3[i] - A3[i, :i] @ xg[:i] - A3[i, i + 1:] @ xg[i + 1:]) / A3[i, i]
    print(f"Q3 k={k + 1}: Jacobi {xj}  GS {xg}")
Tj = np.eye(3) - np.diag(1 / np.diag(A3)) @ A3
print("   exact", xs3, " ||Tj||inf", np.abs(Tj).sum(1).max(), " rho(Tj)", max(abs(np.linalg.eigvals(Tj))))
A4 = np.array([[2.0, 1], [2, 1.01]]); A4i = np.linalg.inv(A4)
k4 = np.abs(A4).sum(1).max() * np.abs(A4i).sum(1).max()
x4 = np.linalg.solve(A4, [3.0, 3.01]); x4p = np.linalg.solve(A4, [3.0, 3.02])
print("Q4 A^-1\n", A4i, " kappa_inf", k4, " x", x4, " perturbed x", x4p, " rel db", 0.01 / 3.02, " rel dx", np.abs(x4p - x4).max() / np.abs(x4).max())
xd = np.array([0.0, 1, 2, 3, 4]); yd = np.array([1.1, 1.9, 3.2, 3.8, 5.0])
a1, a0 = np.polyfit(xd, yd, 1); res = yd - (a0 + a1 * xd)
print(f"Q5 sums {xd.sum()} {yd.sum()} {(xd**2).sum()} {(xd*yd).sum():.1f}; line a0 {a0:.4f} a1 {a1:.4f} SSE {np.sum(res**2):.4f} R2 {1 - np.sum(res**2) / np.sum((yd - yd.mean())**2):.4f}")
c2 = np.polyfit(xd, yd, 2); print("   quadratic", c2.round(5), " SSE", np.sum((yd - np.polyval(c2, xd))**2).round(5))
nodes = 1 + np.cos((2 * np.arange(1, 4) - 1) * np.pi / 6)
print("Q6 Chebyshev nodes on [0,2]:", nodes.round(6), " Chebyshev bound max|prod| = 2((b-a)/4)^3 =", 2 * (2 / 4)**3)
tt = np.linspace(0, 2, 200001); prod_eq = np.abs(tt * (tt - 1) * (tt - 2)).max(); prod_ch = np.abs(np.prod([tt - n for n in nodes], axis=0)).max()
print(f"   max|prod| equispaced {prod_eq:.6f}  Chebyshev {prod_ch:.6f}; f=e^x: bound e^2/6*{prod_ch:.4f} = {np.exp(2) / 6 * prod_ch:.5f}")
Pc = np.polyfit(nodes, np.exp(nodes), 2); print("   actual max error e^x Chebyshev interp:", np.abs(np.exp(tt) - np.polyval(Pc, tt)).max().round(6))
v = np.array([1.0, 0, -1, 2]); print("Q7 DFT", np.fft.fft(v))
fr = lambda x: np.exp(x) - 3 * x
x = 0.5
for k in range(3):
    x = x - fr(x) / (np.exp(x) - 3); print(f"Q8 Newton x{k + 1} = {x:.10f}")

print("=================== EXAM 3 (Semester 2 midterm) ===================")
A = np.array([[3.0, 1, 0], [1, 3, 1], [0, 1, 3]]); x = np.array([1.0, 1, 1])
for k in range(3):
    y = A @ x; mu = np.abs(y).max() * np.sign(y[np.argmax(np.abs(y))]); x = y / mu
    print(f"Q1 k={k + 1}: y {y} mu {mu:.6f} x {x}  Rayleigh {x @ A @ x / (x @ x):.6f}")
print("   eig", np.linalg.eigvalsh(A), " 3+sqrt2", 3 + 2**0.5)
xi = np.array([1.0, 0, 0]); B = A - 1.5 * np.eye(3)
for k in range(3):
    xi = np.linalg.solve(B, xi); xi /= np.linalg.norm(xi)
    print(f"   inverse iteration q=1.5 k={k + 1}: Rayleigh {xi @ A @ xi:.8f}")
x = np.array([3.0, 4.0]); alpha = -5.0; w = (x - alpha * np.array([1, 0])); w /= np.linalg.norm(w); H = np.eye(2) - 2 * np.outer(w, w)
print("Q2 w", w, " H\n", H, " Hx", H @ x)
M2 = np.array([[2.0, 1], [1, 2]]); Q, R = np.linalg.qr(M2); print("   QR of [[2,1],[1,2]]: R\n", R, "\n   RQ\n", R @ Q)
A3 = np.array([[1.0, 1], [1, 1], [1, -1]])
U, s, Vt = np.linalg.svd(A3); print("Q3 A^TA", A3.T @ A3, " sigma", s, "\n V\n", Vt.T, "\n U\n", U, "\n pinv\n", np.linalg.pinv(A3), " x+ for b=(1,2,3):", np.linalg.pinv(A3) @ [1.0, 2, 3])
F = lambda v: np.array([v[0]**2 - v[1] - 1, v[0] + v[1]**2 - 3]); J = lambda v: np.array([[2 * v[0], -1], [1, 2 * v[1]]])
x0 = np.array([1.5, 1.0]); s0 = np.linalg.solve(J(x0), -F(x0)); x1 = x0 + s0
print("Q4 F(x0)", F(x0), " J\n", J(x0), " s", s0, " x1", x1, " F(x1)", F(x1))
y = F(x1) - F(x0); A0 = J(x0); A1 = A0 + np.outer(y - A0 @ s0, s0) / (s0 @ s0); print("   Broyden A1\n", A1, " J(x1)\n", J(x1))
from scipy.optimize import fsolve
print("   root", fsolve(F, x1))
h = 0.25; xi = np.array([0.25, 0.5, 0.75])
M = np.diag(np.full(3, 2 + h * h * 1.0)) - np.eye(3, k=1) - np.eye(3, k=-1)
rhs = -h * h * (-xi); rhs[-1] += 2.0
w = np.linalg.solve(M, rhs)
exact = lambda x: x + np.sinh(x) / np.sinh(1)
print("Q5 y''=y-x, y(0)=0,y(1)=2; matrix diag", 2 + h * h, " rhs", rhs, " w", w, " exact", exact(xi))
lam = 0.5; x = np.linspace(0, 1, 5); u0 = np.sin(np.pi * x)
w1 = u0.copy(); w1[1:-1] = lam * u0[:-2] + (1 - 2 * lam) * u0[1:-1] + lam * u0[2:]
print("Q6 FTCS lambda 0.5 one step:", w1[1:-1], " exact", np.exp(-np.pi**2 * 0.03125) * u0[1:-1], "(k = lambda h^2 = 0.03125)")
lamc = 1.0; n = 3; Tm = np.diag(np.full(n, 1 + lamc)) - lamc / 2 * (np.eye(n, k=1) + np.eye(n, k=-1)); Tp = np.diag(np.full(n, 1 - lamc)) + lamc / 2 * (np.eye(n, k=1) + np.eye(n, k=-1))
wcn = np.linalg.solve(Tm, Tp @ u0[1:-1]); print("   CN lambda 1 one step (k=0.0625):", wcn, " exact", np.exp(-np.pi**2 * 0.0625) * u0[1:-1])

print("=================== EXAM 4 (Semester 2 final) ===================")
n = 12; xbar, ybar, Sxx, Sxy, Syy = 5.0, 20.0, 60.0, 90.0, 150.0
b1 = Sxy / Sxx; b0 = ybar - b1 * xbar; SSE = Syy - b1 * Sxy; s2 = SSE / (n - 2); se1 = np.sqrt(s2 / Sxx); tq = stats.t.ppf(0.975, n - 2)
print(f"Q1 b1 {b1} b0 {b0} SSE {SSE} s2 {s2} se(b1) {se1:.5f} t {b1 / se1:.4f} CI [{b1 - tq * se1:.4f}, {b1 + tq * se1:.4f}] (t_0.975,10 = {tq:.4f}) R2 {b1 * Sxy / Syy:.4f} p {2 * stats.t.sf(b1 / se1, n - 2):.2e}")
R = np.array([[2.0, 4, 2], [0, 3, 1], [0, 0, 2]]); qty = np.array([10.0, 6, 2]); rnorm = 1.5
beta = np.linalg.solve(R, qty); Rinv = np.linalg.inv(R)
print("Q2 beta", beta, " RSS", rnorm**2, " (X^TX)^-1 diag", np.diag(Rinv @ Rinv.T), " s2 (n=8,p=3)", rnorm**2 / 5, " se", np.sqrt(rnorm**2 / 5 * np.diag(Rinv @ Rinv.T)))
sig = np.array([4.0, 1.0]); uty = np.array([8.0, 3.0])
for lam in (0.0, 1.0, 4.0):
    coef = sig * uty / (sig**2 + lam); print(f"Q3 lambda {lam}: coefficients in V-basis {coef}, df {np.sum(sig**2 / (sig**2 + lam)):.4f}")
bo = np.array([2.5, -0.8, 0.3]); print("   lasso lambda 0.5:", np.sign(bo) * np.maximum(np.abs(bo) - 0.5, 0))
C = np.array([[5.0, 2], [2, 2]]); ev, V = np.linalg.eigh(C); ev, V = ev[::-1], V[:, ::-1]
print("Q4 eig", ev, " V\n", V, " explained", ev / ev.sum(), " score of (2,1) centred:", V.T @ np.array([2.0, 1]))
Xl = np.array([[1.0, -1], [1, 0], [1, 1], [1, 2]]); yl = np.array([0.0, 1, 0, 1])
mu = np.full(4, 0.5); W = mu * (1 - mu); z = (yl - mu) / W
b1l = np.linalg.solve(Xl.T @ (W[:, None] * Xl), Xl.T @ (W * z)); print("Q5 X^TWX", Xl.T @ (W[:, None] * Xl), " X^TWz", Xl.T @ (W * z), " beta1", b1l)
Aq = np.diag([1.0, 9.0]); print(f"Q6 GD optimal step {2 / 10}, rate {(9 - 1) / (9 + 1)}, max stable step {2 / 9:.4f}; heavy-ball rate {(3 - 1) / (3 + 1)}")
from scipy.optimize import minimize
r = minimize(lambda v: (v[0] - 2)**2 + (v[1] - 1)**2, [0, 0], constraints=[{"type": "ineq", "fun": lambda v: 1 - v[0] - v[1]}]); print("   KKT min (x-2)^2+(y-1)^2 s.t. x+y<=1:", r.x, " mu", 2 * (2 - r.x[0]))
x1, x2 = 2.0, 1.0; v1 = x1 * x2; v2 = v1 + np.log(x1); fval = v2**2
g1 = 2 * v2 * (x2 + 1 / x1); g2 = 2 * v2 * x1
print(f"Q7 f=(x1 x2 + ln x1)^2 at (2,1): f {fval:.6f} df/dx1 {g1:.6f} df/dx2 {g2:.6f}")
A8 = np.array([[3.0, 1], [1, 3]]); U, s, Vt = np.linalg.svd(A8); print("Q8 sigma", s, " rank-1", s[0] * np.outer(U[:, 0], Vt[0]), " error 2-norm", s[1], " F", s[1])
