"""Numerical answers for exercises/ch09_exercises.md."""
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from numlib import eigen as eg  # noqa: E402
from numlib import linalg_iterative as it  # noqa: E402

np.set_printoptions(precision=6, suppress=True)

# ---------- B ----------
A1 = np.array([[6.0, 2, 1], [1, -5, 0], [2, 1, 4]])
print("B1 discs", eg.gershgorin_discs(A1), " col discs", eg.gershgorin_discs(A1.T), " eig", np.linalg.eigvals(A1))

T = np.array([[2.0, -1, 0], [-1, 2, -1], [0, -1, 2]])
mu, x, hist = eg.power_method_inf(T, [1.0, 0, 0], max_iter=8)
print("B2 inf-norm power mu:", np.round(hist, 6), " x", x, " lambda1", 2 + 2**0.5)
h = np.array(hist)
ait = h[:-2] - (h[1:-1] - h[:-2])**2 / (h[2:] - 2 * h[1:-1] + h[:-2])
print("   Aitken:", np.round(ait, 6))
mu, x, hist = eg.symmetric_power_method(T, [1.0, 0, 0], max_iter=8)
print("B3 symmetric power (Rayleigh):", np.round(hist, 6), " errors", np.array([abs(v - (2 + 2**0.5)) for v in hist]))
lam, v, hist = eg.inverse_iteration(T, shift=0.5, x0=[1.0, 1, 1], tol=1e-14)
print("B4 inverse iteration q=0.5:", np.round(hist[:6], 8), " v", v, " exact", 2 - 2**0.5, " ratio |0.586-0.5|/|2-0.5| =", (2 - 2**0.5 - 0.5) / 1.5)
v1 = np.array([1.0, -2**0.5, 1]); l1 = 2 + 2**0.5
B = eg.wielandt_deflation(T, l1, v1)
print("B5 Wielandt B\n", B, "\n  eig B", np.linalg.eigvals(B), " eig of 2x2 block", np.linalg.eigvals(np.delete(np.delete(B, 1, 0), 1, 1)))
x = np.array([2.0, -1, 2]); alpha = -np.copysign(np.linalg.norm(x), x[0]); w = x - alpha * np.array([1.0, 0, 0]); w /= np.linalg.norm(w)
P = np.eye(3) - 2 * np.outer(w, w)
print("B6a alpha", alpha, "w", w, " P x", P @ x, "\n P\n", P)
A6 = np.array([[5.0, 1, 2, 2], [1, 4, 1, 0], [2, 1, 3, 1], [2, 0, 1, 2]])
T6, Q6 = eg.householder_tridiagonal(A6)
print("B6b tridiagonal\n", T6, "\n  eig A", np.linalg.eigvalsh(A6), " eig T", np.linalg.eigvalsh(T6))
A7 = np.array([[3.0, 1], [1, 1]])
Ak = A7.copy()
for k in range(1, 5):
    Q, R = np.linalg.qr(Ak); Ak = R @ Q
    print(f"B7 QR step {k}:\n", Ak, f"  |a21| ratio {abs(Ak[1, 0]):.6f}")
print("   eig", np.linalg.eigvalsh(A7), " ratio lambda2/lambda1", (2 - 2**0.5) / (2 + 2**0.5))
A8 = np.array([[2.0, 2], [1, -1]])
U, s, Vt = np.linalg.svd(A8)
print("B8a SVD sigma", s, "\n U\n", U, "\n Vt\n", Vt, "\n A^TA", A8.T @ A8)
A8b = np.array([[1.0, 2], [2, 4], [0, 0]])
U, s, Vt = np.linalg.svd(A8b); pinv = np.linalg.pinv(A8b)
print("B8b sigma", s, " pinv\n", pinv, "\n min-norm LS for b=(1,1,1):", pinv @ np.array([1.0, 1, 1]), " residual", A8b @ pinv @ np.array([1.0, 1, 1]) - np.array([1.0, 1, 1]))
A9 = np.array([[3.0, 1], [1, 3], [1, 1]])
U, s, Vt = np.linalg.svd(A9); A1r = s[0] * np.outer(U[:, 0], Vt[0])
print("B9 sigma", s, " u1", U[:, 0], " v1", Vt[0], "\n rank-1\n", A1r, "\n ||A-A1||_2", np.linalg.norm(A9 - A1r, 2), " ||A-A1||_F", np.linalg.norm(A9 - A1r), "A^TA", A9.T @ A9)

# ---------- C ----------
print("C1 iteration counts on symmetric matrices with prescribed spectrum")
rng = np.random.default_rng(0)
n = 200
Qr, _ = np.linalg.qr(rng.standard_normal((n, n)))
for gap in (0.5, 0.9, 0.99):
    ev = np.concatenate([[1.0, gap], rng.uniform(0, gap * 0.9, n - 2)])
    A = (Qr * ev) @ Qr.T
    x0 = rng.standard_normal(n)
    lp, _, hp = eg.power_iteration(A, x0, tol=1e-12)
    li, _, hi = eg.inverse_iteration(A, shift=1.05, x0=x0, tol=1e-12)
    pert = rng.standard_normal(n); x0r = Qr[:, 0] + 0.3 * pert / np.linalg.norm(pert)  # RQI needs a start near v1
    lr, _, hr = eg.rayleigh_quotient_iteration(A, x0r, tol=1e-12)
    print(f"   lambda2/lambda1 = {gap}: power {len(hp) - 1} its (lam {lp:.10f}); inverse shift 1.05 {len(hi) - 1} its; RQI from a 30% perturbation of v1: {len(hr) - 1} its (lam {lr:.10f})")

print("C2 Householder + shifted QR with deflation")
def wilkinson_qr_eigs(A, tol=1e-14):
    T, _ = eg.householder_tridiagonal(A)
    n = T.shape[0]; eigs = []; iters = []
    m = n
    while m > 1:
        k = 0
        while abs(T[m - 1, m - 2]) > tol * (abs(T[m - 1, m - 1]) + abs(T[m - 2, m - 2])):
            a, b, c = T[m - 2, m - 2], T[m - 1, m - 2], T[m - 1, m - 1]
            d = (a - c) / 2; mu = c - np.copysign(b * b, d if d != 0 else 1.0) / (abs(d) + np.hypot(d, b))
            Q, R = np.linalg.qr(T[:m, :m] - mu * np.eye(m)); T[:m, :m] = R @ Q + mu * np.eye(m); k += 1
        eigs.append(T[m - 1, m - 1]); iters.append(k); m -= 1
    eigs.append(T[0, 0])
    return np.sort(eigs), iters
for n in (10, 50, 100):
    S = rng.standard_normal((n, n)); S = (S + S.T) / 2
    e, iters = wilkinson_qr_eigs(S)
    print(f"   n={n}: max |eig - eigh| {np.abs(e - np.linalg.eigvalsh(S)).max():.1e}, total QR steps {sum(iters)}, mean per eigenvalue {np.mean(iters):.2f}")

print("C3 SVD via eig(A^T A) vs Golub-Kahan (np.linalg.svd)")
m, n = 60, 10
U0, _ = np.linalg.qr(rng.standard_normal((m, n))); V0, _ = np.linalg.qr(rng.standard_normal((n, n)))
sv = np.logspace(0, -10, n); A = (U0 * sv) @ V0.T
_, s_eig, _ = eg.svd_via_eig(A); s_svd = np.linalg.svd(A, compute_uv=False)
s_eig = np.pad(s_eig, (0, n - len(s_eig)))
for i in (0, 3, 5, 7, 9):
    print(f"   sigma_{i+1} = {sv[i]:.1e}: rel err eig(A^TA) {abs(s_eig[i] - sv[i]) / sv[i]:.1e}   svd {abs(s_svd[i] - sv[i]) / sv[i]:.1e}")

print("C4 sparse eigensolver for the 2-D Laplacian (m = 100, n = 10^4)")
m = 100; L = it.poisson_2d_sparse(m) * (m + 1)**2
t0 = time.perf_counter(); vals = spla.eigsh(L, k=6, sigma=0, which="LM", return_eigenvectors=False); t1 = time.perf_counter()
vals = np.sort(vals)
h = 1 / (m + 1)
exact = sorted((4 / h**2) * (np.sin(i * np.pi * h / 2)**2 + np.sin(j * np.pi * h / 2)**2) for i in range(1, 4) for j in range(1, 4))[:6]
print("   eigsh (shift-invert) smallest 6:", np.round(vals, 4), f" time {t1 - t0:.2f}s")
print("   exact:", np.round(exact, 4), " continuous pi^2 (i^2+j^2):", np.round(sorted(np.pi**2 * (i * i + j * j) for i in range(1, 4) for j in range(1, 4))[:6], 4))

# ---------- D ----------
print("D1 PCA of handwritten digits")
from sklearn.datasets import load_digits
X = load_digits().data.astype(float); Xc = X - X.mean(0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
var = s**2 / (len(X) - 1); ratio = np.cumsum(var) / var.sum()
print("   shape", X.shape, " top-5 explained ratio", np.round(var[:5] / var.sum(), 4), " k for 80/90/95%:", [int(np.searchsorted(ratio, t) + 1) for t in (0.8, 0.9, 0.95)])
for k in (2, 10, 20, 40):
    Xk = (U[:, :k] * s[:k]) @ Vt[:k]
    print(f"   k={k}: ||Xc - Xk||_F^2 = {np.sum((Xc - Xk)**2):.1f}  sum of discarded sigma^2 = {np.sum(s[k:]**2):.1f}")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
yd = load_digits().target
for k in (2, 10, 20, 64):
    Z = Xc @ Vt[:k].T if k < 64 else Xc
    acc = cross_val_score(LogisticRegression(max_iter=5000), Z, yd, cv=5).mean()
    print(f"   logistic regression on {k} PCs: CV accuracy {acc:.4f}")

print("D2 spectral clustering vs k-means (two moons, two circles)")
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist
for name, (Xs, ys) in (("moons", make_moons(400, noise=0.06, random_state=1)), ("circles", make_circles(400, noise=0.04, factor=0.4, random_state=1))):
    W = np.exp(-cdist(Xs, Xs)**2 / (2 * 0.1**2)); np.fill_diagonal(W, 0)
    d = W.sum(1); Lsym = np.eye(len(d)) - W / np.sqrt(np.outer(d, d))
    ev, V = np.linalg.eigh(Lsym)
    Y = V[:, :2] / np.linalg.norm(V[:, :2], axis=1, keepdims=True)
    lab_sp = KMeans(2, n_init=10, random_state=0).fit_predict(Y)
    lab_km = KMeans(2, n_init=10, random_state=0).fit_predict(Xs)
    print(f"   {name}: smallest eigenvalues of L_sym {np.round(ev[:4], 5)}  ARI spectral {adjusted_rand_score(ys, lab_sp):.3f}  k-means {adjusted_rand_score(ys, lab_km):.3f}")

print("D3 credit-rating Markov chain")
P = np.array([[0.90, 0.08, 0.02, 0.00],
              [0.05, 0.85, 0.08, 0.02],
              [0.01, 0.09, 0.80, 0.10],
              [0.00, 0.00, 0.10, 0.90]])
w, V = np.linalg.eig(P.T); i = np.argmin(abs(w - 1)); pi = np.real(V[:, i]); pi /= pi.sum()
lam = np.sort(np.abs(w))[::-1]
print("   eigenvalues", np.round(np.sort(np.real(w))[::-1], 6), " stationary", np.round(pi, 4), " |lambda2|", round(lam[1], 6))
x = np.array([1.0, 0, 0, 0]); tv = []
for k in range(1, 101):
    x = x @ P; tv.append(0.5 * np.abs(x - pi).sum())
print("   TV distance from state A after 10, 25, 50, 100 steps:", [f"{tv[k - 1]:.4f}" for k in (10, 25, 50, 100)], " steps to TV<0.01:", int(np.argmax(np.array(tv) < 0.01) + 1),
      " ln(0.01)/ln|lambda2| =", round(np.log(0.01) / np.log(lam[1]), 1))

print("D4 low-rank matrix completion by iterative SVD (soft-impute)")
rng = np.random.default_rng(5)
nu, ni, r = 300, 200, 5
M = rng.standard_normal((nu, r)) @ rng.standard_normal((r, ni)) + 0.1 * rng.standard_normal((nu, ni))
mask = rng.random((nu, ni)) < 0.2
test = (~mask) & (rng.random((nu, ni)) < 0.1)
Z = np.zeros_like(M)
for lam_ in (1.0, 5.0, 20.0):
    Z = np.zeros_like(M)
    for itn in range(200):
        Y = np.where(mask, M, Z)
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        s = np.maximum(s - lam_, 0); Z = (U * s) @ Vt
    rmse = np.sqrt(np.mean((Z[test] - M[test])**2)); rank = int(np.sum(s > 0))
    base = np.sqrt(np.mean((M[mask].mean() - M[test])**2))
    print(f"   observed {mask.mean():.1%}, lambda {lam_}: recovered rank {rank}, test RMSE {rmse:.4f} (predict-the-mean baseline {base:.4f}, noise level 0.1)")
