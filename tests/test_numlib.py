"""Correctness tests for numlib. Run with:  python -m pytest -q"""

import math

import numpy as np
import pytest
import scipy.linalg as sla

from numlib import (
    eigen,
    floating,
    integration,
    interpolation,
    least_squares,
    linalg_direct,
    linalg_iterative,
    ode,
    optimization,
    roots,
)

rng = np.random.default_rng(42)


# ---------------- Chapter 1 ----------------
def test_machine_epsilon():
    assert floating.machine_epsilon() == np.finfo(float).eps
    assert floating.machine_epsilon(np.float32) == np.finfo(np.float32).eps


def test_kahan_beats_naive():
    xs = [0.1] * 1_000_000
    exact = math.fsum(xs)
    assert abs(floating.kahan_sum(xs) - exact) < abs(floating.naive_sum(xs) - exact)


def test_stable_quadratic():
    x1, x2 = floating.quadratic_roots_stable(1.0, 1e8, 1.0)
    assert abs(x2 - (-1e-8)) / 1e-8 < 1e-12


def test_logsumexp():
    assert np.isclose(floating.logsumexp([1000, 1000]), 1000 + np.log(2))
    assert np.isclose(floating.softmax([1000, 1000]).sum(), 1.0)


def test_variance_algorithms():
    xs = 1e9 + rng.standard_normal(10_000)
    ref = np.var(xs, ddof=1)
    assert abs(floating.variance_welford(xs) - ref) / ref < 1e-6
    assert abs(floating.variance_two_pass(xs) - ref) / ref < 1e-6


# ---------------- Chapter 2 ----------------
f = lambda x: x ** 3 - 2 * x - 5
df = lambda x: 3 * x ** 2 - 2
ROOT = 2.0945514815423265


def test_bisection():
    x, h = roots.bisection(f, 2, 3, tol=1e-12)
    assert abs(x - ROOT) < 1e-11
    assert len(h) <= roots.bisection_iterations(2, 3, 1e-12) + 1


def test_newton_and_secant():
    assert abs(roots.newton(f, df, 2.0)[0] - ROOT) < 1e-14
    assert abs(roots.secant(f, 2.0, 3.0)[0] - ROOT) < 1e-12
    assert abs(roots.regula_falsi(f, 2.0, 3.0)[0] - ROOT) < 1e-10
    assert abs(roots.safeguarded_newton(f, df, 2.0, 3.0)[0] - ROOT) < 1e-12


def test_fixed_point():
    x, _ = roots.fixed_point(math.cos, 1.0)
    assert abs(x - 0.7390851332151607) < 1e-9


def test_newton_system():
    F = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 4, v[0] * v[1] - 1])
    J = lambda v: np.array([[2 * v[0], 2 * v[1]], [v[1], v[0]]])
    x, _ = roots.newton_system(F, J, [2.0, 0.5])
    assert np.linalg.norm(F(x)) < 1e-12


# ---------------- Chapter 3 ----------------
def test_lu_and_solves():
    A = rng.standard_normal((6, 6))
    b = rng.standard_normal(6)
    P, L, U = linalg_direct.lu_partial_pivot(A)
    assert np.allclose(P @ A, L @ U)
    assert np.allclose(linalg_direct.lu_solve(P, L, U, b), np.linalg.solve(A, b))
    assert np.allclose(linalg_direct.gaussian_elimination(A, b), np.linalg.solve(A, b))
    assert np.isclose(linalg_direct.determinant_via_lu(A), np.linalg.det(A))
    assert np.allclose(linalg_direct.inverse_via_lu(A), np.linalg.inv(A))


def test_cholesky():
    M = rng.standard_normal((5, 5))
    A = M @ M.T + 5 * np.eye(5)
    L = linalg_direct.cholesky(A)
    assert np.allclose(L, np.linalg.cholesky(A))
    with pytest.raises(np.linalg.LinAlgError):
        linalg_direct.cholesky(-A)


def test_thomas():
    n = 8
    a, c = -np.ones(n - 1), -np.ones(n - 1)
    b = 2 * np.ones(n)
    d = rng.standard_normal(n)
    A = np.diag(b) + np.diag(a, -1) + np.diag(c, 1)
    assert np.allclose(linalg_direct.tridiagonal_solve(a, b, c, d), np.linalg.solve(A, d))


# ---------------- Chapter 4 ----------------
def test_iterative_solvers():
    n = 20
    A = linalg_iterative.poisson_1d(n) + 0.5 * np.eye(n)
    b = np.ones(n)
    ref = np.linalg.solve(A, b)
    for solver in (linalg_iterative.jacobi, linalg_iterative.gauss_seidel,
                   linalg_iterative.conjugate_gradient,
                   linalg_iterative.steepest_descent):
        x, _ = solver(A, b, tol=1e-12)
        assert np.allclose(x, ref, atol=1e-9), solver.__name__
    x, res = linalg_iterative.conjugate_gradient(linalg_iterative.poisson_1d(n), b, tol=1e-12)
    assert len(res) - 1 <= n + 1  # CG terminates in <= n steps (exact arithmetic)


def test_sor_omega():
    n = 30
    A = linalg_iterative.poisson_1d(n)
    rho = linalg_iterative.spectral_radius(linalg_iterative.iteration_matrix(A))
    assert np.isclose(rho, np.cos(np.pi / (n + 1)))
    w = linalg_iterative.optimal_sor_omega(rho)
    rho_sor = linalg_iterative.spectral_radius(linalg_iterative.iteration_matrix(A, "sor", w))
    assert rho_sor < rho ** 2


# ---------------- Chapter 5 ----------------
def test_least_squares():
    A = rng.standard_normal((30, 4))
    b = rng.standard_normal(30)
    ref = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(least_squares.normal_equations(A, b), ref)
    assert np.allclose(least_squares.qr_least_squares(A, b), ref)
    assert np.allclose(least_squares.svd_least_squares(A, b), ref)
    Q, R = least_squares.householder_qr(A)
    assert np.allclose(Q @ R, A) and np.allclose(Q.T @ Q, np.eye(30))
    Q, R = least_squares.gram_schmidt(A)
    assert np.allclose(Q @ R, A)
    lam = 0.7
    ridge_ref = np.linalg.solve(A.T @ A + lam * np.eye(4), A.T @ b)
    assert np.allclose(least_squares.ridge(A, b, lam), ridge_ref)


# ---------------- Chapter 6 ----------------
def test_eigen():
    M = rng.standard_normal((6, 6))
    A = M + M.T
    w = np.linalg.eigvalsh(A)
    lam, v, _ = eigen.power_iteration(A @ A)  # positive dominant
    assert np.isclose(lam, np.max(w ** 2), rtol=1e-6)
    lam, v, _ = eigen.inverse_iteration(A, shift=w[2] + 0.01)
    assert np.isclose(lam, w[2])
    assert np.allclose(eigen.qr_algorithm(A), np.sort(w)[::-1], atol=1e-8)


def test_pagerank():
    adj = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    r = eigen.pagerank(adj)
    assert np.isclose(r.sum(), 1.0)
    assert np.argmax(r) == 2


def test_pca():
    X = rng.standard_normal((200, 3)) @ np.diag([5, 1, 0.1])
    scores, comps, ratio = eigen.pca(X, 2)
    assert ratio[0] > 0.9 and scores.shape == (200, 2)


# ---------------- Chapter 7 ----------------
def test_interpolation():
    x = np.array([0.0, 1.0, 2.0, 4.0])
    y = x ** 3 - x + 1
    t = np.linspace(0, 4, 17)
    exact = t ** 3 - t + 1
    assert np.allclose(interpolation.lagrange_eval(x, y, t), exact)
    assert np.allclose(interpolation.barycentric_eval(x, y, t), exact)
    c = interpolation.divided_differences(x, y)
    assert np.allclose(interpolation.newton_eval(x, c, t), exact)
    assert np.allclose(interpolation.horner([1, -1, 0, 1], t), exact)


def test_spline_matches_scipy():
    from scipy.interpolate import CubicSpline

    x = np.linspace(0, 2 * np.pi, 9)
    y = np.sin(x)
    M = interpolation.natural_cubic_spline(x, y)
    t = np.linspace(0, 2 * np.pi, 50)
    ref = CubicSpline(x, y, bc_type="natural")(t)
    assert np.allclose(interpolation.cubic_spline_eval(x, y, M, t), ref)


# ---------------- Chapter 8 ----------------
def test_quadrature():
    exact = 2.0  # int_0^pi sin
    assert abs(integration.trapezoid(np.sin, 0, np.pi, 100) - exact) < 2e-4
    assert abs(integration.simpson(np.sin, 0, np.pi, 100) - exact) < 1e-7
    assert abs(integration.romberg(np.sin, 0, np.pi, 6)[0] - exact) < 1e-10
    assert abs(integration.gauss_legendre(np.sin, 0, np.pi, 10) - exact) < 1e-12
    assert abs(integration.adaptive_simpson(np.sin, 0, np.pi)[0] - exact) < 1e-9
    est, se = integration.monte_carlo(np.sin, 0, np.pi, 100_000, np.random.default_rng(1))
    assert abs(est - exact) < 5 * se


def test_derivatives():
    assert abs(integration.central_diff(np.exp, 1.0) - np.e) < 1e-9
    assert abs(integration.complex_step(np.exp, 1.0) - np.e) < 1e-15
    val, _ = integration.richardson(lambda h: integration.central_diff(np.exp, 1.0, h), 0.1)
    assert abs(val - np.e) < 1e-12


# ---------------- Chapter 9 ----------------
def test_optimizers():
    x, _ = optimization.newton_opt(optimization.rosenbrock_grad, optimization.rosenbrock_hess, [-1.2, 1.0])
    assert np.allclose(x, [1, 1])
    x, _ = optimization.bfgs(optimization.rosenbrock, optimization.rosenbrock_grad, [-1.2, 1.0])
    assert np.allclose(x, [1, 1], atol=1e-6)
    x, _ = optimization.golden_section(lambda t: (t - 2) ** 2, 0, 5)
    assert abs(x - 2) < 1e-6


def test_logistic_gradient():
    X = rng.standard_normal((50, 3))
    y = (rng.random(50) < 0.5).astype(float)
    w = rng.standard_normal(3)
    g_fd = integration.gradient_fd(lambda v: optimization.logistic_loss(v, X, y, 0.1), w)
    assert np.allclose(g_fd, optimization.logistic_grad(w, X, y, 0.1), atol=1e-7)


# ---------------- Chapter 10 ----------------
def test_ode_orders():
    f = lambda t, y: -2 * t * y
    exact = np.exp(-1.0)
    for method, order in ((ode.euler, 1), (ode.heun, 2), (ode.rk4, 4)):
        errs = [abs(method(f, (0, 1), 1.0, h)[1][-1, 0] - exact) for h in (0.02, 0.01)]
        p = np.log2(errs[0] / errs[1])
        assert abs(p - order) < 0.2, method.__name__


def test_rk45_and_implicit():
    f = lambda t, y: -2 * t * y
    t, Y, _ = ode.rk45(f, (0, 1), 1.0, rtol=1e-10, atol=1e-12)
    assert abs(Y[-1, 0] - np.exp(-1)) < 1e-9
    stiff = lambda t, y: -1000 * (y - np.cos(t))
    t, Y = ode.implicit_euler(stiff, (0, 1), 0.0, 0.1)
    assert np.all(np.isfinite(Y)) and abs(Y[-1, 0] - np.cos(1)) < 0.01
