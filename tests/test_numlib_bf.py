"""Tests for the Burden & Faires additions to numlib."""

import math

import numpy as np
import pytest

from numlib import (approximation, bvp, eigen, integration, interpolation, linalg_direct,
                    nonlinear_systems, ode, pde, regression, roots)

rng = np.random.default_rng(7)


def test_aitken_steffensen():
    seq = [math.cos(1.0)]
    for _ in range(10):
        seq.append(math.cos(seq[-1]))
    acc = roots.aitken(seq)
    r = 0.7390851332151607
    assert abs(acc[-1] - r) < abs(seq[-1] - r)
    p, _ = roots.steffensen(math.cos, 1.0)
    assert abs(p - r) < 1e-12


def test_horner_muller():
    coef = [2, -3, 3, -4]  # 2x^3 - 3x^2 + 3x - 4
    p, dp, q = roots.horner_eval(coef, -2.0)
    assert p == 2 * -8 - 3 * 4 + 3 * -2 - 4 and dp == 6 * 4 - 6 * -2 + 3
    x, _ = roots.newton_horner(coef, 1.0)
    assert abs(np.polyval(coef, x)) < 1e-12
    z, _ = roots.muller(lambda x: x ** 2 + 1, 0.5, 1.0, 1.5)
    assert abs(abs(z.imag) - 1) < 1e-10 and abs(z.real) < 1e-10


def test_neville_hermite_clamped():
    x = np.array([1.0, 1.3, 1.6])
    v, Q = interpolation.neville(x, np.exp(x), 1.5)
    assert abs(v - interpolation.lagrange_eval(x, np.exp(x), 1.5)[0]) < 1e-14
    z, c = interpolation.hermite_coefficients([0.0, 1.0], [1.0, 2.0], [0.0, 3.0])
    t = np.linspace(0, 1, 5)
    assert np.allclose(interpolation.newton_eval(z, c, t), t ** 3 + 1)  # x^3+1 matches data
    xs = np.linspace(0, 2, 6)
    M = interpolation.clamped_cubic_spline(xs, xs ** 3, 0.0, 12.0)
    assert np.allclose(interpolation.cubic_spline_eval(xs, xs ** 3, M, t * 2), (t * 2) ** 3)


def test_bf_quadrature_extras():
    assert abs(integration.simpson38(lambda x: x ** 3, 0, 2) - 4) < 1e-14
    assert abs(integration.five_point_midpoint(np.exp, 0.0, 0.1) - 1) < 1e-5
    val = integration.simpson_double(lambda x, y: x * y, 0, 1, 0, lambda x: x, 4, 4)
    assert abs(val - 1 / 8) < 1e-12
    assert abs(integration.gauss_legendre_2d(lambda x, y: x ** 2 * y ** 2, 0, 1, 0, 1, 3) - 1 / 9) < 1e-14
    assert abs(integration.gauss_hermite_expectation(lambda x: x ** 2, 1.0, 2.0, 5) - 5.0) < 1e-12


def test_ode_extras():
    f = lambda t, y: y - t ** 2 + 1
    exact = lambda t: (t + 1) ** 2 - 0.5 * np.exp(t)
    t, Y, hs = ode.rkf45(f, (0, 2), 0.5, tol=1e-5, hmax=0.25, hmin=0.01)
    assert np.max(np.abs(Y[:, 0] - exact(t))) < 1e-4
    for method in (ode.adams_bashforth4, ode.adams_pc4):
        t, Y = method(f, (0, 2), 0.5, 0.2)
        assert abs(Y[-1, 0] - exact(2)) < 5e-3
    t, Y = ode.taylor2(f, lambda t, y: -2 * t, lambda t, y: 1.0, (0, 2), 0.5, 0.2)
    assert abs(Y[-1, 0] - exact(2)) < 0.05


def test_ldlt_scaled():
    M = rng.standard_normal((4, 4))
    A = M @ M.T + np.eye(4)
    L, d = linalg_direct.ldlt(A)
    assert np.allclose(L @ np.diag(d) @ L.T, A)
    B = rng.standard_normal((5, 5))
    b = rng.standard_normal(5)
    assert np.allclose(linalg_direct.gaussian_elimination_scaled(B, b), np.linalg.solve(B, b))
    x, _ = linalg_direct.iterative_refinement(B, b)
    assert np.allclose(x, np.linalg.solve(B, b), atol=1e-10)


def test_approximation():
    x = np.arange(1, 11, dtype=float)
    y = 2 + 3 * x
    assert np.allclose(approximation.linear_fit(x, y), (2, 3))
    a, E = approximation.poly_fit(x, 1 - x + x ** 2, 2)
    assert np.allclose(a, [1, -1, 1]) and E < 1e-15
    b, c = approximation.exp_fit(x, 3 * np.exp(0.2 * x))
    assert np.isclose(b, 3) and np.isclose(c, 0.2)
    phis = approximation.gram_schmidt_polys(3, -1, 1)
    assert np.allclose(phis[2].coeffs, [1, 0, -1 / 3])
    assert np.allclose(approximation.chebyshev_T(3).coeffs, [4, 0, -3, 0])
    p, q = approximation.pade([1, 1, 1 / 2, 1 / 6, 1 / 24, 1 / 120], 3, 2)
    assert np.allclose(q, [1, -2 / 5, 1 / 20])
    sig = rng.standard_normal(16)
    assert np.allclose(approximation.fft_radix2(sig), np.fft.fft(sig))
    xs = -np.pi + np.arange(8) * np.pi / 4
    a, bb = approximation.trig_ls_coeffs(np.cos(2 * xs) + 0.5, 3)
    assert np.allclose(a, [1, 0, 1, 0]) and np.allclose(bb, 0, atol=1e-14)


def test_eigen_extras():
    A = np.array([[4.0, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]])
    T, Q = eigen.householder_tridiagonal(A)
    assert np.allclose(np.triu(T, 2), 0, atol=1e-12)
    assert np.allclose(np.sort(np.linalg.eigvalsh(T)), np.sort(np.linalg.eigvalsh(A)))
    mu, v, _ = eigen.symmetric_power_method(A, np.ones(4))
    assert np.isclose(abs(mu), np.max(np.abs(np.linalg.eigvalsh(A))))
    B = np.array([[4.0, -1, 1], [-1, 3, -2], [1, -2, 3]])
    mu, v, _ = eigen.power_method_inf(B, [1, 0, 0], max_iter=500)
    assert np.isclose(mu, 6.0)
    D = eigen.wielandt_deflation(B, 6.0, v)
    assert np.allclose(np.sort(np.linalg.eigvals(D).real), [0, 1, 3], atol=1e-8)
    M = rng.standard_normal((5, 3))
    U, s, Vt = eigen.svd_via_eig(M)
    assert np.allclose(U * s @ Vt, M)


def test_nonlinear_systems():
    F = lambda v: np.array([3 * v[0] - np.cos(v[1] * v[2]) - 0.5,
                            v[0] ** 2 - 81 * (v[1] + 0.1) ** 2 + np.sin(v[2]) + 1.06,
                            np.exp(-v[0] * v[1]) + 20 * v[2] + (10 * np.pi - 3) / 3])
    x, _ = nonlinear_systems.broyden(F, [0.1, 0.1, -0.1])
    assert np.allclose(x, [0.5, 0, -np.pi / 6], atol=1e-8)
    J = lambda v: nonlinear_systems.jacobian_fd(F, v)
    x, _ = nonlinear_systems.homotopy(F, J, np.array([0.0, 0, 0]), N=8)
    assert np.allclose(x, [0.5, 0, -np.pi / 6], atol=1e-3)
    x, _ = nonlinear_systems.steepest_descent_system(F, J, [0.0, 0, 0], tol=1e-8)
    assert np.linalg.norm(F(x)) < 0.1


def test_bvp():
    # y'' = -2/x y' + 2/x^2 y + sin(ln x)/x^2, y(1)=1, y(2)=2 (B&F classic)
    c2 = (8 - 12 * np.sin(np.log(2)) - 4 * np.cos(np.log(2))) / 70
    c1 = 11 / 10 - c2
    exact = lambda x: c1 * x + c2 / x ** 2 - 3 / 10 * np.sin(np.log(x)) - 1 / 10 * np.cos(np.log(x))
    p = lambda x: -2 / x
    q = lambda x: 2 / x ** 2
    r = lambda x: np.sin(np.log(x)) / x ** 2
    x, w, _ = bvp.linear_shooting(p, q, r, 1, 2, 1, 2, 10)
    assert np.max(np.abs(w - exact(x))) < 1e-6
    x, w = bvp.linear_fd(p, q, r, 1, 2, 1, 2, 9)
    assert np.max(np.abs(w - exact(x))) < 1e-4
    f = lambda x, y, yp: (32 + 2 * x ** 3 - y * yp) / 8
    fy = lambda x, y, yp: -yp / 8
    fyp = lambda x, y, yp: -y / 8
    ex2 = lambda x: x ** 2 + 16 / x
    x, w, _ = bvp.nonlinear_shooting(f, fy, fyp, 1, 3, 17, 43 / 3, 20)
    assert np.max(np.abs(w - ex2(x))) < 1e-4
    x, w, _ = bvp.nonlinear_fd(f, fy, fyp, 1, 3, 17, 43 / 3, 19)
    assert np.max(np.abs(w - ex2(x))) < 0.05


def test_pde():
    x, y, W = pde.poisson_rect(lambda x, y: x * np.exp(y), lambda x, y: x * np.exp(y), 0, 2, 0, 1, 6, 5)
    X, Y = np.meshgrid(x, y, indexing="ij")
    assert np.max(np.abs(W - X * np.exp(Y))) < 1e-2
    u0 = lambda x: np.sin(np.pi * x)
    exact = np.exp(-np.pi ** 2 * 0.5) * np.sin(np.pi * np.linspace(0, 1, 11))
    _, w, _ = pde.heat_crank_nicolson(u0, 1, 1, 0.5, 10, 50)
    assert np.max(np.abs(w - exact)) < 1e-3
    _, w, lam = pde.heat_ftcs(u0, 1, 1, 0.5, 10, 1000)
    assert lam <= 0.5 and np.max(np.abs(w - exact)) < 1e-3
    _, w, _ = pde.heat_btcs(u0, 1, 1, 0.5, 10, 50)
    assert np.max(np.abs(w - exact)) < 5e-3
    _, w, lam = pde.wave_explicit(lambda x: np.sin(np.pi * x), lambda x: 0 * x, 2, 1, 1.0, 10, 20)
    assert np.max(np.abs(w - np.sin(np.pi * np.linspace(0, 1, 11)) * np.cos(2 * np.pi))) < 0.05


def test_regression():
    n = 60
    X = rng.standard_normal((n, 2))
    y = 1 + 2 * X[:, 0] - X[:, 1] + 0.3 * rng.standard_normal(n)
    res = regression.ols(X, y)
    A = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose(res.beta, beta)
    cov = res.sigma2 * np.linalg.inv(A.T @ A)
    assert np.allclose(res.se, np.sqrt(np.diag(cov)))
    assert np.isclose(res.leverage.sum(), 3)
    s = regression.simple_regression(X[:, 0], y)
    r1 = regression.ols(X[:, 0], y)
    assert np.isclose(s["b1"], r1.beta[1]) and np.isclose(s["se_b1"], r1.se[1])
    assert np.isclose(s["R2"], r1.r2)
    xs = np.linspace(0, 10, 30)
    coef, cond = regression.polynomial_regression(xs, 1 + xs - 0.5 * xs ** 2, 2)
    assert np.allclose(coef, [1, 1, -0.5]) and cond < 10
    t = np.linspace(0, 4, 25)
    yy = 2.5 * np.exp(-1.3 * t)
    r = lambda b: b[0] * np.exp(b[1] * t) - yy
    J = lambda b: np.column_stack([np.exp(b[1] * t), b[0] * t * np.exp(b[1] * t)])
    assert np.allclose(regression.gauss_newton(r, J, [2.0, -1.0])[0], [2.5, -1.3])
    assert np.allclose(regression.levenberg_marquardt(r, J, [1.0, 0.0])[0], [2.5, -1.3], atol=1e-6)
    assert np.allclose(regression.wls(X, y, np.ones(n)), beta)
