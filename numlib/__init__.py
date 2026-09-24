"""numlib — reference implementations for the Numerical Analysis course.

Every algorithm taught in the lectures has a short, readable implementation
here. The goal is clarity, not speed: in real work you should call NumPy /
SciPy, and each module's docstring names the library routine to use instead.

Modules
-------
floating         machine epsilon, error measures, stable summation
roots            bisection, fixed point, Newton, secant, Brent-style hybrid
linalg_direct    Gaussian elimination, LU with pivoting, Cholesky, triangular solves
linalg_iterative Jacobi, Gauss-Seidel, SOR, conjugate gradient
least_squares    normal equations, Householder QR, SVD solutions, ridge
eigen            power / inverse / Rayleigh iteration, QR algorithm, PCA, PageRank
interpolation    Vandermonde, Lagrange, Newton divided differences, cubic splines
integration      finite differences, Newton-Cotes, Romberg, Gauss-Legendre, Monte Carlo
optimization     line search, gradient descent, Newton, BFGS, SGD, Adam
ode              Euler, Heun, RK4, adaptive RK45, implicit Euler, systems
"""

from . import (
    floating,
    roots,
    linalg_direct,
    linalg_iterative,
    least_squares,
    eigen,
    interpolation,
    integration,
    optimization,
    ode,
    approximation,
    nonlinear_systems,
    bvp,
    pde,
    regression,
)

__all__ = [
    "floating",
    "roots",
    "linalg_direct",
    "linalg_iterative",
    "least_squares",
    "eigen",
    "interpolation",
    "integration",
    "optimization",
    "ode",
    "approximation",
    "nonlinear_systems",
    "bvp",
    "pde",
    "regression",
]
