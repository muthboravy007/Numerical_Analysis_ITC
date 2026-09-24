# Lecture 7 — Interpolation and Approximation

> **Week 10** · Code: [`numlib/interpolation.py`](../numlib/interpolation.py) · Examples: [`examples/ch07`](../examples/ch07_examples.md) · Exercises: [`exercises/ch07`](../exercises/ch07_exercises.md)

## Learning objectives

1. Construct the interpolating polynomial in monomial, Lagrange, barycentric and Newton forms and compare their costs.
2. Build divided-difference tables.
3. Apply the interpolation error theorem and explain Runge's phenomenon.
4. Use Chebyshev nodes to control interpolation error.
5. Construct piecewise-linear and cubic spline interpolants and explain why splines are preferred for data.
6. Distinguish interpolation (passing through data) from approximation/regression (fitting noisy data).

---

## 7.1 Motivation

* **Filling gaps** in time series (missing sensor readings, resampling irregular timestamps to a regular grid — `pandas.Series.interpolate`).
* **Image resizing**: bilinear and bicubic interpolation.
* **Smooth curves** for calibration tables, yield curves, dose–response.
* **Surrogate models**: replace an expensive simulation with a cheap interpolant.
* **Foundations** for numerical differentiation, quadrature and ODE solvers (Lectures 8, 10) — they all integrate or differentiate an interpolating polynomial.

**Problem.** Given $n+1$ distinct nodes $x_0,\dots,x_n$ and values $y_i = f(x_i)$, find $p \in \mathcal P_n$ (polynomials of degree $\le n$) with $p(x_i) = y_i$.

**Theorem (existence and uniqueness).** Such a $p$ exists and is unique.
*Proof of uniqueness:* if $p, q$ both interpolate, $p-q\in\mathcal P_n$ has $n+1$ roots, hence is zero.

---

## 7.2 Forms of the interpolating polynomial

### Monomial basis (Vandermonde)
$p(x) = \sum c_kx^k$ leads to $V\mathbf c = \mathbf y$ with $V_{ik} = x_i^k$. Simple, but $V$ is **exponentially ill-conditioned** (for 20 equispaced points in $[0,1]$, $\kappa(V)\approx 10^{16}$). Avoid for $n\gtrsim 10$.

### Lagrange form
$$
p(x) = \sum_{k=0}^n y_k L_k(x),\qquad L_k(x) = \prod_{j\ne k}\frac{x - x_j}{x_k - x_j},\qquad L_k(x_i) = \delta_{ik}.
$$
No linear system to solve; $O(n^2)$ per evaluation point.

### Barycentric form (the one to use)
With weights $w_k = 1/\prod_{j\ne k}(x_k-x_j)$ (computed once in $O(n^2)$),
$$
p(x) = \frac{\displaystyle\sum_{k}\frac{w_k}{x - x_k}\,y_k}{\displaystyle\sum_{k}\frac{w_k}{x-x_k}}.
$$
$O(n)$ per evaluation, numerically stable, and adding a node costs $O(n)$. This is `scipy.interpolate.BarycentricInterpolator`.

### Newton form and divided differences
$$
p(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + \cdots
$$
with **divided differences** defined recursively:
$$
f[x_i] = y_i,\qquad f[x_i,\dots,x_{i+k}] = \frac{f[x_{i+1},\dots,x_{i+k}] - f[x_i,\dots,x_{i+k-1}]}{x_{i+k} - x_i}.
$$
Evaluate with a Horner-like nested scheme in $O(n)$. Adding a data point just appends a term.

**Example.** $x = 0,1,2,3$, $y = 1,2,9,28$ ($=x^3+1$):

| $x_i$ | $f[\cdot]$ | 1st | 2nd | 3rd |
|---|---|---|---|---|
| 0 | 1 | | | |
| | | 1 | | |
| 1 | 2 | | 3 | |
| | | 7 | | 1 |
| 2 | 9 | | 6 | |
| | | 19 | | |
| 3 | 28 | | | |

$p(x) = 1 + x + 3x(x-1) + x(x-1)(x-2) = x^3 + 1$. ✓

---

## 7.3 Interpolation error

**Theorem.** If $f\in C^{n+1}[a,b]$ and $p\in\mathcal P_n$ interpolates $f$ at $x_0,\dots,x_n\in[a,b]$, then for each $x\in[a,b]$ there is $\xi\in(a,b)$ with
$$
f(x) - p(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}\,\omega(x),\qquad \omega(x) = \prod_{i=0}^n(x-x_i).
$$

Two factors matter: the smoothness of $f$ (the derivative) and the placement of nodes (the **nodal polynomial** $\omega$).

### Runge's phenomenon
For $f(x) = \dfrac{1}{1+25x^2}$ on $[-1,1]$ with **equispaced** nodes, $\max|f-p_n|$ **grows** with $n$:

| $n$ | equispaced max error | Chebyshev max error |
|---|---|---|
| 5 | 0.43 | 0.56 |
| 10 | 1.92 | 0.11 |
| 15 | 2.11 | 0.083 |
| 20 | 59.8 | 0.015 |

The oscillations appear near the ends of the interval, where $|\omega(x)|$ is largest for equispaced nodes.

### Chebyshev nodes
$$
x_k = \cos\Bigl(\frac{(2k+1)\pi}{2(n+1)}\Bigr),\qquad k = 0,\dots,n,
$$
(the roots of the Chebyshev polynomial $T_{n+1}$) minimise $\max_{[-1,1]}|\omega(x)|$, achieving $2^{-n}$. For any $f$ analytic near $[-1,1]$, Chebyshev interpolation converges **exponentially**. Map to $[a,b]$ by $x\mapsto\frac{a+b}{2}+\frac{b-a}{2}x$. This is the foundation of the `chebfun` system and of `numpy.polynomial.chebyshev`.

**Lesson for data science.** High-degree polynomial features on equispaced (or uniformly sampled) data extrapolate and oscillate wildly — the same phenomenon. Prefer splines, regularisation or orthogonal bases.

---

## 7.4 Piecewise interpolation and splines

When you can't choose the nodes (you're given data), use **low-degree pieces**.

### Piecewise linear
Connect the dots. Error $\le \frac{h^2}{8}\max|f''|$ where $h$ is the max spacing. Continuous but has kinks. (`np.interp`)

### Cubic splines
A cubic spline $S$ is a cubic polynomial on each $[x_i,x_{i+1}]$ with $S, S', S''$ continuous. Counting: $4n$ coefficients; $2n$ interpolation conditions $+\,2(n-1)$ continuity conditions for $S', S''$ $= 4n-2$. Two more conditions are needed:

| End condition | Extra equations | Note |
|---|---|---|
| **Natural** | $S''(x_0)=S''(x_n)=0$ | Minimises $\int (S'')^2$ ("bending energy") |
| **Clamped** | $S'(x_0), S'(x_n)$ given | Best accuracy if slopes known |
| **Not-a-knot** | $S'''$ continuous at $x_1, x_{n-1}$ | SciPy's default |

**Derivation (natural spline).** Let $M_i = S''(x_i)$ and $h_i = x_{i+1}-x_i$. On each interval $S''$ is linear; integrating twice and imposing $S(x_i)=y_i$, $S(x_{i+1})=y_{i+1}$ gives
$$
S(x) = M_i\frac{(x_{i+1}-x)^3}{6h_i} + M_{i+1}\frac{(x-x_i)^3}{6h_i} + \Bigl(\frac{y_i}{h_i}-\frac{M_ih_i}{6}\Bigr)(x_{i+1}-x) + \Bigl(\frac{y_{i+1}}{h_i}-\frac{M_{i+1}h_i}{6}\Bigr)(x-x_i).
$$
Continuity of $S'$ at interior nodes yields the **tridiagonal** system
$$
h_{i-1}M_{i-1} + 2(h_{i-1}+h_i)M_i + h_iM_{i+1} = 6\Bigl(\frac{y_{i+1}-y_i}{h_i} - \frac{y_i-y_{i-1}}{h_{i-1}}\Bigr),\quad i=1,\dots,n-1,
$$
with $M_0 = M_n = 0$. It is strictly diagonally dominant → solve in $O(n)$ with the Thomas algorithm (Lecture 3).

**Error.** For clamped splines, $\max|f - S| \le \frac{5}{384}h^4\max|f^{(4)}|$ — fourth-order accurate.

### Shape-preserving alternatives
Cubic splines may overshoot monotone data (e.g. cumulative counts). **PCHIP** (`scipy.interpolate.PchipInterpolator`) preserves monotonicity at the cost of only $C^1$ smoothness.

---

## 7.5 Interpolation vs. approximation

Interpolation passes *exactly* through the data — appropriate for exact function values, **not** for noisy measurements. For noisy data use least squares (Lecture 5): regression splines, smoothing splines (`scipy.interpolate.make_smoothing_spline`), or penalised models. This is the bias–variance trade-off in numerical clothing.

## 7.6 Multivariate interpolation (brief)

* Regular grids: tensor-product (bilinear/bicubic) — `scipy.interpolate.RegularGridInterpolator`.
* Scattered data: triangulation-based linear (`griddata`), radial basis functions (`RBFInterpolator`), Gaussian processes (kriging — Lecture 3's Cholesky at work).

## 7.7 Summary

* The interpolating polynomial is unique; use barycentric or Newton forms, never Vandermonde for large $n$.
* Error $=\frac{f^{(n+1)}(\xi)}{(n+1)!}\omega(x)$; equispaced high-degree interpolation can diverge (Runge). Chebyshev nodes fix it.
* For data, use cubic splines ($O(h^4)$, tridiagonal solve) or PCHIP; for noisy data, approximate rather than interpolate.

## Further reading

* L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM, 2013 (Chapters 1–5, 13–15).
* Berrut & Trefethen, *Barycentric Lagrange Interpolation*, SIAM Review, 2004.
* C. de Boor, *A Practical Guide to Splines*, 2001.
