# Chapter 3 — Worked Examples: Interpolation and Polynomial Approximation

Companion script: [`ch03_examples.py`](ch03_examples.py) reproduces every number below. · Lecture: [Chapter 3](../lectures/ch03-interpolation.md)

---

## Example 3.1 — Lagrange interpolation of a yield curve

**Problem.** Government bond yields are 3.10% (1 year), 3.50% (2 years) and 4.00% (5 years). Estimate the 3-year yield with the quadratic interpolant, and compare it with linear interpolation.

**Solution.** The Lagrange basis polynomials at $t = 3$ are
$$
L_0(3) = \frac{(3-2)(3-5)}{(1-2)(1-5)} = -\tfrac12,\qquad L_1(3) = \frac{(3-1)(3-5)}{(2-1)(2-5)} = \tfrac43,\qquad L_2(3) = \frac{(3-1)(3-2)}{(5-1)(5-2)} = \tfrac16 .
$$
They sum to 1, as they must, because interpolation reproduces constants. So
$P_2(3) = -\tfrac12(3.10)+\tfrac43(3.50)+\tfrac16(4.00) = \mathbf{3.7833\%}$.
Linear interpolation between the 2-year and 5-year yields gives $3.6667\%$.

**Take-away.** The quadratic captures the concavity of the curve, since yields rise quickly at the short end. A negative basis weight ($L_0<0$) is normal for points outside the nearest interval. Practitioners use splines or Nelson–Siegel curves to avoid wiggles when there are many maturities.

---

## Example 3.2 — Inverse interpolation: solving an equation from a table

**Problem.** The function $g(x) = \cos x-x$ is tabulated at $x = 0.70, 0.72, 0.74, 0.76$. Find its root *without* an iterative solver.

**Solution.** Swap the roles of the variables: interpolate $x$ as a function of $g$, and evaluate at $g = 0$ with Neville's algorithm.

| $g_i$ | $x_i$ | $Q_{i,1}$ | $Q_{i,2}$ | $Q_{i,3}$ |
|---|---|---|---|---|
| 0.064842 | 0.70 | | | |
| 0.031806 | 0.72 | 0.739255 | | |
| −0.001531 | 0.74 | 0.739081 | 0.739085 | |
| −0.035164 | 0.76 | 0.739089 | 0.739085 | **0.7390851348** |

The true root is $0.7390851332$, so the error is $1.6\times10^{-9}$. The table converges quickly because $g$ is monotone here, so its inverse is smooth.

**Take-away.** Inverse interpolation is how root finding is done on *tabulated* data, e.g. finding the break-even price from simulated profit values.

---

## Example 3.3 — Interpolation vs extrapolation: US population

**Problem.** The US census gives (millions) 151.3, 179.3, 203.3, 226.5, 248.7 and 281.4 for 1950–2000. Use the degree-5 interpolating polynomial to estimate the population in 1965 and 1995, and extrapolate to 2010 and 2020.

**Solution.** In decades since 1950, the Newton divided-difference coefficients are $151.3, 28.0, -2.0, 0.5333, -0.1417, 0.1258$.

| year | degree-5 interpolant | least-squares line | actual |
|---|---|---|---|
| 1965 | 191.3 | 189.9 | (194.3) |
| 1995 | 262.1 | 265.5 | (266.3) |
| 2010 | **362.9** | 303.3 | 308.7 |
| 2020 | **573.4** | 328.5 | 331.4 |

**Take-away.** Inside the data range the interpolant is reasonable. Outside it, the factor $\prod(x-x_i)$ in the error formula explodes, and the polynomial "predicts" 573 million people in 2020. Never extrapolate high-degree interpolants. A low-complexity model fitted by least squares (Chapter 8) is far more reliable, though it is also no substitute for a demographic model.

---

## Example 3.4 — Hermite interpolation of a trajectory

**Problem.** A vehicle is at position $0$ with speed $2$ at $t = 0$, and at position $10$ with speed $8$ at $t = 1$. Find the cubic that matches both positions and speeds, and the position at $t = 0.5$.

**Solution.** The doubled nodes are $z = (0,0,1,1)$. The divided-difference table uses $f[z_0,z_1] = f'(0) = 2$ and $f[z_2,z_3] = f'(1) = 8$:
$f[z_1,z_2] = 10$, $f[z_0,z_1,z_2] = 8$, $f[z_1,z_2,z_3] = -2$ and $f[z_0,\dots,z_3] = -10$. Then
$$H_3(t) = 2t+8t^2-10t^2(t-1) = -10t^3+18t^2+2t,\qquad H_3(0.5) = 4.25 .$$
Check: $H_3(1) = 10$ and $H_3'(1) = -30+36+2 = 8$ ✓.

**Take-away.** Hermite interpolation uses derivative information (velocities, marginal costs) as well as values. It is the building block of monotone cubic interpolation (PCHIP) and of dense output in ODE solvers.

---

## Example 3.5 — Splines vs polynomials for a dose–response curve

**Problem.** Response rates at doses $0,\dots,8$ are $0, 0.01, 0.02, 0.10, 0.90, 0.98, 0.99, 1, 1$, a steep sigmoid. Compare the degree-8 interpolating polynomial, the natural cubic spline and a monotone PCHIP interpolant.

**Solution.**

| interpolant | range on $[0,8]$ | value at dose 7.5 | monotone? |
|---|---|---|---|
| degree-8 polynomial | $[-0.894, 1.145]$ | 0.454 | no (wild) |
| natural cubic spline | $[-0.018, 1.018]$ | 0.9998 | no (slight overshoot) |
| PCHIP | $[0, 1]$ | 1.000 | yes |

The spline's second derivatives $M_i$ are $0, 0.075, -0.299, 1.540, -1.540, 0.299, -0.076, 0.004, 0$. The large values sit at the steep part, and the alternating signs cause small overshoots near the plateaus.

**Take-away.** A polynomial through many points is useless for such data. A cubic spline is smooth and local but can overshoot slightly. When the shape must be respected (probabilities, CDFs, monotone dose–response), use a shape-preserving interpolant such as PCHIP.

---

## Example 3.6 — Chebyshev nodes rescue high-degree interpolation

**Problem.** Interpolate the steep logistic function $f(x) = \frac{1}{1+e^{-10x}}$ on $[-1,1]$ at equispaced and at Chebyshev nodes (barycentric formula), for degrees 6–24.

**Solution.**

| degree | equispaced max error | Chebyshev max error |
|---|---|---|
| 6 | $8.2\times10^{-2}$ | $1.1\times10^{-1}$ |
| 10 | $1.3\times10^{-1}$ | $3.6\times10^{-2}$ |
| 16 | $3.9\times10^{-1}$ | $6.1\times10^{-3}$ |
| 24 | $\mathbf{2.0}$ | $\mathbf{5.4\times10^{-4}}$ |

**Take-away.** $f$ has complex poles at $\pm i\pi/10$, close to the interval. With equispaced nodes the error *diverges* (Runge phenomenon). Chebyshev nodes minimise $\max|\prod(x-x_i)|$, and the error decreases geometrically. The barycentric formula makes high-degree evaluation stable.
