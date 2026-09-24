# Chapter 4 — Worked Examples: Numerical Differentiation and Integration

Companion script: [`ch04_examples.py`](ch04_examples.py) reproduces every number below. · Lecture: [Chapter 4](../lectures/ch04-differentiation-integration.md)

---

## Example 4.1 — Acceleration from sensor data

**Problem.** A car's speed (m/s), recorded each second, is $0, 3.1, 6.4, 9.6, 12.5, 15.1, 17.2, 18.9, 20.1$. Estimate the acceleration at every time.

**Solution.** At interior points use the central difference $a(t_i)\approx\frac{v_{i+1}-v_{i-1}}{2h}$ ($O(h^2)$). At the ends use the 3-point one-sided formulas $\frac{-3v_0+4v_1-v_2}{2h}$ and $\frac{3v_8-4v_7+v_6}{2h}$, which are also $O(h^2)$.

| $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| $a$ (m/s²) | 3.00 | 3.20 | 3.25 | 3.05 | 2.75 | 2.35 | 1.90 | 1.45 | 0.95 |

The 5-point formula at $t = 4$ gives $\frac{v_2-8v_3+8v_5-v_6}{12} = 2.767$, close to the central value $2.75$.

**Take-away.** On measured data you cannot shrink $h$, and differentiation amplifies measurement noise by a factor of about $1/h$. Smooth first (spline or Whittaker smoother) if the data are noisy.

---

## Example 4.2 — Distance travelled

**Problem.** Integrate the same speeds over $[0,8]$ to get the distance.

**Solution.** The trapezoidal rule gives $92.85$ m. Composite Simpson ($n = 8$, even) gives $93.03$ m. They differ by $0.18$ m, which is roughly the trapezoidal error, since Simpson is $O(h^4)$.

**Take-away.** The same data give different answers under different rules. The difference between two methods of different order is a practical error estimate, and it is the idea behind Richardson extrapolation and adaptive quadrature.

---

## Example 4.3 — Romberg integration for the normal CDF

**Problem.** Compute $\Phi(1.5) = \frac12+\frac1{\sqrt{2\pi}}\int_0^{1.5}e^{-t^2/2}dt$ with a 5-level Romberg table.

**Solution.** The first column contains trapezoidal values with $1, 2, 4, 8, 16$ intervals ($0.396345, 0.424026, 0.430912, 0.432623, 0.433050$). Each extrapolation column removes one more even power of $h$:

$$
\begin{array}{lllll}
0.396345\\
0.424026&0.433252\\
0.430912&0.433208&0.433205\\
0.432623&0.433194&0.433193&0.433193\\
0.433050&0.433193&0.433193&0.433193&0.433193
\end{array}
$$

$\frac12+R_{4,4} = 0.9331927991$, against the exact $0.9331927987$: an error of $3.4\times10^{-10}$ from only 17 function evaluations.

**Take-away.** For smooth integrands, Romberg turns the humble trapezoidal rule into a very high-order method.

---

## Example 4.4 — Gaussian quadrature for an expectation

**Problem.** Let $X\sim U(0,2)$. Compute $E[\ln(1+X)] = \frac12\int_0^2\ln(1+x)\,dx = \frac{3\ln3-2}{2} = 0.6479184330$.

**Solution.**

| $n$ (nodes) | Gauss–Legendre error | Simpson with a similar number of points |
|---|---|---|
| 1 | $4.5\times10^{-2}$ | — |
| 2 | $1.7\times10^{-3}$ | $2.7\times10^{-3}$ (3 points) |
| 3 | $8.5\times10^{-5}$ | $2.6\times10^{-4}$ (5 points) |
| 4 | $4.6\times10^{-6}$ | $2.6\times10^{-4}$ (5 points) |
| 5 | $2.7\times10^{-7}$ | $5.8\times10^{-5}$ (7 points) |

**Take-away.** With $n$ nodes, Gauss is exact for polynomials of degree $2n-1$, so each extra node gains about 1.3 digits here. Expectations of smooth functions of random variables are ideal for Gaussian rules: Legendre for uniform, Hermite for normal, Laguerre for exponential.

---

## Example 4.5 — Adaptive quadrature for a sharp peak

**Problem.** $\int_0^1\frac{dx}{10^{-4}+(x-0.3)^2}$ has a peak of height $10^4$ and width $0.01$ at $x = 0.3$. The exact value is $\frac{\arctan70+\arctan30}{0.01} = 309.3986915$.

**Solution.**

| method | evaluations | error |
|---|---|---|
| composite Simpson, $n = 100$ | 101 | 7.9 |
| composite Simpson, $n = 1000$ | 1001 | $6\times10^{-11}$ |
| adaptive Simpson, tol $10^{-4}$ | 477 | $7.4\times10^{-7}$ |
| adaptive Simpson, tol $10^{-8}$ | 4593 | $1.4\times10^{-12}$ |

**Take-away.** A uniform grid fails completely until $h$ is smaller than the peak width. Beyond that point it converges extremely fast here, because the integrand is analytic. Adaptive Simpson finds the peak automatically and concentrates the nodes there, which matters most when you do *not* know where the difficulty is. Its error control is conservative: the actual errors are far below the tolerances.

---

## Example 4.6 — A two-dimensional probability

**Problem.** For a standard bivariate normal with correlation $\rho = 0.5$, compute $P(|X|\le1,\ |Y|\le1)$.

**Solution.** Integrate the density over $[-1,1]^2$.

| method | value |
|---|---|
| Gauss–Legendre $2\times2$ | 0.48302 |
| Gauss–Legendre $4\times4$ | 0.497943 |
| Gauss–Legendre $8\times8$ | **0.49797178** |
| Simpson $8\times8$ | 0.49801 |
| Monte Carlo ($10^5$) | $0.4981\pm0.0016$ |
| reference (scipy CDF) | 0.49797178 |

**Take-away.** In 2-D, tensor-product Gauss rules are extremely efficient: 64 evaluations give 8 digits, while Monte Carlo gives 2–3 digits with $10^5$. In 10 or more dimensions the picture reverses (Chapter 4 exercises C4).

---

## Example 4.7 — Gauss–Hermite for option-like payoffs

**Problem.** Let $X\sim N(0.05, 0.2^2)$. Compute $E[e^X]$ (exact $e^{\mu+\sigma^2/2} = 1.0725081813$) and $E[(e^X-1)^+]$ (exact $0.1237832$, a Black–Scholes-type formula) by Gauss–Hermite quadrature.

**Solution.**

| $n$ | $E[e^X]$ error | $E[(e^X-1)^+]$ |
|---|---|---|
| 2 | $1.4\times10^{-4}$ | 0.1420 |
| 3 | $5.7\times10^{-7}$ | 0.1153 |
| 5 | $3.6\times10^{-12}$ | 0.1211 |
| 10 | $0$ | 0.1248 |

**Take-away.** $e^x$ is entire, so Gauss–Hermite converges spectrally. The call payoff has a *kink* at $X = 0$, and the quadrature error then decreases slowly and erratically. The fix is to split the integral at the kink, or to use the closed form. Smoothness, not dimension alone, decides which method wins.
