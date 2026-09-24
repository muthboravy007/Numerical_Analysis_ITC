# Chapter 4 — Solutions

> Exercises: [`exercises/ch04_exercises.md`](../exercises/ch04_exercises.md) · Numbers from [`solutions/code/ch04_solutions.py`](code/ch04_solutions.py).

## A. Theory and proofs

**A1.** $f(x_0\pm h) = f(x_0)\pm hf'(x_0) + \frac{h^2}{2}f''(x_0)\pm\frac{h^3}{6}f'''(\xi_\pm)$. Subtracting and dividing by $2h$: $\frac{f(x_0+h)-f(x_0-h)}{2h} = f'(x_0) + \frac{h^2}{12}[f'''(\xi_+)+f'''(\xi_-)] = f'(x_0) + \frac{h^2}{6}f'''(\xi)$ (IVT). With perturbed values $\tilde f = f + e$, $|e|\le\varepsilon$, the computed quotient differs by at most $\frac{2\varepsilon}{2h} = \frac\varepsilon h$. Total error $E(h)\le\frac\varepsilon h+\frac{h^2}{6}M$; $E'(h) = 0$ gives $h^\ast = \sqrt[3]{3\varepsilon/M}$.

**A2.** Simpson's rule is exact for $1, x, x^2$ by construction (it integrates the quadratic interpolant). On a symmetric interval $[-h,h]$ it gives $\frac h3[(-h)^3 + 0 + h^3] = 0 = \int_{-h}^hx^3dx$, so it is exact for cubics as well. For $x^4$: rule $\frac h3(2h^4) = \frac23h^5\ne\frac25h^5$. Degree of precision 3.

**A3.** On each subinterval the error is $-\frac{h^3}{12}f''(\xi_j)$. Summing: $-\frac{h^3}{12}\sum_{j=1}^nf''(\xi_j) = -\frac{h^3n}{12}\bar{f''}$ where $\bar{f''}$, the average of the values $f''(\xi_j)$, lies between $\min f''$ and $\max f''$; since $f''$ is continuous, the IVT gives $\mu$ with $f''(\mu) = \bar{f''}$. With $nh = b-a$ the error is $-\frac{b-a}{12}h^2f''(\mu)$.

**A4.** $p(x) = \prod(x-x_i)^2\ge0$ and not identically zero, so $\int_{-1}^1p>0$, while the rule gives $\sum c_ip(x_i) = 0$. So the rule fails for this degree-$2n$ polynomial; combined with exactness up to degree $2n-1$, the degree of precision is exactly $2n-1$.

**A5.** For $f$ with period $L = b-a$ the Euler–Maclaurin correction terms are proportional to $f^{(2k-1)}(b) - f^{(2k-1)}(a) = 0$ for all $k$, so the error decays faster than any power of $h$. Equivalently, with the Fourier series $f = \sum c_ke^{2\pi ikx/L}$ the $n$-point trapezoid rule integrates $e^{2\pi ikx/L}$ exactly unless $n\mid k$ (aliasing), so the error is $\sum_{m\ne0}c_{mn}$, which decays like the Fourier coefficients — exponentially for analytic $f$.

## B. Hand computation

**B1.** True $f'(1) = 2e = 5.4365637$.

| $h$ | forward | central | five-point |
|---|---|---|---|
| 0.1 | 5.863008 | 5.454699 | 5.43650920 |
| 0.05 | 5.645037 | 5.441095 | 5.43656026 |

Error ratios when halving $h$: forward $0.426/0.208\approx2$ (order 1), central $0.0181/0.00453\approx4$ (order 2), five-point $5.4\times10^{-5}/3.4\times10^{-6}\approx16$ (order 4).

**B2.** $N(0.2) = 5.509269$, $N(0.1) = 5.454699$; $\frac{4N(0.1)-N(0.2)}{3} = 5.436509$, error $5.4\times10^{-5}$ (vs $1.8\times10^{-2}$ for $N(0.1)$).

**B3.** Trapezoid $\frac12(1+0.5) = 0.75$; Simpson $\frac16(1 + 4\cdot0.8 + 0.5) = 0.78333$; midpoint $g(0.5) = 0.8$. Exact $0.785398$.

**B4.** $n = 4$: $T = 0.7827941$ (error $2.60\times10^{-3}$), $S = 0.7853922$ ($6.0\times10^{-6}$). $n = 8$: $T = 0.7847471$ ($6.5\times10^{-4}$), $S = 0.7853981$ ($3.8\times10^{-8}$). Ratios: trapezoid 4.0 ($h^2$). Simpson's ratio is 159, much more than 16: for this $f$ the leading $h^4$ error term cancels ($f'''(1) - f'''(0) = 0$), so the next term, $O(h^6)$, dominates.

**B5.** Trapezoid: $\frac{\pi}{12}h^2\max|\sin| = \frac{\pi^3}{12n^2}\le10^{-6}$ ⇒ $n\ge1607.4$, i.e. $n = 1608$. Simpson: $\frac{\pi}{180}h^4 = \frac{\pi^5}{180n^4}\le10^{-6}$ ⇒ $n\ge36.1$, i.e. $n = 38$ (even).

**B6.** Romberg table:
$$
\begin{array}{llll}0.78539816\\0.64590905&0.59941268\\0.61419877&0.60362868&0.60390974\\0.60648966&0.60391996&0.60393938&0.60393985\end{array}
$$
Exact $\frac12(1+e^{-\pi/2}) = 0.60393979$; $R_{4,4}$ error $6\times10^{-8}$.

**B7.** $x = 1.5t + 2.5$, $dx = 1.5\,dt$. 2-point: $1.5[\ln(2.5 - 0.866) + \ln(2.5+0.866)] = 2.557122$; 3-point: $2.546084$. Exact $4\ln4 - 3 = 2.545177$ (errors $0.012$ and $9\times10^{-4}$).

**B8.** The result is $0.5$ = exact. The inner integrand $x+y$ is linear in $y$, and the inner integral $\frac32x^2$ is quadratic in $x$; Simpson's rule is exact for both.

**B9.** $P_2(x) = 1 - x + \frac{x^2}{2}$: $\int_0^1x^{-1/2}P_2 = 2 - \frac23 + \frac15 = 1.533333$. The remainder $G(x) = \frac{e^{-x}-P_2(x)}{\sqrt x}$ ($G(0) = 0$) integrated by Simpson ($n = 4$): $-0.039660$. Total $1.493673$; exact $\sqrt\pi\operatorname{erf}1 = 1.493648$ (error $2.5\times10^{-5}$).

## C. Programming (key results)

**C1.** Errors vs $h$: $3.1\times10^{-4}$ ($10^{-1}$), $3.1\times10^{-6}$, $3.1\times10^{-8}$, $1.3\times10^{-8}$ ($10^{-4}$, best), then $1.1\times10^{-6}$, $8.9\times10^{-5}$, $5.4\times10^{-3}$, $0.25$ ($10^{-8}$). Truncation $\frac{h^2}{12}|f^{(4)}|$ vs round-off $\approx\frac{4u|f|}{h^2}$; the optimum is $h\approx(48u|f|/|f^{(4)}|)^{1/4}\approx10^{-4}$.

**C2.** TOL $10^{-4}$: $-0.4444432$ with 65 evaluations; TOL $10^{-8}$: $-0.44444444442$ with 521 evaluations. Points cluster near $x = 0$, where $\sqrt x\ln x$ has unbounded derivatives.

**C3.** $\int|x|$ (exact 1): $1.1547, 1.0425, 1.0115, 1.0030$ — error $O(n^{-2})$ because of the kink at $0$. $\int e^x$: errors $7.7\times10^{-3}, 3.0\times10^{-7}, 9\times10^{-16}, 0$ — spectral convergence for analytic integrands. Gaussian quadrature is only as good as the smoothness of $f$; split the interval at the kink to recover fast convergence.

**C4.**

| $d$ | MC estimate ± s.e. | exact |
|---|---|---|
| 1 | $0.74738\pm0.00045$ | 0.746824 |
| 3 | $0.41585\pm0.00045$ | 0.416538 |
| 6 | $0.17344\pm0.00028$ | 0.173504 |
| 10 | $0.05404\pm0.00012$ | 0.053974 |

All within ~1.5 standard errors. A tensor Simpson rule with 11 points/axis would need $11^{10}\approx2.6\times10^{10}$ evaluations in $d = 10$.

## D. Data-science applications

**D1.** AUC $= 0.755$ (sum of trapezoids; the horizontal segments contribute rectangles).

**D2.** Trapezoid $38.25$, Simpson $40.38$, exact $41.26$. The (linear) trapezoidal rule is the regulatory standard (simple, never negative, robust to noise), but with 2-hour sampling it underestimates the concave peak region: every chord lies below a concave curve. Log-trapezoidal rules on the declining phase and denser sampling near the peak reduce this bias.

**D3.** Exact $0.769696$. Gauss–Hermite: $n = 5$: $0.78319$; $10$: $0.74916$; $20$: $0.76057$ — erratic, slow convergence because $\max(x,0)$ has a kink (not smooth, cf. C3). Monte Carlo ($10^5$): $0.7732\pm0.0027$. For kinked integrands, integrate analytically on each side of the kink or use the closed form.

**D4.** Black–Scholes $7.2558$. Plain MC ($2\times10^5$): $7.214\pm0.033$; antithetic: $7.247\pm0.020$. Standard error ratio $0.62$ ⇒ variance reduced by a factor ≈$2.6$ at no extra random-number cost (the payoff is monotone in $Z$, so $Z$ and $-Z$ are negatively correlated).
