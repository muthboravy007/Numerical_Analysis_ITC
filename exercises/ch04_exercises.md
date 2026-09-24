# Chapter 4 — Exercises: Numerical Differentiation and Integration

> Lecture: [Chapter 4](../lectures/ch04-differentiation-integration.md) · Solutions: [`solutions/ch04_solutions.md`](../solutions/ch04_solutions.md) · Solution code: [`solutions/code/ch04_solutions.py`](../solutions/code/ch04_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★★** Derive the three-point midpoint formula $f'(x_0) = \frac{f(x_0+h)-f(x_0-h)}{2h} - \frac{h^2}{6}f'''(\xi)$ from Taylor's theorem. Then show that if each function value has an error at most $\varepsilon$, the total error is at most $\frac\varepsilon h + \frac{h^2}6M$ ($M = \max|f'''|$) and find the minimising $h$.

**A2 ★★** Show that Simpson's rule on $[a,b]$ is exact for all cubics, and that its degree of precision is exactly 3.

**A3 ★★** Derive the composite trapezoidal error $-\frac{b-a}{12}h^2f''(\mu)$ from the single-interval error, explaining where the Intermediate Value Theorem is used.

**A4 ★★★** Show that the $n$-point Gauss–Legendre rule cannot be exact for the polynomial $\prod_{i=1}^n(x-x_i)^2$ (degree $2n$), so its degree of precision is exactly $2n-1$.

**A5 ★★** Explain why the composite trapezoidal rule converges exponentially fast for smooth periodic functions integrated over a period (use the Euler–Maclaurin formula or the Fourier series of $f$).

## B. Hand computation

**B1 ★** For $f(x) = xe^x$ at $x_0 = 1$ ($f'(1) = 2e$), compute the forward, central and five-point approximations with $h = 0.1$ and $h = 0.05$. Estimate the order of each from the two values of $h$.

**B2 ★** Apply one step of Richardson extrapolation to the central differences $N(0.2)$ and $N(0.1)$ from B1's function and give the error.

**B3 ★** Approximate $\int_0^1\frac{dx}{1+x^2} = \frac\pi4$ by the trapezoidal, Simpson's and midpoint rules (single interval).

**B4 ★★** Repeat B3 with composite trapezoid and Simpson rules for $n = 4$ and $n = 8$. Verify the error ratios predicted by the theory.

**B5 ★★** How many subintervals are needed to approximate $\int_0^\pi\sin x\,dx$ within $10^{-6}$ by the composite trapezoidal rule? By composite Simpson? (Use the error bounds.)

**B6 ★★** Build the Romberg table $R_{4,4}$ for $\int_0^{\pi/2}e^{-x}\cos x\,dx$ and compare with the exact value $\frac12(1+e^{-\pi/2})$.

**B7 ★** Approximate $\int_1^4\ln x\,dx$ with 2- and 3-point Gauss–Legendre quadrature (after changing variables) and compare with $4\ln4 - 3$.

**B8 ★** Apply the composite Simpson double-integral rule with $n = m = 2$ to $\int_0^1\int_0^x(x+y)\,dy\,dx$. Why is the result exact?

**B9 ★★★** Approximate $\int_0^1\frac{e^{-x}}{\sqrt x}dx$ by subtracting the second Taylor polynomial of $e^{-x}$ and applying Simpson's rule with $n = 4$ to the remainder. Compare with $\sqrt\pi\,\operatorname{erf}(1)$.

## C. Programming

**C1 ★★** For $f(x) = \ln x$ at $x = 2$ compute the second-derivative formula $\frac{f(x+h)-2f(x)+f(x-h)}{h^2}$ for $h = 10^{-1},\dots,10^{-8}$. Find the best $h$ and explain the error curve.

**C2 ★★** Implement adaptive Simpson quadrature and apply it to $\int_0^1\sqrt x\ln x\,dx = -\frac49$ with tolerances $10^{-4}$ and $10^{-8}$. Report function evaluations. Where does the algorithm place its points?

**C3 ★★** Compare Gauss–Legendre with $n = 2,4,8,16$ on $\int_{-1}^1|x|\,dx$ and $\int_{-1}^1e^x\,dx$. Explain the very different convergence.

**C4 ★★** Estimate $\int_{[0,1]^d}e^{-\|\mathbf x\|^2}d\mathbf x$ by Monte Carlo ($N = 2\times10^5$) for $d = 1, 3, 6, 10$ with standard errors, and compare with $\bigl(\frac{\sqrt\pi}{2}\operatorname{erf}1\bigr)^d$. How many points would a tensor Simpson rule with 11 points per axis need for $d = 10$?

## D. Data-science applications

**D1 ★** Compute the ROC AUC for the ROC points (FPR, TPR): $(0,0), (0,0.2), (0.1,0.2), (0.1,0.55), (0.4,0.55), (0.4,0.8), (0.7,1), (1,1)$.

**D2 ★★** In pharmacokinetics the drug exposure is the area under the concentration curve. Concentrations $C(t) = 10(e^{-0.2t}-e^{-1.2t})$ are measured every 2 hours from $0$ to $24$ h. Compute the AUC by the trapezoidal and Simpson rules and compare with the exact $\int_0^{24}C\,dt$. Which rule do regulators typically use, and why might it be biased here?

**D3 ★★** For $X\sim N(0.5, 1.2^2)$ compute $E[\max(X,0)]$ (the expected ReLU output) by Gauss–Hermite quadrature with $n = 5, 10, 20$, by Monte Carlo, and exactly ($\mu\Phi(\mu/\sigma)+\sigma\varphi(\mu/\sigma)$). Why does Gauss–Hermite converge slowly here?

**D4 ★★★** Price a European call ($S_0 = 100$, $K = 110$, $r = 0.03$, $\sigma = 0.25$, $T = 1$) by Monte Carlo under geometric Brownian motion with and without antithetic variates, and compare with the Black–Scholes formula. By what factor did the antithetic variates reduce the variance?
