# Lecture 8 — Numerical Differentiation and Integration

> **Week 11** · Code: [`numlib/integration.py`](../numlib/integration.py) · Examples: [`examples/ch08`](../examples/ch08_examples.md) · Exercises: [`exercises/ch08`](../exercises/ch08_exercises.md)

## Learning objectives

1. Derive finite-difference formulas from Taylor series and determine their order.
2. Analyse the truncation–rounding trade-off and choose the step size $h$.
3. Use Richardson extrapolation, the complex-step method, and know how automatic differentiation differs.
4. Derive and apply the midpoint, trapezoidal and Simpson rules (simple and composite) with error bounds.
5. Use Romberg integration, adaptive quadrature and Gaussian quadrature.
6. Apply Monte Carlo integration, understand its $O(N^{-1/2})$ rate and why it wins in high dimension.

---

## Part A — Numerical differentiation

## 8.1 Finite differences

From Taylor's theorem $f(x\pm h) = f(x) \pm hf'(x) + \frac{h^2}{2}f''(x) \pm \frac{h^3}{6}f'''(x) + \cdots$:

| Formula | Expression | Truncation error |
|---|---|---|
| Forward | $\dfrac{f(x+h)-f(x)}{h}$ | $-\tfrac h2 f''(\xi)$ — $O(h)$ |
| Backward | $\dfrac{f(x)-f(x-h)}{h}$ | $O(h)$ |
| Central | $\dfrac{f(x+h)-f(x-h)}{2h}$ | $-\tfrac{h^2}{6}f'''(\xi)$ — $O(h^2)$ |
| Second derivative | $\dfrac{f(x+h)-2f(x)+f(x-h)}{h^2}$ | $-\tfrac{h^2}{12}f^{(4)}(\xi)$ — $O(h^2)$ |

### Choosing $h$: the V-shaped error curve

Each function value carries a rounding error $\approx u|f|$. For the central difference the total error is about
$$
E(h) \approx \frac{h^2}{6}|f'''| + \frac{u|f|}{h},\qquad h^\star \approx \Bigl(\frac{3u|f|}{|f'''|}\Bigr)^{1/3}\approx 10^{-5},\qquad E(h^\star)\approx 10^{-11}.
$$
For the forward difference $h^\star\approx\sqrt u\approx 10^{-8}$ with $E\approx 10^{-8}$.

Errors for $f = e^x$ at $x=1$:

| $h$ | forward error | central error |
|---|---|---|
| $10^{-1}$ | $1.4\times10^{-1}$ | $4.5\times10^{-3}$ |
| $10^{-2}$ | $1.4\times10^{-2}$ | $4.5\times10^{-5}$ |
| $10^{-4}$ | $1.4\times10^{-4}$ | $4.5\times10^{-9}$ |
| $10^{-6}$ | $1.4\times10^{-6}$ | $1.6\times10^{-10}$ |
| $10^{-8}$ | $6.6\times10^{-9}$ | $6.6\times10^{-9}$ |
| $10^{-10}$ | $1.5\times10^{-6}$ | $6.7\times10^{-7}$ |
| $10^{-12}$ | $4.3\times10^{-4}$ | $2.1\times10^{-4}$ |

Smaller $h$ is **not** always better.

## 8.2 Richardson extrapolation

If $D(h) = D + c_1h^p + c_2h^{2p} + \cdots$, combine two step sizes to cancel the leading error:
$$
D \approx \frac{2^pD(h/2) - D(h)}{2^p - 1}\qquad(\text{new error } O(h^{2p})).
$$
Repeating gives a triangular table (the same idea as Romberg integration below). For central differences on $e^x$ at $x=1$ with $h=0.2$, three levels give error $5\times10^{-10}$ using only step sizes $\ge 0.05$.

## 8.3 Complex step and automatic differentiation

**Complex step.** For real-analytic $f$: $f(x+ih) = f(x) + ihf'(x) - \frac{h^2}{2}f''(x) + \cdots$, so
$$
f'(x) \approx \frac{\operatorname{Im} f(x+ih)}{h}
$$
with $O(h^2)$ error and **no subtraction** — take $h = 10^{-20}$ and get full machine precision.

**Automatic differentiation (AD)** applies the chain rule to the program itself, giving derivatives exact to rounding error. *Reverse mode* (back-propagation) computes the gradient of a scalar loss with respect to millions of parameters for about the cost of a few function evaluations — the engine of PyTorch, JAX and TensorFlow. Finite differences would need one evaluation **per parameter**.

**Gradient checking.** Finite differences remain essential for *testing* hand-written or AD gradients: compare $\nabla f$ against central differences on a few random coordinates with $h\approx10^{-5}$ and a relative tolerance of $\sim10^{-6}$.

---

## Part B — Numerical integration (quadrature)

Approximate $I = \int_a^b f(x)\,dx$ by $Q = \sum_i w_if(x_i)$.

## 8.4 Newton–Cotes rules

Integrate the interpolating polynomial on equispaced nodes:

| Rule | Formula on $[a,b]$, $h = b-a$ | Error | Exact for degree |
|---|---|---|---|
| Midpoint | $h\,f\bigl(\tfrac{a+b}{2}\bigr)$ | $+\tfrac{h^3}{24}f''(\xi)$ | 1 |
| Trapezoid | $\tfrac h2\bigl(f(a)+f(b)\bigr)$ | $-\tfrac{h^3}{12}f''(\xi)$ | 1 |
| Simpson | $\tfrac h6\bigl(f(a)+4f(\tfrac{a+b}2)+f(b)\bigr)$ | $-\tfrac{h^5}{2880}f^{(4)}(\xi)$ | 3 |

The **degree of precision** is the highest degree of polynomial integrated exactly. Simpson gains a degree "for free" by symmetry.

## 8.5 Composite rules

Split $[a,b]$ into $n$ subintervals of width $h = (b-a)/n$ and apply a simple rule on each:

$$
T_n = h\Bigl[\tfrac12 f_0 + f_1 + \cdots + f_{n-1} + \tfrac12 f_n\Bigr],\qquad
E_T = -\frac{(b-a)h^2}{12}f''(\xi) = O(h^2),
$$
$$
S_n = \frac h3\Bigl[f_0 + 4f_1 + 2f_2 + 4f_3 + \cdots + 4f_{n-1} + f_n\Bigr]\ (n\text{ even}),\qquad
E_S = -\frac{(b-a)h^4}{180}f^{(4)}(\xi) = O(h^4).
$$

**Example.** $I = \int_0^1e^{-x^2}dx = 0.746824133\ldots$ (the Gaussian integral behind `erf`):

| $n$ | Midpoint | Trapezoid | Simpson |
|---|---|---|---|
| 2 | 0.754598 | 0.731370 | 0.747180 |
| 4 | 0.748747 | 0.742984 | 0.746855 |
| 8 | 0.747304 | 0.745866 | 0.746826 |

Trapezoid error drops ×4 per halving ($O(h^2)$); Simpson ×16 ($O(h^4)$).

**Remarkable fact.** For *periodic* smooth functions integrated over a full period, the trapezoid rule converges **exponentially** (Euler–Maclaurin: all correction terms vanish). The same is true for rapidly decaying integrands on $\mathbb R$.

**Tabulated data.** With measurements at given (possibly irregular) points, use the trapezoid rule directly: `np.trapezoid(y, x)`. E.g. the **AUC** of an ROC curve in `sklearn.metrics.auc` is exactly the trapezoid rule.

## 8.6 Romberg integration

By the Euler–Maclaurin formula, $T(h) = I + c_1h^2 + c_2h^4 + \cdots$. Richardson extrapolation on $T(h), T(h/2), \ldots$:
$$
R_{i,0} = T(h/2^i),\qquad R_{i,j} = R_{i,j-1} + \frac{R_{i,j-1}-R_{i-1,j-1}}{4^j - 1}.
$$
$R_{i,1}$ is Simpson's rule; the diagonal converges very rapidly for smooth $f$. For $\int_0^1e^{-x^2}dx$ with 4 levels (9 function evaluations): $R_{3,3} = 0.74682402$, error $1.1\times10^{-7}$.

## 8.7 Adaptive quadrature

Use small $h$ only where $f$ varies rapidly. **Adaptive Simpson**: compute $S$ on $[a,b]$ and $S_L + S_R$ on the two halves. Since the error of $S_L+S_R$ is about $\frac{1}{15}|S_L+S_R - S|$, accept if this is below the tolerance; otherwise recurse on each half with half the tolerance. `scipy.integrate.quad` (QUADPACK) uses an adaptive Gauss–Kronrod version.

## 8.8 Gaussian quadrature

Let the nodes be free too. With $n$ nodes and $n$ weights we can match $2n$ moments:

**Theorem.** The $n$-point Gauss–Legendre rule on $[-1,1]$, with nodes at the roots of the Legendre polynomial $P_n$ and suitable positive weights, is exact for all polynomials of degree $\le 2n-1$ — the maximum possible.

| $n$ | nodes $x_i$ | weights $w_i$ |
|---|---|---|
| 1 | $0$ | $2$ |
| 2 | $\pm 1/\sqrt3$ | $1, 1$ |
| 3 | $0,\ \pm\sqrt{3/5}$ | $8/9,\ 5/9, 5/9$ |

Map to $[a,b]$: $\int_a^bf\,dx \approx \frac{b-a}{2}\sum w_if\bigl(\tfrac{b-a}{2}x_i+\tfrac{a+b}{2}\bigr)$.

For $\int_0^1e^{-x^2}dx$: 2 points → error $2.3\times10^{-4}$; 3 points → $9.5\times10^{-6}$; 4 points → $3.4\times10^{-7}$. Compare Simpson with $n=8$ (9 evaluations): error $1.9\times10^{-6}$.

Variants: **Gauss–Hermite** for $\int e^{-x^2}f(x)\,dx$ (expectations under a Gaussian — used in GLMMs and Bayesian quadrature), **Gauss–Laguerre** for $\int_0^\infty e^{-x}f(x)\,dx$.

---

## 8.9 Monte Carlo integration

Write the integral as an expectation: with $X\sim\text{Uniform}(a,b)$,
$$
I = (b-a)\,\mathbb E[f(X)] \approx \hat I_N = \frac{b-a}{N}\sum_{i=1}^N f(X_i).
$$

* By the CLT, $\hat I_N$ has standard error $\sigma/\sqrt N$ — **order $N^{-1/2}$, independent of dimension $d$**.
* A tensor-product rule with $m$ points per axis needs $N = m^d$ evaluations and has error $O(N^{-p/d})$: in $d = 10$ even Simpson ($p=4$) is only $O(N^{-0.4})$ — worse than Monte Carlo. This **curse of dimensionality** is why Bayesian inference, option pricing and physics use Monte Carlo.
* Always report the estimate **with its standard error**.

**Example: estimating $\pi$.** Sample points uniformly in $[-1,1]^2$; the fraction inside the unit disc estimates $\pi/4$.

| $N$ | estimate | std. error |
|---|---|---|
| $10^2$ | 2.92 | 0.18 |
| $10^4$ | 3.164 | 0.016 |
| $10^6$ | 3.1415 | 0.0016 |

100× more samples → 10× smaller error.

### Variance reduction
* **Importance sampling:** sample $X\sim q$ and average $f(X)p(X)/q(X)$; choose $q$ proportional to $|f|p$.
* **Control variates:** subtract a correlated quantity with known mean.
* **Antithetic variates:** pair $U$ with $1-U$.
* **Quasi-Monte Carlo:** low-discrepancy (Sobol, Halton) points give nearly $O(N^{-1})$ — `scipy.stats.qmc`.

## 8.10 Summary

* Differentiation is ill-conditioned: balance truncation $O(h^p)$ against rounding $O(u/h)$. Use central differences, Richardson, complex step — or AD.
* Composite trapezoid $O(h^2)$, Simpson $O(h^4)$; Romberg/Gauss for smooth integrands; adaptive for localised features.
* Monte Carlo converges as $N^{-1/2}$ in any dimension — the method of choice for $d\gtrsim 5$.

## Further reading

* Sauer, Chapter 5; Burden & Faires, Chapter 4.
* Baydin et al., *Automatic Differentiation in Machine Learning: a Survey*, JMLR, 2018.
* A. B. Owen, *Monte Carlo Theory, Methods and Examples* (free online).
