# Lecture 2 — Solving Nonlinear Equations

> **Week 2–3** · Code: [`numlib/roots.py`](../numlib/roots.py) · Examples: [`examples/ch02`](../examples/ch02_examples.md) · Exercises: [`exercises/ch02`](../exercises/ch02_exercises.md)

## Learning objectives

1. Apply bisection, fixed-point iteration, Newton's method and the secant method to $f(x)=0$.
2. State and use the convergence theorems for each method (linear, superlinear, quadratic).
3. Choose sensible stopping criteria.
4. Extend Newton's method to systems of equations using the Jacobian.
5. Recognise root-finding inside data-science tasks: maximum-likelihood estimation, quantiles, implied parameters, calibration.

---

## 2.1 Motivation

Many estimation problems reduce to "find the $x$ that makes this expression zero":

* **Maximum likelihood.** The MLE solves the *score equation* $\ell'(\theta) = 0$. For the shape parameter of a Gamma distribution this is $\log\alpha - \psi(\alpha) = \log\bar x - \overline{\log x}$, which has no closed form.
* **Quantiles.** The $p$-quantile of a continuous distribution solves $F(x) - p = 0$.
* **Internal rate of return / implied volatility** in finance.
* **Decision thresholds.** Find the classifier threshold where precision equals recall.

A root (or zero) of $f$ is a number $r$ with $f(r) = 0$.

---

## 2.2 Bisection

**Intermediate Value Theorem.** If $f$ is continuous on $[a,b]$ and $f(a)f(b) < 0$, there is at least one root in $(a,b)$.

```text
Algorithm BISECTION(f, a, b, tol)
  while (b - a)/2 > tol:
      m = a + (b - a)/2
      if f(m) == 0: return m
      if sign f(m) == sign f(a): a = m  else: b = m
  return a + (b - a)/2
```

**Theorem.** After $n$ steps the midpoint $x_n$ satisfies
$$
|x_n - r| \le \frac{b-a}{2^{n+1}}.
$$
So reaching tolerance $\varepsilon$ requires $n \ge \log_2\!\bigl((b-a)/\varepsilon\bigr) - 1$ steps — **known in advance**.

| Pros | Cons |
|---|---|
| Always converges once bracketed | Slow: one bit per step (linear, rate $1/2$) |
| Needs only signs of $f$ | Needs a bracket; misses double roots like $x^2$ |
| Guaranteed error bound | Does not generalise to systems |

---

## 2.3 Fixed-point iteration

Rewrite $f(x)=0$ as $x = g(x)$ and iterate $x_{k+1} = g(x_k)$.

**Example.** $x^3 - 2x - 5 = 0$ can be rewritten as
(a) $x = (x^3-5)/2$, (b) $x = \sqrt[3]{2x+5}$, (c) $x = 5/(x^2-2)$. Only (b) converges from $x_0 = 2$ — why?

**Contraction Mapping Theorem.** Suppose $g$ maps $[a,b]$ into itself and $|g'(x)| \le L < 1$ on $[a,b]$. Then

1. $g$ has a unique fixed point $r \in [a,b]$;
2. $x_{k+1} = g(x_k)$ converges to $r$ for every $x_0 \in [a,b]$;
3. $|x_k - r| \le L^k |x_0 - r|$ and $|x_k - r| \le \dfrac{L^k}{1-L}|x_1 - x_0|$.

*Proof idea.* By the Mean Value Theorem, $x_{k+1} - r = g(x_k) - g(r) = g'(\xi_k)(x_k - r)$, so the error shrinks by at least $L$ each step.

For (b): $g(x) = (2x+5)^{1/3}$, $g'(x) = \tfrac23(2x+5)^{-2/3} \approx 0.15$ near $r \approx 2.0946$ — fast linear convergence. For (a): $g'(r) = 3r^2/2 \approx 6.6 > 1$ — diverges.

### Order of convergence

A sequence $x_k \to r$ has **order $p$** with **rate** $C$ if
$$
\lim_{k\to\infty}\frac{|x_{k+1}-r|}{|x_k-r|^p} = C, \qquad C>0.
$$

* $p=1$, $0<C<1$: **linear** (bisection $C=\tfrac12$; fixed point $C = |g'(r)|$).
* $1<p<2$: **superlinear** (secant, $p \approx 1.618$).
* $p = 2$: **quadratic** (Newton) — the number of correct digits roughly **doubles** each step.

**Theorem.** If $g'(r) = 0$ and $g''$ is continuous, fixed-point iteration converges at least quadratically near $r$, with $C = |g''(r)|/2$.

---

## 2.4 Newton's method

Replace $f$ by its tangent line at $x_k$ and solve the linear equation:

$$
f(x) \approx f(x_k) + f'(x_k)(x - x_k) = 0 \quad\Longrightarrow\quad
\boxed{x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}}.
$$

Newton is fixed-point iteration with $g(x) = x - f(x)/f'(x)$; one checks $g'(r) = f(r)f''(r)/f'(r)^2 = 0$, which explains the quadratic convergence.

**Theorem (local quadratic convergence).** Let $f \in C^2$, $f(r)=0$, $f'(r)\ne0$. Then there is $\delta>0$ such that for $|x_0 - r| < \delta$ Newton's method converges and
$$
\lim_{k\to\infty}\frac{x_{k+1}-r}{(x_k-r)^2} = \frac{f''(r)}{2f'(r)}.
$$

*Proof sketch.* Taylor-expand around $x_k$: $0 = f(r) = f(x_k) + f'(x_k)(r-x_k) + \tfrac12 f''(\xi)(r-x_k)^2$. Divide by $f'(x_k)$ and rearrange to $x_{k+1} - r = \frac{f''(\xi)}{2f'(x_k)}(x_k-r)^2$.

**Example: square roots.** $f(x) = x^2 - a$ gives $x_{k+1} = \tfrac12(x_k + a/x_k)$ — the Babylonian method. For $a = 2$, $x_0 = 1$:
`1, 1.5, 1.41667, 1.4142157, 1.41421356237469, 1.41421356237310` — correct digits: 0, 1, 3, 6, 12, 16.

### When Newton fails

| Failure | Example | Remedy |
|---|---|---|
| $f'(x_k) = 0$ | $f(x) = x^3-2x+2$, $x_0=0$ cycles $0 \to 1 \to 0$ | Different start, safeguard |
| Divergence | $f(x) = \arctan x$, $\lvert x_0\rvert > 1.39$ | Damping/line search |
| Multiple root | $f(x) = (x-1)^2$ — only linear, $C=\tfrac12$ | Use $x_{k+1} = x_k - m f/f'$ for multiplicity $m$ |
| Derivative unavailable | Black-box simulator | Secant method, automatic differentiation |

**Safeguarded Newton** (the idea behind Brent's method and `scipy.optimize.brentq`): keep a bracket $[a,b]$; take the Newton step if it lands inside the bracket, otherwise bisect. You get bisection's guarantee *and* Newton's speed.

---

## 2.5 The secant method

Replace $f'(x_k)$ by the slope through the last two iterates:
$$
x_{k+1} = x_k - f(x_k)\,\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}.
$$

* One new function evaluation per step (Newton needs $f$ **and** $f'$).
* Order $p = \frac{1+\sqrt5}{2} \approx 1.618$ (the golden ratio).
* Per function evaluation, secant is often *faster* than Newton: two secant steps give order $1.618^2 \approx 2.62 > 2$.

**Regula falsi** (false position) uses the secant formula but keeps a bracket. It is guaranteed to converge but can stall with one endpoint fixed (linear convergence).

---

## 2.6 Stopping criteria

No single test is perfect; combine several:

1. **Step size:** $|x_{k+1}-x_k| < \text{tol}_x\,\max(1,|x_{k+1}|)$ (relative + absolute).
2. **Residual:** $|f(x_k)| < \text{tol}_f$ — beware: a flat $f$ has small residual far from the root; a steep $f$ has large residual very close to it.
3. **Iteration cap** $k \le k_{\max}$ — always include, and **report** when it is hit.

The attainable accuracy is limited by conditioning: the root of $f$ has absolute condition number $1/|f'(r)|$. Near a double root ($f'(r) = 0$) only about half the digits can be recovered.

---

## 2.7 Newton's method for systems

For $\mathbf F:\mathbb R^n \to \mathbb R^n$ with Jacobian $J_{ij} = \partial F_i/\partial x_j$:

$$
\mathbf F(\mathbf x_k + \Delta) \approx \mathbf F(\mathbf x_k) + J(\mathbf x_k)\,\Delta = \mathbf 0
\quad\Longrightarrow\quad
\text{solve } J(\mathbf x_k)\,\Delta_k = -\mathbf F(\mathbf x_k),\ \ \mathbf x_{k+1} = \mathbf x_k + \Delta_k.
$$

**Never form $J^{-1}$** — solve a linear system (Lecture 3). Each step costs one Jacobian evaluation plus an $O(n^3)$ solve; quasi-Newton methods (Broyden) update an approximate Jacobian cheaply.

**Example.** Intersect the circle $x^2+y^2=4$ with the hyperbola $xy=1$:
$$
\mathbf F(x,y) = \begin{pmatrix}x^2+y^2-4\\ xy-1\end{pmatrix},\qquad
J = \begin{pmatrix}2x & 2y\\ y & x\end{pmatrix}.
$$
From $(2, 0.5)$ the iterates converge to $(1.931852, 0.517638)$ in four steps.

---

## 2.8 Data-science application: MLE by Newton's method

For i.i.d. data from a Poisson–like or logistic model, the log-likelihood $\ell(\theta)$ is maximised where $\ell'(\theta) = 0$. Newton's method on $\ell'$ is

$$
\theta_{k+1} = \theta_k - \frac{\ell'(\theta_k)}{\ell''(\theta_k)}.
$$

In statistics this is called **Newton–Raphson**; replacing $-\ell''$ by its expectation (the Fisher information) gives **Fisher scoring**, which is exactly how `statsmodels` and R's `glm` fit generalised linear models (Iteratively Reweighted Least Squares). Lecture 9 generalises this to many parameters.

**Example: Cauchy location.** For $x_i \sim \text{Cauchy}(\theta,1)$,
$\ell'(\theta) = \sum_i \frac{2(x_i-\theta)}{1+(x_i-\theta)^2}$ — the likelihood equation can have several roots, so a good starting value (the sample median) matters. This is a practical reminder that Newton is only *locally* convergent.

---

## 2.9 Choosing a method

| Situation | Recommended |
|---|---|
| Scalar $f$, bracket known, robustness matters | `scipy.optimize.brentq` (safeguarded hybrid) |
| Derivative cheap, good initial guess | Newton |
| Derivative unavailable | Secant / Brent |
| System of equations | Newton / `scipy.optimize.root` (hybrid Powell) |
| Polynomial | `np.roots` (eigenvalues of the companion matrix — Lecture 6) |

## 2.10 Summary

* Bisection: guaranteed, linear, error halves each step.
* Fixed point: converges if $|g'| < 1$; rate $|g'(r)|$.
* Newton: quadratic near simple roots; needs $f'$ and a good start.
* Secant: order $1.618$, no derivative.
* Systems: Newton with the Jacobian, solving a linear system each step.

## Further reading

* Sauer, *Numerical Analysis*, Chapter 1; Burden & Faires, *Numerical Analysis*, Chapter 2.
* R. P. Brent, *Algorithms for Minimization without Derivatives*, 1973 (origin of `brentq`).
