# Chapter 2 — Exercises: Solutions of Equations in One Variable

> Lecture: [Chapter 2](../lectures/ch02-equations-in-one-variable.md) · Solutions: [`solutions/ch02_solutions.md`](../solutions/ch02_solutions.md) · Solution code: [`solutions/code/ch02_solutions.py`](../solutions/code/ch02_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★** Prove that the bisection midpoints satisfy $|p_n-p|\le(b-a)/2^n$, and deduce the minimum number of iterations needed for accuracy $\varepsilon$.

**A2 ★★** (a) Prove the uniqueness part of the Fixed-Point Theorem: if $|g'(x)|\le k<1$ on $(a,b)$, $g$ has at most one fixed point in $[a,b]$. (b) Show that Newton's iteration function $g(x) = x - f(x)/f'(x)$ satisfies $g'(p) = 0$ at a simple root $p$, and explain why this implies faster-than-linear convergence.

**A3 ★★** Let $p$ be a root of multiplicity $m\ge2$, $f(x) = (x-p)^mq(x)$, $q(p)\ne0$. Show that Newton's method converges linearly with asymptotic constant $1-\frac1m$, and that $x_{n+1} = x_n - m\frac{f(x_n)}{f'(x_n)}$ restores quadratic convergence.

**A4 ★★★** For the secant method assume $e_{n+1}\approx Ce_ne_{n-1}$. Seeking $|e_{n+1}|\approx K|e_n|^\alpha$, show that $\alpha^2 = \alpha+1$, so $\alpha = \frac{1+\sqrt5}{2}$.

**A5 ★★** Show that Aitken's $\Delta^2$ method reproduces the limit **exactly** for any sequence of the form $p_n = p + c\lambda^n$ ($c\ne0$, $\lambda\ne0,1$). Why does this explain its effectiveness on linearly convergent sequences?

## B. Hand computation

**B1 ★** Show that $f(x) = x^3+2x^2-3x-1$ has a root in $[1,2]$. Perform four bisection steps, and find how many steps guarantee an error below $10^{-4}$.

**B2 ★★** The equation $x^2 - x - 2 = 0$ has the root $p = 2$. For each rearrangement decide (via $|g'(2)|$) whether fixed-point iteration converges near $2$, and iterate six times from $x_0 = 2.5$: (a) $g(x) = x^2-2$; (b) $g(x) = \sqrt{x+2}$; (c) $g(x) = 1+\frac2x$.

**B3 ★** Use Newton's method to compute $\sqrt[3]5$ from $x_0 = 2$ (write the iteration in simplified form). How many correct digits does each iterate have?

**B4 ★★** Apply the secant method to $f(x) = x^2 - e^{-x}$ with $p_0 = 0$, $p_1 = 1$ until $|p_n - p_{n-1}|<10^{-10}$.

**B5 ★★** Apply the method of false position to the same $f$ on $[0,1]$ for four iterations. Compare the number of iterations needed for $10^{-12}$ with the secant method, and explain the difference.

**B6 ★★** Apply Aitken's $\Delta^2$ method to the sequence of B2(b) and compare the errors of $p_n$ and $\hat p_n$.

**B7 ★★** For $P(x) = x^4 - 3x^3 + x^2 + x + 1$: (a) use Horner's method to compute $P(2)$ and $P'(2)$ and the deflated quotient; (b) find the largest real root by Newton–Horner from $x_0 = 2.5$; (c) explain how Müller's method could find the complex roots.

**B8 ★★** Apply Newton's method to $f(x) = (x-1)^3$ from $x_0 = 2$ for six steps. Verify that $\frac{x_{n+1}-1}{x_n-1} = \frac23$, and show the modified method with $m = 3$ reaches the root in one step.

## C. Programming

**C1 ★** Implement bisection, Newton and secant with iteration logs and apply them to $x - 2\sin x = 0$ (positive root). Report iterations to reach $10^{-12}$.

**C2 ★★** Write a function that estimates the order of convergence from a list of iterates, and apply it to the Newton and secant sequences of C1.

**C3 ★★★** Plot the basins of attraction of Newton's method for $z^3 = 1$ on $[-2,2]^2\subset\mathbb C$ (colour each starting point by the root it converges to). Estimate the fraction of starting points converging to each root and comment on the fractal boundary.

**C4 ★★** Implement a safeguarded Newton method (Newton step if it stays in the current bracket, bisection otherwise) and show that it finds the root of $\arctan x$ from the bracket $[-2,5]$, where pure Newton from $x_0 = 5$ diverges.

**C5 ★★★** For $W_{10}(x) = \prod_{k=1}^{10}(x-k)$ compare three ways of computing all roots: `np.roots` (companion-matrix eigenvalues), Newton–Horner with deflation (smallest roots first), and Newton–Horner with deflation (largest first). Report the maximum errors and explain.

## D. Data-science applications

**D1 ★★** A project has cash flows $-100, +230, -132$ (years 0, 1, 2). Show that the NPV equation has **two** internal rates of return and find them. Why is IRR ambiguous here, and what should an analyst report instead?

**D2 ★★** For a zero-truncated Poisson sample (counts are only recorded when positive) the MLE of $\lambda$ solves $\frac{\lambda}{1-e^{-\lambda}} = \bar x$. Solve it by Newton's method for $\bar x = 1.8$.

**D3 ★** Compute the 95th percentile of the Gamma distribution with shape 3 and scale 2 by Newton's method on $F(x) - 0.95$ (use `scipy.stats.gamma.cdf/pdf`), and compare with `scipy.stats.gamma.ppf`.

**D4 ★★** In t-SNE, for each point one finds a precision $\beta$ such that the distribution $p_j\propto e^{-\beta d_j^2}$ has a prescribed **perplexity** $2^{H(p)}$. For squared distances $d^2 = (1, 1.5, 2, 4, 5, 9)$ find $\beta$ with perplexity 3 by bisection. Why is bisection (rather than Newton) the standard choice here?
