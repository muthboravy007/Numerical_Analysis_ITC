# Chapter 2 — Solutions of Equations in One Variable

> **Reference:** Burden & Faires, Chapter 2 (§2.1–2.6) plus §2.7 on data-science applications.
> **Code:** [`numlib/roots.py`](../numlib/roots.py) · **All numbers reproduced by** [`lectures/code/ch02.py`](code/ch02.py)
> **Worked problems:** [`examples/ch02`](../examples/ch02_examples.md) · **Homework:** [`exercises/ch02`](../exercises/ch02_exercises.md)

## Learning outcomes

1. Apply bisection, fixed-point iteration, Newton's method, the secant method and the method of false position, and write each as an algorithm.
2. Prove convergence of bisection and of fixed-point iteration (contraction mapping), and derive the quadratic convergence of Newton's method.
3. Define order of convergence and asymptotic error constant; analyse multiple roots.
4. Accelerate linearly convergent sequences with Aitken's $\Delta^2$ and Steffensen's method.
5. Evaluate and deflate polynomials with Horner's method; find complex roots with Müller's method.
6. Formulate estimation problems in statistics and machine learning (MLE, quantiles, calibration) as root-finding problems.

Throughout, a **root** (zero) of $f$ is a number $p$ with $f(p)=0$. Running example: $f(x) = x^3-2x-5$, whose unique real root is $p = 2.0945514815423265$.

---

## 2.1 The Bisection Method

If $f\in C[a,b]$ and $f(a)f(b)<0$, the IVT guarantees a root in $(a,b)$. Halve the interval and keep the half on which $f$ changes sign.

```text
ALGORITHM 2.1  Bisection
INPUT   a, b; tolerance TOL; maximum iterations N0
OUTPUT  approximate p or failure message
Step 1  i = 1; FA = f(a)
Step 2  while i <= N0:
Step 3      p = a + (b - a)/2;  FP = f(p)          # safer than (a+b)/2
Step 4      if FP == 0 or (b - a)/2 < TOL: OUTPUT p; STOP
Step 5      i = i + 1
Step 6      if FA*FP > 0: a = p; FA = FP  else: b = p
Step 7  OUTPUT "method failed after N0 iterations"
```

**Theorem 2.1.** If $f\in C[a,b]$ and $f(a)f(b)<0$, bisection generates $\{p_n\}$ with
$$
|p_n - p| \le \frac{b-a}{2^n},\qquad n\ge1 .
$$
*Proof.* After $n$ steps the bracket has length $(b-a)/2^{n-1}$ and $p_n$ is its midpoint, while $p$ lies in the bracket, so $|p_n-p|\le\frac12\cdot\frac{b-a}{2^{n-1}}$. ∎

So $p_n = p + O(2^{-n})$: **linear convergence with rate $\tfrac12$**, and the number of iterations for tolerance $\varepsilon$ is known in advance:
$$
n \ge \log_2\frac{b-a}{\varepsilon}.
$$

**Stopping criteria.** Possible tests: (i) $|p_n - p_{n-1}|<\varepsilon$; (ii) $|p_n-p_{n-1}|/|p_n|<\varepsilon$; (iii) $|f(p_n)|<\varepsilon$. Each can fail (Example 2.1.6); the relative test (ii) is usually best when $p\neq 0$. Always cap the number of iterations.

### Examples for §2.1

**Example 2.1.1.** Bisection for $x^3-2x-5=0$ on $[2,3]$ ($f(2) = -1$, $f(3) = 16$):

| $n$ | $p_n$ | $f(p_n)$ | new bracket |
|---|---|---|---|
| 1 | 2.5 | $+5.625$ | $[2, 2.5]$ |
| 2 | 2.25 | $+1.891$ | $[2, 2.25]$ |
| 3 | 2.125 | $+0.3457$ | $[2, 2.125]$ |
| 4 | 2.0625 | $-0.3513$ | $[2.0625, 2.125]$ |
| 5 | 2.09375 | $-0.008942$ | $[2.09375, 2.125]$ |
| 6 | 2.109375 | $+0.1668$ | $[2.09375, 2.109375]$ |
| 7 | 2.1015625 | $+0.07856$ | $[2.09375, 2.1015625]$ |
| 8 | 2.09765625 | $+0.03471$ | $[2.09375, 2.09765625]$ |

For $\varepsilon = 10^{-6}$ Theorem 2.1 requires $n\ge\log_2 10^6 = 19.93$, i.e. **20 iterations**, which is exactly what the algorithm performs. Note that $|f(p_5)|$ is much smaller than $|f(p_8)|$ although $p_8$ is closer to $p$ — the residual is not a reliable error measure.

**Example 2.1.2 (predicting the work).** How many bisection steps give accuracy $10^{-4}$ on $[1,2]$? $n\ge\log_2(1/10^{-4}) = 13.29$, so $n=14$ — independent of $f$!

**Example 2.1.3.** Solve $e^x = 3x$ on $[0,1]$ to within $10^{-3}$. Let $f(x) = e^x-3x$: $f(0) = 1>0$, $f(1) = e-3 = -0.2817<0$.
$p_1 = 0.5\,(+0.1487)$, $p_2 = 0.75\,(-0.1330)$, $p_3 = 0.625\,(-0.00675)$, $p_4 = 0.5625\,(+0.0676)$, $p_5 = 0.59375\,(+0.0295)$, $p_6 = 0.609375\,(+0.0112)$, $p_7 = 0.6171875\,(+0.00215)$, $p_8 = 0.62109375\,(-0.00232)$, $p_9 = 0.619140625\,(-0.0000907)$, $p_{10} = 0.6181640625$.
Stop after $n=10$ ($2^{-10}<10^{-3}$): $p\approx0.618$ (true root $0.619061$).

**Example 2.1.4 ($\sqrt3$ by bisection).** $f(x) = x^2-3$ on $[1,2]$ with $\varepsilon = 10^{-4}$ needs 14 iterations and gives $p_{14} = 1.73199463$, error $5.6\times10^{-5}$. (Newton's method will need 4 iterations for 16 digits — §2.3.)

**Example 2.1.5 (a sign change is not a root).** $f(x) = \frac{1}{x-1}$ on $[0,2.5]$: $f(0) = -1<0<f(2.5)$. Bisection happily "converges" to $1.0000000056$ where $f = 1.8\times10^8$. The IVT hypothesis $f\in C[a,b]$ was violated (pole at $x=1$). *Always check the residual at the end.*

**Example 2.1.6 (a flat function fools the residual test).** $f(x) = (x-1)^{10}$ has $f(1.05) = 9.8\times10^{-14}$ and even $f(1.2) = 1.0\times10^{-7}$: a residual test with $\varepsilon = 10^{-6}$ would accept $x=1.2$, an error of $0.2$. Conversely, bisection cannot even start on this $f$: there is no sign change (even multiplicity).

---

## 2.2 Fixed-Point Iteration

A **fixed point** of $g$ is a number $p$ with $g(p) = p$. Root-finding and fixed-point problems are equivalent: $f(p)=0 \iff p = g(p)$ with e.g. $g(x) = x - f(x)$, or $g(x) = x - \phi(x)f(x)$ for any nonvanishing $\phi$.

**Theorem 2.2 (existence and uniqueness).**
(i) If $g\in C[a,b]$ and $g(x)\in[a,b]$ for all $x\in[a,b]$, then $g$ has at least one fixed point in $[a,b]$.
(ii) If in addition $g'$ exists on $(a,b)$ and $|g'(x)|\le k<1$ for all $x\in(a,b)$, the fixed point is unique.

*Proof.* (i) If $g(a)=a$ or $g(b)=b$ we are done. Otherwise $g(a)>a$ and $g(b)<b$, so $h(x)=g(x)-x$ has $h(a)>0>h(b)$ and the IVT gives a zero of $h$. (ii) If $p\ne q$ were both fixed points, by the MVT $|p-q| = |g(p)-g(q)| = |g'(\xi)||p-q|\le k|p-q|<|p-q|$, a contradiction. ∎

**Theorem 2.3 (Fixed-Point Theorem).** Under the hypotheses of Theorem 2.2(ii), for any $p_0\in[a,b]$ the sequence $p_n = g(p_{n-1})$ converges to the unique fixed point $p$, and
$$
|p_n - p|\le k^n\max\{p_0-a,\ b-p_0\},\qquad |p_n-p|\le\frac{k^n}{1-k}|p_1-p_0|.
$$
*Proof.* $|p_n-p| = |g(p_{n-1})-g(p)| = |g'(\xi_n)||p_{n-1}-p|\le k|p_{n-1}-p|\le\cdots\le k^n|p_0-p|$. For the second bound, $|p_{n+1}-p_n|\le k^n|p_1-p_0|$; summing the geometric series $|p_m-p_n|\le\sum_{j=n}^{m-1}k^j|p_1-p_0|$ and letting $m\to\infty$ gives the result. ∎

**Corollary (local convergence).** If $g\in C^1$ near $p$ and $|g'(p)|<1$, fixed-point iteration converges for all $p_0$ close enough to $p$, and
$$
\lim_{n\to\infty}\frac{|p_{n+1}-p|}{|p_n-p|} = |g'(p)|.
$$
If $|g'(p)|>1$ the iteration diverges from $p$ (except by luck).

### Examples for §2.2

**Example 2.2.1 (verifying the hypotheses).** $g(x) = \frac{x^2+2}{5}$ on $[0,1]$.
*Existence:* $g$ is increasing on $[0,1]$ with $g(0) = 0.4$, $g(1) = 0.6$, so $g([0,1])\subseteq[0.4,0.6]\subseteq[0,1]$.
*Uniqueness:* $|g'(x)| = 2x/5\le 0.4 = k<1$.
The fixed point solves $x^2-5x+2 = 0$: $p = \frac{5-\sqrt{17}}{2} = 0.4384471872$.
From $p_0 = 0$: $0.4,\ 0.432,\ 0.4373248,\ 0.4382506,\ 0.4384127,\ 0.4384411,\ 0.4384461,\ldots$ — each error is about $|g'(p)| = 0.175$ times the previous. Tolerance $10^{-8}$ is met after 11 iterations.

**Example 2.2.2 (hypotheses are sufficient, not necessary).** $g(x) = 3^{-x}$ on $[0,1]$ maps into $[\frac13,1]$, so a fixed point exists. But $g'(0) = -\ln3 = -1.0986$, so $|g'|\le k<1$ fails near $0$ and Theorem 2.3 does not apply directly. Nevertheless $g$ is decreasing, so the fixed point is unique, and at $p = 0.5478086$ we have $|g'(p)| = 0.602<1$: by the corollary the iteration converges (33 iterations to $10^{-8}$ from $p_0=0.5$).

**Example 2.2.3 (a-priori iteration count).** For Example 2.2.1, how many iterations guarantee $|p_n-p|<10^{-5}$?
$\frac{k^n}{1-k}|p_1-p_0| = \frac{0.4^n}{0.6}(0.4)<10^{-5} \iff n > \frac{\ln(1.5\times10^{-5})}{\ln0.4} = 12.1$, so $n = 13$. The actual error after 13 iterations is $3.1\times10^{-11}$: the bound uses the *worst* slope $k=0.4$, but near $p$ the slope is only $0.175$.

**Example 2.2.4 (rate = $|g'(p)|$).** $g(x) = \cos x$, $p_0 = 1$, converges to $p = 0.7390851$ (the *Dottie number*). $|g'(p)| = \sin p = 0.6736$, and the observed error ratios $e_{n+1}/e_n$ are $0.660, 0.683, 0.667, 0.678, 0.671$ → 0.674. Slow: 46 iterations for $10^{-8}$.

**Example 2.2.5 (same root, different $g$).** To compute $\sqrt a$ ($a=2$):
* $g_1(x) = a/x$: $g_1'(\sqrt a) = -1$, and the iteration **oscillates** forever: $1, 2, 1, 2, \ldots$
* $g_2(x) = \frac12\bigl(x + \frac ax\bigr)$: $g_2'(\sqrt a) = \frac12\bigl(1-\frac{a}{a}\bigr) = 0$ — convergence is faster than linear: $1,\ 1.5,\ 1.4166667,\ 1.4142157,\ 1.41421356237469,\ 1.41421356237310$.
When $g'(p) = 0$ the convergence is at least quadratic (Theorem 2.9). $g_2$ is Newton's method in disguise.

**Example 2.2.6 (choosing a convergent form).** For $x^3-2x-5 = 0$: $g_a(x) = (x^3-5)/2$ has $|g_a'(p)| = \frac32p^2 = 6.58$ (diverges); $g_c(x) = 5/(x^2-2)$ has $|g_c'(p)| = 3.68$ (diverges); $g_b(x) = (2x+5)^{1/3}$ has $|g_b'(p)| = \frac{2}{3p^2} = 0.152$ (converges, 10 iterations to $10^{-8}$).

---

## 2.3 Newton's Method and Its Extensions

### 2.3.1 Newton's (Newton–Raphson) method

Suppose $f\in C^2[a,b]$, $p_0\approx p$ with $f'(p_0)\ne0$. Taylor: $0 = f(p) = f(p_0) + (p-p_0)f'(p_0) + \frac{(p-p_0)^2}{2}f''(\xi)$. Dropping the (small) quadratic term and solving for $p$:
$$
\boxed{p_n = p_{n-1} - \frac{f(p_{n-1})}{f'(p_{n-1})}},\qquad n\ge1.
$$
Geometrically, $p_n$ is where the tangent line at $(p_{n-1}, f(p_{n-1}))$ crosses the $x$-axis.

**Theorem 2.6 (local convergence).** Let $f\in C^2[a,b]$ and $p\in(a,b)$ with $f(p)=0$, $f'(p)\neq0$. Then there is $\delta>0$ such that Newton's method converges to $p$ for every $p_0\in[p-\delta,p+\delta]$.
*Proof.* Newton is fixed-point iteration with $g(x) = x - f(x)/f'(x)$, and $g'(x) = \frac{f(x)f''(x)}{f'(x)^2}$, so $g'(p) = 0$. By continuity $|g'(x)|\le k<1$ on a neighbourhood of $p$, which $g$ maps into itself; apply Theorem 2.3. ∎

### 2.3.2 The secant method

Replace $f'(p_{n-1})$ by the slope of the secant through the last two iterates:
$$
p_n = p_{n-1} - \frac{f(p_{n-1})(p_{n-1}-p_{n-2})}{f(p_{n-1})-f(p_{n-2})}.
$$
One new function evaluation per step, no derivative. Order $\alpha = \frac{1+\sqrt5}{2}\approx1.618$ (§2.4).

### 2.3.3 The method of false position (Regula Falsi)

Use the secant formula but keep a bracket $[a_n,b_n]$ with $f(a_n)f(b_n)<0$ (like bisection). Always converges, but often only linearly, because one endpoint may stay fixed.

### Examples for §2.3

**Example 2.3.1 (Newton on the running example).** $p_0 = 2$:

| $n$ | $p_n$ | correct digits |
|---|---|---|
| 0 | 2.0000000000000000 | 1 |
| 1 | 2.1000000000000001 | 2 |
| 2 | 2.0945681211041851 | 5 |
| 3 | 2.0945514816981992 | 10 |
| 4 | 2.0945514815423265 | 16 |

The number of correct digits roughly doubles each step — the signature of **quadratic** convergence. Compare 20 bisection steps for 6 digits.

**Example 2.3.2 (Newton for $x = e^{-x}$).** $f(x) = x - e^{-x}$, $f'(x) = 1 + e^{-x}$, $p_0 = 1$:
$0.537882843,\ 0.566986991,\ 0.567143286,\ 0.567143290409784$ (converged in 4 steps; this is the *omega constant* $W(1)$).

**Example 2.3.3 (secant, same problem).** $p_0 = 0$, $p_1 = 1$:
$0.612699837,\ 0.563838389,\ 0.567170358,\ 0.567143307,\ 0.567143290409705,\ 0.567143290409784$ — 6 new function evaluations; Newton needed 4 steps = 8 evaluations ($f$ and $f'$).

**Example 2.3.4 (false position, same problem).** Bracket $[0,1]$:
$0.612699837,\ 0.572181412,\ 0.567703214,\ 0.567205553,\ 0.567150214,\ 0.567144060,\ 0.567143376,\ldots$ — the left endpoint $0$ never moves after the first step; error ratio $\approx 0.11$ (linear). 16 iterations are needed for $10^{-14}$.

**Example 2.3.5 (how Newton can fail).**
* *Cycling:* $f(x) = x^3-2x+2$, $p_0 = 0$: $p_1 = 0 - \frac{2}{-2} = 1$, $p_2 = 1 - \frac{1}{1} = 0$, … forever.
* *Divergence:* $f(x) = \arctan x$, root $0$. From $p_0 = 1.5$: $-1.694,\ 2.321,\ -5.114,\ 32.30,\ldots$ (diverges). From $p_0 = 1.3$: $-1.162,\ 0.859,\ -0.374,\ 0.034,\ -2.6\times10^{-5}$ (converges). The critical starting value is $|p_0| = 1.3917$, where $p_1 = -p_0$.
* *Remedy:* **safeguarding** — keep a bracket and fall back to bisection when the Newton step leaves it (this is the idea of Brent's method, `scipy.optimize.brentq`).

**Example 2.3.6 (division-free reciprocal).** To compute $1/a$ without dividing, apply Newton to $f(x) = \frac1x - a$: $p_n = p_{n-1} - \frac{1/p_{n-1}-a}{-1/p_{n-1}^2} = p_{n-1}(2 - ap_{n-1})$. For $a = 7$, $p_0 = 0.1$: $0.13,\ 0.1417,\ 0.14284777,\ 0.1428571422,\ 0.142857142857143$. This is how early computers (and some GPUs) implement division.

---

## 2.4 Error Analysis for Iterative Methods

**Definition 2.7 (order of convergence).** Suppose $p_n\to p$, $p_n\ne p$. If there are constants $\lambda>0$, $\alpha\ge1$ with
$$
\lim_{n\to\infty}\frac{|p_{n+1}-p|}{|p_n-p|^\alpha} = \lambda,
$$
then $\{p_n\}$ converges to $p$ **of order $\alpha$** with **asymptotic error constant $\lambda$**. $\alpha = 1$ (with $\lambda<1$): linear; $\alpha = 2$: quadratic.

**Theorem 2.8.** If $g\in C[a,b]$ maps $[a,b]$ into itself, $g'$ is continuous on $(a,b)$, $|g'(x)|\le k<1$, and $g'(p)\ne0$, then fixed-point iteration converges **only linearly** with $\lambda = |g'(p)|$.

**Theorem 2.9.** If $g(p) = p$, $g'(p) = 0$ and $g''$ is continuous with $|g''|<M$ near $p$, then fixed-point iteration converges at least quadratically for $p_0$ near $p$, and $|p_{n+1}-p|<\frac M2|p_n-p|^2$.
*Proof.* Taylor about $p$: $p_{n+1} = g(p_n) = p + 0 + \frac{g''(\xi_n)}{2}(p_n-p)^2$. ∎

For Newton's method this gives $\displaystyle\lim_{n\to\infty}\frac{|p_{n+1}-p|}{|p_n-p|^2} = \frac{|f''(p)|}{2|f'(p)|}$.

**Multiple roots.** $p$ is a zero of **multiplicity $m$** if $f(x) = (x-p)^mq(x)$ with $q(p)\ne0$. Equivalently (**Theorem 2.12**) $f(p)=f'(p)=\dots=f^{(m-1)}(p) = 0\ne f^{(m)}(p)$. $p$ is a **simple** zero iff $f(p)=0\ne f'(p)$. At a multiple root Newton's method converges only linearly (with $\lambda = 1-1/m$). Two fixes:
* if $m$ is known: $p_n = p_{n-1} - m\,f(p_{n-1})/f'(p_{n-1})$;
* in general apply Newton to $\mu(x) = f(x)/f'(x)$, which has a *simple* zero at $p$:
$$
p_n = p_{n-1} - \frac{f(p_{n-1})f'(p_{n-1})}{[f'(p_{n-1})]^2 - f(p_{n-1})f''(p_{n-1})}.
$$

### Examples for §2.4

**Example 2.4.1 (linear vs quadratic).** $p_n = 0.5^n$ (linear, $\lambda = 0.5$) versus $\tilde p_n = 0.5^{2^n}$ (quadratic, $\lambda=1$):

| $n$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| $p_n$ | $0.5$ | $0.25$ | $0.125$ | $0.0625$ | $0.0313$ | $0.0156$ | $7.8\times10^{-3}$ |
| $\tilde p_n$ | $0.25$ | $0.0625$ | $3.9\times10^{-3}$ | $1.5\times10^{-5}$ | $2.3\times10^{-10}$ | $5.4\times10^{-20}$ | $2.9\times10^{-39}$ |

To reach $10^{-38}$ the linear sequence needs 127 terms, the quadratic one 7.

**Example 2.4.2 (Newton at a double root).** $f(x) = e^x-x-1$ has $f(0) = f'(0) = 0$, $f''(0) = 1$: a double root. Newton from $p_0=1$: $0.58198,\ 0.31906,\ 0.16800,\ 0.08635,\ 0.04380,\ldots$ — errors halve each step ($\lambda = 1-\frac12$), 27 iterations for $10^{-8}$. The modified method with $\mu = f/f'$: $-0.23421,\ -0.0084583,\ -1.19\times10^{-5},\ -4.2\times10^{-11}$ — quadratic again.

**Example 2.4.3 (known multiplicity).** For the same $f$ with $m=2$: $p_n = p_{n-1} - 2f/f'$ gives $0.16395,\ 0.0044781,\ 3.34\times10^{-6},\ 1.09\times10^{-11}$. (The last value stagnates at $\sim10^{-11}$: near a double root $f$ is computed with absolute error $\sim u$, so only about $\sqrt u\approx10^{-8}$ relative accuracy in $p$ is meaningful — Chapter 1's conditioning at work.)

**Example 2.4.4 (order of the secant method).** Errors for the running example from $p_0=2$, $p_1=3$: $9.1\times10^{-1},\ 3.6\times10^{-2},\ 1.3\times10^{-2},\ 2.7\times10^{-4},\ 2.1\times10^{-6},\ 3.2\times10^{-10},\ 4.4\times10^{-16}$. The estimates $\alpha\approx\frac{\ln(e_{n+1}/e_n)}{\ln(e_n/e_{n-1})}$ in the asymptotic regime are $1.26,\ 1.80,\ 1.53$ → consistent with $\alpha = 1.618$. (Theory: $e_{n+1}\approx C e_ne_{n-1}$; trying $e_{n+1}\sim e_n^\alpha$ gives $\alpha^2 = \alpha+1$.)

**Example 2.4.5 (checking the asymptotic constant).** Newton for $x^2 - 5$ from $p_0 = 2$: the ratios $e_{n+1}/e_n^2$ are $0.2500,\ 0.2222,\ 0.22360$, approaching $\frac{|f''(p)|}{2|f'(p)|} = \frac{2}{2\cdot2\sqrt5} = 0.223607$. ✓

---

## 2.5 Accelerating Convergence

**Aitken's $\Delta^2$ method.** If $p_n\to p$ linearly, then for large $n$, $\frac{p_{n+1}-p}{p_n-p}\approx\frac{p_{n+2}-p}{p_{n+1}-p}$. Solving for $p$:
$$
\hat p_n = p_n - \frac{(p_{n+1}-p_n)^2}{p_{n+2}-2p_{n+1}+p_n} = p_n - \frac{(\Delta p_n)^2}{\Delta^2p_n},
$$
with forward differences $\Delta p_n = p_{n+1}-p_n$, $\Delta^2p_n = \Delta(\Delta p_n)$.

**Theorem 2.14.** If $p_n\to p$ linearly with $\lim\frac{p_{n+1}-p}{p_n-p}<1$, then $\hat p_n\to p$ **faster**: $\lim\frac{\hat p_n-p}{p_n-p} = 0$.

**Steffensen's method** applies Aitken's formula to a fixed-point iteration and *restarts* from the accelerated value: from $p_0^{(k)}$ compute $p_1 = g(p_0)$, $p_2 = g(p_1)$, then $p_0^{(k+1)} = \hat p_0$. If $g'(p)\neq1$, Steffensen converges **quadratically** without derivatives (Theorem 2.15).

### Examples for §2.5

**Example 2.5.1 (Aitken on $\cos$ iteration).** $p_n = \cos p_{n-1}$, $p_0 = 1$:

| $n$ | $p_n$ | error | $\hat p_n$ | error |
|---|---|---|---|---|
| 0 | 1.000000000 | $2.6\times10^{-1}$ | 0.728010361 | $1.1\times10^{-2}$ |
| 2 | 0.857553216 | $1.2\times10^{-1}$ | 0.736906294 | $2.2\times10^{-3}$ |
| 4 | 0.793480359 | $5.4\times10^{-2}$ | 0.738636097 | $4.5\times10^{-4}$ |
| 6 | 0.763959683 | $2.5\times10^{-2}$ | 0.738992243 | $9.3\times10^{-5}$ |

Aitken gains two to three digits for free from the same data.

**Example 2.5.2 (series acceleration).** The Leibniz series $\pi = 4\sum_{k\ge0}\frac{(-1)^k}{2k+1}$ converges painfully slowly: 10 terms give $3.0418$. Aitken on the partial sums gives $3.14125$ and applying Aitken again gives $3.14158732$ (error $6\times10^{-6}$), from the same ten terms. (Repeated Aitken ≈ Shanks transformation; the same idea underlies Richardson/Romberg in Chapter 4.)

**Example 2.5.3 (Steffensen for $\cos x = x$).** $p_0 = 1$: $0.7280103615,\ 0.7390669669,\ 0.7390851332,\ 0.7390851332151607$ — 4 Steffensen steps (12 evaluations of $\cos$) versus 46 plain iterations.

**Example 2.5.4 (Steffensen on a fast iteration).** $g(x) = (2x+5)^{1/3}$, $p_0 = 2$: $2.0945695285,\ 2.0945514815430,\ 2.094551481542327$ — machine precision in 3 steps.

**Example 2.5.5 (Aitken on $x = e^{-x}$).** Plain errors: $6.7\times10^{-2},\ 3.9\times10^{-2},\ 2.2\times10^{-2},\ 1.3\times10^{-2},\ldots$ (ratio $|g'(p)| = e^{-p} = p = 0.567$). Aitken errors: $4.8\times10^{-4},\ 1.6\times10^{-4},\ 5.0\times10^{-5},\ 1.6\times10^{-5},\ldots$ — about 100× smaller at each index.

---

## 2.6 Zeros of Polynomials and Müller's Method

A polynomial $P(x) = a_nx^n+\dots+a_1x+a_0$ ($a_n\ne0$) has exactly $n$ complex zeros counted with multiplicity (Fundamental Theorem of Algebra). All zeros satisfy the **Cauchy bound** $|z|\le 1 + \max_{i<n}|a_i/a_n|$.

**Horner's method (synthetic division, Theorem 2.19).** Let $b_n = a_n$ and $b_k = a_k + b_{k+1}x_0$ for $k = n-1,\dots,0$. Then $b_0 = P(x_0)$ and
$$
P(x) = (x-x_0)Q(x) + b_0,\qquad Q(x) = b_nx^{n-1}+b_{n-1}x^{n-2}+\dots+b_1 .
$$
Differentiating, $P'(x_0) = Q(x_0)$ — so a second Horner pass on the $b_k$ gives $P'(x_0)$. Newton's method then costs $2n$ multiplications per step.

**Deflation.** Once an approximate root $\hat x_1$ is found, continue with $Q_1(x) = P(x)/(x-\hat x_1)$, and so on. Errors accumulate, so polish each root with Newton on the *original* $P$, and deflate roots of smallest magnitude first.

**Müller's method.** Fit a parabola through $(p_0,f(p_0)), (p_1,f(p_1)), (p_2,f(p_2))$ and take its root nearest $p_2$:
$$
p_3 = p_2 - \frac{2c}{b + \operatorname{sgn}(b)\sqrt{b^2-4ac}},
$$
where $c = f(p_2)$, and $a, b$ come from divided differences. The square root may be complex, so Müller finds **complex roots from real starting values**. Order $\approx1.84$.

### Examples for §2.6

**Example 2.6.1 (Horner by hand).** $P(x) = x^4 - 2x^3 - 5x^2 + 6x + 3$ at $x_0 = 2$:

| | $a_4 = 1$ | $a_3 = -2$ | $a_2 = -5$ | $a_1 = 6$ | $a_0 = 3$ |
|---|---|---|---|---|---|
| $b_{k+1}x_0$ | | 2 | 0 | −10 | −8 |
| $b_k$ | 1 | 0 | −5 | −4 | **−5** = $P(2)$ |
| $c_{k+1}x_0$ | | 2 | 4 | −2 | |
| $c_k$ | 1 | 2 | −1 | **−6** = $P'(2)$ | |

So $P(2) = -5$, $P'(2) = -6$ and $P(x) = (x-2)(x^3 + 0x^2 - 5x - 4) - 5$.

**Example 2.6.2 (Newton–Horner).** From $x_0 = 2$ the first step is $2 - (-5)/(-6) = 1.1667$, then $1.41677,\ 1.394797,\ 1.3947124,\ 1.394712387986677$.

**Example 2.6.3 (deflation for all real roots).** Continuing with the deflated cubic, quadratic and linear factors gives the four roots $1.394712388,\ -0.394712388,\ -1.887360413,\ 2.887360413$, which agree with `np.roots` to 15 digits. (Note the symmetry about $x = 1/2$: substituting $x = y+\frac12$ gives an even polynomial in $y$.)

**Example 2.6.4 (Müller finds complex roots).** $P(x) = x^3+2x^2+3x+4$ from $p_0,p_1,p_2 = 0.5, 1, 1.5$ (all real):
$-0.025-0.9744i,\ -0.2067-1.2952i,\ -0.2195-1.5276i,\ -0.17440-1.54796i,\ -0.174687-1.546867i,\ -0.1746854-1.5468689i$.
The roots are $-1.65062919$ and $-0.1746854\pm1.5468689i$; all lie inside the Cauchy disc $|z|\le 1+4 = 5$.

**Example 2.6.5 (roots as eigenvalues).** The **companion matrix** of the monic $x^3+2x^2+3x+4$,
$$
C = \begin{pmatrix}-2&-3&-4\\1&0&0\\0&1&0\end{pmatrix},
$$
has characteristic polynomial $P$, so its eigenvalues are the roots: $-1.65062919,\ -0.1746854\pm1.54686889i$. This is exactly how `np.roots` works — with the QR algorithm of Chapter 9.

---

## 2.7 Root Finding in Statistics and Data Science *(data-science extension)*

| Task | Equation $f(\theta)=0$ | Recommended method |
|---|---|---|
| Maximum likelihood, 1 parameter | score $\ell'(\theta) = 0$ | Newton (Newton–Raphson / Fisher scoring) |
| Quantile of a distribution | $F(x) - q = 0$ | Newton with $F' = $ density; bracketed if tails heavy |
| Internal rate of return | $\text{NPV}(r) = 0$ | secant / Brent |
| Calibrate a model to a target rate | $\bar\sigma(z+t) - \pi = 0$ | Newton (monotone) |
| Hyperparameter matching (e.g. perplexity in t-SNE) | $H(\beta) - \log_2 \text{Perp} = 0$ | bisection (t-SNE uses it for every point) |

### Examples for §2.7

**Example 2.7.1 (quantile).** $z_{0.975}$ of $N(0,1)$: Newton on $\Phi(z)-0.975$ with $\Phi' = \varphi$ from $z_0 = 2$: $1.9583288,\ 1.9599614,\ 1.95996398454$ — the familiar $1.96$ in three steps.

**Example 2.7.2 (Gamma MLE).** For i.i.d. Gamma$(\alpha,\theta)$ data, $\hat\alpha$ solves $\ln\alpha - \psi(\alpha) = s := \ln\bar x - \overline{\ln x}$ ($\psi$ = digamma). For the data $2.3, 1.1, 4.5, 3.2, 0.8, 2.9, 5.1, 1.7, 2.2, 3.6$: $s = 0.141330$. Starting from Minka's approximation $\alpha_0 = \frac{3-s+\sqrt{(s-3)^2+24s}}{12s} = 3.69067$, Newton (with $\psi'$ = trigamma) gives $3.6961941,\ 3.6962026953,\ 3.6962026953456$. Hence $\hat\alpha = 3.6962$ and $\hat\theta = \bar x/\hat\alpha = 0.7413$.

**Example 2.7.3 (IRR).** Cash flows $-1000, 300, 400, 500$: $\text{NPV}(r) = -1000+\frac{300}{1+r}+\frac{400}{(1+r)^2}+\frac{500}{(1+r)^3}$, with $\text{NPV}(0) = 200$, $\text{NPV}(0.2) = -182.87$. Secant from $0, 0.2$: $0.10447,\ 0.08620,\ 0.08903,\ 0.0889637,\ 0.08896339$. **IRR = 8.896%.**

**Example 2.7.4 (Cauchy location — Newton is only local).** For Cauchy$(\theta,1)$ data $-1.2, 0.3, 0.9, 1.4, 1.8, 2.1, 9.5$ the score is $\ell'(\theta) = \sum_i\frac{2(x_i-\theta)}{1+(x_i-\theta)^2}$. Newton from the sample **median** $1.4$ converges to $\hat\theta = 1.260170$ in 5 iterations ($\ell''<0$: a maximum), and from the mean $2.114$ in 6. From $\theta_0 = 6$ Newton runs away: $9.14,\ 8.46,\ 10.84,\ 17.5,\ 30.7,\ 58.4,\ldots$, because $\ell'(\theta)\to0$ as $\theta\to\infty$ (the outlier $9.5$ creates a flat region). *Lesson: robust starting values matter.*

**Example 2.7.5 (calibrating a classifier's base rate).** A model's logits $z_i$ ($2000$ scores) give an average predicted probability $\overline{\sigma(z)} = 0.3232$, but the known population rate is $0.30$. Find the intercept shift $t$ with $F(t) = \frac1N\sum\sigma(z_i+t) - 0.30 = 0$. Since $F'(t) = \frac1N\sum\sigma(1-\sigma)>0$, $F$ is increasing and Newton from $t_0 = 0$ gives $-0.147235,\ -0.149718,\ -0.1497187457$. After the shift the average prediction is exactly $0.3000$.

---

## Chapter summary

| Method | Needs | Order | Guaranteed? | Cost / step |
|---|---|---|---|---|
| Bisection | bracket | 1 ($\lambda = \frac12$) | yes | 1 $f$ |
| Fixed point | $\lvert g'(p)\rvert<1$ | 1 ($\lambda = \lvert g'(p)\rvert$) | locally | 1 $g$ |
| Newton | $f'$, good start | 2 (simple root) | locally | $f$, $f'$ |
| Secant | 2 starts | 1.618 | locally | 1 $f$ |
| False position | bracket | 1 | yes | 1 $f$ |
| Steffensen | $g$ | 2 | locally | 2 $g$ |
| Müller | 3 starts | 1.84, complex roots | locally | 1 $f$ |

## Further reading

Burden & Faires Ch. 2 · Brent, *Algorithms for Minimization without Derivatives* (1973) · SciPy docs for `scipy.optimize.root_scalar`.
