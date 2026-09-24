# Chapter 1 — Mathematical Preliminaries and Error Analysis

> **Reference:** Burden & Faires, *Numerical Analysis*, Chapter 1 (§1.1–1.3), plus two sections for data science (§1.4–1.5).
> **Code:** [`numlib/floating.py`](../numlib/floating.py) · **Every number in this chapter is reproduced by** [`lectures/code/ch01.py`](code/ch01.py)
> **Worked problems:** [`examples/ch01`](../examples/ch01_examples.md) · **Homework:** [`exercises/ch01`](../exercises/ch01_exercises.md)

## Learning outcomes

After this chapter you should be able to

1. state and apply the Intermediate Value, Mean Value, Extreme Value and Taylor theorems, and bound a Taylor remainder;
2. describe the IEEE-754 floating-point system, compute $k$-digit chopped and rounded representations, and evaluate absolute and relative errors;
3. identify loss of significance and rewrite formulas to avoid it (rationalisation, nested multiplication);
4. distinguish linear from exponential error growth, and stable from unstable algorithms;
5. use big-$O$ notation for rates of convergence of sequences and of functions of $h$;
6. compute the condition number of a problem and relate forward error, backward error and conditioning;
7. recognise and fix floating-point failures in data-science code (sums, variance, softmax, likelihoods).

---

## 1.1 Review of Calculus

Numerical analysis constantly *replaces* a hard problem by an easier one (a function by a polynomial, a derivative by a difference quotient, an integral by a sum) and then *bounds the error* of the replacement. The tools for bounding errors are a handful of classical theorems.

### 1.1.1 Limits and continuity

**Definition 1.1 (limit, continuity).** $\lim_{x\to x_0}f(x) = L$ if for every $\varepsilon>0$ there is $\delta>0$ such that $0<|x-x_0|<\delta \Rightarrow |f(x)-L|<\varepsilon$. $f$ is *continuous at $x_0$* if $\lim_{x\to x_0}f(x) = f(x_0)$. $C[a,b]$ denotes the set of functions continuous on $[a,b]$, and $C^n[a,b]$ those with $n$ continuous derivatives.

**Definition 1.2 (convergent sequence).** $\{x_n\}$ converges to $x$ if $\forall\varepsilon>0\ \exists N$ with $|x_n - x|<\varepsilon$ for all $n>N$.

**Theorem 1.3.** For $f$ defined on a set $X$ and $x_0\in X$, the following are equivalent: (a) $f$ is continuous at $x_0$; (b) for every sequence $x_n\to x_0$ in $X$, $f(x_n)\to f(x_0)$.

Theorem 1.3 is what lets us pass limits through continuous functions — e.g. if a fixed-point iteration $x_{n+1}=g(x_n)$ converges to $p$ and $g$ is continuous, then $p = \lim x_{n+1} = \lim g(x_n) = g(p)$.

### 1.1.2 The key theorems

**Theorem 1.4 (Intermediate Value Theorem, IVT).** If $f\in C[a,b]$ and $K$ lies between $f(a)$ and $f(b)$, then there exists $c\in(a,b)$ with $f(c)=K$.

**Theorem 1.5 (Rolle).** If $f\in C[a,b]$ is differentiable on $(a,b)$ and $f(a)=f(b)$, then $f'(c)=0$ for some $c\in(a,b)$.
*Generalised Rolle:* if $f\in C^n[a,b]$ vanishes at $n+1$ distinct points of $[a,b]$, then $f^{(n)}(c)=0$ for some $c\in(a,b)$. (This is the heart of the interpolation error formula in Chapter 3.)

**Theorem 1.6 (Mean Value Theorem, MVT).** If $f\in C[a,b]$ is differentiable on $(a,b)$, then there is $c\in(a,b)$ with
$$
f'(c) = \frac{f(b)-f(a)}{b-a}.
$$

**Theorem 1.7 (Extreme Value Theorem).** If $f\in C[a,b]$ then $f$ attains a minimum and a maximum on $[a,b]$. If $f$ is also differentiable, the extrema occur at the endpoints or at critical points ($f'=0$).

**Theorem 1.8 (Weighted Mean Value Theorem for Integrals).** If $f\in C[a,b]$, $g$ is integrable and does not change sign on $[a,b]$, then there is $c\in(a,b)$ with
$$
\int_a^b f(x)g(x)\,dx = f(c)\int_a^b g(x)\,dx.
$$
(With $g\equiv1$ this says that $f(c)$ equals the *average value* of $f$.) This theorem is used to simplify every quadrature error term in Chapter 4.

**Theorem 1.9 (Taylor's Theorem).** Let $f\in C^n[a,b]$, $f^{(n+1)}$ exist on $[a,b]$, and $x_0\in[a,b]$. For every $x\in[a,b]$ there is $\xi(x)$ between $x_0$ and $x$ with
$$
f(x) = P_n(x) + R_n(x),\qquad
P_n(x) = \sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k,\qquad
R_n(x) = \frac{f^{(n+1)}(\xi(x))}{(n+1)!}(x-x_0)^{n+1}.
$$
$P_n$ is the **$n$th Taylor polynomial** and $R_n$ the **remainder (truncation error)**. With $x_0 = 0$ it is called a Maclaurin polynomial.

*Proof sketch.* Fix $x$ and define $g(t) = f(t) - P_n(t) - [f(x)-P_n(x)]\frac{(t-x_0)^{n+1}}{(x-x_0)^{n+1}}$. Then $g$ vanishes at $t=x$ and to order $n+1$ at $t=x_0$; applying Rolle's theorem $n+1$ times gives $\xi$ with $g^{(n+1)}(\xi)=0$, which rearranges to the stated remainder. ∎

> **Why a data scientist cares.** Taylor's theorem is the justification of: gradient descent (first-order model), Newton's method (second-order model), the delta method in statistics ($\operatorname{Var} g(\hat\theta)\approx g'(\theta)^2\operatorname{Var}\hat\theta$), Laplace approximations, and every finite-difference or quadrature formula in this course.

### Examples for §1.1

**Example 1.1.1 (IVT — locating a root).** Show that $x^5 - 2x^3 + 3x^2 - 1 = 0$ has a solution in $[0,1]$, and narrow it down.

*Solution.* $f(x) = x^5-2x^3+3x^2-1$ is a polynomial, hence continuous. $f(0) = -1 < 0$ and $f(1) = 1 > 0$, so by the IVT with $K=0$ there is $c\in(0,1)$ with $f(c)=0$. Evaluating further, $f(0.5) = -0.46875 < 0$ and $f(0.75) = 0.0810546875 > 0$, so a root lies in $(0.5, 0.75)$. Repeating this halving is exactly the **bisection method** (Chapter 2).

**Example 1.1.2 (MVT).** Find the number $c$ guaranteed by the MVT for $f(x) = x^3$ on $[0,2]$.

*Solution.* $\dfrac{f(2)-f(0)}{2-0} = \dfrac{8}{2} = 4$, and $f'(c) = 3c^2 = 4 \Rightarrow c = 2/\sqrt3 \approx 1.1547 \in (0,2)$.
*Interpretation:* the MVT also gives the bound $|f(b)-f(a)| \le \max|f'|\,|b-a|$, which is how we will prove that fixed-point iterations contract (Theorem 2.3).

**Example 1.1.3 (Extreme values).** Find $\max_{0\le x\le1}|g(x)|$ for $g(x) = 2 - e^x + 2x$.

*Solution.* $g'(x) = -e^x + 2 = 0 \iff x = \ln 2$. Compare critical point and endpoints:
$g(0) = 1$, $g(\ln 2) = 2 - 2 + 2\ln2 = 1.386294$, $g(1) = 4 - e = 1.281718$.
All values are positive, so $\max|g| = 2\ln 2 \approx 1.3863$, attained at $x=\ln2$.
*Numerical-analysis use:* error bounds contain $\max|f^{(n)}|$; finding it is always an extreme-value problem.

**Example 1.1.4 (Taylor polynomial and remainder).** Approximate $\cos(0.01)$ with the second Maclaurin polynomial and bound the error.

*Solution.* $\cos x = 1 - \frac{x^2}{2} + R$. Because the $x^3$ coefficient of the Maclaurin series is zero, $P_2 = P_3$, so we may use the **sharper** remainder $R_3(x) = \frac{\cos\xi}{24}x^4$:
$$
P_2(0.01) = 1 - 0.00005 = 0.99995,\qquad |R_3(0.01)| \le \frac{(0.01)^4}{24} = 4.1667\times10^{-10}.
$$
The actual error is $\cos(0.01) - 0.99995 = 4.16665\times10^{-10}$ — the bound is essentially sharp. (Using $R_2 = \frac{\sin\xi}{6}x^3$ would only give $|R_2|\le \frac{\sin(0.01)}{6}(0.01)^3 \approx 1.7\times 10^{-9}$, a bound four times weaker.)

**Example 1.1.5 (Integrating a Taylor polynomial).** Approximate $\int_0^{0.1}\cos x\,dx$ and bound the error.

*Solution.* $\int_0^{0.1}\bigl(1 - \frac{x^2}{2}\bigr)dx = 0.1 - \frac{(0.1)^3}{6} = 0.0998333$. The error is
$$
\left|\int_0^{0.1}R_3(x)\,dx\right| \le \int_0^{0.1}\frac{x^4}{24}dx = \frac{(0.1)^5}{120} = 8.33\times10^{-8}.
$$
The exact value $\sin 0.1 = 0.0998334166$ differs by $8.331\times10^{-8}$. ✓

**Example 1.1.6 (How many terms?).** How many terms of $e = \sum_{k\ge0}1/k!$ are needed for an error below $10^{-6}$?

*Solution.* Taylor with $f(x)=e^x$, $x_0=0$, $x=1$: $|R_n(1)| = \frac{e^{\xi}}{(n+1)!} \le \frac{3}{(n+1)!}$ since $e^\xi < e < 3$.

| $n$ | bound $3/(n+1)!$ | actual error |
|---|---|---|
| 6 | $5.95\times10^{-4}$ | $2.26\times10^{-4}$ |
| 7 | $7.44\times10^{-5}$ | $2.79\times10^{-5}$ |
| 8 | $8.27\times10^{-6}$ | $3.06\times10^{-6}$ |
| 9 | $8.27\times10^{-7}$ | $3.03\times10^{-7}$ |

So $n = 9$ (ten terms) suffices. Note the bound is pessimistic by a factor $\approx e$ — typical of error bounds.

**Example 1.1.7 (Weighted MVT).** Show that $\int_0^1 x e^{-x^2}dx = e^{-c^2}\cdot\frac12$ for some $c\in(0,1)$ and find $c$.

*Solution.* $g(x)=x\ge0$ on $[0,1]$, $f(x) = e^{-x^2}$ continuous, so Theorem 1.8 gives $\int_0^1 xe^{-x^2}dx = e^{-c^2}\int_0^1x\,dx = \tfrac12e^{-c^2}$. The integral equals $\frac{1-e^{-1}}{2} = 0.316060$, so $e^{-c^2} = 1-e^{-1}$, $c = \sqrt{-\ln(1-e^{-1})} = 0.6773\in(0,1)$. ✓

---

## 1.2 Round-off Errors and Computer Arithmetic

### 1.2.1 Binary machine numbers (IEEE-754)

A 64-bit **double** stores a sign bit $s$, an 11-bit exponent $c$ and a 52-bit fraction (mantissa) $f$, representing
$$
x = (-1)^s\, 2^{c-1023}\,(1+f),\qquad 1\le c\le 2046 .
$$

| Format | bits | fraction bits | exponent range | $\varepsilon_{\text{mach}}$ | ≈ decimal digits | largest |
|---|---|---|---|---|---|---|
| `float16` | 16 | 10 | $[-14,15]$ | $9.8\times10^{-4}$ | 3 | $6.6\times10^4$ |
| `float32` | 32 | 23 | $[-126,127]$ | $1.19\times10^{-7}$ | 7 | $3.4\times10^{38}$ |
| `float64` | 64 | 52 | $[-1022,1023]$ | $2.22\times10^{-16}$ | 16 | $1.8\times10^{308}$ |

Numbers smaller than about $2.2\times10^{-308}$ in magnitude **underflow** (become subnormal and finally $0$); larger than $1.8\times10^{308}$ **overflow** to `inf`. The gap between $1$ and the next larger double is **machine epsilon** $\varepsilon_{\text{mach}} = 2^{-52}$; the **unit round-off** is $u = 2^{-53}\approx1.1\times10^{-16}$.

### 1.2.2 Decimal machine numbers: chopping and rounding

For hand analysis we use $k$-digit decimal arithmetic. Any real $y = \pm0.d_1d_2\ldots d_kd_{k+1}\ldots\times10^n$ ($d_1\ne0$) has
* **chopping:** $fl(y) = \pm0.d_1\ldots d_k\times10^n$;
* **rounding:** add $5\times10^{n-(k+1)}$ then chop.

**Definition 1.10.** If $p^*$ approximates $p$, the **absolute error** is $|p-p^*|$ and the **relative error** is $|p-p^*|/|p|$ ($p\ne0$). $p^*$ approximates $p$ to **$t$ significant digits** if $t$ is the largest non-negative integer with
$$
\frac{|p-p^*|}{|p|} \le 5\times10^{-t}.
$$

**Theorem 1.11 (relative error of $fl$).** For $k$-digit arithmetic,
$$
\left|\frac{y - fl(y)}{y}\right| \le \begin{cases}10^{-k+1} & \text{chopping}\\ 0.5\times10^{-k+1} & \text{rounding.}\end{cases}
$$
The bound is independent of the size of $y$ — **floating point controls relative, not absolute, error**.

### 1.2.3 Finite-digit arithmetic

We model a machine operation $\odot$ as $x\odot y = fl\bigl(fl(x)\circ fl(y)\bigr)$: round the operands, compute exactly, round the result. Two dangerous situations:

1. **Subtraction of nearly equal numbers** (loss of significance / catastrophic cancellation): the leading digits cancel and only the (already-rounded) trailing digits remain.
2. **Division by a small number** or multiplication by a large one: absolute errors are magnified.

*Remedies:* reformulate (rationalise, use identities), and evaluate polynomials in **nested (Horner) form**
$$
a_nx^n+\dots+a_1x+a_0 = \bigl(\cdots((a_nx + a_{n-1})x + a_{n-2})x+\cdots\bigr)x + a_0,
$$
which needs $n$ multiplications and $n$ additions and usually accumulates less error.

### Examples for §1.2

**Example 1.2.1 (Decoding a double).** What number does `0 10000000010 0110 0000…0` represent, and what is the next larger machine number?

*Solution.* $s=0$; $c = 10000000010_2 = 1024+2 = 1026$; $f = 0.0110_2 = \frac14+\frac18 = 0.375$. Hence $x = 2^{1026-1023}(1.375) = 8\times1.375 = 11$.
Changing the last fraction bit adds $2^{3}\cdot2^{-52} = 2^{-49}\approx1.78\times10^{-15}$: the next double is $11 + 1.78\times10^{-15}$. Every real number in between must be represented by one of these two.

**Example 1.2.2 (Chopping vs rounding).** Five-digit representations of $\pi = 3.14159265\ldots$.

| | $fl(\pi)$ | absolute error | relative error |
|---|---|---|---|
| chopping | $0.31415\times10^1$ | $9.27\times10^{-5}$ | $2.95\times10^{-5}\le 10^{-4}$ |
| rounding | $0.31416\times10^1$ | $7.35\times10^{-6}$ | $2.34\times10^{-6}\le 0.5\times10^{-4}$ |

Both respect Theorem 1.11. Rounding gives 5 significant digits ($2.34\times10^{-6} \le 5\times10^{-6}$); chopping gives only 4.

**Example 1.2.3 (Five-digit rounding arithmetic).** Let $x = 2/3$, $y = 3/11$. Then $fl(x) = 0.66667$, $fl(y) = 0.27273$.

| operation | exact | 5-digit result | absolute error | relative error |
|---|---|---|---|---|
| $x\oplus y$ | 0.9393939394 | 0.93940 | $6.06\times10^{-6}$ | $6.45\times10^{-6}$ |
| $x\ominus y$ | 0.3939393939 | 0.39394 | $6.06\times10^{-7}$ | $1.54\times10^{-6}$ |
| $x\otimes y$ | 0.1818181818 | 0.18182 | $1.82\times10^{-6}$ | $1.00\times10^{-5}$ |
| $x\oslash y$ | 2.4444444444 | 2.4444 | $4.44\times10^{-5}$ | $1.82\times10^{-5}$ |

All relative errors are of order $10^{-5}$, as expected. Now take $u = 0.66666$ (close to $x$): $x - u = 6.6667\times10^{-6}$ exactly, but $fl(x)\ominus u = 0.66667 - 0.66666 = 1.0000\times10^{-5}$ — a **50% relative error**. Only one significant digit survived the cancellation.

**Example 1.2.4 (Loss of significance in the quadratic formula).** Solve $x^2 + 97.43x + 1 = 0$ with four-digit rounding. Exact roots: $x_1 = -0.01026486$, $x_2 = -97.41974$.

*Solution.* $b^2 = 9492.6 \to 9493$, $b^2-4 = 9489$, $\sqrt{9489} = 97.411\to 97.41$.
* Standard formula: $x_1 = \dfrac{-97.43 + 97.41}{2} = -0.01000$ — relative error $2.6\times10^{-2}$ (two digits lost to cancellation).
* Rationalised formula (multiply numerator and denominator by $-b-\sqrt{b^2-4ac}$):
$$
x_1 = \frac{-2c}{b + \sqrt{b^2-4ac}} = \frac{-2}{97.43+97.41} = \frac{-2}{194.8} = -0.01027,
$$
relative error $5.0\times10^{-4}$ — the best possible with four digits.
* For $x_2$ the standard formula adds two numbers of the same sign and is accurate: $x_2 = -97.42$ (relative error $2.0\times10^{-4}$).
*Rule:* compute the root with $-b$ and $-\operatorname{sign}(b)\sqrt{\cdot}$ of the **same sign** by the standard formula, and the other by $x_1x_2 = c/a$.

**Example 1.2.5 (Nested arithmetic).** Evaluate $f(x) = x^3 - 5.2x^2 + 2.7x + 1.4$ at $x=3.83$ using three-digit rounding. Exact: $f(3.83) = -8.355393$.

*Direct:* $x^2 = 14.6689\to14.7$; $x^3 = 14.7\times3.83 = 56.301\to56.3$; $5.2x^2 = 76.44\to76.4$; $2.7x = 10.341\to10.3$.
$f \approx 56.3 - 76.4 + 10.3 + 1.4 = -8.4$ (relative error $5.3\times10^{-3}$).

*Nested:* $f(x) = ((x - 5.2)x + 2.7)x + 1.4$:
$x-5.2 = -1.37$; $\times3.83 = -5.2471\to-5.25$; $+2.7 = -2.55$; $\times3.83 = -9.7665\to-9.77$; $+1.4 = -8.37$.
Relative error $1.7\times10^{-3}$ — three times smaller, and with 3 multiplications instead of 5.

**Example 1.2.6 (Significant digits).** For what values $p^*$ does $p^*$ approximate $p = 1000$ to four significant digits?

*Solution.* $\frac{|p^*-1000|}{1000}\le 5\times10^{-4} \iff |p^*-1000|\le 0.5 \iff p^*\in[999.5,\,1000.5]$. For $p = 0.001$ the interval is $[0.0009995, 0.0010005]$: the *absolute* tolerance scales with $|p|$.

**Example 1.2.7 (Cancellation in double precision).** For $x^2 + 10^8x + 1 = 0$ the true small root is $\approx-10^{-8}$. In `float64` the standard formula returns `-7.450580596923828e-09` (wrong in the **first** digit) while the stable formula of Example 1.2.4 returns `-1e-08`. Even 16 digits are not enough when 16 digits cancel.

---

## 1.3 Algorithms and Convergence

An **algorithm** is a finite, unambiguous sequence of steps. B&F write algorithms in the form

```text
ALGORITHM  name
INPUT      ...
OUTPUT     ...
Step 1     ...
Step 2     while (condition) do Steps 3-4
...
```

### 1.3.1 Stability and error growth

An algorithm is **stable** if small changes in the initial data produce correspondingly small changes in the final result; otherwise it is **unstable**. An algorithm is *conditionally stable* if it is stable only for some inputs.

**Definition 1.12.** Suppose an error $E_0>0$ is introduced at some stage and $E_n$ is the error after $n$ further operations.
* **Linear growth:** $E_n\approx CnE_0$ (usually unavoidable and acceptable);
* **Exponential growth:** $E_n\approx C^nE_0$ with $C>1$ (unacceptable).

### 1.3.2 Rates of convergence

**Definition 1.13.** Suppose $\beta_n\to0$ and $\alpha_n\to\alpha$. If there is $K$ with $|\alpha_n-\alpha|\le K|\beta_n|$ for large $n$, we say $\alpha_n$ converges to $\alpha$ with **rate of convergence $O(\beta_n)$**, and write $\alpha_n = \alpha + O(\beta_n)$. Usually $\beta_n = 1/n^p$.

**Definition 1.14.** Suppose $\lim_{h\to0}G(h)=0$ and $\lim_{h\to0}F(h)=L$. If $|F(h)-L|\le K|G(h)|$ for sufficiently small $h$, we write $F(h) = L + O(G(h))$. Usually $G(h) = h^p$; larger $p$ = faster convergence.

**Estimating $p$ from data.** If $E(h)\approx Ch^p$ then
$$
p \approx \frac{\ln\bigl(E(h_1)/E(h_2)\bigr)}{\ln(h_1/h_2)}.
$$
On a log–log plot of $E$ versus $h$, $p$ is the slope. You will use this in every chapter.

### Examples for §1.3

**Example 1.3.1 (Exponential error growth).** The recurrence
$$
p_n = \tfrac73p_{n-1} - \tfrac23p_{n-2}
$$
has general solution $p_n = c_1(1/3)^n + c_2 2^n$ (roots of $r^2 - \frac73r + \frac23 = 0$ are $r=\frac13, 2$). With $p_0 = 1$, $p_1 = \frac13$ we get $c_1=1$, $c_2=0$: $p_n = (1/3)^n$. With five-digit rounding, $p_1 = 0.33333$ introduces a tiny $c_2\approx -2\times10^{-6}$, which is then multiplied by $2^n$:

| $n$ | exact $(1/3)^n$ | computed | error |
|---|---|---|---|
| 2 | $1.11111\times10^{-1}$ | $1.11090\times10^{-1}$ | $2.1\times10^{-5}$ |
| 4 | $1.23457\times10^{-2}$ | $1.22490\times10^{-2}$ | $9.7\times10^{-5}$ |
| 6 | $1.37174\times10^{-3}$ | $9.82900\times10^{-4}$ | $3.9\times10^{-4}$ |
| 8 | $1.52416\times10^{-4}$ | $-1.40330\times10^{-3}$ | $1.6\times10^{-3}$ |
| 12 | $1.88168\times10^{-6}$ | $-2.48890\times10^{-2}$ | $2.5\times10^{-2}$ |

The error roughly quadruples every two steps ($2^2$) — **exponential growth; the algorithm is unstable.**

**Example 1.3.2 (Linear error growth).** $p_n = 2p_{n-1} - p_{n-2}$ has general solution $c_1 + c_2 n$ (double root $r=1$). With $p_0 = 1$, $p_1 = \frac13$: $p_n = 1 - \frac23n$. Five-digit rounding gives errors $6.7\times10^{-6}$ ($n=2$), $3.3\times10^{-5}$ ($n=4$), $1.0\times10^{-4}$ ($n=6$), $1.7\times10^{-4}$ ($n=8$), $5.0\times10^{-4}$ ($n=12$): growth roughly proportional to $n$ — **stable**.

**Example 1.3.3 (Same formula, two directions).** Let $I_n = \int_0^1x^ne^{x-1}dx$. Integration by parts gives $I_n = 1 - nI_{n-1}$, $I_0 = 1-e^{-1}$.

*Forward* recursion multiplies the error by $n$ at step $n$, so the initial rounding error ($\sim10^{-17}$) is multiplied by $n!$:

| $n$ | forward | backward (from $I_{30}\approx0$) |
|---|---|---|
| 10 | 0.0838770701 | 0.0838770701 |
| 15 | 0.0590337936 | 0.0590175409 |
| 17 | 0.0571918706 | 0.0527711192 |
| 18 | **−0.0294536708** | 0.0501198550 |
| 20 | **−30.19** | 0.0455448841 |

Since $0<I_n<1/(n+1)$, negative values are impossible: the forward algorithm is unstable. *Backward* recursion $I_{n-1} = (1-I_n)/n$ *divides* the error by $n$ each step; starting from the crude guess $I_{30}=0$ (error $<1/31$) the error at $n=20$ is below $10^{-15}$. **Same mathematics, opposite stability.**

**Example 1.3.4 (Rate for a function of $h$).** Show that $\cos h + \frac12h^2 = 1 + O(h^4)$.

*Solution.* Taylor: $\cos h = 1 - \frac12h^2 + \frac{1}{24}h^4\cos\xi(h)$, so $|\cos h + \frac12h^2 - 1| \le \frac{1}{24}h^4$. Numerically: $h=0.1\Rightarrow 4.1653\times10^{-6}$ (bound $4.1667\times10^{-6}$); $h=0.01\Rightarrow4.1667\times10^{-10}$. Each factor 10 in $h$ gives a factor $10^4$ in the error.

**Example 1.3.5 (Comparing sequence rates).** $\alpha_n = \frac{2n+1}{n^2}\to0$ and $\hat\alpha_n = \frac{n+2}{n^3}\to0$.
Since $\frac{2n+1}{n^2}\le\frac{3n}{n^2} = \frac3n$ and $\frac{n+2}{n^3}\le\frac{3n}{n^3}=\frac{3}{n^2}$ for $n\ge1$: $\alpha_n = O(1/n)$, $\hat\alpha_n = O(1/n^2)$.

| $n$ | $\alpha_n$ | $\hat\alpha_n$ |
|---|---|---|
| 10 | 0.21 | $1.2\times10^{-2}$ |
| 100 | 0.0201 | $1.02\times10^{-4}$ |
| 1000 | 0.002001 | $1.002\times10^{-6}$ |

**Example 1.3.6 (Order from data).** $F(h) = \frac{e^h-1}{h}\to 1$. Values of $F(h)-1$: $5.171\times10^{-2}$, $5.017\times10^{-3}$, $5.002\times10^{-4}$ at $h = 0.1, 0.01, 0.001$. The estimated order is $\ln(5.171/0.5017)/\ln 10 = 1.01$: $F(h) = 1 + O(h)$, consistent with $\frac{e^h-1}{h} = 1 + \frac h2 + O(h^2)$.

---

## 1.4 Conditioning, Stability and Backward Error *(data-science extension)*

B&F discuss stability of algorithms; modern practice separates two questions:

* **Conditioning** — a property of the *problem*: how sensitive is the exact answer to perturbations of the data?
* **Stability** — a property of the *algorithm*: does it achieve the accuracy that the conditioning allows?

**Definition 1.15 (relative condition number).** For a differentiable $f$,
$$
\kappa_f(x) = \left|\frac{x f'(x)}{f(x)}\right|,\qquad \frac{|f(x+\Delta x)-f(x)|}{|f(x)|}\approx\kappa_f(x)\frac{|\Delta x|}{|x|}.
$$
If $\kappa\approx10^k$, up to $k$ significant digits may be lost by **any** algorithm.

**Definition 1.16 (forward and backward error).** If an algorithm returns $\hat y$ for $y = f(x)$: forward error $=|\hat y-y|$; backward error $= \min\{|\Delta x| : f(x+\Delta x) = \hat y\}$. An algorithm is **backward stable** if its backward error is always $O(u)$ (relative).

**Rule of thumb (fundamental inequality).**
$$
\text{relative forward error} \;\lesssim\; \kappa \times \text{relative backward error}.
$$

### Examples for §1.4

**Example 1.4.1 (Condition numbers of basic functions).**

| $f(x)$ | $\kappa_f(x)$ | ill-conditioned when |
|---|---|---|
| $\sqrt x$ | $1/2$ | never |
| $e^x$ | $\lvert x\rvert$ | $\lvert x\rvert$ large |
| $\ln x$ | $1/\lvert\ln x\rvert$ | $x\approx1$ |
| $x-a$ | $\lvert x\rvert/\lvert x-a\rvert$ | $x\approx a$ (cancellation) |
| $\sin x$ | $\lvert x\cot x\rvert$ | $x\approx k\pi,\ k\neq0$ |

Subtraction is the *only* ill-conditioned arithmetic operation — this is Example 1.2.3 seen from the problem side.

**Example 1.4.2.** $\kappa_{\ln}(1.001) = 1/\ln(1.001) = 1000.5$. A measurement error of $10^{-6}$ relative in $x$ causes a $10^{-3}$ relative error in $\ln x$, whatever algorithm is used. (If the data is really $\delta = x-1$ known exactly, compute `log1p(δ)` — a *different*, well-conditioned problem.)

**Example 1.4.3.** $f(x) = \tan x$ near $\pi/2$: $\kappa(1.57) = \bigl|\frac{x}{\sin x\cos x}\bigr| \approx 1972$. Indeed $\tan(1.57) = 1255.77$ while $\tan(1.5701) = 1436.11$: a relative input change of $6.4\times10^{-5}$ produced a $14\%$ output change.

**Example 1.4.4 (An ill-conditioned linear system — preview of Chapter 6).** $A = \begin{pmatrix}1&1\\1&1.0001\end{pmatrix}$ has $\kappa_\infty(A) = \|A\|_\infty\|A^{-1}\|_\infty = 40004$. The solution of $A\mathbf x = (2, 2.0001)^T$ is $(1,1)^T$; of $A\mathbf x = (2,2.0002)^T$ it is $(0,2)^T$. A $5\times10^{-5}$ relative change in the data changed the answer by 100%.

**Example 1.4.5 (Wilkinson's polynomial).** $w(x) = \prod_{k=1}^{20}(x-k)$ has well-separated roots $1,\dots,20$. Changing the coefficient of $x^{19}$ from $-210$ to $-210-2^{-23}$ (a relative change of $5.7\times10^{-10}$) moves the roots dramatically — ten of them become complex, e.g. $10.095\pm0.645i$, $11.794\pm1.652i$, $13.992\pm2.519i$, $16.731\pm2.813i$. **Polynomial roots as functions of the coefficients are extremely ill-conditioned**; this is why eigenvalues are never computed from the characteristic polynomial (Chapter 9).

**Example 1.4.6 (Forward vs backward error).** A routine returns $\hat y = 1.4142$ for $\sqrt2$. Forward error $|1.4142-\sqrt2| = 1.36\times10^{-5}$. Backward error: $\hat y$ is the exact root of $1.4142^2 = 1.99996164$, so the backward error is $3.84\times10^{-5}$ (relative $1.92\times10^{-5}$). Check: $\kappa_{\sqrt{}}=\frac12$ and $\frac12\times1.92\times10^{-5} = 9.6\times10^{-6} =$ relative forward error. ✓

---

## 1.5 Floating Point in Data-Science Computations *(data-science extension)*

Every idea above appears in everyday data work. Summary of the standard fixes:

| Computation | Failure | Stable alternative |
|---|---|---|
| $\sum_i x_i$ for large $n$ | error $\sim nu\sum\lvert x_i\rvert$ | pairwise (`np.sum`), Kahan, `math.fsum` |
| sample variance | cancellation in $\sum x^2 - n\bar x^2$ | two-pass, Welford |
| $\log\sum e^{z_i}$, softmax | overflow | subtract $\max z_i$ |
| $\prod p_i$ (likelihood) | underflow to 0 | $\sum\log p_i$ |
| $\log(1+x)$, $e^x-1$ | cancellation for small $x$ | `log1p`, `expm1` |
| $\log\sigma(z)$ (logistic loss) | $\log 0$ | `-np.logaddexp(0, -z)` |
| counters in `float32` | stagnation | accumulate in `float64` |

**Kahan (compensated) summation.**
```text
s = 0; c = 0
for x in data:
    y = x - c        # correct the next term by the lost low-order part
    t = s + y        # low-order digits of y are lost here ...
    c = (t - s) - y  # ... and recovered here
    s = t
```
Its error bound is $2u\sum|x_i| + O(nu^2)$, independent of $n$ to first order.

**Welford's streaming variance.**
$$
\bar x_k = \bar x_{k-1} + \frac{x_k-\bar x_{k-1}}{k},\qquad M_k = M_{k-1} + (x_k-\bar x_{k-1})(x_k-\bar x_k),\qquad s^2 = \frac{M_n}{n-1}.
$$

**Log-sum-exp.** With $m = \max_j z_j$: $\log\sum_je^{z_j} = m + \log\sum_j e^{z_j-m}$ — all exponents $\le0$, at least one term equals 1.

### Examples for §1.5

**Example 1.5.1 (Summation).** Summing $0.1$ one million times (exact decimal answer $100000$; `math.fsum` gives the correctly rounded sum of the stored values): naive loop error $1.33\times10^{-6}$; NumPy pairwise sum $2.9\times10^{-11}$; Kahan $0$.

**Example 1.5.2 (Variance with a large mean).** Data $10^9 + \{4, 7, 13, 16\}$, true $s^2 = 30$. Textbook one-pass formula: **$-170.67$** (a negative variance!). Two-pass and Welford: $30.0$. Here $\sum x_i^2\approx4\times10^{18}$ where doubles are spaced $512$ apart, so the true difference $90$ is pure noise.

**Example 1.5.3 (Softmax).** $z = (1000, 1001, 1002)$: `exp(1000)` overflows. Stable: $\log\sum e^{z_j} = 1002 + \log(e^{-2}+e^{-1}+1) = 1002.40761$ and $\operatorname{softmax}(z) = (0.0900, 0.2447, 0.6652)$.

**Example 1.5.4 (Likelihood underflow).** $\prod_{i=1}^{1000}0.01 = 10^{-2000}$ underflows to `0.0`; the log-likelihood $\sum\ln 0.01 = -4605.17$ is perfectly representable. Maximum-likelihood software always works on the log scale.

**Example 1.5.5 (`float32` accumulation).** Adding `float32(0.1)` $10^4$ times yields $999.9029$; and `float32(2**24) + 1 == 2**24` because the spacing of `float32` numbers at $2^{24}$ is $2$. GPU code that accumulates large counts or losses in single precision silently stagnates.

**Example 1.5.6 (Logistic loss).** For a logit $z=-800$ the naive $\sigma(z) = 1/(1+e^{800})$ is `0.0` (overflow inside), and $\log\sigma(z) = -\infty$ breaks training. The identity $\log\sigma(z) = -\log(1+e^{-z}) = -\texttt{logaddexp}(0,-z)$ returns $-800$ exactly.

---

## Chapter summary

* Calculus theorems (IVT, MVT, EVT, weighted MVT, Taylor) are the tools for **error bounds**.
* $k$-digit arithmetic has relative round-off at most $10^{-k+1}$ (chop) or $\frac12 10^{-k+1}$ (round); in double precision $u\approx1.1\times10^{-16}$.
* Loss of significance comes from subtracting nearly equal numbers; fix by reformulation and nested evaluation.
* Stable algorithms have (at most) linear error growth. Convergence rates are described by $O(n^{-p})$ or $O(h^p)$ and measured by log–log slopes.
* Forward error $\lesssim$ condition number × backward error.

## Further reading

* Burden & Faires, Chapter 1. · N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002. · D. Goldberg, "What every computer scientist should know about floating-point arithmetic", *ACM Computing Surveys* 23 (1991).
