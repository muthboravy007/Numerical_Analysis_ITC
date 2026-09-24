# Lecture 1 — Floating-Point Arithmetic and Error Analysis

> **Week 1–2** · Code: [`numlib/floating.py`](../numlib/floating.py) · Examples: [`examples/ch01`](../examples/ch01_examples.md) · Exercises: [`exercises/ch01`](../exercises/ch01_exercises.md)

## Learning objectives

By the end of this lecture you should be able to

1. describe how real numbers are stored in IEEE-754 double precision;
2. define and compute absolute error, relative error, machine epsilon and unit round-off;
3. separate **truncation error** (the method) from **rounding error** (the computer);
4. recognise **catastrophic cancellation** and rewrite formulas to avoid it;
5. use the **condition number** of a problem and the **stability** of an algorithm to predict how accurate a result can be;
6. apply these ideas to everyday data-science computations: sums, variances, softmax and log-likelihoods.

---

## 1.1 Why a data scientist should care

```python
>>> 0.1 + 0.2 == 0.3
False
>>> 0.1 + 0.2
0.30000000000000004
```

This is not a bug in Python. It is a consequence of storing real numbers in a finite number of bits. In data science the same effect shows up as

* a log-likelihood that becomes `-inf` because a probability underflowed to `0`;
* `softmax` returning `nan` because `exp(1000)` overflowed;
* a variance that comes out **negative** for data with a large mean;
* a regression whose coefficients change completely when you add `1e-10` to one data point;
* a gradient check that "fails" only because the step `h` was chosen too small.

Numerical analysis is the study of algorithms for continuous mathematics **together with the errors they make**. This first lecture gives the vocabulary that every later chapter relies on.

---

## 1.2 Sources of error

| Source | Example | Controlled by |
|---|---|---|
| **Modelling error** | Assuming a linear relationship that is really quadratic | Domain knowledge |
| **Data / measurement error** | Sensor noise, rounding in a CSV file | Data collection |
| **Truncation (discretisation) error** | Replacing $f'(x)$ by $(f(x+h)-f(x))/h$; stopping a series after $n$ terms | Choice of method and step size |
| **Rounding error** | Storing $0.1$ in binary | Floating-point format, algorithm design |

This course concentrates on the last two.

### Measuring error

Let $x$ be the exact value and $\hat{x}$ an approximation.

$$
\text{absolute error} = |\hat{x} - x|, \qquad
\text{relative error} = \frac{|\hat{x} - x|}{|x|}\quad (x \ne 0).
$$

Relative error is scale-free and is almost always the right thing to report. If the relative error is about $10^{-k}$, then $\hat{x}$ has roughly $k$ correct significant digits.

**Example.** $x = \pi$, $\hat{x} = 22/7 = 3.142857\ldots$
Absolute error $\approx 1.26\times 10^{-3}$, relative error $\approx 4.0 \times 10^{-4}$ → about 3 significant digits.

---

## 1.3 Floating-point numbers

A (normalised) binary floating-point number has the form

$$
x = \pm\, (1.d_1 d_2 \ldots d_{t})_2 \times 2^{e}, \qquad e_{\min} \le e \le e_{\max}.
$$

* $d_i \in \{0,1\}$ are the bits of the **mantissa** (significand);
* $t$ is the **precision** (number of fraction bits);
* $e$ is the **exponent**.

### IEEE-754 formats

| Format | Bits | Fraction bits $t$ | Exponent range | Machine epsilon $\varepsilon_{\text{mach}}=2^{-t}$ | ≈ decimal digits | Largest value |
|---|---|---|---|---|---|---|
| half (`float16`) | 16 | 10 | $[-14, 15]$ | $9.8\times10^{-4}$ | 3 | $6.6\times10^{4}$ |
| single (`float32`) | 32 | 23 | $[-126, 127]$ | $1.19\times10^{-7}$ | 7 | $3.4\times10^{38}$ |
| double (`float64`) | 64 | 52 | $[-1022, 1023]$ | $2.22\times10^{-16}$ | 16 | $1.8\times10^{308}$ |

Deep-learning frameworks increasingly train in `float16`/`bfloat16`, so the precision trade-off is a daily practical question, not an academic one.

### Machine epsilon and rounding

**Machine epsilon** $\varepsilon_{\text{mach}}$ is the gap between $1$ and the next larger floating-point number. The **unit round-off** is $u = \varepsilon_{\text{mach}}/2$.

**Fundamental axiom of floating-point arithmetic.** For any real $x$ in range, and for each operation $\circ \in \{+,-,\times,\div\}$,

$$
\mathrm{fl}(x) = x(1+\delta), \qquad \mathrm{fl}(a \circ b) = (a \circ b)(1+\delta), \qquad |\delta| \le u.
$$

In words: *each individual operation is correctly rounded, with relative error at most $u \approx 1.1\times10^{-16}$*. The trouble comes from **combining** many operations, or from operations that amplify the tiny errors already present in the inputs.

```python
import numpy as np
np.finfo(float).eps        # 2.220446049250313e-16
np.finfo(np.float32).eps   # 1.1920929e-07
np.spacing(1e16)           # 2.0  -> 1e16 + 1 == 1e16 !
```

### Special values

* **Overflow** → `inf`; **underflow** → `0` (or a subnormal number).
* `nan` ("not a number") arises from `0/0`, `inf - inf`, `sqrt(-1)`. Any comparison with `nan` is `False` — which is why `x != x` is the classic `nan` test.

---

## 1.4 Catastrophic cancellation

Subtracting two nearly equal numbers **exposes** the rounding errors they already carry. The subtraction itself is exact (Sterbenz lemma); the problem is that the leading correct digits cancel and only the noisy trailing digits remain.

If $\hat a = a(1+\delta_1)$ and $\hat b = b(1+\delta_2)$, then

$$
\frac{|(\hat a - \hat b) - (a-b)|}{|a-b|} \le u\,\frac{|a|+|b|}{|a-b|}.
$$

The factor $\frac{|a|+|b|}{|a-b|}$ can be enormous when $a \approx b$.

### Example 1: the quadratic formula

For $x^2 + 10^8 x + 1 = 0$ the roots are $\approx -10^{8}$ and $\approx -10^{-8}$. The textbook formula computes the small root as $\frac{-b + \sqrt{b^2 - 4ac}}{2a}$, subtracting two numbers that agree in about 16 digits:

```text
naive  : x2 = -7.450580596923828e-09   (wrong in the first digit!)
stable : x2 = -1.0e-08
```

**Fix.** Compute the large root without cancellation, then use Vieta's formula $x_1 x_2 = c/a$:

$$
q = -\tfrac12\bigl(b + \operatorname{sign}(b)\sqrt{b^2-4ac}\bigr), \qquad x_1 = \frac{q}{a},\qquad x_2 = \frac{c}{q}.
$$

### Example 2: algebraic rewriting

| Unstable for | Expression | Stable equivalent |
|---|---|---|
| large $x$ | $\sqrt{x+1} - \sqrt{x}$ | $\dfrac{1}{\sqrt{x+1}+\sqrt{x}}$ |
| small $x$ | $1-\cos x$ | $2\sin^2(x/2)$ |
| small $x$ | $e^x - 1$ | `np.expm1(x)` |
| small $x$ | $\log(1+x)$ | `np.log1p(x)` |
| large $x$ | $\log(1+e^{x})$ (softplus) | `np.logaddexp(0, x)` |

With $x = 10^{-8}$, `(1 - cos(x))/x**2` returns `0.0` while the exact value is $0.5$; the rewritten form returns `0.5`.

---

## 1.5 Conditioning versus stability

Two different questions must be kept apart:

1. **Conditioning** is a property of the *problem*: how much can the exact answer change when the input is perturbed slightly?
2. **Stability** is a property of the *algorithm*: does it deliver an answer that is as good as the conditioning allows?

### Condition number of a function

For $y = f(x)$ the **relative condition number** is

$$
\kappa_f(x) = \left|\frac{x\,f'(x)}{f(x)}\right|,
\qquad\text{so}\qquad
\frac{|\Delta y|}{|y|} \approx \kappa_f(x)\,\frac{|\Delta x|}{|x|}.
$$

| $f(x)$ | $\kappa_f(x)$ | Ill-conditioned when |
|---|---|---|
| $\sqrt{x}$ | $1/2$ | never |
| $e^{x}$ | $\lvert x\rvert$ | $\lvert x\rvert$ large |
| $\log x$ | $1/\lvert\log x\rvert$ | $x \approx 1$ |
| $x - a$ | $\lvert x\rvert/\lvert x-a\rvert$ | $x \approx a$ (cancellation!) |

**Rule of thumb.** If $\kappa \approx 10^{k}$ you may lose up to $k$ significant digits *whatever algorithm you use*.

### Forward and backward error

Suppose an algorithm returns $\hat y$ for input $x$.

* **Forward error:** $|\hat y - f(x)|$.
* **Backward error:** the smallest $|\Delta x|$ such that $\hat y = f(x+\Delta x)$ exactly.

An algorithm is **backward stable** if the backward error is always $O(u)$ relative to $x$. Then

$$
\text{forward error} \;\lesssim\; \kappa \times \text{backward error} \approx \kappa\, u .
$$

This single inequality explains most of numerical linear algebra (Lectures 3–5).

---

## 1.6 Error accumulation in sums

Adding $n$ numbers left to right has an error bound

$$
|\hat S_n - S_n| \lesssim (n-1)\,u \sum_{i=1}^n |x_i|.
$$

Better algorithms exist:

* **Pairwise summation** (`np.sum` uses it): error $O(u \log n)$.
* **Kahan (compensated) summation**: error $O(u)$ independent of $n$.

```python
def kahan_sum(xs):
    s, c = 0.0, 0.0
    for x in xs:
        y = x - c          # subtract the error from last time
        t = s + y          # low bits of y may be lost here...
        c = (t - s) - y    # ...recover them
        s = t
    return s
```

Summing `0.1` one million times: naive error ≈ $1.3\times10^{-6}$, Kahan error ≈ $0$.

---

## 1.7 Data-science case studies

### (a) Variance

The "one-pass" textbook formula
$$
s^2 = \frac{1}{n-1}\Bigl(\sum x_i^2 - n\bar x^2\Bigr)
$$
subtracts two huge, nearly equal numbers when the mean is large compared to the spread. For the data $10^8 + \{4, 7, 13, 16\}$ (true variance $30$) it returns `29.33`; with larger offsets it can even return a **negative** number.

Stable choices:

* **two-pass**: first compute $\bar x$, then $\frac{1}{n-1}\sum (x_i-\bar x)^2$;
* **Welford's streaming algorithm** (one pass, stable) — used in online/streaming analytics:

$$
\bar x_k = \bar x_{k-1} + \frac{x_k - \bar x_{k-1}}{k},\qquad
M_k = M_{k-1} + (x_k - \bar x_{k-1})(x_k - \bar x_k),\qquad
s^2 = \frac{M_n}{n-1}.
$$

### (b) Softmax and log-sum-exp

$\operatorname{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$ overflows for $z_i \gtrsim 710$. Because softmax is invariant to shifting all $z_i$ by a constant, subtract $m = \max_j z_j$:

$$
\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}.
$$

Now every exponent is $\le 0$, no overflow is possible, and at least one term equals $1$ so no total underflow occurs either. Every ML library implements cross-entropy this way.

### (c) Products of probabilities

The likelihood $\prod_{i=1}^{n} p_i$ with $n=1000$ and $p_i \approx 0.01$ is $10^{-2000}$ — it underflows to `0`. Always work with $\sum \log p_i$.

---

## 1.8 Truncation error and Taylor's theorem

Most numerical methods come from truncated Taylor series. If $f$ is $(n+1)$ times continuously differentiable,

$$
f(x+h) = \sum_{k=0}^{n}\frac{f^{(k)}(x)}{k!}h^k + \underbrace{\frac{f^{(n+1)}(\xi)}{(n+1)!}h^{n+1}}_{\text{truncation error}},\qquad \xi \text{ between } x \text{ and } x+h.
$$

**Big-O notation.** We write $E(h) = O(h^p)$ if $|E(h)| \le C h^p$ for small $h$. A method with error $O(h^p)$ is said to be **of order $p$**: halving $h$ divides the error by about $2^p$.

**Estimating the order empirically.** If $E(h) \approx C h^p$ then

$$
p \approx \frac{\log\bigl(E(h_1)/E(h_2)\bigr)}{\log(h_1/h_2)}.
$$

On a log–log plot of error against $h$ the slope is $p$. You will use this in every chapter.

### The truncation–rounding trade-off

For the forward difference $D_h = (f(x+h)-f(x))/h$:

$$
\underbrace{\tfrac{h}{2}|f''(x)|}_{\text{truncation}} + \underbrace{\tfrac{2u|f(x)|}{h}}_{\text{rounding}}
\quad\text{is minimised at}\quad h^\star \approx 2\sqrt{u\,|f|/|f''|}\approx 10^{-8}.
$$

Making $h$ smaller than $h^\star$ makes the answer **worse**. We revisit this in Lecture 8.

---

## 1.9 Summary

* Each floating-point operation has relative error at most $u \approx 1.1\times10^{-16}$ (double precision).
* Loss of accuracy comes from **ill-conditioned problems** or **unstable algorithms** — usually cancellation, overflow/underflow, or error growth.
* Forward error $\lesssim$ condition number × backward error.
* Rewrite formulas to avoid subtracting nearly equal quantities; use `log1p`, `expm1`, `logaddexp`, log-sum-exp, Welford.
* Every numerical method has a **truncation error** $O(h^p)$; measure $p$ with a log–log plot.

## Further reading

* D. Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, ACM Computing Surveys, 1991.
* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002 — Chapters 1–4.
* T. Sauer, *Numerical Analysis*, 3rd ed., Chapter 0.
