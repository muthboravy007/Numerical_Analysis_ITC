# Chapter 1 — Solutions

> Exercises: [`exercises/ch01_exercises.md`](../exercises/ch01_exercises.md) · All numbers are produced by [`solutions/code/ch01_solutions.py`](code/ch01_solutions.py).

## A. Theory and proofs

**A1.** Write $y = 0.d_1d_2\ldots d_kd_{k+1}\ldots\times10^n$. Chopping discards $0.d_{k+1}d_{k+2}\ldots\times10^{n-k}$, so $|y - fl(y)|<10^{n-k}$. Since $d_1\ge1$, $|y|\ge0.1\times10^n = 10^{n-1}$. Hence $\frac{|y-fl(y)|}{|y|}<\frac{10^{n-k}}{10^{n-1}} = 10^{-k+1}$. For rounding the discarded (or added) amount is at most half a unit in the $k$-th digit, $\frac12\times10^{n-k}$, giving $\frac12\times10^{-k+1}$. ∎

**A2.** By the MVT, $\sin a - \sin b = \cos\xi\,(a-b)$ for some $\xi$ between $a$ and $b$, and $|\cos\xi|\le1$. For $g(x) = \frac12\sin x$: $|g(a)-g(b)| = \frac12|\sin a-\sin b|\le\frac12|a-b|$, so $g$ is a contraction with constant $\frac12$ (and the iteration converges to its unique fixed point $0$).

**A3.**
(a) $\kappa = \bigl|\frac{x\cdot nx^{n-1}}{x^n}\bigr| = |n|$ — never ill-conditioned for moderate $n$.
(b) $\kappa = \bigl|\frac{x(-x^{-2})}{x^{-1}}\bigr| = 1$.
(c) $f'(x) = \frac{1}{2\sqrt{1+x}}$, $\kappa = \frac{|x|}{2\sqrt{1+x}\,|\sqrt{1+x}-1|} = \frac{|x|(\sqrt{1+x}+1)}{2\sqrt{1+x}\,|x|} = \frac{\sqrt{1+x}+1}{2\sqrt{1+x}}\to1$ as $x\to0$. The *problem* is well-conditioned; the naive *algorithm* (subtracting two nearly equal numbers) is unstable. The stable form is $\frac{x}{\sqrt{1+x}+1}$.
(d) $\kappa = |x|$: ill-conditioned for large $|x|$.
(e) $\kappa = |x\tan x|$: ill-conditioned near zeros of $\cos$, i.e. $x\approx\frac\pi2 + k\pi$.

**A4.** With $\hat s_k = (\hat s_{k-1}+x_k)(1+\delta_k)$, induction gives $\hat s_n = \sum_{i=1}^nx_i\prod_{k=\max(i,2)}^n(1+\delta_k)$. Each product is $1+\theta_i$ with $|\theta_i|\le(n-\max(i,2)+1)u + O(u^2)\le(n-1)u + O(u^2)$. Hence $|\hat s_n - s_n| = |\sum x_i\theta_i|\le(n-1)u\sum|x_i|$ to first order. The earliest terms are multiplied by the most $(1+\delta)$ factors; adding small numbers first keeps the partial sums (which carry the errors) small, and reduces the chance that a small term is swallowed by a large partial sum.

**A5.** (a) $y_n + 5y_{n-1} = \int_0^1\frac{x^n+5x^{n-1}}{x+5}dx = \int_0^1x^{n-1}dx = \frac1n$; $y_0 = \int_0^1\frac{dx}{x+5} = \ln\frac65$.
(b) If $\tilde y_{n-1} = y_{n-1}+e_{n-1}$ then $\tilde y_n = \frac1n - 5\tilde y_{n-1} = y_n - 5e_{n-1}$, so $e_n = (-5)^ne_0$. An initial error of $10^{-17}$ becomes $5^{25}\times10^{-17}\approx3\times10^{0}$ at $n = 25$. Numerically: $y_{15}$ is correct to 4 digits, $y_{20} = 4.24\times10^{-3}$ (true $8.00\times10^{-3}$), $y_{25} = 11.7$, $y_{30} = -3.7\times10^4$ — although $0<y_n<\frac{1}{5(n+1)}$.
(c) Run the recurrence **backward**: $y_{n-1} = \frac{1/n - y_n}{5}$, which divides errors by 5 per step. Start from a crude value, e.g. $y_{40} = 0$ (true value $<0.005$); by $n = 30$ the error is below $0.005\cdot5^{-10}\approx5\times10^{-10}$ and at $n\le25$ it is at machine precision: $y_{25} = 0.0064503053$, $y_{30} = 0.0054046325$ (quadrature: $0.0054046330$).

## B. Hand computation

**B1.** $P_3(x) = 1 - x + \frac{x^2}2 - \frac{x^3}6$, $P_3(0.5) = 0.6041667$. Remainder $R_3 = \frac{e^{-\xi}}{24}x^4$ with $0<\xi<0.5$, so $|R_3|\le\frac{0.5^4}{24} = 2.60\times10^{-3}$. True value $e^{-0.5} = 0.6065307$; actual error $2.36\times10^{-3}$ — within the bound.

**B2.** For an alternating series with decreasing terms, the error after the term $\frac{(-1)^n}{2n+1}$ is less than the next term $\frac{1}{2n+3}$. We need $\frac1{2n+3}<10^{-3}$, i.e. $n\ge499$: **500 terms** for just three digits of $\pi/4$ (and the error in $\pi$ is 4× larger). The series is useless for computation without acceleration (Aitken/Richardson, Chapter 2/4).

**B3.** $fl(x) = 0.5462$, $fl(y) = 0.5460$, $fl(x)-fl(y) = 0.0002000$ (only one significant digit survives; exact $0.00016$). Then $0.0002/0.001234 = 0.16207\to0.1621$. Exact $(x-y)/z = 0.129660$; relative error $25\%$. The subtraction of nearly equal numbers caused the loss; the division merely propagated it.

**B4.**
(a) $\frac{1-\cos x}{\sin x} = \frac{2\sin^2(x/2)}{2\sin(x/2)\cos(x/2)} = \tan\frac x2$. Naive: $0.0$; stable: $5.0\times10^{-9}$.
(b) $\ln(x-\sqrt{x^2-1}) = -\ln(x+\sqrt{x^2-1})$ (multiply by the conjugate). Naive: $x - \sqrt{x^2-1}$ evaluates to $0$ and the log is $-\infty$; stable: $-19.113828$.
(c) $e^x-e^{-x} = 2\sinh x$ (use `np.sinh`, or the series $2(x + x^3/6+\cdots)$). Naive: $2.00000017\times10^{-10}$ (7 digits correct); stable: $2.0\times10^{-10}$.
(d) $x - \sin x = \frac{x^3}{6} - \frac{x^5}{120}+\cdots$. Naive: $1.66667285\times10^{-16}$ (5 digits); series: $1.66666667\times10^{-16}$.

**B5.** Exact $P(1.72) = -1.677676$. Direct: $t^2 = 2.96$, $t^3 = 5.09$, $1.01t^3 = 5.14$, $4.62t^2 = 13.7$, $3.11t = 5.35$; $5.14 - 13.7 = -8.56$; $-8.56 - 5.35 = -13.9$; $-13.9+12.2 = -1.70$. Nested: $((1.01t - 4.62)t - 3.11)t + 12.2$: $1.74 - 4.62 = -2.88$; $\times1.72 = -4.95$; $-3.11\to-8.06$; $\times1.72 = -13.9$; $+12.2 = -1.70$. Both give $-1.70$ (relative error $1.3\%$) — here the final cancellation ($-13.9 + 12.2$) dominates. Nested form uses 3 multiplications instead of 5 ($t^2$, $t^3$, and three coefficient products).

**B6.** $x^5$: $\kappa = 5$ (lose <1 digit). $1/x$: $1$. $\sqrt{1+x}-1$ at $10^{-3}$: $\approx1.0$ (well-conditioned — use the stable formula). $e^x$ at 50: $\kappa = 50$ (lose ~2 digits: a relative input error $10^{-16}$ becomes $5\times10^{-15}$). $\cos x$ at $\frac\pi2 - 10^{-3}$: $\kappa = |x\tan x|\approx1570$ (lose ~3 digits).

**B7.** (a) $0.625 = \frac12+\frac18 = 0.101_2$. (b) $0.1 = 0.000110011001\ldots_2$ — the block $0011$ repeats forever, so $0.1$ has no finite binary expansion and must be rounded. (c) Sign $1$ ⇒ negative; exponent $10000000001_2 = 1025$, so $2^{1025-1023} = 4$; fraction $0.1010_2 = 0.625$; value $-4\times1.625 = -6.5$.

## C. Programming (key results)

**C1.** Halving gives $9.77\times10^{-4}$ (`float16`), $1.19\times10^{-7}$ (`float32`), $2.22\times10^{-16}$ (`float64`) — identical to `np.finfo`.

**C2.** Exact values $14.3927267$ ($n = 10^6$) and $16.6953114$ ($n = 10^7$):

| $n$ | naive float32 | `np.sum` float32 | Kahan float32 |
|---|---|---|---|
| $10^6$ | 14.35736 (err $3.5\times10^{-2}$) | 14.39273 ($1.7\times10^{-6}$) | 14.39273 ($1.8\times10^{-7}$) |
| $10^7$ | 15.40368 (err $1.3$) | 16.69531 ($1.1\times10^{-6}$) | 16.69531 ($7.7\times10^{-7}$) |

The naive sum stagnates: once the partial sum is ≈15, terms below half its spacing ($\approx4.8\times10^{-7}$), i.e. $1/k$ for $k\gtrsim2\times10^6$, are lost entirely. Pairwise summation has error $O(u\log n)$; Kahan $O(u)$.

**C3.** Errors: $4.3\times10^{-2}$ ($h = 0.1$) decreasing like $h$ to $3.0\times10^{-9}$ at $h = 10^{-8}$, then increasing like $1/h$ to $1.5\times10^{-2}$ at $h = 10^{-15}$. Optimal $h\approx2\sqrt{u|f|/|f''|}\approx10^{-8}$; minimum error $\approx\sqrt u\approx10^{-8}$.

**C4.** $\operatorname{logsumexp}(-1000,-1001) = -1000 + \ln(1+e^{-1}) = -999.686738$ (naive: $\ln0 = -\infty$). $\operatorname{softmax}(10^4, 10^4+1) = (0.268941, 0.731059)$ (naive: `nan`).

**C5.** In `float64` the two-pass and Welford results agree ($0.9550014$) while the textbook formula gives $0.9547$ (3 digits). In `float32` the *data themselves* are rounded to multiples of $0.0625$ near $10^6$ (spacing $2^{19-23}$), so every method sees perturbed data: textbook $0.9561$, two-pass $0.9560$, Welford (mixed-precision loop) $0.985$. Lesson: store large-offset data centred (subtract a reference value) or in double precision.

## D. Data-science applications

**D1.** The likelihood underflows to exactly `0.0` (it is about $e^{-61177}$). The log-likelihood is $-61177.09$, i.e. $-0.61177$ per observation, close to $0.7\ln0.7 + 0.3\ln0.3 = -0.61086$ (the negative entropy of Bernoulli(0.7)) because the sample frequency is ≈0.7. Always work with log-likelihoods.

**D2.** (a) $2^{-24}\approx6.0\times10^{-8}$. (b) `np.float16(1e-8)` is `0.0` — the gradient is lost (underflow). (c) With loss scaling, the scaled gradient $1.024\times10^{-5}$ is representable (`1.025e-05` in float16); dividing by $1024$ in float32 recovers $1.0012\times10^{-8}$ (0.1% error from float16's 11-bit mantissa). Frameworks choose $S$ dynamically, lowering it when overflow (`inf`) appears.
