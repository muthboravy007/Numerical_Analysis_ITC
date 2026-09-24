# Chapter 1 — Worked Examples: Floating Point and Error Analysis

Companion script: [`ch01_examples.py`](ch01_examples.py) reproduces every number below.
Lecture: [Chapter 1](../lectures/ch01-mathematical-preliminaries.md)

---

## Example 1.1 — Absolute error, relative error, significant digits

**Problem.** Compare the approximations $22/7$ and $355/113$ to $\pi$.

**Solution.**

| Approximation | Absolute error $\lvert\hat x - \pi\rvert$ | Relative error | $-\log_{10}(2\cdot\text{rel})$ ≈ correct digits |
|---|---|---|---|
| $22/7 = 3.142857\ldots$ | $1.26\times10^{-3}$ | $4.02\times10^{-4}$ | 3.1 |
| $355/113 = 3.14159292\ldots$ | $2.67\times10^{-7}$ | $8.49\times10^{-8}$ | 6.8 |

$355/113$ gives about 7 correct significant digits using only 6 digits in total — Zu Chongzhi's remarkable approximation (5th century).

**Take-away.** Relative error translates directly into "number of correct digits".

---

## Example 1.2 — Why `0.1 + 0.2 != 0.3`

**Problem.** Explain the output `0.30000000000000004`.

**Solution.** In binary, $0.1 = 0.0\overline{0011}_2$ is a repeating fraction, so it must be rounded to 53 significant bits. The stored value is exactly

```
0.1000000000000000055511151231257827021181583404541015625
```

Similarly $0.2$ and $0.3$ are rounded. The rounding errors of $0.1$ and $0.2$ happen to add up so that the correctly rounded sum is the double *just above* the double nearest to $0.3$ — they differ by one unit in the last place ($2^{-54} \approx 5.6\times10^{-17}$).

**Correct way to compare floats:**
```python
math.isclose(0.1 + 0.2, 0.3)            # True  (rel_tol=1e-9 by default)
np.isclose(a, b, rtol=1e-9, atol=1e-12) # arrays
```

---

## Example 1.3 — Large numbers swallow small ones

**Problem.** What is `1e16 + 1 - 1e16` in double precision? What is `float32(16777216) + 1`?

**Solution.** The spacing between doubles near $10^{16}$ is $\text{spacing} = 2^{\lfloor\log_2 10^{16}\rfloor - 52} = 2^{53-52} = 2$. Adding 1 is exactly half a spacing and "round half to even" gives $10^{16}$ back. So the answer is **0**, not 1.

In single precision, $16\,777\,216 = 2^{24}$ and $t = 23$ fraction bits, so the spacing there is $2^{24-23} = 2$ and again $2^{24}+1 \to 2^{24}$.

**Data-science consequence.** Accumulating counts or sums in `float32` stops increasing once the total is large. Summing `float32(0.1)` ten thousand times returns $999.90$ instead of $1000$. Use `float64` accumulators (NumPy does this for `float32` input when you pass `dtype=np.float64`).

---

## Example 1.4 — Catastrophic cancellation and its fix

**Problem.** Evaluate $f(x) = \sqrt{x+1} - \sqrt{x}$ at $x = 10^8$.

**Solution.**

*Naive:* $\sqrt{100000001} = 10000.00005$ and $\sqrt{10^8}=10^4$ each have relative error $\approx10^{-16}$, i.e. absolute error $\approx 10^{-12}$. Their difference is $\approx 5\times10^{-5}$, so the relative error of the difference is about $10^{-12}/(5\times 10^{-5}) = 2\times10^{-8}$: we have lost 8 digits.

```
naive  : 5.000000055588316e-05
exact  : 4.999999987500000e-05
```

*Rewrite* (multiply by the conjugate):
$$
\sqrt{x+1}-\sqrt x = \frac{(x+1) - x}{\sqrt{x+1}+\sqrt x} = \frac{1}{\sqrt{x+1}+\sqrt x}.
$$
Now we *add* two positive numbers — no cancellation — and get `4.9999999875e-05`, correct to full precision.

---

## Example 1.5 — Condition number of a function

**Problem.** Compute the condition number of $f(x) = \ln x$ at $x = 1.001$ and interpret.

**Solution.** $\kappa_f(x) = \left|\frac{x f'(x)}{f(x)}\right| = \left|\frac{x\cdot(1/x)}{\ln x}\right| = \frac{1}{|\ln x|}$.
At $x = 1.001$: $\kappa = 1/\ln(1.001) \approx 1000.5$.

A relative perturbation of $10^{-6}$ in $x$ (e.g. measurement error) produces a relative change of about $10^{-3}$ in $\ln x$. **No algorithm can fix this** — the problem itself is ill-conditioned near $x=1$. (If your data really is $1+\delta$ with $\delta$ known accurately, compute `np.log1p(delta)` — then the input is $\delta$, and $\log(1+\delta)$ as a function of $\delta$ is well-conditioned.)

---

## Example 1.6 — Variance of data with a large mean

**Problem.** Compute the sample variance of $x = 10^9 + \{4, 7, 13, 16\}$ with (a) the one-pass textbook formula, (b) the two-pass formula, (c) Welford's algorithm.

**Solution.** The true variance: mean offset $10$, deviations $-6,-3,3,6$, so $s^2 = (36+9+9+36)/3 = 30$.

| Method | Result |
|---|---|
| (a) $\frac{1}{n-1}(\sum x_i^2 - n\bar x^2)$ | **$-170.67$** (negative!) |
| (b) two-pass $\frac{1}{n-1}\sum(x_i-\bar x)^2$ | $30.0$ |
| (c) Welford | $30.0$ |

In (a), $\sum x_i^2 \approx 4\times10^{18}$, whose spacing between doubles is $512$. Both terms are rounded to multiples of 512, and their difference (the true value $90$) is pure rounding noise.

---

## Example 1.7 — Stable softmax / log-sum-exp

**Problem.** Compute $\operatorname{softmax}(z)$ and $\log\sum_j e^{z_j}$ for $z = (1000, 1001, 1002)$.

**Solution.** Naively, `np.exp(1000)` overflows to `inf`, and `inf/inf = nan`.

Shift by $m = \max z = 1002$:
$$
\log\sum_j e^{z_j} = 1002 + \log\bigl(e^{-2} + e^{-1} + 1\bigr) = 1002 + \log(1.503214724) = 1002.407606.
$$
$$
\operatorname{softmax}(z) = \frac{(e^{-2}, e^{-1}, 1)}{1.503214724} = (0.0900, 0.2447, 0.6652).
$$

---

## Example 1.8 — Estimating the order of a method from data

**Problem.** A numerical method produced errors $E(0.1) = 4.0\times10^{-3}$, $E(0.05) = 1.0\times10^{-3}$, $E(0.025) = 2.5\times10^{-4}$. What is its order? Predict $E(0.0125)$.

**Solution.** $p \approx \dfrac{\log(E(h_1)/E(h_2))}{\log(h_1/h_2)} = \dfrac{\log 4}{\log 2} = 2$ for both pairs. The method is second order; halving $h$ divides the error by $4$, so $E(0.0125)\approx 6.25\times10^{-5}$.

---

## Example 1.9 — Forward vs backward error

**Problem.** A program claims $\sqrt{2} \approx \hat y = 1.4142$. Compute the forward and backward errors.

**Solution.**
*Forward error:* $|\hat y - \sqrt2| = |1.4142 - 1.41421356| = 1.36\times10^{-5}$.
*Backward error:* $\hat y$ is the exact square root of $\hat y^2 = 1.99996164$, so the backward error is $|2 - 1.99996164| = 3.84\times10^{-5}$ (relative $1.92\times10^{-5}$).
*Check with conditioning:* $\kappa_{\sqrt{\cdot}} = 1/2$, and relative forward error $9.6\times10^{-6} \approx \tfrac12 \times 1.92\times10^{-5}$ ✓.
