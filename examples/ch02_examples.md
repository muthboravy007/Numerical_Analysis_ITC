# Chapter 2 — Worked Examples: Root Finding

Companion script: [`ch02_examples.py`](ch02_examples.py) · Lecture: [Lecture 2](../lectures/02-root-finding.md)

Running example: $f(x) = x^3 - 2x - 5$ (Wallis's classical equation), root $r = 2.0945514815423265$.

---

## Example 2.1 — Bisection by hand

**Problem.** Apply bisection to $f$ on $[2,3]$. How many steps guarantee an error below $10^{-6}$?

**Solution.** $f(2) = -1 < 0$ and $f(3) = 16 > 0$, so a root is bracketed.

| $k$ | $a$ | $b$ | $m$ | $f(m)$ | sign |
|---|---|---|---|---|---|
| 1 | 2 | 3 | 2.5 | 5.625 | + → $b=m$ |
| 2 | 2 | 2.5 | 2.25 | 1.890625 | + → $b=m$ |
| 3 | 2 | 2.25 | 2.125 | 0.345703 | + → $b=m$ |
| 4 | 2 | 2.125 | 2.0625 | −0.351318 | − → $a=m$ |
| 5 | 2.0625 | 2.125 | 2.09375 | −0.008942 | − → $a=m$ |
| 6 | 2.09375 | 2.125 | 2.109375 | 0.166836 | + → $b=m$ |

Error bound after $n$ steps: $(b-a)/2^{n}$ for the endpoints (or $/2^{n+1}$ for the midpoint). We need $1/2^n < 10^{-6}$, i.e. $n > \log_2 10^6 = 19.93$ → **20 iterations**, which is exactly what the code performs.

---

## Example 2.2 — Newton's method and quadratic convergence

**Problem.** Apply Newton's method to $f$ from $x_0 = 2$. Verify quadratic convergence.

**Solution.** $f'(x) = 3x^2 - 2$, so $x_{k+1} = x_k - \dfrac{x_k^3 - 2x_k - 5}{3x_k^2 - 2}$.

| $k$ | $x_k$ | $\lvert x_k - r\rvert$ | $\lvert e_{k}\rvert/\lvert e_{k-1}\rvert^2$ |
|---|---|---|---|
| 0 | 2.0000000000000000 | $9.46\times10^{-2}$ | |
| 1 | 2.1000000000000001 | $5.45\times10^{-3}$ | 0.609 |
| 2 | 2.0945681211041851 | $1.66\times10^{-5}$ | 0.561 |
| 3 | 2.0945514816981992 | $1.56\times10^{-10}$ | 0.563 |
| 4 | 2.0945514815423265 | $0$ | |

The ratio settles at $\dfrac{f''(r)}{2f'(r)} = \dfrac{6r}{2(3r^2-2)} = \dfrac{12.567}{22.321} = 0.563$ ✓ — exactly as the theorem predicts. Correct digits: 1, 2, 5, 10, 16.

---

## Example 2.3 — Choosing a convergent fixed-point form

**Problem.** Which rearrangement of $x^3 - 2x - 5 = 0$ gives a convergent fixed-point iteration near $r\approx 2.09$?
(a) $g_a(x) = (x^3-5)/2$; (b) $g_b(x) = (2x+5)^{1/3}$; (c) $g_c(x) = 5/(x^2-2)$.

**Solution.** Check $|g'(r)|$:

* (a) $g_a'(x) = \tfrac32x^2$, $|g_a'(r)| = 6.58 > 1$ → **diverges**.
* (b) $g_b'(x) = \tfrac23(2x+5)^{-2/3}$, $|g_b'(r)| = \tfrac23 r^{-2} = 0.152 < 1$ → **converges linearly**, error ×0.152 per step. (We used $2r+5 = r^3$.)
* (c) $g_c'(x) = -10x/(x^2-2)^2$, $|g_c'(r)| = 20.95/5.70 = 3.68 > 1$ → **diverges**.

Iterating (b) from $x_0=2$: `2.0, 2.08008, 2.09235, 2.09422, 2.09450, 2.09454, …` — each error is ≈ 0.15 × the previous one; tolerance $10^{-8}$ is reached after 10 iterations.

---

## Example 2.4 — Secant method: internal rate of return

**Problem.** An investment costs \$1000 and returns \$300, \$400, \$500 at the end of years 1–3. Find the internal rate of return $r$, the root of
$$
\text{NPV}(r) = -1000 + \frac{300}{1+r} + \frac{400}{(1+r)^2} + \frac{500}{(1+r)^3}.
$$

**Solution.** $\text{NPV}(0) = 200 > 0$ and $\text{NPV}(0.2) = -182.87 < 0$. Secant from $x_0 = 0$, $x_1 = 0.2$:

| $k$ | $x_k$ |
|---|---|
| 2 | 0.10447400 |
| 3 | 0.08620318 |
| 4 | 0.08903076 |
| 5 | 0.08896369 |
| 6 | 0.08896339 |
| 7 | 0.08896339 |

**IRR ≈ 8.896%.** Bisection on $[0, 0.2]$ with tolerance $10^{-6}$ takes 18 iterations to reach the same answer; secant needs 7 function evaluations and no derivative.

---

## Example 2.5 — Newton for a quantile

**Problem.** Find $z_{0.95}$, the 95th percentile of the standard normal distribution, by solving $\Phi(z) - 0.95 = 0$.

**Solution.** The derivative of the CDF is the PDF: $f'(z) = \varphi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$, so
$$
z_{k+1} = z_k - \frac{\Phi(z_k) - 0.95}{\varphi(z_k)}.
$$
From $z_0 = 1.5$: $1.6297677,\ 1.6446691,\ 1.6448536,\ 1.6448536270$. **$z_{0.95} = 1.6448536$** ✓ (the familiar "1.645"). This is exactly how `scipy.stats.norm.ppf` style routines refine quantiles when no closed form exists.

---

## Example 2.6 — Maximum likelihood: Gamma shape parameter

**Problem.** For i.i.d. data from Gamma$(\alpha, \theta)$ the MLE of $\alpha$ solves
$$
g(\alpha) = \ln\alpha - \psi(\alpha) - s = 0,\qquad s = \ln\bar x - \overline{\ln x},
$$
where $\psi$ is the digamma function. Solve for the data
`2.3, 1.1, 4.5, 3.2, 0.8, 2.9, 5.1, 1.7, 2.2, 3.6`.

**Solution.** $\bar x = 2.74$, $\overline{\ln x} = 0.866627$, so $s = \ln 2.74 - 0.866627 = 0.141330$.
Derivative: $g'(\alpha) = 1/\alpha - \psi'(\alpha)$ (trigamma).
A good starting value (Minka's approximation) is $\alpha_0 = \dfrac{3 - s + \sqrt{(s-3)^2 + 24s}}{12s} = 3.69067$.

Newton: $3.6906686 \to 3.6961941 \to 3.6962027 \to 3.6962027$.

**$\hat\alpha = 3.6962$, $\hat\theta = \bar x/\hat\alpha = 0.7413$.** Three iterations — quadratic convergence from a good start.

---

## Example 2.7 — Newton at a double root

**Problem.** Apply Newton to $f(x) = (x-1)^2$ from $x_0 = 2$. What happens and how can it be fixed?

**Solution.** $x_{k+1} = x_k - \frac{(x_k-1)^2}{2(x_k-1)} = x_k - \frac{x_k-1}{2}$, so $e_{k+1} = e_k/2$: iterates $2, 1.5, 1.25, 1.125, \ldots$ — only **linear** convergence (40 iterations to reach $10^{-12}$).
For a root of known multiplicity $m$, the modified iteration $x_{k+1} = x_k - m\,f(x_k)/f'(x_k)$ restores quadratic convergence; here, with $m=2$, it lands on $x=1$ in one step.

---

## Example 2.8 — Newton's method for a 2×2 system

**Problem.** Solve $x^2 + y^2 = 4$, $xy = 1$ from $(x_0, y_0) = (2, 0.5)$.

**Solution.** $\mathbf F = \begin{pmatrix}x^2+y^2-4\\xy-1\end{pmatrix}$, $J = \begin{pmatrix}2x&2y\\y&x\end{pmatrix}$.

*Step 1.* $\mathbf F(2,0.5) = (0.25, 0)$, $J = \begin{pmatrix}4&1\\0.5&2\end{pmatrix}$. Solve $J\Delta = -\mathbf F$: $\det J = 7.5$,
$\Delta = \frac{1}{7.5}\begin{pmatrix}2&-1\\-0.5&4\end{pmatrix}\begin{pmatrix}-0.25\\0\end{pmatrix} = \begin{pmatrix}-0.066667\\0.016667\end{pmatrix}$ → $(1.933333, 0.516667)$.

Subsequent iterates: $(1.931853, 0.517637)$, $(1.931852, 0.517638)$.
Exact: $x = \sqrt{2+\sqrt3} = 1.9318517$, $y = \sqrt{2-\sqrt3} = 0.5176381$ ✓.
