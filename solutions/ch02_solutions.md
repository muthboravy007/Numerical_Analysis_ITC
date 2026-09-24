# Chapter 2 — Solutions

> Exercises: [`exercises/ch02_exercises.md`](../exercises/ch02_exercises.md) · Numbers from [`solutions/code/ch02_solutions.py`](code/ch02_solutions.py).

## A. Theory and proofs

**A1.** After $n$ halvings the bracket $[a_n,b_n]$ has length $(b-a)/2^{n-1}$ and contains $p$; the midpoint $p_n$ is within half of that: $|p_n-p|\le(b-a)/2^n$. We need $(b-a)/2^n\le\varepsilon\iff n\ge\log_2\frac{b-a}{\varepsilon}$.

**A2.** (a) If $p\ne q$ are fixed points, the MVT gives $|p-q| = |g(p)-g(q)| = |g'(\xi)||p-q|\le k|p-q|<|p-q|$, a contradiction. (b) $g'(x) = 1 - \frac{f'(x)^2 - f(x)f''(x)}{f'(x)^2} = \frac{f(x)f''(x)}{f'(x)^2}$, so $g'(p) = 0$ when $f(p) = 0\ne f'(p)$. Then $p_{n+1}-p = g(p_n)-g(p) = g'(p)(p_n-p) + \frac12g''(\xi)(p_n-p)^2 = \frac12g''(\xi)(p_n-p)^2$ — at least quadratic.

**A3.** $f' = (x-p)^{m-1}[mq + (x-p)q']$, so $g(x) = x - \frac{(x-p)q}{mq+(x-p)q'}$ and $g'(p) = 1 - \frac1m\ne0$: linear convergence with constant $1-\frac1m$. With $g_m(x) = x - m\frac{f}{f'}$: $g_m'(p) = 1 - m\cdot\frac1m = 0$, so convergence is quadratic again.

**A4.** From $|e_{n+1}|\approx K|e_n|^\alpha$ and $|e_n|\approx K|e_{n-1}|^\alpha$ we get $|e_{n-1}|\approx(|e_n|/K)^{1/\alpha}$. Substituting into $|e_{n+1}|\approx C|e_n||e_{n-1}|$: $K|e_n|^\alpha\approx C K^{-1/\alpha}|e_n|^{1+1/\alpha}$. Matching powers, $\alpha = 1+\frac1\alpha$, i.e. $\alpha^2 = \alpha+1$, $\alpha = \frac{1+\sqrt5}2\approx1.618$.

**A5.** $\Delta p_n = c\lambda^n(\lambda-1)$, $\Delta^2p_n = c\lambda^n(\lambda-1)^2$, so $\frac{(\Delta p_n)^2}{\Delta^2p_n} = c\lambda^n$ and $\hat p_n = p + c\lambda^n - c\lambda^n = p$ exactly. A linearly convergent sequence behaves asymptotically like $p + c\lambda^n$ (with $\lambda$ the rate), so Aitken removes the dominant error term; what remains is of higher order (Theorem 2.14).

## B. Hand computation

**B1.** $f(1) = -1<0$, $f(2) = 9>0$ ⇒ root in $(1,2)$. Midpoints: $p_1 = 1.5$ ($f = 2.375$), $p_2 = 1.25$ ($0.328$), $p_3 = 1.125$ ($-0.420$), $p_4 = 1.1875$ ($-0.0676$), so the root is in $[1.1875, 1.25]$ (true root $1.198691$). For $10^{-4}$: $n\ge\log_210^4 = 13.3$ ⇒ **14 iterations**.

**B2.** (a) $g' = 2x$, $|g'(2)| = 4>1$ — diverges: $2.5, 4.25, 16.06, 256, 6.6\times10^4,\ldots$
(b) $g' = \frac{1}{2\sqrt{x+2}}$, $|g'(2)| = \frac14$ — converges linearly: $2.5, 2.12132, 2.03010, 2.00751, 2.00188, 2.00047, 2.00012$ (errors shrink by ≈¼).
(c) $g' = -\frac2{x^2}$, $|g'(2)| = \frac12$ — converges, oscillating: $2.5, 1.8, 2.1111, 1.9474, 2.0270, 1.9867, 2.0067$.

**B3.** $x_{n+1} = x_n - \frac{x_n^3-5}{3x_n^2} = \frac{2x_n^3+5}{3x_n^2}$. Iterates: $2,\ 1.75,\ 1.7108844,\ 1.7099764,\ 1.709975946677,\ 1.709975946677$ — correct digits ≈ 1, 2, 3, 6, 12, 16 (quadratic).

**B4.** $p_2 = 0.6126998$, $p_3 = 0.6934401$, $p_4 = 0.7038381$, $p_5 = 0.7034659$, $p_6 = 0.70346742$, $p_7 = 0.7034674225$, $p_8 = 0.7034674225$ ($|p_8-p_7|<10^{-10}$). Root $p = 0.7034674225$.

**B5.** False position: $0.6126998,\ 0.6934401,\ 0.7023831,\ 0.7033504$. It needs 14 iterations for $10^{-12}$ vs 7 for the secant method: the endpoint $b = 1$ never moves (the function is convex on the bracket), so convergence is only linear.

**B6.** $p_n$: $2.5, 2.12132, 2.03010, 2.00751, 2.00188,\ldots$ (errors ≈ $0.12, 0.03, 0.0075, 0.0019$). Aitken: $\hat p_0 = 2.00116$, $\hat p_1 = 2.000074$, $\hat p_2 = 2.0000047$, $\hat p_3 = 2.00000029$ — errors about 100–400 times smaller.

**B7.** (a) Synthetic division by $x-2$: coefficients $1, -3, 1, 1, 1$ → $b = 1, -1, -1, -1$, remainder $P(2) = -1$; second pass gives $P'(2) = 1$; quotient $x^3 - x^2 - x - 1$. (b) Newton–Horner from $2.5$: $2.34184, 2.29336, 2.28883, 2.2887950, 2.28879499219$ → largest root $2.288795$. The other roots are $1.389391$ and $-0.339093\pm0.446630i$. (c) After deflating the two real roots, the remaining quadratic gives the complex pair; alternatively, Müller's method (parabola through three points, complex square root) converges to complex roots from real starts. (Started at $0, 0.5, 1$ it converges to the real root $1.389391$; started at $-0.5, -0.3, 0$ it converges to $-0.339093-0.446630i$ although all three starting values are real.)

**B8.** $x_{n+1} = x_n - \frac{(x_n-1)^3}{3(x_n-1)^2} = x_n - \frac{x_n-1}{3}$, so $x_{n+1}-1 = \frac23(x_n-1)$: $2, 1.6667, 1.4444, 1.2963, 1.1975, 1.1317, 1.0878$ — ratio exactly $\frac23 = 1-\frac13$. Modified: $x_1 = 2 - 3\cdot\frac{1}{3} = 1$ exactly.

## C. Programming (key results)

**C1.** Root $p = 1.895494267034$. Iterations to $10^{-12}$: bisection on $[1.5,2.5]$: 40; Newton from $2$: 5; secant from $1.5, 2.5$: 7.

**C2.** Estimated orders: Newton $1.96, 2.00$; secant (asymptotic part) $1.88, 1.57, 1.64, 1.61$ → $1.618$. Early estimates are meaningless (pre-asymptotic).

**C3.** Fractions on the square grid: $0.353$ (root $1$), $0.323$, $0.323$ (complex roots). Theory says the basins are congruent under $120°$ rotation; the square window is not rotation-invariant, which biases the fractions. Every boundary point touches all three basins (Wada property) — a fractal boundary, showing why Newton needs good starting values.

**C4.** Safeguarded Newton on $[-2,5]$ starts at the midpoint $1.5$ and produces $-1.694,\ -0.0970,\ 6.1\times10^{-4},\ -1.5\times10^{-10},\ 0$: every Newton step stayed inside the shrinking bracket, so no bisection fallback was needed, yet the bracket guarantees that a wild step (like pure Newton's jump from $x_0 = 5$ to $-45$) would be rejected. Pure Newton from $5$ diverges (Example 2.3.5).

**C5.** `np.roots` max error $2.8\times10^{-9}$; deflation smallest-first max error $4\times10^{-11}$. Deflating the *largest* roots first gives max error $9.8\times10^{-10}$ — about 25× worse — because rounding errors in the quotient coefficients perturb the remaining (smaller) roots. The companion-matrix approach is backward stable for the *matrix*, but polynomial roots are ill-conditioned in the coefficients (Wilkinson), hence ~$10^{-9}$ errors despite double precision.

## D. Data-science applications

**D1.** $\text{NPV}(r) = -100 + \frac{230}{1+r} - \frac{132}{(1+r)^2} = 0$; with $v = 1/(1+r)$: $-132v^2 + 230v - 100 = 0$, $v = \frac{230\pm\sqrt{230^2-52800}}{264} = \frac{230\pm10}{264}$, so $v = 0.9091$ or $0.8333$, i.e. $r = 10\%$ and $r = 20\%$. NPV is $-2$ at $r = 0$, positive ($0.19$) between the roots and negative beyond. With more than one sign change in the cash flows, Descartes' rule allows multiple IRRs; report NPV at the relevant discount rate (or a modified IRR) instead.

**D2.** Newton with $h(\lambda) = \frac{\lambda}{1-e^{-\lambda}} - 1.8$, $h'(\lambda) = \frac{1-e^{-\lambda}-\lambda e^{-\lambda}}{(1-e^{-\lambda})^2}$, from $\lambda_0 = \bar x = 1.8$: $1.33765, 1.31841, 1.3183739, 1.31837394$. $\hat\lambda = 1.3184$ — smaller than $\bar x$ because zeros were not observed.

**D3.** From $x_0 = 6$: $9.331, 11.396, 12.384, 12.584, 12.591579, 12.5915872$ → $12.5915872$, equal to `gamma.ppf(0.95, 3, scale=2)`. (Convergence is slower at first because the CDF is flat/curved far from the answer; a better start is the mean + 2 s.d. $= 6 + 2\sqrt{12} = 12.9$.)

**D4.** $H(\beta)$ decreases monotonically from $\log_26$ ($\beta = 0$) to $0$ ($\beta\to\infty$), so bisection on $[10^{-3}, 20]$ is guaranteed: $\beta = 1.13382$ after 38 iterations, entropy $1.58496 = \log_23$, probabilities $(0.517, 0.293, 0.167, 0.017, 0.006, 0.0001)$. t-SNE solves this for every one of possibly millions of points; robustness and no need for derivatives make bisection the safe default.
