# Chapter 3 — Solutions

> Exercises: [`exercises/ch03_exercises.md`](../exercises/ch03_exercises.md) · Numbers from [`solutions/code/ch03_solutions.py`](code/ch03_solutions.py).

## A. Theory and proofs

**A1.** If $P, Q\in\mathcal P_n$ both interpolate, $P-Q\in\mathcal P_n$ vanishes at $n+1$ distinct points, so it is the zero polynomial. The function $1$ is a polynomial of degree $0\le n$ interpolating the data $y_k = 1$; by uniqueness it equals $\sum_k1\cdot L_{n,k}(x)$.

**A2.** By Theorem 3.3, $f(x) - P_1(x) = \frac{f''(\xi)}{2}(x-x_0)(x-x_1)$. The quadratic $|(x-x_0)(x-x_1)|$ is maximal at the midpoint, where it equals $\frac h2\cdot\frac h2 = \frac{h^2}{4}$. Hence $|f-P_1|\le\frac{h^2}{8}\max|f''|$.

**A3.** The interpolating polynomial does not depend on the order of the nodes, and $f[x_0,\dots,x_n]$ is its leading coefficient (coefficient of $x^n$ in the Newton form) — hence symmetric. Let $g(x) = f(x) - P_n(x)$; $g$ has $n+1$ zeros, so by generalised Rolle $g^{(n)}(\xi) = 0$, i.e. $f^{(n)}(\xi) = P_n^{(n)} = n!\,f[x_0,\dots,x_n]$. For a polynomial of degree $n$ with leading coefficient $a_n$: $f^{(n)}\equiv n!a_n$, so $f[x_0..x_n] = a_n$.

**A4.** Let $e = g - S$; $e(x_i) = 0$. Then $\int(g'')^2 = \int(S'')^2 + 2\int S''e'' + \int(e'')^2$. On $[x_i,x_{i+1}]$ integrate by parts: $\int S''e'' = [S''e']_{x_i}^{x_{i+1}} - \int S'''e'$, and since $S'''$ is a constant $c_i$ there, $\int S'''e' = c_i[e(x_{i+1}) - e(x_i)] = 0$. Summing over $i$ the boundary terms telescope to $S''(b)e'(b) - S''(a)e'(a) = 0$ (natural conditions). Hence $\int(g'')^2 = \int(S'')^2+\int(e'')^2\ge\int(S'')^2$. ∎

**A5.** $\cos((n+1)\theta)+\cos((n-1)\theta) = 2\cos\theta\cos n\theta$ gives $T_{n+1}+T_{n-1} = 2xT_n$. Zeros: $\cos n\theta = 0\iff\theta = \frac{(2k-1)\pi}{2n}$, i.e. $x_k = \cos\frac{(2k-1)\pi}{2n}$, $k = 1..n$. Extrema: $\theta = \frac{k\pi}n$, $x_k' = \cos\frac{k\pi}n$, $T_n(x_k') = (-1)^k$, $k = 0..n$ — $n+1$ alternating values; scaling by $2^{1-n}$ (which makes the leading coefficient 1, since $T_n$ has leading coefficient $2^{n-1}$) gives alternation between $\pm2^{1-n}$.

## B. Hand computation

**B1.** $L_0 = \frac{(x-1)(x-3)}{3}$, $L_1 = -\frac{x(x-3)}{2}$, $L_2 = \frac{x(x-1)}{6}$; $P(x) = L_0 + 3L_1 + 2L_2 = -\frac56x^2 + \frac{17}6x + 1$. $P(2) = \frac{10}{3} = 3.3333$.

**B2.** With nodes $4, 9, 1, 16$ and $x = 5$:

| node | $Q_{i,0}$ | $Q_{i,1}$ | $Q_{i,2}$ | $Q_{i,3}$ |
|---|---|---|---|---|
| 4 | 2 | | | |
| 9 | 3 | 2.2 | | |
| 1 | 1 | 2.0 | 2.26667 | |
| 16 | 4 | 1.8 | 2.11429 | 2.25397 |

True $\sqrt5 = 2.23607$. The quadratic through $4, 9, 1$ ($2.2667$, error $0.031$) and the cubic ($2.2540$, error $0.018$) are best; $\sqrt x$ has large derivatives near $0$ and the node spacing is uneven, so accuracy is modest. The ordering by distance made the early entries informative.

**B3.** Table (top diagonal in bold): $f[\,] = \mathbf{-1}, 1, 1, 5$; first $\mathbf{2}, 0, 4$; second $\mathbf{-1}, 2$; third $\mathbf1$. $P_3(x) = -1 + 2(x+1) - (x+1)x + (x+1)x(x-1) = x^3 - x^2 + 1$. Adding $(3, 20)$ gives the new coefficient $f[x_0..x_4] = \frac1{24} = 0.0417$: the data are *not* from the cubic (which predicts $P_3(3) = 19$); $P_4 = P_3 + \frac1{24}(x+1)x(x-1)(x-2)$.

**B4.** $f = 0, 0.095310, 0.182322, 0.262364$; $\Delta f_0 = 0.095310$, $\Delta^2f_0 = -0.008298$, $\Delta^3f_0 = 0.001328$. With $s = 0.5$: $\binom{0.5}{1} = 0.5$, $\binom{0.5}{2} = -0.125$, $\binom{0.5}{3} = 0.0625$, so $P(1.05) = 0.5(0.095310) - 0.125(-0.008298) + 0.0625(0.001328) = 0.047655 + 0.001037 + 0.000083 = 0.048775$. True $\ln1.05 = 0.048790$; error $1.5\times10^{-5}$ (partly from the 6-decimal table).

**B5.** Data: $f(1) = 1$, $f'(1) = -1$, $f(2) = 0.5$, $f'(2) = -0.25$. With $z = 1,1,2,2$: coefficients $1, -1, 0.5, -0.25$, so $H(x) = 1 - (x-1) + 0.5(x-1)^2 - 0.25(x-1)^2(x-2)$. $H(1.5) = 0.65625$; true $0.66667$; error $0.0104$. Bound: $\frac{\max|f^{(4)}|}{4!}(x-1)^2(x-2)^2 = \frac{24}{24}(0.25)(0.25) = 0.0625$ ✓.

**B6.** $h = 1$: $4M_1 = 6[(0-2)-(2-1)] = -18$ ⇒ $M_1 = -4.5$, $M_0 = M_2 = 0$.
$S_0(x) = 1 + 1.75x - 0.75x^3$ on $[0,1]$; $S_1(x) = 2 - 0.5(x-1) - 2.25(x-1)^2 + 0.75(x-1)^3$ on $[1,2]$. $S(0.5) = 1.78125$, $S(1.5) = 1.28125$.

**B7.** Clamped equations: $2M_0 + M_1 = 6(1 - 0) = 6$; $M_0 + 4M_1 + M_2 = -18$; $M_1 + 2M_2 = 6(0 - (-2)) = 12$ ⇒ $M = (7.5, -9, 10.5)$. $S_0(x) = 1 + 0x + 3.75x^2 - 2.75x^3$; $S_1(x) = 2 - 0.75(x-1) - 4.5(x-1)^2 + 3.25(x-1)^3$. $S(0.5) = 1.59375$, $S(1.5) = 0.90625$. Forcing zero slopes at the ends makes the clamped spline flatter near the boundary and more curved inside (larger $|M_i|$).

**B8.** $t = \frac13$: level 1: $(0.3333, 1)$, $(2, 3)$, $(4.3333, 2)$; level 2: $(0.8889, 1.6667)$, $(2.7778, 2.6667)$; level 3: $B(\frac13) = (1.5185, 2)$.

## C. Programming (key results)

**C1.**

| $n$ | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| equispaced | 0.433 | 1.92 | 59.8 | $1.0\times10^5$ |
| Chebyshev | 0.556 | 0.109 | 0.0153 | $2.9\times10^{-4}$ |

**C2.** Equispaced $\Lambda_n$: $2.21, 10.9, 89.3, 935$ (exponential growth). Chebyshev: $1.99, 2.36, 2.60, 2.77$ — just below $\frac2\pi\ln(n+1)+1 = 2.03, 2.40, 2.63, 2.80$.

**C3.** Max errors $1.07\times10^{-3},\ 6.31\times10^{-5},\ 3.89\times10^{-6},\ 2.42\times10^{-7}$ (ratios ≈16 ⇒ $O(h^4)$); difference from SciPy's clamped spline $\le4\times10^{-16}$.

**C4.** RMSE on the missing days: linear $0.236$, cubic spline $0.0039$, PCHIP $0.073$. For smooth signals the $C^2$ spline is best; PCHIP trades accuracy for shape preservation.

## D. Data-science applications

**D1.** The cubic spline decreases on 452 of 1000 grid intervals (minimum slope $-1.06$), e.g. it overshoots to $36.3$ at $t = 3$ and must then come *down* to $32$ at $t = 5$ — impossible for cumulative counts. PCHIP (monotone piecewise cubic Hermite) has minimum slope $\ge0$ and gives $30.9$ at $t = 3$. Recommend PCHIP (or linear) whenever monotonicity or positivity must be preserved.

**D2.** Row 2 of the $8\times8$ result — nearest neighbour: $0, 0, 1, 1, 2, 2, 1, 1$ (blocky, repeated pixels); bilinear: $0, 0.37, 0.74, 1.14, 1.57, 1.74, 1.37, 1.00$ (smooth ramps). Bilinear removes blockiness at the cost of slight blurring; bicubic (order 3) sharpens edges further but may overshoot.

**D3.** The Chebyshev surrogate (9 evaluations) has maximum error $0.0071$ over $[-5,1]$; its maximiser is $\ell = -1.876$ (surrogate value $0.9033$) vs the true maximiser $-1.705$ (value $0.9075$). The next step in practice: evaluate the true function near $-1.8$ and refine the surrogate locally (sequential design). Bayesian optimisation replaces the polynomial by a Gaussian process (§6.7), whose uncertainty guides where to sample next.
