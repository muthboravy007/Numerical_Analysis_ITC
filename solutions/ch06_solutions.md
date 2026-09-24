# Chapter 6 — Solutions

> Exercises: [`exercises/ch06_exercises.md`](../exercises/ch06_exercises.md) · Numbers from [`solutions/code/ch06_solutions.py`](code/ch06_solutions.py).

## A. Theory and proofs

**A1.** At step $i$ ($i = 1..n-1$) we compute $n-i$ multipliers (divisions) and update $(n-i)$ rows of $n-i$ remaining entries plus the right-hand side: $(n-i)(n-i+1)$ multiplications. Total $\sum_{i=1}^{n-1}(n-i)(n-i+2) = \sum_{k=1}^{n-1}(k^2+2k) = \frac{2n^3+3n^2-5n}{6}$. Back substitution needs $1+2+\cdots+n = \frac{n^2+n}{2}$. Sum: $\frac{n^3}{3}+n^2-\frac n3$.

**A2.** Step $k$ computes $A^{(k+1)} = M^{(k)}A^{(k)}$ with $\mathbf m^{(k)} = (0,\dots,0,m_{k+1,k},\dots,m_{nk})^T$. So $U = M^{(n-1)}\cdots M^{(1)}A$ and $A = (M^{(1)})^{-1}\cdots(M^{(n-1)})^{-1}U$. The product $\prod_k(I+\mathbf m^{(k)}\mathbf e_k^T) = I + \sum_k\mathbf m^{(k)}\mathbf e_k^T$ (cross terms vanish because $\mathbf e_k^T\mathbf m^{(l)} = 0$ for $l\ge k$), which is unit lower triangular with the multipliers in column $k$.

**A3.** Suppose $A\mathbf x = \mathbf 0$, $\mathbf x\ne\mathbf 0$, and $|x_k| = \max|x_j|>0$. Row $k$: $a_{kk}x_k = -\sum_{j\ne k}a_{kj}x_j$, so $|a_{kk}||x_k|\le\sum_{j\ne k}|a_{kj}||x_j|\le|x_k|\sum_{j\ne k}|a_{kj}|$, contradicting strict diagonal dominance. Hence $A$ is nonsingular.

**A4.** (a) $a_{ii} = \mathbf e_i^TA\mathbf e_i>0$. (b) For a principal submatrix indexed by $I$, take $\mathbf x$ supported on $I$: $\mathbf x_I^TA_{II}\mathbf x_I = \mathbf x^TA\mathbf x>0$. (c) The $2\times2$ principal submatrix $\begin{pmatrix}a_{ii}&a_{ij}\\a_{ij}&a_{jj}\end{pmatrix}$ is PD, so its determinant $a_{ii}a_{jj}-a_{ij}^2>0$.

**A5.** $n = 1$: $\ell_{11} = \sqrt{a_{11}}$. For $n>1$, $a_{11}>0$ (A4), and
$$
A = \begin{pmatrix}\sqrt{a_{11}}&0\\\mathbf w/\sqrt{a_{11}}&I\end{pmatrix}\begin{pmatrix}1&0\\0&K-\mathbf w\mathbf w^T/a_{11}\end{pmatrix}\begin{pmatrix}\sqrt{a_{11}}&\mathbf w^T/\sqrt{a_{11}}\\0&I\end{pmatrix}.
$$
The Schur complement $S = K-\mathbf w\mathbf w^T/a_{11}$ is SPD (congruence preserves definiteness), so by induction $S = L_SL_S^T$, and $L = \begin{pmatrix}\sqrt{a_{11}}&0\\\mathbf w/\sqrt{a_{11}}&L_S\end{pmatrix}$. Uniqueness: the first column of $L$ is forced ($\ell_{11}>0$ and $\ell_{i1} = a_{i1}/\ell_{11}$), and by induction so is $L_S$.

## B. Hand computation

**B1.** $E_2-2E_1$, $E_3-3E_1$: $\left[\begin{smallmatrix}1&1&1&|&6\\0&1&-1&|&-1\\0&-2&1&|&-1\end{smallmatrix}\right]$; $E_3+2E_2$: $\left[\begin{smallmatrix}1&1&1&|&6\\0&1&-1&|&-1\\0&0&-1&|&-3\end{smallmatrix}\right]$. Back substitution: $x_3 = 3$, $x_2 = -1+3 = 2$, $x_1 = 6-2-3 = 1$.

**B2.** No pivoting: $m_{21} = 2.14/0.0012 = 1783$, $a_{22} = -1.78 - 1783(3.47) = -6189$, $b_2 = 2.5 - \mathrm{fl}(1783\cdot3.472) = 2.5 - 6191 \to -6189$, so $x_2 = 1.000$, and $x_1 = \frac{3.472 - 3.470}{0.0012} = 1.667$: the subtraction $3.472 - 3.470$ has only one significant digit left, and dividing by the tiny pivot magnifies it. Result $(1.667, 1.000)$ — 17% error in $x_1$. With partial pivoting (swap rows; $m = 0.0012/2.14 = 0.000561$): $(2.000, 1.000)$.

**B3.** Scale factors $s = (1000, 3, 4)$; ratios $|a_{i1}|/s_i = (0.002, 0.333, 1)$. No pivoting: $(-5.0, -1.36, 2.73)$ — useless. Partial pivoting (pivot $4$, row 3) and scaled pivoting (ratio 1, also row 3): $(1.00, -0.999, 2.00)$. Here both pivoting strategies pick row 3; partial pivoting would be fooled only if row 1 had a first entry larger than 4.

**B4.** $L = \begin{pmatrix}1&0&0\\2&1&0\\3&-2&1\end{pmatrix}$, $U = \begin{pmatrix}1&1&1\\0&1&-1\\0&0&-1\end{pmatrix}$, $\det A = 1\cdot1\cdot(-1) = -1$. Solutions: $(1,2,3)$ and, for $\mathbf e_1$, $(-11, 5, 7)$ (the first column of $A^{-1}$).

**B5.** Row 3 has the largest first entry: $P$ swaps rows 1 and 3. $L = \begin{pmatrix}1&0&0\\2/3&1&0\\1/3&2/7&1\end{pmatrix}$, $U = \begin{pmatrix}3&1&4\\0&7/3&-5/3\\0&0&1/7\end{pmatrix}$ (check: $\det = -(3\cdot\frac73\cdot\frac17) = -1$ with one row swap).

**B6.** $\ell_{11} = 3$, $\ell_{21} = 1$, $\ell_{31} = -1$, $\ell_{22} = \sqrt{5-1} = 2$, $\ell_{32} = (1-(-1)(1))/2 = 1$, $\ell_{33} = \sqrt{11-1-1} = 3$: $L = \begin{pmatrix}3&0&0\\1&2&0\\-1&1&3\end{pmatrix}$. $LDL^T$: $L = \begin{pmatrix}1&&\\1/3&1&\\-1/3&1/2&1\end{pmatrix}$, $D = \operatorname{diag}(9,4,9)$. Solution of $S\mathbf x = (3,9,9)$: $\mathbf x = (0, 5/3, 2/3)$.

**B7.** Minors: $2$, $3$, $\det T = 3k-2$. PD iff $k>\frac23$. Eigenvalues: $k = 1$: $0.198, 1.555, 3.247$; $k = 2$: $0.586, 2, 3.414$; $k = 2.5$: $0.687, 2.242, 3.571$; $k = 3$: $0.753, 2.445, 3.802$ — all positive, as predicted.

**B8.** $\ell = (4, 3.75, 3.7333, 3.7321)$, $u = (0.25, 0.2667, 0.2679)$, forward $z = (1.25, 1.2667, 1.2679, 1)$, back substitution $\mathbf x = (1,1,1,1)$.

**B9.** $A^{-1} = \frac18\begin{pmatrix}5&-2&1\\-2&4&-2\\1&-2&5\end{pmatrix}$ (dense, although $A$ is tridiagonal).

## C. Programming (key results)

**C1.** Pure-Python elimination: $0.017, 0.069, 0.30, 1.38$ s for $n = 100..800$ (≈×4.5 per doubling at these sizes, tending to ×8); LAPACK: $0.0008$–$0.017$ s. LAPACK is ~100× faster thanks to compiled, cache-blocked (BLAS-3) code — same $O(n^3)$ algorithm.

**C2.**

| $n$ | $\kappa_2$ | rel. error | rel. residual |
|---|---|---|---|
| 4 | $1.6\times10^4$ | $3\times10^{-14}$ | $0$ |
| 8 | $1.5\times10^{10}$ | $3.7\times10^{-8}$ | $6\times10^{-17}$ |
| 10 | $1.6\times10^{13}$ | $2.3\times10^{-5}$ | $1\times10^{-16}$ |
| 12 | $1.6\times10^{16}$ | $0.13$ | $1\times10^{-16}$ |
| 14 | $3\times10^{17}$ | $3.0$ | $5\times10^{-17}$ |

GEPP is backward stable: it solves a nearby system exactly, so the residual is at round-off level; the forward error is $\approx\kappa u$ ($\log_{10}(\kappa u)$ matches the error's exponent).

**C3.** Mean (max) growth: $n = 10$: $1.7$ ($3.9$); $50$: $3.6$ ($6.0$); $100$: $5.1$ ($8.3$); $200$: $7.2$ ($11.8$) — growing roughly like $n^{1/2}$, astronomically below $2^{n-1}$.

## D. Data-science applications

**D1.** Sample correlations $0.800, 0.302, 0.500$ ✓. Portfolio volatility $0.01719$ (theory $0.02\sqrt{\mathbf w^T\mathbf R\mathbf w} = 0.01720$); 99% one-day VaR (loss quantile) $= 4.01\%$ ≈ $2.33\sigma$, as expected for Gaussian returns.

**D2.** Posterior mean/sd: $x_\ast = 2.5$: $-0.785\pm0.082$ (true $-0.801$); $4.2$: $-0.467\pm0.061$ (true $-0.490$); $7.0$: $0.111\pm0.984$ (true $0.754$). Log marginal likelihood $-5.19$. At $x_\ast = 7$ (two length-scales outside the data) the GP reverts to its prior mean $0$ with prior variance $\approx1$: it honestly reports that it does not know.

**D3.** $N = \begin{pmatrix}4.130&3.261&1.304\\2.174&4.348&1.739\\1.304&2.609&3.043\end{pmatrix}$, $B = NR = \begin{pmatrix}0.576&0.424\\0.435&0.565\\0.261&0.739\end{pmatrix}$ (columns: leave, purchase). Purchase probability: $42.4\%$ from *browse*, $56.5\%$ from *cart*, $73.9\%$ from *checkout*. Expected steps before absorption $N\mathbf 1 = (8.70, 8.26, 6.96)$. (Compute $B$ and $N\mathbf 1$ by solving with $I-Q$, not by inverting, for large chains.)
