# Chapter 6 — Worked Examples: Direct Methods for Linear Systems

Companion script: [`ch06_examples.py`](ch06_examples.py) reproduces every number below (timings are machine-dependent). · Lecture: [Chapter 6](../lectures/ch06-direct-linear-systems.md)

---

## Example 6.1 — Leontief input–output economics

**Problem.** An economy has three sectors: agriculture, manufacturing and services. Producing one unit of each requires the inputs in the columns of
$$A = \begin{pmatrix}0.2&0.3&0.1\\0.1&0.1&0.3\\0.2&0.2&0.1\end{pmatrix}.$$
Final demand is $\mathbf d = (50, 80, 120)$. Find the gross output $\mathbf x$ with $\mathbf x = A\mathbf x+\mathbf d$.

**Solution.** Solve $(I-A)\mathbf x = \mathbf d$ by Gaussian elimination: $\mathbf x = (153.83, 174.77, 206.36)$. The Leontief inverse is
$$(I-A)^{-1} = \begin{pmatrix}1.402&0.542&0.336\\0.280&1.308&0.467\\0.374&0.411&1.290\end{pmatrix},$$
with column sums (output multipliers) $2.06$, $2.26$ and $2.09$. An extra 10 units of demand for services requires $(3.36, 4.67, 12.90)$ more output from the three sectors.

**Take-away.** Here the inverse matrix itself is the object of interest: its entries are economic multipliers. For merely *solving* the system, never form the inverse (Example 6.2).

---

## Example 6.2 — Factor once, solve many times

**Problem.** Solve $B\mathbf x_j = \mathbf r_j$ for $500$ right-hand sides with $n = 1000$.

**Solution.**

| strategy | time |
|---|---|
| LU once, then 500 triangular solves | **0.04 s** |
| solving from scratch each time | ≈ 8.8 s (extrapolated) |
| explicit inverse, then multiply | 0.08 s |

The residuals are $1.1\times10^{-15}$ (LU) and $1.4\times10^{-15}$ (inverse). The LU factorisation costs $\frac23n^3 = 6.7\times10^8$ flops, while each solve costs only $2n^2 = 2\times10^6$.

**Take-away.** In data science the same matrix recurs constantly: a kernel matrix for many targets, a covariance matrix in every iteration, a leave-one-out computation. Factor once, reuse the factors, and prefer LU or Cholesky to the inverse, which is slower and usually slightly less accurate.

---

## Example 6.3 — Is this correlation matrix valid?

**Problem.** Pairwise-estimated correlations are $\rho_{12} = 0.9$, $\rho_{13} = 0.7$ and $\rho_{23} = 0.3$. Can they come from real data?

**Solution.** Cholesky factorisation fails, with "not positive definite". The eigenvalues are $-0.0073, 0.7106, 2.2967$, and $\det = -0.012<0$. The matrix is not a valid correlation matrix: if $X_1$ is strongly correlated with both $X_2$ and $X_3$, then $X_2$ and $X_3$ cannot be almost uncorrelated. Repair it by clipping the negative eigenvalue to about $0$ and rescaling to unit diagonal. This gives $\rho_{12} = 0.894$, $\rho_{13} = 0.696$, $\rho_{23} = 0.301$, which is (numerically) PSD, and Cholesky now succeeds.

**Take-away.** Cholesky is the cheapest test for positive definiteness, costing $n^3/3$ flops. Pairwise-deletion estimates with missing data, or hand-specified stress-test scenarios, often yield invalid matrices. Higham's nearest-correlation-matrix algorithm is the principled fix.

---

## Example 6.4 — Tridiagonal systems in linear time

**Problem.** Solve $\operatorname{tridiag}(-1, 2.01, -1)\mathbf x = \mathbf 1$ for $n = 10^3$, $10^5$ and $10^6$.

**Solution.**

| $n$ | Thomas (Python loops) | LAPACK banded | memory a dense matrix would need |
|---|---|---|---|
| $10^3$ | 0.002 s | 0.0004 s | 8 MB |
| $10^5$ | 0.16 s | 0.003 s | 80 GB |
| $10^6$ | 1.8 s | 0.08 s | 8 TB |

The two solutions agree to $10^{-13}$.

**Take-away.** Exploit structure: $O(n)$ instead of $O(n^3)$ work, and $O(n)$ instead of $O(n^2)$ memory. Tridiagonal systems arise in cubic splines, 1-D diffusion and Whittaker smoothing.

---

## Example 6.5 — Log-likelihood of a multivariate normal

**Problem.** Evaluate $\log p(\mathbf x) = -\frac12\bigl(p\ln2\pi+\ln\det\Sigma+\mathbf x^T\Sigma^{-1}\mathbf x\bigr)$ for $p = 600$, with the eigenvalues of $\Sigma$ in $[0.01, 0.5]$.

**Solution.** `det(Σ)` underflows to $0.0$, so $\ln\det = -\infty$ and the naive formula fails. With $\Sigma = LL^T$, however, $\ln\det\Sigma = 2\sum\ln\ell_{ii} = -969.03$ (matches `slogdet`), and $\mathbf x^T\Sigma^{-1}\mathbf x = \|L^{-1}\mathbf x\|^2$ needs only one triangular solve. The log-likelihood is $-367.022$, identical to `scipy.stats.multivariate_normal`.

**Take-away.** Gaussian processes, Kalman filters and Gaussian mixture models all evaluate this quantity thousands of times. The Cholesky route is stable, costs $p^3/3$ flops, and never forms $\Sigma^{-1}$ or $\det\Sigma$.

---

## Example 6.6 — Fill-in and the importance of ordering

**Problem.** An "arrow" matrix of size $2000$ has a dense first row and column and a diagonal elsewhere (5998 nonzeros). How many nonzeros do the LU factors have?

**Solution.**

| ordering | nonzeros in $L+U$ |
|---|---|
| natural (dense row and column first) | **4 002 000** (complete fill-in) |
| COLAMD reordering | 7 998 |
| dense row and column moved last | 7 998 |

Eliminating the first variable couples every other variable to every other one. Eliminating it last creates no fill-in.

**Take-away.** For sparse matrices the *order* of elimination decides whether a direct solver is feasible. Sparse solvers (SuperLU, CHOLMOD) always reorder first, using minimum degree or nested dissection.
