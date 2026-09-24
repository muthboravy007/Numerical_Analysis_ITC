# Chapter 6 — Exercises: Direct Methods for Solving Linear Systems

> Lecture: [Chapter 6](../lectures/ch06-direct-linear-systems.md) · Solutions: [`solutions/ch06_solutions.md`](../solutions/ch06_solutions.md) · Solution code: [`solutions/code/ch06_solutions.py`](../solutions/code/ch06_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★★** Derive the operation count $\frac{n^3}{3}+n^2-\frac n3$ multiplications/divisions for Gaussian elimination with backward substitution.

**A2 ★★** Show that if Gaussian elimination is performed without row interchanges, then $A = LU$ with $L$ unit lower triangular containing the multipliers. (Hint: each elimination step is multiplication by an elementary matrix $M^{(k)} = I - \mathbf m^{(k)}\mathbf e_k^T$ with $(M^{(k)})^{-1} = I + \mathbf m^{(k)}\mathbf e_k^T$.)

**A3 ★★** Prove that a strictly diagonally dominant matrix is nonsingular. (Hint: if $A\mathbf x = \mathbf 0$, look at the component with largest $|x_i|$.)

**A4 ★★** Show that if $A$ is symmetric positive definite then (a) $a_{ii}>0$; (b) every principal submatrix is PD; (c) $a_{ij}^2<a_{ii}a_{jj}$ for $i\ne j$.

**A5 ★★★** Prove that the Cholesky factorisation of a symmetric positive definite matrix exists and is unique, by induction on $n$ using the block form $A = \begin{pmatrix}a_{11}&\mathbf w^T\\\mathbf w&K\end{pmatrix}$ and the Schur complement $K - \mathbf w\mathbf w^T/a_{11}$.

## B. Hand computation

**B1 ★** Solve $x_1+x_2+x_3 = 6$, $2x_1+3x_2+x_3 = 11$, $3x_1+x_2+4x_3 = 17$ by Gaussian elimination, showing the augmented matrix after each step.

**B2 ★★** Solve $0.0012x_1 + 3.47x_2 = 3.4724$, $2.14x_1 - 1.78x_2 = 2.5$ (exact solution $(2,1)$) in four-digit rounding arithmetic without pivoting and with partial pivoting.

**B3 ★★** The system $2x_1 + 1000x_2 + 500x_3 = 2$, $x_1 + 2x_2 + 3x_3 = 5$, $4x_1 + x_2 - x_3 = 1$ has solution $(1,-1,2)$. Solve it in three-digit arithmetic with no pivoting, partial pivoting and scaled partial pivoting, listing the scale factors and ratios used.

**B4 ★** Find the $LU$ factorisation (Doolittle) of the matrix of B1, compute $\det A$ from it, and use it to solve for the right-hand sides $(6,11,17)^T$ and $(1,0,0)^T$.

**B5 ★★** Compute $PA = LU$ with partial pivoting for the matrix of B1.

**B6 ★★** Compute the Cholesky and $LDL^T$ factorisations of $S = \begin{pmatrix}9&3&-3\\3&5&1\\-3&1&11\end{pmatrix}$ and solve $S\mathbf x = (3,9,9)^T$.

**B7 ★★** For which $k$ is $T = \begin{pmatrix}2&-1&0\\-1&2&-1\\0&-1&k\end{pmatrix}$ positive definite? Use leading principal minors and confirm with eigenvalues for $k = 1, 2, 2.5, 3$.

**B8 ★** Solve the tridiagonal system with diagonal $4$, off-diagonals $1$ ($4\times4$) and right-hand side $(5,6,6,5)^T$ by the Crout/Thomas algorithm.

**B9 ★** Find $A^{-1}$ for $A = \begin{pmatrix}2&1&0\\1&3&1\\0&1&2\end{pmatrix}$ by Gauss–Jordan elimination.

## C. Programming

**C1 ★** Time your own Gaussian elimination (Python loops) against `np.linalg.solve` for $n = 100, 200, 400, 800$. Verify the $O(n^3)$ growth and explain the constant-factor difference.

**C2 ★★** Solve $H_n\mathbf x = H_n\mathbf 1$ for Hilbert matrices $n = 4,\dots,14$. Tabulate $\kappa_2(H_n)$, the relative error and the relative residual. Explain why the residual stays tiny while the error explodes.

**C3 ★★** Estimate the growth factor $\max|u_{ij}|/\max|a_{ij}|$ of GEPP for 50 random Gaussian matrices of sizes $10, 50, 100, 200$. Compare with the worst-case bound $2^{n-1}$.

## D. Data-science applications

**D1 ★★** Simulate $2\times10^5$ daily returns of three assets with volatility 2% and correlation matrix $\begin{pmatrix}1&0.8&0.3\\0.8&1&0.5\\0.3&0.5&1\end{pmatrix}$ via Cholesky. Check the sample correlations, and estimate the volatility and 99% Value-at-Risk of the portfolio $(0.5, 0.3, 0.2)$.

**D2 ★★** Gaussian-process regression: training inputs $x = 0,1,\dots,5$, targets $\cos x$, squared-exponential kernel with length-scale 1 and noise variance $10^{-4}$. Using a single Cholesky factorisation compute the posterior mean and standard deviation at $x_\ast = 2.5, 4.2, 7.0$ and the log marginal likelihood. Interpret the result at $x_\ast = 7$.

**D3 ★★** A customer-journey Markov chain has transient states *browse*, *cart*, *checkout* with
$Q = \begin{pmatrix}0.6&0.3&0\\0.2&0.5&0.2\\0&0.3&0.5\end{pmatrix}$ and absorbing states *leave*, *purchase* with $R = \begin{pmatrix}0.1&0\\0.05&0.05\\0&0.2\end{pmatrix}$. Compute the fundamental matrix $N = (I-Q)^{-1}$, the purchase probability from each state and the expected number of steps before absorption.
