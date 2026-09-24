# Lecture 3 — Direct Methods for Linear Systems

> **Week 4–5** · Code: [`numlib/linalg_direct.py`](../numlib/linalg_direct.py) · Examples: [`examples/ch03`](../examples/ch03_examples.md) · Exercises: [`exercises/ch03`](../exercises/ch03_exercises.md)

## Learning objectives

1. Solve triangular systems and count their cost.
2. Perform Gaussian elimination and interpret it as an **LU factorisation**.
3. Explain why **pivoting** is necessary and compute $PA = LU$.
4. Use the **Cholesky factorisation** for symmetric positive definite (SPD) matrices.
5. Define vector and matrix norms and the **condition number** $\kappa(A)$; bound the error in a computed solution.
6. Exploit structure (banded, tridiagonal, sparse) and know when to call which library routine.

---

## 3.1 Where linear systems appear in data science

* **Linear regression:** normal equations $X^\top X\,\beta = X^\top y$.
* **Gaussian processes / kriging:** solve $(K + \sigma^2 I)\alpha = y$ with a kernel matrix $K$.
* **Newton's method** (Lectures 2, 9): one linear solve per iteration.
* **Multivariate Gaussians:** evaluating $\log\det\Sigma$ and $x^\top\Sigma^{-1}x$.
* **Markov chains:** stationary distributions, absorption probabilities.
* **Splines and PDEs** (Lectures 7, 10): banded systems.

Throughout, $A \in \mathbb R^{n\times n}$ is nonsingular and we solve $A\mathbf x = \mathbf b$.

> **Golden rule.** Never compute $A^{-1}$ to solve $A\mathbf x=\mathbf b$. Solving via a factorisation is ~3× cheaper and more accurate. Write `np.linalg.solve(A, b)`, never `np.linalg.inv(A) @ b`.

---

## 3.2 Triangular systems

For lower-triangular $L$, **forward substitution**:
$$
x_i = \frac{1}{\ell_{ii}}\Bigl(b_i - \sum_{j<i}\ell_{ij}x_j\Bigr),\qquad i = 1,\dots,n.
$$
For upper-triangular $U$, **back substitution** runs $i = n,\dots,1$ with $\sum_{j>i}$.

**Cost.** Row $i$ needs about $2i$ flops, so the total is $\sum 2i \approx n^2$ flops.

---

## 3.3 Gaussian elimination = LU factorisation

At step $k$, for every row $i>k$ subtract $m_{ik} = a_{ik}/a_{kk}$ times row $k$. The number $a_{kk}$ is the **pivot**; $m_{ik}$ are the **multipliers**.

If we store the multipliers in a unit lower-triangular matrix $L$ and the final upper-triangular matrix in $U$, then
$$
\boxed{A = LU}.
$$

**Worked example.**
$$
A=\begin{pmatrix}2&1&1\\4&-6&0\\-2&7&2\end{pmatrix}
\;\xrightarrow{\;m_{21}=2,\ m_{31}=-1\;}\;
\begin{pmatrix}2&1&1\\0&-8&-2\\0&8&3\end{pmatrix}
\;\xrightarrow{\;m_{32}=-1\;}\;
\begin{pmatrix}2&1&1\\0&-8&-2\\0&0&1\end{pmatrix}=U,
\quad
L=\begin{pmatrix}1&0&0\\2&1&0\\-1&-1&1\end{pmatrix}.
$$

**Solving with LU.** $A\mathbf x = \mathbf b \iff L(U\mathbf x) = \mathbf b$: solve $L\mathbf y = \mathbf b$ (forward), then $U\mathbf x = \mathbf y$ (back).

**Cost.** Factorisation: $\tfrac23 n^3$ flops. Each subsequent solve: $2n^2$. Hence *factor once, solve many times* — crucial when the same $A$ is used with many right-hand sides (e.g. cross-validation folds, time steps, Newton iterations with a frozen Jacobian).

| $n$ | $\tfrac23 n^3$ flops | time at $10^{10}$ flop/s |
|---|---|---|
| $10^3$ | $6.7\times10^{8}$ | 0.07 s |
| $10^4$ | $6.7\times10^{11}$ | 67 s |
| $10^5$ | $6.7\times10^{14}$ | 18 hours |

This cubic wall is why Lecture 4 (iterative methods) exists.

---

## 3.4 Pivoting

Elimination fails if a pivot is zero and is **unstable** if a pivot is tiny. Consider
$$
A = \begin{pmatrix}10^{-20} & 1\\ 1 & 1\end{pmatrix},\qquad \mathbf b = \begin{pmatrix}1\\2\end{pmatrix},\qquad \mathbf x \approx \begin{pmatrix}1\\1\end{pmatrix}.
$$
Without pivoting, $m_{21} = 10^{20}$ and $u_{22} = 1 - 10^{20}$ rounds to $-10^{20}$; the computed solution is $(0, 1)$ — **completely wrong** in $x_1$. The matrix is perfectly well-conditioned ($\kappa\approx 2.6$): the algorithm, not the problem, is at fault.

**Partial pivoting.** At step $k$, swap row $k$ with the row $p \ge k$ that has the largest $|a_{pk}|$. Then all $|m_{ik}| \le 1$ and we obtain
$$
\boxed{PA = LU}
$$
with a permutation matrix $P$. This is what `scipy.linalg.lu_factor` and LAPACK's `getrf` compute.

**Stability.** Gaussian elimination with partial pivoting (GEPP) is backward stable *in practice*: the computed $\hat{\mathbf x}$ solves $(A+E)\hat{\mathbf x}=\mathbf b$ with $\|E\| \lesssim \rho_n\, u\,\|A\|$, where the **growth factor** $\rho_n$ is almost always small (the theoretical worst case $2^{n-1}$ essentially never occurs in real problems).

---

## 3.5 Cholesky factorisation

$A$ is **symmetric positive definite (SPD)** if $A = A^\top$ and $\mathbf x^\top A\mathbf x > 0$ for all $\mathbf x \ne \mathbf 0$. Covariance matrices, kernel (Gram) matrices and $X^\top X$ with full column rank are SPD.

**Theorem.** $A$ is SPD iff $A = LL^\top$ with $L$ lower triangular and positive diagonal. The factor is unique.

$$
\ell_{jj} = \sqrt{a_{jj} - \sum_{k<j}\ell_{jk}^2},\qquad
\ell_{ij} = \frac{1}{\ell_{jj}}\Bigl(a_{ij} - \sum_{k<j}\ell_{ik}\ell_{jk}\Bigr),\ i>j.
$$

* Cost $\tfrac13 n^3$ — half of LU. No pivoting needed; stable.
* A negative number under the square root proves $A$ is **not** positive definite — the cheapest PD test.

**Data-science uses of Cholesky**
* Sampling from $\mathcal N(\boldsymbol\mu, \Sigma)$: draw $\mathbf z \sim \mathcal N(0, I)$ and return $\boldsymbol\mu + L\mathbf z$.
* Log-determinant: $\log\det\Sigma = 2\sum_i \log \ell_{ii}$ (never compute `det` directly — it over/underflows).
* Mahalanobis distance: $\mathbf x^\top\Sigma^{-1}\mathbf x = \|L^{-1}\mathbf x\|_2^2$ via one triangular solve.
* Gaussian-process regression — `sklearn.gaussian_process` does exactly this.

---

## 3.6 Norms and conditioning

### Vector norms
$$
\|\mathbf x\|_1 = \sum|x_i|,\qquad \|\mathbf x\|_2 = \Bigl(\sum x_i^2\Bigr)^{1/2},\qquad \|\mathbf x\|_\infty = \max|x_i|.
$$

### Induced matrix norms
$\|A\| = \max_{\mathbf x\ne 0}\|A\mathbf x\|/\|\mathbf x\|$. Useful formulas:
$$
\|A\|_1 = \max_j\sum_i|a_{ij}|\ \ (\text{max column sum}),\qquad
\|A\|_\infty = \max_i\sum_j|a_{ij}|\ \ (\text{max row sum}),\qquad
\|A\|_2 = \sigma_{\max}(A).
$$

### Condition number
$$
\kappa(A) = \|A\|\,\|A^{-1}\| \;\ge 1, \qquad \kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}.
$$

**Theorem (perturbation of the right-hand side).** If $A\mathbf x = \mathbf b$ and $A(\mathbf x + \delta\mathbf x) = \mathbf b + \delta\mathbf b$, then
$$
\frac{\|\delta\mathbf x\|}{\|\mathbf x\|} \le \kappa(A)\,\frac{\|\delta\mathbf b\|}{\|\mathbf b\|}.
$$

*Proof.* $\delta\mathbf x = A^{-1}\delta\mathbf b$ gives $\|\delta\mathbf x\| \le \|A^{-1}\|\|\delta\mathbf b\|$; and $\|\mathbf b\| \le \|A\|\|\mathbf x\|$. Multiply.

A similar bound holds for perturbations of $A$. Combined with backward stability:

$$
\boxed{\frac{\|\hat{\mathbf x} - \mathbf x\|}{\|\mathbf x\|} \lesssim \kappa(A)\, u.}
$$

**Example.** $A = \begin{pmatrix}1&1\\1&1.0001\end{pmatrix}$ has $\kappa_\infty \approx 4\times10^4$. Changing $b_2$ from $2.0001$ to $2.0002$ (relative change $5\times 10^{-5}$) changes the solution from $(1,1)$ to $(0,2)$ — a 100% change.

**The Hilbert matrix** $H_{ij} = 1/(i+j-1)$ is the classic ill-conditioned example:

| $n$ | $\kappa_2(H_n)$ | max error solving $H\mathbf x = H\mathbf 1$ |
|---|---|---|
| 4 | $1.6\times10^{4}$ | $5\times10^{-14}$ |
| 8 | $1.5\times10^{10}$ | $7\times10^{-8}$ |
| 12 | $1.6\times10^{16}$ | $0.28$ |

Note how "digits lost ≈ $\log_{10}\kappa$" predicts the error. Polynomial regression with monomial features $1, x, x^2,\dots$ on $[0,1]$ leads to Hilbert-like matrices — one reason to standardise features and to prefer QR (Lecture 5).

**Residual vs. error.** A small residual $\|\mathbf b - A\hat{\mathbf x}\|$ does **not** imply a small error when $\kappa$ is large: $\frac{\|\hat{\mathbf x}-\mathbf x\|}{\|\mathbf x\|} \le \kappa(A)\frac{\|\mathbf r\|}{\|\mathbf b\|}$.

---

## 3.7 Exploiting structure

| Structure | Algorithm | Cost | Library |
|---|---|---|---|
| General dense | LU with partial pivoting | $\tfrac23n^3$ | `scipy.linalg.lu_factor` |
| SPD | Cholesky | $\tfrac13n^3$ | `scipy.linalg.cho_factor` |
| Tridiagonal | Thomas algorithm | $8n$ | `scipy.linalg.solve_banded` |
| Banded, bandwidth $p$ | Banded LU | $O(np^2)$ | `solve_banded` |
| Sparse | Sparse LU with fill-reducing ordering | problem dependent | `scipy.sparse.linalg.spsolve` |
| Triangular | Substitution | $n^2$ | `scipy.linalg.solve_triangular` |

### Thomas algorithm (tridiagonal)
For $a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i$: forward sweep
$c_i' = \dfrac{c_i}{b_i - a_i c_{i-1}'},\ d_i' = \dfrac{d_i - a_i d_{i-1}'}{b_i - a_i c_{i-1}'}$, then back substitution $x_i = d_i' - c_i' x_{i+1}$. Stable when $A$ is diagonally dominant.

---

## 3.8 Iterative refinement

Given an approximate solution $\hat{\mathbf x}$ from an LU factorisation:
1. $\mathbf r = \mathbf b - A\hat{\mathbf x}$ (ideally in higher precision);
2. solve $A\mathbf d = \mathbf r$ reusing the factors ($2n^2$ flops);
3. $\hat{\mathbf x} \leftarrow \hat{\mathbf x} + \mathbf d$.

Modern GPUs use this in **mixed precision**: factor in `float16`/`float32`, refine to `float64` accuracy.

---

## 3.9 Summary

* LU with partial pivoting: $PA = LU$, $\tfrac23n^3$ flops, backward stable in practice.
* Cholesky for SPD: $LL^\top$, $\tfrac13n^3$, also a positive-definiteness test.
* Accuracy is governed by $\kappa(A)$: expect to lose about $\log_{10}\kappa(A)$ digits.
* Never invert; factor once and reuse; exploit structure.

## Further reading

* Trefethen & Bau, *Numerical Linear Algebra*, Lectures 20–23.
* Golub & Van Loan, *Matrix Computations*, Chapters 3–4.
