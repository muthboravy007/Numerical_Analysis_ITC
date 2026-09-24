# Lecture 4 — Iterative Methods for Large Sparse Systems

> **Week 6** · Code: [`numlib/linalg_iterative.py`](../numlib/linalg_iterative.py) · Examples: [`examples/ch04`](../examples/ch04_examples.md) · Exercises: [`exercises/ch04`](../exercises/ch04_exercises.md)

## Learning objectives

1. Explain when iterative methods beat direct factorisation (size, sparsity, fill-in).
2. Derive Jacobi, Gauss–Seidel and SOR from a matrix splitting and state their convergence criteria.
3. Relate convergence speed to the **spectral radius** of the iteration matrix.
4. Understand the **conjugate gradient** method as optimisation of a quadratic and know its convergence bound in terms of $\kappa$.
5. Apply **preconditioning** and use `scipy.sparse` data structures.

---

## 4.1 Why iterate?

A graph with $10^6$ nodes (a social network, a web crawl, a recommender user–item matrix) gives a matrix with $10^{12}$ entries — 8 TB dense — but perhaps only $10^7$ non-zeros. Gaussian elimination destroys sparsity ("fill-in") and costs $O(n^3)$. Iterative methods only need the ability to compute **matrix–vector products** $A\mathbf v$, which cost $O(\text{nnz})$.

### Sparse storage (CSR)

Compressed Sparse Row stores three arrays: `data` (non-zero values), `indices` (their column indices) and `indptr` (where each row starts).

```python
import scipy.sparse as sp
A = sp.csr_matrix([[4, 0, 1], [0, 3, 0], [1, 0, 2]])
A.data, A.indices, A.indptr   # [4 1 3 1 2], [0 2 1 0 2], [0 2 3 5]
```

---

## 4.2 Stationary iterative methods

Split $A = M - N$ with $M$ easy to invert. Then $A\mathbf x = \mathbf b \iff M\mathbf x = N\mathbf x + \mathbf b$, suggesting

$$
\mathbf x^{(k+1)} = M^{-1}N\mathbf x^{(k)} + M^{-1}\mathbf b = T\mathbf x^{(k)} + \mathbf c.
$$

Write $A = D - L - U$ with $D$ diagonal, $-L$ strictly lower and $-U$ strictly upper part.

| Method | $M$ | Component form |
|---|---|---|
| **Jacobi** | $D$ | $x_i^{(k+1)} = \dfrac{1}{a_{ii}}\Bigl(b_i - \sum_{j\ne i} a_{ij}x_j^{(k)}\Bigr)$ |
| **Gauss–Seidel** | $D-L$ | $x_i^{(k+1)} = \dfrac{1}{a_{ii}}\Bigl(b_i - \sum_{j<i}a_{ij}x_j^{(k+1)} - \sum_{j>i}a_{ij}x_j^{(k)}\Bigr)$ |
| **SOR($\omega$)** | $\tfrac1\omega D - L$ | $x_i^{(k+1)} = (1-\omega)x_i^{(k)} + \omega\,x_i^{\text{GS}}$ |

Gauss–Seidel uses new values as soon as they are available; Jacobi updates all components simultaneously (and is therefore trivially parallel).

### Convergence theory

The error $\mathbf e^{(k)} = \mathbf x^{(k)} - \mathbf x$ satisfies $\mathbf e^{(k+1)} = T\mathbf e^{(k)}$, so $\mathbf e^{(k)} = T^k\mathbf e^{(0)}$.

**Theorem.** The iteration converges for every $\mathbf x^{(0)}$ iff the **spectral radius** $\rho(T) = \max_i|\lambda_i(T)| < 1$. Asymptotically each iteration reduces the error by a factor $\rho(T)$, so gaining one decimal digit needs about $-1/\log_{10}\rho(T)$ iterations.

**Sufficient conditions (easy to check).**
* If $A$ is **strictly diagonally dominant** ($|a_{ii}| > \sum_{j\ne i}|a_{ij}|$ for all $i$) then Jacobi and Gauss–Seidel both converge.
* If $A$ is SPD then Gauss–Seidel converges, and SOR converges iff $0<\omega<2$ (Ostrowski–Reich).

**Optimal SOR.** For the important class of *consistently ordered* matrices (e.g. the 1-D/2-D Poisson matrices),
$\rho_{\text{GS}} = \rho_{\text{J}}^2$ and the best relaxation parameter is
$$
\omega^\star = \frac{2}{1+\sqrt{1-\rho_J^2}},\qquad \rho(T_{\omega^\star}) = \omega^\star - 1.
$$

**Model problem.** The $n\times n$ matrix $\operatorname{tridiag}(-1,2,-1)$ has $\rho_J = \cos\frac{\pi}{n+1} \approx 1 - \frac{\pi^2}{2(n+1)^2}$. For $n = 100$, $\rho_J \approx 0.9995$: Jacobi needs tens of thousands of iterations. Optimal SOR brings $\rho$ down to about $0.94$ — a huge improvement, but CG does better still.

---

## 4.3 The conjugate gradient method

Assume $A$ is SPD. Solving $A\mathbf x = \mathbf b$ is equivalent to **minimising the quadratic**
$$
\phi(\mathbf x) = \tfrac12\mathbf x^\top A\mathbf x - \mathbf b^\top\mathbf x,\qquad \nabla\phi(\mathbf x) = A\mathbf x - \mathbf b = -\mathbf r.
$$
This links linear algebra to optimisation (Lecture 9).

**Steepest descent** moves along $\mathbf r_k$ with exact line search $\alpha_k = \frac{\mathbf r_k^\top\mathbf r_k}{\mathbf r_k^\top A\mathbf r_k}$. It converges like $\left(\frac{\kappa-1}{\kappa+1}\right)^k$ — slow and zig-zagging when $\kappa$ is large.

**Conjugate gradient** instead chooses search directions that are **$A$-conjugate**: $\mathbf p_i^\top A\mathbf p_j = 0$ for $i\ne j$. Minimising along each once is then enough.

```text
Algorithm CG(A, b, x0)
  r = b - A x0;  p = r
  for k = 0, 1, 2, ...
      alpha = (r·r) / (p·A p)
      x = x + alpha p
      r_new = r - alpha A p
      if ||r_new|| small: stop
      beta = (r_new·r_new) / (r·r)
      p = r_new + beta p;  r = r_new
```

One matrix–vector product and a few dot products per iteration; $O(n)$ extra memory.

**Theorem.** In exact arithmetic CG converges in at most $n$ iterations. More usefully,
$$
\|\mathbf x_k - \mathbf x\|_A \le 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k\|\mathbf x_0-\mathbf x\|_A,
\qquad \|\mathbf v\|_A = \sqrt{\mathbf v^\top A\mathbf v}.
$$
The dependence on $\sqrt\kappa$ rather than $\kappa$ is the key advantage. Also, if $A$ has only $m$ distinct eigenvalues, CG converges in $m$ steps.

**Comparison on $\operatorname{tridiag}(-1,2,-1)$, tolerance $10^{-8}$:**

| $n$ | $\kappa$ | Gauss–Seidel iterations | CG iterations |
|---|---|---|---|
| 50 | $1.1\times10^{3}$ | 4 828 | 25 |
| 100 | $4.1\times10^{3}$ | 18 934 | 50 |
| 200 | $1.6\times10^{4}$ | 74 983 | 100 |

---

## 4.4 Preconditioning

Solve the equivalent system $M^{-1}A\mathbf x = M^{-1}\mathbf b$ where $M \approx A$ but $M\mathbf z = \mathbf r$ is cheap to solve, so that $\kappa(M^{-1}A) \ll \kappa(A)$.

| Preconditioner | $M$ | Comment |
|---|---|---|
| Jacobi (diagonal) | $\operatorname{diag}(A)$ | Free; fixes bad row scaling |
| Incomplete Cholesky / ILU | $\tilde L\tilde L^\top$ keeping sparsity pattern | Workhorse for PDE matrices |
| Multigrid | Hierarchy of coarse grids | Optimal $O(n)$ for elliptic problems |

In data science, **feature standardisation** is diagonal preconditioning of $X^\top X$: it is why gradient-based solvers converge faster on scaled features.

---

## 4.5 Beyond SPD

For non-symmetric systems use **GMRES** (minimises the residual over a Krylov subspace $\operatorname{span}\{\mathbf r_0, A\mathbf r_0, \dots, A^{k-1}\mathbf r_0\}$) or **BiCGSTAB**. For least-squares problems use **LSQR/LSMR** (CG applied implicitly to the normal equations). All are in `scipy.sparse.linalg`.

---

## 4.6 Data-science applications

* **PageRank** (Lecture 6) solves $(I - dP)\mathbf r = \frac{1-d}{n}\mathbf 1$ — Jacobi iteration on this system *is* the power method.
* **Graph Laplacian systems**: semi-supervised label propagation solves $(L + \lambda I)\mathbf f = \lambda\mathbf y$ with a sparse Laplacian $L$ — ideal for CG.
* **Large ridge regression**: CG/LSQR on $(X^\top X + \lambda I)\beta = X^\top\mathbf y$ without ever forming $X^\top X$.
* **Kernel methods at scale**: CG with fast kernel mat-vecs (GPyTorch).

## 4.7 Summary

* Stationary methods converge iff $\rho(T) < 1$; diagonal dominance or SPD are sufficient conditions.
* Gauss–Seidel ≈ twice as fast as Jacobi for model problems; optimal SOR much faster.
* CG for SPD: $O(\sqrt\kappa)$ iterations, each costing one mat-vec. Precondition to reduce $\kappa$.
* Use `scipy.sparse` formats and `scipy.sparse.linalg.{cg, gmres, lsqr}`.

## Further reading

* J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*, 1994 (free online).
* Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
