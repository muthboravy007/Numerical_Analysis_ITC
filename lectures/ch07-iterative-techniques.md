# Chapter 7 — Iterative Techniques in Matrix Algebra

> **Reference:** Burden & Faires, Chapter 7 (§7.1–7.6) plus §7.7 for data science.
> **Code:** [`numlib/linalg_iterative.py`](../numlib/linalg_iterative.py) · **All numbers reproduced by** [`lectures/code/ch07.py`](code/ch07.py)
> **Worked problems:** [`examples/ch07`](../examples/ch07_examples.md) · **Homework:** [`exercises/ch07`](../exercises/ch07_exercises.md)

## Learning outcomes

1. Compute vector norms ($\ell_1,\ell_2,\ell_\infty$) and induced matrix norms, and use norm equivalences.
2. Relate eigenvalues, spectral radius and matrix norms; characterise convergent matrices.
3. Derive the Jacobi, Gauss–Seidel and SOR methods from matrix splittings and prove convergence conditions.
4. Use the condition number to bound errors from residuals; apply iterative refinement.
5. Derive and apply the (preconditioned) conjugate gradient method and state its convergence rate.
6. Use sparse storage and iterative solvers in data-science problems (PageRank, graph learning, large regressions).

---

## 7.1 Norms of Vectors and Matrices

**Definition 7.1.** A **vector norm** on $\mathbb R^n$ is a function $\|\cdot\|$ with (i) $\|\mathbf x\|\ge0$; (ii) $\|\mathbf x\| = 0\iff\mathbf x = \mathbf 0$; (iii) $\|\alpha\mathbf x\| = |\alpha|\|\mathbf x\|$; (iv) $\|\mathbf x+\mathbf y\|\le\|\mathbf x\|+\|\mathbf y\|$.
$$
\|\mathbf x\|_1 = \sum|x_i|,\qquad\|\mathbf x\|_2 = \Bigl(\sum x_i^2\Bigr)^{1/2},\qquad\|\mathbf x\|_\infty = \max|x_i|.
$$

**Theorem 7.3 (Cauchy–Buniakowsky–Schwarz).** $|\mathbf x^T\mathbf y|\le\|\mathbf x\|_2\|\mathbf y\|_2$.

**Equivalence.** $\|\mathbf x\|_\infty\le\|\mathbf x\|_2\le\sqrt n\|\mathbf x\|_\infty$ and $\|\mathbf x\|_2\le\|\mathbf x\|_1\le\sqrt n\|\mathbf x\|_2$. All norms on $\mathbb R^n$ are equivalent, so a sequence converges in one norm iff it converges in all (Theorem 7.7: iff it converges componentwise).

**Definition 7.8.** A **matrix norm** satisfies the vector-norm axioms plus $\|AB\|\le\|A\|\|B\|$. The **natural (induced)** norm is $\|A\| = \max_{\|\mathbf x\| = 1}\|A\mathbf x\|$; it satisfies $\|A\mathbf z\|\le\|A\|\|\mathbf z\|$.
$$
\|A\|_\infty = \max_i\sum_j|a_{ij}|\ \ (\text{Theorem 7.11}),\qquad \|A\|_1 = \max_j\sum_i|a_{ij}|,\qquad \|A\|_2 = \sqrt{\rho(A^TA)}\ \ (\text{Theorem 7.15}).
$$
The Frobenius norm $\|A\|_F = (\sum a_{ij}^2)^{1/2}$ is a matrix norm but not induced; $\|A\|_2\le\|A\|_F$.

### Examples for §7.1

**Example 7.1.1.** $\mathbf x = (-1,1,-2)^T$: $\|\mathbf x\|_1 = 4$, $\|\mathbf x\|_2 = \sqrt6 = 2.4495$, $\|\mathbf x\|_\infty = 2$.

**Example 7.1.2 (convergence of vectors).** $\mathbf x^{(k)} = \bigl(1,\ 2+\frac1k,\ \frac{3}{k^2},\ e^{-k}\sin k\bigr)^T\to(1,2,0,0)^T$: the $\ell_\infty$ distances are $3,\ 0.1,\ 0.01,\ 0.001$ for $k = 1, 10, 100, 1000$ (each component converges).

**Example 7.1.3 (matrix norms).** $A = \begin{pmatrix}1&2&-1\\0&3&-1\\5&-1&1\end{pmatrix}$: row sums $4,4,7$ ⇒ $\|A\|_\infty = 7$; column sums $6,6,3$ ⇒ $\|A\|_1 = 6$; $\|A\|_2 = 5.2824$ (from the eigenvalues of $A^TA$); $\|A\|_F = \sqrt{43} = 6.5574$.

**Example 7.1.4 (equivalence and Cauchy–Schwarz numerically).** For random $\mathbf x\in\mathbb R^4$: $0.6404\le0.6741\le1.2808$ ($\|\cdot\|_\infty\le\|\cdot\|_2\le2\|\cdot\|_\infty$), and $|\mathbf x^T\mathbf y| = 0.346\le\|\mathbf x\|_2\|\mathbf y\|_2 = 0.867$.

**Example 7.1.5 (measuring an approximate solution).** True $\mathbf x = (1,1,1)$, computed $\tilde{\mathbf x} = (1.2001, 0.99991, 0.92538)$: $\|\tilde{\mathbf x}-\mathbf x\|_\infty = 0.2001$, $\|\tilde{\mathbf x}-\mathbf x\|_2 = 0.2136$. Report errors in a norm and say which one.

**Example 7.1.6.** $B = \begin{pmatrix}3&-1\\2&4\end{pmatrix}$: $\|B\|_2 = 4.5150<\|B\|_F = \sqrt{30} = 5.4772$.

---

## 7.2 Eigenvalues and Eigenvectors

$\lambda$ is an eigenvalue of $A$ if $\det(A-\lambda I) = 0$; a nonzero $\mathbf x$ with $A\mathbf x = \lambda\mathbf x$ is an eigenvector. The **spectral radius** is $\rho(A) = \max|\lambda|$.

**Theorem 7.15.** For any natural norm, $\rho(A)\le\|A\|$; and $\|A\|_2 = \sqrt{\rho(A^TA)}$ (so $\|A\|_2 = \rho(A)$ if $A$ is symmetric).

**Definition 7.16.** $A$ is **convergent** if $\lim_{k\to\infty}(A^k)_{ij} = 0$ for all $i,j$.

**Theorem 7.17.** The following are equivalent: (i) $A$ is convergent; (ii) $\lim\|A^k\| = 0$ for some natural norm; (iii) $\rho(A)<1$; (iv) $\lim A^k\mathbf x = \mathbf 0$ for every $\mathbf x$.

**Neumann series.** If $\rho(A)<1$ then $I-A$ is invertible and $(I-A)^{-1} = I + A + A^2 + \cdots$.

### Examples for §7.2

**Example 7.2.1.** $A = \begin{pmatrix}4&1\\2&3\end{pmatrix}$: $\det(A-\lambda I) = \lambda^2-7\lambda+10 = (\lambda-5)(\lambda-2)$. Eigenvectors: $\lambda = 5$: $(1,1)^T$; $\lambda = 2$: $(1,-2)^T$. $\rho(A) = 5$.

**Example 7.2.2 (the 2-norm from $A^TA$).** $B = \begin{pmatrix}1&2\\0&2\end{pmatrix}$: $B^TB = \begin{pmatrix}1&2\\2&8\end{pmatrix}$ with eigenvalues $0.4689, 8.5311$, so $\|B\|_2 = \sqrt{8.5311} = 2.9208$, while $\rho(B) = 2$ — for non-symmetric matrices $\rho<\|\cdot\|_2$ in general.

**Example 7.2.3 (convergent despite a large norm).** $C = \begin{pmatrix}0.5&1\\0&0.5\end{pmatrix}$ has $\|C\|_\infty = 1.5>1$ but $\rho(C) = 0.5$. $C^k = \begin{pmatrix}0.5^k&k0.5^{k-1}\\0&0.5^k\end{pmatrix}$: the (1,2) entry is $1, 1, 0.3125, 0.0195, 3.8\times10^{-5}$ for $k = 1,2,5,10,20$ — it first *grows* (transient) and then decays. The spectral radius, not the norm, decides.

**Example 7.2.4 ($\rho\le\|\cdot\|$).** $M = \begin{pmatrix}1&2&0\\0&-1&1\\1&0&2\end{pmatrix}$: $\rho(M) = 2.4142$ while $\|M\|_1 = 3$, $\|M\|_2 = 2.4495$, $\|M\|_\infty = 3$ — every natural norm is an upper bound.

**Example 7.2.5 (Leontief input–output model).** An economy with 3 sectors and consumption matrix $A = \begin{pmatrix}0.2&0.3&0.1\\0.4&0.1&0.3\\0.1&0.2&0.2\end{pmatrix}$ must meet final demand $\mathbf d = (100,200,150)$: production $\mathbf x$ solves $(I-A)\mathbf x = \mathbf d$. Since $\rho(A) = 0.634<1$ the Neumann series converges: partial sums $\mathbf d$, $\mathbf d + A\mathbf d = (195,305,230)$, , …, after 20 terms $(357.10, 499.95, 357.11)$, limit $(357.14, 500, 357.14)$. Each term has an interpretation: direct demand, first-round inputs, second-round inputs, …

---

## 7.3 The Jacobi and Gauss–Seidel Iterative Techniques

Write $A = D - L - U$ ($D$ diagonal, $-L$ strictly lower, $-U$ strictly upper). A general stationary iteration is $\mathbf x^{(k)} = T\mathbf x^{(k-1)} + \mathbf c$.

**Jacobi:** $\mathbf x^{(k)} = D^{-1}(L+U)\mathbf x^{(k-1)} + D^{-1}\mathbf b$, i.e.
$$
x_i^{(k)} = \frac{1}{a_{ii}}\Bigl[b_i - \sum_{j\ne i}a_{ij}x_j^{(k-1)}\Bigr],\qquad T_j = D^{-1}(L+U).
$$
**Gauss–Seidel:** use new components as soon as they are available:
$$
x_i^{(k)} = \frac{1}{a_{ii}}\Bigl[b_i - \sum_{j<i}a_{ij}x_j^{(k)} - \sum_{j>i}a_{ij}x_j^{(k-1)}\Bigr],\qquad T_g = (D-L)^{-1}U.
$$

**Theorem 7.19.** For any $\mathbf x^{(0)}$, $\mathbf x^{(k)} = T\mathbf x^{(k-1)}+\mathbf c$ converges to the unique solution of $\mathbf x = T\mathbf x+\mathbf c$ **iff** $\rho(T)<1$.
**Corollary 7.20.** If $\|T\|<1$ for some natural norm, the sequence converges and
$$
\|\mathbf x-\mathbf x^{(k)}\|\le\|T\|^k\|\mathbf x^{(0)}-\mathbf x\|,\qquad \|\mathbf x-\mathbf x^{(k)}\|\le\frac{\|T\|^k}{1-\|T\|}\|\mathbf x^{(1)}-\mathbf x^{(0)}\|.
$$
**Theorem 7.21.** If $A$ is strictly diagonally dominant, both Jacobi and Gauss–Seidel converge for every $\mathbf x^{(0)}$.
**Theorem 7.22 (Stein–Rosenberg).** If $a_{ij}\le0$ for $i\ne j$ and $a_{ii}>0$, then exactly one of: $0\le\rho(T_g)<\rho(T_j)<1$; $1<\rho(T_j)<\rho(T_g)$; $\rho(T_j) = \rho(T_g) = 0$; $\rho(T_j) = \rho(T_g) = 1$. (For such matrices, if one method converges, both do, and Gauss–Seidel is faster.)

### Examples for §7.3

**Example 7.3.1 (Jacobi).** The SDD system
$$
\begin{pmatrix}5&-1&1&0\\-1&6&-2&1\\1&-2&7&-1\\0&1&-1&4\end{pmatrix}\mathbf x = \begin{pmatrix}2\\14\\-11\\7\end{pmatrix},\qquad\mathbf x = (1,2,-1,1)^T.
$$
From $\mathbf x^{(0)} = \mathbf 0$: $x_1^{(1)} = \frac{2}{5} = 0.4$, $x_2^{(1)} = \frac{14}{6} = 2.3333$, $x_3^{(1)} = \frac{-11}{7} = -1.5714$, $x_4^{(1)} = \frac74 = 1.75$;
$\mathbf x^{(2)} = (1.1810, 1.5845, -0.7119, 0.7738)$; $\mathbf x^{(3)} = (0.8593, 2.1639, -1.1769, 1.1759)$.
32 iterations reach $\|\mathbf r\|/\|\mathbf b\|<10^{-8}$.

**Example 7.3.2 (Gauss–Seidel, same system).** $x_1^{(1)} = 0.4$, then $x_2^{(1)} = \frac{14 + 0.4}{6} = 2.4$ (uses the new $x_1$), $x_3^{(1)} = -0.9429$, $x_4^{(1)} = 0.9143$;
$\mathbf x^{(2)} = (1.0686, 2.0448, -1.0093, 0.9865)$; $\mathbf x^{(3)} = (1.0108, 2.0010, -1.0032, 0.9990)$. Only 10 iterations to $10^{-8}$.

**Example 7.3.3 (spectral radii explain the speed).** For this $A$: $\rho(T_j) = 0.558$, $\rho(T_g) = 0.146$. Digits gained per iteration: $-\log_{10}0.558 = 0.25$ vs $-\log_{10}0.146 = 0.83$ — a ratio of about 3.3, matching $32/10$.

**Example 7.3.4 (neither method dominates in general).** For $A = \begin{pmatrix}1&2&-2\\1&1&1\\2&2&1\end{pmatrix}$: $\rho(T_j) = 0$ ($T_j$ is nilpotent, $T_j^3 = 0$) so **Jacobi converges exactly in 3 iterations**, whereas $\rho(T_g) = 2$ and Gauss–Seidel diverges (after 20 iterations the entries are $\sim3\times10^7$). For $A = \begin{pmatrix}2&-1&1\\2&2&2\\-1&-1&2\end{pmatrix}$ it is the other way round: $\rho(T_j) = 1.118$, $\rho(T_g) = 0.5$.

**Example 7.3.5 (a-priori iteration count).** For Example 7.3.1, $\|T_j\|_\infty = \frac23$ and $\|\mathbf x^{(1)}-\mathbf x^{(0)}\|_\infty = 2.3333$. Corollary 7.20 guarantees $\|\mathbf x-\mathbf x^{(k)}\|_\infty<10^{-6}$ once $\frac{(2/3)^k}{1/3}2.3333<10^{-6}$, i.e. $k\ge39$. (Pessimistic: the asymptotic rate is $\rho = 0.558$, not $0.667$.)

**Example 7.3.6 (slow convergence for PDE matrices).** For $\operatorname{tridiag}(-1,2,-1)$ of size $n$ (not strictly SDD), $\rho(T_j) = \cos\frac{\pi}{n+1}$ and $\rho(T_g) = \rho(T_j)^2$. Iterations to reach $10^{-6}$:

| $n$ | $\rho(T_j)$ | Jacobi | Gauss–Seidel |
|---|---|---|---|
| 10 | 0.9595 | 333 | 168 |
| 20 | 0.9888 | 1223 | 613 |
| 40 | 0.9971 | 4670 | 2337 |
| 80 | 0.9992 | 18233 | 9118 |

Doubling $n$ roughly quadruples the work: $1-\rho\approx\frac{\pi^2}{2(n+1)^2}$. This motivates SOR and CG.

---

## 7.4 Relaxation Techniques for Solving Linear Systems

The **residual** of $\tilde{\mathbf x}$ is $\mathbf r = \mathbf b - A\tilde{\mathbf x}$. Gauss–Seidel can be written $x_i^{(k)} = x_i^{(k-1)} + r_{ii}^{(k)}/a_{ii}$. **Successive over-relaxation** takes a larger step:
$$
x_i^{(k)} = (1-\omega)x_i^{(k-1)} + \frac{\omega}{a_{ii}}\Bigl[b_i - \sum_{j<i}a_{ij}x_j^{(k)} - \sum_{j>i}a_{ij}x_j^{(k-1)}\Bigr],
$$
$$
T_\omega = (D-\omega L)^{-1}[(1-\omega)D + \omega U].
$$
$\omega<1$: under-relaxation; $\omega = 1$: Gauss–Seidel; $\omega>1$: over-relaxation.

**Theorem 7.24 (Kahan).** If $a_{ii}\ne0$, then $\rho(T_\omega)\ge|\omega-1|$; hence SOR can converge only if $0<\omega<2$.
**Theorem 7.25 (Ostrowski–Reich).** If $A$ is positive definite and $0<\omega<2$, SOR converges for any $\mathbf x^{(0)}$.
**Theorem 7.26.** If $A$ is positive definite and tridiagonal, then $\rho(T_g) = \rho(T_j)^2<1$ and the optimal parameter is
$$
\omega^\ast = \frac{2}{1+\sqrt{1-\rho(T_j)^2}},\qquad\rho(T_{\omega^\ast}) = \omega^\ast - 1 .
$$

### Examples for §7.4

**Example 7.4.1 (SOR by hand).** $\begin{pmatrix}4&-1&0\\-1&4&-1\\0&-1&4\end{pmatrix}\mathbf x = \begin{pmatrix}2\\4\\10\end{pmatrix}$, $\mathbf x = (1,2,3)^T$, $\mathbf x^{(0)} = \mathbf 0$.
* $\omega = 1$ (GS): $(0.5,\ 1.125,\ 2.78125)$, $(0.78125,\ 1.89063,\ 2.97266)$, $(0.97266,\ 1.98633,\ 2.99658)$ — 12 iterations to $10^{-10}$.
* $\omega = 1.1$: $x_1^{(1)} = 1.1\cdot\frac{2}{4} = 0.55$, $x_2^{(1)} = 1.1\cdot\frac{4+0.55}{4} = 1.25125$, $x_3^{(1)} = 1.1\cdot\frac{10+1.25125}{4} = 3.09409$; then $(0.83909, 2.05650, 3.00613)$, $(1.03163, 2.00473, 3.00069)$ — 11 iterations.

**Example 7.4.2 (optimal $\omega$).** For this matrix $\rho(T_j) = \frac{\sqrt2}{4} = 0.35355$, $\rho(T_g) = 0.125$, and
$\omega^\ast = \frac{2}{1+\sqrt{1-0.125}} = 1.03337$ with $\rho(T_{\omega^\ast}) = 0.03337$. (Strong diagonal dominance means GS is already good; the gain is modest.)

**Example 7.4.3 (where SOR shines).** $\operatorname{tridiag}(-1,2,-1)$, $n = 50$, tolerance $10^{-8}$:

| $\omega$ | 1.0 | 1.5 | 1.8 | 1.884 ($=\omega^\ast$) | 1.9 | 1.95 | 1.99 |
|---|---|---|---|---|---|---|---|
| $\rho(T_\omega)$ | 0.99621 | 0.98859 | 0.96342 | 0.88599 | 0.90000 | 0.95000 | 0.99000 |
| iterations | 4828 | 1600 | 502 | 189 | 204 | 407 | 1887 |

Optimal SOR is 25× faster than Gauss–Seidel. Note the asymmetry: overestimating $\omega^\ast$ is less harmful than underestimating it.

**Example 7.4.4 (Kahan's bound).** For the matrix of Example 7.4.1, $\rho(T_2) = 1.0$ and $\rho(T_{2.1}) = 1.1$: no convergence for $\omega\ge2$.

**Example 7.4.5 (2-D Poisson).** For the 5-point Laplacian on a $20\times20$ grid ($n = 400$) $\omega^\ast = \frac{2}{1+\sin(\pi/21)} = 1.7406$: Gauss–Seidel needs 609 iterations for $10^{-6}$, SOR($\omega^\ast$) 62.

---

## 7.5 Error Bounds and Iterative Refinement

**Theorem 7.27 (residual bound).** If $\tilde{\mathbf x}$ approximates the solution of $A\mathbf x = \mathbf b$ with residual $\mathbf r = \mathbf b - A\tilde{\mathbf x}$, then for any natural norm
$$
\|\mathbf x-\tilde{\mathbf x}\|\le\|\mathbf r\|\,\|A^{-1}\|,\qquad\frac{\|\mathbf x-\tilde{\mathbf x}\|}{\|\mathbf x\|}\le K(A)\frac{\|\mathbf r\|}{\|\mathbf b\|},\qquad K(A) = \|A\|\,\|A^{-1}\|.
$$
$K(A)\ge1$ is the **condition number**; $A$ is well-conditioned if $K(A)\approx1$ and ill-conditioned if $K(A)\gg1$.

**Theorem 7.29 (perturbations of $A$ and $\mathbf b$).** If $\|\delta A\|<1/\|A^{-1}\|$, the solution $\tilde{\mathbf x}$ of $(A+\delta A)\tilde{\mathbf x} = \mathbf b+\delta\mathbf b$ satisfies
$$
\frac{\|\mathbf x-\tilde{\mathbf x}\|}{\|\mathbf x\|}\le\frac{K(A)}{1-K(A)\frac{\|\delta A\|}{\|A\|}}\Bigl(\frac{\|\delta\mathbf b\|}{\|\mathbf b\|}+\frac{\|\delta A\|}{\|A\|}\Bigr).
$$

**Iterative refinement.** Given $\tilde{\mathbf x}$ from an $LU$ factorisation computed in $t$-digit arithmetic: compute $\mathbf r = \mathbf b - A\tilde{\mathbf x}$ in **higher** precision, solve $A\mathbf y = \mathbf r$ with the existing factors, update $\tilde{\mathbf x}\leftarrow\tilde{\mathbf x}+\mathbf y$, repeat. If $K(A)\approx10^q$ and $q<t$, each step gains about $t-q$ digits. The first correction also gives the estimate $K(A)\approx\frac{\|\mathbf y\|}{\|\tilde{\mathbf x}\|}10^t$. Modern GPUs use exactly this in **mixed precision** (factor in `float16/32`, refine in `float64`).

### Examples for §7.5

**Example 7.5.1 (small residual, large error).** $A = \begin{pmatrix}1&1\\1&1.0001\end{pmatrix}$, $\mathbf b = (2, 2.0001)^T$, $\mathbf x = (1,1)^T$. The poor approximation $\tilde{\mathbf x} = (2,0)^T$ has residual $\mathbf r = (0, 0.0001)^T$ — tiny — but error $\|\mathbf x-\tilde{\mathbf x}\|_\infty = 1$. Explanation: $K_\infty(A) = 40004$ and Theorem 7.27 allows a relative error up to $40004\times\frac{10^{-4}}{2.0001} = 2.0$.

**Example 7.5.2 (computing $K_\infty$).** $A = \begin{pmatrix}1&2&-1\\0&3&-1\\5&-1&1\end{pmatrix}$, $A^{-1} = \frac17\begin{pmatrix}2&-1&1\\-5&6&1\\-15&11&3\end{pmatrix}$. $\|A\|_\infty = 7$, $\|A^{-1}\|_\infty = \frac{29}{7} = 4.143$, so $K_\infty(A) = 29$ — well-conditioned.

**Example 7.5.3 (perturbing the matrix).** Changing $a_{22}$ in Example 7.5.1 from $1.0001$ to $1.0002$ (relative change $5\times10^{-5}$ in $\|A\|_\infty$) moves the solution of $A\mathbf x = (2,2.0001)^T$ from $(1,1)$ to $(1.5, 0.5)$ — a 50% change, consistent with $K\approx4\times10^4$.

**Example 7.5.4 (iterative refinement).** $H_6$ (Hilbert, $K_\infty = 2.9\times10^7$), $\mathbf x = \mathbf 1$; factor in single precision (~7 digits), compute residuals in double:

| step | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| max error | $1.6\times10^{-1}$ | $8.6\times10^{-3}$ | $4.7\times10^{-4}$ | $2.5\times10^{-5}$ | $1.4\times10^{-6}$ |

Each step gains about $1.3$ digits. With $t\approx7$ digits and $K\approx10^{7.5}$ this problem is at the very edge of what single precision can factor, yet refinement (with residuals computed in double) still recovers six correct digits.

**Example 7.5.5 (estimating $K$ from refinement).** From the first correction: $\frac{\|\mathbf y\|_\infty}{\|\tilde{\mathbf x}\|_\infty}10^7 = 1.3\times10^6$ — the right order of magnitude (true $2.9\times10^7$) at no extra cost.

**Example 7.5.6 (Hilbert matrices).**

| $n$ | 3 | 5 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| $K_\infty(H_n)$ | $7.5\times10^2$ | $9.4\times10^5$ | $3.4\times10^{10}$ | $3.5\times10^{13}$ | $4.0\times10^{16}$ |
| error in double precision | $10^{-14}$ | $2\times10^{-12}$ | $7\times10^{-8}$ | $4\times10^{-5}$ | $0.28$ |

Digits lost ≈ $\log_{10}K$. Polynomial regression with monomials on $[0,1]$ produces Hilbert-like normal equations (Chapter 8).

---

## 7.6 The Conjugate Gradient Method

For **symmetric positive definite** $A$, solving $A\mathbf x = \mathbf b$ is equivalent to minimising
$$
g(\mathbf x) = \mathbf x^TA\mathbf x - 2\mathbf x^T\mathbf b\qquad(\text{B&F}; \text{equivalently }\tfrac12\mathbf x^TA\mathbf x - \mathbf b^T\mathbf x).
$$
(Theorem 7.31: $\mathbf x^\ast$ solves $A\mathbf x = \mathbf b$ iff it minimises $g$.) Along a direction $\mathbf v$ the minimising step is $t = \frac{\mathbf v^T\mathbf r}{\mathbf v^TA\mathbf v}$.

* **Steepest descent** uses $\mathbf v = \mathbf r$: converges with factor $\frac{K-1}{K+1}$ per step — slow.
* **Conjugate gradient** uses **$A$-orthogonal** directions ($\mathbf v_i^TA\mathbf v_j = 0$, $i\ne j$):

```text
ALGORITHM (CG)   r0 = b - A x0;  v1 = r0
for k = 1, 2, ...:
    t_k   = (r_{k-1}·r_{k-1}) / (v_k·A v_k)
    x_k   = x_{k-1} + t_k v_k
    r_k   = r_{k-1} - t_k A v_k
    stop if ||r_k|| small
    s_k   = (r_k·r_k) / (r_{k-1}·r_{k-1})
    v_{k+1} = r_k + s_k v_k
```

**Theorem 7.33.** The residuals are mutually orthogonal and the directions $A$-orthogonal; in exact arithmetic CG terminates in at most $n$ steps. More useful in practice:
$$
\|\mathbf x-\mathbf x_k\|_A\le2\Bigl(\frac{\sqrt K-1}{\sqrt K+1}\Bigr)^k\|\mathbf x-\mathbf x_0\|_A,\qquad K = K_2(A),
$$
and if $A$ has only $m$ distinct eigenvalues, CG converges in $m$ steps.

**Preconditioning.** Apply CG to $C^{-1}AC^{-T}$ where $M = CC^T\approx A$ is cheap to invert (Jacobi: $M = D$; incomplete Cholesky; multigrid). This reduces $K$ and the iteration count. Each iteration costs one mat-vec with $A$ plus one solve with $M$.

### Examples for §7.6

**Example 7.6.1 (CG by hand).** $A = \begin{pmatrix}4&1\\1&3\end{pmatrix}$, $\mathbf b = (1,2)^T$, $\mathbf x_0 = \mathbf 0$: $\mathbf r_0 = \mathbf v_1 = (1,2)$.
$t_1 = \frac{5}{\mathbf v_1^TA\mathbf v_1} = \frac{5}{20} = 0.25$, $\mathbf x_1 = (0.25, 0.5)$, $\mathbf r_1 = (1,2) - 0.25(6,7) = (-0.5, 0.25)$, $s_1 = \frac{0.3125}{5} = 0.0625$, $\mathbf v_2 = (-0.5,0.25)+0.0625(1,2) = (-0.4375, 0.375)$.
$t_2 = \frac{0.3125}{\mathbf v_2^TA\mathbf v_2} = 0.36364$, $\mathbf x_2 = (0.090909, 0.636364) = (\frac1{11},\frac7{11})$ — exact after $n = 2$ steps.

**Example 7.6.2 (three steps for 3×3).** $A = \begin{pmatrix}4&-1&1\\-1&4&-2\\1&-2&4\end{pmatrix}$, $\mathbf b = (5,1,9)^T$: relative residuals $1,\ 0.530,\ 0.0558,\ 8\times10^{-17}$ — exact solution $(1,2,3)$ at step 3.

**Example 7.6.3 (CG vs steepest descent).** $\operatorname{tridiag}(-1,2,-1)$, tolerance $10^{-8}$:

| $n$ | $K_2$ | CG | steepest descent |
|---|---|---|---|
| 50 | $1.1\times10^3$ | 25 | 9 762 |
| 100 | $4.1\times10^3$ | 50 | 38 446 |
| 200 | $1.6\times10^4$ | 100 | 152 562 |
| 400 | $6.5\times10^4$ | 200 | 607 696 |

CG iterations grow like $\sqrt K\propto n$; steepest descent like $K\propto n^2$.

**Example 7.6.4 (preconditioning).** An SPD matrix with well-behaved spectrum ($K = 10$) is ruined by bad row/column scaling $DAD$ with $D_{ii}\in[10^{-3},10^3]$: $K = 1.2\times10^{12}$. Plain CG does **not** converge in 20 000 iterations; Jacobi-preconditioned CG converges in 44, because $D^{-1/2}(DAD)D^{-1/2}$ has $K = 10.1$.

**Example 7.6.5 (clustered spectra).** $A = Q\Lambda Q^T$ ($n = 200$) with only $m$ distinct eigenvalues: CG converges in exactly $3, 5, 10$ iterations for $m = 3, 5, 10$ — independent of $n$.

---

## 7.7 Sparse Iterative Methods in Data Science *(data-science extension)*

**Sparse storage.** CSR (compressed sparse row) stores `data` (non-zeros), `indices` (their columns) and `indptr` (row starts). A mat-vec costs $O(\text{nnz})$.

| Application | System | Method |
|---|---|---|
| PageRank | $(I-dP)\mathbf r = \frac{1-d}{n}\mathbf 1$ | Jacobi = power iteration |
| semi-supervised learning (label propagation) | $(L+\Lambda)\mathbf f = \Lambda\mathbf y$, $L$ graph Laplacian | CG |
| large ridge regression | $(X^TX+\lambda I)\boldsymbol\beta = X^T\mathbf y$ | CG with mat-vecs $X^T(X\mathbf v)$ (never form $X^TX$) |
| least squares, sparse $X$ | $\min\|X\boldsymbol\beta-\mathbf y\|$ | LSQR/LSMR |
| Gaussian processes at scale | $(K+\sigma^2I)\boldsymbol\alpha = \mathbf y$ | preconditioned CG (GPyTorch) |

### Examples for §7.7

**Example 7.7.1 (CSR).** $A = \begin{pmatrix}4&0&1&0\\0&3&0&0\\1&0&2&5\\0&0&5&1\end{pmatrix}$: `data = [4 1 3 1 2 5 5 1]`, `indices = [0 2 1 0 2 3 2 3]`, `indptr = [0 2 3 6 8]`. Row $i$ occupies `data[indptr[i]:indptr[i+1]]`.

**Example 7.7.2 (PageRank as a linear system).** Four pages with links $1\to2,3$; $2\to3$; $3\to1$; $4\to3$ and $d = 0.85$. Solving $(I-dP)\mathbf r = \frac{0.15}{4}\mathbf 1$ directly: $\mathbf r = (0.3725, 0.1958, 0.3941, 0.0375)$. The Jacobi iteration $\mathbf r_{k+1} = dP\mathbf r_k + \frac{0.15}{4}\mathbf 1$ (= power method) has errors $0.175,\ 0.023,\ 0.0025,\ 1.0\times10^{-5}$ after $1, 5, 10, 20$ iterations — rate $d = 0.85$ ($\rho(dP)\le d$), independent of the number of pages.

**Example 7.7.3 (label propagation).** A 6-node graph (two triangles joined by an edge) with node 1 labelled $+1$ and node 6 labelled $-1$. Solving $(L+\Lambda)\mathbf f = \Lambda\mathbf y$ ($\Lambda = \operatorname{diag}(1,0,0,0,0,1)$) by CG (3 iterations — few distinct eigenvalues): $\mathbf f = (0.54, 0.38, 0.23, -0.23, -0.38, -0.54)$. Nodes 1–3 are classified $+$, nodes 4–6 $-$.

**Example 7.7.4 (ridge regression on sparse data).** $X$: $20\,000\times500$ with $10^5$ non-zeros. CG on $(X^TX+I)\boldsymbol\beta = X^T\mathbf y$, using only products $X\mathbf v$ and $X^T\mathbf u$, converges in 16 iterations to within $8\times10^{-11}$ of the dense solution — without ever forming the $500\times500$ matrix $X^TX$ (which for $10^6$ features would be impossible).

**Example 7.7.5 (feature scaling is preconditioning).** Three features with scales $1, 100, 0.01$: $K(X^TX) = 1.05\times10^8$; steepest descent (= full-batch gradient descent with exact line search) does not converge in $2\times10^5$ iterations. After standardisation $K = 1.04$ and it converges in 4. (CG, which adapts to the spectrum, needs only $n = 3$ iterations in either case in exact arithmetic.)

---

## Chapter summary

* Norms measure errors; $\rho(A)\le\|A\|$; $A^k\to0\iff\rho(A)<1$.
* Stationary methods converge iff $\rho(T)<1$ (SDD ⇒ Jacobi and GS converge; SPD ⇒ GS and SOR($0<\omega<2$) converge).
* Relative error $\le K(A)\times$ relative residual; iterative refinement recovers accuracy when $K(A)<10^t$.
* CG for SPD: $O(\sqrt K)$ iterations, one mat-vec each; precondition to reduce $K$.

## Further reading

Burden & Faires Ch. 7 · Saad, *Iterative Methods for Sparse Linear Systems* · Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain* (1994) · Zhu, Ghahramani & Lafferty, "Semi-supervised learning using Gaussian fields and harmonic functions", ICML 2003.
