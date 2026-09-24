# Chapter 7 — Exercises: Iterative Techniques in Matrix Algebra

> Lecture: [Chapter 7](../lectures/ch07-iterative-techniques.md) · Solutions: [`solutions/ch07_solutions.md`](../solutions/ch07_solutions.md) · Solution code: [`solutions/code/ch07_solutions.py`](../solutions/code/ch07_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★** Prove that $\|\mathbf x\|_\infty\le\|\mathbf x\|_2\le\sqrt n\,\|\mathbf x\|_\infty$ and $\|\mathbf x\|_2\le\|\mathbf x\|_1\le\sqrt n\,\|\mathbf x\|_2$. Show that each constant is attained by some vector.

**A2 ★★** (a) Prove $\rho(A)\le\|A\|$ for every natural (induced) matrix norm. (b) Prove $\|A\|_2 = \sqrt{\rho(A^TA)}$, and deduce $\|A\|_2 = \rho(A)$ when $A$ is symmetric. (c) Give a $2\times2$ matrix with $\rho(A) = 0$ but $\|A\|_2 = 1$.

**A3 ★★** Let $\|T\|<1$ for some natural norm. Show that $\mathbf x^{(k)} = T\mathbf x^{(k-1)}+\mathbf c$ converges to the unique fixed point $\mathbf x$ for every $\mathbf x^{(0)}$, and prove the two bounds
$$
\|\mathbf x-\mathbf x^{(k)}\|\le\|T\|^k\|\mathbf x-\mathbf x^{(0)}\|,\qquad \|\mathbf x-\mathbf x^{(k)}\|\le\frac{\|T\|^k}{1-\|T\|}\|\mathbf x^{(1)}-\mathbf x^{(0)}\|.
$$

**A4 ★★** Prove that if $A$ is strictly diagonally dominant then $\|T_j\|_\infty<1$, so the Jacobi method converges. *(Challenge ★★★: show that Gauss–Seidel also converges, by proving $\|T_g\|_\infty\le\max_i\frac{\sum_{j>i}|a_{ij}|}{|a_{ii}|-\sum_{j<i}|a_{ij}|}<1$.)*

**A5 ★★** Prove Kahan's theorem: $\rho(T_\omega)\ge|\omega-1|$, so SOR can only converge for $0<\omega<2$. (Hint: $T_\omega = (D-\omega L)^{-1}[(1-\omega)D+\omega U]$; compute $\det T_\omega$.)

**A6 ★★★** For conjugate gradients applied to an SPD matrix $A$:
(a) prove by induction that the residuals are mutually orthogonal and the search directions are mutually $A$-conjugate;
(b) deduce that CG terminates in at most $n$ steps in exact arithmetic;
(c) show that $\mathbf x^{(k)}$ minimises $\|\mathbf x-\mathbf x^\ast\|_A$ over $\mathbf x^{(0)}+\mathcal K_k(A,\mathbf r^{(0)})$.

## B. Hand computation

**B1 ★** For $\mathbf x = (3,-4,0,12)^T$ compute $\|\mathbf x\|_1,\|\mathbf x\|_2,\|\mathbf x\|_\infty$. For $A = \begin{pmatrix}1&-2\\3&4\end{pmatrix}$ compute $\|A\|_1$, $\|A\|_\infty$, $\|A\|_2$ and $\rho(A)$, and check that $\rho(A)\le\|A\|$ in each case.

**B2 ★** Find the eigenvalues and eigenvectors of $B = \begin{pmatrix}2&1&0\\1&2&1\\0&1&2\end{pmatrix}$, its spectral radius, $\|B\|_2$ and $\|B\|_\infty$.

**B3 ★** The system $8x_1+x_2-2x_3 = 3$, $x_1+7x_2+2x_3 = -2$, $-x_1+2x_2+9x_3 = 15$ has solution $(1,-1,2)$. Perform three iterations of Jacobi and of Gauss–Seidel from $\mathbf x^{(0)} = \mathbf 0$ and give the error $\|\mathbf x-\mathbf x^{(3)}\|_\infty$ for each.

**B4 ★★** For B3 write out $T_j$ and $T_g$, compute $\|T_j\|_\infty$, $\|T_g\|_\infty$ and the spectral radii. Use the spectral radii to predict the ratio of iterations the two methods need for a given tolerance, and compare with the actual counts for $10^{-8}$.

**B5 ★★** For $A = \operatorname{tridiag}(-1,4,-1)$ ($3\times3$) and $\mathbf b = (2,4,10)^T$ perform two iterations of Gauss–Seidel and of SOR with $\omega = 1.1$. Compute $\rho(T_j)$, the optimal $\omega$ by Theorem 7.26, and $\rho(T_{\omega^\ast})$.

**B6 ★★** $A = \begin{pmatrix}1&2\\1.0001&2\end{pmatrix}$, $\mathbf b = (3, 3.0001)^T$ (solution $(1,1)$). Compute $A^{-1}$ and $K_\infty(A)$. Solve with $\mathbf b$ replaced by $(3, 3.0003)^T$, compare the relative changes of $\mathbf b$ and $\mathbf x$ with the bound $K_\infty\frac{\|\delta\mathbf b\|}{\|\mathbf b\|}$. What is the residual of $\tilde{\mathbf x} = (3,0)^T$ for the original system?

**B7 ★★** Apply conjugate gradients by hand to $\begin{pmatrix}3&1\\1&2\end{pmatrix}\mathbf x = \begin{pmatrix}5\\5\end{pmatrix}$ from $\mathbf x^{(0)} = \mathbf 0$. Show $\alpha_k,\beta_k$, the residuals and search directions, and verify $\mathbf r^{(1)}\perp\mathbf r^{(0)}$ and $\mathbf p^{(0)T}A\mathbf p^{(1)} = 0$.

**B8 ★★** Apply steepest descent to the system of B7 for five iterations. Compute $\kappa_2(A)$ and the guaranteed contraction factor $\frac{\kappa-1}{\kappa+1}$ of $\|\mathbf e\|_A$; compare with the observed factor.

## C. Programming

**C1 ★★** For the 2-D Poisson matrix on a $20\times20$ interior grid ($n = 400$) and $\mathbf b = \mathbf 1$, count the iterations Jacobi, Gauss–Seidel and SOR ($\omega\in\{1.2,1.4,\dots,1.95\}$) need to reach relative residual $10^{-6}$. Compare the best $\omega$ with $\omega^\ast = \frac{2}{1+\sin(\pi h)}$, $h = \frac1{m+1}$.

**C2 ★★** For the 2-D Poisson matrix with $m = 16, 32, 64, 128$ count the iterations to $10^{-8}$ of steepest descent (small $m$ only), CG and CG preconditioned by an incomplete LU factorisation. Relate the counts to $\kappa$ and $\sqrt\kappa$.

**C3 ★★★** *Mixed-precision iterative refinement* (used on GPUs for fast solvers). Factor $A$ in `float32`, then refine: $\mathbf r = \mathbf b-A\mathbf x$ in `float64`, solve $A\mathbf d = \mathbf r$ with the `float32` factors, $\mathbf x\leftarrow\mathbf x+\mathbf d$. Test on random $200\times200$ matrices with $\kappa = 10^3$ and $10^6$ and compare with a `float64` direct solve. When does refinement fail?

## D. Data-science applications

**D1 ★★** Ridge regression $(X^TX+\lambda I)\boldsymbol\beta = X^T\mathbf y$ with $X\in\mathbb R^{5000\times200}$ of badly scaled columns. Solve it **matrix-free** with CG (using only products $X\mathbf v$ and $X^T\mathbf w$) for $\lambda = 10^{-4},10^{-2},1,100$. Tabulate $\kappa$ and the CG iteration counts. Why can the count exceed $p = 200$?

**D2 ★★** PageRank solves $(I-\alpha P^T)\mathbf x = (1-\alpha)\mathbf v$. For a random 2000-page web graph with hub pages and $\alpha = 0.85$ compare the power iteration (which is the Jacobi iteration here), Gauss–Seidel and a sparse direct solve. Explain the iteration counts using Stein–Rosenberg and the second eigenvalue of $P$.

**D3 ★★★** *Graph-based semi-supervised learning.* On the two-moons data (600 points, noise 0.08) build a Gaussian similarity graph ($\sigma = 0.1$), graph Laplacian $L = D-W$, and label only 3 points per class. The harmonic solution solves $L_{uu}\mathbf f_u = -L_{ul}\mathbf f_l$. Solve it with Jacobi-preconditioned CG, classify by $f>0.5$ and compare the accuracy with logistic regression trained on the 6 labelled points.
