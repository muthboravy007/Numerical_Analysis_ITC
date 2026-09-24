# Chapter 7 — Solutions

> Exercises: [`exercises/ch07_exercises.md`](../exercises/ch07_exercises.md) · Numbers from [`solutions/code/ch07_solutions.py`](code/ch07_solutions.py).

## A. Theory and proofs

**A1.** Let $|x_k| = \|\mathbf x\|_\infty$. Then $\|\mathbf x\|_\infty^2 = x_k^2\le\sum x_i^2\le n\|\mathbf x\|_\infty^2$. Next, $\|\mathbf x\|_1^2 = \sum_i x_i^2 + \sum_{i\ne j}|x_i||x_j|\ge\|\mathbf x\|_2^2$, and by Cauchy–Schwarz $\|\mathbf x\|_1 = \sum|x_i|\cdot1\le\|\mathbf x\|_2\sqrt n$. Equality: $\mathbf e_1$ attains the lower bounds; $\mathbf 1 = (1,\dots,1)$ attains the upper bounds.

**A2.** (a) If $A\mathbf v = \lambda\mathbf v$ with $\|\mathbf v\| = 1$, then $|\lambda| = \|A\mathbf v\|\le\|A\|$. (For complex $\lambda$, apply the argument to the norm extended to $\mathbb C^n$, or use $\rho(A)^k = \rho(A^k)\le\|A^k\|\le\|A\|^k$ together with Gelfand's formula.) (b) $\|A\mathbf x\|_2^2 = \mathbf x^TA^TA\mathbf x$. $A^TA$ is symmetric positive semidefinite with orthonormal eigenvectors $\mathbf q_i$ and eigenvalues $\mu_i\ge0$. Writing $\mathbf x = \sum c_i\mathbf q_i$ with $\sum c_i^2 = 1$ gives $\mathbf x^TA^TA\mathbf x = \sum\mu_ic_i^2\le\mu_{\max}$, with equality at $\mathbf q_{\max}$. So $\|A\|_2 = \sqrt{\mu_{\max}}$. If $A = A^T$ then $A^TA = A^2$ has eigenvalues $\lambda_i^2$, so $\|A\|_2 = \max|\lambda_i| = \rho(A)$. (c) $A = \begin{pmatrix}0&1\\0&0\end{pmatrix}$ has $\rho = 0$ and $\|A\|_2 = 1$.

**A3.** The fixed point exists and is unique: if $\mathbf x = T\mathbf x+\mathbf c$ and $\mathbf y = T\mathbf y+\mathbf c$ then $\|\mathbf x-\mathbf y\|\le\|T\|\|\mathbf x-\mathbf y\|$, which forces $\mathbf x = \mathbf y$. Existence follows because $I-T$ is invertible, since $\rho(T)\le\|T\|<1$. Subtracting the fixed-point equation from the iteration gives $\mathbf x-\mathbf x^{(k)} = T(\mathbf x-\mathbf x^{(k-1)}) = T^k(\mathbf x-\mathbf x^{(0)})$, which proves the first bound. For the second bound, $\|\mathbf x^{(k+1)}-\mathbf x^{(k)}\|\le\|T\|^k\|\mathbf x^{(1)}-\mathbf x^{(0)}\|$, so for $m>k$
$\|\mathbf x^{(m)}-\mathbf x^{(k)}\|\le\sum_{j=k}^{m-1}\|T\|^j\|\mathbf x^{(1)}-\mathbf x^{(0)}\|\le\frac{\|T\|^k}{1-\|T\|}\|\mathbf x^{(1)}-\mathbf x^{(0)}\|$. Let $m\to\infty$.

**A4.** $(T_j)_{ij} = -a_{ij}/a_{ii}$ for $j\ne i$, so $\|T_j\|_\infty = \max_i\sum_{j\ne i}|a_{ij}|/|a_{ii}|<1$ by strict diagonal dominance. *Gauss–Seidel:* let $\mathbf y = T_g\mathbf x$ with $\|\mathbf x\|_\infty = 1$. Then $a_{ii}y_i = -\sum_{j<i}a_{ij}y_j - \sum_{j>i}a_{ij}x_j$. Take $i$ with $|y_i| = \|\mathbf y\|_\infty$. This gives $|a_{ii}||y_i|\le\sum_{j<i}|a_{ij}||y_i| + \sum_{j>i}|a_{ij}|$, so $\|\mathbf y\|_\infty\le\frac{\sum_{j>i}|a_{ij}|}{|a_{ii}|-\sum_{j<i}|a_{ij}|}<1$. The last inequality holds because $|a_{ii}|>\sum_{j<i}|a_{ij}|+\sum_{j>i}|a_{ij}|$.

**A5.** $D-\omega L$ is lower triangular with diagonal $D$, and $(1-\omega)D+\omega U$ is upper triangular with diagonal $(1-\omega)D$. So $\det T_\omega = \frac{\det[(1-\omega)D]}{\det D} = (1-\omega)^n$. Since $\det T_\omega = \prod\lambda_i$, we get $\rho(T_\omega)^n\ge|\prod\lambda_i| = |1-\omega|^n$. If $\omega\le0$ or $\omega\ge2$ then $\rho\ge1$ and SOR does not converge for every $\mathbf x^{(0)}$.

**A6.** (a) With $\mathbf r_{k+1} = \mathbf r_k-\alpha_kA\mathbf p_k$, $\alpha_k = \frac{\mathbf r_k^T\mathbf r_k}{\mathbf p_k^TA\mathbf p_k}$ and $\mathbf p_{k+1} = \mathbf r_{k+1}+\beta_k\mathbf p_k$, $\beta_k = \frac{\mathbf r_{k+1}^T\mathbf r_{k+1}}{\mathbf r_k^T\mathbf r_k}$. Assume $\mathbf r_i^T\mathbf r_j = 0$ and $\mathbf p_i^TA\mathbf p_j = 0$ for $i\ne j\le k$. Then:
- For $j<k$: $\mathbf r_{k+1}^T\mathbf r_j = \mathbf r_k^T\mathbf r_j - \alpha_k\mathbf p_k^TA\mathbf r_j$. Here $\mathbf r_j = \mathbf p_j-\beta_{j-1}\mathbf p_{j-1}$, so both terms vanish.
- For $j = k$: $\mathbf r_{k+1}^T\mathbf r_k = \mathbf r_k^T\mathbf r_k - \alpha_k\mathbf p_k^TA\mathbf r_k$. Since $\mathbf p_k^TA\mathbf r_k = \mathbf p_k^TA\mathbf p_k$ (conjugacy), this is $0$ by the choice of $\alpha_k$.
- $\mathbf p_{k+1}^TA\mathbf p_j = \mathbf r_{k+1}^TA\mathbf p_j + \beta_k\mathbf p_k^TA\mathbf p_j$. For $j<k$ both terms are $0$, because $A\mathbf p_j = (\mathbf r_j-\mathbf r_{j+1})/\alpha_j$ is orthogonal to $\mathbf r_{k+1}$. For $j = k$ the choice of $\beta_k$ makes it $0$ (a short computation using $A\mathbf p_k = (\mathbf r_k-\mathbf r_{k+1})/\alpha_k$).

(b) At most $n$ nonzero vectors in $\mathbb R^n$ can be mutually orthogonal, so $\mathbf r_n = \mathbf 0$. (c) $\mathbf x_k-\mathbf x_0\in\operatorname{span}\{\mathbf p_0,\dots,\mathbf p_{k-1}\} = \mathcal K_k$. The error $\mathbf e_k = \mathbf x^\ast-\mathbf x_k$ satisfies $A\mathbf e_k = \mathbf r_k\perp\mathcal K_k$, i.e. $\mathbf e_k$ is $A$-orthogonal to $\mathcal K_k$. That is exactly the optimality condition for minimising $\|\mathbf x^\ast-\mathbf x\|_A$ over $\mathbf x_0+\mathcal K_k$ (Pythagoras in the $A$-inner product).

## B. Hand computation

**B1.** $\|\mathbf x\|_1 = 19$, $\|\mathbf x\|_2 = 13$, $\|\mathbf x\|_\infty = 12$.
- $\|A\|_1 = \max(4,6) = 6$ and $\|A\|_\infty = \max(3,7) = 7$.
- $A^TA = \begin{pmatrix}10&10\\10&20\end{pmatrix}$ has eigenvalues $15\pm5\sqrt2 = 26.180, 3.820$, so $\|A\|_2 = \sqrt{26.180} = 5.117$.
- The eigenvalues of $A$ are $\frac{5\pm i\sqrt{15}}{2}$, so $\rho(A) = \sqrt{10} = 3.162$. This is below all three norms.

**B2.** The eigenvalues are $2-\sqrt2, 2, 2+\sqrt2 = 0.586, 2, 3.414$, with eigenvectors $(1,-\sqrt2,1)/2$, $(1,0,-1)/\sqrt2$ and $(1,\sqrt2,1)/2$. $\rho(B) = \|B\|_2 = 2+\sqrt2 = 3.414$ ($B$ is symmetric), while $\|B\|_\infty = 4$.

**B3.**

| $k$ | Jacobi | Gauss–Seidel |
|---|---|---|
| 1 | $(0.375, -0.28571, 1.66667)$ | $(0.375, -0.33929, 1.78373)$ |
| 2 | $(0.82738, -0.81548, 1.77183)$ | $(0.86334, -0.91869, 1.96675)$ |
| 3 | $(0.91989, -0.91015, 1.93981)$ | $(0.98152, -0.98786, 1.99525)$ |

$\|\mathbf x-\mathbf x^{(3)}\|_\infty$: Jacobi $0.090$, Gauss–Seidel $0.018$.

**B4.** $T_j = \begin{pmatrix}0&-\frac18&\frac14\\-\frac17&0&-\frac27\\\frac19&-\frac29&0\end{pmatrix}$, with $\|T_j\|_\infty = \frac37 = 0.429$ and $\rho(T_j) = 0.375$. $T_g = \begin{pmatrix}0&-0.125&0.25\\0&0.01786&-0.32143\\0&-0.01786&0.09921\end{pmatrix}$ (first column zero), with $\|T_g\|_\infty = 0.375$ and $\rho(T_g) = 0.145$. Predicted ratio of iteration counts: $\frac{\ln0.145}{\ln0.375} = 1.97$. Actual counts to $10^{-8}$: Jacobi 19, Gauss–Seidel 11, a ratio of $1.73$. The asymptotic rate only takes over after a few transient iterations.

**B5.** Iterates:

| | $\mathbf x^{(1)}$ | $\mathbf x^{(2)}$ |
|---|---|---|
| Gauss–Seidel | $(0.5, 1.125, 2.78125)$ | $(0.78125, 1.89063, 2.97266)$ |
| SOR, $\omega = 1.1$ | $(0.55, 1.25125, 3.09409)$ | $(0.83909, 2.05650, 3.00613)$ |

The exact solution is $(1,2,3)$. Since $T_j = \frac14\operatorname{tridiag}(1,0,1)$, $\rho(T_j) = \frac14\cdot2\cos\frac\pi4 = \frac{\sqrt2}{4} = 0.35355$ and $\rho(T_g) = 0.125$. Then $\omega^\ast = \frac{2}{1+\sqrt{1-1/8}} = 1.03337$ and $\rho(T_{\omega^\ast}) = \omega^\ast-1 = 0.0334$. Iterations to $10^{-10}$: $\omega = 1$: 12; $\omega^\ast$: 9; $\omega = 1.1$: 11.

**B6.** $A^{-1} = \begin{pmatrix}-10000&10000\\5000.5&-5000\end{pmatrix}$, $\|A\|_\infty = 3.0001$ and $\|A^{-1}\|_\infty = 20000$, so $K_\infty = 60002$. The perturbed right-hand side gives $\tilde{\mathbf x} = (3,0)$. The relative change in $\mathbf b$ is $6.7\times10^{-5}$ and the relative change in $\mathbf x$ is $2$. The bound allows $60002\times6.7\times10^{-5} = 4.0$, so the actual change is within the bound. The residual of $(3,0)$ in the original system is $(0,-0.0002)$, which is tiny, although the error $(2,-1)$ is 200% of $\|\mathbf x\|$.

**B7.** Start with $\mathbf r_0 = \mathbf p_0 = (5,5)$ and $A\mathbf p_0 = (20,15)$.
- $\alpha_0 = \frac{50}{175} = \frac27$, giving $\mathbf x_1 = (\frac{10}7,\frac{10}7)$ and $\mathbf r_1 = (-\frac57,\frac57)$.
- $\beta_0 = \frac{50/49}{50} = \frac1{49}$, giving $\mathbf p_1 = (-\frac{30}{49},\frac{40}{49})$.
- Checks: $\mathbf r_1^T\mathbf r_0 = 0$ and $\mathbf p_0^TA\mathbf p_1 = 0$.
- $\alpha_1 = 0.7$, giving $\mathbf x_2 = (1,2)$ exactly and $\mathbf r_2 = \mathbf 0$. The method terminates after $n = 2$ steps.

**B8.** $\lambda(A) = \frac{5\pm\sqrt5}{2} = 1.382, 3.618$, so $\kappa = 2.618$ and the guaranteed factor is $\frac{\kappa-1}{\kappa+1} = 0.447$. Iterates: $(1.42857,1.42857)$, $(0.95238,1.90476)$, $(1.02041,1.97279)$, $(0.99773,1.99546)$, $(1.00097,1.99870)$. The step lengths alternate between $\frac27$ and $\frac23$ (zig-zag). The observed $\|\mathbf e\|_A$ factor is $0.218$ at every step, below the worst-case bound $0.447$. The bound is attained only for the worst starting vector.

## C. Programming (key results)

**C1.** Theory gives $\rho(T_j) = \cos\frac{\pi}{21} = 0.98883$, which matches the numerical value, and $\omega^\ast = 1.7406$.

| method | Jacobi | GS | 1.2 | 1.4 | 1.6 | 1.7 | 1.74 ($\omega^\ast$) | 1.75 | 1.8 | 1.9 | 1.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iterations | 1216 | 609 | 404 | 257 | 142 | 90 | 62 | 60 | 78 | 148 | 298 |

Gauss–Seidel needs exactly half the Jacobi count ($\rho_g = \rho_j^2$), and optimal SOR is 20× faster than Gauss–Seidel. The minimum is flat but asymmetric: overestimating $\omega$ costs less than underestimating it.

**C2.**

| $m$ | $n$ | $\kappa$ | $\sqrt\kappa$ | steepest descent | CG | ILU-PCG |
|---|---|---|---|---|---|---|
| 16 | 256 | 117 | 10.8 | 1044 | 28 | 9 |
| 32 | 1024 | 441 | 21.0 | 4039 | 59 | 16 |
| 64 | 4096 | 1712 | 41.4 | — | 119 | 29 |
| 128 | 16384 | 6744 | 82.1 | — | 239 | 50 |

Steepest descent scales like $\kappa$ (×4 per doubling of $m$) and CG like $\sqrt\kappa$ (×2). ILU($10^{-2}$)-PCG, with 2.8× fill, cuts the count by a further factor of about 5. *Pitfall:* `spilu` pivots and reorders by default, which makes the preconditioner non-symmetric. CG then fails to converge, and it did stall in our first attempt. Use `permc_spec="NATURAL", diag_pivot_thresh=0`, or a genuine incomplete Cholesky factorisation.

**C3.** Relative errors after 0, 1, …, 6 refinement steps:
- $\kappa = 10^3$: $2.4\times10^{-5}\to4\times10^{-10}\to3\times10^{-14}$ (converged). A float64 direct solve gives $4\times10^{-14}$.
- $\kappa = 10^6$: $8.5\times10^{-3}\to7.5\times10^{-5}\to4.4\times10^{-7}\to4\times10^{-9}\to3\times10^{-11}\to1.2\times10^{-11}$. A float64 direct solve gives $1.8\times10^{-11}$.

Each step multiplies the error by about $\kappa u_{32}$ (about $10^{-4}$ and $10^{-2}$ respectively), so refinement converges while $\kappa u_{32}<1$, i.e. $\kappa\lesssim10^7$. The final accuracy is float64 quality ($\approx\kappa u_{64}$) at float32 factorisation cost.

## D. Data-science applications

**D1.**

| $\lambda$ | $\kappa(X^TX+\lambda I)$ | CG iterations to $10^{-10}$ | relative difference from direct solve |
|---|---|---|---|
| $10^{-4}$ | $1.07\times10^4$ | 523 | $4\times10^{-8}$ |
| $10^{-2}$ | $1.05\times10^4$ | 523 | $4\times10^{-8}$ |
| $1$ | $3.4\times10^3$ | 331 | $2\times10^{-8}$ |
| $100$ | $51$ | 62 | $6\times10^{-10}$ |

Ridge regularisation is also a *preconditioner*: it caps $\kappa$ at $\frac{\lambda_{\max}+\lambda}{\lambda}$. The count exceeds $p = 200$ because finite termination holds only in exact arithmetic. In floating point the residuals gradually lose orthogonality, so CG behaves like an iterative method with rate $\approx\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$. Matrix-free CG never forms $X^TX$ ($O(Np^2)$ flops) and never squares the condition number in storage precision.

**D2.** Power/Jacobi needs 25 iterations to reach $\|\cdot\|_1<10^{-10}$. Gauss–Seidel needs 67. Both agree with the direct solve to $\le2\times10^{-10}$, and $\sum x_i = 1$. The top five pages are all hubs (indices < 50). Stein–Rosenberg applies ($I-\alpha P^T$ has nonpositive off-diagonal entries) and gives $\rho(T_g)<\rho(T_j) = \alpha = 0.85$. So why is Gauss–Seidel slower here? The power-iteration error $\mathbf e_k$ has zero sum, which removes the component along the dominant eigenvector. Power iteration therefore contracts at $\alpha|\lambda_2(P)|$, which is small for this random, expander-like graph (observed factor ≈0.4 per step). Gauss–Seidel does not preserve that invariant subspace and contracts at $\rho(T_g)\approx0.71$. Lesson: the spectral radius governs the *worst case*, and structure in the error can make an "inferior" method faster. On real web graphs, where $|\lambda_2|\approx1$, Gauss–Seidel is typically about twice as fast.

**D3.** With 6 labels and 594 unlabelled points, $W$ has 55 440 nonzero entries after thresholding. Jacobi-PCG converges in 57 iterations. The harmonic solution classifies **100%** of the unlabelled points correctly. Logistic regression on the same 6 points reaches only **80.1%**, because a linear boundary cannot follow the moons. The Laplacian system propagates labels along the data manifold, and the solution $f_u$ is the probability that a random walk started at $u$ hits a class-1 label first.
