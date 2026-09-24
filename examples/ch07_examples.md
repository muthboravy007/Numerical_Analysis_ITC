# Chapter 7 — Worked Examples: Iterative Techniques in Matrix Algebra

Companion script: [`ch07_examples.py`](ch07_examples.py) reproduces every number below. · Lecture: [Chapter 7](../lectures/ch07-iterative-techniques.md)

---

## Example 7.1 — How good is an approximate solution?

**Problem.** For $A = \begin{pmatrix}4&1&0\\1&3&1\\0&1&2\end{pmatrix}$ and $\mathbf b = A(1,-2,3)^T$, someone reports $\tilde{\mathbf x} = (1.01, -2.02, 3.015)$. Bound the relative error without knowing the true solution.

**Solution.** The residual is $\mathbf r = \mathbf b-A\tilde{\mathbf x} = (-0.02, 0.035, -0.01)$, so $\|\mathbf r\|_\infty/\|\mathbf b\|_\infty = 8.75\times10^{-3}$. With $K_\infty(A) = 4.444$, Theorem 7.27 gives
$$\frac{\|\mathbf x-\tilde{\mathbf x}\|}{\|\mathbf x\|}\le K(A)\frac{\|\mathbf r\|}{\|\mathbf b\|} = 3.9\times10^{-2}.$$
The actual relative error is $6.7\times10^{-3}$, well inside the bound.

**Take-away.** The residual is always computable, and the condition number converts it into an error bound. For well-conditioned systems a small residual means a small error. For ill-conditioned systems it does not (Chapter 6, Hilbert matrices).

---

## Example 7.2 — Steady temperature in a plate

**Problem.** A square plate has its top edge at 100° and the other edges at 0°. Discretise it on a $15\times15$ interior grid (5-point Laplacian) and solve with Jacobi and Gauss–Seidel to $10^{-6}$.

**Solution.**

| method | iterations | observed residual ratio | theory |
|---|---|---|---|
| Jacobi | 571 | 0.98079 | $\rho(T_J) = \cos\frac{\pi}{16} = 0.98079$ |
| Gauss–Seidel | 290 | 0.96194 | $\rho(T_{GS}) = \rho_J^2 = 0.96194$ |

The centre temperature is $24.9994$. The exact limit is $25$: by superposition, four copies of the problem, one per hot edge, sum to a plate at 100° everywhere, so each contributes $\frac{100}4$ at the centre.

**Take-away.** The measured contraction rates match the spectral radii to 5 digits. Gauss–Seidel halves the work, but both methods slow down like $h^{-2}$ as the grid is refined.

---

## Example 7.3 — Choosing the SOR relaxation parameter

**Problem.** For $\operatorname{tridiag}(-1,2,-1)$ with $n = 50$, find the best $\omega$ and verify it numerically.

**Solution.** $\rho_J = \cos\frac{\pi}{51} = 0.998103$, so Theorem 7.26 gives $\omega^\ast = \frac{2}{1+\sqrt{1-\rho_J^2}} = 1.8840$ and $\rho(T_{\omega^\ast}) = \omega^\ast-1 = 0.8840$.

| $\omega$ | 1.0 | 1.5 | 1.8 | 1.85 | **1.884** | 1.95 | 1.99 |
|---|---|---|---|---|---|---|---|
| $\rho(T_\omega)$ | 0.9962 | 0.9886 | 0.9634 | 0.9449 | **0.8840** | 0.9500 | 0.9900 |
| iterations to $10^{-8}$ | 4828 | 1600 | 502 | 338 | **189** | 407 | 1887 |

**Take-away.** Optimal SOR is 25 times faster than Gauss–Seidel here. The minimum is sharp: $\rho(T_\omega)$ has an infinite slope on the left of $\omega^\ast$, so it is better to overestimate $\omega$ than to underestimate it.

---

## Example 7.4 — Conjugate gradients for Gaussian-process regression

**Problem.** Gaussian-process regression requires solving $(K+\sigma^2I)\boldsymbol\alpha = \mathbf y$ with the RBF kernel matrix $K$ (300 points, length-scale 1). How does the CG iteration count depend on the noise (jitter) $\sigma^2$?

**Solution.**

| $\sigma^2$ | $\kappa$ | CG iterations to $10^{-8}$ | $\sqrt\kappa$ |
|---|---|---|---|
| $10^{-1}$ | $7.5\times10^2$ | 27 | 27 |
| $10^{-2}$ | $7.5\times10^3$ | 38 | 87 |
| $10^{-4}$ | $7.5\times10^5$ | 83 | 866 |
| $10^{-6}$ | $7.5\times10^7$ | 198 | 8663 |

**Take-away.** The iteration count grows far more slowly than $\sqrt\kappa$. The kernel's eigenvalues decay fast: only a few are large and the rest cluster near $\sigma^2$. CG exploits such clustered spectra, which is why it (with preconditioning) powers scalable GP libraries.

---

## Example 7.5 — Diagonal (Jacobi) preconditioning

**Problem.** An SPD matrix has a benign spectrum, but its variables are scaled over two orders of magnitude, like features in different units: $A = DBD$ with $\kappa(B) = 10$ and $D = \operatorname{diag}(10^{0\dots2})$. Compare CG with Jacobi-preconditioned CG.

**Solution.** $\kappa(A) = 1.8\times10^4$ but $\kappa(D_A^{-1/2}AD_A^{-1/2}) = 10.1$. Plain CG needs 798 iterations to $10^{-10}$, while Jacobi-PCG needs **38**.

**Take-away.** Diagonal preconditioning is the linear-algebra version of *standardising your features*. It is free and often removes most of the ill-conditioning caused by units.

---

## Example 7.6 — CG as regularisation: semi-convergence

**Problem.** A signal (a box plus a bump) is blurred by a Gaussian kernel $K$ (width 0.03, $n = 200$), and noise with $\sigma = 0.01$ is added. Apply CG to the normal equations $K^TK\mathbf x = K^T\mathbf y$ and track the error against the true signal.

**Solution.** $\kappa(K) = 2.3\times10^{19}$.

| CG iterations | 1 | 5 | 10 | **19** | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| relative error | 0.306 | 0.196 | 0.184 | **0.172** | 0.426 | 2.05 | 5.92 |

The "exact" solution of the normal equations has relative error $1.5\times10^3$: it is pure noise amplification.

**Take-away.** For ill-posed inverse problems, early CG iterations capture the large singular components (signal), and later ones fit noise. Stopping early *is* a regulariser, analogous to truncated SVD or ridge. The same phenomenon underlies early stopping in machine learning.
