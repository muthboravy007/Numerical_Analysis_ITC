# Lecture 5 — Least Squares, QR and the Singular Value Decomposition

> **Week 7–8** · Code: [`numlib/least_squares.py`](../numlib/least_squares.py) · Examples: [`examples/ch05`](../examples/ch05_examples.md) · Exercises: [`exercises/ch05`](../exercises/ch05_exercises.md)

## Learning objectives

1. Formulate linear regression as the least-squares problem $\min_{\mathbf x}\|A\mathbf x - \mathbf b\|_2$ and derive the normal equations geometrically.
2. Explain why the normal equations can be inaccurate ($\kappa(A^\top A) = \kappa(A)^2$).
3. Compute QR factorisations by Gram–Schmidt and Householder reflections, and solve least squares with QR.
4. State the SVD, interpret its factors, and use it for pseudoinverses, rank determination and low-rank approximation.
5. Understand ridge regression as regularisation of small singular values.

---

## 5.1 The least-squares problem

Given data $(t_i, y_i)$, $i=1,\dots,m$, fit a model $y \approx \sum_{j=1}^n x_j\,\phi_j(t)$ with $m > n$. With the **design matrix** $A_{ij} = \phi_j(t_i)$, we want

$$
\min_{\mathbf x\in\mathbb R^n}\ \|A\mathbf x - \mathbf b\|_2^2 = \sum_{i=1}^m\bigl(b_i - (A\mathbf x)_i\bigr)^2 .
$$

The system $A\mathbf x = \mathbf b$ is **overdetermined**; in general no exact solution exists.

### Geometry and the normal equations

$A\mathbf x$ ranges over the column space $\mathcal R(A)$. The closest point to $\mathbf b$ is the **orthogonal projection**; the residual $\mathbf r = \mathbf b - A\mathbf x$ must be orthogonal to every column:
$$
A^\top(\mathbf b - A\mathbf x) = \mathbf 0 \quad\Longleftrightarrow\quad \boxed{A^\top A\,\mathbf x = A^\top\mathbf b.}
$$
If $A$ has full column rank, $A^\top A$ is SPD and the solution is unique: $\mathbf x = (A^\top A)^{-1}A^\top\mathbf b$. The **hat matrix** $H = A(A^\top A)^{-1}A^\top$ projects $\mathbf b$ onto $\mathcal R(A)$.

**Example.** Fit $y = x_1 + x_2 t$ to $(0,1), (1,2.9), (2,5.2), (3,7.1), (4,8.8)$:
$$
A^\top A = \begin{pmatrix}5&10\\10&30\end{pmatrix},\quad A^\top\mathbf y = \begin{pmatrix}25\\69.8\end{pmatrix}
\ \Rightarrow\ x_1 = 1.04,\ x_2 = 1.98,\ R^2 = 0.9976.
$$

### The trouble with normal equations

$\kappa_2(A^\top A) = \kappa_2(A)^2$. If $\kappa_2(A) = 10^{8}$ (common with polynomial features or collinear predictors), the normal equations have $\kappa = 10^{16}$ and **all** digits may be lost, even though the least-squares problem itself only warrants losing about 8. Remedy: never form $A^\top A$ — use QR or SVD.

---

## 5.2 QR factorisation

**Theorem.** Every $A\in\mathbb R^{m\times n}$ with $m\ge n$ can be written $A = QR$ with $Q\in\mathbb R^{m\times m}$ orthogonal ($Q^\top Q = I$) and $R\in\mathbb R^{m\times n}$ upper triangular. The **thin (reduced) QR** keeps the first $n$ columns: $A = \hat Q\hat R$, $\hat Q\in\mathbb R^{m\times n}$, $\hat R\in\mathbb R^{n\times n}$.

### Least squares via QR
Orthogonal matrices preserve the 2-norm, so
$$
\|A\mathbf x - \mathbf b\|_2^2 = \|Q^\top A\mathbf x - Q^\top\mathbf b\|_2^2 = \|\hat R\mathbf x - \hat Q^\top\mathbf b\|_2^2 + \|(\text{rest of }Q^\top\mathbf b)\|_2^2 .
$$
Minimise by solving the triangular system $\boxed{\hat R\mathbf x = \hat Q^\top\mathbf b}$. The condition number involved is $\kappa(A)$, not $\kappa(A)^2$. Cost $\approx 2mn^2 - \tfrac23n^3$ flops.

### Gram–Schmidt
Orthonormalise the columns $\mathbf a_1,\dots,\mathbf a_n$ one at a time:
$$
\mathbf v_j = \mathbf a_j - \sum_{i<j}(\mathbf q_i^\top\mathbf a_j)\,\mathbf q_i,\qquad r_{ij} = \mathbf q_i^\top\mathbf a_j,\qquad r_{jj} = \|\mathbf v_j\|,\qquad \mathbf q_j = \mathbf v_j/r_{jj}.
$$
*Classical* GS loses orthogonality badly in floating point; *modified* GS (subtract each projection from the updated vector immediately) is much better but still loses orthogonality proportional to $\kappa(A)\,u$.

### Householder reflections (what LAPACK uses)
A **Householder reflector** $H = I - 2\mathbf v\mathbf v^\top/\mathbf v^\top\mathbf v$ is symmetric and orthogonal. Choosing
$$
\mathbf v = \mathbf x + \operatorname{sign}(x_1)\|\mathbf x\|_2\,\mathbf e_1
$$
maps $\mathbf x$ to $-\operatorname{sign}(x_1)\|\mathbf x\|_2\mathbf e_1$, zeroing all but the first entry. (The sign choice avoids cancellation.) Applying $H_1, H_2, \ldots, H_n$ column by column gives $H_n\cdots H_1A = R$, so $Q = H_1\cdots H_n$. Householder QR is **backward stable** and $Q$ is orthogonal to machine precision.

**Givens rotations** zero one entry at a time; useful for sparse matrices and for updating a QR when a new data row arrives (online regression).

---

## 5.3 The singular value decomposition

**Theorem (SVD).** Every $A\in\mathbb R^{m\times n}$ can be written
$$
A = U\Sigma V^\top = \sum_{i=1}^{r}\sigma_i\,\mathbf u_i\mathbf v_i^\top,
$$
with $U\in\mathbb R^{m\times m}$, $V\in\mathbb R^{n\times n}$ orthogonal and $\Sigma$ diagonal with $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$, $r = \operatorname{rank}(A)$.

**Interpretation.** $A$ maps the unit sphere to an ellipsoid; the $\sigma_i$ are the semi-axis lengths, $\mathbf v_i$ the input directions and $\mathbf u_i$ the output directions: $A\mathbf v_i = \sigma_i\mathbf u_i$.

**Relation to eigenvalues.** $A^\top A = V\Sigma^2V^\top$ and $AA^\top = U\Sigma^2U^\top$, so $\sigma_i^2$ are eigenvalues of $A^\top A$. (But don't *compute* the SVD this way — same squaring problem.)

**What the SVD tells you**

| Quantity | From the SVD |
|---|---|
| $\|A\|_2$ | $\sigma_1$ |
| $\kappa_2(A)$ | $\sigma_1/\sigma_n$ |
| Numerical rank | # of $\sigma_i > \text{tol}$ (e.g. $\text{tol}=\max(m,n)\,u\,\sigma_1$) |
| $\|A\|_F$ | $\sqrt{\sum\sigma_i^2}$ |
| Range / null space | first $r$ columns of $U$ / last $n-r$ columns of $V$ |

### Pseudoinverse and minimum-norm solutions
$$
A^{+} = V\Sigma^{+}U^\top,\qquad \Sigma^+ = \operatorname{diag}(1/\sigma_1,\dots,1/\sigma_r,0,\dots).
$$
$\mathbf x^+ = A^+\mathbf b = \sum_{i=1}^r \frac{\mathbf u_i^\top\mathbf b}{\sigma_i}\mathbf v_i$ is the least-squares solution of **minimum norm** — well-defined even when $A$ is rank-deficient (perfectly collinear features). This is what `np.linalg.lstsq` returns.

### Low-rank approximation

**Eckart–Young–Mirsky Theorem.** The best rank-$k$ approximation to $A$ in both the 2-norm and Frobenius norm is the truncated SVD $A_k = \sum_{i=1}^k\sigma_i\mathbf u_i\mathbf v_i^\top$, and
$$
\|A - A_k\|_2 = \sigma_{k+1},\qquad \|A-A_k\|_F = \sqrt{\textstyle\sum_{i>k}\sigma_i^2}.
$$

Applications: image compression (store $k(m+n+1)$ numbers instead of $mn$), latent semantic analysis of term–document matrices, recommender systems (matrix factorisation), denoising, and PCA (Lecture 6).

---

## 5.4 Regularisation: ridge regression

Small singular values amplify noise: the component $\frac{\mathbf u_i^\top\mathbf b}{\sigma_i}$ blows up when $\sigma_i\approx 0$. **Ridge (Tikhonov) regression** solves
$$
\min_{\mathbf x}\ \|A\mathbf x - \mathbf b\|_2^2 + \lambda\|\mathbf x\|_2^2 \quad\Longleftrightarrow\quad (A^\top A + \lambda I)\mathbf x = A^\top\mathbf b,
$$
$$
\mathbf x_\lambda = \sum_{i}\underbrace{\frac{\sigma_i^2}{\sigma_i^2+\lambda}}_{\text{filter factor}}\frac{\mathbf u_i^\top\mathbf b}{\sigma_i}\mathbf v_i .
$$
Directions with $\sigma_i^2 \gg \lambda$ pass through; those with $\sigma_i^2\ll\lambda$ are damped. Also $\kappa(A^\top A+\lambda I) = \frac{\sigma_1^2+\lambda}{\sigma_n^2+\lambda}$ — regularisation *improves conditioning*. Computing the SVD once lets you evaluate $\mathbf x_\lambda$ for many $\lambda$ cheaply (efficient cross-validation).

Equivalently, ridge is ordinary least squares on the augmented system $\begin{pmatrix}A\\ \sqrt\lambda I\end{pmatrix}\mathbf x \approx \begin{pmatrix}\mathbf b\\ \mathbf 0\end{pmatrix}$ — solve it with QR.

---

## 5.5 Choosing an algorithm

| Method | Cost ($m\gg n$) | Accuracy depends on | Use when |
|---|---|---|---|
| Normal equations + Cholesky | $mn^2$ | $\kappa(A)^2$ | Well-conditioned, huge $m$, speed critical |
| Householder QR | $2mn^2$ | $\kappa(A)$ | **Default** (`scipy.linalg.lstsq(lapack_driver='gelsy')`) |
| SVD | $\approx 2mn^2 + 11n^3$ | $\kappa(A)$, handles rank deficiency | Rank-deficient / diagnostics (`np.linalg.lstsq`) |
| Iterative (LSQR, CG) | $k\cdot\text{nnz}$ | $\kappa$ via iterations | Sparse/huge |

## 5.6 Summary

* Least squares = orthogonal projection; normal equations are the optimality condition but square $\kappa$.
* QR (Householder) solves least squares stably via $R\mathbf x = Q^\top\mathbf b$.
* SVD reveals rank, conditioning, and gives the pseudoinverse and optimal low-rank approximations.
* Ridge regression filters small singular values and improves conditioning.

## Further reading

* Trefethen & Bau, Lectures 4–11 and 18–19.
* Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, §3.4 (ridge via SVD).
* G. Strang, *Linear Algebra and Learning from Data*, 2019.
