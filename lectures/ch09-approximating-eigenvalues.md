# Chapter 9 — Approximating Eigenvalues

> **Reference:** Burden & Faires, Chapter 9 (§9.1–9.6) plus §9.7 for data science.
> **Code:** [`numlib/eigen.py`](../numlib/eigen.py) · **All numbers reproduced by** [`lectures/code/ch09.py`](code/ch09.py)
> **Worked problems:** [`examples/ch09`](../examples/ch09_examples.md) · **Homework:** [`exercises/ch09`](../exercises/ch09_exercises.md)
> **See also:** Chapter 14 (symmetric matrices, SVD and PCA in depth).

## Learning outcomes

1. Localise eigenvalues with the Gershgorin circle theorem; use linear independence and orthogonality of eigenvectors.
2. State the spectral theorem and Schur's theorem; use orthogonal similarity transformations.
3. Apply the power, symmetric power, inverse power and deflation methods, and state their convergence rates.
4. Reduce symmetric matrices to tridiagonal (and general matrices to Hessenberg) form with Householder reflections.
5. Explain the QR algorithm with shifts and deflation.
6. Compute and interpret the singular value decomposition.
7. Apply eigen-methods to Markov chains, PageRank, spectral clustering, PCA and link analysis.

**Never compute eigenvalues from the characteristic polynomial.** Its roots are extremely sensitive to its coefficients (Wilkinson, Example 1.4.5). All practical methods work with the matrix directly.

---

## 9.1 Linear Algebra and Eigenvalues

**Theorem 9.1 (Gershgorin circle theorem).** Let $R_i = \{z\in\mathbb C: |z-a_{ii}|\le\sum_{j\ne i}|a_{ij}|\}$. Every eigenvalue of $A$ lies in $\bigcup_iR_i$. Moreover, any union of $k$ discs that is disjoint from the other $n-k$ discs contains exactly $k$ eigenvalues (counted with multiplicity).

Since $A$ and $A^T$ have the same eigenvalues, column sums may be used as well.

**Theorem 9.2.** Eigenvectors belonging to *distinct* eigenvalues are linearly independent. Hence a matrix with $n$ distinct eigenvalues has a basis of eigenvectors.

**Orthogonality.** $\{\mathbf v_i\}$ is orthogonal if $\mathbf v_i^T\mathbf v_j = 0$ ($i\ne j$) and orthonormal if in addition $\|\mathbf v_i\|_2 = 1$. Orthogonal nonzero vectors are linearly independent (Theorem 9.4). Gram–Schmidt converts any independent set into an orthonormal one:
$$
\mathbf u_k = \mathbf v_k - \sum_{j<k}(\mathbf v_k^T\mathbf q_j)\mathbf q_j,\qquad\mathbf q_k = \mathbf u_k/\|\mathbf u_k\|_2 .
$$

### Examples for §9.1

**Example 9.1.1 (Gershgorin discs).** $A = \begin{pmatrix}5&1&0\\1&-2&1\\0.5&0.5&8\end{pmatrix}$: row discs centred at $5, -2, 8$ with radii $1, 2, 1$: $R_1 = [4,6]$, $R_2 = [-4,0]$, $R_3 = [7,9]$ (on the real line; in $\mathbb C$ they are discs). They are pairwise disjoint, so each contains exactly one eigenvalue: $5.1135\in R_1$, $-2.1815\in R_2$, $8.0681\in R_3$.

**Example 9.1.2 (real eigenvalues from disjoint discs).** A real matrix has complex eigenvalues in conjugate pairs. Since each disc of Example 9.1.1 is isolated and symmetric about the real axis, its single eigenvalue must be real — even though $A$ is not symmetric. Column discs (radii $1.5, 1.5, 1$) give an alternative localisation; intersecting row and column information sharpens it.

**Example 9.1.3 (independent eigenvectors).** $B = \begin{pmatrix}2&1&0\\0&3&1\\0&0&5\end{pmatrix}$ (upper triangular) has eigenvalues $2,3,5$ (the diagonal). The eigenvectors $(1,0,0)$, $(1,1,0)/\sqrt2$, $(1,3,6)/\sqrt{46}$ form a nonsingular matrix (rank 3), as Theorem 9.2 promises.

**Example 9.1.4 (Gram–Schmidt).** From $(1,1,0)$, $(1,0,1)$, $(0,1,1)$: $\mathbf q_1 = \frac1{\sqrt2}(1,1,0)$; $\mathbf u_2 = (1,0,1) - \frac12(1,1,0) = (\frac12,-\frac12,1)$, $\mathbf q_2 = \frac{1}{\sqrt6}(1,-1,2)$; $\mathbf q_3 = \frac{1}{\sqrt3}(-1,1,1)$. $QQ^T = I$.

**Example 9.1.5 (bounding a covariance spectrum).** $\Sigma = \begin{pmatrix}4&0.8&-0.3\\0.8&2&0.5\\-0.3&0.5&1.5\end{pmatrix}$: discs $[2.9,5.1]$, $[0.7,3.3]$, $[0.7,2.3]$. All lie in $(0,\infty)$, so $\Sigma$ is positive definite, and $\lambda_{\max}\le5.1$ (true eigenvalues $1.026, 2.189, 4.286$) — obtained without any computation.

---

## 9.2 Orthogonal Matrices and Similarity Transformations

$Q$ is **orthogonal** if $Q^TQ = I$ (columns orthonormal). Then $Q^{-1} = Q^T$, $\|Q\mathbf x\|_2 = \|\mathbf x\|_2$, and products of orthogonal matrices are orthogonal. $A$ and $B$ are **similar** if $A = S^{-1}BS$; similar matrices have the same eigenvalues (with multiplicities).

**Theorem 9.10 (diagonalisation).** $A$ is similar to a diagonal matrix $D$ iff $A$ has $n$ linearly independent eigenvectors; then $D = S^{-1}AS$ with the eigenvectors as columns of $S$.

**Theorem 9.12 (Schur).** For every $A$ there is a unitary $U$ with $T = U^*AU$ upper triangular; the diagonal of $T$ holds the eigenvalues. (For real $A$ a real *quasi*-triangular form with $2\times2$ blocks for complex pairs exists.)

**Theorem 9.13 (spectral theorem).** A real symmetric $A$ has real eigenvalues and an orthonormal basis of eigenvectors: $A = QDQ^T$.

**Theorem 9.14.** A symmetric $A$ is positive definite iff all its eigenvalues are positive.

### Examples for §9.2

**Example 9.2.1 (orthogonal diagonalisation).** $S = \begin{pmatrix}2&1&0\\1&2&1\\0&1&2\end{pmatrix}$ has eigenvalues $2-\sqrt2 = 0.5858$, $2$, $2+\sqrt2 = 3.4142$ with orthonormal eigenvectors $\frac12(1,-\sqrt2,1)$, $\frac1{\sqrt2}(1,0,-1)$, $\frac12(1,\sqrt2,1)$ (up to sign), and $Q^TSQ = \operatorname{diag}(0.5858, 2, 3.4142)$.

**Example 9.2.2 (a defective matrix).** $J = \begin{pmatrix}2&1\\0&2\end{pmatrix}$ has the double eigenvalue $2$ but only one independent eigenvector $(1,0)^T$: not diagonalisable. Numerically, `eig` returns two (almost) parallel vectors — a sign of a defective or nearly defective matrix, for which eigenvectors are ill-conditioned.

**Example 9.2.3 (similarity preserves eigenvalues).** $M = \begin{pmatrix}1&2\\3&0\end{pmatrix}$ (eigenvalues $3, -2$) and $P = \begin{pmatrix}1&1\\0&1\end{pmatrix}$: $P^{-1}MP = \begin{pmatrix}-2&0\\3&3\end{pmatrix}$ — a lower-triangular matrix with the same eigenvalues $-2, 3$ on its diagonal.

**Example 9.2.4 (Schur form).** $N = \begin{pmatrix}1&2&3\\0&4&5\\1&0&6\end{pmatrix}$: the real Schur form is upper triangular with diagonal $7.0410,\ 1.0885,\ 2.8705$ — the eigenvalues — obtained with orthogonal (perfectly conditioned) transformations. This is what the QR algorithm computes.

**Example 9.2.5 (orthogonal matrices preserve length; PD via eigenvalues).** A rotation $R_\theta$ ($\theta = 30^\circ$) has $R^TR = I$, eigenvalues $e^{\pm i\pi/6} = 0.866\pm0.5i$ (modulus 1), and $\|R(3,4)^T\|_2 = 5$. The symmetric $K = \begin{pmatrix}2&-1\\-1&2\end{pmatrix}$ has eigenvalues $1, 3>0$, so it is PD; the quadratic form $\mathbf x^TK\mathbf x$ equals $2 = 2\lambda_1$ along $(1,1)$ and $6 = 2\lambda_2$ along $(1,-1)$ — the extremes of the Rayleigh quotient.

---

## 9.3 The Power Method

Assume $|\lambda_1|>|\lambda_2|\ge\cdots\ge|\lambda_n|$ with independent eigenvectors $\mathbf v_i$. Writing $\mathbf x^{(0)} = \sum\beta_j\mathbf v_j$ ($\beta_1\neq0$),
$$
A^k\mathbf x^{(0)} = \lambda_1^k\Bigl[\beta_1\mathbf v_1 + \sum_{j\ge2}\beta_j\Bigl(\frac{\lambda_j}{\lambda_1}\Bigr)^k\mathbf v_j\Bigr]\ \longrightarrow\ \text{direction of }\mathbf v_1 .
$$

**Power method (B&F Algorithm 9.1, $\ell_\infty$ scaling).** Normalise so that $\|\mathbf x^{(k)}\|_\infty = 1$ with $x_{p}^{(k)} = 1$; then $\mathbf y = A\mathbf x^{(k)}$, $\mu^{(k)} = y_{p}$, $p\leftarrow\arg\max|y_i|$, $\mathbf x^{(k+1)} = \mathbf y/y_p$. Then $\mu^{(k)}\to\lambda_1$ **linearly** with ratio $|\lambda_2/\lambda_1|$. Aitken's $\Delta^2$ (Chapter 2) accelerates it.

**Symmetric power method (Algorithm 9.2).** For symmetric $A$, use $\ell_2$ scaling and the **Rayleigh quotient** $\mu^{(k)} = \mathbf x^{(k)T}A\mathbf x^{(k)}$; the error is $O(|\lambda_2/\lambda_1|^{2k})$ — twice as many digits per iteration.

**Inverse power method (Algorithm 9.3).** The eigenvalues of $(A-qI)^{-1}$ are $1/(\lambda_i-q)$. The power method applied to it (solving $(A-qI)\mathbf y = \mathbf x$ with a single $LU$ factorisation) converges to the eigenvalue **closest to $q$** with ratio $\frac{|\lambda-q|}{|\lambda'-q|}$ ($\lambda'$ = second closest). Updating $q$ by the Rayleigh quotient gives cubic convergence (Rayleigh quotient iteration).

**Deflation.** After finding $\lambda_1,\mathbf v_1$: for symmetric $A$, $A - \lambda_1\mathbf v_1\mathbf v_1^T$ ($\|\mathbf v_1\|_2 = 1$) has eigenvalues $0,\lambda_2,\dots,\lambda_n$ (Hotelling). **Wielandt deflation** (general $A$): with $i$ such that $v_{1,i}\ne0$, let $\mathbf x = \frac{1}{\lambda_1v_{1,i}}(a_{i1},\dots,a_{in})^T$; then $B = A - \lambda_1\mathbf v_1\mathbf x^T$ has eigenvalues $0,\lambda_2,\dots,\lambda_n$, row $i$ of $B$ is zero, and deleting row and column $i$ gives an $(n-1)\times(n-1)$ matrix.

### Examples for §9.3

**Example 9.3.1 (power method).** $A = \begin{pmatrix}4&-1&1\\-1&3&-2\\1&-2&3\end{pmatrix}$ (eigenvalues $6,3,1$), $\mathbf x^{(0)} = (1,0,0)^T$:

| $k$ | $\mathbf x^{(k)}$ | $\mu^{(k)}$ |
|---|---|---|
| 1 | $(1, -0.25, 0.25)$ | 4.0000000 |
| 2 | $(1, -0.5, 0.5)$ | 4.5000000 |
| 3 | $(1, -0.7, 0.7)$ | 5.0000000 |
| 5 | $(1, -0.911765, 0.911765)$ | 5.6666667 |
| 10 | $(1, -0.997076, 0.997076)$ | 5.9883268 |

The errors $2, 1.5, 1, 0.6, 0.33, 0.18,\ldots$ halve each step: ratio $\lambda_2/\lambda_1 = 3/6$. Eigenvector $\to(1,-1,1)$.

**Example 9.3.2 (Aitken acceleration).** Applying $\Delta^2$ to $\mu^{(k)}$: $7.0,\ 6.2,\ 6.0476,\ 6.0118,\ 6.0029$ — errors $0.012, 0.003$ where the plain method has $0.33, 0.18$.

**Example 9.3.3 (symmetric power method).** Rayleigh quotients from $\mathbf x^{(0)} = (1,0,0)$: $4,\ 5,\ 5.6667,\ 5.9091,\ 5.9767,\ 5.9942,\ 5.9985,\ 5.9996$ — errors $2, 1, 0.33, 0.091, 0.023, 0.0058, 0.0015, 0.00037$ divide by **4** each step ($=(3/6)^2$).

**Example 9.3.4 (inverse power method).** With shift $q = 0.8$: $\mu = 2,\ 1.0067,\ 1.000047,\ 1.0000000$ → $\lambda_3 = 1$ (ratio $\frac{0.2}{2.2} = 0.09$). With $q = 2.9$: $2,\ 2.98507,\ 2.999956,\ 3.0000000$ → $\lambda_2 = 3$ (ratio $\frac{0.1}{1.9}$). Any eigenvalue can be targeted with a good shift.

**Example 9.3.5 (Wielandt deflation).** With $\lambda_1 = 6$, $\mathbf v_1 = (1,-1,1)^T$, $i = 1$: $\mathbf x = \frac{1}{6}(4,-1,1)^T$,
$$
B = A - 6\mathbf v_1\mathbf x^T = \begin{pmatrix}0&0&0\\3&2&-1\\-3&-1&2\end{pmatrix},
$$
eigenvalues $0, 3, 1$. The power method on the $2\times2$ block $\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$ returns $\lambda_2 = 3$.

**Example 9.3.6 (how many iterations?).** Reaching $10^{-8}$ needs about $\frac{\ln10^{-8}}{\ln r}$ iterations: $r = 0.5$: 27; $r = 0.9$: 175; $r = 0.99$: 1833. When $|\lambda_2|\approx|\lambda_1|$ use shifts, inverse iteration or Lanczos.

---

## 9.4 Householder's Method

A **Householder reflector** $P = I - 2\mathbf w\mathbf w^T$ ($\|\mathbf w\|_2 = 1$) is symmetric and orthogonal ($P^2 = I$). Given $\mathbf x$, choosing $\mathbf w\propto\mathbf x - \alpha\mathbf e_1$ with $\alpha = -\operatorname{sign}(x_1)\|\mathbf x\|_2$ gives $P\mathbf x = \alpha\mathbf e_1$ (the sign avoids cancellation).

**Householder tridiagonalisation (B&F Algorithm 9.5).** For symmetric $A$, apply $P^{(k)}$ that zeros entries $k+2,\dots,n$ of column $k$ while leaving row/column $1..k$ intact:
$$
A^{(k+1)} = P^{(k)}A^{(k)}P^{(k)},\qquad k = 1,\dots,n-2 .
$$
The result is symmetric tridiagonal and **orthogonally similar** to $A$ (same eigenvalues, perfectly stable). Cost $\approx\frac23n^3$. For non-symmetric $A$ the same process produces **upper Hessenberg** form ($a_{ij} = 0$ for $i>j+1$).

Why bother? A QR step costs $O(n^3)$ for a full matrix but only $O(n)$ for tridiagonal and $O(n^2)$ for Hessenberg matrices, and the QR algorithm preserves these forms.

### Examples for §9.4

**Example 9.4.1 (tridiagonalising a 4×4).** $A = \begin{pmatrix}4&2&2&1\\2&-3&1&1\\2&1&3&1\\1&1&1&2\end{pmatrix}$ becomes
$$
T = \begin{pmatrix}4&-3&0&0\\-3&2&3.1623&0\\0&3.1623&-1.4&-0.2\\0&0&-0.2&1.4\end{pmatrix},
$$
with the same eigenvalues $-3.6458,\ 1.3542,\ 1.6458,\ 6.6458$. (First step: $\|(2,2,1)\| = 3$, so $t_{21} = -\operatorname{sign}(2)\cdot3 = -3$.)

**Example 9.4.2 (one reflector).** $\mathbf x = (3,4,0)^T$: $\alpha = -5$, $\mathbf w\propto\mathbf x+5\mathbf e_1 = (8,4,0)$, $\mathbf w = (0.8944, 0.4472, 0)$, $P = \begin{pmatrix}-0.6&-0.8&0\\-0.8&0.6&0\\0&0&1\end{pmatrix}$, $P\mathbf x = (-5,0,0)^T$.

**Example 9.4.3 (Hessenberg form).** The non-symmetric $\begin{pmatrix}4&1&-2&2\\1&2&0&1\\2&1&3&-2\\1&3&-1&5\end{pmatrix}$ reduces to an upper Hessenberg matrix with subdiagonal $(-2.4495,\ -1.2802,\ -1.3285)$ and the same eigenvalues $0.8023,\ 3.1391\pm1.7650i,\ 6.9194$.

**Example 9.4.4 (cost).** $n = 1000$: tridiagonalisation $\approx6.7\times10^8$ flops once; afterwards each QR step costs $\approx1.4\times10^4$ instead of $\approx1.3\times10^9$.

**Example 9.4.5 (stability).** For Example 9.4.1, $\|Q^TQ - I\| = 2.7\times10^{-16}$ and $\|Q^TAQ - T\| = 5.6\times10^{-16}$: orthogonal transformations do not amplify rounding errors.

---

## 9.5 The QR Algorithm

```text
A(1) = A (tridiagonal or Hessenberg)
for k = 1, 2, ...
    choose shift s_k
    A(k) - s_k I = Q(k) R(k)          (QR factorisation, e.g. by Givens rotations)
    A(k+1) = R(k) Q(k) + s_k I        (= Q(k)^T A(k) Q(k): orthogonally similar)
    if some subdiagonal entry is negligible: deflate (split the problem)
```
**Convergence (unshifted, symmetric, distinct $|\lambda_i|$):** $a^{(k)}_{i+1,i}\to0$ like $|\lambda_{i+1}/\lambda_i|^k$ and the diagonal tends to the eigenvalues in decreasing order of magnitude. **Shifts** $s_k\approx\lambda_n$ make $a_{n,n-1}$ converge quadratically (Rayleigh shift) or cubically for symmetric matrices (Wilkinson shift: the eigenvalue of the trailing $2\times2$ block closer to $a_{nn}$). Complex conjugate eigenvalues of real matrices appear as $2\times2$ blocks (double-shift QR, Francis). Total cost ≈ $10n^3$ (general) or $\frac43n^3$ (symmetric, eigenvalues only). This is LAPACK's `dsyev`/`dgeev`, i.e. `np.linalg.eig`.

### Examples for §9.5

**Example 9.5.1 (unshifted QR).** $T = \begin{pmatrix}3&1&0\\1&3&1\\0&1&3\end{pmatrix}$, eigenvalues $3+\sqrt2,\ 3,\ 3-\sqrt2$:

| $k$ | diagonal | $\lvert a_{21}\rvert$ | $\lvert a_{32}\rvert$ |
|---|---|---|---|
| 1 | $(3.6, 3.1297, 2.2703)$ | 0.860 | 0.897 |
| 3 | $(4.1767, 3.1727, 1.6506)$ | 0.535 | 0.309 |
| 8 | $(4.4084, 3.0057, 1.5859)$ | 0.0907 | 0.0122 |
| 12 | $(4.4139, 3.0003, 1.5858)$ | 0.0194 | 0.00095 |

$|a_{21}|$ decays by $\frac{3}{4.414} = 0.68$ and $|a_{32}|$ by $\frac{1.586}{3} = 0.53$ per step — slow.

**Example 9.5.2 (Wilkinson shift).** Same matrix: $|a_{32}| = 0.707,\ 0.0304,\ 4.5\times10^{-7},\ 4.6\times10^{-22}$ — **cubic** convergence (the exponent triples each step). After 3 steps $\lambda_3 = 3-\sqrt2$ is known to machine precision and the problem deflates to $2\times2$.

**Example 9.5.3 (complex eigenvalues).** For a real matrix with eigenvalues $2.1084$ and $0.0958\pm0.9934i$, unshifted QR converges to a block upper-triangular form: $a_{11} = 2.1084$, and a trailing $2\times2$ block whose eigenvalues are $0.0958\pm0.9934i$ (its entries keep rotating; only the block's eigenvalues converge).

**Example 9.5.4 (polynomial roots by QR).** The companion matrix of $x^3-6x^2+11x-6$ is $\begin{pmatrix}6&-11&6\\1&0&0\\0&1&0\end{pmatrix}$; shifted QR gives $3, 2, 1$ — exactly what `np.roots` does (Chapter 2).

**Example 9.5.5 (deflation).** For $\begin{pmatrix}4&1&0&0\\1&3&1&0\\0&1&2&1\\0&0&1&1\end{pmatrix}$, 5 shifted QR steps make $|a_{43}|<10^{-12}$, giving $\lambda_4 = 0.254719$; the leading $3\times3$ block has eigenvalues $1.8227, 3.1773, 4.7453$ — the remaining eigenvalues.

---

## 9.6 Singular Value Decomposition

**Theorem 9.25 (SVD).** Every $A\in\mathbb R^{m\times n}$ factors as
$$
A = U\Sigma V^T,\qquad U\in\mathbb R^{m\times m},\ V\in\mathbb R^{n\times n}\text{ orthogonal},\quad\Sigma = \operatorname{diag}(s_1\ge s_2\ge\cdots\ge s_r>0,0,\dots)\in\mathbb R^{m\times n}.
$$
The **singular values** $s_i$ are the square roots of the nonzero eigenvalues of $A^TA$ (or $AA^T$); columns of $V$ (right singular vectors) are eigenvectors of $A^TA$, columns of $U$ of $AA^T$, and $A\mathbf v_i = s_i\mathbf u_i$.

**Consequences.** $\operatorname{rank}A = r$; $\|A\|_2 = s_1$; $K_2(A) = s_1/s_n$; $A = \sum_{i=1}^rs_i\mathbf u_i\mathbf v_i^T$; the best rank-$k$ approximation is $A_k = \sum_{i\le k}s_i\mathbf u_i\mathbf v_i^T$ with $\|A-A_k\|_2 = s_{k+1}$ (Eckart–Young); the pseudoinverse $A^+ = V\Sigma^+U^T$ gives the minimum-norm least-squares solution.

**Computation.** Do *not* form $A^TA$ (it squares $K$). Golub–Kahan: bidiagonalise $A$ with Householder reflections from both sides, then apply an implicit QR iteration to the bidiagonal matrix.

### Examples for §9.6

**Example 9.6.1 (SVD by hand).** $A = \begin{pmatrix}3&2&2\\2&3&-2\end{pmatrix}$. $AA^T = \begin{pmatrix}17&8\\8&17\end{pmatrix}$ has eigenvalues $25, 9$, so $s_1 = 5$, $s_2 = 3$; $\mathbf u_1 = \frac1{\sqrt2}(1,1)$, $\mathbf u_2 = \frac1{\sqrt2}(1,-1)$. Then $\mathbf v_i = A^T\mathbf u_i/s_i$: $\mathbf v_1 = \frac1{\sqrt2}(1,1,0)$, $\mathbf v_2 = \frac{1}{\sqrt{18}}(1,-1,4)$, and $\mathbf v_3 = \frac13(2,-2,-1)$ completes $V$. (Signs may differ from software output.)

**Example 9.6.2 (via $A^TA$).** $A = \begin{pmatrix}1&1\\0&1\\1&0\end{pmatrix}$: $A^TA = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ with eigenvalues $3, 1$, so $s = (\sqrt3, 1) = (1.7321, 1)$ — matching `np.linalg.svd`. Fine for teaching and for well-conditioned $A$.

**Example 9.6.3 (rank deficiency and the pseudoinverse).** $X = \begin{pmatrix}1&2&3\\1&2&3\\2&4&6\\1&0&1\end{pmatrix}$ has singular values $9.2279,\ 0.9195,\ 0$ — rank 2 (perfectly collinear columns: col 3 = col 1 + col 2). The normal equations are singular, but $X^+\mathbf y$ with $\mathbf y = (1,2,3,1)$ gives the minimum-norm least-squares solution $(0.5833, -0.1667, 0.4167)$ (= `np.linalg.lstsq`).

**Example 9.6.4 (low-rank structure in ratings).** A $5\times4$ user–item rating matrix with two "taste groups" has singular values $12.02,\ 8.46,\ 1.58,\ 1.22$. The rank-2 approximation $A_2 = s_1\mathbf u_1\mathbf v_1^T + s_2\mathbf u_2\mathbf v_2^T$ reproduces the pattern (e.g. row 1 ≈ $(5.00, 5.01, 0.48, 0.52)$ vs data $(5,5,0,1)$). Checks: $\|A - A_1\|_2 = s_2 = 8.456$ and $\|A - A_2\|_F = \sqrt{s_3^2+s_4^2} = 1.999$. This is the idea behind matrix-factorisation recommenders.

**Example 9.6.5 (geometry and conditioning).** $A = \begin{pmatrix}2&1\\1&1.2\end{pmatrix}$ maps the unit circle to an ellipse with semi-axes $s_1 = 2.6770$ (direction $\mathbf u_1$) and $s_2 = 0.5230$; $A\mathbf v_1 = s_1\mathbf u_1 = (-2.2168, -1.5008)$; $K_2(A) = s_1/s_2 = 5.119$.

---

## 9.7 Eigenvalue Problems in Data Science *(data-science extension)*

| Application | Matrix | Eigen-object |
|---|---|---|
| stationary distribution of a Markov chain | transition matrix $P$ | left eigenvector for $\lambda = 1$ (power method; rate $\lvert\lambda_2\rvert$) |
| PageRank | $G = dP + \frac{1-d}{n}\mathbf 1\mathbf 1^T$ | dominant eigenvector (rate $d$) |
| spectral clustering | graph Laplacian $L = D - W$ | Fiedler vector (2nd smallest) |
| PCA | covariance $S$ | top eigenvectors (Chapter 14) |
| HITS hubs/authorities | adjacency $A$ | top singular vectors |

### Examples for §9.7

**Example 9.7.1 (Markov chain — weather model).** States sunny/cloudy/rainy with transition matrix $P = \begin{pmatrix}0.7&0.2&0.1\\0.3&0.5&0.2\\0.2&0.3&0.5\end{pmatrix}$ (rows sum to 1). Starting from "sunny": $\boldsymbol\pi_1 = (0.7,0.2,0.1)$, $\boldsymbol\pi_2 = (0.57,0.27,0.16)$, $\boldsymbol\pi_{10} = (0.4636, 0.3170, 0.2194)$, converging to the stationary distribution $\boldsymbol\pi = (0.4634, 0.3171, 0.2195)$ (left eigenvector for $\lambda = 1$). The error decays like $|\lambda_2|^k = 0.462^k$.

**Example 9.7.2 (PageRank with a dangling page).** Five pages, page 5 has no out-links (treated as linking to every page). With $d = 0.85$: $\mathbf r = (0.350, 0.188, 0.365, 0.040, 0.056)$. Page 3, which receives links from three pages, ranks highest.

**Example 9.7.3 (spectral clustering).** Two dense groups $\{1,2,3,4\}$ and $\{5,6,7,8\}$ joined by a single edge. The Laplacian has eigenvalues $0,\ 0.2907,\ 2,\ 2.81,\dots$; the small second eigenvalue signals a weak cut. The **Fiedler vector** $(-0.43, -0.37, -0.37, -0.20, 0.20, 0.37, 0.37, 0.43)$ has the sign pattern $(-,-,-,-,+,+,+,+)$ — exactly the two communities.

**Example 9.7.4 (PCA preview).** 500 samples from a 2-D Gaussian: sample covariance $\begin{pmatrix}2.916&1.456\\1.456&0.961\end{pmatrix}$ with eigenvalues $3.692, 0.185$. The first principal direction $(0.882, 0.470)$ explains $95.2\%$ of the variance. (Chapter 14 develops PCA via the SVD.)

**Example 9.7.5 (HITS — hubs and authorities).** For a 4-page link graph with adjacency $A$, hub scores are the top left singular vector of $A$ and authority scores the top right singular vector: hubs $(0.28, 0.34, 0.17, 0.21)$, authorities $(0.10, 0.16, 0.46, 0.29)$. Page 3 is the best authority (most linked-to by good hubs); page 2 the best hub.

---

## Chapter summary

| Task | Method | Rate / cost |
|---|---|---|
| locate eigenvalues | Gershgorin | free |
| dominant eigenpair (sparse) | power / symmetric power | $\lvert\lambda_2/\lambda_1\rvert$ / its square |
| eigenvalue near $q$ | inverse power, RQI | $\frac{\lvert\lambda-q\rvert}{\lvert\lambda'-q\rvert}$ / cubic |
| all eigenvalues (dense) | Householder + shifted QR | $O(n^3)$, cubic local convergence |
| singular values | Golub–Kahan SVD | $O(mn^2)$ |

## Further reading

Burden & Faires Ch. 9 · Trefethen & Bau, Lectures 24–31 · Golub & Van Loan, Ch. 7–8 · von Luxburg, "A tutorial on spectral clustering", *Statistics and Computing* 17 (2007) · Langville & Meyer, *Google's PageRank and Beyond*.
