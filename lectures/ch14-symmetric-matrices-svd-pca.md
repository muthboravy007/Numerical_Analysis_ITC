# Chapter 14 — Symmetric Matrices, the SVD and Principal Component Analysis

> **Data-science chapter** extending Burden & Faires §9.1–9.2 (symmetric matrices, similarity) and §9.6 (SVD).
> **Code:** [`numlib/eigen.py`](../numlib/eigen.py), [`numlib/least_squares.py`](../numlib/least_squares.py) · **All numbers reproduced by** [`lectures/code/ch14.py`](code/ch14.py)
> **Worked problems:** [`examples/ch14`](../examples/ch14_examples.md) · **Homework:** [`exercises/ch14`](../exercises/ch14_exercises.md)

## Learning outcomes

1. Prove and use the spectral theorem; form matrix functions ($A^{1/2}$, $e^A$) from the eigen-decomposition; characterise eigenvalues by the Rayleigh quotient and the min–max principle.
2. Test positive (semi)definiteness in several ways; interpret quadratic forms geometrically; use Schur complements (conditional Gaussians) and Hessians (convexity).
3. Derive the SVD from the spectral theorem, identify the four fundamental subspaces, and relate singular values to norms, rank and conditioning.
4. Apply the Eckart–Young theorem to compression and denoising; use randomized SVD for large matrices.
5. Derive PCA as variance maximisation and as best low-rank reconstruction, compute it stably with the SVD, and apply it (standardisation, explained variance, whitening, PCA before regression/classification).
6. Use SVD-based methods in text mining (LSA), recommender systems, image analysis and multidimensional scaling.

---

## 14.1 Symmetric Matrices and the Spectral Theorem

**Theorem 14.1 (spectral theorem).** If $A\in\mathbb R^{n\times n}$ is symmetric, then (i) all eigenvalues are real; (ii) eigenvectors of distinct eigenvalues are orthogonal; (iii) $A = Q\Lambda Q^T = \sum_{i=1}^n\lambda_i\mathbf q_i\mathbf q_i^T$ with $Q$ orthogonal.

*Proof of (i)–(ii).* If $A\mathbf x = \lambda\mathbf x$ ($\mathbf x\in\mathbb C^n$), then $\bar{\mathbf x}^TA\mathbf x = \lambda\|\mathbf x\|^2$ and, taking conjugates and using $A = A^T$ real, $\bar{\mathbf x}^TA\mathbf x = \bar\lambda\|\mathbf x\|^2$; hence $\lambda = \bar\lambda$. If $A\mathbf x = \lambda\mathbf x$, $A\mathbf y = \mu\mathbf y$, $\lambda\ne\mu$: $\lambda\mathbf y^T\mathbf x = \mathbf y^TA\mathbf x = (A\mathbf y)^T\mathbf x = \mu\mathbf y^T\mathbf x$, so $\mathbf y^T\mathbf x = 0$. (iii) follows by induction using Schur's theorem (§9.2): a triangular matrix that is also symmetric is diagonal. ∎

**Matrix functions.** For symmetric $A = Q\Lambda Q^T$ and a function $f$: $f(A) = Q\,f(\Lambda)\,Q^T$. E.g. $A^{1/2}$ (for PSD $A$), $A^{-1}$, $e^{A}$, $\log A$.

**Rayleigh quotient and min–max.** $R(\mathbf x) = \frac{\mathbf x^TA\mathbf x}{\mathbf x^T\mathbf x}$ satisfies $\lambda_{\min}\le R(\mathbf x)\le\lambda_{\max}$, with equality at the corresponding eigenvectors. More generally (Courant–Fischer), with $\lambda_1\ge\cdots\ge\lambda_n$:
$$
\lambda_k = \max_{\dim S = k}\ \min_{\mathbf x\in S\setminus\{0\}}R(\mathbf x),\qquad\lambda_2 = \max_{\mathbf x\perp\mathbf q_1}R(\mathbf x).
$$
**Weyl's inequality.** For symmetric $A, E$: $|\lambda_k(A+E) - \lambda_k(A)|\le\|E\|_2$ — symmetric eigenvalues are perfectly conditioned.

### Examples for §14.1

**Example 14.1.1.** $A = \begin{pmatrix}6&2&1\\2&3&1\\1&1&1\end{pmatrix}$ has eigenvalues $0.5789, 2.1331, 7.2880$ (all real) and an orthonormal eigenvector matrix $Q$ with $Q^TQ = I$ to machine precision.

**Example 14.1.2 (spectral decomposition).** $A = 0.5789\,\mathbf q_1\mathbf q_1^T + 2.1331\,\mathbf q_2\mathbf q_2^T + 7.2880\,\mathbf q_3\mathbf q_3^T$ (reconstruction error $5\times10^{-15}$). Each term is a rank-one projection scaled by its eigenvalue; keeping only the largest gives the best rank-one approximation of $A$.

**Example 14.1.3 (matrix functions).** $A^{1/2} = Q\Lambda^{1/2}Q^T = \begin{pmatrix}2.3894&0.4758&0.2535\\0.4758&1.6288&0.3474\\0.2535&0.3474&0.9028\end{pmatrix}$, and $\|(A^{1/2})^2 - A\| = 10^{-14}$. Likewise $Qe^{\Lambda}Q^T$ agrees with `scipy.linalg.expm(A)` to $7\times10^{-10}$ (relative to $\|e^A\|\approx1500$). Matrix square roots are used to whiten data and to sample Gaussians; $e^{-tL}$ is the graph heat kernel (Chapter 12).

**Example 14.1.4 (Rayleigh quotient bounds).** Over $10^5$ random vectors, $R(\mathbf x)$ ranges over $[0.57897, 7.28790]$ — inside $[\lambda_{\min},\lambda_{\max}] = [0.57893, 7.28799]$, approaching the ends.

**Example 14.1.5 (min–max).** Maximising $R$ over vectors orthogonal to $\mathbf q_3$ (the top eigenvector) gives $2.13307 = \lambda_2$. This deflation argument is exactly how the second principal component is defined (§14.5).

**Example 14.1.6 (Weyl).** Perturbing $A$ by a symmetric $E$ with $\|E\|_2 = 0.0175$ moves the eigenvalues by $-0.0051, 0.0118, 0.0023$ — each less than $\|E\|_2$.

---

## 14.2 Positive Definite Matrices and Quadratic Forms

For symmetric $A$ the following are equivalent (**positive definite**, PD): (a) $\mathbf x^TA\mathbf x>0$ for all $\mathbf x\ne\mathbf 0$; (b) all eigenvalues $>0$; (c) all leading principal minors $>0$ (Sylvester); (d) all pivots of Gaussian elimination $>0$; (e) $A = LL^T$ with $L$ nonsingular (Cholesky); (f) $A = B^TB$ with $B$ of full column rank. Replacing $>$ by $\ge$ gives **positive semidefinite** (PSD) (except (c), which then needs all principal minors).

**Where PSD matrices come from.** Gram matrices $X^TX$, covariance matrices, kernel matrices $K_{ij} = k(\mathbf x_i,\mathbf x_j)$, graph Laplacians, Hessians of convex functions, Fisher information matrices.

**Geometry.** $\{\mathbf x: \mathbf x^TA\mathbf x = 1\}$ for PD $A$ is an ellipsoid with axes along the eigenvectors and semi-axis lengths $1/\sqrt{\lambda_i}$. The Hessian's definiteness classifies critical points (min/max/saddle, Chapter 15).

**Schur complement.** For $A = \begin{pmatrix}A_{11}&A_{12}\\A_{21}&A_{22}\end{pmatrix}$ with $A_{22}$ PD, $A$ is PD iff $S = A_{11} - A_{12}A_{22}^{-1}A_{21}$ is PD. For a Gaussian vector, $S$ is the **conditional covariance** of the first block given the second:
$$
\mathbf x_1\mid\mathbf x_2\sim N\bigl(\boldsymbol\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf x_2-\boldsymbol\mu_2),\ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\bigr).
$$

### Examples for §14.2

**Example 14.2.1 (five tests agree).** $M = \operatorname{tridiag}(-1,2,-1)$ ($3\times3$): eigenvalues $0.586, 2, 3.414>0$; leading minors $2, 3, 4>0$; Cholesky succeeds with $L = \begin{pmatrix}1.4142&&\\-0.7071&1.2247&\\0&-0.8165&1.1547\end{pmatrix}$ (pivots $\ell_{ii}^2 = 2, 1.5, 1.333$). PD.

**Example 14.2.2 (Gram matrices are PSD).** For $X\in\mathbb R^{50\times4}$ random, $X^TX$ has eigenvalues $27.5, 51.9, 64.9, 101.3$ (PD: full column rank). For $Z\in\mathbb R^{3\times5}$, $Z^TZ$ ($5\times5$) has eigenvalues $0, 0, 0.72, 5.54, 8.67$ — PSD with rank 3: more features than observations always gives a singular Gram/covariance matrix.

**Example 14.2.3 (the ellipse of a quadratic form).** $K = \begin{pmatrix}5&2\\2&2\end{pmatrix}$ has eigenvalues $1$ and $6$ with eigenvectors $(0.447,-0.894)$ and $(0.894, 0.447)$. The curve $\mathbf x^TK\mathbf x = 1$ is an ellipse with semi-axes $1/\sqrt1 = 1$ and $1/\sqrt6 = 0.408$ along these directions — the same picture as the contours of a Gaussian density or of a quadratic loss.

**Example 14.2.4 (Hessians classify critical points).** $f(x,y) = x^4 - 2x^2 + y^2$ has critical points $(0,0)$ and $(\pm1,0)$. Hessian eigenvalues: at $(0,0)$: $-4, 2$ (indefinite ⇒ saddle); at $(\pm1,0)$: $2, 8$ (PD ⇒ strict local minima).

**Example 14.2.5 (conditioning a Gaussian = Schur complement).** $\Sigma = \begin{pmatrix}4&1.2&0.8\\1.2&1&0.3\\0.8&0.3&2\end{pmatrix}$, $\boldsymbol\mu = (1,0,2)$. Given $(x_2,x_3) = (0.5, 2.5)$: $E[x_1\mid\cdot] = 1.681$ and $\operatorname{Var}[x_1\mid\cdot] = 4 - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} = 2.459$ (down from the marginal $4$): observing correlated variables reduces uncertainty. This formula is Gaussian-process prediction (§6.7) and the Kalman filter update.

**Example 14.2.6 (numerically semidefinite kernels).** The RBF kernel matrix of 30 points on $[0,1]$ ($\ell = 0.3$) is PD in theory, but its computed smallest eigenvalue is $-6\times10^{-16}$ and its largest $17.3$: numerically singular, so Cholesky may fail. Adding "jitter" $10^{-6}I$ gives $\kappa = 1.7\times10^7$ and a safe factorisation — standard practice in Gaussian-process software.

---

## 14.3 The Singular Value Decomposition

**Theorem 14.2 (SVD).** Every $A\in\mathbb R^{m\times n}$ of rank $r$ can be written $A = U\Sigma V^T = \sum_{i=1}^r\sigma_i\mathbf u_i\mathbf v_i^T$ with $\sigma_1\ge\cdots\ge\sigma_r>0$.

*Proof.* $A^TA$ is symmetric PSD; let $A^TA = V\Lambda V^T$ with $\lambda_1\ge\dots\ge\lambda_r>0 = \lambda_{r+1} = \cdots$. Put $\sigma_i = \sqrt{\lambda_i}$ and $\mathbf u_i = A\mathbf v_i/\sigma_i$ ($i\le r$). Then $\mathbf u_i^T\mathbf u_j = \mathbf v_i^TA^TA\mathbf v_j/(\sigma_i\sigma_j) = \delta_{ij}$; extend $\{\mathbf u_i\}$ to an orthonormal basis. For $i>r$, $\|A\mathbf v_i\|^2 = \lambda_i = 0$. Hence $AV = U\Sigma$. ∎

**The four fundamental subspaces.** $\operatorname{col}(A) = \operatorname{span}\{\mathbf u_1..\mathbf u_r\}$, $\operatorname{null}(A^T) = \operatorname{span}\{\mathbf u_{r+1}..\mathbf u_m\}$, $\operatorname{row}(A) = \operatorname{span}\{\mathbf v_1..\mathbf v_r\}$, $\operatorname{null}(A) = \operatorname{span}\{\mathbf v_{r+1}..\mathbf v_n\}$.

**Norms and conditioning.** $\|A\|_2 = \sigma_1$, $\|A\|_F = \sqrt{\sum\sigma_i^2}$, nuclear norm $\|A\|_* = \sum\sigma_i$, $\kappa_2(A) = \sigma_1/\sigma_n$. For symmetric $A$, $\sigma_i = |\lambda_i|$. **Perturbation:** $|\sigma_i(A+E)-\sigma_i(A)|\le\|E\|_2$.

**Computation.** Never via $A^TA$ in floating point (it squares $\kappa$); LAPACK uses Householder bidiagonalisation + implicit QR (Golub–Kahan) or divide-and-conquer.

### Examples for §14.3

**Example 14.3.1 (a small SVD).** $B = \begin{pmatrix}1&1\\1&-1\\2&0\end{pmatrix}$: $B^TB = \operatorname{diag}(6,2)$, so $\sigma = (\sqrt6,\sqrt2) = (2.449, 1.414)$, $V = I$ (up to sign), $\mathbf u_1 = B\mathbf e_1/\sqrt6 = (1,1,2)/\sqrt6$, $\mathbf u_2 = (1,-1,0)/\sqrt2$, and $\mathbf u_3 = (1,1,-1)/\sqrt3$ spans $\operatorname{null}(B^T)$.

**Example 14.3.2 (subspaces from the SVD).** $C = \begin{pmatrix}1&2&3\\2&4&6\\1&0&1\end{pmatrix}$: $\sigma = (8.435, 0.918, 0)$, rank 2. $\mathbf v_3 = (-1,-1,1)/\sqrt3$ spans $\operatorname{null}(C)$ ($C\mathbf v_3 = \mathbf 0$: column 3 = column 1 + column 2); $\mathbf u_3 = (-2,1,0)/\sqrt5$ spans $\operatorname{null}(C^T)$ (row 2 = 2 × row 1).

**Example 14.3.3 (norms).** $D = \begin{pmatrix}3&1\\1&3\end{pmatrix}$: $\sigma = (4, 2)$; $\|D\|_2 = 4$, $\|D\|_F = \sqrt{20} = 4.472$, $\|D\|_* = 6$, $\kappa_2 = 2$.

**Example 14.3.4 (eigenvalues vs singular values).** The symmetric but indefinite $\begin{pmatrix}2&3\\3&-2\end{pmatrix}$ has eigenvalues $\pm\sqrt{13} = \pm3.606$ and singular values $3.606, 3.606$ — for symmetric matrices $\sigma_i = |\lambda_i|$, and the SVD "forgets" the sign.

**Example 14.3.5 (stability of singular values).** Adding noise $E$ with $\|E\|_2 = 0.0024$ to $C$ changes its singular values by $-0.0003, 0.0004, 0.0016$ — all bounded by $\|E\|_2$. Note the zero singular value became $0.0016$: numerical rank must be decided with a tolerance.

**Example 14.3.6 (why not $A^TA$).** For the $20\times8$ Vandermonde matrix on $[0,1]$: smallest singular value via `svd` $5.6766554\times10^{-5}$; via $\sqrt{\lambda_{\min}(A^TA)}$: $5.6766568\times10^{-5}$ — only 6 correct digits, because $\kappa(A^TA) = \kappa(A)^2\approx10^{10}$.

---

## 14.4 Low-Rank Approximation

**Theorem 14.3 (Eckart–Young–Mirsky).** For $k<r$, $A_k = \sum_{i=1}^k\sigma_i\mathbf u_i\mathbf v_i^T$ satisfies
$$
\min_{\operatorname{rank}B\le k}\|A-B\|_2 = \|A-A_k\|_2 = \sigma_{k+1},\qquad\min_{\operatorname{rank}B\le k}\|A-B\|_F = \|A-A_k\|_F = \Bigl(\sum_{i>k}\sigma_i^2\Bigr)^{1/2}.
$$
**Storage:** $k(m+n+1)$ numbers instead of $mn$.

**Choosing $k$:** a scree plot (look for an elbow); an energy threshold $\sum_{i\le k}\sigma_i^2/\sum\sigma_i^2\ge0.9$; for data "signal + i.i.d. noise of level $\sigma$", singular values of pure noise concentrate below $\approx\sigma(\sqrt m+\sqrt n)$ (Marchenko–Pastur edge), so keep singular values above it.

**Randomized SVD** (Halko–Martinsson–Tropp). For large $A$ and small $k$: $Y = A\Omega$ with Gaussian $\Omega\in\mathbb R^{n\times(k+p)}$ (oversampling $p\approx10$), optionally a few power iterations $Y\leftarrow A(A^TY)$, $Q = \operatorname{qr}(Y)$, then an SVD of the small $B = Q^TA$. Cost $O(mnk)$ instead of $O(mn\min(m,n))$.

### Examples for §14.4

**Example 14.4.1 (image compression).** A $427\times640$ grayscale photograph has singular values $327.2, 60.4, 38.3, 22.6, 19.3,\ldots$

| $k$ | relative error $\lVert A-A_k\rVert_F/\lVert A\rVert_F$ | storage fraction |
|---|---|---|
| 5 | 0.184 | 0.020 |
| 20 | 0.136 | 0.078 |
| 50 | 0.103 | 0.195 |
| 100 | 0.074 | 0.391 |

In every case the error equals $\sqrt{\sum_{i>k}\sigma_i^2}/\|A\|_F$ exactly (Eckart–Young).

**Example 14.4.2 (energy).** For the same image 90% of the energy is in $k = 1$ (the mean brightness!), 99% needs $k = 54$ and 99.9% needs $k = 216$. Energy thresholds on uncentred data can mislead — this is why PCA centres first.

**Example 14.4.3 (denoising a low-rank matrix).** $L$ = rank-3 $200\times100$ signal, noise $\sigma = 0.5$. Singular values of $L+N$: $164.0, 144.0, 126.9$, then $12.1, 11.9, 11.4,\ldots$ — a clear gap at the noise edge $0.5(\sqrt{200}+\sqrt{100}) = 12.07$. Truncating to rank 3 reduces the relative error from $0.283$ (noisy data) to $0.059$.

**Example 14.4.4 (randomized SVD).** $A\in\mathbb R^{2000\times1500}$ with decaying spectrum: the top 10 singular values from randomized SVD ($p = 10$, one power iteration) have relative error $\le6\times10^{-7}$, and $\|A - A_{10}\|_2 = 181.7656$ equals the optimal $\sigma_{11} = 181.7656$; time $0.025$ s vs $0.96$ s for the full SVD.

**Example 14.4.5 (latent factors in ratings).** The $5\times4$ ratings matrix of §9.6 has singular values $12.02, 8.46, 1.58, 1.22$: two factors explain $98.2\%$ of the energy — two "taste" dimensions.

---

## 14.5 Principal Component Analysis

Let $X\in\mathbb R^{n\times p}$ be the data (rows = observations), $X_c$ its column-centred version and $S = \frac{1}{n-1}X_c^TX_c$ the sample covariance.

**Two derivations, one answer.**
1. *Maximum variance:* $\mathbf w_1 = \arg\max_{\|\mathbf w\| = 1}\operatorname{Var}(X_c\mathbf w) = \arg\max\mathbf w^TS\mathbf w$ — by §14.1 the top eigenvector of $S$, with variance $\lambda_1$; $\mathbf w_k$ maximises the variance subject to $\mathbf w_k\perp\mathbf w_1,\dots,\mathbf w_{k-1}$ (min–max).
2. *Best reconstruction:* the rank-$k$ projection $X_cW_kW_k^T$ minimising $\|X_c - X_cWW^T\|_F$ over orthonormal $W\in\mathbb R^{p\times k}$ is given by the top-$k$ eigenvectors (Eckart–Young applied to $X_c$), with error $(n-1)\sum_{j>k}\lambda_j$.

**Computation via the SVD** of $X_c = U\Sigma V^T$: principal directions (**loadings**) $V$; variances $\lambda_j = \sigma_j^2/(n-1)$; **scores** $T = X_cV = U\Sigma$. (No need to form $S$; `sklearn.decomposition.PCA` does this.)

**Explained variance ratio** $\lambda_j/\sum\lambda$; choose $k$ by cumulative proportion, scree plot or downstream CV.
**Scale matters:** PCA on the covariance is not scale-invariant. If units differ, use the **correlation** matrix (standardise columns).
**Whitening:** $Z = X_cV\Lambda^{-1/2}$ has identity covariance.
**Uses:** visualisation (2-D scores), compression, noise reduction, decorrelating features, principal component regression (PCR) against multicollinearity, anomaly detection (reconstruction error).

### Examples for §14.5

Fisher's **Iris** data: 150 flowers, 4 measurements (cm): sepal length, sepal width, petal length, petal width.

**Example 14.5.1 (PCA by eigen-decomposition).** Means $(5.843, 3.057, 3.758, 1.199)$. Covariance
$$
S = \begin{pmatrix}0.686&-0.042&1.274&0.516\\-0.042&0.190&-0.330&-0.122\\1.274&-0.330&3.116&1.296\\0.516&-0.122&1.296&0.581\end{pmatrix}
$$
has eigenvalues $4.228, 0.243, 0.078, 0.024$; explained variance $92.5\%, 5.3\%, 1.7\%, 0.5\%$. PC1 $= \pm(0.361, -0.085, 0.857, 0.358)$: dominated by petal length — a "size" axis.

**Example 14.5.2 (the same via SVD).** $\sigma_j^2/(n-1)$ reproduces $4.228, 0.243, 0.078, 0.024$ and $|V_{:,1}|$ equals the eigenvector above. Scores of the first three flowers on (PC1, PC2): $(-2.684, -0.319)$, $(-2.714, 0.177)$, $(-2.889, 0.145)$ (signs are arbitrary).

**Example 14.5.3 (scaling matters).** On standardised data (correlation matrix) the eigenvalues are $2.919, 0.914, 0.147, 0.021$ ($73.0\%, 22.9\%, 3.7\%, 0.5\%$) — a different, more balanced picture. If sepal length were recorded in millimetres instead of centimetres, covariance PCA would give PC1 $98.5\%$ — driven by units, not by structure.

**Example 14.5.4 (PCA reveals clusters).** Mean scores on (PC1, PC2) by species (with the sign convention of `eigh`): setosa $(2.64, 0.19)$, versicolor $(-0.53, -0.25)$, virginica $(-2.11, 0.06)$ — PC1 alone separates the species almost perfectly, although PCA never saw the labels.

**Example 14.5.5 (reconstruction error = discarded variance).**

| $k$ | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| $\lVert X_c - X_cW_kW_k^T\rVert_F^2/(n-1)$ | 0.3447 | 0.1020 | 0.0238 | 0 |
| $\sum_{j>k}\lambda_j$ | 0.3447 | 0.1020 | 0.0238 | 0 |

**Example 14.5.6 (whitening).** $Z = X_cV\Lambda^{-1/2}$ has covariance $I_4$ exactly — decorrelated, unit-variance features (used before ICA, some clustering methods and neural-network training).

**Example 14.5.7 (PCA as preprocessing).** Handwritten digits ($8\times8$ images, 64 features, 1797 samples): 13, 21 and 29 components explain 80%, 90% and 95% of the variance. Logistic regression (5-fold CV accuracy) on PCA scores: 2 components $0.587$; 10: $0.889$; 20: $0.895$; all 64 raw features: $0.913$ — a 3× smaller representation keeps almost all the predictive information.

---

## 14.6 SVD in Data-Science Applications

**Latent semantic analysis (LSA).** A term–document matrix $A$ ($t\times d$, counts or tf–idf) is replaced by $A_k$; documents are represented by the rows of $V_k\Sigma_k$ and compared by cosine similarity. Synonyms and co-occurring terms are merged into "concepts". A query vector $\mathbf q$ is folded in as $\hat{\mathbf q} = \Sigma_k^{-1}U_k^T\mathbf q$.

**Matrix completion / recommender systems.** With missing ratings, fit $R\approx PQ^T$ ($P\in\mathbb R^{u\times k}$, $Q\in\mathbb R^{i\times k}$) on the observed entries only, with ridge penalties; **alternating least squares (ALS)** fixes $Q$ and solves a small ridge problem for each row of $P$, then vice versa.

**Subspace models (eigenfaces).** PCA of a class of images gives a basis; the distance of a new image from the class subspace (reconstruction error) is a classification/anomaly score.

**Classical multidimensional scaling (MDS).** Given only pairwise distances $D$, double centring $B = -\frac12JD^{(2)}J$ ($J = I-\frac1n\mathbf 1\mathbf 1^T$, $D^{(2)}$ = squared distances) gives the Gram matrix of centred coordinates; its top eigenvectors scaled by $\sqrt{\lambda}$ recover the configuration (up to rotation).

**Total least squares.** When both variables are noisy, the line minimising *perpendicular* distances is PC1 of the centred data — the smallest singular vector of $[\mathbf x\ \mathbf y]$ gives the normal direction.

### Examples for §14.6

**Example 14.6.1 (LSA).** 8 terms × 6 documents: documents 1–3 are about data/models/learning, 4–6 about football. $\sigma = 6.16, 5.75, 1.91, 1.63$: two strong concepts. In the 2-D LSA space documents 1–3 have coordinates with positive second component and 4–6 negative. $\cos(d_1,d_3)$ rises from $0.937$ (raw counts) to $0.9998$; $\cos(d_1,d_4)$ stays $\approx0$ ($-0.018$). The query "matrix" — a word appearing in only two documents — has cosine $0.998$–$0.999$ with **all three** data documents and $\le0$ with the football ones: LSA retrieves by concept, not by exact word match.

**Example 14.6.2 (ALS matrix completion).** A $5\times4$ ratings matrix with 5 missing entries, rank $k = 2$, $\lambda = 0.1$: after 200 ALS sweeps the training RMSE is $0.055$ and the predicted missing ratings are $1.00, 3.13, 5.89, 1.05, 1.81$ — consistent with each user's taste group (e.g. user 3, who likes football-type items, gets $5.89$ for item 3).

**Example 14.6.3 ("eigen-threes").** PCA of all handwritten 3s (10 components): a new 3 is reconstructed with error $10.7$, an 8 with error $25.5$ — the subspace model recognises its own class.

**Example 14.6.4 (classical MDS).** From the $5\times5$ distance matrix of points $(0,0), (3,0), (3,4), (0,4), (1.5,2)$ alone, $B = -\frac12JD^{(2)}J$ has eigenvalues $16, 9, 0, 0, 0$ (rank 2 ⇒ exactly planar), and the 2-D MDS coordinates reproduce all distances to $10^{-15}$ (rotated/reflected).

**Example 14.6.5 (total least squares vs OLS).** 500 points with $x_b = 0.8x_a + 0.6\cdot$noise (both standardised-ish): OLS slope of $x_b$ on $x_a$: $0.811$; inverse of the OLS slope of $x_a$ on $x_b$: $1.228$; PC1 (TLS) slope $0.998$ — between the two regressions, and symmetric in the roles of the variables.

---

## Chapter summary

| Concept | Key fact | Computation |
|---|---|---|
| symmetric $A$ | $A = Q\Lambda Q^T$, real $\lambda$, orthogonal $Q$ | `eigh` (tridiagonal QR / divide & conquer) |
| PD test | Cholesky succeeds | $n^3/6$ |
| any $A$ | $A = U\Sigma V^T$; $\sigma_i^2 = \lambda_i(A^TA)$ | `svd` (Golub–Kahan) — not via $A^TA$ |
| best rank-$k$ | $A_k$, error $\sigma_{k+1}$ | truncated / randomized SVD |
| PCA | loadings $V$, scores $U\Sigma$, variances $\sigma^2/(n-1)$ | SVD of centred data |

## Further reading

Strang, *Linear Algebra and Learning from Data* (2019) · Jolliffe & Cadima, "Principal component analysis: a review and recent developments", *Phil. Trans. R. Soc. A* (2016) · Halko, Martinsson & Tropp, "Finding structure with randomness", *SIAM Review* 53 (2011) · Koren, Bell & Volinsky, "Matrix factorization techniques for recommender systems", *IEEE Computer* (2009) · Deerwester et al., "Indexing by latent semantic analysis", *JASIS* (1990).
