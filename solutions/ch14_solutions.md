# Chapter 14 — Solutions

> Exercises: [`exercises/ch14_exercises.md`](../exercises/ch14_exercises.md) · Numbers from [`solutions/code/ch14_solutions.py`](code/ch14_solutions.py).

## A. Theory and proofs

**A1.** $A$ has a real eigenvalue $\lambda_1$ (Chapter 9, A1), so it has a real unit eigenvector $\mathbf q_1$. Let $Q = [\mathbf q_1\ Q_2]$ be orthogonal. Then $Q^TAQ = \begin{pmatrix}\lambda_1&\mathbf q_1^TAQ_2\\Q_2^TA\mathbf q_1&Q_2^TAQ_2\end{pmatrix}$ with $Q_2^TA\mathbf q_1 = \lambda_1Q_2^T\mathbf q_1 = \mathbf 0$. By symmetry the top-right block also vanishes. $Q_2^TAQ_2$ is symmetric of order $n-1$, so by induction it equals $\tilde Q\tilde\Lambda\tilde Q^T$. Combining the two orthogonal matrices gives $A = Q\Lambda Q^T$. Expanding the product column by column gives $\sum\lambda_i\mathbf q_i\mathbf q_i^T$.

**A2.** Write $\mathbf x = Q\mathbf y$. Then $R(\mathbf x) = \frac{\sum\lambda_iy_i^2}{\sum y_i^2}$, a weighted average of the $\lambda_i$. It lies in $[\lambda_n,\lambda_1]$, with the extremes attained at $\mathbf q_n$ and $\mathbf q_1$. For the min-max: with $V = \operatorname{span}(\mathbf q_1..\mathbf q_k)$ the minimum of $R$ over $V$ is $\lambda_k$. Any $k$-dimensional $V$ meets $\operatorname{span}(\mathbf q_k..\mathbf q_n)$, which has dimension $n-k+1$, and there $R\le\lambda_k$. For PCA, the variance of the projection onto a unit $\mathbf w$ is $\mathbf w^TS\mathbf w = R_S(\mathbf w)$, which is maximised by the top eigenvector of $S$.

**A3.** (i)⇒(ii): $\lambda = \mathbf q^TA\mathbf q>0$. (ii)⇒(iii): $R = \Lambda^{1/2}Q^T$. (iii)⇒(i): $\mathbf x^TR^TR\mathbf x = \|R\mathbf x\|^2>0$ because $R$ is nonsingular. (iv)⇒(i) by induction: $a_{11}>0$, so write $A = \begin{pmatrix}1&0\\\mathbf l&I\end{pmatrix}\begin{pmatrix}a_{11}&0\\0&S\end{pmatrix}\begin{pmatrix}1&\mathbf l^T\\0&I\end{pmatrix}$ with Schur complement $S$. The leading minors of $A$ equal $a_{11}$ times those of $S$, so $S$ satisfies (iv) and is PD by induction. $A$ is congruent to $\operatorname{diag}(a_{11},S)$, so $A$ is PD. (i)⇒(iv): leading principal submatrices of a PD matrix are PD, so their determinants, being products of positive eigenvalues, are positive.

**A4.** (a) Let $B$ have rank $k$. By Weyl's inequality for singular values, $\sigma_{i+k}(A)\le\sigma_i(A-B)+\sigma_{k+1}(B) = \sigma_i(A-B)$. So $\|A-B\|_F^2 = \sum_i\sigma_i(A-B)^2\ge\sum_{i>k}\sigma_i(A)^2$, with equality for the truncated SVD. (b) Each of $AA^+A = A$, $A^+AA^+ = A^+$, $(AA^+)^T = AA^+$ and $(A^+A)^T = A^+A$ reduces to the same identity for $\Sigma$ and $\Sigma^+$. For $\min\|A\mathbf x-\mathbf b\|$, write $\mathbf x = V\mathbf y$. The residual is $\|\Sigma\mathbf y-U^T\mathbf b\|$, which fixes $y_i = \mathbf u_i^T\mathbf b/\sigma_i$ for $\sigma_i>0$ and leaves the other $y_i$ free. The minimum norm sets them to zero, and this gives $\mathbf x = A^+\mathbf b$.

**A5.** (a) $S = \frac{X_c^TX_c}{n-1} = \frac{V\Sigma U^TU\Sigma V^T}{n-1}$. (b) $X_cV = U\Sigma V^TV = U\Sigma$. (c) For orthonormal $W\in\mathbb R^{p\times k}$, $\|X_c-X_cWW^T\|_F^2 = \|X_c\|_F^2-\|X_cW\|_F^2$ by Pythagoras, because $WW^T$ is an orthogonal projector. So minimising the reconstruction error is the same as maximising $\|X_cW\|_F^2 = (n-1)\operatorname{tr}(W^TSW)$, which is maximised by $W = V_k$ (Ky Fan). The minimum error is $\sum_{i>k}\sigma_i^2$.

**A6.** (a) By Courant–Fischer, $\lambda_i(A+E) = \max_V\min_{\mathbf x\in V}[R_A(\mathbf x)+R_E(\mathbf x)]\le\lambda_i(A)+\lambda_{\max}(E)$, and similarly from below, so $|\Delta\lambda_i|\le\|E\|_2$. For singular values, apply this to the symmetric matrices $\begin{pmatrix}0&A\\A^T&0\end{pmatrix}$, whose eigenvalues are $\pm\sigma_i$. (b) Davis–Kahan: if $\mathbf v$ and $\tilde{\mathbf v}$ are the eigenvectors of $\lambda_i$ for $A$ and $A+E$, then $\sin\angle(\mathbf v,\tilde{\mathbf v})\le\frac{\|E\|_2}{\text{gap}}$, where the gap is the distance from $\lambda_i$ to the rest of the spectrum. Eigenvalues are always well conditioned. Eigenvectors are well conditioned only when the gap is large. PCA loadings with nearly equal variances are therefore arbitrary, and only the subspace they span is stable.

## B. Hand computation

**B1.** The eigenvalues are $6$, with eigenvector $\frac1{\sqrt3}(1,1,1)$, and $3$ (twice), with eigenspace $\mathbf 1^\perp$, e.g. $\frac1{\sqrt2}(1,-1,0)$ and $\frac1{\sqrt6}(1,1,-2)$. The projectors are $P_1 = \frac13\mathbf 1\mathbf 1^T$ and $P_2 = I-P_1$, and $S = 6P_1+3P_2$.
- $S^{-1} = \frac16P_1+\frac13P_2$, with diagonal $0.27778 = \frac5{18}$ and off-diagonal $-0.05556 = -\frac1{18}$.
- $S^{1/2} = \sqrt6P_1+\sqrt3P_2$, with diagonal $1.971197$ and off-diagonal $0.239146$. This matches `sqrtm`.

**B2.** $\begin{pmatrix}3&1\\1&3\end{pmatrix}$ has eigenvalues $2$ and $4$, with axes along $(1,-1)/\sqrt2$ and $(1,1)/\sqrt2$. The curve is an ellipse with semi-axes $1/\sqrt2$ along $(1,-1)$ and $1/2$ along $(1,1)$. $x^2+4xy+y^2$ has eigenvalues $-1$ and $3$, so it is indefinite (a hyperbola). For $A$ the leading minors are $4, 16, 64>0$, so $A$ is PD. Its eigenvalues are $1.388, 6.341, 7.271$, and $L = \begin{pmatrix}2&0&0\\1&2&0\\-1&1&2\end{pmatrix}$ (note $\det = 2^6 = 64$).

**B3.** $A^TA = \begin{pmatrix}25&20\\20&25\end{pmatrix}$ has eigenvalues $45$ and $5$, so $\sigma = 3\sqrt5 = 6.7082$ and $\sqrt5 = 2.2361$. The right singular vectors are $\mathbf v_1 = \frac{(1,1)}{\sqrt2}$ and $\mathbf v_2 = \frac{(-1,1)}{\sqrt2}$. Then $\mathbf u_1 = A\mathbf v_1/\sigma_1 = \frac{(1,3)}{\sqrt{10}}$ and $\mathbf u_2 = A\mathbf v_2/\sigma_2 = \frac{(-3,1)}{\sqrt{10}}$. The condition number is $\kappa_2 = 3$, and since $A$ is invertible, $A^+ = A^{-1} = \begin{pmatrix}1/3&0\\-4/15&1/5\end{pmatrix}$.

**B4.** $\mathbf a = (1,2,2)$ and $\mathbf a^+ = \mathbf a^T/\|\mathbf a\|^2 = \mathbf a^T/9$, so $\mathbf x^+ = \mathbf a^T = (1,2,2)$ with norm $3$. Every other solution adds a component orthogonal to $\mathbf a$ and is longer; e.g. $(9,0,0)$ has norm 9.

**B5.** The mean is $(5,5)$ and $S = \begin{pmatrix}6.5&6.5\\6.5&7.5\end{pmatrix}$. The eigenvalues are $13.5192$ and $0.4808$, explaining 96.6% and 3.4% of the variance. The axes are PC1 $= (0.6795, 0.7337)$ and PC2 $= (-0.7337, 0.6795)$. The PC1 scores are $(-4.973, -2.093, 0, 3.560, 3.506)$. The rank-1 reconstruction is $(1.621,1.351)$, $(3.578,3.465)$, $(5,5)$, $(7.419,7.612)$, $(7.382,7.572)$. Its SSE is $1.92319 = 4\times0.48080$.

**B6.** $\Sigma^{-1} = \begin{pmatrix}0.375&-0.25\\-0.25&0.5\end{pmatrix}$. Both points have Euclidean distance $2.828$. Their Mahalanobis distances are $1.225$ for $(2,2)$ and $2.345$ for $(2,-2)$. $(2,-2)$ is much more unusual because it goes against the positive correlation. The whitening matrices are $L^{-1} = \begin{pmatrix}0.5&0\\-0.3536&0.7071\end{pmatrix}$ and $\Sigma^{-1/2} = \begin{pmatrix}0.5792&-0.1988\\-0.1988&0.6786\end{pmatrix}$, and both give identity covariance. ZCA (symmetric) whitening stays closest to the original coordinates. Cholesky whitening depends on the variable order.

**B7.** $R(1,0,0) = 4$, $R(1,1,1) = 6 = \lambda_{\max}$ and $R(1,-1,0) = 3 = \lambda_{\min}$. All lie in $[3,6]$, and the two eigenvectors attain the ends.

## C. Programming (key results)

**C1.** The full SVD takes 0.20 s, and the optimum is $\sigma_{21} = 0.0476$.

| $p$ | $q$ | error | ratio to optimum | time | max relative error in the top 20 $\sigma$ |
|---|---|---|---|---|---|
| 0 | 0 | 0.1153 | 2.42 | 5.5 ms | 0.52 |
| 10 | 0 | 0.0871 | 1.83 | 6.8 ms | 0.26 |
| 10 | 1 | 0.0480 | 1.009 | 12 ms | 0.021 |
| 10 | 2 | 0.0476 | 1.000 | 25 ms | 0.004 |

With the slowly decaying spectrum $1/i$, oversampling alone is not enough. Power iterations replace $A$ by $(AA^T)^qA$, whose singular values $\sigma_i^{2q+1}$ decay much faster. The method is 8–40× faster than the full SVD.

**C2.**

| gap $\delta$ | eigenvalue change | eigenvector rotation | Davis–Kahan bound $\|E\|/\text{gap}$ |
|---|---|---|---|
| $10^{-1}$ | $10^{-7}$ | $0.057°$ | $10^{-3}$ rad |
| $10^{-3}$ | $10^{-5}$ | $5.7°$ | $0.1$ rad |
| $10^{-5}$ | $9.5\times10^{-5}$ | $43.6°$ | vacuous |

The eigenvalues always move by at most $10^{-4} = \|E\|$ (Weyl). The eigenvector rotation grows as the gap shrinks. With a nearly degenerate pair, the individual eigenvectors are meaningless.

**C3.** $\|\hat R-R_{true}\|_F = 0.017$, $\det\hat R = 1$, and the result matches `scipy.linalg.orthogonal_procrustes`. The residuals are $0.585$ for Procrustes and $0.550$ for unconstrained least squares. The unconstrained fit is slightly smaller, but it fits noise with a non-orthogonal (shearing) matrix. Procrustes is used for aligning shapes, word embeddings and point clouds.

**C4.**

| | slope | intercept |
|---|---|---|
| OLS | 1.638 | — |
| predicted OLS attenuation | $2\cdot\frac{3}{3+0.64} = 1.648$ | — |
| TLS | **2.015** | 0.897 |

OLS is biased towards zero (regression dilution) when $x$ is noisy. TLS minimises the orthogonal distances and is consistent when the $x$ and $y$ noise variances are equal. With unequal variances, rescale first (Deming regression).

## D. Data-science applications

**D1.**

| rank $k$ | storage | relative Frobenius error | PSNR |
|---|---|---|---|
| 5 | 2.0% | 0.184 | 18.4 dB |
| 20 | 7.8% | 0.136 | 21.0 dB |
| 50 | 19.5% | 0.103 | 23.4 dB |
| 100 | 39.1% | 0.074 | 26.4 dB |

The relative error equals $\sqrt{\sum_{i>k}\sigma_i^2}/\|A\|_F$ exactly (Eckart–Young). The top 1, 10 and 50 singular values carry 91.5%, 97.4% and 98.9% of the energy. The DCT (Chapter 8) achieves 22.7 dB with 5% of its coefficients against 21.0 dB here with 7.8% of the storage. A fixed basis that suits natural images can beat the data-adapted but global SVD. The SVD is *optimal* only among rank-$k$ matrices.

**D2.** The singular values are $3.28, 3.06, 2.08, \dots$, and the two dominant ones correspond to the two topics. For the query "algebra", the raw cosines are nonzero only for d2, d3 and d4, the documents that contain the word ($0.41$, $0.58$, $0.71$). In the latent space, **d1 ("matrix eigenvalue vector") gets cosine 0.998** although it does not contain "algebra", while the football documents stay near 0 ($-0.10$ to $0.07$) and the mixed document d9 gets $0.60$. The rank-2 SVD groups co-occurring terms into a "linear-algebra" direction. It thus handles synonymy, because documents about the same topic match even without shared words. This is the ancestor of word embeddings.

**D3.** The first five PCs explain $77.9\%, 9.4\%, 2.7\%, 1.6\%, 1.5\%$ of the variance of daily changes, with 90% in three. The remaining PCs are the i.i.d. measurement noise, which is spread evenly over all directions. The loadings:
- PC1 is all positive ($0.41\to0.20$): the **level**. All yields move together, with the short end moving more because the Nelson–Siegel slope factor also loads there.
- PC2 goes from negative to positive ($-0.43\to+0.42$): the **slope** (steepening or flattening).
- PC3 is negative, then positive, then negative ($-0.44, \dots, +0.48, \dots, -0.40$): the **curvature**.

These three are the famous factors of Litterman–Scheinkman, and fixed-income risk management hedges exactly these.

**D4.**

| $k$ | explained variance | AUC (Q-statistic) | AUC ($T^2$) |
|---|---|---|---|
| 1 | 0.30 | 0.921 | 0.905 |
| 3 | 0.68 | 0.887 | 0.940 |
| 5 | 0.83 | 0.865 | **0.945** |
| 10 | 0.95 | 0.907 | 0.913 |
| 20 | 0.996 | **0.943** | 0.918 |

The two statistics are complementary. $T^2$ detects anomalies *within* the normal subspace (unusually large values of the usual patterns), and $Q$ detects departures *from* it (new correlation patterns). Malignant tumours are mostly "larger than usual" along the main size axes, so $T^2$ with a few PCs works well. In process monitoring both are charted, with control limits from $F$ and $\chi^2$ approximations.
