# Chapter 14 — Exercises: Symmetric Matrices, the SVD and PCA

> Lecture: [Chapter 14](../lectures/ch14-symmetric-matrices-svd-pca.md) · Solutions: [`solutions/ch14_solutions.md`](../solutions/ch14_solutions.md) · Solution code: [`solutions/code/ch14_solutions.py`](../solutions/code/ch14_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★★** *(Spectral theorem.)* Prove by induction that every real symmetric $A$ can be written as $A = Q\Lambda Q^T$ with $Q$ orthogonal. (Take a real unit eigenvector $\mathbf q_1$, complete it to an orthogonal $[\mathbf q_1\ Q_2]$, and show that $Q_2^TAQ_2$ is symmetric and that the off-diagonal blocks vanish.) Deduce $A = \sum\lambda_i\mathbf q_i\mathbf q_i^T$.

**A2 ★★** *(Rayleigh quotient and Courant–Fischer.)* For symmetric $A$ with $\lambda_1\ge\cdots\ge\lambda_n$, prove $\lambda_1 = \max_{\mathbf x\ne0}\frac{\mathbf x^TA\mathbf x}{\mathbf x^T\mathbf x}$ and $\lambda_n = \min$. Prove $\lambda_k = \max_{\dim V = k}\min_{\mathbf x\in V\setminus0}R(\mathbf x)$. Explain why the first principal component maximises the projected sample variance.

**A3 ★★** Prove that the following are equivalent for symmetric $A$: (i) $\mathbf x^TA\mathbf x>0$ for all $\mathbf x\ne0$; (ii) all eigenvalues are positive; (iii) $A = R^TR$ with $R$ nonsingular; (iv) all leading principal minors are positive (Sylvester). (Prove (i)⇔(ii)⇔(iii) fully and (iv)⇒(i) by induction with the Cholesky step.)

**A4 ★★** (a) Prove the Eckart–Young theorem in the Frobenius norm: $\min_{\operatorname{rank}B\le k}\|A-B\|_F^2 = \sum_{i>k}\sigma_i^2$. (b) Show that $A^+ = V\Sigma^+U^T$ satisfies the four Moore–Penrose conditions, and that $\mathbf x^+ = A^+\mathbf b$ is the minimum-norm least-squares solution.

**A5 ★★** Let $X_c$ be a centred $n\times p$ data matrix with SVD $U\Sigma V^T$. Show: (a) the sample covariance is $S = \frac{1}{n-1}V\Sigma^2V^T$, so the principal axes are the $\mathbf v_i$ and the variances are $\frac{\sigma_i^2}{n-1}$; (b) the scores are $X_cV = U\Sigma$; (c) the $k$-dimensional subspace that *maximises* the projected variance is the one that *minimises* the reconstruction error $\|X_c-X_cV_kV_k^T\|_F^2$, and the minimum equals $\sum_{i>k}\sigma_i^2$.

**A6 ★★★** *(Perturbation.)* (a) Prove Weyl's inequality $|\lambda_i(A+E)-\lambda_i(A)|\le\|E\|_2$ for symmetric $A$ and $E$ (use Courant–Fischer), and the analogous result for singular values. (b) State the Davis–Kahan $\sin\theta$ theorem and explain why eigen*vectors* (and PCA loadings) are unstable when the eigengap is small.

## B. Hand computation

**B1 ★** Orthogonally diagonalise $S = \begin{pmatrix}4&1&1\\1&4&1\\1&1&4\end{pmatrix}$. Write $S = 6P_1+3P_2$ with orthogonal projectors $P_1, P_2$, and use it to compute $S^{-1}$ and $S^{1/2}$.

**B2 ★** Find the principal axes of $3x^2+2xy+3y^2 = 1$ and classify $x^2+4xy+y^2$. Use Sylvester's criterion to show that $A = \begin{pmatrix}4&2&-2\\2&5&1\\-2&1&6\end{pmatrix}$ is positive definite, and find its Cholesky factor.

**B3 ★★** Compute the SVD of $A = \begin{pmatrix}3&0\\4&5\end{pmatrix}$ by hand, together with $\kappa_2(A)$ and $A^+$.

**B4 ★** Find the minimum-norm solution of the underdetermined equation $x+2y+2z = 9$ using the pseudoinverse.

**B5 ★★** PCA by hand for the five points $(2,1), (3,4), (5,5), (7,8), (8,7)$. Give the mean, the covariance matrix, the principal axes, the explained-variance ratios, the PC1 scores and the rank-1 reconstruction. Verify that the reconstruction SSE equals $(n-1)\lambda_2$.

**B6 ★★** Let $\Sigma = \begin{pmatrix}4&2\\2&3\end{pmatrix}$. Compare the Euclidean and Mahalanobis distances from the mean for $(2,2)$ and $(2,-2)$. Find the two whitening matrices $L^{-1}$ (Cholesky) and $\Sigma^{-1/2}$ (ZCA) and check that each whitens.

**B7 ★** For $S$ in B1 compute the Rayleigh quotients of $(1,0,0)$, $(1,1,1)$ and $(1,-1,0)$, and check that they lie in $[\lambda_{\min},\lambda_{\max}]$.

## C. Programming

**C1 ★★** *Randomised SVD* (Halko–Martinsson–Tropp). For $A\in\mathbb R^{2000\times500}$ with $\sigma_i = 1/i$, compute a rank-20 approximation with oversampling $p\in\{0,10\}$ and power iterations $q\in\{0,1,2\}$. Compare the error with the optimum $\sigma_{21}$, and the timing with the full SVD.

**C2 ★★** Illustrate Weyl and Davis–Kahan: perturb $\operatorname{diag}(1,1+\delta,3)$ by a symmetric $E$ with $\|E\| = 10^{-4}$ for $\delta = 10^{-1},10^{-3},10^{-5}$. Report the eigenvalue changes and the rotation angle of the first eigenvector.

**C3 ★★** *Orthogonal Procrustes.* Rotate a 50-point 3-D cloud by a known rotation, add noise, and recover the rotation from the SVD of $P^TB$. Compare the residual with unconstrained least squares.

**C4 ★★** *Total least squares.* For data with noise in both $x$ and $y$ (true line $y = 2x+1$, noise SD 0.8, $x\sim U(-3,3)$), compare the OLS slope, its predicted attenuation, and the TLS slope from the smallest right singular vector.

## D. Data-science applications

**D1 ★★** *Image compression.* Compute rank-$k$ SVD approximations of the grayscale `china.jpg` for $k = 5, 20, 50, 100$. Report the storage ratio, the relative Frobenius error (check it against Eckart–Young) and the PSNR. Compare with the DCT results of Chapter 8 D3.

**D2 ★★** *Latent semantic analysis.* Build the term–document count matrix of the 9 mini-documents in the solution script (linear-algebra and football topics, plus one mixed document). Compute its rank-2 SVD, and compare raw and latent cosine similarities for the query "algebra". Why do documents without the word "algebra" become relevant?

**D3 ★★** *Yield-curve factors.* Simulate 1000 days of Nelson–Siegel yield curves (10 maturities) with random-walk level, slope and curvature factors plus noise. Run PCA on the daily changes: report the explained variance and interpret the first three loading vectors.

**D4 ★★** *Anomaly detection.* Fit PCA to 200 benign breast-cancer samples (standardised) and score the rest by (i) the squared prediction error (Q-statistic) and (ii) Hotelling's $T^2$ in the PC space. Report the AUC for detecting malignant samples with $k = 1,3,5,10,20$ components.
