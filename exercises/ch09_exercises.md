# Chapter 9 — Exercises: Approximating Eigenvalues and the SVD

> Lecture: [Chapter 9](../lectures/ch09-approximating-eigenvalues.md) · Solutions: [`solutions/ch09_solutions.md`](../solutions/ch09_solutions.md) · Solution code: [`solutions/code/ch09_solutions.py`](../solutions/code/ch09_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★** (a) Prove that eigenvectors belonging to distinct eigenvalues are linearly independent. (b) Prove that a real symmetric matrix has real eigenvalues and that eigenvectors for distinct eigenvalues are orthogonal.

**A2 ★★** Prove the Gershgorin circle theorem: every eigenvalue of $A$ lies in $\bigcup_iR_i$, where $R_i = \{z:|z-a_{ii}|\le\sum_{j\ne i}|a_{ij}|\}$. Explain (without a full proof) why a union of $k$ discs that is disjoint from the others contains exactly $k$ eigenvalues. (Hint: use continuity in $t$ of the eigenvalues of $D+t(A-D)$.)

**A3 ★★** Let $A$ be diagonalisable with $|\lambda_1|>|\lambda_2|\ge\cdots$, and let $\mathbf x^{(0)}$ have a nonzero component along $\mathbf v_1$. (a) Show that the normalised iterates of the power method converge to $\pm\mathbf v_1$ with error $O(|\lambda_2/\lambda_1|^k)$. (b) If $A$ is symmetric, show that the Rayleigh quotient $\frac{\mathbf x^{(k)T}A\mathbf x^{(k)}}{\mathbf x^{(k)T}\mathbf x^{(k)}}$ has error $O(|\lambda_2/\lambda_1|^{2k})$.

**A4 ★★** Let $P = I-2\mathbf w\mathbf w^T$ with $\|\mathbf w\|_2 = 1$. (a) Show that $P$ is symmetric and orthogonal and has eigenvalues $-1$ (once) and $1$ ($n-1$ times). (b) Given $\mathbf x$, show that $\mathbf w = (\mathbf x-\alpha\mathbf e_1)/\|\mathbf x-\alpha\mathbf e_1\|$ with $\alpha = -\operatorname{sign}(x_1)\|\mathbf x\|$ gives $P\mathbf x = \alpha\mathbf e_1$. Why is this choice of sign numerically preferable?

**A5 ★★** Let $A_k = Q_kR_k$ and $A_{k+1} = R_kQ_k$. (a) Show that $A_{k+1} = Q_k^TA_kQ_k$, so all $A_k$ have the same eigenvalues. (b) Show that if $A_k$ is symmetric tridiagonal, so is $A_{k+1}$. (c) Show that the shifted step $A_k-\mu I = QR$, $A_{k+1} = RQ+\mu I$ is also a similarity transformation.

**A6 ★★★** (a) Derive the SVD $A = U\Sigma V^T$ from the spectral decomposition of $A^TA$. (b) Show that $\|A\|_2 = \sigma_1$ and $\|A\|_F^2 = \sum\sigma_i^2$. (c) Prove the Eckart–Young theorem in the 2-norm: for every matrix $B$ of rank $\le k$, $\|A-B\|_2\ge\sigma_{k+1}$, with equality for $A_k = \sum_{i\le k}\sigma_i\mathbf u_i\mathbf v_i^T$. (Hint: $\ker B$ has dimension $\ge n-k$ and must meet $\operatorname{span}\{\mathbf v_1,\dots,\mathbf v_{k+1}\}$.)

## B. Hand computation

**B1 ★** Draw the Gershgorin row and column discs of $A = \begin{pmatrix}6&2&1\\1&-5&0\\2&1&4\end{pmatrix}$. What can you conclude about the location and the reality of the eigenvalues?

**B2 ★★** Apply the $\ell_\infty$-scaled power method (B&F Algorithm 9.1) to $T = \operatorname{tridiag}(-1,2,-1)$ ($3\times3$) from $\mathbf x^{(0)} = (1,0,0)^T$ for 8 iterations. Accelerate the sequence of $\mu$ values with Aitken's $\Delta^2$ method. (Exact $\lambda_1 = 2+\sqrt2$.)

**B3 ★** Repeat B2 with the symmetric power method (Rayleigh quotients). Verify that the error ratio approaches $(\lambda_2/\lambda_1)^2 = 0.343$.

**B4 ★★** Use inverse iteration with shift $q = 0.5$ and $\mathbf x^{(0)} = (1,1,1)^T$ to find the smallest eigenvalue of $T$. Predict the convergence factor.

**B5 ★★** Given $\lambda_1 = 2+\sqrt2$ and $\mathbf v_1 = (1,-\sqrt2,1)^T$ for $T$, perform Wielandt deflation and find the remaining eigenvalues from the deflated $2\times2$ matrix.

**B6 ★★** (a) Construct the Householder matrix $P$ with $P(2,-1,2)^T = \alpha\mathbf e_1$. (b) Reduce $A = \begin{pmatrix}5&1&2&2\\1&4&1&0\\2&1&3&1\\2&0&1&2\end{pmatrix}$ to symmetric tridiagonal form.

**B7 ★★** Perform four steps of the unshifted QR algorithm on $\begin{pmatrix}3&1\\1&1\end{pmatrix}$. Show that $|a_{21}^{(k)}|$ decreases by the factor $\lambda_2/\lambda_1$ at each step.

**B8 ★★** (a) Compute the SVD of $\begin{pmatrix}2&2\\1&-1\end{pmatrix}$ by hand. (b) For the rank-one $A = \begin{pmatrix}1&2\\2&4\\0&0\end{pmatrix}$ find $\sigma_1$, the pseudoinverse $A^+$ and the minimum-norm least-squares solution of $A\mathbf x = (1,1,1)^T$.

**B9 ★** Find the best rank-one approximation of $\begin{pmatrix}3&1\\1&3\\1&1\end{pmatrix}$ and its error in the 2-norm and the Frobenius norm.

## C. Programming

**C1 ★★** Build random symmetric $200\times200$ matrices with $\lambda_1 = 1$ and $\lambda_2/\lambda_1 = 0.5, 0.9, 0.99$. Count the iterations to $10^{-12}$ of the power method, inverse iteration with shift $1.05$, and Rayleigh-quotient iteration.

**C2 ★★★** Implement the symmetric eigenvalue algorithm: Householder tridiagonalisation, then QR steps with the Wilkinson shift and deflation. Compare with `eigh` for $n = 10, 50, 100$ and report the average number of QR steps per eigenvalue.

**C3 ★★** Build $A\in\mathbb R^{60\times10}$ with singular values $10^0,\dots,10^{-10}$ (log-spaced). Compare the relative accuracy of the singular values from `eig(A^TA)` and from `np.linalg.svd`.

**C4 ★★** Use `scipy.sparse.linalg.eigsh` in shift-invert mode to compute the six smallest eigenvalues of the 2-D Dirichlet Laplacian on a $100\times100$ grid ($n = 10^4$). Compare with the exact discrete eigenvalues $\frac4{h^2}(\sin^2\frac{i\pi h}2+\sin^2\frac{j\pi h}2)$ and the continuous ones $\pi^2(i^2+j^2)$.

## D. Data-science applications

**D1 ★★** *PCA of handwritten digits* (`sklearn.datasets.load_digits`, $1797\times64$). Report the explained-variance ratios, the number of components needed for 80/90/95% of the variance, and verify Eckart–Young: $\|X_c-X_k\|_F^2 = \sum_{i>k}\sigma_i^2$. How does 5-fold CV accuracy of logistic regression depend on the number of PCs?

**D2 ★★** *Spectral clustering.* For two moons and two concentric circles (400 points each), build a Gaussian similarity graph ($\sigma = 0.1$) and the normalised Laplacian $L_{sym} = I-D^{-1/2}WD^{-1/2}$. Cluster the rows of its two bottom eigenvectors (row-normalised) with k-means. Compare the adjusted Rand index with k-means on the raw coordinates.

**D3 ★★** *Credit-rating migration.* A four-state chain (A, B, C, D) has monthly transition matrix
$P = \begin{pmatrix}0.90&0.08&0.02&0\\0.05&0.85&0.08&0.02\\0.01&0.09&0.80&0.10\\0&0&0.10&0.90\end{pmatrix}$.
Find the stationary distribution as a left eigenvector, the second-largest eigenvalue modulus, and the number of steps until the total-variation distance from the stationary distribution (starting in A) drops below 0.01. Compare with $\ln0.01/\ln|\lambda_2|$.

**D4 ★★★** *Recommender systems.* Generate a $300\times200$ rank-5 rating matrix plus noise ($\sigma = 0.1$) and observe 20% of its entries. Recover it with soft-impute: repeat $Z\leftarrow\mathcal S_\lambda(P_\Omega(M)+P_{\Omega^c}(Z))$, where $\mathcal S_\lambda$ soft-thresholds the singular values. Try $\lambda = 1, 5, 20$ and report the recovered rank and the test RMSE.
