# Chapter 9 — Solutions

> Exercises: [`exercises/ch09_exercises.md`](../exercises/ch09_exercises.md) · Numbers from [`solutions/code/ch09_solutions.py`](code/ch09_solutions.py).

## A. Theory and proofs

**A1.** (a) Induction on the number $k$ of eigenvectors. Suppose $\sum_{i\le k}c_i\mathbf v_i = \mathbf 0$. Apply $A-\lambda_kI$: $\sum_{i<k}c_i(\lambda_i-\lambda_k)\mathbf v_i = \mathbf 0$. By induction all $c_i(\lambda_i-\lambda_k) = 0$, so $c_i = 0$ for $i<k$, and then $c_k = 0$. (b) If $A\mathbf v = \lambda\mathbf v$ with $\mathbf v\in\mathbb C^n$, then $\bar{\mathbf v}^TA\mathbf v = \lambda\|\mathbf v\|^2$. Its conjugate equals $\bar{\mathbf v}^TA^T\mathbf v = \bar{\mathbf v}^TA\mathbf v$, so the quantity is real and $\lambda$ is real. For orthogonality, $\lambda_1\mathbf v_2^T\mathbf v_1 = \mathbf v_2^TA\mathbf v_1 = (A\mathbf v_2)^T\mathbf v_1 = \lambda_2\mathbf v_2^T\mathbf v_1$. So $(\lambda_1-\lambda_2)\mathbf v_2^T\mathbf v_1 = 0$.

**A2.** Let $A\mathbf x = \lambda\mathbf x$ and choose $i$ with $|x_i| = \|\mathbf x\|_\infty$. Row $i$ gives $(\lambda-a_{ii})x_i = \sum_{j\ne i}a_{ij}x_j$, so $|\lambda-a_{ii}|\le\sum_{j\ne i}|a_{ij}|\frac{|x_j|}{|x_i|}\le r_i$. For the counting statement, let $A(t) = D+t(A-D)$ for $t\in[0,1]$. Its discs have the same centres and radii $tr_i$, so they grow continuously from points. The eigenvalues depend continuously on $t$. The union of the $k$ discs stays disjoint from the others for all $t$, so no eigenvalue can jump between the two regions. At $t = 0$ the region holds exactly $k$ eigenvalues, namely the centres $a_{ii}$.

**A3.** (a) Write $\mathbf x^{(0)} = \sum c_i\mathbf v_i$ with $c_1\ne0$. Then $A^k\mathbf x^{(0)} = \lambda_1^k\bigl[c_1\mathbf v_1+\sum_{i\ge2}c_i(\lambda_i/\lambda_1)^k\mathbf v_i\bigr]$. After normalisation, the bracketed sum is $O(|\lambda_2/\lambda_1|^k)$. (b) For symmetric $A$ take orthonormal $\mathbf v_i$. With $\mathbf y = A^k\mathbf x^{(0)}$, the Rayleigh quotient is $\frac{\sum c_i^2\lambda_i^{2k+1}}{\sum c_i^2\lambda_i^{2k}} = \lambda_1\frac{1+\sum_{i\ge2}(c_i/c_1)^2(\lambda_i/\lambda_1)^{2k+1}}{1+\sum_{i\ge2}(c_i/c_1)^2(\lambda_i/\lambda_1)^{2k}}$. Its error is therefore $O(|\lambda_2/\lambda_1|^{2k})$. The eigenvector error $\epsilon$ enters the Rayleigh quotient only as $\epsilon^2$, because the Rayleigh quotient is stationary at eigenvectors.

**A4.** (a) $P^T = P$, and $P^2 = I-4\mathbf w\mathbf w^T+4\mathbf w(\mathbf w^T\mathbf w)\mathbf w^T = I$. Also $P\mathbf w = -\mathbf w$, and $P\mathbf u = \mathbf u$ for every $\mathbf u\perp\mathbf w$. (b) $\|\mathbf x\| = |\alpha|$ gives $\|\mathbf x-\alpha\mathbf e_1\|^2 = 2(\|\mathbf x\|^2-\alpha x_1)$ and $\mathbf w^T\mathbf x\propto\|\mathbf x\|^2-\alpha x_1$. So $P\mathbf x = \mathbf x-2\mathbf w(\mathbf w^T\mathbf x) = \mathbf x-(\mathbf x-\alpha\mathbf e_1) = \alpha\mathbf e_1$. With $\alpha = -\operatorname{sign}(x_1)\|\mathbf x\|$ the first entry of $\mathbf x-\alpha\mathbf e_1$ is $x_1+\operatorname{sign}(x_1)\|\mathbf x\|$, which adds two numbers of the same sign. The opposite sign risks catastrophic cancellation when $\mathbf x\approx\|\mathbf x\|\mathbf e_1$.

**A5.** (a) $R_k = Q_k^TA_k$, so $A_{k+1} = Q_k^TA_kQ_k$. (b) $A_{k+1}$ is symmetric by (a). $Q_k = A_kR_k^{-1}$ (if $A_k$ is nonsingular) is upper Hessenberg, as the product of a Hessenberg and an upper triangular matrix. $R_kQ_k$ is upper triangular times Hessenberg, hence Hessenberg. A symmetric Hessenberg matrix is tridiagonal. (The singular case follows by continuity, or by using Givens rotations directly.) (c) $A_{k+1} = RQ+\mu I = Q^T(A_k-\mu I)Q+\mu I = Q^TA_kQ$.

**A6.** (a) $A^TA = V\Lambda V^T$ with $\lambda_i\ge0$. Set $\sigma_i = \sqrt{\lambda_i}$ and $\mathbf u_i = A\mathbf v_i/\sigma_i$ for $\sigma_i>0$. These $\mathbf u_i$ are orthonormal, since $\mathbf u_i^T\mathbf u_j = \mathbf v_i^TA^TA\mathbf v_j/(\sigma_i\sigma_j) = \delta_{ij}$. Extend them to an orthonormal basis to get $AV = U\Sigma$. (b) Orthogonal factors preserve both norms, so $\|A\|_2 = \|\Sigma\|_2 = \sigma_1$ and $\|A\|_F^2 = \|\Sigma\|_F^2$. (c) $\dim\ker B\ge n-k$, so there is a unit vector $\mathbf z\in\ker B\cap\operatorname{span}\{\mathbf v_1..\mathbf v_{k+1}\}$. Then $\|(A-B)\mathbf z\|^2 = \|A\mathbf z\|^2 = \sum_{i\le k+1}\sigma_i^2(\mathbf v_i^T\mathbf z)^2\ge\sigma_{k+1}^2$. For $A_k$, $A-A_k = \sum_{i>k}\sigma_i\mathbf u_i\mathbf v_i^T$ has norm $\sigma_{k+1}$.

## B. Hand computation

**B1.** The row discs are $|z-6|\le3$, $|z+5|\le1$ and $|z-4|\le3$. The column discs are $|z-6|\le3$, $|z+5|\le3$ and $|z-4|\le1$. The disc around $-5$ is isolated, so it contains exactly one eigenvalue. That eigenvalue is real, because a non-real eigenvalue would force its conjugate into the same disc. Intersecting with the column discs puts it in $[-6,-4]$. The union of the other two row discs contains two eigenvalues, with real parts in $[1,9]$. The computed eigenvalues are $6.8895$, $3.2831$ and $-5.1726$, all real.

**B2.** The $\mu^{(k)}$ are $2,\ 2.5,\ 2.8,\ 3.428571,\ 3.416667,\ 3.414634,\ 3.414286,\ 3.414226$, converging to $3.414214$. The final $\mathbf x = (-0.7267, 1, -0.6875)$ tends to $(-\frac1{\sqrt2},1,-\frac1{\sqrt2})$. Aitken's $\Delta^2$ gives $3.25, 2.226, 3.416888, 3.414216, 3.414214$. The early values are erratic because the sequence is not yet in its asymptotic regime (the index $p$ of the largest component still changes), but from the fourth term Aitken gains about 3 digits.

**B3.** The Rayleigh quotients are $2, 2.8, 3.142857, 3.308411, 3.376054, 3.400884, 3.409611, 3.412631$. The error ratios tend to $0.0046/0.0133 = 0.345$, close to $(2/3.414)^2 = 0.343$.

**B4.** $\mu = 0.666667,\ 0.585859,\ 0.585786,\dots$, converging to $2-\sqrt2 = 0.585786$ with $\mathbf v = (\frac12,\frac1{\sqrt2},\frac12)$. The factor for the eigenvector is $\frac{|\lambda_3-q|}{|\lambda_2-q|} = \frac{0.0858}{1.5} = 0.057$, and the Rayleigh quotient error squares this. That is why convergence is so fast.

**B5.** The largest component of $\mathbf v_1$ is at $i = 2$, so $\mathbf x = \frac{(-1,2,-1)}{\lambda_1v_2}$. The deflated matrix is $B = T-\lambda_1\mathbf v_1\mathbf x^T = \begin{pmatrix}1.292893&0.414214&-0.707107\\0&0&0\\-0.707107&0.414214&1.292893\end{pmatrix}$. Deleting row and column 2 leaves $\begin{pmatrix}1.2929&-0.7071\\-0.7071&1.2929\end{pmatrix}$, with eigenvalues $1.2929\pm0.7071 = 2,\ 0.585786$. These are the remaining eigenvalues of $T$.

**B6.** (a) $\|\mathbf x\| = 3$ and $\alpha = -3$. Then $\mathbf w = (5,-1,2)/\sqrt{30}$ and $P = \frac1{15}\begin{pmatrix}-10&5&-10\\5&14&2\\-10&2&11\end{pmatrix}$, with $P\mathbf x = (-3,0,0)$. (b) The result is
$T = \begin{pmatrix}5&-3&0&0\\-3&4&-1&0\\0&-1&3&1\\0&0&1&2\end{pmatrix}$.
It has the same eigenvalues as $A$: $0.92259, 1.71389, 3.72643, 7.63709$. For the first step, column 1 below the diagonal is $(1,2,2)$ with norm 3, so the new subdiagonal entry is $-3$.

**B7.**
- $A_1 = \begin{pmatrix}3.4&-0.2\\-0.2&0.6\end{pmatrix}$
- $A_2 = \begin{pmatrix}3.413793&0.034483\\0.034483&0.586207\end{pmatrix}$
- $A_3$ has diagonal $(3.414201, 0.585799)$ and $|a_{21}| = 0.005917$.
- $A_4$ has diagonal $(3.414213, 0.585787)$ and $|a_{21}| = 0.001015$.

The ratios of successive off-diagonal entries are $0.172$, which equals $\lambda_2/\lambda_1 = \frac{2-\sqrt2}{2+\sqrt2} = 0.1716$.

**B8.** (a) $A^TA = \begin{pmatrix}5&3\\3&5\end{pmatrix}$ has eigenvalues $8$ and $2$, so $\sigma = 2\sqrt2,\sqrt2$, with $\mathbf v_1 = \frac{(1,1)}{\sqrt2}$ and $\mathbf v_2 = \frac{(1,-1)}{\sqrt2}$. Then $\mathbf u_1 = A\mathbf v_1/\sigma_1 = (1,0)$ and $\mathbf u_2 = (0,1)$. So $A = I\cdot\operatorname{diag}(2\sqrt2,\sqrt2)\cdot V^T$. (The rows of $A$ are already orthogonal.) (b) $A = \mathbf u\mathbf v^T\sigma_1$ with $\mathbf u = (1,2,0)/\sqrt5$, $\mathbf v = (1,2)/\sqrt5$ and $\sigma_1 = 5$. Then $A^+ = \frac1{\sigma_1}\mathbf v\mathbf u^T = \frac1{25}\begin{pmatrix}1&2&0\\2&4&0\end{pmatrix}$. The minimum-norm least-squares solution is $\mathbf x = A^+(1,1,1)^T = (0.12, 0.24)$, and the residual is $(-0.4, 0.2, -1)$.

**B9.** $A^TA = \begin{pmatrix}11&7\\7&11\end{pmatrix}$ gives $\sigma = \sqrt{18} = 4.2426$ and $2$. With $\mathbf v_1 = \frac{(1,1)}{\sqrt2}$ and $\mathbf u_1 = \frac{(2,2,1)}{3}$, $A_1 = \begin{pmatrix}2&2\\2&2\\1&1\end{pmatrix}$. The error is $\|A-A_1\|_2 = \|A-A_1\|_F = \sigma_2 = 2$. The two norms agree because only one singular value is discarded.

## C. Programming (key results)

**C1.**

| $\lambda_2/\lambda_1$ | power | inverse ($q = 1.05$) | RQI |
|---|---|---|---|
| 0.5 | 21 | 8 | 4 |
| 0.9 | 122 | 14 | 4 |
| 0.99 | 929 | 60 | 4 |

The power method needs about $\frac{\ln10^{-12}}{2\ln r}$ iterations (Rayleigh quotient). Inverse iteration improves the ratio to $\frac{|1-1.05|}{|r-1.05|}$. RQI converges cubically and is independent of the gap. It must be started close to $\mathbf v_1$, though: from a random start it converged to an interior eigenvalue (0.2457) in our first attempt.

**C2.** The errors against `eigh` are $5\times10^{-15}$, $3\times10^{-14}$ and $7\times10^{-14}$ for $n = 10, 50, 100$. The QR steps per eigenvalue are $2.56$, $2.12$ and $2.05$. That is about 2 steps per eigenvalue, which reflects the (generically cubic) convergence of the Wilkinson shift. Total cost: $\frac43n^3$ for the tridiagonalisation plus $O(n^2)$ for all the QR steps on the tridiagonal matrix.

**C3.**

| $\sigma_i$ | `eig(A^TA)` | `svd` |
|---|---|---|
| $1$ | $0$ | $3\times10^{-16}$ |
| $4.6\times10^{-4}$ | $4\times10^{-11}$ | $2\times10^{-15}$ |
| $2.8\times10^{-6}$ | $1.6\times10^{-6}$ | $4\times10^{-13}$ |
| $1.7\times10^{-8}$ | $5.8\times10^{-2}$ | $1.1\times10^{-10}$ |
| $10^{-10}$ | $1.0$ (lost) | $3\times10^{-9}$ |

Forming $A^TA$ squares the singular values. Anything with $\sigma_i^2<u\,\sigma_1^2$, i.e. $\sigma_i<10^{-8}$, drowns in round-off. The SVD works with $A$ directly and achieves an absolute error of about $u\sigma_1$.

**C4.** `eigsh` in shift-invert mode returns $19.7376, 49.3345, 49.3345, 78.9314, 98.6308, 98.6308$ in 0.16 s. These agree with the exact discrete eigenvalues to all digits shown. The continuous values are $19.7392, 49.348, \dots$, a relative discretisation error of $O(h^2)$, about $8\times10^{-5}$. The repeated eigenvalues ($i\ne j$ pairs) come from the symmetry of the square.

## D. Data-science applications

**D1.** The top five explained-variance ratios are $0.149, 0.136, 0.118, 0.084, 0.058$. Reaching 80/90/95% of the variance needs **13 / 21 / 29** components. Eckart–Young is verified exactly: for $k = 2, 10, 20, 40$, $\|X_c-X_k\|_F^2$ is $1.5435\times10^6$, $5.652\times10^5$, $2.282\times10^5$ and $2.547\times10^4$, each equal to $\sum_{i>k}\sigma_i^2$. Logistic regression CV accuracy is 0.587 (2 PCs), 0.889 (10), 0.895 (20) and 0.913 (all 64). Twenty components keep 98% of the accuracy with less than a third of the features.

**D2.** For the moons, the smallest eigenvalues of $L_{sym}$ are $0$, $4\times10^{-5}$ and then a jump to $0.0056$. For the circles they are $0, 0$ and then $0.0052$. The eigengap after the second eigenvalue reveals the two clusters. ARI for spectral clustering is **1.000** on both data sets. ARI for k-means is 0.243 on the moons and $-0.002$ on the circles, no better than chance: k-means assumes convex clusters, whereas the Laplacian follows connectivity.

**D3.** The eigenvalues are $1, 0.924875, 0.821313, 0.703812$ and the stationary distribution is $\pi = (0.151, 0.247, 0.277, 0.326)$. The total-variation distance from A is 0.344, 0.113, 0.016 and 0.0003 after 10, 25, 50 and 100 months. It first drops below 0.01 at month **57**, close to the prediction $\ln0.01/\ln0.924875 = 59.0$. The mixing time is governed by $|\lambda_2|$: the "memory" of the initial rating decays like $0.925^k$.

**D4.**

| $\lambda$ | recovered rank | test RMSE |
|---|---|---|
| 1 | 52 | 0.651 (overfits the noise) |
| **5** | **5** | **0.370** |
| 20 | 5 | 1.125 (over-shrinks) |

Predicting the mean gives 2.29. $\lambda$ trades bias against variance, just like ridge or lasso; in practice choose it by validation. The remaining gap to the noise level 0.1 is shrinkage bias, which can be reduced by re-fitting the singular values on the selected subspace ("debiasing").
