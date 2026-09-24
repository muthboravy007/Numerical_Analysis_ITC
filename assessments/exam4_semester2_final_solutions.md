# Semester 2 Final — Solutions and Marking Scheme

> All numbers are verified by [`code/verify_exams.py`](code/verify_exams.py).

**Q1.** (a) $\hat\beta_1 = S_{xy}/S_{xx} = 1.5$ and $\hat\beta_0 = 20-1.5(5) = 12.5$. Then SSE $= S_{yy}-\hat\beta_1S_{xy} = 150-135 = 15$, $s^2 = 15/10 = 1.5$ and $R^2 = 135/150 = 0.90$. [7]
(b) $SE(\hat\beta_1) = \sqrt{1.5/60} = 0.1581$ and $t = 9.49$ ($p = 2.6\times10^{-6}$). The 95% CI is $1.5\pm2.228(0.1581) = [1.148, 1.852]$. [6]
(c) The assumptions are linearity, independence, constant variance and normal errors (or large $n$). Check them with residual-vs-fitted plots, a scale-location plot, a normal Q–Q plot of the residuals, and a plot against time or order for independence. [3]

**Q2.** (a) Back substitution: $2\beta_3 = 2$ gives $\beta_3 = 1$; $3\beta_2+1 = 6$ gives $\beta_2 = 5/3$; $2\beta_1+4(5/3)+2 = 10$ gives $\beta_1 = 2/3$. So $\hat{\boldsymbol\beta} = (0.6667, 1.6667, 1)$. [4]
(b) $s^2 = 1.5^2/(8-3) = 0.45$. The diagonal of $R^{-1}R^{-T}$ is $(0.7222, 0.1389, 0.25)$, so the standard errors are $\sqrt{0.45\cdot\text{diag}} = (0.570, 0.250, 0.335)$. [7]
(c) QR is backward stable and its error is proportional to $\kappa(X)$. Forming $X^TX$ squares the condition number and can lose all accuracy for ill-conditioned designs. [3]

**Q3.** (a) The coefficient on $\mathbf v_i$ is $\frac{\sigma_i(\mathbf u_i^T\mathbf y)}{\sigma_i^2+\lambda}$, and $\text{df} = \sum\frac{\sigma_i^2}{\sigma_i^2+\lambda}$. [8]

| $\lambda$ | coefficients | df |
|---|---|---|
| 0 | $(2, 3)$ | 2 |
| 1 | $(1.882, 1.5)$ | 1.44 |
| 4 | $(1.6, 0.6)$ | 1.0 |

The direction with the small singular value ($\sigma_2 = 1$) is shrunk much more: its factor falls from $1$ to $\frac1{1+4} = 0.2$, against $\frac{16}{20} = 0.8$. That is the poorly determined, high-variance direction.
(b) $S_{0.5}(2.5, -0.8, 0.3) = (2.0, -0.3, 0)$. The $\ell_1$ ball has corners on the axes, so the elliptical RSS contours typically first touch it at a corner, where some coordinates are zero. The $\ell_2$ ball is smooth, so the tangency point is generically off the axes. [6]

**Q4.** (a) $\det(\Sigma-\lambda I) = \lambda^2-7\lambda+6$, so $\lambda = 6, 1$. The eigenvectors are $\mathbf v_1 = \frac{(2,1)}{\sqrt5}$ and $\mathbf v_2 = \frac{(-1,2)}{\sqrt5}$. The proportions of variance explained are $6/7 = 85.7\%$ and $1/7 = 14.3\%$. [8]
(b) The scores are $\mathbf v_1^T(2,1) = \sqrt5 = 2.236$ and $\mathbf v_2^T(2,1) = 0$: the point lies exactly on the first axis. [3]
(c) PCA maximises variance, so variables with large units dominate. Standardising (PCA on the correlation matrix) makes the result independent of measurement units. [3]

**Q5.** (a) $\ell = \sum[y_i\eta_i-\ln(1+e^{\eta_i})]$, $\nabla\ell = X^T(\mathbf y-\boldsymbol\mu)$ and $\nabla^2\ell = -X^TWX$ with $W = \operatorname{diag}(\mu_i(1-\mu_i))$. [4]
(b) At $\boldsymbol\beta = \mathbf 0$: $\mu_i = \frac12$ and $w_i = \frac14$, so $z_i = \pm2$. Then $X^TWX = \begin{pmatrix}1&0.5\\0.5&1.5\end{pmatrix}$ and $X^TW\mathbf z = X^T(\mathbf y-\boldsymbol\mu) = (0, 1)$, giving $\boldsymbol\beta^{(1)} = (-0.4, 0.8)$. [8]
(c) Under separation the MLE does not exist: $\|\boldsymbol\beta\|\to\infty$ and $W\to0$, so IRLS diverges or becomes ill-conditioned. Remedies are a ridge penalty, Firth's correction, or early stopping. [2]

**Q6.** (a) The eigenvalues are $1$ and $9$. The optimal step is $\frac2{1+9} = 0.2$, with rate $\frac{9-1}{9+1} = 0.8$. GD is stable only for step $<\frac29 = 0.222$. Heavy ball achieves $\frac{\sqrt9-1}{\sqrt9+1} = 0.5$. [8]
(b) The unconstrained minimiser $(2,1)$ violates $x+y\le1$, so the constraint is active. Stationarity gives $2(x-2)+\mu = 0$ and $2(y-1)+\mu = 0$, so $x-2 = y-1$. With $x+y = 1$ this gives $(1,0)$ and $\mu = 2\ge0$. Interpretation: relaxing the constraint to $x+y\le1+\epsilon$ lowers the optimal value by about $\mu\epsilon = 2\epsilon$ (a shadow price). [8]

**Q7.** (a) The graph is $v_1 = x_1x_2 = 2$, $v_2 = \ln x_1 = 0.693147$, $v_3 = v_1+v_2 = 2.693147$ and $f = v_3^2 = 7.253042$. The reverse sweep: $\bar v_3 = 2v_3 = 5.386294$ and $\bar v_1 = \bar v_2 = \bar v_3$. Then $\bar x_1 = \bar v_1x_2+\bar v_2/x_1 = 5.386294(1+0.5) = 8.079442$ and $\bar x_2 = \bar v_1x_1 = 10.772589$. [8]
(b) The stochastic gradient has variance $\sigma^2>0$ even at the minimiser, so with a constant step $\eta$ the iterates keep fluctuating, in a neighbourhood of size $O(\eta\sigma^2)$. Convergence requires $\sum\eta_k = \infty$ and $\sum\eta_k^2<\infty$ (e.g. $\eta_k\propto1/k$), or mini-batches of growing size, or variance reduction. In machine learning, the statistical error ($\sim1/\sqrt n$) exceeds the optimisation error anyway, and the noise can even help generalisation. [6]

**Q8.** (a) The matrix is symmetric PD, so its SVD is its eigen-decomposition: $\sigma = (4, 2)$ with $\mathbf u_1 = \mathbf v_1 = \frac{(1,1)}{\sqrt2}$ and $\mathbf u_2 = \mathbf v_2 = \frac{(1,-1)}{\sqrt2}$. Then $A_1 = 4\mathbf v_1\mathbf v_1^T = \begin{pmatrix}2&2\\2&2\end{pmatrix}$, and $\|A-A_1\|_2 = \|A-A_1\|_F = \sigma_2 = 2$. [8]
(b) Eckart–Young: among all matrices of rank $\le k$, the truncated SVD $A_k$ minimises $\|A-B\|_2$ and $\|A-B\|_F$, with errors $\sigma_{k+1}$ and $\sqrt{\sum_{i>k}\sigma_i^2}$. In PCA, it makes the top-$k$ components the best $k$-dimensional reconstruction. In recommender systems, it is the ideal (fully observed) low-rank model. In image compression, it gives the optimal rank-$k$ storage. [6]
(c) Davis–Kahan: an eigenvector perturbed by $E$ rotates by an angle of about $\|E\|/\text{gap}$. With nearly equal variances, sampling noise can rotate the loadings arbitrarily within their common subspace. Only the subspace is meaningful. [4]
