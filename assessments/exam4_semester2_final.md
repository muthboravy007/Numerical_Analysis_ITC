# Semester 2 — Final Examination (Chapters 9–15, emphasis on 13–15)

**Time:** 3 hours · **Total:** 120 points · A scientific calculator is allowed, two handwritten A4 formula sheets, and the $t$/normal tables provided. Show all working.

Solutions: [`exam4_semester2_final_solutions.md`](exam4_semester2_final_solutions.md)

---

**Question 1 (16 points) — Simple linear regression.**
For $n = 12$ observations: $\bar x = 5$, $\bar y = 20$, $S_{xx} = 60$, $S_{xy} = 90$, $S_{yy} = 150$.
(a) Compute $\hat\beta_1$, $\hat\beta_0$, SSE, $s^2$ and $R^2$. (7 pts)
(b) Compute $SE(\hat\beta_1)$, the $t$-statistic, and a 95% confidence interval for $\beta_1$ ($t_{0.975,10} = 2.228$). (6 pts)
(c) State the assumptions needed for the interval to be valid, and say how you would check each one. (3 pts)

**Question 2 (14 points) — Least squares via QR.**
A regression with $n = 8$ observations and $p = 3$ coefficients has thin QR factorisation $X = QR$ with
$R = \begin{pmatrix}2&4&2\\0&3&1\\0&0&2\end{pmatrix}$, $Q^T\mathbf y = (10, 6, 2)^T$ and residual norm $\|\mathbf y-X\hat{\boldsymbol\beta}\| = 1.5$.
(a) Compute $\hat{\boldsymbol\beta}$ by back substitution. (4 pts)
(b) Compute $s^2$ and the standard errors, using $\operatorname{Cov}\hat{\boldsymbol\beta} = s^2R^{-1}R^{-T}$. (7 pts)
(c) Explain why this route is more accurate than forming $X^TX$. (3 pts)

**Question 3 (14 points) — Ridge and lasso.**
(a) A design has singular values $\sigma = (4, 1)$ and $U^T\mathbf y = (8, 3)$. Compute the ridge coefficients in the basis of right singular vectors for $\lambda = 0, 1, 4$, and the effective degrees of freedom. Which direction is shrunk more, and why? (8 pts)
(b) For an orthonormal design with OLS estimates $(2.5, -0.8, 0.3)$, give the lasso estimates for $\lambda = 0.5$. Explain geometrically why the lasso produces exact zeros while ridge does not. (6 pts)

**Question 4 (14 points) — PCA.**
Two standardised-scale variables have covariance matrix $\Sigma = \begin{pmatrix}5&2\\2&2\end{pmatrix}$.
(a) Find the principal components and the proportion of variance explained. (8 pts)
(b) Compute the PC scores of the centred observation $(2,1)$. (3 pts)
(c) Why should variables usually be standardised before PCA? (3 pts)

**Question 5 (14 points) — Logistic regression by IRLS.**
The data $x = (-1,0,1,2)$, $y = (0,1,0,1)$, with an intercept.
(a) Write the log-likelihood, its gradient and its Hessian. (4 pts)
(b) Perform the first IRLS/Newton step from $\boldsymbol\beta = \mathbf 0$. (8 pts)
(c) What happens to IRLS when the classes are perfectly separable, and how is this remedied? (2 pts)

**Question 6 (16 points) — Optimisation.**
(a) For $f(\mathbf x) = \frac12(x_1^2+9x_2^2)$, find the step size that minimises the worst-case contraction factor of gradient descent, the resulting rate, and the largest stable step. What rate can optimally tuned heavy-ball momentum achieve? (8 pts)
(b) Solve $\min(x-2)^2+(y-1)^2$ subject to $x+y\le1$ using the KKT conditions. Give the multiplier and interpret it. (8 pts)

**Question 7 (14 points) — Automatic differentiation and SGD.**
(a) For $f(x_1,x_2) = (x_1x_2+\ln x_1)^2$ at $(2,1)$, draw the computational graph, perform the forward pass, and compute $\nabla f$ with one reverse sweep. (8 pts)
(b) Explain why SGD with a constant learning rate does not converge to the exact minimiser, and give conditions on the step sizes that ensure convergence. Why is a noisy solution often acceptable in machine learning? (6 pts)

**Question 8 (18 points) — SVD and low-rank structure.**
(a) Find the SVD of $A = \begin{pmatrix}3&1\\1&3\end{pmatrix}$, the best rank-1 approximation $A_1$, and $\|A-A_1\|_2$ and $\|A-A_1\|_F$. (8 pts)
(b) State the Eckart–Young theorem and explain its role in PCA, recommender systems and image compression. (6 pts)
(c) Why are PCA loadings unstable when two eigenvalues are nearly equal? (4 pts)
