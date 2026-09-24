# Chapter 13 — Exercises: Regression and Least-Squares Data Fitting

> Lecture: [Chapter 13](../lectures/ch13-regression-least-squares.md) · Solutions: [`solutions/ch13_solutions.md`](../solutions/ch13_solutions.md) · Solution code: [`solutions/code/ch13_solutions.py`](../solutions/code/ch13_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging. This is a core data-science chapter, so the problem set is longer. Assign a subset.

## A. Theory and proofs

**A1 ★** For simple regression $y_i = \beta_0+\beta_1x_i+\varepsilon_i$ derive $\hat\beta_1 = S_{xy}/S_{xx}$ and $\hat\beta_0 = \bar y-\hat\beta_1\bar x$. Show that $\sum e_i = 0$ and $\sum x_ie_i = 0$, that $SST = SSR+SSE$, and that $R^2 = r_{xy}^2$.

**A2 ★★** *(Gauss–Markov.)* Assume $\mathbf y = X\boldsymbol\beta+\boldsymbol\varepsilon$, $E\boldsymbol\varepsilon = \mathbf 0$, $\operatorname{Cov}\boldsymbol\varepsilon = \sigma^2I$, and $X$ of full column rank $p$. (a) Show $\operatorname{Cov}\hat{\boldsymbol\beta} = \sigma^2(X^TX)^{-1}$. (b) Prove that $\hat{\boldsymbol\beta}$ is BLUE: for any linear unbiased estimator $C\mathbf y$, $\operatorname{Cov}(C\mathbf y)-\operatorname{Cov}\hat{\boldsymbol\beta}\succeq0$. (c) Show $E[\text{RSS}] = \sigma^2(n-p)$ using $\operatorname{tr}(I-H) = n-p$.

**A3 ★★** Let $H = X(X^TX)^{-1}X^T = QQ^T$ be the hat matrix. (a) Show that $H$ is symmetric and idempotent, $\operatorname{tr}H = p$, and $0\le h_{ii}\le1$. (b) Prove the leave-one-out identity $y_i-\hat y_{(i)} = \frac{e_i}{1-h_{ii}}$ (use Sherman–Morrison), which gives LOOCV at the cost of a single fit.

**A4 ★★** *(Ridge regression.)* (a) Show $\hat{\boldsymbol\beta}_\lambda = (X^TX+\lambda I)^{-1}X^T\mathbf y = \sum_i\frac{\sigma_i}{\sigma_i^2+\lambda}(\mathbf u_i^T\mathbf y)\mathbf v_i$, and interpret the shrinkage factors $\frac{\sigma_i^2}{\sigma_i^2+\lambda}$. (b) Show that ridge is OLS on the augmented data $\begin{pmatrix}X\\\sqrt\lambda I\end{pmatrix}$, $\begin{pmatrix}\mathbf y\\\mathbf 0\end{pmatrix}$. (c) Compute the effective degrees of freedom $\operatorname{tr}S_\lambda$. (d) Show that the bias is $-\lambda(X^TX+\lambda I)^{-1}\boldsymbol\beta$.

**A5 ★★** *(Lasso.)* For $\min\frac1{2n}\|\mathbf y-X\mathbf b\|^2+\lambda\|\mathbf b\|_1$: (a) write the subgradient optimality (KKT) conditions; (b) for an orthonormal design ($X^TX = nI$) show $\hat b_j = S_\lambda(\hat b_j^{OLS})$ (soft-thresholding); (c) derive the coordinate-descent update and show that all coefficients are zero iff $\lambda\ge\lambda_{\max} = \|X^T\mathbf y\|_\infty/n$.

**A6 ★★** *(Logistic regression.)* Show that the log-likelihood $\ell(\boldsymbol\beta) = \sum[y_i\eta_i-\ln(1+e^{\eta_i})]$ has gradient $X^T(\mathbf y-\boldsymbol\mu)$ and Hessian $-X^TWX$ with $W = \operatorname{diag}(\mu_i(1-\mu_i))$. Deduce that $\ell$ is concave and that Newton's method is the same as IRLS with working response $\mathbf z = X\boldsymbol\beta+W^{-1}(\mathbf y-\boldsymbol\mu)$.

**A7 ★★** For $f(\boldsymbol\beta) = \frac12\|\mathbf r(\boldsymbol\beta)\|^2$ show $\nabla^2f = J^TJ+\sum_ir_i\nabla^2r_i$. Explain why Gauss–Newton converges quadratically for zero-residual problems but only linearly otherwise, and how Levenberg–Marquardt interpolates between Gauss–Newton and gradient descent.

## B. Hand computation

**B1 ★** The data $x = 1,\dots,8$, $y = 3.1, 4.9, 6.2, 7.8, 9.1, 11.2, 12.4, 14.3$. Compute $\hat\beta_0$, $\hat\beta_1$, $s$, $SE(\hat\beta_1)$, the $t$-statistic, $R^2$, a 95% CI for $\beta_1$, and the 95% confidence and prediction intervals at $x_0 = 5.5$.

**B2 ★★** For $X = \{(1,2),(2,1),(3,4),(4,3),(5,6),(6,5)\}$ and $\mathbf y = (5.1, 6.2, 10.8, 12.1, 17.2, 17.9)$, form the normal equations of $y = \beta_0+\beta_1x_1+\beta_2x_2$ and solve them. Report the residuals, $R^2$, $\hat\sigma^2$ and the standard errors, and give the $R$ factor of the QR decomposition of the design.

**B3 ★★** Two groups ($d = 0,1$) are measured at $x = 1,\dots,5$: $y = (2.1, 3.9, 6.2, 7.8, 10.1)$ for $d = 0$ and $y = (4.0, 7.1, 9.8, 13.2, 16.0)$ for $d = 1$. Fit $y = \beta_0+\beta_1x+\beta_2d+\beta_3xd$. Interpret each coefficient, write the two fitted lines, and test whether the slopes differ.

**B4 ★★** For $x = (1,\dots,7,20)$ and $y = (2.2, 4.1, 5.8, 8.3, 9.9, 12.1, 13.8, 22.0)$ compute the leverages, the internally studentised residuals and Cook's distances. Refit without the last point. Is point 8 an outlier, a high-leverage point, or both?

**B5 ★★** Two standardised predictors have correlation $\rho = 0.95$. Compute the VIF, the eigenvalues and condition number of the correlation matrix, and the ridge solutions for $\lambda = 0, 0.1, 1, 10$ when the OLS solution is $(2, 0.5)$. Which eigen-direction is shrunk first?

**B6 ★** With an orthonormal design and OLS coefficients $(3, -1.2, 0.4, 0.05)$, give the lasso and ridge estimates for $\lambda = 0.1, 0.5, 1.5, 3.5$. Which estimator performs variable selection?

**B7 ★★** Fit $y = ae^{bx}$ to $(0,2.0)$, $(1,3.0)$, $(2,5.2)$, $(3,8.1)$. Do one Gauss–Newton step by hand from $(a,b) = (2, 0.4)$ (give $J$, $\mathbf r$ and the step), iterate to convergence, and compare with the log-linearised fit.

**B8 ★★** Logistic regression on $x = (0.5, 1, 1.5, 2, 2.5, 3)$, $y = (0,0,1,0,1,1)$. Do the first IRLS step from $\boldsymbol\beta = \mathbf 0$ by hand ($\boldsymbol\mu$, $W$, $\mathbf z$, $X^TWX$, $X^TW\mathbf z$), iterate to convergence, and report the odds ratio and $P(y = 1\mid x = 1.75)$.

## C. Programming

**C1 ★★** *The Longley benchmark* (`statsmodels.datasets.longley`). Fit the 7-parameter model by normal equations, QR and SVD in float64, and by normal equations and QR in float32. Compare with the NIST certified values using the log relative error (LRE = number of correct digits). Compute $\kappa(X)$, $\kappa(X^TX)$ and the VIFs.

**C2 ★★** Implement lasso by cyclic coordinate descent and verify it against `sklearn.linear_model.Lasso` on the standardised diabetes data for $\lambda = 20, 5, 1, 0.1$. Report the active sets and $\lambda_{\max}$.

**C3 ★★** Generate heteroscedastic data $y = 1+0.5x+(0.2+0.3x)\varepsilon$, $n = 200$. Compare the classical, HC3 sandwich, pairs-bootstrap and true (Monte Carlo) standard errors. Then fit WLS with the correct weights and quantify its efficiency gain.

**C4 ★★** Fit $y = a_1e^{-b_1t}+a_2e^{-b_2t}$ (40 points, noise 0.02, true $(3, 0.4, 2, 2.5)$) with plain Gauss–Newton and with Levenberg–Marquardt from three starting points. Explain the failures and the "label switching" of the solution.

## D. Data-science applications

**D1 ★★** *Diabetes progression* (`sklearn.datasets.load_diabetes`). Using a 70/30 split and 5-fold CV for tuning, compare OLS, ridge and lasso on test MSE and $R^2$. Which variables does lasso drop, and how does this relate to the VIFs of `s1`–`s4`?

**D2 ★★** *Insurance claims.* Simulate 5000 policies with exposure $\in[0.2,1]$ years, age and an urban indicator. Fit a Poisson GLM with offset $\log(\text{exposure})$ by IRLS and verify it against `statsmodels`. Report the urban rate ratio with a 95% CI, show what goes wrong if the offset is omitted, and check the Pearson dispersion.

**D3 ★★** *Breast-cancer diagnosis.* Fit logistic regression by IRLS on three standardised features (mean radius, texture, smoothness). Report the coefficients, SEs and odds ratios per SD, the AUC and accuracy, and a calibration table.

**D4 ★★** *Robustness.* Contaminate a clean linear data set with 5%, 10% and 20% gross outliers in $y$, and separately with 10 high-leverage outliers. Compare OLS and Huber IRLS. When does Huber fail, and what estimators would you use instead?
