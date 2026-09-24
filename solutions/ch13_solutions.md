# Chapter 13 — Solutions

> Exercises: [`exercises/ch13_exercises.md`](../exercises/ch13_exercises.md) · Numbers from [`solutions/code/ch13_solutions.py`](code/ch13_solutions.py).

## A. Theory and proofs

**A1.** Setting $\partial/\partial\beta_0$ and $\partial/\partial\beta_1$ of $\sum(y_i-\beta_0-\beta_1x_i)^2$ to zero gives $\sum e_i = 0$ and $\sum x_ie_i = 0$. The first gives $\hat\beta_0 = \bar y-\hat\beta_1\bar x$. Substituting into the second gives $\hat\beta_1 = S_{xy}/S_{xx}$. Next, $\sum(y_i-\bar y)^2 = \sum(\hat y_i-\bar y+e_i)^2$, and the cross term $\sum(\hat y_i-\bar y)e_i = \hat\beta_1\sum(x_i-\bar x)e_i = 0$, so $SST = SSR+SSE$. Finally $SSR = \hat\beta_1^2S_{xx} = S_{xy}^2/S_{xx}$, so $R^2 = SSR/SST = S_{xy}^2/(S_{xx}S_{yy}) = r^2$.

**A2.** (a) $\hat{\boldsymbol\beta}-\boldsymbol\beta = (X^TX)^{-1}X^T\boldsymbol\varepsilon$, so its covariance is $(X^TX)^{-1}X^T\sigma^2IX(X^TX)^{-1} = \sigma^2(X^TX)^{-1}$. (b) Unbiasedness for all $\boldsymbol\beta$ forces $CX = I$. Write $C = (X^TX)^{-1}X^T+D$; then $DX = 0$. Now $\operatorname{Cov}(C\mathbf y) = \sigma^2CC^T = \sigma^2[(X^TX)^{-1}+DD^T]$, because the cross terms $(X^TX)^{-1}X^TD^T = (X^TX)^{-1}(DX)^T$ vanish. Since $DD^T\succeq0$, the claim follows. (c) $\mathbf e = (I-H)\boldsymbol\varepsilon$, so $E\|\mathbf e\|^2 = \sigma^2\operatorname{tr}(I-H) = \sigma^2(n-\operatorname{tr}H)$. By the cyclic property of the trace, $\operatorname{tr}H = \operatorname{tr}((X^TX)^{-1}X^TX) = p$.

**A3.** (a) $H^T = H$ and $H^2 = X(X^TX)^{-1}X^TX(X^TX)^{-1}X^T = H$. Then $\operatorname{tr}H = p$ as in A2. Also $h_{ii} = (H^2)_{ii} = \sum_jh_{ij}^2\ge h_{ii}^2$, so $0\le h_{ii}\le1$. (b) Removing row $\mathbf x_i$ gives $(X^TX-\mathbf x_i\mathbf x_i^T)^{-1} = A^{-1}+\frac{A^{-1}\mathbf x_i\mathbf x_i^TA^{-1}}{1-h_{ii}}$, where $A = X^TX$. Then $\hat{\boldsymbol\beta}_{(i)} = \hat{\boldsymbol\beta}-\frac{A^{-1}\mathbf x_ie_i}{1-h_{ii}}$ and $y_i-\mathbf x_i^T\hat{\boldsymbol\beta}_{(i)} = e_i+\frac{h_{ii}e_i}{1-h_{ii}} = \frac{e_i}{1-h_{ii}}$. So $\text{LOOCV} = \frac1n\sum\bigl(\frac{e_i}{1-h_{ii}}\bigr)^2$ (PRESS).

**A4.** (a) With $X = U\Sigma V^T$, $X^TX+\lambda I = V(\Sigma^2+\lambda I)V^T$, which gives the formula. OLS corresponds to $\lambda = 0$, with factor $1/\sigma_i$. Directions with small $\sigma_i$ (the ill-determined ones) are shrunk most: $\frac{\sigma_i^2}{\sigma_i^2+\lambda}\approx0$ when $\sigma_i^2\ll\lambda$. (b) $\|\mathbf y-X\mathbf b\|^2+\lambda\|\mathbf b\|^2$ is exactly the squared norm of the augmented residual. This is also how to *compute* ridge stably by QR. (c) $S_\lambda = X(X^TX+\lambda I)^{-1}X^T = U\operatorname{diag}\frac{\sigma_i^2}{\sigma_i^2+\lambda}U^T$, so $\operatorname{df} = \sum\frac{\sigma_i^2}{\sigma_i^2+\lambda}$, which falls from $p$ to $0$. (d) $E\hat{\boldsymbol\beta}_\lambda = (X^TX+\lambda I)^{-1}X^TX\boldsymbol\beta = \boldsymbol\beta-\lambda(X^TX+\lambda I)^{-1}\boldsymbol\beta$.

**A5.** (a) $\frac1n\mathbf x_j^T(\mathbf y-X\hat{\mathbf b}) = \lambda\operatorname{sign}(\hat b_j)$ if $\hat b_j\ne0$, and $\bigl|\frac1n\mathbf x_j^T(\mathbf y-X\hat{\mathbf b})\bigr|\le\lambda$ if $\hat b_j = 0$. (b) With $X^TX = nI$ the problem separates into $\frac12(b_j-\hat b_j^{OLS})^2+\lambda|b_j|$, whose minimiser is $S_\lambda(z) = \operatorname{sign}(z)(|z|-\lambda)_+$. (c) Fix all coordinates except $j$ and let $\mathbf r^{(j)} = \mathbf y-\sum_{k\ne j}\mathbf x_kb_k$. Then $b_j = S_\lambda(\mathbf x_j^T\mathbf r^{(j)}/n)/(\|\mathbf x_j\|^2/n)$. At $\mathbf b = \mathbf 0$ the KKT conditions read $|\mathbf x_j^T\mathbf y|/n\le\lambda$ for all $j$, i.e. $\lambda\ge\lambda_{\max}$.

**A6.** $\partial\ell/\partial\eta_i = y_i-\mu_i$ and $\partial\mu_i/\partial\eta_i = \mu_i(1-\mu_i)$, so $\nabla\ell = X^T(\mathbf y-\boldsymbol\mu)$ and $\nabla^2\ell = -X^TWX\preceq0$: $\ell$ is concave. The Newton step is $\boldsymbol\beta^+ = \boldsymbol\beta+(X^TWX)^{-1}X^T(\mathbf y-\boldsymbol\mu) = (X^TWX)^{-1}X^TW[X\boldsymbol\beta+W^{-1}(\mathbf y-\boldsymbol\mu)]$. This is a weighted least-squares solve with working response $\mathbf z$.

**A7.** $\nabla f = J^T\mathbf r$ and $\nabla^2f = J^TJ+\sum r_i\nabla^2r_i$. Gauss–Newton drops the second term. If $\mathbf r(\boldsymbol\beta^\ast) = \mathbf 0$ the dropped term vanishes at the solution, and convergence is quadratic like Newton's. Otherwise the error contracts by a factor of about $\|(J^TJ)^{-1}\sum r_i\nabla^2r_i\|$ per step: linear convergence, or divergence if that factor exceeds 1. Levenberg–Marquardt solves $(J^TJ+\mu D)\boldsymbol\delta = -J^T\mathbf r$. With $\mu\to0$ it is Gauss–Newton; with $\mu\to\infty$ it takes a short step along $-D^{-1}\nabla f$ (scaled gradient descent). $\mu$ is adapted by whether the cost decreased, which is a trust-region idea.

## B. Hand computation

**B1.** $\bar x = 4.5$, $\bar y = 8.625$, $S_{xx} = 42$, $S_{xy} = 66.1$, $S_{yy} = 104.275$. The fit is $\hat\beta_1 = 1.573810$ and $\hat\beta_0 = 1.542857$, with $SSE = 0.24619$ and $s = 0.20256$. Then $SE(\hat\beta_1) = s/\sqrt{S_{xx}} = 0.031256$, $t = 50.35$ and $R^2 = 0.99764$. With $t_{0.975,6} = 2.4469$, the 95% CI for $\beta_1$ is $[1.4973, 1.6503]$. At $x_0 = 5.5$, $\hat y = 10.1988$. The CI for the mean response is $[10.0076, 10.3900]$ and the prediction interval is $[9.6676, 10.7301]$. The PI is wider because it adds the noise of a new observation.

**B2.** $X^TX = \begin{pmatrix}6&21&21\\21&91&88\\21&88&91\end{pmatrix}$ and $X^T\mathbf y = (69.3, 291.7, 288.6)$. The solution is $\hat{\boldsymbol\beta} = (1.1375, 2.004167, 0.970833)$. The residuals are $(0.017, 0.083, -0.233, 0.033, 0.217, -0.117)$, $R^2 = 0.99914$ and $\hat\sigma^2 = 0.041111$ with $n-p = 3$. The standard errors are $(0.1958, 0.0866, 0.0866)$. $R = \begin{pmatrix}-2.449&-8.573&-8.573\\0&4.183&3.466\\0&0&-2.342\end{pmatrix}$, and $R^TR = X^TX$.

**B3.** $\hat{\boldsymbol\beta} = (0.05, 1.99, 0.94, 1.02)$ with SEs $(0.189, 0.057, 0.267, 0.080)$.
- Group 0: $\hat y = 0.05+1.99x$.
- Group 1: $\hat y = 0.99+3.01x$.
- $\beta_2$ is the vertical shift *at $x = 0$*.
- $\beta_3 = 1.02$ is the difference in slopes, with $t = 12.7$ and $p = 1.5\times10^{-5}$: the slopes differ significantly.
- With an interaction in the model, $\beta_2$ alone is not "the group effect". The group effect depends on $x$.

**B4.** Leverages: $h_{88} = 0.903$, while the others are $0.125$–$0.224$ (the average is $p/n = 0.25$). The studentised residual of point 8 is $-2.44$ and its Cook's $D = 27.7$; all other points have $D<0.26$. The full fit is $\hat y = 3.81+0.994x$. Without point 8 the fit is $0.186+1.961x$, which predicts $39.4$ at $x = 20$ against the observed $22$. Point 8 has extreme leverage *and* does not follow the trend of the rest, so it drags the line towards itself. Its raw residual is modest ($-1.69$) only because it pulled the fit. Always look at leverage and influence, not just residuals. PRESS LOOCV $= 42.6$ is dominated by this point.

**B5.** VIF $= \frac1{1-0.95^2} = 10.26$. The eigenvalues are $1\pm\rho = 0.05$ and $1.95$, so $\kappa = 39$.

| $\lambda$ | $\hat{\boldsymbol\beta}_\lambda$ | $\|\hat{\boldsymbol\beta}_\lambda\|$ | shrink factor on $0.05$ | shrink factor on $1.95$ |
|---|---|---|---|---|
| 0 | $(2, 0.5)$ | 2.06 | 1 | 1 |
| 0.1 | $(1.439, 0.939)$ | 1.72 | 0.33 | 0.95 |
| 1 | $(0.862, 0.791)$ | 1.17 | 0.048 | 0.66 |
| 10 | $(0.208, 0.200)$ | 0.29 | 0.005 | 0.16 |

The contrast direction $(1,-1)/\sqrt2$ (eigenvalue 0.05) is shrunk first. The coefficients are pulled towards each other, because correlated predictors share credit.

**B6.**

| $\lambda$ | lasso | ridge ($\hat b/(1+\lambda)$) |
|---|---|---|
| 0.1 | $(2.9, -1.1, 0.3, 0)$ | $(2.727, -1.091, 0.364, 0.045)$ |
| 0.5 | $(2.5, -0.7, 0, 0)$ | $(2.0, -0.8, 0.267, 0.033)$ |
| 1.5 | $(1.5, 0, 0, 0)$ | $(1.2, -0.48, 0.16, 0.02)$ |
| 3.5 | $(0,0,0,0)$ | $(0.667, -0.267, 0.089, 0.011)$ |

Only the lasso sets coefficients exactly to zero. Ridge shrinks proportionally and never selects.

**B7.** $J(\mathbf b_0)$ has columns $e^{bx_i}$ and $ax_ie^{bx_i}$: $\begin{pmatrix}1&0\\1.4918&2.9836\\2.2255&8.9022\\3.3201&19.9207\end{pmatrix}$, and $\mathbf r = (0, -0.0164, -0.7489, -1.4598)$. The step is $\boldsymbol\delta = (-0.0573, 0.0845)$, giving $(a,b) = (1.9427, 0.4845)$ and reducing the SSE from $2.692$ to $0.0778$. Six iterations reach $(1.958904, 0.474715)$ with SSE $0.044309$. The log-linearised fit $(1.95647, 0.47462)$ has SSE $0.044540$, which is close but not optimal for the original criterion.

**B8.** At $\boldsymbol\beta = \mathbf 0$: $\mu_i = 0.5$, $w_i = 0.25$ and $z_i = (y_i-0.5)/0.25 = \pm2$. Then $X^TWX = \begin{pmatrix}1.5&2.625\\2.625&5.6875\end{pmatrix}$ and $X^TW\mathbf z = (0, 1.75)$, so $\boldsymbol\beta^{(1)} = (-2.8, 1.6)$. The iterates continue $(-3.884, 2.220)$, $(-4.221, 2.412)$, $(-4.2489, 2.4280)$, and 7 iterations give $\hat{\boldsymbol\beta} = (-4.2491, 2.4281)$ with SE $(3.39, 1.83)$. The odds ratio is $e^{2.428} = 11.3$ per unit of $x$, and $P(y = 1\mid1.75) = 0.500$: the fitted 50% point is $-\hat\beta_0/\hat\beta_1 = 1.75$. With $n = 6$ the SEs are huge. The slope is not significant ($z = 1.33$), despite the large odds ratio.

## C. Programming (key results)

**C1.** $\kappa(X) = 4.9\times10^9$ and $\kappa(X^TX) = 2.4\times10^{19}$. Correct digits (LRE; minimum over coefficients):

| method | float64 | float32 |
|---|---|---|
| normal equations | 7.4 | **−0.2** (no correct digit) |
| QR | 10.9 | 4.9 |
| SVD | 10.9 | — |

Normal equations lose about twice as many digits as QR, because $\kappa^2$ enters. Our float64 normal equations still get 7 digits: the *column-scaled* condition number is much smaller than the raw $2.4\times10^{19}$ (the columns have wildly different scales), and LAPACK's pivoting benefits from that. After standardising, $\kappa = 111$. The VIFs are $136, 1789, 34, 3.6, 399, 759$, so the multicollinearity is intrinsic, and scaling does not remove it.

**C2.** Our coordinate descent matches sklearn to $\le10^{-9}$.

| $\lambda$ | nonzero | sweeps | active set |
|---|---|---|---|
| 20 | 3 | 21 | bmi, bp, s5 |
| 5 | 5 | 29 | + sex, s3 |
| 1 | 7 | 42 | + s1, s6 |
| 0.1 | 9 | 556 | + age, s2, s4 (s3 dropped) |

$\lambda_{\max} = 45.16$. Small $\lambda$ needs many sweeps because correlated predictors (s1–s4) make coordinate descent zig-zag. Warm starts along a decreasing $\lambda$ path fix this. The active set is not monotone: s3 leaves when s2 and s4 enter, because of the correlations.

**C3.**

| SE | classical | HC3 | bootstrap | true (Monte Carlo) |
|---|---|---|---|---|
| slope | 0.0508 | 0.0503 | 0.0489 | 0.0482 |
| intercept | **0.314** | 0.184 | 0.182 | 0.176 |

The classical formula assumes constant variance, and here it overstates the intercept SE by 78%. The sandwich and bootstrap estimates are consistent under heteroscedasticity. WLS with weights $1/\sigma_i^2$ has true slope SE $0.0297$, i.e. **2.6×** smaller variance than OLS. OLS is unbiased but no longer efficient (Gauss–Markov fails).

**C4.**

| start | Gauss–Newton | Levenberg–Marquardt |
|---|---|---|
| $(1,1,1,3)$ | diverges (SSE $10^{17}$) | 12 steps → $(2.998, 0.400, 2.010, 2.491)$, SSE 0.0136 |
| $(1,0.1,1,0.2)$ | explodes in 2 steps | 11 steps → same |
| $(5,1,0.5,1.1)$ | explodes in 1 step | 17 steps → $(2.010, 2.491, 2.998, 0.400)$ |

Near the start $J$ is nearly rank-deficient (two similar exponentials), so the Gauss–Newton step is huge. LM damps it. The third solution is the same model with the two terms swapped: the parameters are identifiable only up to permutation. Impose an ordering such as $b_1<b_2$ to fix it. Sums of exponentials are a classic ill-conditioned inverse problem.

## D. Data-science applications

**D1.**

| model | test MSE | test $R^2$ | nonzero coefficients | tuning |
|---|---|---|---|---|
| OLS | 3097 | 0.393 | 10 | — |
| ridge | **3072** | **0.398** | 10 | $\alpha = 25$ |
| lasso | 3118 | 0.389 | 8 | $\alpha = 0.92$ |

Lasso zeroes `s2` and `s4`. The VIFs are s1 56, s2 36, s3 16, s4 9.5 (these are blood-serum measurements that are strongly correlated with each other). OLS assigns them large coefficients of opposite sign ($-23.9$ vs $+11.5$), whereas lasso keeps one representative per correlated group. The differences in test MSE are within noise, since $n_{test} = 133$. With a moderate $p/n$, regularisation mainly buys interpretability and stability here, not accuracy.

**D2.** IRLS takes 7 iterations to reach $\hat{\boldsymbol\beta} = (-1.9373, 0.02349, 0.32758)$ with SE $(0.0828, 0.0145, 0.0859)$, identical to statsmodels. The urban rate ratio is $1.388$ with 95% CI $[1.173, 1.642]$ (true $e^{0.4} = 1.49$). Omitting the offset gives intercept $-2.468$ instead of $-1.937$. The model then estimates claims *per policy* instead of per policy-year, and any covariate correlated with exposure would also be biased. The Pearson dispersion is $1.03\approx1$, so there is no overdispersion and the Poisson model fits. Values well above 1 would call for quasi-Poisson or negative binomial models.

**D3.** IRLS takes 9 iterations, giving $\hat{\boldsymbol\beta} = (-1.002, 4.919, 1.635, 2.033)$ with SE $(0.20, 0.54, 0.25, 0.27)$, identical to statsmodels. The odds ratios per SD are 137 for radius, 5.1 for texture and 7.6 for smoothness. In-sample AUC is 0.981 and accuracy is 93.3%. Calibration, as (mean predicted, observed) in the $p$-bins $[0,0.2)$, …, $[0.8,1]$: $(0.027, 0.031)$, $(0.29, 0.21)$, $(0.50, 0.59)$, $(0.67, 0.56)$, $(0.97, 0.97)$. The model is good at the extremes and noisier in the middle bins, which contain few patients. Use held-out data for honest AUC and calibration.

**D4.**

| contamination | OLS $(\beta_0,\beta_1)$ | Huber $(\beta_0,\beta_1)$ |
|---|---|---|
| none | $(2.00, 1.53)$ | $(1.99, 1.53)$ |
| 5% in $y$ | $(2.31, 1.77)$ | $(2.03, 1.55)$ |
| 10% | $(4.08, 1.71)$ | $(2.13, 1.54)$ |
| 20% | $(7.87, 1.56)$ | $(2.54, 1.54)$ |
| 10 high-leverage outliers | $(6.58, 0.14)$ | $(2.45, 1.39)$ |

The true values are $(2, 1.5)$. Huber downweights the gross outliers (weights down to 0.045) and stays close to the truth under $y$-contamination. High-leverage outliers still pull it: an M-estimator bounds the influence of large *residuals*, but a leverage point makes its own residual small. Use high-breakdown estimators for that case (LTS, MM-estimators, or RANSAC).
