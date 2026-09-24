# Chapter 13 — Regression and Least Squares for Data Science

> **Data-science chapter** building on Burden & Faires §6.6 (Cholesky), §8.1 (discrete least squares), §9.6 (SVD) and §10 (nonlinear systems).
> **Code:** [`numlib/regression.py`](../numlib/regression.py), [`numlib/least_squares.py`](../numlib/least_squares.py) · **All numbers reproduced by** [`lectures/code/ch13.py`](code/ch13.py)
> **Worked problems:** [`examples/ch13`](../examples/ch13_examples.md) · **Homework:** [`exercises/ch13`](../exercises/ch13_exercises.md)

## Learning outcomes

1. Derive the least-squares estimates of simple and multiple linear regression and their statistical properties (unbiasedness, covariance, Gauss–Markov).
2. Carry out inference: standard errors, $t$- and $F$-tests, confidence and prediction intervals, $R^2$ and adjusted $R^2$; diagnose with residuals and leverage.
3. Choose a numerically sound algorithm (normal equations, QR, SVD) and recognise multicollinearity, scaling problems and rank deficiency.
4. Build polynomial, spline, interaction and categorical models and select their complexity by cross-validation and information criteria.
5. Apply ridge and lasso regularisation; relate ridge to SVD filter factors and lasso to soft-thresholding; choose $\lambda$ by (generalised) cross-validation.
6. Handle heteroscedastic, correlated and outlier-contaminated data with WLS, GLS and robust (IRLS) regression.
7. Fit nonlinear models by Gauss–Newton and Levenberg–Marquardt, and GLMs (logistic, Poisson) by IRLS.

**The model.** $\mathbf y = X\boldsymbol\beta + \boldsymbol\varepsilon$ with $X\in\mathbb R^{n\times p}$ (first column usually $\mathbf 1$), $E[\boldsymbol\varepsilon] = \mathbf 0$, $\operatorname{Cov}(\boldsymbol\varepsilon) = \sigma^2I$. The least-squares estimate minimises $\|\mathbf y - X\boldsymbol\beta\|_2^2$.

---

## 13.1 Simple Linear Regression

For $y_i = \beta_0 + \beta_1x_i + \varepsilon_i$, with $\bar x,\bar y$ the means,
$$
S_{xx} = \sum(x_i-\bar x)^2,\quad S_{xy} = \sum(x_i-\bar x)(y_i-\bar y),\quad S_{yy} = \sum(y_i-\bar y)^2,
$$
$$
\hat\beta_1 = \frac{S_{xy}}{S_{xx}},\qquad\hat\beta_0 = \bar y - \hat\beta_1\bar x .
$$
(The centred form is numerically better than the raw-sum formula of §8.1 — it avoids the cancellation in $m\sum x^2 - (\sum x)^2$, cf. Chapter 1.)

**Properties.** $\hat\beta_0,\hat\beta_1$ are unbiased, with
$$
\operatorname{Var}(\hat\beta_1) = \frac{\sigma^2}{S_{xx}},\qquad\operatorname{Var}(\hat\beta_0) = \sigma^2\Bigl(\frac1n + \frac{\bar x^2}{S_{xx}}\Bigr),\qquad s^2 = \frac{\text{SSE}}{n-2},\ \ \text{SSE} = S_{yy} - \hat\beta_1S_{xy}.
$$
**Goodness of fit.** $R^2 = 1 - \frac{\text{SSE}}{S_{yy}} = r^2$ with $r = \frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}$.
**Inference** (normal errors): $T = \frac{\hat\beta_1-\beta_1}{s/\sqrt{S_{xx}}}\sim t_{n-2}$. At $x_0$, with $\hat y_0 = \hat\beta_0+\hat\beta_1x_0$:
$$
\text{CI for }E[y_0]:\ \hat y_0\pm t_{n-2}s\sqrt{\tfrac1n + \tfrac{(x_0-\bar x)^2}{S_{xx}}},\qquad
\text{PI for }y_0:\ \hat y_0\pm t_{n-2}s\sqrt{1+\tfrac1n + \tfrac{(x_0-\bar x)^2}{S_{xx}}}.
$$

### Examples for §13.1

Data (advertising spend $x$ in \$1000, sales $y$ in 1000 units), $n = 10$: $x = 1,\dots,10$, $y = 3.2, 4.8, 5.1, 7.4, 8.0, 9.9, 10.1, 12.6, 13.0, 14.8$.

**Example 13.1.1 (estimates).** $\bar x = 5.5$, $\bar y = 8.89$, $S_{xx} = 82.5$, $S_{xy} = 104.65$, $S_{yy} = 134.549$. Hence $\hat\beta_1 = 104.65/82.5 = 1.26848$ and $\hat\beta_0 = 8.89 - 1.26848(5.5) = 1.91333$: each extra \$1000 of advertising is associated with $\approx1270$ more units sold.

**Example 13.1.2 (fit quality).** Residuals $0.018, 0.350, -0.619, 0.413, -0.256, 0.376, -0.693, 0.539, -0.330, 0.202$ (they sum to zero). $\text{SSE} = 134.549 - 1.26848(104.65) = 1.80206$, $s^2 = 0.22526$, $s = 0.4746$, $R^2 = 0.98661$, $r = 0.99328$.

**Example 13.1.3 (inference on the slope).** $\text{se}(\hat\beta_1) = \frac{0.4746}{\sqrt{82.5}} = 0.05225$, $t = 24.28$ ($p = 9\times10^{-9}$). With $t_{8,0.975} = 2.306$: 95% CI $[1.148, 1.389]$. $\text{se}(\hat\beta_0) = 0.3242$.

**Example 13.1.4 (confidence vs prediction interval).** At $x_0 = 12$ (slightly beyond the data): $\hat y_0 = 17.135$; 95% CI for the mean response $[16.28, 17.99]$; 95% prediction interval for a new observation $[15.75, 18.52]$ — wider because it includes the noise of the new observation, and both widen with $(x_0-\bar x)^2$.

**Example 13.1.5 (Anscombe's quartet — always look at residuals).** Four classic datasets all give $\hat\beta_0 = 3.00$, $\hat\beta_1 = 0.500$, $R^2 = 0.667$. But the residuals differ completely: set 2 has strongly autocorrelated residuals (lag correlation $0.73$ — a curved relationship); set 3 has one residual of $3.24$ (an outlier); set 4's slope is determined by a single point at $x = 19$. Identical summary statistics, different stories.

**Example 13.1.6 (leverage).** Adding the point $(25, 12)$ to the advertising data changes the slope from $1.268$ to $0.373$. Its **leverage** $h_{ii} = \frac1n + \frac{(x_i-\bar x)^2}{S_{xx}} = 0.825$ (average leverage $p/n = 0.18$): far-out $x$ values pull the line toward themselves.

---

## 13.2 Multiple Linear Regression

Setting $\nabla_{\boldsymbol\beta}\|\mathbf y - X\boldsymbol\beta\|^2 = -2X^T(\mathbf y-X\boldsymbol\beta) = \mathbf 0$ gives the **normal equations** $X^TX\hat{\boldsymbol\beta} = X^T\mathbf y$; if $X$ has full column rank,
$$
\hat{\boldsymbol\beta} = (X^TX)^{-1}X^T\mathbf y,\qquad\hat{\mathbf y} = H\mathbf y,\quad H = X(X^TX)^{-1}X^T\ (\text{hat matrix}).
$$
$H$ is the orthogonal projector onto $\operatorname{col}(X)$: $H^2 = H = H^T$, $\operatorname{tr}H = p$; the residuals $\mathbf e = (I-H)\mathbf y$ are orthogonal to every column of $X$.

**Theorem 13.1 (Gauss–Markov).** Under $E\boldsymbol\varepsilon = 0$, $\operatorname{Cov}\boldsymbol\varepsilon = \sigma^2I$, $\hat{\boldsymbol\beta}$ is unbiased with $\operatorname{Cov}(\hat{\boldsymbol\beta}) = \sigma^2(X^TX)^{-1}$, and it has the smallest variance among all linear unbiased estimators (BLUE).

**Estimation of $\sigma^2$:** $s^2 = \frac{\|\mathbf e\|^2}{n-p}$; $\text{se}(\hat\beta_j) = s\sqrt{[(X^TX)^{-1}]_{jj}}$; $t_j = \hat\beta_j/\text{se}_j\sim t_{n-p}$.
**Overall $F$-test:** $F = \frac{\text{SSR}/(p-1)}{\text{SSE}/(n-p)}\sim F_{p-1,n-p}$.
**Adjusted $R^2$:** $\bar R^2 = 1 - (1-R^2)\frac{n-1}{n-p}$ — penalises useless predictors.

### Examples for §13.2

House prices (\$10k) vs size (100 m²) and age (years), $n = 8$:
size $= 1.2, 1.5, 1.7, 2.0, 2.2, 2.5, 2.8, 3.0$; age $= 20, 15, 18, 10, 8, 12, 5, 3$; price $= 12.5, 15.8, 16.2, 21.0, 23.1, 24.9, 29.8, 32.0$.

**Example 13.2.1 (normal equations).**
$$
X^TX = \begin{pmatrix}8&16.9&91\\16.9&38.51&167.7\\91&167.7&1291\end{pmatrix},\quad X^T\mathbf y = \begin{pmatrix}175.3\\400.75\\1717.2\end{pmatrix}\ \Rightarrow\ \hat{\boldsymbol\beta} = (6.937,\ 8.517,\ -0.2652).
$$
Each extra 100 m² adds \$85 170; each year of age subtracts \$2 652, *holding the other variable fixed*.

**Example 13.2.2 (leverage).** $h_{ii} = 0.43, 0.33, 0.41, 0.26, 0.29, 0.57, 0.30, 0.41$; $\sum h_{ii} = 3 = p$. House 6 (large but relatively old) has the highest leverage.

**Example 13.2.3 (inference).** $s = 0.481$; standard errors $(2.315, 0.713, 0.0747)$; $t = (3.00, 11.95, -3.55)$ with $p$-values $(0.030, 7\times10^{-5}, 0.016)$. $R^2 = 0.99653$, $\bar R^2 = 0.99514$, overall $F = 718.2$ ($p = 7\times10^{-7}$).

**Example 13.2.4 (partial vs marginal effects).** Regressing price on age *alone* gives slope $-1.082$ — four times the multiple-regression coefficient $-0.265$. Size and age are strongly correlated ($r = -0.915$): newer houses are bigger, so the simple regression attributes the size effect to age (omitted-variable bias).

**Example 13.2.5 (adjusted $R^2$).** Adding a column of pure noise raises $R^2$ from $0.99653$ to $0.99759$ (it can never decrease) and here even $\bar R^2$ from $0.99514$ to $0.99579$ — with $n = 8$ chance fits are easy. Hypothesis tests, validation data or cross-validation are needed to reject such variables.

**Example 13.2.6 (Gauss–Markov by simulation).** $20\,000$ simulated datasets from $y = 1 + 2t - t^2 + \varepsilon$ ($\sigma = 0.5$, 30 points): the average estimate is $(1.001, 1.995, -0.993)$ (unbiased) and the sampling variances $(0.0665, 1.420, 1.324)$ match $\sigma^2\operatorname{diag}(X^TX)^{-1} = (0.0658, 1.411, 1.317)$.

---

## 13.3 Numerical Methods for Least Squares

| method | cost ($n\gg p$) | error ∝ | notes |
|---|---|---|---|
| normal equations + Cholesky | $np^2 + p^3/3$ | $\kappa(X)^2u$ | fastest; fine if $\kappa(X)\ll10^8$ |
| Householder QR: $X = QR$, $R\hat{\boldsymbol\beta} = Q^T\mathbf y$ | $2np^2$ | $\kappa(X)u$ | **default** (statsmodels, R `lm`) |
| SVD: $\hat{\boldsymbol\beta} = V\Sigma^+U^T\mathbf y$ | $\approx4np^2$ | $\kappa(X)u$ | rank-deficient data (`np.linalg.lstsq`) |
| CG/LSQR on $X$ | $k\cdot\text{nnz}(X)$ | iterative | huge sparse $X$ (Chapter 7) |

With $X = QR$: $\|\mathbf y - X\boldsymbol\beta\|^2 = \|Q^T\mathbf y - R\boldsymbol\beta\|^2$, minimised by $R_1\boldsymbol\beta = (Q^T\mathbf y)_{1:p}$, and the residual norm is $\|(Q^T\mathbf y)_{p+1:n}\|$. Also $(X^TX)^{-1} = R_1^{-1}R_1^{-T}$ and $h_{ii} = \|Q_{i,1:p}\|^2$, so standard errors and leverages come for free.

**Multicollinearity.** If columns are nearly linearly dependent, $\kappa(X)$ is large and coefficients are unstable (large standard errors), even though predictions may be fine. The **variance inflation factor** $\text{VIF}_j = \frac{1}{1-R_j^2}$ ($R_j^2$ from regressing $x_j$ on the others) measures it; $\text{VIF}>10$ is a warning.

### Examples for §13.3

**Example 13.3.1 (all methods agree on a well-conditioned problem).** For the house data, normal equations, Householder QR and SVD all give $(6.93741, 8.51678, -0.26520)$. Here $\kappa(X) = 183$ and $\kappa(X^TX) = 3.3\times10^4$.

**Example 13.3.2 (they disagree on an ill-conditioned one).** Degree-10 monomial regression on 40 points in $[0,1]$ with exact coefficients all equal to 1: $\kappa(X) = 2.0\times10^7$, $\kappa(X^TX) = 4.1\times10^{14}$. Coefficient errors: normal equations $2.9\times10^{-3}$; Householder QR $1.1\times10^{-9}$; SVD $2.7\times10^{-10}$ — six orders of magnitude better, exactly as $\kappa^2$ vs $\kappa$ predicts.

**Example 13.3.3 (QR by hand).** $A = \begin{pmatrix}3&-6\\4&-8\\0&1\end{pmatrix}$, $\mathbf b = (-1,7,2)^T$. Gram–Schmidt: $\mathbf q_1 = (0.6, 0.8, 0)$, $r_{11} = 5$, $r_{12} = \mathbf q_1^T\mathbf a_2 = -10$, $\mathbf a_2 - r_{12}\mathbf q_1 = (0,0,1)$ so $\mathbf q_2 = (0,0,1)$, $r_{22} = 1$. $Q^T\mathbf b = (5, 2)$, and $R\mathbf x = Q^T\mathbf b$ gives $\mathbf x = (5, 2)$. Householder gives the same $R$ up to signs, plus a third component of $Q^T\mathbf b$ equal to $-5$: the residual norm $\|\mathbf b - A\mathbf x\| = 5$.

**Example 13.3.4 (multicollinearity).** $x_2 = x_1 + 0.05\cdot$noise, $x_3$ independent, $y = 1 + x_1 + x_3 + \varepsilon$. VIFs: $328, 329, 1.05$. OLS gives $\hat\beta_1 = 0.539$ (se $1.56$) and $\hat\beta_2 = 0.346$ (se $1.55$): individually meaningless, although their *sum* ($0.885\approx1$) is well determined; $\hat\beta_3 = 0.972$ (se $0.082$) is fine.

**Example 13.3.5 (units matter).** The house design with size in m² and age in days has $\kappa(X) = 6.3\times10^4$; after standardising the predictors (mean 0, s.d. 1) $\kappa = 4.8$. Standardisation is diagonal preconditioning (Chapter 7) and makes coefficients comparable.

**Example 13.3.6 (the dummy-variable trap).** Intercept plus one-hot columns for all 3 neighbourhoods: the 4 columns satisfy $\mathbf 1 = \mathbf d_1+\mathbf d_2+\mathbf d_3$, so $X$ has rank 3 (singular values $3.28, 1.73, 1.49, 0$) and $X^TX$ is singular. The SVD returns the minimum-norm solution $(15.79, -1.64, 7.28, 10.15)$; dropping one dummy (the reference category) gives the identifiable $(25.93, -11.78, -2.87)$. Both produce identical fitted values.

---

## 13.4 Polynomial, Spline and Feature-Based Regression

Linear regression is linear in the **parameters**; the columns of $X$ may be any functions of the inputs: $x^k$, splines, $\sin/\cos$ (Fourier features, §8.7), interactions $x_1x_2$, indicator (dummy) variables.

**Model selection.** Training error always decreases with complexity. Use
* **$K$-fold cross-validation:** average squared prediction error on held-out folds;
* **AIC** $= n\ln(\text{RSS}/n) + 2p$ and **BIC** $= n\ln(\text{RSS}/n) + p\ln n$ (Gaussian errors, up to constants).

**Regression splines.** A cubic spline with knots $\kappa_1,\dots,\kappa_K$ has basis $1, x, x^2, x^3, (x-\kappa_k)_+^3$ ($K+4$ parameters): flexible locally without global oscillation. For numerical stability use **B-splines** (`scipy.interpolate.make_lsq_spline`) instead of truncated powers.

### Examples for §13.4

60 noisy samples of $e^x\sin3x$ on $[-1,1]$ ($\sigma = 0.2$).

**Example 13.4.1 (choosing the degree).**

| degree | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| RSS | 11.20 | 10.40 | 3.04 | 1.540 | 1.504 | 1.504 | 1.472 | 1.442 | 1.404 |
| AIC | −96.7 | −99.2 | −171.0 | **−209.7** | −209.2 | −207.2 | −204.4 | −201.7 | −199.3 |
| BIC | −92.5 | −92.9 | −162.6 | **−199.3** | −196.6 | −192.5 | −185.6 | −178.7 | −172.1 |
| 10-fold CV | 0.203 | 0.197 | 0.063 | 0.0308 | **0.0307** | 0.0327 | 0.0379 | 0.0436 | 0.158 |

AIC and BIC choose degree 4; CV is essentially tied between 4 and 5. RSS alone would pick 12.

**Example 13.4.2 (conditioning of the basis).** Same model space, different bases: $\kappa$ of the design matrix for degree 5/10/15 is $56 / 6.2\times10^3 / 7.7\times10^5$ (monomials) vs $6.6 / 45 / 486$ (Legendre polynomials in $x\in[-1,1]$).

**Example 13.4.3 (a regression spline).** A cubic spline with knots at $-0.5, 0, 0.5$ (7 parameters) fits with maximum deviation $0.46$ from the true curve (degree-6 polynomial: $0.42$). Splines' advantage is local control and stable extrapolation behaviour; the truncated-power basis already has $\kappa = 909$ — B-splines keep it $O(1)$.

**Example 13.4.4 (interactions and dummies).** Wage vs years of education with a dummy `female` and interaction `edu × female`: $\widehat{\text{wage}} = -13.41 + 2.407\,\text{edu} - 0.078\,\text{female} - 0.150\,\text{edu}\times\text{female}$. The return to a year of education is $2.41$ for men and $2.41 - 0.15 = 2.26$ for women (tiny illustrative data — not a real finding).

**Example 13.4.5 (extrapolation).** At $x = 1.2$ and $1.5$ (outside the data range) the true values are $-1.47$ and $-4.38$. Degree 3 predicts $-0.52$ and $-4.07$; degree 12 predicts $-401$ and $-12\,816$. High-degree polynomials must never be used outside the data range.

---

## 13.5 Regularisation: Ridge and Lasso

**Ridge regression** minimises $\|\mathbf y - X\boldsymbol\beta\|^2 + \lambda\|\boldsymbol\beta\|^2$ (predictors standardised, intercept not penalised):
$$
\hat{\boldsymbol\beta}_\lambda = (X^TX+\lambda I)^{-1}X^T\mathbf y = \sum_j\frac{s_j^2}{s_j^2+\lambda}\cdot\frac{\mathbf u_j^T\mathbf y}{s_j}\mathbf v_j\qquad(X = U\Sigma V^T).
$$
The **filter factors** $\frac{s_j^2}{s_j^2+\lambda}$ damp directions with small singular values — exactly the unstable directions of multicollinearity. Effective degrees of freedom $\operatorname{df}(\lambda) = \operatorname{tr}H_\lambda = \sum\frac{s_j^2}{s_j^2+\lambda}$.

**Choosing $\lambda$ without refitting.** For linear smoothers $\hat{\mathbf y} = H\mathbf y$:
$$
\text{LOOCV} = \frac1n\sum_i\Bigl(\frac{y_i-\hat y_i}{1-h_{ii}}\Bigr)^2,\qquad\text{GCV} = \frac{\frac1n\|\mathbf y-\hat{\mathbf y}\|^2}{(1-\operatorname{tr}H/n)^2}.
$$

**Lasso** minimises $\frac{1}{2n}\|\mathbf y - X\boldsymbol\beta\|^2 + \lambda\|\boldsymbol\beta\|_1$. Not differentiable at 0, so no closed form; **coordinate descent** updates one coefficient at a time by **soft-thresholding**:
$$
\beta_j\leftarrow\frac{S\bigl(\frac1n\mathbf x_j^T\mathbf r^{(j)},\ \lambda\bigr)}{\frac1n\|\mathbf x_j\|^2},\qquad S(z,\gamma) = \operatorname{sign}(z)\max(|z|-\gamma,0),
$$
where $\mathbf r^{(j)}$ is the residual without feature $j$. Lasso sets coefficients **exactly** to zero (variable selection); ridge only shrinks.

### Examples for §13.5

**Example 13.5.1 (ridge as an SVD filter).** For the collinear data of Example 13.3.4 (standardised), the singular values are $10.14, 6.86, 0.276$. The small one corresponds to the direction $x_1 - x_2$.

| $\lambda$ | filter factors | $\hat{\boldsymbol\beta}$ |
|---|---|---|
| 0 | $(1, 1, 1)$ | $(0.469, 0.305, 0.916)$ |
| 0.1 | $(0.999, 0.998, 0.432)$ | $(0.421, 0.353, 0.914)$ |
| 1 | $(0.990, 0.979, 0.071)$ | $(0.388, 0.382, 0.898)$ |
| 10 | $(0.911, 0.825, 0.008)$ | $(0.362, 0.366, 0.770)$ |
| 100 | $(0.507, 0.320, 0.001)$ | $(0.218, 0.221, 0.325)$ |

A small $\lambda$ already neutralises the unstable direction (the two collinear coefficients become equal) while barely affecting the others.

**Example 13.5.2 (bias–variance trade-off).** 2000 simulations with true $\boldsymbol\beta = (1,0,1)$ on the collinear design:

| $\lambda$ | bias² | variance | MSE |
|---|---|---|---|
| 0 (OLS) | 0.0000 | 3.140 | 3.140 |
| 1 | 0.433 | 0.024 | **0.456** |
| 5 | 0.500 | 0.007 | 0.507 |
| 25 | 0.615 | 0.004 | 0.618 |

OLS is unbiased but its variance is enormous; ridge trades a little bias for a 100-fold variance reduction.

**Example 13.5.3 (choosing $\lambda$ by LOOCV/GCV).**

| $\lambda$ | 0.01 | 0.1 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|---|
| df | 2.88 | 2.43 | 2.04 | 1.94 | 1.74 | 1.39 |
| LOOCV | 0.2929 | 0.2873 | **0.2832** | 0.2849 | 0.3088 | 0.4316 |
| GCV | 0.2929 | 0.2873 | **0.2831** | 0.2846 | 0.3070 | 0.4249 |

Both pick $\lambda\approx1$, computed from a single fit per $\lambda$.

**Example 13.5.4 (lasso selects variables).** 100 observations, 8 standardised predictors, true $\boldsymbol\beta = (3,-2,0,0,1.5,0,0,0)$:

| $\lambda$ | $\hat{\boldsymbol\beta}$ | non-zeros |
|---|---|---|
| 0.01 | $(3.48, -1.91, -0.05, 0.04, 1.51, -0.04, 0.01, 0.02)$ | 8 |
| 0.1 | $(3.40, -1.80, 0, 0, 1.42, 0, 0, 0)$ | **3** |
| 0.5 | $(3.02, -1.36, 0, 0, 1.03, 0, 0, 0)$ | 3 |
| 1.0 | $(2.53, -0.80, 0, 0, 0.55, 0, 0, 0)$ | 3 |
| 2.0 | $(1.55, 0, 0, 0, 0, 0, 0, 0)$ | 1 |

At $\lambda = 0.1$ it recovers exactly the true support; coordinate descent converged in 8 sweeps and matches `sklearn.linear_model.Lasso`.

**Example 13.5.5 (ridge does not select; soft-thresholding).** Ridge on the same data with $\lambda = 1, 100, 1000$ shrinks all coefficients but never to exactly zero (e.g. $\lambda = 100$: $(1.74, -0.88, -0.01, 0.05, 0.81, 0.04, 0.18, 0.09)$). The lasso's building block: $S(2.5,1) = 1.5$, $S(0.4,1) = 0$, $S(-1.7,1) = -0.7$.

---

## 13.6 Weighted, Generalised and Robust Least Squares

**Weighted LS (WLS).** If $\operatorname{Var}(\varepsilon_i) = \sigma^2/w_i$ (heteroscedasticity), minimise $\sum w_i(y_i - \mathbf x_i^T\boldsymbol\beta)^2$: $\hat{\boldsymbol\beta} = (X^TWX)^{-1}X^TW\mathbf y$ — ordinary LS on rows scaled by $\sqrt{w_i}$.

**Generalised LS (GLS).** If $\operatorname{Cov}(\boldsymbol\varepsilon) = \sigma^2\Sigma$ (e.g. autocorrelated time series), factor $\Sigma = LL^T$ (Cholesky) and "whiten": $L^{-1}\mathbf y = L^{-1}X\boldsymbol\beta + L^{-1}\boldsymbol\varepsilon$ has uncorrelated errors, so OLS on the whitened data is BLUE: $\hat{\boldsymbol\beta} = (X^T\Sigma^{-1}X)^{-1}X^T\Sigma^{-1}\mathbf y$.

**Robust regression (M-estimation) by IRLS.** Minimise $\sum\rho(r_i/s)$ with a loss that grows slowly for large residuals, e.g. Huber's $\rho(u) = \frac12u^2$ ($|u|\le c$), $c|u| - \frac12c^2$ otherwise. The optimality conditions are a weighted LS problem with weights $w_i = \psi(u_i)/u_i = \min(1, c/|u_i|)$, which depend on the fit — iterate: fit, compute weights, refit (**iteratively reweighted least squares**). $s$ = robust scale (MAD/0.6745); $c = 1.345$ gives 95% efficiency under normality. With $w_i = 1/|r_i|$, IRLS computes the least-absolute-deviation (median) regression.

### Examples for §13.6

**Example 13.6.1 (heteroscedasticity).** $y = 2 + 3x + \varepsilon$ with $\operatorname{sd}(\varepsilon)\propto x$, 40 points on $[1,10]$. OLS: $(0.94, 3.04)$ with (misleading) standard errors $(2.03, 0.33)$; WLS with $w = 1/x^2$: $(1.74, 2.86)$ with standard errors $(0.69, 0.23)$ — about 3× more precise for the intercept.

**Example 13.6.2 (autocorrelated errors — GLS).** 100 time points, AR(1) errors with $\phi = 0.8$. OLS gives $(1.448, 0.418)$ with naive standard errors $(0.334, 0.058)$, but the true standard errors of OLS under autocorrelation are $(0.936, 0.160)$ — the naive ones are 3× too small, so tests are wildly overconfident. GLS (Cholesky whitening) gives $(1.314, 0.450)$ with correct standard errors $(0.903, 0.153)$.

**Example 13.6.3 (outliers — Huber IRLS).** 30 points from $y = 1 + 2x$ with three gross outliers ($+15, -20, +25$). OLS: $(1.19, 2.08)$. Huber IRLS (12 iterations): $(0.95, 2.004)$; the outliers receive weights $0.078, 0.060, 0.047$ — automatically down-weighted.

**Example 13.6.4 (regression on group means).** Means $10.2, 12.1, 13.8, 16.3$ at $x = 1,2,3,4$ computed from $50, 5, 20, 2$ observations. Unweighted fit: $8.10 + 2.00x$; weighting each mean by its count (variance $\sigma^2/n_i$): $8.34 + 1.85x$ — the fit follows the precise means.

**Example 13.6.5 (least absolute deviations).** IRLS with $w_i = 1/|r_i|$ on the outlier data gives $(0.59, 2.071)$: also robust (the median of residuals is minimised), but less efficient than Huber for the Gaussian bulk.

---

## 13.7 Nonlinear Least Squares

Minimise $\|\mathbf r(\boldsymbol\beta)\|^2$ with residuals $r_i = f(x_i;\boldsymbol\beta) - y_i$ and Jacobian $J_{ij} = \partial r_i/\partial\beta_j$.

**Gauss–Newton.** Linearise $\mathbf r(\boldsymbol\beta+\boldsymbol\delta)\approx\mathbf r + J\boldsymbol\delta$ and solve the *linear* least-squares problem $\min\|J\boldsymbol\delta + \mathbf r\|$ (by QR — never form $J^TJ$ unnecessarily): $\boldsymbol\beta\leftarrow\boldsymbol\beta+\boldsymbol\delta$. It is Newton's method on $\nabla\|\mathbf r\|^2 = 2J^T\mathbf r$ with the Hessian approximated by $2J^TJ$ (dropping $\sum r_i\nabla^2r_i$): fast (near-quadratic) for small-residual problems, but may diverge from poor starts.

**Levenberg–Marquardt.** $(J^TJ + \lambda D)\boldsymbol\delta = -J^T\mathbf r$, $D = \operatorname{diag}(J^TJ)$: $\lambda\to0$ gives Gauss–Newton, $\lambda\to\infty$ a short steepest-descent step. Decrease $\lambda$ after a successful step, increase it after a failure (a trust-region method). This is `scipy.optimize.least_squares` / `curve_fit`.

**Uncertainty.** At the optimum, $\operatorname{Cov}(\hat{\boldsymbol\beta})\approx s^2(J^TJ)^{-1}$ with $s^2 = \frac{\|\mathbf r\|^2}{n-p}$.

### Examples for §13.7

**Example 13.7.1 (Gauss–Newton for exponential decay).** Drug concentration $c = 10.1, 6.2, 3.6, 2.3, 1.4, 0.8, 0.5$ at $t = 0,\dots,6$ h; model $c = Ae^{-kt}$. From $(A,k) = (8, 0.3)$:

| $k$ | $(A, k)$ | SSE |
|---|---|---|
| 0 | $(8, 0.3)$ | 8.6829 |
| 1 | $(9.9156, 0.47336)$ | 0.13681 |
| 2 | $(10.1122, 0.50025)$ | 0.022754 |
| 3 | $(10.11445, 0.50089)$ | 0.022694 |

Converged in 3 iterations (small residuals ⇒ fast convergence).

**Example 13.7.2 (Michaelis–Menten kinetics).** Reaction rates $V = 1.2, 1.9, 2.7, 3.3, 3.8, 4.0$ at substrate concentrations $S = 0.5, 1, 2, 4, 8, 16$; model $V = \frac{V_{\max}S}{K_m+S}$. LM from $(1,1)$: $\hat V_{\max} = 4.361$, $\hat K_m = 1.276$ in 9 iterations (SSE $0.0055$). (The classical Lineweaver–Burk linearisation distorts the errors, cf. §8.1.)

**Example 13.7.3 (logistic growth curve).** 15 cumulative-adoption observations, model $y = \frac{K}{1+e^{-r(t-t_0)}}$: LM gives $\hat K = 51.20$, $\hat r = 0.484$, $\hat t_0 = 6.97$ (the inflection time) in 9 iterations.

**Example 13.7.4 (starting values matter).** Same model:
* from $(10, 0.1, 1)$: Gauss–Newton diverges ($|\beta|\sim10^{47}$); LM stops at a poor local solution (SSE $4.6\times10^3$) with $r<0$;
* from $(100, 2, 20)$: Gauss–Newton diverges again; LM converges to the correct fit (SSE $0.948$).
LM is far more robust than GN, but nonlinear least squares is non-convex: use sensible starts (e.g. $K\approx\max y$, $t_0\approx$ time of half-maximum) and check the fit.

**Example 13.7.5 (standard errors).** For the decay fit: $s^2 = 0.022694/5$, $\text{se}(\hat A, \hat k) = (0.0629, 0.00567)$, 95% CI for $k$: $[0.4863, 0.5155]$; half-life $\ln2/\hat k = 1.384$ h.

---

## 13.8 Generalised Linear Models by IRLS

A GLM has $E[y_i] = \mu_i = g^{-1}(\mathbf x_i^T\boldsymbol\beta)$ with an exponential-family distribution. For canonical links the log-likelihood gradient is $X^T(\mathbf y-\boldsymbol\mu)$ and the Hessian $-X^TWX$ with $W = \operatorname{diag}(\operatorname{Var}(y_i))$:

| model | link | $\mu$ | $W_{ii}$ |
|---|---|---|---|
| logistic | logit | $\sigma(\eta) = 1/(1+e^{-\eta})$ | $\mu_i(1-\mu_i)$ |
| Poisson | log | $e^\eta$ | $\mu_i$ |

**Newton = IRLS:** with working response $\mathbf z = \boldsymbol\eta + W^{-1}(\mathbf y-\boldsymbol\mu)$,
$$
\boldsymbol\beta^{(k+1)} = (X^TWX)^{-1}X^TW\mathbf z,
$$
a weighted least-squares problem at each step (solve it by QR of $W^{1/2}X$). Standard errors: $\sqrt{\operatorname{diag}(X^TWX)^{-1}}$ at convergence (inverse Fisher information). An **offset** $\log(\text{exposure})$ models rates in Poisson regression.

### Examples for §13.8

**Example 13.8.1 (logistic regression — hours studied vs passing).** 20 students; IRLS from $\boldsymbol\beta = \mathbf 0$: $(-2.616, 0.938)$, $(-3.661, 1.339)$, $(-4.035, 1.487)$, $(-4.0772, 1.50445)$, $(-4.07771, 1.50465)$ — converged in 5–6 iterations (quadratic). $\hat{\boldsymbol\beta} = (-4.078, 1.505)$, standard errors $(1.761, 0.629)$. Each extra hour multiplies the odds of passing by $e^{1.505} = 4.50$; $P(\text{pass}\mid2\text{ h}) = 0.256$; the 50% point is $4.078/1.505 = 2.71$ hours.

**Example 13.8.2 (Poisson regression with exposure).** Event counts $2,3,6,7,8,9,10,12,15$ over exposures $10,12,15,14,13,12,11,12,13$ at $x = 0,\dots,8$; model $\log\mu_i = \log(\text{exposure}_i) + \beta_0 + \beta_1x_i$. IRLS (6 iterations): $\hat\beta_1 = 0.1973$ (se $0.0494$) — the event rate rises by a factor $e^{0.197} = 1.218$ (22%) per unit of $x$.

**Example 13.8.3 (separation — the MLE does not exist).** Data $x = 1..6$, $y = 0,0,0,1,1,1$ are perfectly separated at $x = 3.5$. IRLS slopes: $0, 1.03, 1.82, 2.88, 4.50, 6.54, 8.60, 10.6, 12.6, 14.6, 16.6$, then NaN; the negative log-likelihood decreases toward 0 ($4.16, 1.47, 0.83, 0.45, 0.20, 0.075, \ldots, 0.0005$) without attaining it. The numerical failure reveals a statistical one: use penalisation (ridge/Firth) or report separation.

**Example 13.8.4 (cross-check with a library).** `statsmodels.GLM(..., family=Binomial())` returns $(-4.07771, 1.50465)$ with standard errors $(1.76098, 0.62872)$ — identical to our IRLS.

**Example 13.8.5 (IRLS vs a general optimiser).** Minimising the logistic negative log-likelihood with BFGS (Chapter 15) gives the same $(-4.0777, 1.5047)$; IRLS uses the exact Hessian and needs 7 iterations, each a small weighted least-squares solve.

---

## Chapter summary

* $\hat{\boldsymbol\beta} = \arg\min\|\mathbf y - X\boldsymbol\beta\|^2$; statistics need $(X^TX)^{-1}$, but compute through QR/SVD.
* Diagnose: residual plots, leverage $h_{ii}$, VIF, condition numbers; standardise predictors.
* Complexity by CV/AIC/BIC; regularise with ridge (SVD filter) or lasso (soft-thresholding, sparsity).
* Non-constant or correlated noise: WLS/GLS (Cholesky whitening); outliers: Huber/LAD by IRLS.
* Nonlinear models: Gauss–Newton (fast, fragile) and Levenberg–Marquardt (robust); GLMs: IRLS = Newton.

## Further reading

Montgomery, Peck & Vining, *Introduction to Linear Regression Analysis* · Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Ch. 3 · Björck, *Numerical Methods for Least Squares Problems* (SIAM) · McCullagh & Nelder, *Generalized Linear Models* · Friedman, Hastie & Tibshirani, "Regularization paths for GLMs via coordinate descent", *J. Stat. Software* 33 (2010).
