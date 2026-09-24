# Chapter 13 — Worked Examples: Regression and Least Squares

Companion script: [`ch13_examples.py`](ch13_examples.py) reproduces every number below. · Lecture: [Chapter 13](../lectures/ch13-regression-least-squares.md)

---

## Example 13.1 — A complete simple-regression analysis: TV advertising and sales

**Problem.** Twenty markets report TV advertising spend (thousands) and sales (thousands of units). Fit $\text{sales} = \beta_0+\beta_1\,\text{TV}$, test the slope, and give confidence and prediction intervals at TV $= 100$.

**Solution.** $\hat\beta_0 = 6.850$ and $\hat\beta_1 = 0.05569$, with $SE = 0.00769$, $t = 7.24$ (df 18), $R^2 = 0.745$ and $s = 2.82$. The 95% CI for the slope is $[0.0395, 0.0718]$: each extra 100 units of spend is associated with about 5.6 (4.0–7.2) extra units of sales. At TV $= 100$ the fitted mean is $12.42$. Its confidence interval is $[11.06, 13.78]$, and the prediction interval for a single new market is $[6.34, 18.50]$. The correlation between $|e_i|$ and TV is $0.39$: the spread grows with spend (heteroscedasticity), so robust SEs or a log model deserve a look.

**Take-away.** Report the effect size with uncertainty, distinguish predicting a *mean* from predicting an *individual*, and check the model's assumptions with the residuals.

---

## Example 13.2 — Omitted-variable bias

**Problem.** Wages depend on education and (unobserved) ability: $\text{wage} = 5+1\cdot\text{educ}+3\cdot\text{ability}+\varepsilon$, and education is correlated with ability. What happens if ability is left out?

**Solution.** With $n = 500$ simulated workers, the short regression $\text{wage}\sim\text{educ}$ gives slope $2.239$ ($SE$ 0.033), far from the true $1$ and very "significant". Including ability gives $(0.973, 3.079)$. The bias formula $\beta_{\text{ability}}\cdot\frac{\operatorname{cov}(\text{educ},\text{ability})}{\operatorname{var}(\text{educ})} = 3\times0.411 = 1.234$ matches the observed bias of $1.239$.

**Take-away.** Least squares is numerically exact but *causally* naive. A small standard error does not protect against a misspecified model. The bias formula follows from the normal equations of the partitioned regression.

---

## Example 13.3 — Bias and variance of polynomial regression

**Problem.** Draw 25 noisy points ($\sigma = 0.3$) from $\sin2\pi x$ on $[0,1]$, repeated 200 times. For each polynomial degree, estimate the squared bias and the variance of the fitted curve.

**Solution.**

| degree | bias² | variance | expected test MSE ($+\sigma^2$) |
|---|---|---|---|
| 1 | 0.200 | 0.027 | 0.317 |
| **3** | **0.005** | **0.019** | **0.114** |
| 5 | 0.001 | 0.088 | 0.179 |
| 7 | 0.014 | 3.4 | 3.5 |
| 9 | 1.4 | 473 | 474 |

**Take-away.** Bias falls and variance rises with model complexity. With only 25 randomly placed points, high-degree fits explode near the edges, where data are sparse: occasional samples have gaps, and the fit extrapolates wildly. That is why cross-validation or regularisation is essential. Degree 3 is optimal here.

---

## Example 13.4 — The ridge path with nearly collinear predictors

**Problem.** Ten predictors, where $x_2 = x_1+$ small noise, and true coefficients $(1, 1, 0.5, 0, \dots, -0.5, 0)$. Trace the ridge solution $\hat{\boldsymbol\beta}_\lambda = V\operatorname{diag}\bigl(\frac{\sigma_i}{\sigma_i^2+\lambda}\bigr)U^T\mathbf y$.

**Solution.** The singular values run from $11.0$ down to $\mathbf{0.23}$, the collinear direction, so $\kappa = 47.5$.

| $\lambda$ | df | $(\hat\beta_1,\hat\beta_2)$ | $\|\hat{\boldsymbol\beta}\|$ |
|---|---|---|---|
| 0 (OLS) | 10 | $(1.846, 0.175)$ | 2.00 |
| 0.1 | 9.33 | $(1.302, 0.715)$ | 1.67 |
| 1 | 8.84 | $(1.039, 0.951)$ | 1.59 |
| 10 | 7.36 | $(0.898, 0.886)$ | 1.40 |
| 100 | 3.08 | $(0.487, 0.485)$ | 0.74 |

**Take-away.** OLS splits the shared effect arbitrarily ($1.85$ vs $0.18$). A small ridge penalty first shrinks the tiny-$\sigma$ direction ($\beta_1-\beta_2$) and restores the sensible split (about 1, 1), long before it shrinks everything else. The effective degrees of freedom quantify the model complexity.

---

## Example 13.5 — Weighted least squares for averaged data

**Problem.** Only group means are available, for 8 groups whose sizes alternate between 5 and 50. The mean of a group of size $m$ has variance $\sigma^2/m$. Fit the trend by OLS and by WLS with weights $m$.

**Solution.** On one data set, OLS gives $(1.944, 0.498)$ and WLS gives $(2.097, 0.477)$ (truth $(2, 0.5)$). Over 2000 repetitions the slope SD is $0.052$ for OLS and **$0.029$** for WLS, a variance reduction of $3.2\times$.

**Take-away.** When observations have known, unequal precision (survey strata, aggregated data, replicate measurements), weight by the inverse variance. WLS is OLS on rows scaled by $\sqrt{w_i}$, so it is as cheap and as stable.

---

## Example 13.6 — Enzyme kinetics: don't linearise the noise

**Problem.** The Michaelis–Menten model is $v = \frac{V_{\max}S}{K_m+S}$ (truth $V_{\max} = 10$, $K_m = 3$), with 5% multiplicative noise at $S = 0.5,\dots,32$. Compare the classic Lineweaver–Burk linearisation ($1/v$ against $1/S$) with nonlinear least squares by Gauss–Newton.

**Solution.**

| method | $V_{\max}$ | $K_m$ | SSE |
|---|---|---|---|
| Lineweaver–Burk | 10.62 | 3.61 | 0.764 |
| Gauss–Newton (5 iterations, started from the linearised estimate) | **9.86** | **3.03** | **0.255** |

**Take-away.** Taking reciprocals hugely inflates the noise of the small-$S$ measurements, which then dominate the fit. The linearised fit is still an excellent *starting point* for nonlinear least squares.

---

## Example 13.7 — Poisson regression for counts

**Problem.** Counts satisfy $y_i\sim\text{Poisson}(e^{0.5+0.8x_i})$ for 300 observations with $x\in[0,2]$. Fit the log-linear model by IRLS.

**Solution.** From $\boldsymbol\beta = \mathbf 0$ the iterates are $(0.026, 3.033)\to(-0.477, 2.767)\to(-0.447, 2.223)\to(0.050, 1.480)\to(0.413, 0.966)\to\cdots$. After 10 iterations they converge to $\hat{\boldsymbol\beta} = (0.515, 0.791)$ with SE $(0.070, 0.054)$. The rate ratio is $e^{0.791} = 2.21$ per unit of $x$. With an intercept in the model, the score equation forces the mean of the fitted values to equal the mean of $y$: both are $3.850$.

**Take-away.** IRLS (Newton's method) starts far away, because $\boldsymbol\beta = \mathbf 0$ is a poor guess, overshoots, and then converges quadratically. Poisson regression models counts and rates without transforming $y$, and its coefficients are log rate ratios.
