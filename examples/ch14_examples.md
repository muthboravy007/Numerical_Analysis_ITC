# Chapter 14 — Worked Examples: Symmetric Matrices, the SVD and PCA

Companion script: [`ch14_examples.py`](ch14_examples.py) reproduces every number below. · Lecture: [Chapter 14](../lectures/ch14-symmetric-matrices-svd-pca.md)

---

## Example 14.1 — Risk factors hidden in a correlation matrix

**Problem.** Three stocks have correlation matrix $R = \begin{pmatrix}1&0.6&0.5\\0.6&1&0.7\\0.5&0.7&1\end{pmatrix}$. Interpret its spectral decomposition.

**Solution.** The eigenvalues are $2.204, 0.514, 0.283$, carrying 73.5%, 17.1% and 9.4% of the total variance. The eigenvectors are:
- $\mathbf q_1 = (0.54, 0.61, 0.58)$: all stocks move together. This is the **market factor**.
- $\mathbf q_2 = (-0.81, 0.18, 0.56)$: stock 1 against stock 3. This is a sector or style contrast.
- $\mathbf q_3 = (0.23, -0.77, 0.59)$: the least risky combination.

The decomposition $R = \sum\lambda_i\mathbf q_i\mathbf q_i^T$ is verified numerically. With unit volatilities the equal-weight portfolio has variance $0.733$. Among unit-norm portfolios, the minimum variance is $\lambda_{\min} = 0.283$, attained along $\mathbf q_3$ (Rayleigh quotient).

**Take-away.** The spectral theorem turns a covariance matrix into independent risk factors. Its largest eigenvalue is the dominant common risk, and its smallest eigenvalues identify hedges. They also show where estimation noise is most dangerous: a nearly singular covariance leads to extreme "optimal" portfolios.

---

## Example 14.2 — The 95% confidence ellipse

**Problem.** For $\mathbf X\sim N(\mathbf 0,\Sigma)$ with $\Sigma = \begin{pmatrix}4&1.5\\1.5&1\end{pmatrix}$, find the region $\{\mathbf x:\mathbf x^T\Sigma^{-1}\mathbf x\le\chi^2_{2,0.95}\}$.

**Solution.** $\chi^2_{2,0.95} = 5.991$. The eigenvalues of $\Sigma$ are $4.621$ and $0.379$, so the semi-axes are $\sqrt{5.991\lambda_i} = 5.262$ and $1.506$. The major axis points along $(0.924, 0.383)$, i.e. at $22.5°$. The area is $\pi\sqrt{\det\Sigma}\,\chi^2 = 24.90$. In $10^5$ simulated points the coverage is $0.9501$.

**Take-away.** The quadratic form $\mathbf x^T\Sigma^{-1}\mathbf x$ (squared Mahalanobis distance) describes an ellipse whose axes are the eigenvectors of $\Sigma$, with lengths $\propto\sqrt{\lambda_i}$. This geometry underlies confidence regions for regression coefficients, anomaly detection and whitening.

---

## Example 14.3 — What the SVD finds in a ratings matrix

**Problem.** Four users rate three movies (action 1, action 2, romance):
$M = \begin{pmatrix}5&5&0\\4&5&1\\1&0&5\\0&1&4\end{pmatrix}$. Interpret the SVD and the best rank-2 approximation.

**Solution.** The singular values are $9.74, 6.24, 1.12$, carrying 70.2%, 28.9% and 0.9% of the energy. The movie factors are:
- $\mathbf v_1 = (0.65, 0.72, 0.23)$: general "likes movies".
- $\mathbf v_2 = (0.15, 0.18, -0.97)$: action vs romance taste.

In the user factors $U$, users 1–2 load positively on factor 2 (action fans) and users 3–4 negatively (romance fans). The rank-2 approximation $\begin{pmatrix}4.73&5.25&0.01\\4.29&4.74&0.99\\0.51&0.44&5.01\\0.55&0.51&3.99\end{pmatrix}$ has error $\|M-M_2\|_F = 1.125 = \sigma_3$ (Eckart–Young).

**Take-away.** The SVD discovers latent "taste" dimensions without being told the genres. This is the core of latent-factor recommenders (project P1).

---

## Example 14.4 — The condition number predicts sensitivity

**Problem.** Fit a degree-5 polynomial in the monomial basis to 20 points on $[0,1]$. Perturb $\mathbf y$ by a relative amount of $4\times10^{-7}$. How much do the coefficients change?

**Solution.** The singular values of $X$ range from $5.79$ to $0.0018$, so $\kappa(X) = 3.2\times10^3$. The coefficients change by a relative $4.4\times10^{-5}$, an amplification of about 100, within the bound $\kappa$. The *fitted values* change by only $1.9\times10^{-7}$.

**Take-away.** Ill-conditioning makes individual coefficients unstable. Predictions can still be stable, because they depend on well-determined combinations (the large singular directions). Interpret coefficients only when $\kappa$ is moderate, or use an orthogonal basis.

---

## Example 14.5 — PCA depends on the units

**Problem.** A data set has height (cm), weight (kg) and income (\$) for 300 people, with height and weight correlated. Compare PCA on the raw data and on standardised data.

**Solution.**

| data | variance shares | PC1 loadings (height, weight, income) |
|---|---|---|
| raw | $(1.000, 0.000, 0.000)$ | $(0, 0, 1)$ |
| standardised | $(0.575, 0.333, 0.092)$ | $(0.71, 0.71, -0.04)$ |

**Take-away.** On raw data PCA simply finds "income", the variable with the largest numbers: its variance is about $10^8$ against about $10^2$ for the others. After standardising (PCA of the correlation matrix), PC1 is the meaningful "body size" factor. Always standardise variables measured in different units.

---

## Example 14.6 — More features than samples

**Problem.** $n = 20$ samples and $p = 100$ features, with only 5 nonzero true coefficients. Least squares has infinitely many exact solutions. Which one does $X^+\mathbf y$ pick, and how does it relate to ridge?

**Solution.** $\operatorname{rank}X = 20$. The pseudoinverse gives the **minimum-norm** interpolant: training residual $2\times10^{-14}$ and $\|\mathbf b\| = 2.04$, much smaller than $\|\mathbf b_{true}\| = 4.36$. As $\lambda\to0$ ridge converges to it: $\|\mathbf b_\lambda-\mathbf b^+\| = 3\times10^{-3}, 3\times10^{-5}, 3\times10^{-8}$ for $\lambda = 10^{-1}, 10^{-3}, 10^{-6}$. The test RMSE is $3.85$, against $4.36$ for predicting 0, and the correlation between the estimated and true coefficients is only $0.47$.

**Take-away.** With $p>n$, interpolating the training data is easy and says nothing about generalisation. The minimum-norm solution is the implicit choice of gradient descent started from zero, and it spreads the signal over all features. Sparse structure calls for the lasso (Chapter 13), and prediction calls for honest regularisation and validation.
