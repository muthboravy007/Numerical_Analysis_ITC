# P2 — A Regression Engine from Scratch: Stable Least Squares, Regularisation and Inference

**Chapters:** 6 (direct methods), 7 (conditioning, CG), 13 (regression) · **Difficulty:** ★★ · **Duration:** 6–8 weeks

## Question
Build a small but *numerically trustworthy* regression library, the core of statsmodels or R's `lm`. Show when naive implementations silently give wrong answers, and apply the library to a real prediction-and-inference problem.

## Mathematical background
- **OLS by normal equations, Householder QR and SVD.** Forward-error bounds: normal equations have error of order $\kappa(X)^2u$, while QR has error of order $\kappa(X)u+\kappa(X)^2u\,\frac{\|\mathbf r\|}{\|X\|\|\boldsymbol\beta\|}$.
- **Inference.** $\operatorname{Cov}\hat{\boldsymbol\beta} = \sigma^2(X^TX)^{-1} = \sigma^2R^{-1}R^{-T}$; $t$- and $F$-tests; confidence and prediction intervals; leverage $h_{ii} = \|Q_{i,:}\|^2$; Cook's distance; the sandwich (HC0–HC3) estimators.
- **Regularisation.** Ridge via the augmented QR or the SVD, with the GCV formula. Lasso via coordinate descent, with warm starts along the $\lambda$ path and KKT checks.
- **Iterative solvers** for large sparse designs: LSQR or CG on the normal equations (CGLS). Their convergence depends on $\kappa(X)$.

## Required tasks
1. Implement `fit(X, y, method={"normal","qr","svd"})` (Householder QR *yourself*, not `np.linalg.qr`), with a summary table (coefficients, SE, $t$, $p$, $R^2$, adjusted $R^2$, $F$).
2. **Accuracy benchmark:** use the NIST StRD linear regression datasets (Norris, Longley, Filip, Wampler1–5). Report the log relative error (LRE) of every method against the certified values. Explain the Filip failure of the normal equations with condition numbers.
3. **Diagnostics:** leverage, studentised residuals, Cook's distance, VIF, and a residuals-vs-fitted plot. Heteroscedasticity-robust standard errors; show their coverage by simulation.
4. **Regularisation:** ridge (with GCV/LOOCV in closed form) and lasso (coordinate descent with a path). Validate against scikit-learn to $10^{-8}$.
5. **Real data:** the Ames housing data (or the diabetes data for a smaller version). Model log(price) with feature engineering (splines or polynomial terms, categorical dummies). Compare OLS, ridge and lasso by CV, and interpret a final model with valid inference.

## Going further
- CGLS or LSQR for a sparse design with $10^5$ rows (one-hot features). Compare with QR on time and memory.
- The bootstrap (pairs and residual) for CIs; compare its coverage with the analytic intervals.
- Updating or downdating QR when rows are added (Givens rotations) for streaming data.

## Weekly milestones
| Week | Goal |
|---|---|
| 1 | Proposal; run starter (QR-based OLS + NIST Longley check) |
| 2 | Householder QR implementation + tests (orthogonality $\|Q^TQ-I\|$, backward error) |
| 3 | Inference tables; agreement with statsmodels |
| 4 | NIST benchmark (LRE table); progress report |
| 5 | Diagnostics + robust SEs, coverage simulation |
| 6 | Ridge/lasso + CV; real-data modelling |
| 7 | Extension |
| 8 | Report + talk |

## Data
- NIST StRD: <https://www.itl.nist.gov/div898/strd/lls/lls.shtml> (certified values to 15 digits).
- Ames housing: `sklearn.datasets.fetch_openml(name="house_prices")`, or <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>.
- Offline fallback: `sklearn.datasets.load_diabetes` and `statsmodels.datasets.longley`.

## Project-specific rubric additions
- The Householder QR is correct and stable: $\|Q^TQ-I\|\approx u$, with the backward error demonstrated.
- The LRE table and its explanation connect to $\kappa(X)$ and $\kappa(X)^2$.
- Inference is validated by simulated coverage (about 95% for nominal 95% intervals).

## Starter code
[`starter/p2_regression_engine.py`](starter/p2_regression_engine.py) contains an OLS class with QR, an inference table, a Longley LRE check, and a TODO for your own Householder QR.

## References
- Björck, *Numerical Methods for Least Squares Problems*, SIAM (1996).
- McCullough, "Assessing the reliability of statistical software", *Am. Statistician* 52 (1998).
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, ch. 3.
