# P5 — Seasonality and Forecasting with Fourier Analysis and Splines

**Chapters:** 3 (splines), 8 (trigonometric LS, FFT), 13 (regression, CV), 11 (smoothing as a BVP) · **Difficulty:** ★★ · **Duration:** 6–8 weeks

## Question
Decompose a real time series into trend, seasonal cycles and noise using numerical-analysis tools, and build a forecast. How do Fourier regression, smoothing splines and the FFT compare with standard statistical forecasting baselines?

## Mathematical background
- **Discrete Fourier transform and FFT:** Cooley–Tukey complexity; aliasing and the Nyquist frequency; leakage and windowing (Hann); the periodogram as an estimate of spectral density.
- **Harmonic regression:** $y_t = T(t)+\sum_{k=1}^K[a_k\cos\frac{2\pi kt}{P}+b_k\sin\frac{2\pi kt}{P}]+\varepsilon_t$. Orthogonality on equispaced complete cycles makes the least-squares normal equations diagonal (Chapter 8). Multiple seasonalities (daily, weekly, yearly) are handled the same way.
- **Trend:** cubic regression splines (B-spline basis) or penalised smoothing splines, $\min\sum(y_i-f(t_i))^2+\lambda\int f''^2$. The discrete analogue is the Whittaker smoother $(I+\lambda D_2^TD_2)\mathbf z = \mathbf y$, solved in $O(n)$ with a banded Cholesky factorisation. Choose $\lambda$ by GCV.
- **Forecast evaluation:** rolling-origin CV; MAE, RMSE and MASE; prediction intervals from the residual bootstrap.

## Required tasks
1. Implement the radix-2 FFT (or Bluestein for arbitrary $n$) and a periodogram with windowing. Verify against `numpy.fft` and with synthetic signals, showing leakage with and without the window.
2. Implement harmonic regression with a spline trend, selecting $K$ by AIC/BIC/CV.
3. Implement the Whittaker smoother with a banded solver (your own banded Cholesky or `scipy.linalg.solveh_banded`) and GCV. Show $O(n)$ scaling up to $n = 10^6$.
4. **Data set A (Mauna Loa CO₂, monthly 1958–today):** decomposition, seasonal amplitude over time (is it growing?), and 5-year-ahead forecasts with intervals.
5. **Data set B (hourly electricity load or bike-share counts):** multiple seasonalities, detected via the periodogram. Compare with seasonal-naive and ETS/ARIMA baselines (`statsmodels`).

## Going further
- Short-time Fourier transform (spectrogram) to detect changing periodicity.
- Temperature as an exogenous regressor, with splines for the non-linear effect.
- The Lomb–Scargle periodogram for unevenly sampled data.

## Weekly milestones
| Week | Goal |
|---|---|
| 1 | Proposal; starter: CO₂ periodogram + harmonic fit |
| 2 | Derivations: DFT orthogonality, leakage, spline smoothing Euler–Lagrange, GCV |
| 3 | FFT + windowed periodogram + tests |
| 4 | Harmonic regression + spline trend; progress report |
| 5 | Whittaker smoother, banded solver, scaling |
| 6 | Data set B, baselines |
| 7 | Rolling-origin evaluation, intervals, extension |
| 8 | Report + talk |

## Data
- CO₂: <https://gml.noaa.gov/ccgg/trends/data.html>. The offline copy is `statsmodels.datasets.co2`, weekly 1958–2001.
- Electricity: UCI "ElectricityLoadDiagrams20112014", or the PJM hourly load on Kaggle.
- Bike sharing: UCI "Bike Sharing Dataset" (hourly counts with weather).

## Project-specific rubric additions
- The FFT and the periodogram are implemented and validated, and leakage is demonstrated.
- The smoothing parameter is chosen in a principled way (GCV or CV), and the banded solver is shown to scale as $O(n)$.
- Forecasts are evaluated honestly against strong baselines.

## Starter code
[`starter/p5_fourier_forecasting.py`](starter/p5_fourier_forecasting.py) contains the CO₂ data loader, a periodogram, a harmonic-regression forecast with a holdout RMSE, and TODOs for your own FFT and spline trend.

## References
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. (free online), ch. 7, 12.
- Eilers, "A perfect smoother", *Analytical Chemistry* 75 (2003).
- Brigham, *The Fast Fourier Transform and Its Applications*.
