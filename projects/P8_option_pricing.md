# P8 — Computational Finance: Option Pricing with PDEs, Monte Carlo and Root Finding

**Chapters:** 2 (root finding), 4 (quadrature, Monte Carlo), 5 (SDE/ODE time stepping), 12 (parabolic PDEs, Crank–Nicolson) · **Difficulty:** ★★★ · **Duration:** 6–8 weeks

## Question
Price European, American and exotic options with three independent numerical approaches, verify that they agree, and calibrate the model to market prices through implied volatilities. Where does each method win?

## Mathematical background
- **Black–Scholes:** $dS = rS\,dt+\sigma S\,dW$ under the risk-neutral measure. The price solves
$$V_t+\tfrac12\sigma^2S^2V_{SS}+rSV_S-rV = 0,\qquad V(S,T) = \text{payoff},$$
  and the closed form exists for European options. The change of variables $x = \ln S$ gives a constant-coefficient heat equation.
- **Finite differences:**
  - explicit (with its stability condition), implicit, and Crank–Nicolson (Rannacher start for non-smooth payoffs);
  - boundary conditions and grid truncation;
  - convergence $O(\Delta S^2+\Delta t^2)$ for CN;
  - Greeks read off the grid.
- **American options:** the linear complementarity problem $\min(-\mathcal LV,\ V-\text{payoff}) = 0$, solved by projected SOR (Chapter 7) or a penalty method. The free exercise boundary.
- **Monte Carlo:**
  - $O(N^{-1/2})$ error with confidence intervals;
  - variance reduction: antithetic variates and control variates;
  - quasi-Monte Carlo (Sobol sequences);
  - path-dependent options (Asian), where no simple PDE exists;
  - Euler–Maruyama time stepping of SDEs: strong and weak orders.
- **Quadrature:** the price as an integral against the log-normal density, computed with Gauss–Hermite quadrature (Chapter 4).
- **Implied volatility:** solve $BS(\sigma) = V_{mkt}$ with Newton's method (vega), with a bisection safeguard. Existence and uniqueness follow from monotonicity in $\sigma$.

## Required tasks
1. **European options, three ways:** CN finite differences, Monte Carlo (with antithetic and control variates), and Gauss–Hermite quadrature. Show convergence plots against the closed form, including the $O(h^2)$ vs $O(N^{-1/2})$ behaviour, and a work–precision diagram.
2. **Stability:** demonstrate the explicit-scheme instability, and CN oscillations with the digital (step) payoff. Fix them with Rannacher smoothing.
3. **American put:** PSOR and a penalty method. Compare with a binomial tree (CRR), and plot the exercise boundary.
4. **Asian option** by Monte Carlo with a geometric-Asian control variate, which has a closed form. Report the variance reduction factor, and compare with Sobol QMC.
5. **Implied volatility and calibration:** a safeguarded Newton solver. Build the implied-volatility smile from real option quotes. Fit a simple parametric smile (e.g. quadratic in log-moneyness, or SVI) by nonlinear least squares. Discuss why Black–Scholes cannot explain the smile.

## Going further
- The Heston stochastic-volatility model: pricing by Fourier inversion (FFT, Carr–Madan; Chapter 8) and calibration to the smile.
- The ADI method for two-asset (2-D) options.
- Pathwise and likelihood-ratio Greeks in Monte Carlo.

## Weekly milestones
| Week | Goal |
|---|---|
| 1 | Proposal; starter: BS closed form, CN European put, MC |
| 2 | Derivations: BS PDE, transformation to heat eq., CN stability, LCP |
| 3 | CN + MC + quadrature with convergence tests |
| 4 | Stability experiments; progress report |
| 5 | American options (PSOR, penalty, CRR) |
| 6 | Asian/MC variance reduction, QMC |
| 7 | Implied vol + calibration on real data; extension |
| 8 | Report + talk |

## Data
- Public option-chain snapshots via `yfinance` (e.g. SPY), or CBOE delayed quotes. Save a snapshot for reproducibility.
- All pricing experiments work offline with simulated parameters.

## Project-specific rubric additions
- The three pricing methods agree with each other and with the closed form, and their convergence orders are verified.
- The American-option solver is validated against a binomial tree.
- The implied-volatility solver is robust (it handles deep in-the-money and out-of-the-money options), and the smile is analysed critically.

## Starter code
[`starter/p8_option_pricing.py`](starter/p8_option_pricing.py) contains the closed-form Black–Scholes price, CN for a European put, plain Monte Carlo, and a Newton implied-volatility solver, with TODOs for PSOR, variance reduction and calibration.

## References
- Hull, *Options, Futures, and Other Derivatives*, 10th ed.
- Wilmott, Howison & Dewynne, *The Mathematics of Financial Derivatives*, Cambridge (1995).
- Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer (2003).
