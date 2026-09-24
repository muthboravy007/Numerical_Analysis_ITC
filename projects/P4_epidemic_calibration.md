# P4 — Epidemic Modelling: ODE Simulation, Calibration and Forecasting

**Chapters:** 5 (IVPs, stiffness, adaptive RK), 10 (Newton/Gauss–Newton), 13 (nonlinear least squares, inference) · **Difficulty:** ★★★ · **Duration:** 6–8 weeks

## Question
Can a compartmental ODE model, fitted to early case data, forecast the course of an epidemic? How uncertain are its parameters, and which of them can the data actually identify?

## Mathematical background
- **SEIR model:** $S' = -\beta SI/N$, $E' = \beta SI/N-\sigma E$, $I' = \sigma E-\gamma I$, $R' = \gamma I$. Key quantities: $R_0 = \beta/\gamma$; the final-size relation $\ln\frac{S_\infty}{S_0} = -R_0\bigl(1-\frac{S_\infty}N\bigr)$; the early exponential growth rate, obtained from the linearisation (eigenvalues of the Jacobian at the disease-free equilibrium).
- **Numerics:**
  - RK4 and embedded RK45 with error control (implement your own);
  - convergence order checks;
  - when time-varying contact rates $\beta(t)$ make the problem non-smooth.
- **Calibration:** $\min_{\boldsymbol\theta}\sum_t(\log y_t-\log\hat y_t(\boldsymbol\theta))^2$, or a Poisson or negative-binomial likelihood. Sensitivities $\partial\mathbf x/\partial\boldsymbol\theta$ come from the **variational (sensitivity) equations** $\mathbf s' = J_f\mathbf s+\partial f/\partial\boldsymbol\theta$. They give the Jacobian for Gauss–Newton or Levenberg–Marquardt; compare with finite differences.
- **Identifiability:** the singular values of the sensitivity matrix; profile likelihoods; the correlation of $(\beta,\gamma)$ in early data (only $\beta-\gamma$ is identified).

## Required tasks
1. Implement RK45 with adaptive steps. Verify its order on a problem with a known solution, and compare with `solve_ivp`.
2. Derive and implement the sensitivity equations for SEIR. Check them against finite differences.
3. **Synthetic study:** simulate with known parameters and Poisson noise. Fit using only the first 20, 40 or 60 days. Report the estimates, standard errors (from $(J^TJ)^{-1}$ and from a parametric bootstrap), and forecast intervals.
4. **Identifiability:** the singular values of the sensitivity matrix over time, a profile likelihood for $R_0$, and a demonstration that early data identify the growth rate but not $R_0$.
5. **Real data:** the daily cases or deaths of one country's first wave (Our World in Data). Model an intervention as a piecewise-constant $\beta(t)$ with a change point, and estimate the change point by profiling. Discuss under-reporting.

## Going further
- An age-structured model (contact matrices). The next-generation matrix gives $R_0$ as a spectral radius (Chapter 9).
- Stochastic simulation (Gillespie) against the ODE; extinction probability.
- Optimal vaccination allocation as an optimisation problem (Chapter 15).

## Weekly milestones
| Week | Goal |
|---|---|
| 1 | Proposal; starter SEIR simulation + least-squares fit |
| 2 | Derivations: $R_0$, final size, linear growth rate, sensitivity equations |
| 3 | RK45 + sensitivities with tests |
| 4 | Synthetic calibration study; progress report |
| 5 | Identifiability analysis |
| 6 | Real-data fit with change point |
| 7 | Forecast evaluation (rolling origin), extension |
| 8 | Report + talk |

## Data
- Our World in Data COVID-19: <https://github.com/owid/covid-19-data> (`owid-covid-data.csv`).
- Historical: the 1978 boarding-school influenza outbreak (763 pupils, a classic SIR data set, printed in Murray, *Mathematical Biology*).
- The starter generates synthetic data offline.

## Project-specific rubric additions
- The sensitivity equations are correct, validated, and used in Gauss–Newton.
- Uncertainty is quantified honestly, with forecast intervals whose coverage is checked on synthetic data.
- There is a clear discussion of identifiability and model misspecification.

## Starter code
[`starter/p4_epidemic.py`](starter/p4_epidemic.py) contains the SEIR right-hand side, synthetic Poisson data, a fit with `scipy.optimize.least_squares` (finite-difference Jacobian), and TODOs for sensitivities and your own RK45.

## References
- Keeling & Rohani, *Modeling Infectious Diseases in Humans and Animals*, Princeton (2008).
- Chowell, "Fitting dynamic models to epidemic outbreaks with quantified uncertainty", *Infectious Disease Modelling* 2 (2017).
- Raue et al., "Structural and practical identifiability analysis ... profile likelihood", *Bioinformatics* 25 (2009).
