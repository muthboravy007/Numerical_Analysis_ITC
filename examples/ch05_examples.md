# Chapter 5 — Worked Examples: Initial-Value Problems

Companion script: [`ch05_examples.py`](ch05_examples.py) reproduces every number below. · Lecture: [Chapter 5](../lectures/ch05-initial-value-problems.md)

---

## Example 5.1 — Logistic growth of an app's user base

**Problem.** Users (in thousands) follow $y' = 0.8\,y(1-y/100)$ with $y(0) = 1$. Compute $y(10)$ with Euler and RK4 for $h = 1, 0.5, 0.25$, and compare with the exact solution $y = \frac{100}{1+99e^{-0.8t}}$.

**Solution.** The exact value is $y(10) = 96.785670$, and the inflection point (fastest growth) is at $t = \frac{\ln99}{0.8} = 5.74$.

| $h$ | Euler | error | RK4 | error |
|---|---|---|---|---|
| 1 | 94.2326 | 2.55 | 96.743652 | $4.2\times10^{-2}$ |
| 0.5 | 95.9079 | 0.88 | 96.783118 | $2.6\times10^{-3}$ |
| 0.25 | 96.3952 | 0.39 | 96.785508 | $1.6\times10^{-4}$ |

Halving $h$ divides the Euler error by about 2–3 (asymptotically 2, order 1) and the RK4 error by about 16 (order 4).

**Take-away.** For the same cost (4 Euler steps ≈ 1 RK4 step), RK4 is thousands of times more accurate on smooth problems.

---

## Example 5.2 — Repeated drug dosing

**Problem.** A drug is eliminated by $C' = -0.2C$ (per hour), with volume $V = 20$ L. A 100 mg bolus is given every 8 h. Simulate 8 doses with RK4 ($h = 0.1$) and compare the peaks and troughs with the steady-state theory.

**Solution.** Each dose adds $D/V = 5$ mg/L, and the concentration then decays for 8 h.

| dose | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| peak | 5.000 | 6.010 | 6.213 | 6.254 | 6.263 | 6.264 | 6.265 | 6.265 |
| trough | 1.010 | 1.213 | 1.254 | 1.263 | 1.264 | 1.265 | 1.265 | 1.265 |

Theory: the steady-state peak is $\frac{D/V}{1-e^{-k\tau}} = 6.2649$ and the trough is $1.2649$. The accumulation factor is $1/(1-e^{-1.6}) = 1.253$. Steady state is reached within about 5 doses, i.e. about 5 half-lives.

**Take-away.** ODE models with *events* (doses, interventions) are integrated piece by piece, restarting at each event. This is how pharmacometrics software and epidemic-intervention models work.

---

## Example 5.3 — Adaptive step size for a fast transient

**Problem.** Solve $y' = -50(y-\cos t)$, $y(0) = 0$, on $[0,2]$ with RKF45 (tolerance $10^{-6}$).

**Solution.** RKF45 takes 274 accepted steps. On $[0,0.1]$ the steps are $0.001$–$0.0034$ (resolving the $e^{-50t}$ transient). On $[1,2]$ they average $0.0106$. The maximum error is $2.8\times10^{-8}$. Fixed-step RK4 with the same number of steps ($h = 0.0073$) has maximum error $7.4\times10^{-5}$, about 2700 times worse, because it wastes steps where the solution is smooth and is too coarse in the transient.

**Take-away.** Error control puts the work where the solution changes. The step size on $[1,2]$ is still limited by *stability* ($h|\lambda|\lesssim3$ for RK45), which is a first hint of stiffness.

---

## Example 5.4 — Predator–prey dynamics and a conserved quantity

**Problem.** Lotka–Volterra with $a = 1$, $b = 0.1$, $c = 1.5$, $d = 0.075$ from $(10, 5)$ conserves $H = dx-c\ln x+by-a\ln y$. Integrate to $t = 50$ with $h = 0.01$ and measure the drift in $H$.

**Solution.**

| method | relative drift of $H$ | final (prey, predator) |
|---|---|---|
| Euler | $1.3\times10^{-1}$ | $(4.93, 8.04)$ — wrong orbit |
| Heun | $1.9\times10^{-6}$ | $(15.03, 3.36)$ |
| RK4 | $2.7\times10^{-10}$ | $(15.02, 3.36)$ |

The prey peaks at $40.6$ at $t = 2.49$. The equilibrium is $(c/d, a/b) = (20, 10)$.

**Take-away.** Euler spirals outwards: it systematically adds "energy", so long-time behaviour is qualitatively wrong. Checking conserved quantities is a powerful diagnostic. Symplectic integrators preserve them by design.

---

## Example 5.5 — Stiffness: explicit vs implicit Euler

**Problem.** $y' = -1000(y-\sin t)+\cos t$, $y(0) = 1$. The exact solution quickly approaches $\sin t$, so $y(1)\approx0.841471$.

**Solution.**

| $h$ | explicit Euler $y(1)$ | implicit Euler $y(1)$ |
|---|---|---|
| 0.001 | 0.841471 | 0.841471 |
| 0.0021 | $-1.2\times10^{19}$ | 0.841470 |
| 0.01 | $2.7\times10^{95}$ | 0.841467 |
| 0.1 | $9.0\times10^{19}$ | 0.841430 |

Explicit Euler needs $|1-1000h|\le1$, i.e. $h\le0.002$. Just past the limit it explodes. Implicit Euler ($|1/(1+1000h)|<1$ for all $h$) is stable and accurate even with 10 steps.

**Take-away.** The solution is smooth, but the *equation* has a fast mode. That is stiffness. Use implicit (BDF, Radau) solvers for chemical kinetics, circuit models and many pharmacokinetic or epidemic models with fast compartments.

---

## Example 5.6 — A second-order ODE as a system; multistep vs one-step

**Problem.** Damped oscillator $u''+2\zeta\omega u'+\omega^2u = 0$ with $\omega = 2$, $\zeta = 0.1$, $u(0) = 1$, $u'(0) = 0$. Rewrite it as a system and compare Adams predictor–corrector (4th order) with RK4.

**Solution.** The system is $(u,v)' = (v, -0.4v-4u)$. Maximum errors on $[0,10]$:

| $h$ | Adams PC error | $f$-evaluations | RK4 error | $f$-evaluations |
|---|---|---|---|---|
| 0.1 | $2.0\times10^{-4}$ | ~200 | $5.0\times10^{-5}$ | 400 |
| 0.05 | $1.1\times10^{-5}$ | ~400 | $3.1\times10^{-6}$ | 800 |
| 0.025 | $6.3\times10^{-7}$ | ~800 | $1.9\times10^{-7}$ | 1600 |

Both are 4th order (ratio about 16–19). The predictor–corrector uses half the function evaluations for about 3–4 times the error: for expensive $f$ it wins on cost per accuracy.

**Take-away.** Every higher-order ODE becomes a first-order system. Multistep methods reuse past evaluations and are cheaper per step, but they need starting values and are awkward with step-size changes or events.
