# Lecture 10 — Ordinary Differential Equations

> **Week 14** · Code: [`numlib/ode.py`](../numlib/ode.py) · Examples: [`examples/ch10`](../examples/ch10_examples.md) · Exercises: [`exercises/ch10`](../exercises/ch10_exercises.md)

## Learning objectives

1. Formulate initial value problems (IVPs), including systems and higher-order equations rewritten as first-order systems.
2. Derive Euler's method and analyse local and global truncation error.
3. Apply Heun, midpoint and classical Runge–Kutta (RK4) methods and verify their order empirically.
4. Explain adaptive step-size control with embedded Runge–Kutta pairs.
5. Define stability regions, recognise **stiff** problems and use implicit methods.
6. Simulate data-science-relevant dynamical models: epidemics (SIR), population growth, predator–prey, and understand the link to gradient flow and neural ODEs.

---

## 10.1 Initial value problems

$$
\mathbf y'(t) = \mathbf f\bigl(t, \mathbf y(t)\bigr),\qquad \mathbf y(t_0) = \mathbf y_0,\qquad \mathbf y(t)\in\mathbb R^d.
$$

**Existence and uniqueness (Picard–Lindelöf).** If $\mathbf f$ is continuous in $t$ and Lipschitz in $\mathbf y$ ($\|\mathbf f(t,\mathbf u)-\mathbf f(t,\mathbf v)\|\le L\|\mathbf u-\mathbf v\|$), the IVP has a unique solution on some interval around $t_0$.

**Higher order → first order.** $y'' + \gamma y' + \omega^2 y = 0$ becomes, with $\mathbf y = (y, y')$,
$$
\mathbf y' = \begin{pmatrix} y_2\\ -\omega^2y_1 - \gamma y_2\end{pmatrix}.
$$

### Models you will meet in data science
* **Logistic growth** $y' = ry(1-y/K)$ — product adoption, user growth, COVID case curves.
* **SIR epidemic model**
$$
S' = -\beta SI,\qquad I' = \beta SI - \gamma I,\qquad R' = \gamma I,
$$
with basic reproduction number $R_0 = \beta/\gamma$. Fitting $\beta,\gamma$ to case data combines this lecture with Lecture 9 (Project 5).
* **Lotka–Volterra** predator–prey; **compartment models** in pharmacokinetics.
* **Gradient flow** $\mathbf w' = -\nabla F(\mathbf w)$: gradient descent *is* Euler's method on this ODE with step $\alpha$.
* **Neural ODEs**: a ResNet block $\mathbf h_{k+1} = \mathbf h_k + h\,\mathbf f_\theta(\mathbf h_k)$ is an Euler step.

## 10.2 Euler's method

On a grid $t_k = t_0 + kh$, replace $\mathbf y'(t_k)$ by the forward difference:
$$
\boxed{\mathbf y_{k+1} = \mathbf y_k + h\,\mathbf f(t_k,\mathbf y_k).}
$$
Geometrically: follow the tangent line for a time $h$.

### Error analysis
* **Local truncation error** (error made in one step starting from the exact solution): by Taylor, $\tau_{k+1} = \frac{h^2}{2}y''(\xi) = O(h^2)$.
* **Global error** at a fixed time $T$: about $T/h$ steps each contributing $O(h^2)$, amplified by at most $e^{L(t-t_0)}$:

**Theorem.** If $\mathbf f$ is $L$-Lipschitz and $\|y''\|\le M$, then
$$
|y(t_k) - y_k| \le \frac{hM}{2L}\bigl(e^{L(t_k-t_0)} - 1\bigr) = O(h).
$$

Euler's method is **first order**: halving $h$ halves the error. A method with local error $O(h^{p+1})$ has global error $O(h^p)$ and is called **order $p$**.

**Example.** $y' = y - t^2 + 1$, $y(0)=0.5$, exact solution $y = (t+1)^2 - \tfrac12e^t$. With $h = 0.5$:

| $t$ | Euler | Heun | RK4 | Exact |
|---|---|---|---|---|
| 0.0 | 0.5 | 0.5 | 0.5 | 0.5 |
| 0.5 | 1.25 | 1.375 | 1.425130 | 1.425639 |
| 1.0 | 2.25 | 2.515625 | 2.639603 | 2.640859 |
| 1.5 | 3.375 | 3.775391 | 4.006819 | 4.009155 |
| 2.0 | 4.4375 | 4.916260 | 5.301605 | 5.305472 |

## 10.3 Runge–Kutta methods

Use several slope evaluations per step, combined to cancel Taylor terms.

**Heun (improved Euler, RK2):** predictor–corrector with the trapezoid rule
$$
\mathbf k_1 = \mathbf f(t_k,\mathbf y_k),\quad \mathbf k_2 = \mathbf f(t_k+h,\mathbf y_k + h\mathbf k_1),\quad \mathbf y_{k+1} = \mathbf y_k + \tfrac h2(\mathbf k_1+\mathbf k_2).
$$

**Explicit midpoint (RK2):** $\mathbf y_{k+1} = \mathbf y_k + h\,\mathbf f\bigl(t_k + \tfrac h2, \mathbf y_k + \tfrac h2\mathbf k_1\bigr)$.

**Classical RK4:**
$$
\begin{aligned}
\mathbf k_1 &= \mathbf f(t_k,\mathbf y_k), &
\mathbf k_2 &= \mathbf f\bigl(t_k+\tfrac h2,\ \mathbf y_k+\tfrac h2\mathbf k_1\bigr),\\
\mathbf k_3 &= \mathbf f\bigl(t_k+\tfrac h2,\ \mathbf y_k+\tfrac h2\mathbf k_2\bigr), &
\mathbf k_4 &= \mathbf f(t_k+h,\ \mathbf y_k+h\mathbf k_3),\\
\mathbf y_{k+1} &= \mathbf y_k + \tfrac h6(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4). &&
\end{aligned}
$$
For $y' = f(t)$ this reduces to Simpson's rule. Global error $O(h^4)$.

**Observed errors at $t=2$** for the example above:

| $h$ | Euler | Heun | RK4 |
|---|---|---|---|
| 0.2 | $4.4\times10^{-1}$ | $7.2\times10^{-2}$ | $1.1\times10^{-4}$ |
| 0.1 | $2.4\times10^{-1}$ | $1.9\times10^{-2}$ | $7.0\times10^{-6}$ |
| 0.05 | $1.3\times10^{-1}$ | $4.8\times10^{-3}$ | $4.4\times10^{-7}$ |
| ratio per halving | ≈ 2 | ≈ 4 | ≈ 16 |

**Cost vs. accuracy.** RK4 needs 4 evaluations per step; Euler 1. For an accuracy of $10^{-6}$, Euler would need $h\approx10^{-6}$ (millions of steps); RK4 needs $h\approx0.1$. Higher order wins whenever the solution is smooth.

## 10.4 Adaptive step size

Embedded pairs (e.g. **Dormand–Prince 5(4)**, `solve_ivp(method="RK45")`) compute a 5th- and a 4th-order solution from the *same* stages. Their difference estimates the local error $\text{err}$. Scaled by $\text{atol} + \text{rtol}\,|\mathbf y|$:

* if $\text{err}\le1$: accept the step;
* in both cases update $h \leftarrow h\cdot\min\bigl(5,\max(0.2,\ 0.9\,\text{err}^{-1/5})\bigr)$.

Small steps are taken automatically where the solution changes quickly (e.g. the epidemic peak) and large steps where it is flat.

## 10.5 Stability and stiffness

Apply a method to the **test equation** $y' = \lambda y$ ($\operatorname{Re}\lambda<0$, so $y\to0$). Euler gives $y_{k+1} = (1+h\lambda)y_k$, which decays only if
$$
|1 + h\lambda| < 1\qquad(\text{for real }\lambda<0:\ h < 2/|\lambda|).
$$
The set $\{z = h\lambda : |R(z)|<1\}$ is the method's **region of absolute stability**. For explicit RK methods it is bounded.

**Stiffness.** A system is stiff when it contains very fast decaying components (large $|\lambda|$) alongside the slow dynamics we care about. Explicit methods must use $h \lesssim 2/|\lambda_{\max}|$ for **stability**, far smaller than accuracy requires.

**Example.** $y' = -50y$, $y(0)=1$, $h = 0.05$ ($h\lambda = -2.5$):
* Euler: $y_k = (-1.5)^k$: $1, -1.5, 2.25, -3.375, 5.06, \ldots$ — explodes.
* Implicit Euler: $y_k = (1/3.5)^k$: $1, 0.286, 0.082, 0.023, \ldots$ — decays correctly.

### Implicit methods
**Backward (implicit) Euler:**
$$
\mathbf y_{k+1} = \mathbf y_k + h\,\mathbf f(t_{k+1},\mathbf y_{k+1}).
$$
For $y'=\lambda y$: $y_{k+1} = \frac{1}{1-h\lambda}y_k$, stable for **every** $h>0$ when $\operatorname{Re}\lambda<0$ (**A-stable**). The price: each step requires solving a (generally nonlinear) system for $\mathbf y_{k+1}$ — by Newton's method (Lecture 2) with the Jacobian $I - h\,\partial\mathbf f/\partial\mathbf y$ (Lecture 3).

**Trapezoidal rule / Crank–Nicolson:** $\mathbf y_{k+1} = \mathbf y_k + \frac h2\bigl(\mathbf f_k + \mathbf f_{k+1}\bigr)$ — A-stable and second order.

Production stiff solvers: BDF (`solve_ivp(method="BDF")`), Radau, and LSODA (switches automatically).

## 10.6 Beyond this course
* **Symplectic integrators** (leapfrog/Verlet) conserve energy over long times — used in **Hamiltonian Monte Carlo** (Stan, PyMC).
* **Boundary value problems**: shooting (root finding + IVP) and finite differences (tridiagonal systems) — `solve_bvp`.
* **PDEs**: method of lines turns e.g. the heat equation into a large stiff ODE system.
* **Stochastic differential equations**: Euler–Maruyama; diffusion models in generative AI integrate a reverse-time SDE/ODE.

## 10.7 Summary

| Method | Order | Evaluations/step | Stability |
|---|---|---|---|
| Euler | 1 | 1 | small region |
| Heun / midpoint | 2 | 2 | small region |
| RK4 | 4 | 4 | moderate region |
| RK45 (adaptive) | 5(4) | 6 (FSAL) | moderate region |
| Implicit Euler | 1 | Newton solve | A-stable |
| Trapezoidal | 2 | Newton solve | A-stable |

* Global error $O(h^p)$; verify order with a log–log plot.
* Stiff problems need implicit methods.

## Further reading

* Sauer, Chapter 6; Burden & Faires, Chapter 5.
* Hairer, Nørsett & Wanner, *Solving Ordinary Differential Equations I* (non-stiff) and *II* (stiff).
* Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018.
