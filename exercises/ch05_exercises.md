# Chapter 5 — Exercises: Initial-Value Problems for ODEs

> Lecture: [Chapter 5](../lectures/ch05-initial-value-problems.md) · Solutions: [`solutions/ch05_solutions.md`](../solutions/ch05_solutions.md) · Solution code: [`solutions/code/ch05_solutions.py`](../solutions/code/ch05_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★** Show that $f(t,y) = t^2\sin y$ satisfies a Lipschitz condition in $y$ on $\{0\le t\le2\}\times\mathbb R$ and find $L$. Is $f(t,y) = \sqrt{|y|}$ Lipschitz near $y = 0$?

**A2 ★★** Prove Lemma 5.8: if $a_{i+1}\le(1+s)a_i + t$ with $s>0$, $t\ge0$, then $a_{i+1}\le e^{(i+1)s}\bigl(a_0 + \frac ts\bigr) - \frac ts$. Use it to derive the Euler error bound $\frac{hM}{2L}(e^{L(t_i-a)}-1)$.

**A3 ★★** Show that all two-stage explicit Runge–Kutta methods of order 2 satisfy $a_1+a_2 = 1$, $a_2\alpha_2 = a_2\delta_2 = \frac12$, and verify that the midpoint and modified Euler methods are members of this family.

**A4 ★★** Derive the two-step Adams–Bashforth formula $w_{i+1} = w_i + \frac h2(3f_i - f_{i-1})$ and its local truncation error $\frac{5}{12}y'''(\xi)h^2$.

**A5 ★★★** For the test equation $y' = \lambda y$ find the stability function $Q(h\lambda)$ of (a) Euler, (b) backward Euler, (c) the trapezoidal method, (d) RK4. Which are A-stable? Show that for the trapezoidal method $|Q(z)|\to1$ as $z\to-\infty$ and explain the consequence for stiff problems.

## B. Hand computation

**B1 ★** Use Euler's method with $h = 0.5$ and $h = 0.25$ for $y' = te^{-t} - y$, $0\le t\le1$, $y(0) = 0$; compare with the exact solution $y = \frac12t^2e^{-t}$.

**B2 ★★** For B1 find $L$ and $M = \max|y''|$ and evaluate the error bound of Theorem 5.9 at $t = 1$ with $h = 0.25$. Compare with the actual error.

**B3 ★★** Apply the Taylor method of order 2 with $h = 0.25$ to $y' = \frac yt + t^2$, $1\le t\le2$, $y(1) = 1$ (exact $y = \frac{t^3+t}{2}$).

**B4 ★** Perform one RK4 step ($h = 0.25$) for the problem of B3 showing $k_1,\dots,k_4$, and then complete the integration to $t = 2$.

**B5 ★** Solve B3 with the midpoint and modified Euler methods ($h = 0.25$) and compare $w(2)$.

**B6 ★★** Solve B3 with the Adams–Bashforth four-step method and the Adams predictor–corrector with $h = 0.2$ (RK4 starting values).

**B7 ★★** For $y' = -20y$ compute the amplification factors of Euler, RK4 and backward Euler for $h = 0.05, 0.10, 0.15$. For which $h$ is each method stable?

**B8 ★★** Find the characteristic roots of (a) the leapfrog method $w_{i+1} = w_{i-1} + 2hf_i$ and (b) BDF2: $w_{i+1} = \frac43w_i - \frac13w_{i-1} + \frac23hf_{i+1}$. Classify each as strongly stable, weakly stable or unstable.

## C. Programming

**C1 ★** Verify numerically the orders of Euler, Heun (modified Euler) and RK4 on B1's problem with $h = 0.1, 0.05, 0.025, 0.0125$.

**C2 ★★** Implement Runge–Kutta–Fehlberg (B&F Algorithm 5.3) and apply it to B1's problem with TOL $= 10^{-6}$, $h_{\max} = 0.25$. Report the number of steps, the range of step sizes and the maximum error.

**C3 ★★** The pendulum $\theta'' = -\frac gL\sin\theta$ ($g/L = 9.81$) with $\theta(0) = \theta_0$, $\theta'(0) = 0$. Compute the period for $\theta_0 = 0.1, 1.0, 2.5$ (use event detection for $\theta' = 0$) and compare with the small-angle period $2\pi\sqrt{L/g}$.

**C4 ★★** Integrate Robertson's stiff kinetics to $t = 40$ with `solve_ivp` using RK45, LSODA and BDF. Tabulate the numbers of function evaluations.

## D. Data-science applications

**D1 ★★** SEIR epidemic: $S' = -\beta SI$, $E' = \beta SI - \sigma E$, $I' = \sigma E - \gamma I$, $R' = \gamma I$, $\beta = 0.5$, $\sigma = 0.2$, $\gamma = 0.1$, $S(0) = 0.999$, $I(0) = 0.001$. Find the peak day, peak prevalence and final attack rate. Repeat with contact reductions of 30% and 50% ($\beta\to0.7\beta, 0.5\beta$) and discuss "flattening the curve".

**D2 ★★** One-compartment pharmacokinetics: a 100 mg bolus, concentration $C' = -kC$, $C(0) = 100/V$. Simulate noisy measurements every 2 h for 24 h with $k = 0.25$, $V = 10$, and estimate $(k, V)$ by nonlinear least squares where each residual evaluation solves the ODE numerically. Report the half-life.

**D3 ★★** Gradient descent on $F(\mathbf w) = \frac12(w_1^2 + 25w_2^2)$ is Euler's method for the gradient flow. Predict the largest stable learning rate, then verify with $\alpha = 0.01, 0.07, 0.079, 0.081$ (200 iterations from $(1,1)$). Solve the gradient flow with a stiff solver and compare with $e^{-t}$ and $e^{-25t}$.
