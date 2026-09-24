# Chapter 10 — Worked Examples: Nonlinear Systems of Equations

Companion script: [`ch10_examples.py`](ch10_examples.py) reproduces every number below. · Lecture: [Chapter 10](../lectures/ch10-nonlinear-systems.md)

---

## Example 10.1 — Trilateration (how GPS-like positioning works)

**Problem.** A device at an unknown position $\mathbf p$ measures its distances to beacons. (a) With two beacons at $(0,0)$ and $(10,0)$ and exact distances $5$ and $8.0623$, solve $\|\mathbf p-\mathbf b_i\|^2 = d_i^2$ by Newton's method. (b) With four beacons at the corners of $[0,10]^2$ and noisy ranges (SD 0.1), estimate $\mathbf p$ by Gauss–Newton.

**Solution.** (a) With $J = 2(\mathbf p-\mathbf b_i)^T$ and start $(5,5)$, the iterates are $(3, 4.5)\to(3, 4.0278)\to(3, 4.000096)\to(3,4)$. Two circles meet in two points, and from $(5,-5)$ Newton finds the mirror solution $(3,-4)$. (b) Six Gauss–Newton iterations give $\hat{\mathbf p} = (3.023, 3.970)$, with standard errors $(0.073, 0.069)$ from $\sigma^2(J^TJ)^{-1}$.

**Take-away.** With exactly as many equations as unknowns the answer may not be unique, and the starting point decides which solution Newton's method finds. Redundant measurements turn the problem into nonlinear least squares, which averages out the noise and provides uncertainty estimates.

---

## Example 10.2 — Endemic equilibrium of an epidemic

**Problem.** An SIR model with births and deaths ($\mu = 0.02$), transmission $\beta = 0.5$ and recovery $\gamma = 0.1$ has equilibria given by
$\mu-\beta SI-\mu S = 0$ and $\beta SI-(\gamma+\mu)I = 0$. Find the endemic equilibrium and assess its stability.

**Solution.** Newton from $(0.5, 0.05)$ converges in 9 iterations to $(S^\ast, I^\ast) = (0.24, 0.126667)$. This matches the theory: $R_0 = \frac{\beta}{\gamma+\mu} = 4.167$, $S^\ast = 1/R_0$ and $I^\ast = \mu(R_0-1)/\beta$. The Jacobian there has eigenvalues $-0.0417\pm0.0766i$, so the equilibrium is stable, and it is approached through damped oscillations (epidemic waves). From $(0.9, 10^{-4})$, Newton converges to the disease-free equilibrium $(1, 0)$, which is unstable because $R_0>1$.

**Take-away.** Newton finds equilibria of dynamical systems, and the same Jacobian then tells you about their stability. Both steady states are solutions: the algorithm has no notion of which one is "physical".

---

## Example 10.3 — Broyden saves Jacobians

**Problem.** Solve the discretised Bratu-type system $-w_{i-1}+2w_i-w_{i+1}-2h^2e^{w_i} = 0$ ($n = 50$) with Newton and with Broyden (one initial Jacobian).

**Solution.** Newton needs 5 iterations, 5 $F$-evaluations and 5 Jacobians. Broyden needs 7 iterations, 7 $F$-evaluations and **1** Jacobian. The two solutions agree to $10^{-16}$, with $\max w = 0.328853$.

**Take-away.** When the Jacobian is expensive, e.g. from finite differences that need $n$ extra $F$-evaluations or from a complex simulation, Broyden's secant updates trade a couple of extra iterations for far less work per iteration.

---

## Example 10.4 — Finding all solutions

**Problem.** Find all real solutions of $x^2+y^2 = 4$, $xy = 1$.

**Solution.** By symmetry the solutions are $(\pm a,\pm b)$ and $(\pm b,\pm a)$ with $a = \sqrt{2+\sqrt3} = 1.9318517$ and $b = \sqrt{2-\sqrt3} = 0.5176381$. Newton from $(2,0.5)$, $(0.5,2)$, $(-2,-0.5)$ and $(-0.5,-2)$ finds each of the four solutions in 4 iterations. From $(1, 1.1)$, close to the line $x = y$ where $J = \begin{pmatrix}2x&2y\\y&x\end{pmatrix}$ is singular, it needs 9 iterations.

**Take-away.** Use problem structure (symmetry) and multiple starting points, or homotopy methods, to find *all* solutions. Starting near a singular Jacobian gives erratic first steps.

---

## Example 10.5 — Best-response dynamics is a fixed-point iteration

**Problem.** Two firms with costs $c_1 = 10$ and $c_2 = 16$ face inverse demand $p = 100-q_1-q_2$. Each firm's best response is $q_i = \frac{100-c_i-q_j}2$. Iterate best responses from $(0,0)$.

**Solution.** The iterates are $(45, 42)\to(24, 19.5)\to(35.25, 30)\to(30, 24.375)\to\cdots\to(32, 26)$, reached to $10^{-10}$ in 40 iterations. The Cournot equilibrium is $\bigl(\frac{A-2c_1+c_2}3, \frac{A-2c_2+c_1}3\bigr) = (32, 26)$. The map has $\|J_G\|_\infty = \frac12$, so the error halves at each step, oscillating around the equilibrium.

**Take-away.** Many economic and learning processes (best responses, EM, $k$-means, alternating least squares) are fixed-point iterations. Their convergence rate is the contraction factor, and Newton's method on the same equations would converge in one step here, because the system is linear.

---

## Example 10.6 — Circle fitting: algebraic vs geometric

**Problem.** Fit a circle to 30 noisy points on a quarter arc (true centre $(2,-1)$, radius 3, noise SD 0.05).

**Solution.** The algebraic (Kåsa) fit solves the *linear* least-squares problem $x^2+y^2 = Ax+By+C$. It gives centre $(2.028, -1.009)$ and radius $3.009$, with RMS geometric residual $0.0473$. The geometric fit minimises the true distances $\sum(\|\mathbf p_i-\mathbf c\|-R)^2$ with Levenberg–Marquardt, starting from the Kåsa fit. It gives centre $(1.976, -1.062)$ and radius $3.074$, with RMS residual $0.0468$.

**Take-away.** The geometric fit is optimal for its criterion, but on a short arc the centre and radius are strongly correlated and poorly determined. Two quite different circles fit almost equally well. A good linear "algebraic" solution is the standard starting point for the nonlinear refinement.
