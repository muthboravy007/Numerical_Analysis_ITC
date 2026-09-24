# Chapter 15 — Exercises: Numerical Optimisation for Data Science

> Lecture: [Chapter 15](../lectures/ch15-numerical-optimization.md) · Solutions: [`solutions/ch15_solutions.md`](../solutions/ch15_solutions.md) · Solution code: [`solutions/code/ch15_solutions.py`](../solutions/code/ch15_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★** Let $f$ be convex and differentiable. Prove that $f(\mathbf y)\ge f(\mathbf x)+\nabla f(\mathbf x)^T(\mathbf y-\mathbf x)$, that every stationary point is a global minimiser, and that a strongly convex $f$ has at most one minimiser.

**A2 ★★** *(Descent lemma and GD rates.)* Let $\nabla f$ be $L$-Lipschitz. (a) Prove $f(\mathbf y)\le f(\mathbf x)+\nabla f(\mathbf x)^T(\mathbf y-\mathbf x)+\frac L2\|\mathbf y-\mathbf x\|^2$. (b) Deduce that GD with step $1/L$ decreases $f$ by at least $\frac1{2L}\|\nabla f\|^2$ per step. (c) For convex $f$, show $f(\mathbf x_k)-f^\ast\le\frac{L\|\mathbf x_0-\mathbf x^\ast\|^2}{2k}$. (d) For $\mu$-strongly convex $f$, show $f(\mathbf x_k)-f^\ast\le(1-\mu/L)^k(f(\mathbf x_0)-f^\ast)$.

**A3 ★★** For $f = \frac12\mathbf x^TA\mathbf x-\mathbf b^T\mathbf x$ with $\lambda(A)\subset[\mu,L]$, show that fixed-step GD has error-contraction factor $\max(|1-\alpha\mu|,|1-\alpha L|)$. This factor is minimised by $\alpha = \frac2{\mu+L}$, which gives $\frac{\kappa-1}{\kappa+1}$. Show that GD diverges for $\alpha>2/L$.

**A4 ★★** (a) Show that Newton's method is affine invariant: minimising $g(\mathbf y) = f(B\mathbf y)$ for nonsingular $B$ produces iterates $\mathbf y_k = B^{-1}\mathbf x_k$. (b) Show that Newton's method minimises a strictly convex quadratic in one step. (c) Explain why pure Newton can increase $f$, and how a line search (damped Newton) or a trust region fixes this.

**A5 ★★** Show that the BFGS inverse update $H_{+} = (I-\rho\mathbf s\mathbf y^T)H(I-\rho\mathbf y\mathbf s^T)+\rho\mathbf s\mathbf s^T$, with $\rho = 1/\mathbf y^T\mathbf s$, (a) satisfies the secant equation $H_+\mathbf y = \mathbf s$; (b) is positive definite whenever $H$ is PD and $\mathbf y^T\mathbf s>0$. (c) Show that the Wolfe curvature condition $\nabla f(\mathbf x_+)^T\mathbf p\ge c_2\nabla f(\mathbf x)^T\mathbf p$ guarantees $\mathbf y^T\mathbf s>0$.

**A6 ★★** (a) Write the KKT conditions for $\min\frac12\|\mathbf x-\mathbf v\|^2$ subject to $\mathbf x\ge0$, $\mathbf 1^T\mathbf x = 1$, and derive the simplex projection $x_i = (v_i-\tau)_+$. (b) For $\min\frac12\mathbf x^TQ\mathbf x-\mathbf c^T\mathbf x$ subject to $A\mathbf x = \mathbf b$, derive the linear KKT system $\begin{pmatrix}Q&A^T\\A&0\end{pmatrix}\begin{pmatrix}\mathbf x\\\boldsymbol\lambda\end{pmatrix} = \begin{pmatrix}\mathbf c\\\mathbf b\end{pmatrix}$.

**A7 ★★** *(SGD.)* Let $f = \frac1n\sum f_i$ and let $\mathbf g_k = \nabla f_{i_k}(\mathbf x_k)$ with $i_k$ uniform. (a) Show $E[\mathbf g_k\mid\mathbf x_k] = \nabla f(\mathbf x_k)$. (b) For $f_i = \frac12(x-a_i)^2$ in 1-D, show that SGD with constant step $\eta$ has $E(x_k-\bar a)^2\to\frac{\eta\,\sigma_a^2}{2-\eta}$. So a constant step converges only to a neighbourhood. (c) State the Robbins–Monro conditions $\sum\eta_k = \infty$, $\sum\eta_k^2<\infty$.

**A8 ★** Explain why reverse-mode automatic differentiation computes the full gradient of $f:\mathbb R^n\to\mathbb R$ at a cost of a small constant times the cost of evaluating $f$, while forward mode and finite differences cost $O(n)$ evaluations.

## B. Hand computation

**B1 ★** Perform five golden-section steps for $f(x) = x^2-2\sin x$ on $[0,2]$ and one parabolic-interpolation step through $x = 0,1,2$. Compare both with the true minimiser.

**B2 ★** For $f(x,y) = x^2+5y^2$ from $(5,1)$: three GD steps with step $0.1$, and three steps with exact line search. What is the optimal fixed step, and what happens for steps $0.19$ and $0.21$?

**B3 ★★** For Rosenbrock's function at $(-1.2, 1)$, compute $f$, $\nabla f$, $\nabla^2f$ and its eigenvalues, and one Newton step. Then iterate pure Newton and comment on the sequence of $f$ values.

**B4 ★★** Apply BFGS with exact line search to $f = \frac12(x^2+5y^2)$ from $(1,1)$ with $H_0 = I$. Show both steps, the secant check, and that $H_2 = A^{-1}$.

**B5 ★★** Solve with KKT conditions: (a) $\min x^2+y^2$ subject to $x+2y\ge4$; (b) $\min x+y$ subject to $x^2+y^2\le2$. Give the multipliers.

**B6 ★** Project $\mathbf v = (0.8, 0.6, -0.2, 0.4)$ onto the probability simplex using the sort-based algorithm.

**B7 ★★** For $f(x_1,x_2) = (x_1x_2+\sin x_1)e^{x_2}$ at $(1,2)$, draw the computational graph, compute $\partial f/\partial x_1$ with forward mode, and compute the full gradient with one reverse sweep. Check against central differences.

**B8 ★★** Heavy-ball momentum on $f = \frac\lambda2x^2$ satisfies $x_{k+1} = (1+\beta-\alpha\lambda)x_k-\beta x_{k-1}$. Compute the characteristic roots for $\lambda = 1, 10$ with $(\alpha,\beta)\in\{(0.1,0),(0.1,0.5),(0.1,0.9),(0.15,0.9)\}$. Derive the optimal $(\alpha,\beta)$ for the spectrum $[1,10]$.

## C. Programming

**C1 ★★** Compare GD (fixed step and backtracking), Nesterov, Newton, BFGS and L-BFGS on Rosenbrock from $(-1.2,1)$ and on a 100-dimensional quadratic with $\kappa = 10^3$. Tabulate iterations, function and gradient evaluations, final error and time.

**C2 ★★** For quadratics with $\kappa = 10, 10^2, 10^3, 10^4$, count the iterations of GD (optimal step), optimally tuned heavy ball and CG to a relative residual of $10^{-6}$. Compare with $\frac\kappa2\ln10^6$ and $\frac{\sqrt\kappa}2\ln10^6$.

**C3 ★** *Gradient checking.* Compare the analytic gradient of the regularised logistic loss (breast-cancer data) with central differences for $h = 10^{-2},\dots,10^{-10}$. Then plant a bug (a factor 2 in the ridge term) and show that the check catches it.

**C4 ★★★** Implement a minimal reverse-mode AD class (`Var` with `+`, `*`, `tanh`, `log1pexp` and `backward`). Use it to differentiate the logistic loss of a 2-3-1 tanh network with respect to all 12 parameters, and verify the result by finite differences.

## D. Data-science applications

**D1 ★★** *Training logistic regression.* On the standardised breast-cancer data (31 parameters, $\lambda = 10^{-3}$), compare GD with step $1/L$, Nesterov, Newton, BFGS and L-BFGS by the number of iterations to a relative objective gap of $10^{-6}$. Then compare SGD (with decaying and constant step) and Adam after 10, 50 and 100 epochs.

**D2 ★★** *A neural network from scratch.* Train a 2-16-1 tanh network with hand-coded backpropagation on two moons (1000 points, noise 0.2) using mini-batch SGD and Adam with three learning rates each. Compare the train and test accuracy with linear logistic regression.

**D3 ★★** *Minimum-variance portfolio.* For an 8-asset factor-model covariance, find the long-only minimum-variance weights by projected gradient onto the simplex. Verify them against SLSQP and check the KKT conditions. Compare with the unconstrained (short-selling) solution.

**D4 ★★** *Proximal methods.* Solve the lasso on the diabetes data ($\lambda = 1$) with ISTA and FISTA (step $1/L$), and compare the objective gaps at iterations 10, 100, 1000 with the $O(1/k)$ and $O(1/k^2)$ theory. Use coordinate descent as the reference.
