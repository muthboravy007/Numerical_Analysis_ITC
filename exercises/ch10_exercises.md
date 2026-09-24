# Chapter 10 — Exercises: Numerical Solutions of Nonlinear Systems of Equations

> Lecture: [Chapter 10](../lectures/ch10-nonlinear-systems.md) · Solutions: [`solutions/ch10_solutions.md`](../solutions/ch10_solutions.md) · Solution code: [`solutions/code/ch10_solutions.py`](../solutions/code/ch10_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★★** Let $D = \{\mathbf x: a_i\le x_i\le b_i\}$ and let $\mathbf G: D\to D$ be continuous with $\left|\frac{\partial g_i}{\partial x_j}\right|\le\frac Kn$ on $D$ for some $K<1$. Prove that $\mathbf G$ is a contraction in $\|\cdot\|_\infty$. Deduce that the fixed-point iteration converges to the unique fixed point, with $\|\mathbf x^{(k)}-\mathbf p\|_\infty\le\frac{K^k}{1-K}\|\mathbf x^{(1)}-\mathbf x^{(0)}\|_\infty$.

**A2 ★★** Let $\mathbf F(\mathbf p) = \mathbf 0$, let $J(\mathbf p)$ be nonsingular with $\|J(\mathbf p)^{-1}\|\le\beta$, and let $\|J(\mathbf x)-J(\mathbf y)\|\le L\|\mathbf x-\mathbf y\|$ near $\mathbf p$. Show that Newton's method satisfies $\|\mathbf e_{k+1}\|\le\beta L\|\mathbf e_k\|^2$ (up to a constant factor) for $\mathbf x^{(k)}$ close enough to $\mathbf p$. (Hint: $\mathbf F(\mathbf x) = \int_0^1J(\mathbf p+t(\mathbf x-\mathbf p))(\mathbf x-\mathbf p)\,dt$.)

**A3 ★★** (a) Show that the Broyden update $A_1 = A_0+\frac{(\mathbf y-A_0\mathbf s)\mathbf s^T}{\mathbf s^T\mathbf s}$ satisfies the secant equation $A_1\mathbf s = \mathbf y$ and agrees with $A_0$ on $\mathbf s^\perp$. (b) Prove it is the solution of $\min\|A-A_0\|_F$ subject to $A\mathbf s = \mathbf y$. (c) Prove the Sherman–Morrison formula $(A+\mathbf u\mathbf v^T)^{-1} = A^{-1}-\frac{A^{-1}\mathbf u\mathbf v^TA^{-1}}{1+\mathbf v^TA^{-1}\mathbf u}$ and count the operations of one inverse update.

**A4 ★★** For $g(\mathbf x) = \sum_iF_i(\mathbf x)^2$ show that $\nabla g = 2J^T\mathbf F$. Deduce that every stationary point of $g$ at which $J$ is nonsingular is a root of $\mathbf F$. Construct an example where steepest descent on $g$ stops at a point that is not a root.

**A5 ★** For the homotopy $H(\mathbf x,\lambda) = \mathbf F(\mathbf x)+(\lambda-1)\mathbf F(\mathbf x(0))$, show that the solution path satisfies $\mathbf x'(\lambda) = -J(\mathbf x(\lambda))^{-1}\mathbf F(\mathbf x(0))$. What can go wrong along the path?

## B. Hand computation

**B1 ★★** Let $\mathbf G(x,y) = \left(\frac{1+\cos y}{4},\ \frac{1+\sin x}{3}\right)$. Show that $\mathbf G$ maps $[0,1]^2$ into itself and satisfies the hypotheses of A1. Iterate from $(0,0)$ to $10^{-8}$, and compare with the Gauss–Seidel variant.

**B2 ★★** (a) Use Newton's method on $x^2+y^2 = 5$, $x-y = -1$ from $(2,3)$ and verify quadratic convergence. (b) Perform the first Newton step by hand for
$x+y+z = 3$, $x^2+y^2+z^2 = 5$, $e^x+yz = 3$ from $(0.1, 1.2, 1.7)$, then iterate to convergence. (The root is $(0,1,2)$.)

**B3 ★★** Apply Broyden's method to B2(a) from $(2,3)$ with $A_0 = J(\mathbf x^{(0)})$. Compute the first update $A_1$ by hand, verify the secant equation, compare $A_1$ with $J(\mathbf x^{(1)})$, and compare the convergence with Newton's.

**B4 ★★** Apply steepest descent (B&F Algorithm 10.3) to $g = F_1^2+F_2^2$ for the system of B2(a), first from $(0,0)$ and then from $(0,1)$. Explain what happens in the first case.

**B5 ★** Apply the homotopy method (RK4, $N$ steps) to B2(a) from $(1,0)$ with $N = 1, 2, 4, 8$. Show the path for $N = 2$ and the order of accuracy of the end point.

**B6 ★★** The circle $x^2+y^2 = 2$ and the line $x+y = 2$ are tangent at $(1,1)$. Apply Newton's method from $(2,0.5)$ and explain the linear convergence with ratio $\frac12$.

## C. Programming

**C1 ★★** The *Broyden tridiagonal function* $F_i = (3-2x_i)x_i-x_{i-1}-2x_{i+1}+1$ ($x_0 = x_{n+1} = 0$) from $\mathbf x^{(0)} = -\mathbf 1$. Compare Newton, Broyden, `scipy.optimize.root(method="hybr")` and Newton–Krylov (`method="krylov"`) for $n = 100$ and $1000$: iterations, $F$-evaluations and time.

**C2 ★★** Newton's method for $z^3 = 1$, viewed as a real $2\times2$ system. Iterate from a $401\times401$ grid on $[-2,2]^2$. Report the fraction of starting points attracted to each root, the non-convergent fraction and the mean number of iterations, and plot the basins.

**C3 ★★** For $\mathbf F = (\arctan x+0.1y,\ y-0.5x)$ compare pure Newton with damped Newton (backtracking on $\|\mathbf F\|$) from 1000 random starts in $[-10,10]^2$. Also find the approximate radius of convergence of pure Newton along the $x$-axis.

**C4 ★** For the $3\times3$ system of B2(b) compute the forward- and central-difference Jacobians at $\mathbf x^{(0)}$ for $h = 10^{-2},\dots,10^{-12}$. Explain the optimal $h$ for each.

## D. Data-science applications

**D1 ★★** *Gamma maximum likelihood.* Draw 400 observations from a Gamma distribution with shape 2.5 and scale 1.6. Solve the score equations $\partial\ell/\partial k = \partial\ell/\partial\theta = 0$ by Newton's method from the method-of-moments estimates. Report the standard errors from the observed information $-\nabla^2\ell$, a 95% CI for $k$ and the correlation of the estimators, and compare with `scipy.stats.gamma.fit`.

**D2 ★★** *Cournot equilibrium.* Three firms with marginal costs $c = (10,12,15)$ face inverse demand $p(Q) = 100\,Q^{-1/\eta}$ with $\eta = 1.5$. Each firm's first-order condition is $p(Q)+q_ip'(Q) = c_i$. Solve the $3\times3$ system with Newton's method. Report quantities, price, market shares and markups, and verify the Lerner condition $\frac{p-c_i}{p} = \frac{s_i}{\eta}$.

**D3 ★★★** *Multinomial logistic regression.* On the standardised iris data, fit a 3-class softmax model (class 0 as reference, ridge penalty $\lambda = 1$ on the non-intercept weights) by applying Newton's method to the gradient system $\nabla\ell(\mathbf W) = \mathbf 0$. Derive the block Hessian. Report $\|\nabla\ell\|$ per iteration and the training accuracy, and cross-check with BFGS.
