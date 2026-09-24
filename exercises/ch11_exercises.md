# Chapter 11 — Exercises: Boundary-Value Problems for ODEs

> Lecture: [Chapter 11](../lectures/ch11-boundary-value-problems.md) · Solutions: [`solutions/ch11_solutions.md`](../solutions/ch11_solutions.md) · Solution code: [`solutions/code/ch11_solutions.py`](../solutions/code/ch11_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★★** *(Maximum principle.)* Let $q(x)>0$ on $[a,b]$ and let $u$ satisfy $u'' = p(x)u'+q(x)u$ with $u(a) = u(b) = 0$. Show that $u$ cannot have a positive interior maximum or a negative interior minimum, so $u\equiv0$. Deduce that the linear BVP $y'' = py'+qy+r$, $y(a) = \alpha$, $y(b) = \beta$ has at most one solution.

**A2 ★★** (a) Show that the centred finite-difference scheme for $y'' = py'+qy+r$ has local truncation error $\frac{h^2}{12}\bigl(y^{(4)}(\xi)-2p\,y'''(\eta)\bigr)$. (b) Show that if $q\ge0$ and $h\max|p|<2$, the tridiagonal matrix has nonpositive off-diagonal entries and is weakly diagonally dominant (strictly in at least one row), hence nonsingular.

**A3 ★** In linear shooting, let $y_1$ solve the IVP with $y(a) = \alpha$, $y'(a) = 0$ and let $y_2$ solve the homogeneous IVP with $y(a) = 0$, $y'(a) = 1$. Show that $y = y_1+\frac{\beta-y_1(b)}{y_2(b)}y_2$ solves the BVP. Why is $y_2(b)\ne0$ when $q>0$?

**A4 ★★★** *(Rayleigh–Ritz / Galerkin.)* For $-(p(x)y')'+q(x)y = f$ with $y(0) = y(1) = 0$, $p>0$, $q\ge0$:
(a) show that the solution minimises $I[u] = \int_0^1\bigl(p u'^2+qu^2-2fu\bigr)dx$ over $H_0^1$;
(b) show that the Galerkin solution $u_h$ in a subspace $V_h$ satisfies Galerkin orthogonality $a(y-u_h,v) = 0$ for all $v\in V_h$;
(c) deduce that $u_h$ is the best approximation of $y$ in the energy norm $\|v\|_a = \sqrt{a(v,v)}$.

**A5 ★★** For $y'' = k^2y$ on $[0,1]$ show that the homogeneous IVP solution used by linear shooting satisfies $y_2(1) = \sinh(k)/k$, while the BVP solution with $y(0) = 1$, $y(1) = e^{-k}$ is $e^{-kx}$. Explain why single shooting loses about $\log_{10}e^{2k}$ digits.

## B. Hand computation

**B1 ★** For $y'' = 4y$, $y(0) = 1$, $y(1) = 2$: the IVP solutions are $y_1 = \cosh2x$ and $y_2 = \frac12\sinh2x$. Build the shooting solution and $y'(0)$. Then run linear shooting with RK4, $h = 0.25$, and compare with the exact solution.

**B2 ★** Solve $y'' = -y+x$, $y(0) = 0$, $y(1) = 2$ by finite differences with $h = 0.25$ (three unknowns), writing out the $3\times3$ system. The exact solution is $y = x+\frac{\sin x}{\sin1}$. Repeat with $h = 1/8, 1/16, 1/32$ and confirm $O(h^2)$.

**B3 ★** Solve $y'' = -2y'$, $y(0) = 0$, $y(1) = 1$ by finite differences with $h = 0.25$. Give the sub- and superdiagonal entries and compare with $y = \frac{1-e^{-2x}}{1-e^{-2}}$.

**B4 ★★** *Convection dominance.* For $y'' = 50y'$, $y(0) = 0$, $y(1) = 1$ (boundary layer at $x = 1$), compare centred differences with $h = 0.1$ and $h = 0.02$, and the first-order *upwind* scheme ($y'\approx(w_i-w_{i-1})/h$). Relate the oscillations to the cell Péclet number $h|p|/2$.

**B5 ★★** Solve $y'' = -(y')^2$, $y(0) = 0$, $y(1) = \ln2$ (exact $y = \ln(1+x)$) by Newton shooting from $t_0 = \ln2$. Carry out the first Newton update analytically: show $y(1;t) = \ln(1+t)$ and $z(1;t) = \partial y(1;t)/\partial t = \frac1{1+t}$.

**B6 ★★** Apply nonlinear finite differences to B5 with $h = 0.25$. Write $\mathbf F(\mathbf w^{(0)})$ and $J(\mathbf w^{(0)})$ for the linear initial guess, perform one Newton step, then iterate to convergence and verify $O(h^2)$.

**B7 ★★** Use the Rayleigh–Ritz method with piecewise-linear hat functions ($h = 0.25$) for $-y''+y = 1$, $y(0) = y(1) = 0$. Assemble the stiffness-plus-mass matrix $\frac1h\operatorname{tridiag}(-1,2,-1)+\frac h6\operatorname{tridiag}(1,4,1)$ and the load vector, solve, and compare with $y = 1-\frac{\cosh(x-\frac12)}{\cosh\frac12}$.

## C. Programming

**C1 ★★** For B2 compare the maximum errors of linear shooting (RK4) and finite differences for $h = 1/4,\dots,1/64$. Apply two levels of Richardson extrapolation to the FD value at $x = 0.5$.

**C2 ★★★** *The Bratu problem* $y''+\lambda e^y = 0$, $y(0) = y(1) = 0$. (a) Using the exact solution $y = -2\ln\frac{\cosh((x-\frac12)\theta/2)}{\cosh(\theta/4)}$ with $\theta = \sqrt{2\lambda}\cosh(\theta/4)$, find the critical $\lambda_c$ beyond which no solution exists. (b) For $\lambda = 1,2,3,3.5$ compute both solutions by nonlinear FD ($h = 0.005$) from the initial guesses $0.1\sin\pi x$ and $6\sin\pi x$. How does the Newton iteration count behave as $\lambda\to\lambda_c$?

**C3 ★★** Implement P1 finite elements with 3-point Gauss quadrature for $-((1+x^2)y')'+y = f$, with $f$ chosen so that $y = \sin\pi x$. Measure the nodal, $L^2$ and $H^1$-seminorm errors for $h = 1/8,\dots,1/128$ and the convergence rates.

**C4 ★★** For $y'' = k^2y$, $y(0) = 1$, $y(1) = e^{-k}$ with $k = 10, 20, 40$ compare linear shooting (RK4, 1000 steps) with finite differences (999 interior points). Explain the result with A5.

## D. Data-science applications

**D1 ★★** *Whittaker smoothing with GCV.* 300 noisy samples ($\sigma = 0.25$) of $\sin(4\pi t)e^{-2t}+t$. For $\lambda = 10^{-1},\dots,10^6$ compute $\mathbf z = (I+\lambda D_2^TD_2)^{-1}\mathbf y$, the effective degrees of freedom $\operatorname{tr}S_\lambda$, the GCV score $\frac{n\|\mathbf y-\mathbf z\|^2}{(n-\operatorname{tr}S)^2}$ and the RMSE against the truth. This is a discrete BVP: explain the connection to $z-\lambda z'''' = y$.

**D2 ★★** *Identifiability.* A concentration profile obeys $-Dc''+kc = 0$, $c(0) = 1$, $c(1) = 0$. From 9 noisy measurements ($\sigma = 0.005$, true $D = 0.5$, $k = 8$), estimate $(\log D,\log k)$ by nonlinear least squares, where each residual evaluation solves the BVP by finite differences. Examine the singular values of the Jacobian and results from different starting points, then refit using $m = \sqrt{k/D}$.

**D3 ★★** *Gambler's ruin for a diffusion.* $dX = \mu\,dt+\sigma\,dW$ with $\mu = 0.3$, $\sigma = 1$, started at $x_0 = 2$ in $(0,5)$. The ruin probability $u$ and the expected exit time $T$ solve $\frac{\sigma^2}2u''+\mu u' = 0$ ($u(0) = 1$, $u(5) = 0$) and $\frac{\sigma^2}2T''+\mu T' = -1$ ($T(0) = T(5) = 0$). Solve both by finite differences and compare with the closed forms and with Monte Carlo (Euler–Maruyama with $\Delta t = 10^{-2}, 10^{-3}$, with and without the Brownian-bridge crossing correction).
