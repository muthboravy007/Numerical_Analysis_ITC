# Chapter 10 — Solutions

> Exercises: [`exercises/ch10_exercises.md`](../exercises/ch10_exercises.md) · Numbers from [`solutions/code/ch10_solutions.py`](code/ch10_solutions.py).

## A. Theory and proofs

**A1.** By the mean value theorem along the segment from $\mathbf y$ to $\mathbf x$ (convex $D$), $g_i(\mathbf x)-g_i(\mathbf y) = \sum_j\frac{\partial g_i}{\partial x_j}(\boldsymbol\xi_i)(x_j-y_j)$. Hence $|g_i(\mathbf x)-g_i(\mathbf y)|\le n\cdot\frac Kn\|\mathbf x-\mathbf y\|_\infty$, so $\|\mathbf G(\mathbf x)-\mathbf G(\mathbf y)\|_\infty\le K\|\mathbf x-\mathbf y\|_\infty$. $D$ is closed and $\mathbf G(D)\subset D$, so the contraction mapping theorem applies. The iterates are Cauchy, since $\|\mathbf x^{(k+1)}-\mathbf x^{(k)}\|\le K^k\|\mathbf x^{(1)}-\mathbf x^{(0)}\|$. The limit is a fixed point by continuity, and it is unique because two fixed points would satisfy $\|\mathbf p-\mathbf q\|\le K\|\mathbf p-\mathbf q\|$. Summing the geometric tail gives the bound.

**A2.** Let $\mathbf e_k = \mathbf x^{(k)}-\mathbf p$. Then
$\mathbf e_{k+1} = \mathbf e_k-J_k^{-1}\mathbf F(\mathbf x^{(k)}) = J_k^{-1}\bigl[J_k\mathbf e_k-\int_0^1J(\mathbf p+t\mathbf e_k)\mathbf e_k\,dt\bigr] = J_k^{-1}\int_0^1[J_k-J(\mathbf p+t\mathbf e_k)]\mathbf e_k\,dt$, where $J_k = J(\mathbf x^{(k)})$. The Lipschitz bound gives $\|J_k-J(\mathbf p+t\mathbf e_k)\|\le L(1-t)\|\mathbf e_k\|$, so $\|\mathbf e_{k+1}\|\le\|J_k^{-1}\|\frac L2\|\mathbf e_k\|^2$. Near $\mathbf p$, perturbation theory (Banach lemma) gives $\|J_k^{-1}\|\le2\beta$, hence $\|\mathbf e_{k+1}\|\le\beta L\|\mathbf e_k\|^2$.

**A3.** (a) $A_1\mathbf s = A_0\mathbf s+(\mathbf y-A_0\mathbf s) = \mathbf y$, and for $\mathbf z\perp\mathbf s$, $A_1\mathbf z = A_0\mathbf z$. (b) Any $A$ with $A\mathbf s = \mathbf y$ satisfies $(A-A_0)\mathbf s = \mathbf y-A_0\mathbf s$. Then $\|A_1-A_0\|_F = \frac{\|(A-A_0)\mathbf s\mathbf s^T\|_F}{\|\mathbf s\|^2}\le\|A-A_0\|_F\frac{\|\mathbf s\mathbf s^T\|_2}{\|\mathbf s\|^2} = \|A-A_0\|_F$. The minimiser is unique because the problem is strictly convex. (c) Multiply $(A+\mathbf u\mathbf v^T)$ by the proposed inverse: $I+A^{-1}$-terms cancel, since $\mathbf u\mathbf v^TA^{-1}-\frac{\mathbf u(1+\mathbf v^TA^{-1}\mathbf u)\mathbf v^TA^{-1}}{1+\mathbf v^TA^{-1}\mathbf u} = 0$. With $A^{-1}$ known, the update costs two matrix–vector products and one outer product, i.e. $O(n^2)$ instead of $O(n^3)$.

**A4.** $\partial g/\partial x_j = \sum_i2F_i\,\partial F_i/\partial x_j = (2J^T\mathbf F)_j$. If $\nabla g = \mathbf 0$ and $J$ is nonsingular, then $J^T\mathbf F = \mathbf 0$ implies $\mathbf F = \mathbf 0$. Example: in B4 below, $J$ is singular along $y = -x$. Steepest descent from $(0,0)$ stays on that line and stops at $(-1.473,1.473)$, where $g = 4.22$. Another example is an inconsistent system (Example 10.4.5).

**A5.** Differentiate $\mathbf F(\mathbf x(\lambda))+(\lambda-1)\mathbf F(\mathbf x(0)) = \mathbf 0$: $J\mathbf x'+\mathbf F(\mathbf x(0)) = \mathbf 0$. Along the path, $J$ can become singular (a turning point, where $\lambda$ must stop increasing), the path can go to infinity, or it can lead to a different root than intended. Pseudo-arclength continuation handles turning points.

## B. Hand computation

**B1.** On $[0,1]^2$, $g_1\in[\frac{1+\cos1}{4},\frac12] = [0.385,0.5]$ and $g_2\in[\frac13,\frac{1+\sin1}3] = [0.333,0.614]$, so $\mathbf G(D)\subset D$. The partial derivatives satisfy $|\partial g_1/\partial y| = |\sin y|/4\le0.21$ and $|\partial g_2/\partial x| = |\cos x|/3\le\frac13$, while the other two are $0$. All are $\le K/2$ with $K = \frac23$. The iterates are $(0,0)\to(0.5,0.333333)\to(0.486239,0.493142)\to(0.470212,0.489101)\to\cdots$. They converge to $\mathbf p = (0.47120898, 0.48465461)$ in **12** iterations, and in **7** with Gauss–Seidel. The actual rate is $\rho(J_{\mathbf G}(\mathbf p)) = 0.186$, much better than $K = 0.667$.

**B2.** (a) $J = \begin{pmatrix}2x&2y\\1&-1\end{pmatrix}$. The iterates are $(2,3)\to(1.2,2.2)\to(1.011765,2.011765)\to(1.0000458,2.0000458)\to(1,2)$, with errors $1,\ 0.2,\ 1.2\times10^{-2},\ 4.6\times10^{-5},\ 7\times10^{-10}$. Each error is about the square of the previous one, times a constant near 0.3. The linear equation is satisfied exactly after the first step. (b) $\mathbf F(\mathbf x^{(0)}) = (0, -0.66, 0.145171)$ and $J = \begin{pmatrix}1&1&1\\0.2&2.4&3.4\\1.105171&1.7&1.2\end{pmatrix}$. Solving $J\mathbf s = -\mathbf F$ gives $\mathbf s = (-0.109055, -0.311025, 0.420080)$, so $\mathbf x^{(1)} = (-0.009055, 0.888975, 2.120080)$. The errors are then $0.30, 0.12, 1.1\times10^{-2}, 1.1\times10^{-4}, 1.3\times10^{-8}, 4\times10^{-16}$, which is quadratic.

**B3.** $\mathbf s_1 = (-0.8,-0.8)$, $\mathbf x^{(1)} = (1.2,2.2)$ and $\mathbf y_1 = \mathbf F(\mathbf x^{(1)})-\mathbf F(\mathbf x^{(0)}) = (-6.72, 0)$. The update is
$A_1 = A_0+\frac{(\mathbf y-A_0\mathbf s)\mathbf s^T}{\mathbf s^T\mathbf s} = \begin{pmatrix}3.2&5.2\\1&-1\end{pmatrix}$, compared with $J(\mathbf x^{(1)}) = \begin{pmatrix}2.4&4.4\\1&-1\end{pmatrix}$. The secant equation holds, $A_1\mathbf s_1-\mathbf y_1 = \mathbf 0$. Row 2 stays exact because $F_2$ is linear. Row 1 is off by $(0.8,0.8)$, i.e. only along $\mathbf s$. Then $\mathbf x^{(2)} = (1.047619, 2.047619)$. The errors are $1, 0.2, 0.048, 2.9\times10^{-3}, 4.6\times10^{-5}, 4.5\times10^{-8}, 7\times10^{-13}$: superlinear, with 7 iterations against Newton's 5, but no Jacobian evaluations.

**B4.** From $(0,0)$: $g = 26$ and $\nabla g = (2,-2)$, so the search direction is $(-1,1)/\sqrt2$. The iterates are $(-0.7071,0.7071)$ with $g = 16.17$, then $(-1.4142,1.4142)$ with $g = 4.343$, then $(-1.4726,1.4726)$ with $g = 4.2232$, where the method stops. The iteration never leaves the line $y = -x$, where $\det J = -2(x+y) = 0$, because the gradient $2J^T\mathbf F$ is parallel to $(-1,1)$ along it. It converges to a stationary point of $g$ that is **not** a root (cf. A4). From $(0,1)$, 9 iterations reach $(1.00006, 1.99983)$ with $g = 3.5\times10^{-7}$, and Newton then finishes to machine precision. This is the hybrid strategy.

**B5.** The path for $N = 2$ is $(1,0)\to(1.2569,1.2569)\to(1.0266,2.0266)$.

| $N$ | end-point error |
|---|---|
| 1 | $0.157$ |
| 2 | $0.027$ |
| 4 | $3.1\times10^{-3}$ |
| 8 | $2.6\times10^{-4}$ |

The ratios of about 6–12 approach $2^4$ asymptotically (RK4). A few Newton steps then polish the end point.

**B6.** The errors are $0.417, 0.208, 0.104, \dots$, with ratio exactly $0.5$. At $(1,1)$, $J = \begin{pmatrix}2&2\\1&1\end{pmatrix}$ is singular. The root is double (tangency), just as $f(x) = (x-1)^2$ is for scalar Newton, where $e_{k+1} = \frac12e_k$. Remedies are to deflate, to use $\mathbf x_{k+1} = \mathbf x_k+2\Delta\mathbf x$ (multiplicity 2), or to reformulate the problem.

## C. Programming (key results)

**C1.**

| | Newton | Broyden | hybr (MINPACK) | Newton–Krylov |
|---|---|---|---|---|
| $n = 100$: iterations / $F$-evaluations / time | 5 / 5 / 3 ms | 12 / 12 / 2 ms | — / 115 / 5 ms | — / 119 / 27 ms |
| $n = 1000$ | 5 / 5 / 0.30 s | 12 / 12 / 0.23 s | — / 1015 / 1.24 s | — / 117 / 0.02 s |

All four methods reach $\max|F_i|\le1.5\times10^{-10}$.
- Newton's time at $n = 1000$ is dominated by the dense $O(n^3)$ solves. A tridiagonal solver would make it $O(n)$.
- `hybr` builds a finite-difference Jacobian at a cost of $n$ $F$-evaluations.
- Newton–Krylov never forms $J$: GMRES uses directional differences, so its cost is independent of $n$. This is the method of choice for large sparse systems.

**C2.** The basins take $35.3\%$, $32.3\%$ and $32.3\%$ of the grid. Root $1$ gets slightly more because the grid includes the real axis, which belongs to its basin. Only $10^{-5}$ of the starting points fail, namely points hitting $z = 0$ or landing on the basin boundary. The mean is 7.6 iterations: 10.2 for $|z_0|<0.5$, where the first step throws the point far away, and 7.2 for $0.5\le|z_0|<1.5$. The basin boundaries are fractal (Julia set): arbitrarily close to any boundary point, all three roots are reached. The plot is `solutions/code/figures/ch10_newton_basins.png`.

**C3.** From random starts in $[-10,10]^2$, pure Newton converges from only **15.8%** of the starts, while damped Newton converges from **100%** (mean 8.6 iterations against 4.5 for the successful pure runs). Along the $x$-axis pure Newton converges for $x_0\le1.5$ and diverges from $1.6$ on. For scalar $\arctan$ the critical value is $1.3917$; the coupling shifts it slightly. $\arctan$ is flat for large $|x|$, so the Newton step overshoots, and the overshoot grows until the iteration diverges. Backtracking on $\|\mathbf F\|$ enforces descent and globalises the method.

**C4.**

| $h$ | $10^{-2}$ | $10^{-4}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ | $10^{-12}$ |
|---|---|---|---|---|---|---|
| forward error | $10^{-2}$ | $10^{-4}$ | $10^{-6}$ | $3.7\times10^{-8}$ | $1.6\times10^{-6}$ | $2.8\times10^{-4}$ |
| central error | $1.8\times10^{-5}$ | $1.8\times10^{-9}$ | $3.8\times10^{-10}$ | $3\times10^{-8}$ | $1.6\times10^{-6}$ | $3\times10^{-4}$ |

The forward difference has truncation error $O(h)$ and round-off error $O(u/h)$, so the optimum is at $h\approx\sqrt u\approx10^{-8}$. The central difference has truncation error $O(h^2)$, so its optimum is at $h\approx u^{1/3}\approx10^{-5}$–$10^{-6}$.

## D. Data-science applications

**D1.** The log-likelihood is $\ell = (k-1)\sum\ln x_i-\sum x_i/\theta-nk\ln\theta-n\ln\Gamma(k)$. The score is $(\sum\ln x_i-n\ln\theta-n\psi(k),\ \sum x_i/\theta^2-nk/\theta)$. The Hessian involves the trigamma function $\psi'(k)$. Starting from the method-of-moments estimates $(2.3416, 1.5227)$, Newton takes 5 iterations to reach $\hat k = 2.486025$, $\hat\theta = 1.434208$, identical to `scipy.stats.gamma.fit`. The standard errors are $(0.165, 0.106)$ and the 95% CI for $k$ is $[2.16, 2.81]$, which contains the true 2.5. The estimators are strongly negatively correlated, $-0.90$, because the mean $k\theta$ is well determined but the split between $k$ and $\theta$ is not. A reparametrisation by (mean, shape) decorrelates them.

**D2.** Newton from $\mathbf q = (5,5,5)$ takes 5 iterations to reach $\mathbf q = (8.774, 5.778, 1.284)$, with $Q = 15.837$ and $p = 15.857$. The market shares are $(0.554, 0.365, 0.081)$ and the markups are $(0.369, 0.243, 0.054)$, which equal $s_i/\eta$ exactly: the Lerner condition holds. The Jacobian is $J = p'(Q)(\mathbf 1\mathbf 1^T+I)+p''(Q)\,\mathbf q\mathbf 1^T$. The least efficient firm produces little, because its cost of 15 is close to the price.

**D3.** Newton takes 8 iterations. $\|\nabla\ell\|$ goes $95\to22\to10\to3.6\to0.60\to0.020\to1.8\times10^{-5}\to1.3\times10^{-11}$: a damped-looking start, then the quadratic phase in the last 3 steps. Training accuracy is 96%. BFGS on the same objective needs 27 iterations and agrees to $2.4\times10^{-9}$. The Hessian blocks are $H_{ij} = X^T\operatorname{diag}(p_i(\delta_{ij}-p_j))X+\lambda R$. It is positive definite, so Newton for the score equations is the same as Newton for minimising $-\ell$. For $K$ classes and $p$ features each step solves a $(K-1)p$-dimensional system; that is cheap here but motivates quasi-Newton or SGD methods for large $p$ (Chapter 15).
