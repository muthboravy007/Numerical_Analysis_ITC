# Chapter 15 — Solutions

> Exercises: [`exercises/ch15_exercises.md`](../exercises/ch15_exercises.md) · Numbers from [`solutions/code/ch15_solutions.py`](code/ch15_solutions.py).

## A. Theory and proofs

**A1.** Convexity gives $f(\mathbf x+t(\mathbf y-\mathbf x))\le f(\mathbf x)+t(f(\mathbf y)-f(\mathbf x))$. Subtract $f(\mathbf x)$, divide by $t$, and let $t\downarrow0$: $\nabla f(\mathbf x)^T(\mathbf y-\mathbf x)\le f(\mathbf y)-f(\mathbf x)$. If $\nabla f(\mathbf x) = \mathbf 0$ then $f(\mathbf y)\ge f(\mathbf x)$ for all $\mathbf y$. For uniqueness under strong convexity, suppose $\mathbf x\ne\mathbf y$ are both minimisers. Then $f(\frac{\mathbf x+\mathbf y}2)\le f^\ast-\frac\mu8\|\mathbf x-\mathbf y\|^2<f^\ast$, which is a contradiction.

**A2.** (a) $f(\mathbf y)-f(\mathbf x)-\nabla f(\mathbf x)^T(\mathbf y-\mathbf x) = \int_0^1[\nabla f(\mathbf x+t(\mathbf y-\mathbf x))-\nabla f(\mathbf x)]^T(\mathbf y-\mathbf x)dt\le\int_0^1Lt\|\mathbf y-\mathbf x\|^2dt$. (b) Put $\mathbf y = \mathbf x-\frac1L\nabla f$ in (a). (c) Combine (b) with convexity, $f(\mathbf x)\le f^\ast+\nabla f^T(\mathbf x-\mathbf x^\ast)$, to get $f(\mathbf x_{k+1})-f^\ast\le\frac L2(\|\mathbf x_k-\mathbf x^\ast\|^2-\|\mathbf x_{k+1}-\mathbf x^\ast\|^2)$. Summing telescopes, and $f(\mathbf x_k)$ is monotone. (d) Strong convexity gives the Polyak–Łojasiewicz inequality $\|\nabla f\|^2\ge2\mu(f-f^\ast)$. With (b), $f_{k+1}-f^\ast\le(1-\frac\mu L)(f_k-f^\ast)$.

**A3.** $\mathbf e_{k+1} = (I-\alpha A)\mathbf e_k$, and the eigenvalues of $I-\alpha A$ are $1-\alpha\lambda_i$. The worst case is $\max(|1-\alpha\mu|,|1-\alpha L|)$. This is minimised where the two terms are equal, $1-\alpha\mu = \alpha L-1$, i.e. $\alpha = \frac2{\mu+L}$, giving $\frac{L-\mu}{L+\mu}$. For $\alpha>2/L$, $|1-\alpha L|>1$ and the $L$-component grows.

**A4.** (a) $\nabla g(\mathbf y) = B^T\nabla f(B\mathbf y)$ and $\nabla^2g = B^T\nabla^2fB$. So the Newton step is $\Delta\mathbf y = -B^{-1}(\nabla^2f)^{-1}B^{-T}B^T\nabla f = B^{-1}\Delta\mathbf x$. (b) $\mathbf x-A^{-1}(A\mathbf x-\mathbf b) = A^{-1}\mathbf b$. (c) Far from the minimiser, the quadratic model can be a poor fit, or indefinite, so the Newton step may overshoot (B3: $f$ jumps to 1410). Backtracking on $f$ along the Newton direction, which is a descent direction when $\nabla^2f\succ0$, restores monotone decrease. Trust regions limit the step to where the model is trusted.

**A5.** (a) $(I-\rho\mathbf y\mathbf s^T)\mathbf y = \mathbf y-\rho\mathbf y(\mathbf s^T\mathbf y) = \mathbf 0$, so $H_+\mathbf y = \rho\mathbf s\mathbf s^T\mathbf y = \mathbf s$. (b) Let $\mathbf w = (I-\rho\mathbf y\mathbf s^T)\mathbf z$. Then $\mathbf z^TH_+\mathbf z = \mathbf w^TH\mathbf w+\rho(\mathbf s^T\mathbf z)^2\ge0$. Equality would need both $\mathbf w = \mathbf 0$ and $\mathbf s^T\mathbf z = 0$, but then $\mathbf w = \mathbf z$, so $\mathbf z = \mathbf 0$. (c) $\mathbf y^T\mathbf s = \alpha(\nabla f_+-\nabla f)^T\mathbf p\ge\alpha(c_2-1)\nabla f^T\mathbf p>0$, since $c_2<1$ and $\nabla f^T\mathbf p<0$.

**A6.** (a) The Lagrangian is $\frac12\|\mathbf x-\mathbf v\|^2-\boldsymbol\mu^T\mathbf x+\tau(\mathbf 1^T\mathbf x-1)$. Stationarity gives $x_i = v_i-\tau+\mu_i$, with $\mu_i\ge0$ and $\mu_ix_i = 0$. So $x_i = \max(v_i-\tau,0)$, where $\tau$ is chosen so that $\sum x_i = 1$. Sorting $\mathbf v$ in decreasing order, $\tau = \frac{\sum_{j\le\rho}u_j-1}{\rho}$ with $\rho$ the largest $j$ such that $u_j>\frac{\sum_{i\le j}u_i-1}{j}$. (b) Stationarity of $\frac12\mathbf x^TQ\mathbf x-\mathbf c^T\mathbf x+\boldsymbol\lambda^T(A\mathbf x-\mathbf b)$ is $Q\mathbf x+A^T\boldsymbol\lambda = \mathbf c$. Together with feasibility $A\mathbf x = \mathbf b$ this is the saddle-point system, which is symmetric but indefinite.

**A7.** (a) $E[\mathbf g_k] = \frac1n\sum_i\nabla f_i(\mathbf x_k) = \nabla f(\mathbf x_k)$. (b) $x_{k+1}-\bar a = (1-\eta)(x_k-\bar a)+\eta(\bar a-a_{i_k})$, where the last term has mean zero and variance $\eta^2\sigma_a^2$ and is independent of $x_k$. The variance recursion $V_{k+1} = (1-\eta)^2V_k+\eta^2\sigma_a^2$ has fixed point $\frac{\eta^2\sigma_a^2}{1-(1-\eta)^2} = \frac{\eta\sigma_a^2}{2-\eta}$. (c) $\sum\eta_k = \infty$ lets the iterates travel any distance, and $\sum\eta_k^2<\infty$ makes the accumulated noise variance finite and vanishing. An example is $\eta_k = c/k$.

**A8.** Reverse mode records the evaluation graph, and then one backward sweep applies the chain rule $\bar v_i = \sum_{j\in\text{children}}\bar v_j\,\partial v_j/\partial v_i$. Each elementary operation contributes a constant amount of work, so the total cost is a small multiple (about 3–5) of one evaluation, independent of $n$. It needs memory proportional to the graph. Forward mode propagates one directional derivative per sweep, so the full gradient needs $n$ sweeps, and finite differences need $n$ or $2n$ evaluations and are also inexact. This is why backpropagation (reverse mode) makes training networks with $10^9$ parameters feasible.

## B. Hand computation

**B1.** The golden-section brackets shrink by $0.618$ each step: $[0,1.2361]$, $[0.4721,1.2361]$, $[0.4721,0.9443]$, $[0.6525,0.9443]$, $[0.6525,0.8328]$ (width 0.180). The true minimiser, the root of $x = \cos x$, is $0.739085$ with $f^\ast = -0.800977$. The parabolic step through $0,1,2$ gives $0.6925$. Golden section needs 40 steps to reach width $10^{-8}$: it is robust, but linear with rate $0.618$.

**B2.** With $A = \operatorname{diag}(2,10)$:
- Fixed step 0.1 gives $(4,0)$, $(3.2,0)$, $(2.56,0)$. The $y$-factor is $1-0.1\cdot10 = 0$, so $y$ is solved in one step, while $x$ shrinks by $0.8$ per step.
- Exact line search uses $t = 1/6$ each time: $(3.333,-0.667)$, $(2.222,0.444)$, $(1.481,-0.296)$, a zig-zag with $f$ falling by the factor $\frac49 = (\frac{\kappa-1}{\kappa+1})^2$ per step.
- The optimal fixed step is $2/(2+10) = 1/6$ (rate $2/3$).

Iterations to $\|\nabla f\|<10^{-8}$:

| step | iterations |
|---|---|
| 0.1 | 93 |
| 1/6 | 52 |
| 0.19 | 197 (factor $\lvert1-1.9\rvert = 0.9$) |
| 0.21 | diverges ($0.21>2/L = 0.2$) |

**B3.** At $(-1.2,1)$: $f = 24.2$ and $\nabla f = (-215.6,-88)$. The Hessian is $\begin{pmatrix}1330&480\\480&200\end{pmatrix}$ with eigenvalues $23.6$ and $1506$, so it is PD. The Newton step is $(0.0247, 0.3807)$, giving $\mathbf x_1 = (-1.1753, 1.3807)$ with $f = 4.73$. Pure Newton takes 7 steps, with $f$ values $24.2, 4.73, \mathbf{1410}, 0.056, 0.313, 1.9\times10^{-11}, 3\times10^{-20}, 0$. $f$ is *not* monotone, since the third iterate jumps far up the valley wall. The final quadratic convergence is evident. A damped Newton would reject the bad step.

**B4.** Step 1: $\mathbf p = -(1,5)$, $t = 0.20635$, $\mathbf s = (-0.2063,-1.0317)$, $\mathbf y = A\mathbf s$, giving $\mathbf x_1 = (0.7937,-0.0317)$ and $H_1 = \begin{pmatrix}1.0315&-0.0013\\-0.0013&0.2001\end{pmatrix}$, with $H_1\mathbf y = \mathbf s$. Step 2: $\mathbf p = (-0.8188,0.0328)$ and $t = 0.9692$, giving $\mathbf x_2 = (0,0)$ exactly and $H_2 = \operatorname{diag}(1,0.2) = A^{-1}$. With exact line searches on a quadratic, BFGS generates conjugate directions and terminates in $n$ steps.

**B5.** (a) Stationarity gives $(2x,2y) = \mu(1,2)$, and the constraint is active: $x+2y = 4$. Hence $(x,y) = (0.8,1.6)$ with $\mu = 1.6\ge0$. This is the projection of the origin onto the half-plane. (b) Stationarity gives $(1,1)+2\mu(x,y) = 0$, and the constraint is active, so $(x,y) = (-1,-1)$ with $\mu = \frac12$. SLSQP confirms both.

**B6.** Sorted, $\mathbf u = (0.8,0.6,0.4,-0.2)$ with cumulative sums $(0.8,1.4,1.8,1.6)$. The test $u_j-\frac{c_j-1}{j} = (1, 0.4, 0.133, -0.35)$ gives $\rho = 3$, so $\tau = \frac{1.8-1}3 = 0.2667$. The projection is $\mathbf x = (0.5333, 0.3333, 0, 0.1333)$, which sums to 1. The negative entry and the smallest positive one are clipped.

**B7.** The graph is $v_1 = x_1x_2 = 2$, $v_2 = \sin x_1 = 0.841471$, $v_3 = v_1+v_2 = 2.841471$, $v_4 = e^{x_2} = 7.389056$, and $f = v_3v_4 = 20.995789$.
- Forward mode with seed $\dot x_1 = 1$: $\dot v_1 = x_2 = 2$, $\dot v_2 = \cos1 = 0.540302$, $\dot v_3 = 2.540302$, $\dot f = \dot v_3v_4 = 18.770436$.
- Reverse mode: $\bar v_3 = v_4 = 7.389056$ and $\bar v_4 = v_3 = 2.841471$. Then $\bar x_1 = \bar v_3(x_2+\cos x_1) = 18.770436$ and $\bar x_2 = \bar v_3x_1+\bar v_4e^{x_2} = 28.384845$.

Central differences agree to 6 digits. Reverse mode obtained both partials in one sweep.

**B8.** The roots of $r^2-(1+\beta-\alpha\lambda)r+\beta = 0$:

| $(\alpha,\beta)$ | $\lambda = 1$: max $\lvert r\rvert$ | $\lambda = 10$: max $\lvert r\rvert$ |
|---|---|---|
| (0.1, 0) | 0.900 | 0 (the step equals $1/\lambda$) |
| (0.1, 0.5) | 0.707 | 0.707 |
| (0.1, 0.9) | 0.949 | 0.949 |
| (0.15, 0.9) | 0.949 | 0.949 |

When the roots are complex, $|r| = \sqrt\beta$ regardless of $\alpha$. Too much momentum gives slow, oscillating convergence. The optimum for $[\mu,L] = [1,10]$ is $\alpha = \frac4{(\sqrt L+\sqrt\mu)^2} = 0.2309$ and $\beta = \bigl(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\bigr)^2 = 0.2699$. This gives rate $0.519$, against $0.818$ for optimal GD: $\sqrt\kappa$ replaces $\kappa$.

## C. Programming (key results)

**C1.** Rosenbrock:

| method | iterations | $f$-evals | $\nabla f$-evals | error | time |
|---|---|---|---|---|---|
| GD, fixed $10^{-3}$ | 32 076 | 0 | 32 077 | $2.5\times10^{-6}$ | 0.30 s |
| GD, backtracking | 13 756 | 150 555 | 13 757 | $1.8\times10^{-6}$ | 0.59 s |
| Nesterov ($5\times10^{-4}$, 0.95) | 2 908 | 0 | 5 818 | $2.5\times10^{-6}$ | 0.04 s |
| Newton | 6 | 0 | 7 (+ Hessians) | $2\times10^{-11}$ | 0.001 s |
| BFGS | 34 | 88 | 104 | $2\times10^{-10}$ | 0.003 s |
| L-BFGS ($m = 5$) | 36 | 185 | 121 | $3\times10^{-8}$ | 0.003 s |

Quadratic with $\kappa = 10^3$, $n = 100$: GD needs 14 230 iterations and backtracking GD 6 331. Nesterov needs **533**. Newton needs 1. BFGS needs 100 ($= n$, as expected on a quadratic) and L-BFGS 281. Nesterov with $\beta = 0.99$ and step $10^{-3}$ *diverged* on Rosenbrock in our first attempt: momentum methods need a step comfortably below $1/L$, and $L$ varies along the valley.

**C2.**

| $\kappa$ | GD | heavy ball | CG | $\frac\kappa2\ln10^6$ | $\frac{\sqrt\kappa}2\ln10^6$ |
|---|---|---|---|---|---|
| 10 | 61 | 25 | 21 | 69 | 22 |
| 100 | 594 | 82 | 47 | 691 | 69 |
| 1000 | 6152 | 245 | 82 | 6908 | 218 |
| 10000 | 61183 | 963 | 137 | 69078 | 691 |

GD scales like $\kappa$ and heavy ball like $\sqrt\kappa$, both matching theory. CG does even better: it adapts to the actual spectrum and, with 50 distinct eigenvalues, terminates in at most 50 steps in exact arithmetic.

**C3.** The relative error $\frac{\|g_{fd}-g\|}{\|g_{fd}+g\|}$ is $3.6\times10^{-6}$, $3.6\times10^{-10}$, $1.1\times10^{-10}$, $9.6\times10^{-9}$ and $1.1\times10^{-6}$ for $h = 10^{-2},\dots,10^{-10}$. The best $h$ is about $10^{-5}$–$10^{-6}$, as for central differences in Chapter 10 C4. The planted bug produces $1.9\times10^{-3}$, orders of magnitude above the round-off floor, so the check clearly flags it. Rule of thumb: $<10^{-7}$ is fine and $>10^{-4}$ is almost surely a bug.

**C4.** The loss is $0.369811$, identical to numpy. The largest difference between AD and finite differences over the 12 parameters is $1.1\times10^{-10}$. The implementation is under 40 lines, and the same design, scaled up with tensor operations, is PyTorch's autograd.

## D. Data-science applications

**D1.** $f^\ast = 0.05982947$. The smoothness constant is $L = \lambda_{\max}(X^TX)/4n+\lambda = 3.32$ and the strong-convexity constant is $\mu\ge\lambda = 10^{-3}$, so $\kappa\approx3300$.

| method | iterations to relative gap $10^{-6}$ |
|---|---|
| GD ($1/L$) | 13 528 |
| Nesterov | 1 303 |
| **Newton** | **8** |
| BFGS | 96 |
| L-BFGS | 37 |

Stochastic methods, relative gap after 10 / 50 / 100 epochs:

| method | 10 | 50 | 100 |
|---|---|---|---|
| SGD, decaying step | $1.7\times10^{-1}$ | $6\times10^{-2}$ | $4\times10^{-2}$ |
| SGD, constant step | $1.2\times10^{-1}$ | $1.3\times10^{-2}$ | $3.3\times10^{-3}$ |
| Adam | $3.4\times10^{-1}$ | $3.3\times10^{-2}$ | $5.6\times10^{-3}$ |

With $p = 31$, Newton is unbeatable. The Hessian is cheap and the problem is smooth and strongly convex. Stochastic methods reach only moderate accuracy, which is often enough statistically: the estimation error is about $1/\sqrt n\approx0.04$. They win when $n$ is huge. The decaying step shrinks too fast here ($\eta_{100} = 0.045$).

**D2.**

| optimiser | learning rate | train accuracy | test accuracy |
|---|---|---|---|
| SGD | 0.01 | 0.880 | 0.890 |
| SGD | 0.1 | 0.967 | 0.964 |
| SGD | 1.0 | 0.970 | **0.969** |
| Adam | 0.001 | 0.958 | 0.955 |
| Adam | 0.01 | 0.970 | 0.967 |
| Adam | 0.1 | 0.972 | 0.963 |

Linear logistic regression reaches only $0.867$. The hidden layer learns the curved boundary. A learning rate that is too small under-trains in 200 epochs. Adam is less sensitive to the learning rate, and its default $10^{-3}$ is slow here. The noise level (0.2) limits achievable accuracy to about 97%.

**D3.** Projected gradient with step $1/L$ takes 335 iterations to reach $\mathbf w = (0.183, 0.117, 0, 0.148, 0, 0.179, 0.090, 0.284)$ with variance $0.005562$. SLSQP gives the identical result. The unconstrained optimum shorts assets 3 and 5 ($-0.018$ and $-0.047$) and has the lower variance $0.005385$. KKT check: $(2\Sigma\mathbf w)_i = 0.011124$, equal to $-\tau$, for every held asset, and it is larger ($0.0148$, $0.0173$) for the two excluded assets. Adding those assets would increase the variance, which is exactly complementary slackness.

**D4.** $L = 4.024$.

| iteration | ISTA gap | FISTA gap |
|---|---|---|
| 10 | 7.7 | 3.2 |
| 100 | $1.9\times10^{-2}$ | $3.9\times10^{-7}$ |
| 1000 | $\approx0$ | $\approx0$ |

Coordinate descent gives $f^\ast = 1533.7687$ with 7 nonzeros. Both methods beat their worst-case bounds of $O(1/k)$ and $O(1/k^2)$ because the lasso here is locally strongly convex on the active set, and ISTA then converges linearly. FISTA is far ahead at intermediate accuracy. Proximal gradient is the workhorse for $\ell_1$, nuclear-norm and other nonsmooth regularisers.
