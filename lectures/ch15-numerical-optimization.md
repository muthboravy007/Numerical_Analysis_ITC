# Chapter 15 — Numerical Optimization for Data Science

> **Data-science chapter** extending Burden & Faires §10.4 (steepest descent) and building on Chapters 2, 5, 7, 10, 13 and 14.
> **Code:** [`numlib/optimization.py`](../numlib/optimization.py) · **All numbers reproduced by** [`lectures/code/ch15.py`](code/ch15.py)
> **Worked problems:** [`examples/ch15`](../examples/ch15_examples.md) · **Homework:** [`exercises/ch15`](../exercises/ch15_exercises.md)

## Learning outcomes

1. Formulate model fitting as (regularised) empirical-risk minimisation; state first- and second-order optimality conditions; recognise convexity, strong convexity and the condition number $\kappa = L/\mu$.
2. Minimise one-dimensional functions by golden-section search, parabolic interpolation and Newton's method; use exact and inexact (Armijo, Wolfe) line searches.
3. Analyse gradient descent ($1-1/\kappa$ rate) and accelerated methods (heavy-ball, Nesterov, $1-1/\sqrt\kappa$).
4. Apply Newton's method (with Hessian modification) and quasi-Newton methods (BFGS, L-BFGS).
5. Use stochastic gradient methods (SGD, mini-batches, momentum, Adam) and choose learning rates and schedules.
6. Solve constrained problems via Lagrange multipliers/KKT conditions, projected gradients, penalties and linear/quadratic programming.
7. Compute gradients by automatic differentiation (forward and reverse mode) and verify them by gradient checking.

---

## 15.1 Optimization Problems in Data Science

Training a model means solving
$$
\min_{\mathbf w\in\mathbb R^d}F(\mathbf w) = \frac1n\sum_{i=1}^n\ell(\mathbf w;\mathbf x_i,y_i) + \lambda R(\mathbf w)
$$
(least squares, logistic regression, SVMs, neural networks, matrix factorisation, $k$-means, maximum likelihood).

**Optimality conditions.** If $\mathbf w^\ast$ is a local minimiser of a $C^2$ function then $\nabla F(\mathbf w^\ast) = \mathbf 0$ and $\nabla^2F(\mathbf w^\ast)\succeq0$. Conversely $\nabla F = \mathbf 0$ with $\nabla^2F\succ0$ implies a strict local minimum. (Indefinite Hessian ⇒ saddle point.)

**Convexity.** $F$ is convex iff $F(\theta\mathbf a+(1-\theta)\mathbf b)\le\theta F(\mathbf a)+(1-\theta)F(\mathbf b)$; for $C^2$ functions iff $\nabla^2F\succeq0$ everywhere. **Every local minimiser of a convex function is global.** Least squares, ridge, lasso, logistic regression and SVMs are convex; neural networks and matrix factorisation are not.

**Smoothness and strong convexity.** If $\mu I\preceq\nabla^2F\preceq LI$ then $F$ is $\mu$-strongly convex and $L$-smooth, with **condition number** $\kappa = L/\mu$. For a quadratic $\frac12\mathbf w^TA\mathbf w - \mathbf b^T\mathbf w$ this is $\kappa_2(A)$ — the link to Chapters 7 and 14.

### Examples for §15.1

**Example 15.1.1 (least squares as optimisation).** $F(\mathbf w) = \frac15\sum(\beta_0+\beta_1x_i - y_i)^2$ for the data of Example 6.7.5: $\nabla F(\mathbf 0) = (-7.28, -24.84)$, $\nabla^2F = \frac25X^TX = \begin{pmatrix}2&6\\6&22\end{pmatrix}$ with eigenvalues $0.338, 23.66$ (PD ⇒ strictly convex, $\kappa = 70$). The unique minimiser $(1.39, 0.75)$ gives $F^\ast = 0.0054$.

**Example 15.1.2 (classifying critical points).** $f(x,y) = x^3 - 3x + y^2$: $\nabla f = (3x^2-3, 2y) = \mathbf 0$ at $(\pm1, 0)$. Hessian eigenvalues at $(1,0)$: $2, 6$ (minimum); at $(-1,0)$: $-6, 2$ (saddle). $f$ is unbounded below, so $(1,0)$ is only a local minimum.

**Example 15.1.3 (logistic loss is convex).** $\nabla^2F = \frac1nX^TDX$ with $D = \operatorname{diag}(p_i(1-p_i))\succ0$ is PSD for every $\mathbf w$; at a random $\mathbf w$ (30 samples, 3 features) its eigenvalues are $0.073, 0.112, 0.257$.

**Example 15.1.4 (a non-convex function).** $f(x) = x^4 - 3x^2 + x$ has critical points $-1.3008$ (global minimum, $f = -3.514$, $f'' = 14.3$), $0.1699$ (local maximum, $f'' = -5.65$) and $1.1309$ (local minimum, $f = -1.070$, $f'' = 9.35$). A local method started at $x_0>0.17$ finds the worse minimum.

**Example 15.1.5 (condition number of a quadratic).** $F = \frac12(10w_1^2 + w_2^2)$: $\mu = 1$, $L = 10$, $\kappa = 10$. Its level sets are ellipses with axis ratio $\sqrt{10}$; the larger $\kappa$, the narrower the valley and the slower first-order methods become (§15.3).

---

## 15.2 One-Dimensional Minimisation and Line Searches

**Golden-section search.** For $f$ unimodal on $[a,b]$, evaluate at $c = b - \varphi^{-1}(b-a)$ and $d = a + \varphi^{-1}(b-a)$, $\varphi^{-1} = \frac{\sqrt5-1}{2} = 0.618$; if $f(c)<f(d)$ keep $[a,d]$, else $[c,b]$. One of the two interior points is re-used, so each step costs **one** evaluation and shrinks the interval by $0.618$ (linear convergence, guaranteed, derivative-free).

**Parabolic interpolation.** Fit a parabola through three points and jump to its vertex:
$$
x_{\text{new}} = b - \frac12\,\frac{(b-a)^2[f(b)-f(c)] - (b-c)^2[f(b)-f(a)]}{(b-a)[f(b)-f(c)] - (b-c)[f(b)-f(a)]}.
$$
Superlinear (order ≈1.32) near a smooth minimum. **Brent's method** (`scipy.optimize.minimize_scalar`) combines both safely.

**Newton's method in 1-D:** $x_{k+1} = x_k - f'(x_k)/f''(x_k)$ (quadratic convergence if $f''(x^\ast)>0$).

**Line searches in $\mathbb R^d$.** Along a descent direction $\mathbf p$ ($\nabla F^T\mathbf p<0$) choose a step $\alpha$:
* exact (quadratics only): $\alpha = -\frac{\mathbf g^T\mathbf p}{\mathbf p^TA\mathbf p}$;
* **Armijo (sufficient decrease)**, by backtracking $\alpha = 1, \frac12, \frac14,\ldots$: $F(\mathbf w+\alpha\mathbf p)\le F(\mathbf w) + c_1\alpha\nabla F^T\mathbf p$ ($c_1 = 10^{-4}$);
* **Wolfe** conditions add the curvature condition $\nabla F(\mathbf w+\alpha\mathbf p)^T\mathbf p\ge c_2\nabla F^T\mathbf p$ ($c_2 = 0.9$), which guarantees $\mathbf s^T\mathbf y>0$ for quasi-Newton updates (§15.4).

### Examples for §15.2

**Example 15.2.1 (golden section).** Minimise $f(x) = x^2 - \sin x$ on $[0,2]$:

| $k$ | $[a,b]$ | $c$ | $d$ | $f(c)$ | $f(d)$ |
|---|---|---|---|---|---|
| 0 | $[0, 2]$ | 0.763932 | 1.236068 | −0.108174 | 0.583364 |
| 1 | $[0, 1.236068]$ | 0.472136 | 0.763932 | −0.231877 | −0.108174 |
| 2 | $[0, 0.763932]$ | 0.291796 | 0.472136 | −0.202528 | −0.231877 |
| 3 | $[0.291796, 0.763932]$ | 0.472136 | 0.583592 | −0.231877 | −0.210445 |
| 4 | $[0.291796, 0.583592]$ | 0.403252 | 0.472136 | −0.229799 | −0.231877 |

Note how one point is re-used each step. After 40 steps (tolerance $10^{-8}$): $x^\ast = 0.4501836$, the root of $f'(x) = 2x - \cos x = 0$.

**Example 15.2.2 (parabolic interpolation).** Through $x = 0, 0.5, 1$ the vertex is $0.43581$; one more step (with $0.5, 0.43581, 1$) gives $0.45117$ — two evaluations for 2–3 correct digits.

**Example 15.2.3 (Newton in 1-D).** $f(x) = x^4 - 3x^3 + 2$: $x_0 = 3$: $2.5,\ 2.29167,\ 2.251462,\ 2.2500019,\ 2.2500000000032$ → $x^\ast = \frac94$ (quadratic convergence).

**Example 15.2.4 (tuning a hyperparameter).** The GCV criterion of ridge regression (§13.5) as a function of $\log_{10}\lambda$ is smooth and unimodal on $[-4,3]$ for a collinear dataset. Golden-section search finds $\lambda^\ast = 0.551$ (GCV $0.26825$) with 24 evaluations; a 701-point grid gives the same minimum value — 30× fewer model fits.

**Example 15.2.5 (exact line search on a quadratic).** $F = \frac12\mathbf w^T\operatorname{diag}(1,10)\mathbf w$ at $\mathbf w = (10,1)$: $\mathbf g = (10,10)$, $\alpha = \frac{\mathbf g^T\mathbf g}{\mathbf g^TA\mathbf g} = \frac{200}{1100} = 0.181818$, new point $(8.1818, -0.8182)$.

---

## 15.3 Gradient Descent and Acceleration

$$
\mathbf w_{k+1} = \mathbf w_k - \alpha_k\nabla F(\mathbf w_k).
$$
**Theorem 15.1.** If $F$ is $L$-smooth and $\mu$-strongly convex, gradient descent with $\alpha = 1/L$ satisfies
$$
F(\mathbf w_k) - F^\ast\le\Bigl(1-\frac1\kappa\Bigr)^k\bigl(F(\mathbf w_0)-F^\ast\bigr),
$$
so about $\kappa\ln(1/\varepsilon)$ iterations are needed. For merely convex $L$-smooth $F$: $F(\mathbf w_k) - F^\ast\le\frac{L\|\mathbf w_0-\mathbf w^\ast\|^2}{2k}$. (Step $\alpha>2/L$ diverges — the Euler stability limit of Chapter 5.)

**Momentum.** *Heavy ball* (Polyak): $\mathbf v_{k+1} = \beta\mathbf v_k - \alpha\nabla F(\mathbf w_k)$, $\mathbf w_{k+1} = \mathbf w_k+\mathbf v_{k+1}$; with $\alpha = \frac{4}{(\sqrt L+\sqrt\mu)^2}$, $\beta = \bigl(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\bigr)^2$ it converges like $\bigl(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\bigr)^k$ on quadratics. *Nesterov* evaluates the gradient at the look-ahead point $\mathbf w_k+\beta\mathbf v_k$ and achieves $1-1/\sqrt\kappa$ (strongly convex) and $O(1/k^2)$ (convex) — optimal among first-order methods. These are the optimisation analogues of CG (Chapter 7).

### Examples for §15.3

**Example 15.3.1 (zig-zagging).** $F = \frac12(w_1^2+10w_2^2)$ from $(10,1)$ with exact line search: every step has $\alpha = 2/11$ and the iterates are $(8.182, -0.818)$, $(6.694, 0.669)$, $(5.477, -0.548)$, $(4.481, 0.448)$ — bouncing across the valley, with $F$ reduced by the factor $\bigl(\frac{\kappa-1}{\kappa+1}\bigr)^2 = 0.669$ per step.

**Example 15.3.2 (iterations grow linearly with $\kappa$).** $F = \frac12(w_1^2+\kappa w_2^2)$, $\alpha = 1/L$, stop at $\|\nabla F\|<10^{-6}$: $\kappa = 10$: 132 iterations; $100$: 1375; $1000$: 13 809 (theory $\approx\kappa\ln(\sqrt2\cdot10^6)$: 142, 1416, 14 162).

**Example 15.3.3 (Rosenbrock's valley).** $F = (1-x)^2 + 100(y-x^2)^2$ from $(-1.2, 1)$ with Armijo backtracking: $F = 4.3\times10^{-3}$ after 100 iterations, $7.5\times10^{-4}$ after 1000, $2.7\times10^{-10}$ after 10 000; 19 435 iterations to $\|\nabla F\|<10^{-8}$. The curved, narrow valley has local $\kappa\approx2500$.

**Example 15.3.4 (acceleration).** $\kappa = 100$, tolerance $10^{-8}$: gradient descent ($\alpha = 1/L$) 1833 iterations; heavy ball ($\alpha = 0.0331$, $\beta = 0.669$) 143; Nesterov ($\alpha = 1/L$, $\beta = \frac{\sqrt\kappa-1}{\sqrt\kappa+1}$) 204 — an order of magnitude fewer, as $\sqrt\kappa = 10$ predicts.

**Example 15.3.5 (feature scaling).** Least squares with two features of scale 1 and 100: $\kappa(X^TX/n) = 10\,088$, gradient descent needs 146 202 iterations; after standardisation $\kappa = 1.2$ and it needs 8. Standardisation is the cheapest "optimiser" improvement there is.

---

## 15.4 Newton and Quasi-Newton Methods

**Newton's method** minimises the local quadratic model:
$$
\nabla^2F(\mathbf w_k)\,\mathbf p_k = -\nabla F(\mathbf w_k),\qquad\mathbf w_{k+1} = \mathbf w_k+\alpha_k\mathbf p_k .
$$
Quadratically convergent near a minimiser with $\nabla^2F\succ0$ and **affine invariant** (insensitive to scaling/conditioning). Costs: Hessian ($O(nd^2)$ for GLMs) and a $d\times d$ solve ($O(d^3)$). If the Hessian is indefinite, $\mathbf p_k$ may not be a descent direction and pure Newton can converge to saddles or maxima: **modify** it, e.g. $\nabla^2F + \tau I$ with $\tau$ making it PD (Levenberg–Marquardt / trust-region idea), and use a line search. For logistic regression Newton = IRLS (§13.8).

**BFGS.** Maintain $H_k\approx(\nabla^2F)^{-1}$; with $\mathbf s_k = \mathbf w_{k+1}-\mathbf w_k$, $\mathbf y_k = \nabla F_{k+1}-\nabla F_k$, $\rho_k = 1/\mathbf y_k^T\mathbf s_k$:
$$
H_{k+1} = (I-\rho_k\mathbf s_k\mathbf y_k^T)H_k(I-\rho_k\mathbf y_k\mathbf s_k^T) + \rho_k\mathbf s_k\mathbf s_k^T .
$$
It satisfies the secant condition $H_{k+1}\mathbf y_k = \mathbf s_k$ (cf. Broyden, §10.3), stays PD if $\mathbf y_k^T\mathbf s_k>0$ (guaranteed by a **Wolfe** line search), converges superlinearly, costs $O(d^2)$ per iteration.

**L-BFGS** keeps only the last $m$ pairs and applies $H_k$ by the **two-loop recursion** in $O(md)$ — the default for medium/large smooth problems (`scipy.optimize.minimize(method="L-BFGS-B")`, scikit-learn's `LogisticRegression`).

### Examples for §15.4

**Example 15.4.1 (Newton on Rosenbrock).** From $(-1.2,1)$ (pure Newton, no line search):

| $k$ | $\mathbf w_k$ | $F$ | $\lVert\nabla F\rVert$ |
|---|---|---|---|
| 1 | $(-1.1753, 1.3807)$ | 4.73 | 4.6 |
| 2 | $(0.7631, -3.1750)$ | 1412 | 1370 |
| 3 | $(0.7634, 0.5828)$ | 0.056 | 0.47 |
| 5 | $(0.999996, 0.999991)$ | $1.9\times10^{-11}$ | $8.6\times10^{-6}$ |
| 6 | $(1, 1)$ | $3\times10^{-20}$ | $8\times10^{-9}$ |

Note the wild step 2 (a line search would prevent the increase), then quadratic convergence: 7 iterations vs 19 435 for gradient descent.

**Example 15.4.2 (quasi-Newton).** Rosenbrock from $(-1.2,1)$: BFGS (Wolfe line search) 35 iterations, L-BFGS ($m = 5$) 37, SciPy's BFGS 32 — no Hessians required.

**Example 15.4.3 (logistic regression).** 200 samples, 3 parameters: Newton/IRLS converges in 6 iterations to $\hat{\mathbf w} = (-0.422, 1.998, -1.357)$; gradient descent with backtracking needs 356.

**Example 15.4.4 (when Newton goes wrong).** $f = x^4-2x^2+y^2$ at $(0.3, 0.5)$: the Hessian has eigenvalues $-2.92, 2$ (indefinite). The Newton direction $(-0.374, -0.5)$ still has $\nabla f^T\mathbf p = -0.092<0$, but its $x$-component *increases* $f$ ($g_xp_x = +0.41$): it heads for the saddle, and pure Newton converges to $(0,0)$ — a saddle point. The modified Newton method ($\tau$ chosen so that the smallest eigenvalue is 1) moves to $(1.392, 0.331)$, $(1.121, 0)$, $(1.017, 0)$, $(1.0004,0)$ → the minimiser $(1,0)$.

**Example 15.4.5 (cost vs iterations).** A 300-dimensional strongly convex quadratic ($\kappa = 5$): Newton 1 iteration (it solves the linear system exactly); BFGS 23 iterations; L-BFGS ($m = 10$) 21 iterations and the fastest wall-clock time, since each iteration is $O(md)$ instead of $O(d^2)$ or $O(d^3)$.

---

## 15.5 Stochastic Gradient Methods

For $F = \frac1n\sum f_i$ with huge $n$, use an unbiased estimate of the gradient from a random **mini-batch** $B_k$:
$$
\mathbf w_{k+1} = \mathbf w_k - \alpha_k\frac{1}{|B_k|}\sum_{i\in B_k}\nabla f_i(\mathbf w_k).
$$
* Cost per step $O(|B|d)$ instead of $O(nd)$; one **epoch** = one pass over the data.
* With a constant step the iterates reach a noise ball of radius $\propto\alpha\sigma^2/|B|$; convergence to the optimum needs decreasing steps ($\sum\alpha_k = \infty$, $\sum\alpha_k^2<\infty$, e.g. $\alpha_k = \frac{\alpha_0}{1+\gamma k}$) or growing batches. Rate $O(1/k)$ for strongly convex problems.
* In practice: constant rate with step decay/cosine schedules, momentum, and **Adam**:
$$
\mathbf m_k = \beta_1\mathbf m_{k-1}+(1-\beta_1)\mathbf g_k,\quad\mathbf v_k = \beta_2\mathbf v_{k-1}+(1-\beta_2)\mathbf g_k^{2},\quad\mathbf w_{k+1} = \mathbf w_k - \alpha\frac{\mathbf m_k/(1-\beta_1^k)}{\sqrt{\mathbf v_k/(1-\beta_2^k)}+\varepsilon}
$$
(per-coordinate step sizes = a cheap diagonal preconditioner).

### Examples for §15.5

Linear regression with $n = 5000$, $d = 5$; distances to the least-squares solution $\mathbf w_{LS}$.

**Example 15.5.1 (step size and decay).** Plain SGD (batch 1): $\alpha = 0.01$: $\|\mathbf w-\mathbf w_{LS}\| = 0.051, 0.136, 0.065$ after 1, 5, 20 epochs (fluctuating in a noise ball); $\alpha = 0.05$: $0.154, 0.278, 0.202$ (bigger ball); $\alpha_t = \frac{0.05}{1+0.5t}$: $0.154, 0.170, 0.045$ (decay shrinks the ball).

**Example 15.5.2 (mini-batch size).** 10 epochs: batch 1 (5000 updates/epoch): $0.144$; batch 32 (157): $0.028$; batch 512 (10): $0.022$; full batch (1 update/epoch, larger step): $0.0033$. Larger batches reduce gradient noise; smaller ones make more updates per epoch — on GPUs, batch sizes of 32–512 are the usual compromise.

**Example 15.5.3 (Adam on badly scaled features).** Logistic regression with features of scale 1 and 50 (optimal loss $0.38856$ at $\mathbf w^\ast = (0.314, 1.513, 0.042)$), 30 epochs, batch 32: SGD $\alpha = 10^{-3}$: loss $0.4647$ ($\mathbf w = (0.06, 0.32, 0.03)$, far from converged); SGD $\alpha = 10^{-2}$: $0.4241$; Adam $\alpha = 10^{-2}$: $0.3898$, $\mathbf w = (0.307, 1.538, 0.046)$. Adam's per-coordinate scaling compensates for the scale mismatch.

**Example 15.5.4 (why SGD for big data).** $n = 200\,000$, $d = 10$: one SGD epoch (batch 64) reaches an optimality gap of $9.3\times10^{-4}$; full-batch gradient descent needs 6 passes over the data for the same gap. (On this easy, well-conditioned problem the timings are similar; for large, redundant datasets and expensive models, SGD's advantage grows with $n$.)

**Example 15.5.5 (too large a learning rate).** Mini-batch 64, 2 epochs: $\alpha = 0.1$: error $0.032$; $0.5$: $0.23$; $1.2$: $0.64$; $2.5$: $2.4\times10^{33}$ — divergence once $\alpha$ exceeds roughly $2/L$.

---

## 15.6 Constrained Optimisation

**Equality constraints — Lagrange multipliers.** For $\min f(\mathbf x)$ s.t. $\mathbf h(\mathbf x) = \mathbf 0$: at a regular minimiser there is $\boldsymbol\lambda$ with $\nabla f + \sum\lambda_j\nabla h_j = \mathbf 0$ (stationarity of the Lagrangian $\mathcal L = f + \boldsymbol\lambda^T\mathbf h$).

**Inequalities — KKT conditions.** For $\min f$ s.t. $g_i(\mathbf x)\le0$, $h_j(\mathbf x) = 0$: stationarity $\nabla f+\sum\mu_i\nabla g_i+\sum\lambda_j\nabla h_j = \mathbf 0$; primal feasibility; dual feasibility $\mu_i\ge0$; complementary slackness $\mu_ig_i(\mathbf x) = 0$. For convex problems KKT is necessary and sufficient.

**Methods.** Projected gradient $\mathbf w\leftarrow P_C(\mathbf w-\alpha\nabla F)$ for simple sets (boxes, balls, simplex); penalty methods $\min f + \rho\sum\max(0,g_i)^2$ ($\rho\to\infty$); augmented Lagrangian/ADMM; interior-point and simplex methods for LP/QP (`scipy.optimize.linprog`, `cvxpy`). Many ML problems are constrained: SVMs, portfolio optimisation, non-negative matrix factorisation, fairness constraints.

### Examples for §15.6

**Example 15.6.1 (Lagrange multipliers).** Maximise $xy$ s.t. $x+y = 10$: $\nabla(xy) = \lambda\nabla(x+y)$ gives $y = \lambda$, $x = \lambda$, so $x = y = 5$, $\lambda = 5$, maximum $25$. Interpretation: $\lambda = \frac{d(\text{optimal value})}{d(\text{budget})}$ — raising the budget to $10.1$ raises the maximum by about $0.5$.

**Example 15.6.2 (KKT with an inequality).** $\min x^2+y^2$ s.t. $x+y\ge1$ ($g = 1-x-y\le0$): stationarity $2x - \mu = 0$, $2y-\mu = 0$; the constraint must be active (else $x = y = 0$ violates it), so $x = y = \frac12$, $\mu = 1\ge0$ ✓. A numerical solver (SLSQP) returns $(0.5, 0.5)$.

**Example 15.6.3 (long-only minimum-variance portfolio).** Three assets with covariance $\Sigma$ (volatilities 20%, 30%, 40%); minimise $\mathbf w^T\Sigma\mathbf w$ over the simplex $\{\mathbf w\ge0, \sum w_i = 1\}$ by projected gradient (projection onto the simplex by sorting): $\mathbf w = (0.578, 0.273, 0.148)$, variance $0.02626$ (volatility 16.2% — lower than any single asset), expected return 8.28%; 115 iterations.

**Example 15.6.4 (linear programming).** Maximise profit $3x_1+5x_2$ subject to $x_1\le4$, $2x_2\le12$, $3x_1+2x_2\le18$, $\mathbf x\ge0$: optimum $(2,6)$, profit $36$. The dual values $(0, 1.5, 1)$ are shadow prices: one more unit of resource 2 is worth $1.5$, of resource 3 worth $1$, of resource 1 nothing (slack).

**Example 15.6.5 (Markowitz mean–variance by KKT).** $\min\mathbf w^T\Sigma\mathbf w$ s.t. $\boldsymbol\mu^T\mathbf w = 0.10$, $\mathbf 1^T\mathbf w = 1$ (short sales allowed): the KKT conditions are the *linear* system
$$
\begin{pmatrix}2\Sigma&\boldsymbol\mu&\mathbf 1\\\boldsymbol\mu^T&0&0\\\mathbf 1^T&0&0\end{pmatrix}\begin{pmatrix}\mathbf w\\\lambda_1\\\lambda_2\end{pmatrix} = \begin{pmatrix}\mathbf 0\\0.10\\1\end{pmatrix},
$$
giving $\mathbf w = (0.308, 0.384, 0.308)$, variance $0.03272$.

**Example 15.6.6 (penalty method).** $\min x^2+y^2 + \rho\max(0,1-x-y)^2$: $\rho = 1$: $(0.333, 0.333)$; $\rho = 10$: $(0.476, 0.476)$; $\rho = 100$: $(0.4975, 0.4975)$; $\rho = 1000$: $(0.49975, 0.49975)$. The constraint violation is $\approx\frac{1}{1+\rho}$: accurate solutions need large $\rho$, which makes the problem ill-conditioned — hence augmented Lagrangian methods.

---

## 15.7 Automatic Differentiation and Gradient Checking

Finite differences (Chapter 4) need $d+1$ (or $2d$) function evaluations for a gradient and lose half the digits. **Automatic differentiation (AD)** applies the chain rule to the *program*:
* **Forward mode:** propagate (value, derivative) pairs — *dual numbers* $a + b\varepsilon$ with $\varepsilon^2 = 0$: $(a+b\varepsilon)(c+d\varepsilon) = ac + (ad+bc)\varepsilon$, $\sin(a+b\varepsilon) = \sin a + b\cos a\,\varepsilon$. One pass per input direction.
* **Reverse mode (back-propagation):** record the computational graph forward, then propagate adjoints $\bar v = \partial f/\partial v$ backward. The whole gradient of a scalar loss costs a small constant multiple (≈2–4) of one function evaluation, **independent of $d$** — the reason deep networks with $10^9$ parameters are trainable (PyTorch, JAX, TensorFlow).

**Gradient checking.** Compare an analytic/AD gradient $\mathbf g$ with central differences $\mathbf g_{FD}$ ($h\approx10^{-5}$) via $\frac{\|\mathbf g-\mathbf g_{FD}\|}{\|\mathbf g+\mathbf g_{FD}\|}$: $\sim10^{-9}$–$10^{-11}$ means correct; $\gtrsim10^{-3}$ means a bug.

### Examples for §15.7

**Example 15.7.1 (forward mode with dual numbers).** $f(x) = x\sin x$ at $x = 2$: $(2+\varepsilon)\sin(2+\varepsilon) = (2+\varepsilon)(\sin2 + \cos2\,\varepsilon) = 2\sin2 + (\sin2 + 2\cos2)\varepsilon = 1.818595 + 0.077004\,\varepsilon$ — value and exact derivative in one pass.

**Example 15.7.2 (reverse mode by hand).** $f(x,y) = (xy+\sin x)^2$ at $(1,2)$. Forward: $a_1 = xy = 2$, $a_2 = \sin x = 0.841471$, $a_3 = a_1+a_2 = 2.841471$, $f = a_3^2 = 8.073957$. Backward: $\bar a_3 = 2a_3 = 5.682942$; $\bar a_1 = \bar a_2 = 5.682942$; $\bar x = \bar a_1y + \bar a_2\cos x = 14.436391$; $\bar y = \bar a_1x = 5.682942$. Finite differences confirm $(14.436391, 5.682942)$.

**Example 15.7.3 (gradient checking).** The analytic logistic-loss gradient (with ridge term) vs central differences: relative error $9\times10^{-12}$ ✓. A deliberately buggy gradient (forgetting the $\frac1n$) gives $0.99$ — caught immediately.

**Example 15.7.4 (training a neural network with back-propagation).** A 2–16–1 network ($\tanh$ hidden layer, sigmoid output, 65 parameters) on the two-moons data (400 points). Back-propagated gradient vs finite differences: relative error $2.7\times10^{-10}$. Full-batch gradient descent ($\alpha = 0.5$): loss $0.741\to0.298\to0.104\to0.087$ and accuracy $0.50\to0.885\to0.960\to0.970$ after 1, 100, 1000, 3000 epochs — a non-convex problem solved well by first-order methods.

**Example 15.7.5 (cost comparison).** For the 65-parameter network a central-difference gradient needs 130 loss evaluations; back-propagation costs about two to three. For a model with $10^8$ parameters the ratio is $10^8$ — finite differences are for *checking*, AD is for *computing*.

---

## Chapter summary

| Method | Per-iteration cost | Convergence | Typical use |
|---|---|---|---|
| golden section / Brent | 1 $f$ | linear / superlinear | 1-D, hyperparameters |
| gradient descent | $\nabla F$ | $1-1/\kappa$ | baseline, huge $d$ |
| heavy ball / Nesterov | $\nabla F$ | $1-1/\sqrt\kappa$ | smooth convex |
| Newton (modified) | $\nabla^2F$, $O(d^3)$ | quadratic | small $d$, GLMs |
| BFGS / L-BFGS | $O(d^2)$ / $O(md)$ | superlinear | medium–large, full batch |
| SGD / Adam | $O(\lvert B\rvert d)$ | sublinear (noise) | huge $n$, deep learning |
| projected gradient / penalty / KKT / LP | varies | varies | constraints |

## Further reading

Nocedal & Wright, *Numerical Optimization* (2nd ed.) · Boyd & Vandenberghe, *Convex Optimization* (free online) · Bottou, Curtis & Nocedal, "Optimization methods for large-scale machine learning", *SIAM Review* 60 (2018) · Kingma & Ba, "Adam: a method for stochastic optimization", ICLR 2015 · Baydin et al., "Automatic differentiation in machine learning: a survey", *JMLR* 18 (2018).
