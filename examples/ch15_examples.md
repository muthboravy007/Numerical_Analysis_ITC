# Chapter 15 — Worked Examples: Numerical Optimisation

Companion script: [`ch15_examples.py`](ch15_examples.py) reproduces every number below. · Lecture: [Chapter 15](../lectures/ch15-numerical-optimization.md)

---

## Example 15.1 — Armijo backtracking, step by step

**Problem.** For $f(x,y) = x^2+10y^2$ at $(1,1)$, find the step size along $-\nabla f$ accepted by Armijo backtracking ($c_1 = 10^{-4}$, halving from $\alpha = 1$).

**Solution.** $f = 11$, $\nabla f = (2, 20)$, and the slope along $\mathbf p = -\nabla f$ is $-404$.

| $\alpha$ | $f(\mathbf x+\alpha\mathbf p)$ | Armijo threshold $f+c_1\alpha\nabla f^T\mathbf p$ | accepted? |
|---|---|---|---|
| 1 | 3611 | 10.960 | no |
| 0.5 | 810 | 10.980 | no |
| 0.25 | 160.25 | 10.990 | no |
| 0.125 | 23.06 | 10.995 | no |
| **0.0625** | **1.391** | 10.997 | **yes** |

The exact line-search step would be $\frac{\mathbf g^T\mathbf g}{\mathbf g^TA\mathbf g} = 0.0504$.

**Take-away.** Backtracking needs only function values. It finds a step within a factor of 2 of a sensible one, and it guarantees sufficient decrease, which is all the convergence theory needs.

---

## Example 15.2 — Newton vs gradient descent for a logistic MLE

**Problem.** Fit a one-parameter logistic model $P(y = 1) = \sigma(\beta x)$ to 8 observations by maximum likelihood.

**Solution.** Newton from $\beta = 0$ gives $0.54113, 0.60550, 0.60765, 0.60765, \dots$. The errors $6\times10^{-1}\to7\times10^{-2}\to2\times10^{-3}\to2\times10^{-6}\to3\times10^{-12}$ show quadratic convergence. Gradient descent with the safe step $1/L$, where $L = \sum x_i^2/4 = 4.62$ bounds the curvature, needs 14 iterations for $10^{-8}$. The standard error from the Fisher information is $1/\sqrt{I(\hat\beta)} = 0.554$: the estimate is not significantly different from 0 with $n = 8$.

**Take-away.** For small, smooth problems Newton's method is unbeatable, and its Hessian doubles as the inverse covariance matrix for inference.

---

## Example 15.3 — Momentum cures ill-conditioning

**Problem.** Minimise $f = \frac12(x^2+100y^2)$ ($\kappa = 100$) from $(1,1)$ until $\|\nabla f\|<10^{-8}$.

**Solution.**

| method | iterations |
|---|---|
| GD, step $1/L$ | 1833 |
| GD, optimal step $2/(\mu+L)$ | 1152 |
| heavy ball, optimal $(\alpha,\beta)$ | **143** |
| Nesterov, step $1/L$, $\beta = 0.8$ | 279 |

The gradient norm must shrink by $10^{-10}$. At the rates $\frac{\kappa-1}{\kappa+1} = 0.980$ and $\frac{\sqrt\kappa-1}{\sqrt\kappa+1} = 0.818$, that predicts about 1151 and 115 iterations. The GD count matches exactly. Heavy ball needs a few transient iterations more.

**Take-away.** Acceleration replaces $\kappa$ by $\sqrt\kappa$, a 10× saving here and 1000× when $\kappa = 10^6$. It needs sensible parameter choices, and it is the idea behind momentum in deep learning.

---

## Example 15.4 — How much memory does L-BFGS need?

**Problem.** Minimise the 100-dimensional extended Rosenbrock function from $(-1.2, 1, -1.2, 1, \dots)$ with L-BFGS for memory $m = 1,\dots,20$, and with full BFGS.

**Solution.**

| method | iterations | stored numbers |
|---|---|---|
| L-BFGS, $m = 1$ | 73 | 200 |
| L-BFGS, $m = 3$ | 36 | 600 |
| L-BFGS, $m = 5$ | 36 | 1000 |
| L-BFGS, $m = 20$ | 36 | 4000 |
| full BFGS, $H_0 = I$ (ours) | 923 | 10 000 |
| full BFGS, $H_0 = I$ (scipy) | 467 | 10 000 |
| scipy L-BFGS-B | 36 | — |

**Take-away.** A handful of curvature pairs is enough. Full BFGS started from $H_0 = I$ is *slower* here: its initial matrix has the wrong scale, and it must learn 5000 entries. L-BFGS rescales $H_0 = \gamma I$ with $\gamma = \mathbf s^T\mathbf y/\mathbf y^T\mathbf y$ at every iteration. L-BFGS is the default for smooth problems with thousands to millions of variables (logistic regression, CRFs, physics-informed models).

---

## Example 15.5 — SGD: step size, batch size and the noise floor

**Problem.** Least squares with $n = 2000$ and $p = 10$ (noise SD 0.5), trained by SGD for 30 epochs. Report the distance to the exact least-squares solution.

**Solution.**

| setting | after 5 epochs | after 30 epochs |
|---|---|---|
| constant $\eta = 0.05$, batch 1 | 0.211 | 0.316 |
| constant $\eta = 0.01$, batch 1 | 0.106 | 0.112 |
| constant $\eta = 0.05$, batch 32 | 0.025 | 0.043 |
| decaying $\eta = 0.05/(1+t)$, batch 1 | 0.106 | **0.041** |

**Take-away.** With a constant step SGD does not converge: it hovers in a noise ball whose size is proportional to $\eta\sigma^2/b$ (Chapter 15, A7). Smaller steps, larger batches and decaying steps all shrink the ball, trading off speed. Note that these errors ($\approx0.04$) are already smaller than the statistical uncertainty of $\mathbf w_{LS}$ itself (about $0.5/\sqrt{2000}\cdot\sqrt{10}\approx0.035$).

---

## Example 15.6 — Maximum entropy with a constraint: Newton on the dual

**Problem.** Find the distribution $p_1,\dots,p_6$ on the faces of a die that maximises the entropy $-\sum p_k\ln p_k$ subject to $\sum p_k = 1$ and mean $\sum kp_k = 4.5$.

**Solution.** The Lagrange (KKT) conditions give the Gibbs form $p_k\propto e^{\lambda k}$. The single multiplier $\lambda$ solves $E_\lambda[k] = 4.5$. Newton's method on this 1-D dual equation uses the derivative $\operatorname{Var}_\lambda(k)$. It gives $0.34286\to0.37059\to0.37105\to0.371049$, with quadratic convergence. The result is $p = (0.054, 0.079, 0.114, 0.165, 0.240, 0.347)$, with mean $4.5$ and entropy $1.6136$ (uniform: $\ln6 = 1.7918$).

**Take-away.** Duality reduces a 6-dimensional constrained problem to a 1-dimensional root-finding problem. The exponential-family form is exactly what logistic regression, softmax and CRFs are.

---

## Example 15.7 — Non-negative least squares by projected gradient

**Problem.** Mixture proportions and concentrations must be non-negative. Solve $\min\|X\mathbf b-\mathbf y\|^2$ subject to $\mathbf b\ge0$, with $X\in\mathbb R^{50\times8}$ and true $\mathbf b = (2,0,1.5,0,0,0.5,0,0)$.

**Solution.** Projected gradient ($\mathbf b\leftarrow\max(\mathbf b-\frac1L\nabla,0)$) converges in 596 iterations to $(1.981, 0.046, 1.483, 0.009, 0, 0.470, 0.010, 0)$. This agrees with `scipy.optimize.nnls` to $2\times10^{-11}$. The unconstrained solution has negative entries $-0.017$ and $-0.020$, which the constraint sets to exactly zero.

**Take-away.** Projection makes simple constraints (non-negativity, boxes, simplices) as easy as unconstrained gradient descent. The KKT conditions explain the zeros: those coordinates have gradients pointing outward. Non-negativity is not full sparsity, though: the tiny true zeros at indices 1, 3 and 6 get small positive values from noise.
