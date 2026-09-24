# Lecture 9 — Numerical Optimization for Data Science

> **Week 12–13** · Code: [`numlib/optimization.py`](../numlib/optimization.py) · Examples: [`examples/ch09`](../examples/ch09_examples.md) · Exercises: [`exercises/ch09`](../exercises/ch09_exercises.md)

## Learning objectives

1. State first- and second-order optimality conditions and recognise convex problems.
2. Apply golden-section search for 1-D minimisation.
3. Implement gradient descent with fixed step and with backtracking line search; relate its speed to the condition number of the Hessian.
4. Implement Newton's method and BFGS and compare their cost/convergence.
5. Explain stochastic gradient descent, mini-batches, momentum and Adam, and why they dominate machine learning.
6. Fit logistic regression by gradient descent and by Newton's method (IRLS).

---

## 9.1 Training a model *is* an optimisation problem

Almost every model in data science is fitted by minimising an empirical risk
$$
\min_{\mathbf w\in\mathbb R^d}\ F(\mathbf w) = \frac1m\sum_{i=1}^m \ell\bigl(\mathbf w;\ \mathbf x_i, y_i\bigr) + \lambda R(\mathbf w).
$$
Least squares (Lecture 5) is the rare case with a closed-form solution. Logistic regression, neural networks, matrix factorisation, SVMs, $k$-means and maximum-likelihood estimation all need iterative optimisation.

## 9.2 Optimality conditions and convexity

For smooth $F$:
* **First-order necessary:** $\nabla F(\mathbf w^\star) = \mathbf 0$ (stationary point).
* **Second-order sufficient:** $\nabla F(\mathbf w^\star) = \mathbf 0$ and Hessian $\nabla^2F(\mathbf w^\star)$ positive definite ⇒ strict local minimum.

$F$ is **convex** if $F(\theta\mathbf a + (1-\theta)\mathbf b) \le \theta F(\mathbf a) + (1-\theta)F(\mathbf b)$; for $C^2$ functions, iff $\nabla^2F\succeq0$ everywhere. For convex $F$ **every stationary point is a global minimum**. Least squares, ridge, lasso, logistic regression and SVMs are convex; neural networks are not.

**Strong convexity and smoothness.** If $\mu I\preceq\nabla^2F\preceq LI$, the ratio $\kappa = L/\mu$ is the **condition number** of the optimisation problem; it controls the speed of first-order methods just like $\kappa(A)$ controlled CG in Lecture 4.

## 9.3 One-dimensional search: golden section

For a unimodal $f$ on $[a,b]$, evaluate at two interior points $c<d$ and discard the part of the interval that cannot contain the minimum. Placing them at fractions $1-\varphi^{-1}$ and $\varphi^{-1}$ of the interval ($\varphi^{-1} = \frac{\sqrt5-1}{2}\approx0.618$) lets one point be re-used, so each step costs **one** evaluation and shrinks the interval by $0.618$ — linear convergence, derivative-free, guaranteed. (Used for tuning a single hyperparameter, or inside line searches.) `scipy.optimize.minimize_scalar` uses Brent's method, which adds parabolic interpolation.

## 9.4 Gradient descent

$$
\mathbf w_{k+1} = \mathbf w_k - \alpha_k\nabla F(\mathbf w_k).
$$
$-\nabla F$ is the direction of steepest local decrease.

### Step size
* **Fixed step** $\alpha = 1/L$ guarantees decrease for $L$-smooth $F$. Too large → divergence; too small → crawl.
* **Backtracking (Armijo) line search:** start with $\alpha=1$ and halve until
$$
F(\mathbf w_k - \alpha\nabla F_k) \le F(\mathbf w_k) - c\,\alpha\|\nabla F_k\|^2,\qquad c = 10^{-4}.
$$
* **Exact line search** for quadratics: $\alpha = \frac{\mathbf g^\top\mathbf g}{\mathbf g^\top A\mathbf g}$ (= steepest descent of Lecture 4).

### Convergence

**Theorem.** For $\mu$-strongly convex, $L$-smooth $F$ with $\alpha = 1/L$:
$$
F(\mathbf w_k) - F^\star \le \Bigl(1 - \frac1\kappa\Bigr)^k\bigl(F(\mathbf w_0)-F^\star\bigr).
$$
Linear convergence, but with $\kappa = 10^4$ it takes about $\kappa\ln 10 \approx 2.3\times 10^4$ iterations per decimal digit.

**Zig-zagging.** For $F(\mathbf w) = \tfrac12(w_1^2 + 10w_2^2)$ from $(10,1)$ with exact line search, every step has $\alpha = 2/11$ and the iterates bounce: $(8.18,-0.82)$, $(6.69, 0.67)$, $(5.48, -0.55)$, … The error shrinks by only $\frac{\kappa-1}{\kappa+1} = \frac{9}{11}$ per step. **Feature scaling** (standardisation) reduces $\kappa$ and is the single most effective "optimiser tweak" in practice.

### Momentum and acceleration
**Heavy-ball momentum** $\mathbf v_{k+1} = \beta\mathbf v_k - \alpha\nabla F(\mathbf w_k)$, $\mathbf w_{k+1} = \mathbf w_k + \mathbf v_{k+1}$ damps zig-zags. With tuned $\alpha,\beta$ (and Nesterov's variant for general convex $F$) the rate improves to $1 - 1/\sqrt\kappa$ — the same $\sqrt\kappa$ improvement CG achieves over steepest descent.

## 9.5 Newton's method

Minimise the local quadratic model $F(\mathbf w_k+\mathbf p)\approx F_k + \nabla F_k^\top\mathbf p + \tfrac12\mathbf p^\top\nabla^2F_k\,\mathbf p$:
$$
\nabla^2F(\mathbf w_k)\,\mathbf p_k = -\nabla F(\mathbf w_k),\qquad \mathbf w_{k+1} = \mathbf w_k + \mathbf p_k.
$$
This is Lecture 2's Newton's method applied to $\nabla F = \mathbf 0$.

* **Quadratic convergence** near a minimiser with positive definite Hessian; **affine invariant** — insensitive to scaling and conditioning.
* Cost per step: forming the Hessian ($O(md^2)$ for GLMs) and solving ($O(d^3)$, Cholesky). Fine for $d\lesssim10^4$, impossible for deep nets with $d\sim10^9$.
* Far from the minimum the Hessian may be indefinite → add a line search, or damping $(\nabla^2F + \lambda I)$ (Levenberg–Marquardt), or a trust region.

**Example (1-D).** $f(x) = x^4 - 3x^3 + 2$ from $x_0 = 3$: $2.5,\ 2.2917,\ 2.25146,\ 2.2500019,\ 2.25000000000$ → quadratic convergence to $x^\star = 9/4$.

**Rosenbrock's banana** $F(x,y) = (1-x)^2 + 100(y-x^2)^2$ from $(-1.2, 1)$:

| Method | iterations to $\|\nabla F\|<10^{-8}$ |
|---|---|
| Gradient descent + backtracking | ≈ 19 400 |
| BFGS + backtracking | 35 |
| Newton (pure) | 7 |

## 9.6 Quasi-Newton: BFGS and L-BFGS

Build an approximation $H_k\approx(\nabla^2F)^{-1}$ from gradient differences. With $\mathbf s_k = \mathbf w_{k+1}-\mathbf w_k$, $\mathbf y_k = \nabla F_{k+1}-\nabla F_k$ and $\rho_k = 1/\mathbf y_k^\top\mathbf s_k$:
$$
H_{k+1} = (I - \rho_k\mathbf s_k\mathbf y_k^\top)\,H_k\,(I-\rho_k\mathbf y_k\mathbf s_k^\top) + \rho_k\mathbf s_k\mathbf s_k^\top .
$$
This satisfies the **secant condition** $H_{k+1}\mathbf y_k = \mathbf s_k$ (compare the secant method of Lecture 2) and preserves positive definiteness when $\mathbf y_k^\top\mathbf s_k>0$. Convergence is **superlinear**, cost $O(d^2)$ per step, no Hessian needed.

**L-BFGS** stores only the last $m\approx 10$ pairs $(\mathbf s,\mathbf y)$ — $O(md)$ memory. It is the default solver for `sklearn.linear_model.LogisticRegression` and `scipy.optimize.minimize(method="L-BFGS-B")`.

## 9.7 Stochastic gradient methods

When $m$ (number of samples) is huge, even one full gradient is expensive. **SGD** uses an unbiased estimate from a random mini-batch $B_k$:
$$
\mathbf w_{k+1} = \mathbf w_k - \alpha_k\,\frac{1}{|B_k|}\sum_{i\in B_k}\nabla\ell_i(\mathbf w_k).
$$

* Each step costs $O(|B|d)$ instead of $O(md)$.
* The gradient noise prevents convergence with a constant step; theory requires $\sum\alpha_k=\infty$, $\sum\alpha_k^2<\infty$ (e.g. $\alpha_k = \alpha_0/(1+\gamma k)$). In practice: constant rate + scheduled decay.
* Rate: $O(1/k)$ for strongly convex problems — slow in iterations but very cheap per iteration, and good enough because we only need to optimise to the level of the statistical error.
* Noise may help escape saddle points and sharp minima in deep learning.

**Adam** keeps exponential moving averages of the gradient ($\mathbf m$) and its elementwise square ($\mathbf v$):
$$
\mathbf m_k = \beta_1\mathbf m_{k-1}+(1-\beta_1)\mathbf g_k,\quad
\mathbf v_k = \beta_2\mathbf v_{k-1}+(1-\beta_2)\mathbf g_k^2,\quad
\mathbf w_{k+1} = \mathbf w_k - \alpha\frac{\hat{\mathbf m}_k}{\sqrt{\hat{\mathbf v}_k}+\varepsilon},
$$
with bias corrections $\hat{\mathbf m}_k = \mathbf m_k/(1-\beta_1^k)$, $\hat{\mathbf v}_k = \mathbf v_k/(1-\beta_2^k)$. The per-coordinate scaling is a cheap diagonal preconditioner.

## 9.8 Case study: logistic regression

With $\sigma(z) = 1/(1+e^{-z})$ and labels $y_i\in\{0,1\}$, the negative log-likelihood is
$$
F(\mathbf w) = \frac1m\sum_{i=1}^m\Bigl[\log\bigl(1+e^{\mathbf x_i^\top\mathbf w}\bigr) - y_i\,\mathbf x_i^\top\mathbf w\Bigr] + \frac\lambda2\|\mathbf w\|^2,
$$
$$
\nabla F = \frac1mX^\top(\boldsymbol\sigma - \mathbf y) + \lambda\mathbf w,\qquad
\nabla^2F = \frac1mX^\top DX + \lambda I,\quad D = \operatorname{diag}\bigl(\sigma_i(1-\sigma_i)\bigr).
$$
* $F$ is convex; strictly convex if $\lambda>0$.
* Compute $\log(1+e^z)$ as `np.logaddexp(0, z)` (Lecture 1!).
* Newton's step $\bigl(X^\top DX + m\lambda I\bigr)\mathbf p = -X^\top(\boldsymbol\sigma-\mathbf y) - m\lambda\mathbf w$ is a **weighted least-squares** problem — hence the name *Iteratively Reweighted Least Squares*. It typically converges in 5–10 iterations.
* Without regularisation on **separable** data the MLE does not exist ($\|\mathbf w\|\to\infty$): the optimiser never "converges". Numerical behaviour exposes a statistical issue.

## 9.9 Constrained optimisation (preview)

* Simple bounds: projected gradient, L-BFGS-B.
* Equality constraints: Lagrange multipliers, KKT conditions.
* $\ell_1$ penalties (lasso): proximal gradient / coordinate descent (`sklearn.linear_model.Lasso`).
* Linear/quadratic programs: `scipy.optimize.linprog`, `cvxpy`.

## 9.10 Summary

| Method | Per-iteration cost | Convergence | Typical use |
|---|---|---|---|
| Gradient descent | $O(md)$ | linear, $1-1/\kappa$ | Baseline |
| Momentum / Nesterov | $O(md)$ | linear, $1-1/\sqrt\kappa$ | Smooth convex |
| Newton | $O(md^2 + d^3)$ | quadratic | Small $d$, GLMs |
| BFGS / L-BFGS | $O(md + d^2)$ / $O(md)$ | superlinear | Medium $d$, full batch |
| SGD / Adam | $O(\lvert B\rvert d)$ | sublinear | Huge $m$, deep learning |

## Further reading

* Nocedal & Wright, *Numerical Optimization*, 2nd ed., Chapters 2–3, 6–7.
* Boyd & Vandenberghe, *Convex Optimization* (free online), Chapter 9.
* Bottou, Curtis & Nocedal, *Optimization Methods for Large-Scale Machine Learning*, SIAM Review, 2018.
