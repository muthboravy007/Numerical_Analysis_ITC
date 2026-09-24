# Chapter 10 — Numerical Solutions of Nonlinear Systems of Equations

> **Reference:** Burden & Faires, Chapter 10 (§10.1–10.5) plus §10.6 for data science.
> **Code:** [`numlib/nonlinear_systems.py`](../numlib/nonlinear_systems.py), [`numlib/roots.py`](../numlib/roots.py) · **All numbers reproduced by** [`lectures/code/ch10.py`](code/ch10.py)
> **Worked problems:** [`examples/ch10`](../examples/ch10_examples.md) · **Homework:** [`exercises/ch10`](../exercises/ch10_exercises.md)
> **See also:** Chapter 15 (optimisation): minimising $g(\mathbf x)$ and solving $\nabla g = \mathbf 0$ are two views of the same problem.

## Learning outcomes

1. Extend fixed-point iteration to $\mathbb R^n$ and verify contraction conditions via the Jacobian.
2. Derive and apply Newton's method for systems; analyse its quadratic convergence and globalise it by damping.
3. Derive Broyden's quasi-Newton update and the Sherman–Morrison formula; compare costs with Newton.
4. Use steepest descent on $g = \sum f_i^2$ to obtain starting values, and recognise its limitations.
5. Solve hard problems by homotopy/continuation.
6. Formulate multi-parameter estimation (MLE, logistic regression, EM, $k$-means, equilibria) as nonlinear systems or fixed-point problems.

We solve $\mathbf F(\mathbf x) = \mathbf 0$ with $\mathbf F = (f_1,\dots,f_n)^T:\mathbb R^n\to\mathbb R^n$ and **Jacobian** $J(\mathbf x)_{ij} = \partial f_i/\partial x_j$.

---

## 10.1 Fixed Points for Functions of Several Variables

$\mathbf p$ is a fixed point of $\mathbf G:D\subset\mathbb R^n\to\mathbb R^n$ if $\mathbf G(\mathbf p) = \mathbf p$. Iterate $\mathbf x^{(k)} = \mathbf G(\mathbf x^{(k-1)})$.

**Theorem 10.6.** Let $D = \{\mathbf x: a_i\le x_i\le b_i\}$ and $\mathbf G:D\to\mathbb R^n$ continuous with $\mathbf G(\mathbf x)\in D$ for $\mathbf x\in D$. Then $\mathbf G$ has a fixed point in $D$. If moreover the partial derivatives are continuous and there is $K<1$ with
$$
\Bigl|\frac{\partial g_i(\mathbf x)}{\partial x_j}\Bigr|\le\frac Kn\qquad\text{for all }\mathbf x\in D,\ i,j,
$$
then the fixed point is unique, the iteration converges for every $\mathbf x^{(0)}\in D$, and $\|\mathbf x^{(k)}-\mathbf p\|_\infty\le\frac{K^k}{1-K}\|\mathbf x^{(1)}-\mathbf x^{(0)}\|_\infty$.

(A sharper local criterion: convergence near $\mathbf p$ if $\rho(J_{\mathbf G}(\mathbf p))<1$, with asymptotic rate $\rho(J_{\mathbf G}(\mathbf p))$.)

**Gauss–Seidel acceleration.** Use each newly computed component immediately, as in Chapter 7.

### Examples for §10.1

**Example 10.1.1 (a contraction in $\mathbb R^2$).** $x = 0.5\cos y + 0.1$, $y = 0.3\sin x + 0.4$. From $(0,0)$:
$(0.6, 0.4)$, $(0.56053, 0.56939)$, $(0.52111, 0.55949)$, $(0.52376, 0.54935)$, $(0.52643, 0.55004)$, … → $\mathbf p = (0.526093, 0.550648)$ after 18 iterations (tolerance $10^{-10}$).

**Example 10.1.2 (checking Theorem 10.6).** On $D = [0,1]^2$: $\mathbf G(D)\subset[0.1,0.6]\times[0.4,0.66]\subset D$. Partial derivatives: $|\partial g_1/\partial y| = 0.5|\sin y|\le0.5$, $|\partial g_2/\partial x| = 0.3|\cos x|\le0.3$, others 0. With $n = 2$ we need $\frac Kn\ge0.5$, i.e. $K = 1$ — the *sufficient* condition just fails. But at the fixed point $\|J_{\mathbf G}(\mathbf p)\|_\infty = 0.262<1$, so the iteration converges locally (as observed). Hypotheses of convergence theorems are often pessimistic.

**Example 10.1.3 (Gauss–Seidel variant).** Updating $x$ first and then using it in $y = 0.3\sin x + 0.4$ converges in 10 iterations instead of 18.

**Example 10.1.4 (a bad rearrangement).** For $F(x,y) = (x^2+y-11,\ x+y^2-7)$ (root $(3,2)$), the naive $\mathbf G(\mathbf x) = \mathbf x - \mathbf F(\mathbf x)$ from $(2.9, 2.1)$ gives $(3.39,1.79)$, $(1.11,2.20)$, $(8.69,3.27)$, $(-59,-9.1)$, … — divergence. Indeed $J_{\mathbf G}(3,2) = I - J_{\mathbf F} = \begin{pmatrix}-5&-1\\-1&-3\end{pmatrix}$ has $\rho = 5.41>1$.

**Example 10.1.5 (a good rearrangement).** Solving each equation for "its" variable: $x = \sqrt{11-y}$, $y = \sqrt{7-x}$. At $(3,2)$, $J_{\mathbf G} = \begin{pmatrix}0&-\frac{1}{2\sqrt9}\\-\frac{1}{2\sqrt4}&0\end{pmatrix}$ with $\rho = 0.204$. From $(2.5,2.5)$: converges to $(3,2)$ in 15 iterations; the error ratios alternate $0.173, 0.241$ (geometric mean $0.204$).

---

## 10.2 Newton's Method

Linearise $\mathbf F$ at $\mathbf x^{(k-1)}$: $\mathbf F(\mathbf x)\approx\mathbf F(\mathbf x^{(k-1)}) + J(\mathbf x^{(k-1)})(\mathbf x-\mathbf x^{(k-1)})$. Setting this to zero:
$$
\boxed{J(\mathbf x^{(k-1)})\,\mathbf y^{(k-1)} = -\mathbf F(\mathbf x^{(k-1)}),\qquad\mathbf x^{(k)} = \mathbf x^{(k-1)} + \mathbf y^{(k-1)}.}
$$
Never invert $J$ — solve the linear system (Chapter 6). Newton is fixed-point iteration with $\mathbf G(\mathbf x) = \mathbf x - J(\mathbf x)^{-1}\mathbf F(\mathbf x)$, whose Jacobian vanishes at $\mathbf p$.

**Theorem 10.7 (local quadratic convergence).** If $\mathbf F$ is $C^2$ near a root $\mathbf p$ and $J(\mathbf p)$ is nonsingular, then for $\mathbf x^{(0)}$ sufficiently close to $\mathbf p$, Newton converges and $\|\mathbf x^{(k)}-\mathbf p\|\le C\|\mathbf x^{(k-1)}-\mathbf p\|^2$.

**Costs per iteration.** One evaluation of $\mathbf F$ ($n$ functions) and $J$ ($n^2$ partial derivatives, or $n$ extra $\mathbf F$-evaluations with finite differences $J_{:,j}\approx\frac{\mathbf F(\mathbf x+h\mathbf e_j)-\mathbf F(\mathbf x)}{h}$), plus an $O(n^3)$ linear solve.

**Globalisation (damping).** Far from the root, a full step can increase $\|\mathbf F\|$. Damped Newton takes $\mathbf x^{(k)} = \mathbf x^{(k-1)} + t\mathbf y^{(k-1)}$ with $t = 1, \frac12, \frac14,\ldots$ until $\|\mathbf F\|$ decreases sufficiently.

### Examples for §10.2

**Example 10.2.1 (two curves).** $x^2+y^2 = 4$, $xy = 1$ from $(2, 0.5)$:

| $k$ | $\mathbf x^{(k)}$ | $\lVert\mathbf F\rVert_2$ |
|---|---|---|
| 0 | $(2, 0.5)$ | $2.5\times10^{-1}$ |
| 1 | $(1.9333333, 0.5166667)$ | $4.9\times10^{-3}$ |
| 2 | $(1.9318527, 0.5176371)$ | $3.5\times10^{-6}$ |
| 3 | $(1.9318517, 0.5176381)$ | $2.5\times10^{-12}$ |

The residual is roughly squared each step. Exact: $\bigl(\sqrt{2+\sqrt3}, \sqrt{2-\sqrt3}\bigr)$.

**Example 10.2.2 (a 3×3 system by hand, first step).** $x_1+x_2+x_3 = 6$, $x_1^2+x_2^2+x_3^2 = 14$, $x_1x_2x_3 = 6$ (solutions: permutations of $(1,2,3)$), $\mathbf x^{(0)} = (0.5, 1.5, 3.5)$:
$$
\mathbf F(\mathbf x^{(0)}) = \begin{pmatrix}-0.5\\0.75\\-3.375\end{pmatrix},\quad J(\mathbf x^{(0)}) = \begin{pmatrix}1&1&1\\1&3&7\\5.25&1.75&0.75\end{pmatrix},\quad\mathbf y^{(0)} = \begin{pmatrix}0.64583\\0.09375\\-0.23958\end{pmatrix}.
$$
$\mathbf x^{(1)} = (1.14583, 1.59375, 3.26042)$; then errors $9.8\times10^{-2}$, $6.3\times10^{-3}$, $3.9\times10^{-5}$, $1.5\times10^{-9}$, $3\times10^{-15}$.

**Example 10.2.3 (quadratic convergence).** $x^2+y-11 = 0$, $x+y^2-7 = 0$ from $(1,1)$ converges to $(3,2)$; errors $2.0, 2.3, 0.51, 0.039, 2.5\times10^{-4}, 1.1\times10^{-8}$, with $e_{k+1}/e_k^2\to0.17$.

**Example 10.2.4 (which root? basins of attraction).** The same system has four real roots. Newton from $(-3,3)$ → $(-2.8051, 3.1313)$; from $(-3,-3)$ → $(-3.7793, -3.2832)$; from $(4,-2)$ → $(3.5844, -1.8481)$; from $(0,0)$ → $(3,2)$. The root found depends on the start; the basin boundaries are fractal.

**Example 10.2.5 (finite-difference Jacobian).** For Example 10.2.2 at $\mathbf x^{(0)}$, forward differences with $h = 10^{-7}$ reproduce $J$ with max error $3.5\times10^{-7}\approx\sqrt u$ — accurate enough for Newton, at the cost of $n+1 = 4$ evaluations of $\mathbf F$ per iteration.

**Example 10.2.6 (damping).** $\mathbf F = (\arctan x + 0.1y,\ y - 0.5x)$, root $(0,0)$, start $(3,0)$. Pure Newton overshoots and diverges: $(-6.33,-3.16)$, $(16.9, 8.5)$, $(-27.2,-13.6)$, $(29.2,14.6)$, … Damped Newton ($t$ halved until $\|\mathbf F\|$ decreases) produces $(-6.33,-3.16)$, $(5.31,2.65)$, $(-4.48,-2.24)$, $(3.60,1.80)$, $(-2.48,-1.24)$, $(0.98,0.49)$, $(-0.49,-0.25)$, … and converges to $(0,0)$.

---

## 10.3 Quasi-Newton Methods

Evaluating $J$ ($n^2$ derivatives) and solving with it ($n^3$) every step is expensive. **Broyden's method** replaces $J$ by a matrix $A_k$ updated by a rank-one correction satisfying the **secant equation** $A_k\mathbf s_k = \mathbf y_k$ (a multidimensional secant method), where $\mathbf s_k = \mathbf x^{(k)}-\mathbf x^{(k-1)}$, $\mathbf y_k = \mathbf F(\mathbf x^{(k)})-\mathbf F(\mathbf x^{(k-1)})$:
$$
A_k = A_{k-1} + \frac{(\mathbf y_k - A_{k-1}\mathbf s_k)\mathbf s_k^T}{\|\mathbf s_k\|_2^2}.
$$
**Theorem 10.8 (Sherman–Morrison).** If $A$ is nonsingular and $\mathbf v^TA^{-1}\mathbf u\ne-1$, then
$$
(A+\mathbf u\mathbf v^T)^{-1} = A^{-1} - \frac{A^{-1}\mathbf u\mathbf v^TA^{-1}}{1+\mathbf v^TA^{-1}\mathbf u}.
$$
This updates $A_k^{-1}$ directly in $O(n^2)$:
$$
A_k^{-1} = A_{k-1}^{-1} + \frac{(\mathbf s_k - A_{k-1}^{-1}\mathbf y_k)\,\mathbf s_k^TA_{k-1}^{-1}}{\mathbf s_k^TA_{k-1}^{-1}\mathbf y_k}.
$$
Broyden needs **one** $\mathbf F$-evaluation and $O(n^2)$ work per step and converges **superlinearly**: $\lim\frac{\|\mathbf x^{(k+1)}-\mathbf p\|}{\|\mathbf x^{(k)}-\mathbf p\|} = 0$. (BFGS in Chapter 15 is the analogous update for optimisation.)

### Examples for §10.3

**Example 10.3.1 (Broyden vs Newton).** Circle/hyperbola system from $(2,0.5)$ with $A_0 = J(\mathbf x^{(0)})$: errors $6.8\times10^{-2},\ 1.5\times10^{-3},\ 4.3\times10^{-5},\ 4.0\times10^{-7},\ 6.3\times10^{-9},\ 1.2\times10^{-11},\ 4\times10^{-16}$. Newton needed 4 iterations, Broyden 6 — but without computing any further derivatives.

**Example 10.3.2 (one update in detail).** After the first step $\mathbf s_1 = (-0.06667, 0.01667)$, $\mathbf y_1 = (-0.24528, -0.00111)$, and
$$
A_1 = \begin{pmatrix}3.93333&1.01667\\0.51569&1.99608\end{pmatrix},\qquad A_1\mathbf s_1 = \mathbf y_1\ \checkmark.
$$
The Sherman–Morrison update of $A_0^{-1}$ gives $\begin{pmatrix}0.27243&-0.13876\\-0.07038&0.53683\end{pmatrix}$, identical to inverting $A_1$ directly.

**Example 10.3.3 (superlinear convergence).** The ratios $e_{k+1}/e_k$ are $0.022,\ 0.029,\ 0.0095,\ 0.016,\ 0.0019$ — tending to 0 (superlinear), but not like $e_k$ (not quadratic).

**Example 10.3.4 (counting function evaluations).** For the $3\times3$ system of Example 10.2.2: Broyden (analytic $A_0$) converges in 9 iterations with **9** evaluations of $\mathbf F$; Newton with a finite-difference Jacobian needs 7 iterations and **35** evaluations. When $\mathbf F$ is expensive (a simulation, a likelihood over millions of rows), Broyden wins.

**Example 10.3.5 (why $O(n^2)$ matters).** For $n = 1000$: Newton's linear solve $\approx\frac23n^3 = 6.7\times10^8$ flops per iteration, a Sherman–Morrison update $\approx4n^2 = 4\times10^6$.

---

## 10.4 Steepest Descent Techniques

Newton and Broyden need a good start. Define
$$
g(\mathbf x) = \sum_{i=1}^nf_i(\mathbf x)^2 = \|\mathbf F(\mathbf x)\|_2^2\ \ge0,
$$
which is zero exactly at the roots. Its gradient is $\nabla g(\mathbf x) = 2J(\mathbf x)^T\mathbf F(\mathbf x)$. **Steepest descent** moves along $\mathbf z = -\nabla g/\|\nabla g\|$ with a step $\alpha$ chosen by a one-dimensional search. B&F (Algorithm 10.3) uses quadratic interpolation: evaluate $g$ at $\alpha_1 = 0$, $\alpha_3$ (halved until $g(\alpha_3)<g(0)$), $\alpha_2 = \alpha_3/2$, fit the parabola $P(\alpha)$ through the three points, and take $\alpha_0$ = its minimiser (or $\alpha_3$ if better).

* Converges (to a local minimum of $g$) from almost any start — **globally**, but only **linearly** and slowly.
* A local minimum with $g>0$ is not a root.
* Typical use: a few steepest-descent steps, then Newton/Broyden.

### Examples for §10.4

**Example 10.4.1 (getting close from far away).** $\mathbf F = (x^2+y-11,\ x+y^2-7)$ from $(0,0)$, where $g = 170$:
$g$: $170 \to 130.5 \to 74.4 \to 24.6 \to 4.10 \to 0.468\to\cdots$; after 10 iterations $\mathbf x = (2.99972, 2.00096)$, $g = 1.3\times10^{-5}$.

**Example 10.4.2 (hybrid strategy).** From the steepest-descent point, Newton reaches $(3,2)$ to machine precision in 3 iterations.

**Example 10.4.3 (one line search in detail).** At $\mathbf x = (0,0)$: $\nabla g = 2J^T\mathbf F = (-14,-22)$, $\mathbf z = (-0.5369,-0.8437)$ (so we move along $-\mathbf z$). $g(0) = 170$, $\alpha_3 = 1$: $g = 130.46<170$ ✓; $\alpha_2 = 0.5$: $g = 153.33$. Divided differences $h_1 = -33.34$, $h_2 = -45.74$, $h_3 = -12.40$; the parabola is concave ($h_3<0$), its critical point $\alpha_0 = -1.09$ is a maximum with $g = 179.4$, so the algorithm takes $\alpha = \alpha_3 = 1$.

**Example 10.4.4 (slow near the solution).** From $(2.5,1.5)$, steepest descent needs 16 iterations to reach $g\approx10^{-13}$; Newton needs 5. Near a root $g\approx(\mathbf x-\mathbf p)^TJ^TJ(\mathbf x-\mathbf p)$ and steepest descent zig-zags with rate governed by $K(J^TJ)$ (Chapter 7).

**Example 10.4.5 (a minimum that is not a root).** $x^2+y^2 = 1$ and $x^2+y^2 = 4$ is inconsistent. Steepest descent from $(1,0.5)$ converges to $(1.4142,0.7071)$ with $r^2 = 2.5$, $\nabla g = \mathbf 0$ but $g = 2(1.5)^2 = 4.5>0$. It found the *least-squares* compromise, not a solution — always check $g$ at the end.

---

## 10.5 Homotopy and Continuation Methods

Embed the hard problem in a family: with $\mathbf x(0)$ given,
$$
H(\mathbf x,\lambda) = \mathbf F(\mathbf x) + (\lambda - 1)\mathbf F(\mathbf x(0)),\qquad\lambda\in[0,1].
$$
$H(\cdot,0)$ has the known solution $\mathbf x(0)$ and $H(\cdot,1) = \mathbf F$. If the solution curve $\mathbf x(\lambda)$ exists, differentiating $H(\mathbf x(\lambda),\lambda) = \mathbf 0$ gives the IVP
$$
\frac{d\mathbf x}{d\lambda} = -J(\mathbf x(\lambda))^{-1}\mathbf F(\mathbf x(0)),\qquad 0\le\lambda\le1,
$$
which B&F (Algorithm 10.4) integrates with RK4 in $N$ steps; the endpoint $\mathbf x(1)$ is then polished by Newton. The method can fail at **turning points** where $J$ is singular along the path.

**Natural parameter continuation** is the simpler everyday version: to solve $\mathbf F(\mathbf x;\lambda) = \mathbf 0$ for a sequence of parameters (a regularisation path, a sweep over prices, a bifurcation study), use the solution for $\lambda_{j}$ as the Newton starting value for $\lambda_{j+1}$.

### Examples for §10.5

**Example 10.5.1 (homotopy path).** $\mathbf F = (x^2+y-11,\ x+y^2-7)$, $\mathbf x(0) = (1,1)$, $N = 4$ RK4 steps: $\lambda = 0.25$: $(1.742, 1.230)$; $0.5$: $(2.238, 1.505)$; $0.75$: $(2.646, 1.763)$; $1$: $(3.0026, 2.0004)$. Newton then gives $(3,2)$ exactly.

**Example 10.5.2 (where plain Newton struggles).** $\arctan x = 1$ (with the trivial second equation $y = x$) from $x = 3$: Newton gives $0.51, 1.176, 1.495, 1.556,\ldots$ — it survives here but only after an overshoot to the far side. The homotopy with $N = 10$ follows the path monotonically to $1.5574079$ (true $\tan1 = 1.5574077$).

**Example 10.5.3 (accuracy of the path integration).** Endpoint error for Example 10.5.1: $N = 1$: $0.169$; $2$: $0.024$; $4$: $0.0026$; $8$: $2.1\times10^{-4}$; $16$: $1.3\times10^{-5}$ — RK4's $O(h^4)$ with $h = 1/N$. A crude path plus Newton polishing is usually enough.

**Example 10.5.4 (natural continuation).** Solve $x^3 + x = \lambda$ for $\lambda = 0, 0.25,\dots,2$, starting each Newton solve from the previous root: $0,\ 0.2367,\ 0.4239,\ 0.5674,\ 0.6823,\ 0.7784,\ 0.8612,\ 0.9343,\ 1.0$ — each solve converges in 2–3 iterations.

**Example 10.5.5 (a turning point).** For $x^3 - x = \lambda$ following the branch that starts at $x = -1.32$ ($\lambda = -1$): $\lambda = 0$: $x = -1$; $\lambda = 0.2$: $x = -0.879$; at $\lambda = 0.4$ Newton **fails**, because the branch folds back at $\lambda = \frac{2}{3\sqrt3} = 0.385$ (where $F'(x) = 3x^2-1 = 0$); for larger $\lambda$ only the other branch ($x\approx1.22$ at $\lambda = 0.6$) exists. Detecting such folds requires arc-length continuation.

---

## 10.6 Nonlinear Systems in Data Science *(data-science extension)*

| Problem | Equations | Method |
|---|---|---|
| multi-parameter MLE | score $\nabla\ell(\boldsymbol\theta) = \mathbf 0$ | Newton (Hessian = $J$ of the score), Fisher scoring |
| logistic regression | $X^T(\mathbf y - \sigma(X\mathbf w)) = \mathbf 0$ | Newton = IRLS |
| mixture models | EM update $\boldsymbol\theta\mapsto M(\boldsymbol\theta)$ | fixed-point iteration (linear rate) |
| $k$-means | Lloyd update of centres | fixed-point iteration (finite, local) |
| economic equilibrium | supply = demand | Newton / Broyden |

### Examples for §10.6

**Example 10.6.1 (Weibull MLE).** For $n = 200$ Weibull observations (true shape $k = 1.8$, scale $\lambda = 3$), the score equations
$$
\frac nk + \sum\ln x_i - n\ln\lambda - \sum\Bigl(\frac{x_i}\lambda\Bigr)^k\ln\frac{x_i}\lambda = 0,\qquad -\frac{nk}{\lambda} + \frac k\lambda\sum\Bigl(\frac{x_i}{\lambda}\Bigr)^k = 0
$$
are solved by Newton (finite-difference Jacobian) from $(1,\bar x)$ in 6 iterations: $\hat k = 1.8755$, $\hat\lambda = 3.2375$.

**Example 10.6.2 (logistic regression = Newton on the score).** Data $x = 0.5,1,\dots,4$ with labels $0,0,1,0,1,0,1,1$. Newton on $\nabla\ell(\mathbf w) = X^T(\mathbf y-\boldsymbol\sigma) = \mathbf 0$ with Jacobian $-X^TDX$ ($D = \operatorname{diag}\sigma_i(1-\sigma_i)$) from $\mathbf w = \mathbf 0$: $(-2.1429, 0.9524)$, $(-2.6215, 1.1651)$, $(-2.6728, 1.1879)$, $(-2.67338, 1.18817)$ — converged in 4 iterations: $\hat w_0 = -2.6734$, $\hat w_1 = 1.1882$.

**Example 10.6.3 (EM is a fixed-point iteration).** EM for a two-component Gaussian mixture (500 points, true weights $0.6/0.4$, means $-2, 3$) converges to $\hat\pi = 0.600$, $\hat\mu = (-2.048, 2.990)$, $\hat\sigma = (1.050, 1.439)$. The error in $\mu_1$ is $1.6\times10^{-2},\ 2.9\times10^{-3},\ 9\times10^{-5},\ 8\times10^{-8}$ at iterations $5,10,20,40$ — linear convergence with ratio $\approx0.71$ (the spectral radius of the EM map's Jacobian, which equals the "fraction of missing information"). Newton or Aitken-type acceleration can speed EM up.

**Example 10.6.4 (market equilibrium).** Two goods with demands $100 - 2p_1 + p_2$, $80 - 3p_2 + 0.5p_1$ and supplies $10 + 3p_1$, $5 + 2p_2^{1.2}$. Newton (finite-difference Jacobian) on "demand − supply = 0" from $(10,10)$ gives equilibrium prices $(20.683, 13.415)$ in 4 iterations.

**Example 10.6.5 ($k$-means as a fixed point).** Lloyd's algorithm alternates assignment and centre updates: a fixed-point iteration that terminates in finitely many steps at a *local* optimum. On three well-separated clusters: a spread-out initialisation reaches the right centres $(0,0), (3,3), (0,3.8)$ in 2 iterations (SSE $61.5$); three initial centres from the same cluster get stuck in a poor fixed point (SSE $297.3$). Hence $k$-means++ initialisation and multiple restarts.

---

## Chapter summary

| Method | Needs | Convergence | Cost / iteration |
|---|---|---|---|
| fixed point | contraction $\rho(J_G)<1$ | linear | 1 $\mathbf G$ |
| Newton | $J$, good start | quadratic | $\mathbf F$, $J$, $O(n^3)$ solve |
| damped Newton | $J$ | global-ish + quadratic | + line search |
| Broyden | $A_0$ | superlinear | 1 $\mathbf F$, $O(n^2)$ |
| steepest descent on $\lVert\mathbf F\rVert^2$ | $J$ | linear, global to local min | $\mathbf F$, $J$, line search |
| homotopy | $J$ | robust path following | $N$ RK4 steps |

## Further reading

Burden & Faires Ch. 10 · Dennis & Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear Equations* (SIAM Classics) · Kelley, *Solving Nonlinear Equations with Newton's Method* · McLachlan & Krishnan, *The EM Algorithm and Extensions*.
