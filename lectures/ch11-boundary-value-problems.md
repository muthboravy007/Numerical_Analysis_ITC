# Chapter 11 — Boundary-Value Problems for Ordinary Differential Equations

> **Reference:** Burden & Faires, Chapter 11 (§11.1–11.5) plus §11.6 for data science.
> **Code:** [`numlib/bvp.py`](../numlib/bvp.py) · **All numbers reproduced by** [`lectures/code/ch11.py`](code/ch11.py)
> **Worked problems:** [`examples/ch11`](../examples/ch11_examples.md) · **Homework:** [`exercises/ch11`](../exercises/ch11_exercises.md)

## Learning outcomes

1. State existence/uniqueness conditions for two-point boundary-value problems (BVPs).
2. Solve linear BVPs by the linear shooting method (superposition of two IVPs) and recognise its instability for rapidly growing solutions.
3. Solve nonlinear BVPs by shooting with Newton's or the secant method on the initial slope.
4. Discretise linear and nonlinear BVPs by centred finite differences, solve the resulting tridiagonal systems, and improve accuracy by Richardson extrapolation.
5. Derive the Rayleigh–Ritz (finite-element) method with piecewise-linear basis functions.
6. Recognise BVP structure in smoothing, trend filtering, graph-based interpolation and parameter calibration.

We consider
$$
y'' = f(x,y,y'),\qquad a\le x\le b,\qquad y(a) = \alpha,\quad y(b) = \beta,
$$
and the linear case $y'' = p(x)y' + q(x)y + r(x)$. Unlike an IVP, information is given at **both** ends.

---

## 11.1 The Linear Shooting Method

**Theorem 11.1 (existence and uniqueness).** If $f$ and $f_y$, $f_{y'}$ are continuous on $D = \{a\le x\le b,\ -\infty<y,y'<\infty\}$ and (i) $f_y(x,y,y')>0$, (ii) $|f_{y'}(x,y,y')|\le M$ on $D$, then the BVP has a unique solution.

**Corollary 11.2 (linear case).** If $p,q,r$ are continuous on $[a,b]$ and $q(x)>0$, then $y'' = py'+qy+r$, $y(a)=\alpha$, $y(b)=\beta$ has a unique solution.

**Linear shooting.** Solve two IVPs:
$$
\begin{aligned}
y_1'' &= py_1' + qy_1 + r, & y_1(a) &= \alpha, & y_1'(a) &= 0,\\
y_2'' &= py_2' + qy_2, & y_2(a) &= 0, & y_2'(a) &= 1.
\end{aligned}
$$
By linearity, $y = y_1 + \frac{\beta - y_1(b)}{y_2(b)}y_2$ satisfies the ODE and both boundary conditions (provided $y_2(b)\ne0$, which Corollary 11.2 guarantees). Each IVP is solved by RK4 (Chapter 5), so the method inherits $O(h^4)$ accuracy.

**Instability.** If $y_1$ and $y_2$ grow rapidly (e.g. $e^{kx}$ with large $k$), $y_1(b)$ and $y_2(b)$ are huge, and forming $y_1 + cy_2$ subtracts nearly equal large numbers — cancellation (Chapter 1). Remedies: shoot backward, multiple shooting, or finite differences.

### Examples for §11.1

**Example 11.1.1.** $y'' = y + x$, $y(0) = 0$, $y(1) = 1$ ($p = 0$, $q = 1>0$, $r = x$). Exact solution $y = \frac{2\sinh x}{\sinh1} - x$. Linear shooting with RK4, $h = 0.1$:

| $x$ | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|
| $w$ | 0.14264098 | 0.29903332 | 0.48348030 | 0.71141108 |
| $y$ | 0.14264091 | 0.29903320 | 0.48348015 | 0.71141096 |
| error | $6.7\times10^{-8}$ | $1.2\times10^{-7}$ | $1.5\times10^{-7}$ | $1.2\times10^{-7}$ |

**Example 11.1.2 (the two IVPs).** Here $y_1 = \sinh x - x$ and $y_2 = \sinh x$. RK4 gives $y_1(1) = 0.1751999849$ and $y_2(1) = 1.1751999849$ (exact $0.1752011936$, $1.1752011936$), so $c = \frac{1 - y_1(1)}{y_2(1)} = 0.701838$ and $y = y_1 + 0.701838\,y_2$.

**Example 11.1.3 (fourth-order accuracy).** Max errors for $N = 5, 10, 20, 40$ steps: $1.9\times10^{-6}$, $1.5\times10^{-7}$, $1.0\times10^{-8}$, $6.6\times10^{-10}$ — ratio ≈ 15 per halving.

**Example 11.1.4 (shooting can be unstable).** $y'' = 400y$, $y(0) = 1$, $y(1) = e^{-20}$ (exact $y = e^{-20x}$). The IVP solutions grow like $e^{20x}\approx4.9\times10^8$ at $x = 1$. With $N = 20$ the max error is $7.1\times10^{-3}$ and $w(0.5) = 5.5\times10^{-5}$ vs the exact $4.54\times10^{-5}$ (20% relative error); $N = 50$: $1.1\times10^{-4}$; $N = 100$: $5.8\times10^{-6}$. Very small steps are needed not for truncation but to control amplification.

**Example 11.1.5 (heat in a rod with convective loss).** Steady temperature: $T'' = h'(T - T_a)$ with $h' = 0.01\ \text{m}^{-2}$, $T_a = 20$, $T(0) = 40$, $T(10) = 200$. In the standard form $p = 0$, $q = h'$, $r = -h'T_a$. Shooting with $h = 0.5$: $T(2.5) = 72.686$, $T(5) = 108.68189$ (exact $108.68189$), $T(7.5) = 150.249$.

---

## 11.2 Shooting for Nonlinear Problems

For $y'' = f(x,y,y')$, solve the IVP $y'' = f$, $y(a) = \alpha$, $y'(a) = t$, and choose the slope $t$ so that the endpoint condition holds: find a root of
$$
\phi(t) = y(b,t) - \beta = 0 .
$$
**Newton's method:** $t_k = t_{k-1} - \frac{y(b,t_{k-1})-\beta}{z(b,t_{k-1})}$ where $z = \partial y/\partial t$ satisfies the **variational equation**
$$
z'' = f_y(x,y,y')\,z + f_{y'}(x,y,y')\,z',\qquad z(a) = 0,\quad z'(a) = 1,
$$
solved together with the original IVP (B&F Algorithm 11.2). The **secant method** avoids $z$ altogether.

### Examples for §11.2

**Example 11.2.1 (Newton shooting).** $y'' = 2y^3$, $y(1) = \frac12$, $y(2) = \frac13$, exact $y = \frac{1}{x+1}$ (so the true slope is $y'(1) = -\frac14$). Here $f_y = 6y^2$, $f_{y'} = 0$. From $t_0 = 0$: $t_1 = -0.234949$, $t_2 = -0.249955$, $t_3 = -0.2500003$ (the RK4-discrete optimum, $3.4\times10^{-7}$ from $-\frac14$). The resulting solution ($h = 0.1$) has max error $5\times10^{-8}$: $w(1.2) = 0.45454550$, $w(1.6) = 0.38461543$.

**Example 11.2.2 (secant shooting).** Same problem with $t_0 = 0$, $t_1 = -0.5$: $-0.262254,\ -0.249481,\ -0.2500016,\ -0.2500003$ — no variational equation needed, slightly more iterations.

**Example 11.2.3 (why iterate?).** With the naive slope $t = 0$, the IVP gives $y(2) = 0.6434$ instead of $0.3333$: the "shot" misses by $0.31$. Each Newton iteration corrects the aim.

**Example 11.2.4 (multiple solutions — the Bratu problem).** $y'' + e^y = 0$, $y(0) = y(1) = 0$ (a model of combustion). $\phi(t)$ has **two** roots: $t = 0.549353$ (lower solution, $\max y = 0.1405$) and $t = 10.846900$ (upper solution, $\max y = 4.0915$). A nonlinear BVP need not have a unique solution; the starting guess decides which one you get.

**Example 11.2.5 (the original "shooting" problem).** A ball thrown upward with quadratic drag, $h'' = -g - 0.02\,h'|h'|$, must return to the ground after 3 s: $h(0) = h(3) = 0$. Shooting on the launch speed gives $v_0 = 16.49$ m/s (vacuum: $gT/2 = 14.72$ m/s) and a maximum height of $11.03$ m.

---

## 11.3 Finite-Difference Methods for Linear Problems

Divide $[a,b]$ into $N+1$ subintervals, $h = \frac{b-a}{N+1}$, $x_i = a + ih$. Replace derivatives by centred differences (Chapter 4):
$$
y''(x_i) = \frac{y_{i+1}-2y_i+y_{i-1}}{h^2} - \frac{h^2}{12}y^{(4)}(\xi_i),\qquad y'(x_i) = \frac{y_{i+1}-y_{i-1}}{2h} - \frac{h^2}6y'''(\eta_i).
$$
This gives, for $i = 1,\dots,N$ (with $w_0 = \alpha$, $w_{N+1} = \beta$),
$$
-\Bigl(1+\frac h2p(x_i)\Bigr)w_{i-1} + \bigl(2+h^2q(x_i)\bigr)w_i - \Bigl(1-\frac h2p(x_i)\Bigr)w_{i+1} = -h^2r(x_i),
$$
a **tridiagonal** system, solved in $O(N)$ by Crout/Thomas (Chapter 6).

**Theorem 11.3.** If $p,q,r$ are continuous and $q(x)\ge0$ on $[a,b]$, the tridiagonal system has a unique solution provided $h<2/L$, $L = \max|p(x)|$ (then the matrix is diagonally dominant). The error is $O(h^2)$ if $y\in C^4$.

**Richardson extrapolation.** Since the error expands in even powers of $h$, combine solutions at $h$, $h/2$, $h/4$ at common nodes: $\text{Ext}_1 = \frac{4w(h/2) - w(h)}{3}$, $\text{Ext}_2 = \frac{16\,\text{Ext}_1(h/2) - \text{Ext}_1(h)}{15}$.

### Examples for §11.3

**Example 11.3.1 (FD by hand, $h = 0.2$).** For $y'' = y + x$: $-w_{i-1} + 2.04w_i - w_{i+1} = -0.04x_i$, $i = 1..4$, with $w_0 = 0$, $w_5 = 1$ (right-hand side $(-0.008, -0.016, -0.024, 0.968)$ after moving $w_5$). Solution: $w = (0.14281, 0.29933, 0.48383, 0.71168)$; errors $1.7, 3.0, 3.5, 2.7\ (\times10^{-4})$.

**Example 11.3.2 (second-order convergence).** Max errors for $h = 0.2, 0.1, 0.05, 0.025$: $3.5\times10^{-4}$, $8.8\times10^{-5}$, $2.2\times10^{-5}$, $5.5\times10^{-6}$ — ratio 4 per halving. (Shooting with RK4 was more accurate for this smooth problem.)

**Example 11.3.3 (Richardson).** At $x = 0.5$: $w(h{=}0.5) = 0.388889$, $w(0.25) = 0.387348$, $w(0.125) = 0.386952$. $\text{Ext}_1$: $0.386835,\ 0.386820$; $\text{Ext}_2 = 0.38681892$ vs exact $0.38681888$ (error $3.8\times10^{-8}$) — from three very coarse grids. The $h = 0.25$ system is
$$
\begin{pmatrix}2.0625&-1&0\\-1&2.0625&-1\\0&-1&2.0625\end{pmatrix}\mathbf w = \begin{pmatrix}-0.015625\\-0.03125\\0.953125\end{pmatrix},\ \mathbf w = (0.18023, 0.38735, 0.64993).
$$

**Example 11.3.4 (rod, coarse grid).** Example 11.1.5 with $h = 1$ (9 unknowns): $T(5) = 108.690$ (error $0.0085$).

**Example 11.3.5 (conditioning of FD matrices).** For $\frac{1}{h^2}$-scaled problems the matrix $\operatorname{tridiag}(-1, 2+h^2, -1)$ has $\kappa_2\approx\frac{4}{\pi^2h^2}$: $36$ ($N = 9$), $3.7\times10^3$ ($N=99$), $3.7\times10^5$ ($N = 999$). Refining the grid reduces truncation error like $h^2$ but increases round-off amplification like $h^{-2}$ — for $h\approx10^{-4}$ the two balance in double precision.

---

## 11.4 Finite-Difference Methods for Nonlinear Problems

Discretising $y'' = f(x,y,y')$ gives the nonlinear system
$$
-w_{i-1} + 2w_i - w_{i+1} + h^2f\Bigl(x_i,\ w_i,\ \frac{w_{i+1}-w_{i-1}}{2h}\Bigr) = 0,\qquad i = 1,\dots,N,
$$
solved by **Newton's method for systems** (Chapter 10). The Jacobian is tridiagonal:
$$
J_{i,i} = 2 + h^2f_y,\qquad J_{i,i\pm1} = -1\pm\frac h2f_{y'},
$$
so each Newton step is an $O(N)$ Thomas solve. Start from the straight line $w_i = \alpha + \frac{\beta-\alpha}{b-a}(x_i-a)$. Under the conditions of Theorem 11.1 and $h<2/M$, the method converges with $O(h^2)$ accuracy (Theorem 11.5).

### Examples for §11.4

**Example 11.4.1.** $y'' = 2y^3$, $y(1) = \frac12$, $y(2) = \frac13$ with $h = 0.1$: Newton converges in 4 iterations from the linear initial guess; $w = (0.47620, 0.45457, 0.43481, 0.41669, 0.40003, 0.38464, 0.37039, 0.35716, 0.34483)$, max error $2.6\times10^{-5}$.

**Example 11.4.2 (two Bratu solutions).** With $h = 0.02$, Newton from the initial guess $0.1\sin\pi x$ converges (4 iterations) to the lower solution ($\max w = 0.14054$); from $4\sin\pi x$ (5 iterations) to the upper one ($\max w = 4.0913$) — matching the shooting results of Example 11.2.4.

**Example 11.4.3 ($O(h^2)$).** Max errors for $h = 0.2, 0.1, 0.05, 0.025$: $1.04\times10^{-4},\ 2.64\times10^{-5},\ 6.61\times10^{-6},\ 1.65\times10^{-6}$.

**Example 11.4.4 (the Jacobian).** With $f = 2y^3$: $f_y = 6y^2$, $f_{y'} = 0$, so $J = \operatorname{tridiag}(-1,\ 2 + 6h^2w_i^2,\ -1)$ — symmetric and strictly diagonally dominant (as $w_i\ne0$); at the initial guess with $h = 0.2$ the diagonal is $(2.0523, 2.0451, 2.0384, 2.0323)$.

**Example 11.4.5 (shooting vs FD).** On this smooth, stable problem nonlinear shooting (RK4, $h = 0.1$) has max error $5\times10^{-8}$ vs $2.6\times10^{-5}$ for FD. FD wins on unstable problems (Example 11.1.4) and generalises to PDEs (Chapter 12).

---

## 11.5 The Rayleigh–Ritz Method

Consider the self-adjoint problem
$$
-\frac{d}{dx}\Bigl(p(x)\frac{dy}{dx}\Bigr) + q(x)y = f(x),\quad0\le x\le1,\quad y(0) = y(1) = 0,
$$
with $p\ge\delta>0$, $q\ge0$.

**Theorem 11.6 (variational principle).** $y$ solves the BVP iff it minimises the energy functional
$$
I[u] = \int_0^1\bigl\{p(x)[u'(x)]^2 + q(x)[u(x)]^2 - 2f(x)u(x)\bigr\}dx
$$
over all $u\in C^2_0[0,1]$ (twice differentiable, vanishing at the endpoints).

**Rayleigh–Ritz.** Minimise $I$ over a finite-dimensional subspace spanned by basis functions $\phi_1,\dots,\phi_n$: $u = \sum c_i\phi_i$. Setting $\partial I/\partial c_j = 0$ gives $A\mathbf c = \mathbf b$ with
$$
a_{ij} = \int_0^1[p\phi_i'\phi_j' + q\phi_i\phi_j]\,dx,\qquad b_i = \int_0^1f\phi_i\,dx .
$$
**Piecewise-linear "hat" functions** on $x_i = ih$ ($\phi_i(x_j) = \delta_{ij}$) give a **tridiagonal, symmetric positive definite** matrix; for constant $p,q$:
$$
a_{ii} = \frac{2p}{h} + \frac{2qh}{3},\qquad a_{i,i\pm1} = -\frac ph + \frac{qh}{6}.
$$
This is the one-dimensional **finite element method**; the error is $O(h^2)$ in the maximum norm (for $p = 1$, $q = 0$ the nodal values are even exact).

### Examples for §11.5

**Example 11.5.1 (FEM by hand).** $-y'' = \pi^2\sin\pi x$, $y(0)=y(1)=0$ (exact $y = \sin\pi x$), three interior nodes ($h = 0.25$): $A = \frac1h\operatorname{tridiag}(-1,2,-1) = \begin{pmatrix}8&-4&0\\-4&8&-4\\0&-4&8\end{pmatrix}$, $\mathbf b = (1.65685, 2.34315, 1.65685)$ (integrals of $f$ times the hats). Solution $\mathbf c = (0.70711, 1, 0.70711) = (\sin\frac\pi4, \sin\frac\pi2, \sin\frac{3\pi}4)$ — **exact at the nodes**.

**Example 11.5.2 (with a reaction term).** $-y'' + y = x$, $y(0) = y(1) = 0$, exact $y = x - \frac{\sinh x}{\sinh1}$. Nodal max error: $n = 3$: $2.7\times10^{-4}$; $n = 7$: $6.9\times10^{-5}$; $n = 15$: $1.7\times10^{-5}$ — $O(h^2)$.

**Example 11.5.3 (minimum energy).** For Example 11.5.1 the discrete energy $\frac12\mathbf c^TA\mathbf c - \mathbf b^T\mathbf c$ equals $-2.343146$ at the FE solution; perturbing all coefficients by $0.01$ raises it to $-2.342746$. The FE solution is the best approximation in the energy norm — Galerkin orthogonality.

**Example 11.5.4 (nodal vs global error).** For $-y'' = \pi^2\sin\pi x$: nodal errors are $\sim10^{-16}$ for every $n$, while the max error between nodes (piecewise-linear interpolation of $\sin\pi x$) is $0.070$ ($n=3$), $0.019$ ($n=7$), $0.0048$ ($n = 15$), $0.0012$ ($n = 31$) — $O(h^2)$.

**Example 11.5.5 (FEM vs FD).** For $p = 1$, $q = 0$ the FE stiffness matrix $\frac1h\operatorname{tridiag}(-1,2,-1)$ is exactly $h$ times the FD matrix; the methods differ only in the right-hand side ($\int f\phi_i$ vs $hf(x_i)$). The variational viewpoint, however, extends naturally to irregular meshes, variable coefficients and 2-D/3-D domains.

---

## 11.6 Boundary-Value Structure in Data Science *(data-science extension)*

* **Whittaker smoother / smoothing splines.** Minimising $\sum(y_i-z_i)^2 + \lambda\sum(\Delta^2z_i)^2$ gives $(I + \lambda D^TD)\mathbf z = \mathbf y$, a banded (pentadiagonal) system — a discrete version of the BVP for a smoothing spline, $z'''' \propto (y - z)$.
* **Hodrick–Prescott filter** (economics): the same with $\lambda = 1600$ for quarterly data, separating trend and cycle.
* **Harmonic functions on graphs** (label propagation, Chapter 7): $L\mathbf f = \mathbf 0$ at unlabelled nodes with fixed values at labelled nodes is a discrete Dirichlet BVP.
* **Parameter calibration:** fit coefficients of a BVP to measurements by nonlinear least squares, solving the BVP inside every residual evaluation.
* **Eigenvalue BVPs:** $-y'' = \lambda y$ discretised gives the graph-Laplacian spectrum of a path — the basis of Laplacian eigenmaps and spectral methods.

### Examples for §11.6

**Example 11.6.1 (Whittaker smoothing).** 200 noisy samples of $\sin2\pi t + 0.5t$ ($\sigma = 0.3$). RMSE of $\mathbf z$ against the true curve: $\lambda = 1$: $0.163$; $\lambda = 100$: $0.092$; $\lambda = 10^4$: $0.037$; $\lambda = 10^6$: $0.309$ (over-smoothed toward a straight line). Choosing $\lambda$ is the bias–variance trade-off; the solve is $O(n)$.

**Example 11.6.2 (HP filter).** A 120-month series = smooth trend + 12-month cycle + noise. The HP filter ($\lambda = 1600$) recovers the trend with RMSE $0.77$, and the extracted cycle $\mathbf y - \boldsymbol\tau$ has correlation $0.80$ with the true cycle.

**Example 11.6.3 (harmonic interpolation).** On a path of 11 nodes with $f_0 = 0$, $f_{10} = 1$, solving $-f_{i-1} + 2f_i - f_{i+1} = 0$ gives $f_i = 0.1i$ — the discrete analogue of $y'' = 0$ (linear interpolation). On general graphs the same equations propagate labels.

**Example 11.6.4 (calibrating a physical coefficient).** Temperatures measured at $x = 2,4,6,8$ in the rod of Example 11.1.5: $66.4, 93.4, 125.0, 159.3$. Minimising the misfit over $h'$ (a shooting solve per evaluation) gives $\hat h' = 0.00989$ (true $0.01$) with residuals below $0.5$ °C.

**Example 11.6.5 (Laplacian eigenvalues).** The lowest eigenvalues of $\frac{1}{h^2}\operatorname{tridiag}(-1,2,-1)$ approximate those of $-y'' = \lambda y$, $y(0) = y(1) = 0$, namely $(k\pi)^2 = 9.870, 39.478, 88.826$: $N = 9$: $9.789, 38.197, 82.443$; $N = 49$: $9.866, 39.426, 88.564$; $N = 199$: $9.8694, 39.4752, 88.8100$ ($O(h^2)$ convergence, worse for higher modes).

---

## Chapter summary

| Method | Problem | Accuracy | Notes |
|---|---|---|---|
| linear shooting | linear | $O(h^4)$ (RK4) | 2 IVPs; unstable for fast-growing modes |
| nonlinear shooting | nonlinear | $O(h^4)$ | Newton/secant on the slope; multiple solutions possible |
| FD (linear) | linear | $O(h^2)$, Richardson → higher | tridiagonal solve; robust |
| FD (nonlinear) | nonlinear | $O(h^2)$ | Newton with tridiagonal Jacobian |
| Rayleigh–Ritz / FEM | self-adjoint | $O(h^2)$ | energy minimisation; extends to 2-D/3-D |

## Further reading

Burden & Faires Ch. 11 · Ascher, Mattheij & Russell, *Numerical Solution of Boundary Value Problems for ODEs* · Eilers, "A perfect smoother", *Analytical Chemistry* 75 (2003) · Hodrick & Prescott, "Postwar U.S. business cycles", *J. Money, Credit and Banking* 29 (1997).
