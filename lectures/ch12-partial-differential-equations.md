# Chapter 12 — Numerical Solutions to Partial Differential Equations

> **Reference:** Burden & Faires, Chapter 12 (§12.1–12.3) plus §12.4 for data science.
> **Code:** [`numlib/pde.py`](../numlib/pde.py) · **All numbers reproduced by** [`lectures/code/ch12.py`](code/ch12.py)
> **Worked problems:** [`examples/ch12`](../examples/ch12_examples.md) · **Homework:** [`exercises/ch12`](../exercises/ch12_exercises.md)

## Learning outcomes

1. Classify second-order linear PDEs (elliptic, parabolic, hyperbolic) and match each with appropriate boundary/initial conditions.
2. Discretise Poisson's equation with the five-point stencil and solve the sparse system directly or iteratively.
3. Solve the heat equation by the forward-difference, backward-difference and Crank–Nicolson methods; analyse stability by eigenvalues (matrix method) and state orders of accuracy.
4. Solve the wave equation by the explicit scheme and apply the CFL condition; recognise numerical diffusion in upwind advection.
5. Use diffusion PDEs in data science: signal/image smoothing, inpainting, option pricing and diffusion on graphs.

For $Au_{xx} + 2Bu_{xy} + Cu_{yy} + \dots = 0$ the type is determined by $B^2 - AC$: **elliptic** ($<0$, e.g. Poisson), **parabolic** ($=0$, e.g. heat), **hyperbolic** ($>0$, e.g. wave).

| type | model equation | conditions | physical meaning |
|---|---|---|---|
| elliptic | $u_{xx}+u_{yy} = f(x,y)$ | boundary values on $\partial R$ | steady state (temperature, potential) |
| parabolic | $u_t = \alpha^2u_{xx}$ | $u(x,0)$ + boundary values | diffusion, smoothing |
| hyperbolic | $u_{tt} = \alpha^2u_{xx}$ | $u(x,0)$, $u_t(x,0)$ + boundary values | waves, transport |

---

## 12.1 Elliptic Partial Differential Equations

**Poisson's equation** $\nabla^2u = u_{xx} + u_{yy} = f(x,y)$ on $R = (a,b)\times(c,d)$ with $u = g$ on $\partial R$ (Laplace's equation if $f\equiv0$).

**Five-point finite-difference method.** With $h = \frac{b-a}{n}$, $k = \frac{d-c}{m}$, $x_i = a+ih$, $y_j = c+jk$ and centred second differences:
$$
2\Bigl[\Bigl(\frac hk\Bigr)^2+1\Bigr]w_{ij} - (w_{i+1,j}+w_{i-1,j}) - \Bigl(\frac hk\Bigr)^2(w_{i,j+1}+w_{i,j-1}) = -h^2f(x_i,y_j),
$$
for interior nodes, with $w = g$ on the boundary. Local truncation error $O(h^2+k^2)$.

**The linear system.** With $(n-1)(m-1)$ unknowns ordered row by row, the matrix is symmetric positive definite, block tridiagonal, has at most 5 non-zeros per row and bandwidth $\approx n$. Solution methods:
* **Gauss–Seidel / SOR** (B&F Algorithm 12.1) — simple, memory-light; optimal $\omega = \frac{2}{1+\sin(\pi h)}$ for the unit square;
* **conjugate gradient** (Chapter 7) — $O(\sqrt\kappa) = O(1/h)$ iterations;
* sparse direct (banded Cholesky / nested dissection); fast Poisson solvers (FFT); multigrid ($O(N)$).

**Maximum principle.** Solutions of Laplace's equation attain their max and min on the boundary; the five-point scheme inherits this (each interior value is the average of its four neighbours) — a useful sanity check.

### Examples for §12.1

**Example 12.1.1 (steady plate temperature by hand).** A square plate with the top edge at 100° and the other edges at 0°, $h = \frac13$: four interior unknowns ($w_1, w_2$ top row, $w_3, w_4$ bottom row). The equations $4w_1 - w_2 - w_3 = 100$, $4w_2 - w_1 - w_4 = 100$, $4w_3 - w_1 - w_4 = 0$, $4w_4 - w_2 - w_3 = 0$ give $w_1 = w_2 = 37.5$, $w_3 = w_4 = 12.5$ — each interior value is the mean of its neighbours, and all lie between 0 and 100 (maximum principle).

**Example 12.1.2 (Poisson with a known solution).** $\nabla^2u = -2\pi^2\sin\pi x\sin\pi y$ on the unit square, $u = 0$ on the boundary; exact $u = \sin\pi x\sin\pi y$ with $u(\frac12,\frac12) = 1$. With $h = k = 0.25$: $w(0.5,0.5) = 1.053029$.

**Example 12.1.3 ($O(h^2)$ convergence).**

| $h$ | 1/4 | 1/8 | 1/16 | 1/32 |
|---|---|---|---|---|
| $w(\frac12,\frac12)$ | 1.053029 | 1.012951 | 1.003219 | 1.000804 |
| max error | $5.3\times10^{-2}$ | $1.3\times10^{-2}$ | $3.2\times10^{-3}$ | $8.0\times10^{-4}$ |

**Example 12.1.4 (iterative solvers).** $-\nabla^2u = 1$ on a $20\times20$ interior grid (400 unknowns), tolerance $10^{-8}$: Gauss–Seidel 814 iterations, SOR with $\omega^\ast = 1.7406$ 79 iterations, CG 36 iterations. (For the right-hand side of Example 12.1.2 CG converges in *one* step, because $\sin\pi x\sin\pi y$ sampled on the grid is an eigenvector of the discrete Laplacian — cf. Example 7.6.5.)

**Example 12.1.5 (sparsity).** $m\times m$ interior grid:

| $m$ | unknowns | non-zeros | dense entries |
|---|---|---|---|
| 10 | 100 | 460 | $10^4$ |
| 100 | $10^4$ | 49 600 | $10^8$ |
| 1000 | $10^6$ | $\approx5\times10^6$ | $10^{12}$ |

A dense solver for $m = 1000$ would need 8 TB of memory; sparse methods need ~100 MB.

---

## 12.2 Parabolic Partial Differential Equations

The heat (diffusion) equation $u_t = \alpha^2u_{xx}$, $0<x<l$, $t>0$, with $u(0,t) = u(l,t) = 0$ and $u(x,0) = f(x)$. Grid: $x_i = ih$ ($h = l/m$), $t_j = jk$; let $\lambda = \alpha^2k/h^2$.

**Forward-difference (explicit, FTCS).**
$$
w_{i,j+1} = (1-2\lambda)w_{ij} + \lambda(w_{i+1,j}+w_{i-1,j}),\qquad\mathbf w^{(j+1)} = A\mathbf w^{(j)},\ A = \operatorname{tridiag}(\lambda, 1-2\lambda, \lambda).
$$
Truncation error $O(k+h^2)$. **Stability (matrix method):** errors propagate as $\mathbf e^{(j)} = A^j\mathbf e^{(0)}$, so we need $\rho(A)\le1$. The eigenvalues of $A$ are
$$
\mu_i = 1 - 4\lambda\sin^2\Bigl(\frac{i\pi}{2m}\Bigr),\qquad i = 1,\dots,m-1,
$$
and $\rho(A)\le1$ iff $\lambda\le\frac12$: the method is **conditionally stable** ($k\le\frac{h^2}{2\alpha^2}$ — halving $h$ requires quartering $k$).

**Backward-difference (implicit, BTCS).**
$$
(1+2\lambda)w_{ij} - \lambda(w_{i+1,j}+w_{i-1,j}) = w_{i,j-1},
$$
a tridiagonal solve per step. Eigenvalues of the iteration matrix $\frac{1}{1+4\lambda\sin^2(i\pi/2m)}\in(0,1)$: **unconditionally stable**. Error $O(k+h^2)$.

**Crank–Nicolson.** Average the two:
$$
(1+\lambda)w_{i,j+1} - \frac\lambda2(w_{i+1,j+1}+w_{i-1,j+1}) = (1-\lambda)w_{ij} + \frac\lambda2(w_{i+1,j}+w_{i-1,j}).
$$
Unconditionally stable and $O(k^2+h^2)$ (B&F Algorithm 12.3). Its amplification factors tend to $-1$ for high frequencies when $\lambda$ is large, so non-smooth data can produce slowly decaying oscillations.

### Examples for §12.2

**Example 12.2.1 (the stability limit is real).** $u_t = u_{xx}$ on $[0,1]$ with the tent $u(x,0) = 1 - |2x-1|$, $h = 0.1$, $t = 0.1$ (exact $u(0.5,0.1) = 0.302118$ from the Fourier series):

| $k$ | $\lambda$ | steps | $w(0.5,0.1)$ | max error |
|---|---|---|---|---|
| 0.004 | 0.40 | 25 | 0.301098 | $1.0\times10^{-3}$ |
| 0.005 | 0.50 | 20 | 0.307083 | $9.6\times10^{-3}$ |
| 0.00588 | 0.59 | 17 | **−1.372968** | 1.68 |

At $\lambda = 0.59$ the numerical solution oscillates wildly ($-0.42, 1.15, -1.11, 1.87,\ldots$) — the highest-frequency mode is amplified by $|1-4(0.59)\sin^2(0.45\pi)|\approx1.3$ per step.

**Example 12.2.2 (eigenvalues explain it).** For $m = 10$: $\lambda = 0.4$: eigenvalues of $A$ in $[-0.561, 0.961]$, $\rho = 0.961$; $\lambda = 0.5$: $[-0.951, 0.951]$; $\lambda = 0.6$: $[-1.341, 0.941]$, $\rho = 1.341>1$.

**Example 12.2.3 (backward difference is always stable).** Smooth data $u(x,0) = \sin\pi x + 0.5\sin3\pi x$ (exact $u = e^{-\pi^2t}\sin\pi x + 0.5e^{-9\pi^2t}\sin3\pi x$), $h = 0.1$, $t = 0.5$: BTCS with $k = 0.01$ ($\lambda = 1$): max error $2.2\times10^{-3}$; with $k = 0.05$ ($\lambda = 5$): $1.1\times10^{-2}$ — stable for large $\lambda$ but only first-order accurate in time.

**Example 12.2.4 (Crank–Nicolson).** Same problem: $k = 0.01$: max error $2.7\times10^{-4}$; $k = 0.05$: $4.4\times10^{-4}$ — second-order in time lets CN take large steps accurately.

**Example 12.2.5 (convergence study at $t = 0.5$).**

| $m$ | FTCS ($k = 0.4h^2$) | BTCS ($k = h/5$) | CN ($k = h/5$) |
|---|---|---|---|
| 10 | $4.1\times10^{-4}$ (125 steps) | $4.3\times10^{-3}$ (25 steps) | $1.8\times10^{-4}$ |
| 20 | $1.0\times10^{-4}$ (500) | $1.9\times10^{-3}$ (50) | $4.4\times10^{-5}$ |
| 40 | $2.6\times10^{-5}$ (2000) | $9.2\times10^{-4}$ (100) | $1.1\times10^{-5}$ |

FTCS and CN errors drop ×4 per halving of $h$ ($O(h^2)$); BTCS only ×2 (dominated by $O(k) = O(h)$). FTCS needs 20× more steps than CN for similar accuracy.

---

## 12.3 Hyperbolic Partial Differential Equations

The wave equation $u_{tt} = \alpha^2u_{xx}$, $0<x<l$, with $u(0,t) = u(l,t) = 0$, $u(x,0) = f(x)$, $u_t(x,0) = g(x)$. With $\lambda = \alpha k/h$, centred differences in both variables give the explicit scheme
$$
w_{i,j+1} = 2(1-\lambda^2)w_{ij} + \lambda^2(w_{i+1,j}+w_{i-1,j}) - w_{i,j-1},
$$
started with $w_{i,0} = f(x_i)$ and the second-order first step
$$
w_{i,1} = (1-\lambda^2)f(x_i) + \frac{\lambda^2}{2}[f(x_{i+1})+f(x_{i-1})] + kg(x_i).
$$
Truncation error $O(h^2+k^2)$. **CFL condition:** the scheme is stable iff $\lambda\le1$ — the numerical domain of dependence must contain the physical one ($\alpha k\le h$). When $\lambda = 1$ the scheme is *exact* at the grid points (it reproduces d'Alembert's solution $u = \frac12[F(x-\alpha t)+F(x+\alpha t)]$).

**First-order transport** $u_t + cu_x = 0$ (advection) with the **upwind** scheme $w_{i,j+1} = w_{ij} - \frac{ck}{h}(w_{ij}-w_{i-1,j})$ is stable for $0\le\frac{ck}{h}\le1$ but introduces **numerical diffusion**: it actually solves $u_t + cu_x = \frac{ch}{2}(1-\frac{ck}{h})u_{xx}$ to second order.

### Examples for §12.3

**Example 12.3.1.** $u_{tt} = u_{xx}$, $u(x,0) = \sin\pi x$, $u_t(x,0) = 0$ (exact $u = \sin\pi x\cos\pi t$), $h = 0.1$, at $t = 1$:

| $\lambda$ | $w(0.5,1)$ | max error |
|---|---|---|
| 1.0 | −1.000000 | $3\times10^{-16}$ |
| 0.769 | −0.999986 | $1.4\times10^{-5}$ |
| 0.5 | −0.999953 | $4.7\times10^{-5}$ |

$\lambda = 1$ is exact; smaller $\lambda$ adds dispersion error.

**Example 12.3.2 (violating CFL).** Same problem to $t = 5$: $\lambda = 1$: $\max|w| = 1.00$; $\lambda = 1.11$: $3.33$; $\lambda = 1.25$: $1.4\times10^6$.

**Example 12.3.3 (a plucked string).** $u(x,0)$ = tent with peak 1 at $x = 0.5$, $u_t = 0$, $h = 0.1$, $\lambda = 1$. At $t = 0.3$: $w = (0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0)$ — identical (to $5\times10^{-16}$) to d'Alembert's solution: two half-height tents travelling outward, flattening the middle.

**Example 12.3.4 (periodicity).** With $\alpha = 1$, $l = 1$ the solution has period 2. After one period ($t = 2$, $\lambda = 1$, $h = 0.02$) the numerical solution returns to $\sin\pi x$ within $4\times10^{-15}$.

**Example 12.3.5 (numerical diffusion).** Advect a Gaussian pulse $e^{-200(x-0.25)^2}$ with $c = 1$ to $t = 0.5$ by upwind ($h = 0.01$, Courant number $0.5$). The pulse arrives at $x = 0.75$ as it should, but its peak has dropped from 1 to $0.707$ and it has spread — artificial viscosity $\frac{ch}{2}(1-0.5) = 0.0025$. Higher-order or flux-limited schemes reduce this.

---

## 12.4 Diffusion and Waves in Data Science *(data-science extension)*

* **The heat equation is Gaussian smoothing.** On $\mathbb R$, $u(\cdot,t) = G_\sigma * f$ with $\sigma = \sqrt{2t}$ ($\alpha = 1$). Explicit heat steps = repeated local averaging = blur; scale-space theory in computer vision.
* **Image processing:** isotropic diffusion denoises but blurs edges; Perona–Malik (nonlinear) diffusion preserves them; **harmonic inpainting** fills missing pixels by solving Laplace's equation.
* **Finance:** the Black–Scholes PDE $V_t + \frac12\sigma^2S^2V_{SS} + rSV_S - rV = 0$ is a (backward) heat equation; Crank–Nicolson is the industry workhorse.
* **Graphs:** the graph heat equation $\mathbf u' = -L\mathbf u$ has solution $e^{-tL}\mathbf u_0$ — diffusion kernels, graph neural networks (message passing), label smoothing.
* **Generative models:** diffusion models add noise by a forward diffusion and learn to reverse it.

### Examples for §12.4

**Example 12.4.1 (smoothing a noisy signal by heat flow).** A square wave (3 periods on $[0,1]$, 256 samples) plus noise ($\sigma = 0.3$, RMSE $0.304$). Running the explicit heat equation to time $t$ (equivalent Gaussian width $\sqrt{2t}$): $t = 10^{-5}$ ($\sigma_G = 0.0045$): RMSE $0.209$; $t = 3\times10^{-5}$: $0.228$; $t = 10^{-4}$: $0.282$; $t = 10^{-3}$: $0.486$ (edges destroyed). Isotropic diffusion must stop early — the motivation for edge-preserving (nonlinear) diffusion.

**Example 12.4.2 (2-D image denoising).** A $64\times64$ image of nested squares with noise (PSNR $14.0$ dB). Explicit 2-D heat steps ($\lambda = 0.2$): after 2 steps PSNR $20.1$ dB; 5 steps $19.0$; 10 steps $17.8$; 20 steps $16.2$. Early stopping acts as the regulariser.

**Example 12.4.3 (Black–Scholes by Crank–Nicolson).** European call, $K = 100$, $r = 0.05$, $\sigma = 0.2$, $T = 1$, grid $S\in[0,300]$ with $\Delta S = 1$, 200 time steps. CN gives $V(100,0) = 10.4481$; the closed-form Black–Scholes price is $10.4506$ (error $0.0025$). Same tridiagonal machinery as the heat equation.

**Example 12.4.4 (diffusion on a graph).** On a 6-node graph (two triangles joined by an edge) the signal $\mathbf f_0 = (1, 0.2, 0.9, -0.8, -1.1, -0.1)$ evolves as $e^{-tL}\mathbf f_0$: $t = 0.1$: $(0.92, 0.32, 0.71, -0.63, -0.98, -0.24)$; $t = 0.5$: $(0.67, 0.49, 0.40, -0.35, -0.67, -0.44)$; $t = 2$: $(0.32, 0.32, 0.19, -0.15, -0.29, -0.28)$. Values equalise within each cluster first and the mean ($0.0167$) is conserved — the basis of graph smoothing and GNN aggregation.

**Example 12.4.5 (harmonic inpainting).** An $8\times44$ strip of the image is deleted and filled by solving Laplace's equation with the surrounding pixels as boundary data (Gauss–Seidel sweeps): RMSE in the hole $0.195$, compared with $0.674$ for filling with zeros. (Sharp edges crossing the hole are smoothed — higher-order PDE or learned inpainting does better.)

---

## Chapter summary

| Equation | Scheme | Accuracy | Stability |
|---|---|---|---|
| Poisson | 5-point | $O(h^2+k^2)$ | SPD sparse system (SOR, CG, multigrid) |
| heat | forward (FTCS) | $O(k+h^2)$ | $\lambda = \alpha^2k/h^2\le\frac12$ |
| heat | backward (BTCS) | $O(k+h^2)$ | unconditional |
| heat | Crank–Nicolson | $O(k^2+h^2)$ | unconditional |
| wave | explicit centred | $O(k^2+h^2)$ | CFL $\alpha k/h\le1$ |
| advection | upwind | $O(k+h)$ | $0\le ck/h\le1$; numerical diffusion |

## Further reading

Burden & Faires Ch. 12 · LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations* (SIAM 2007) · Weickert, *Anisotropic Diffusion in Image Processing* · Wilmott, Howison & Dewynne, *The Mathematics of Financial Derivatives*.
