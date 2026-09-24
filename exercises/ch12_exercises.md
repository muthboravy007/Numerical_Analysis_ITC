# Chapter 12 — Exercises: Numerical Solutions to Partial Differential Equations

> Lecture: [Chapter 12](../lectures/ch12-partial-differential-equations.md) · Solutions: [`solutions/ch12_solutions.md`](../solutions/ch12_solutions.md) · Solution code: [`solutions/code/ch12_solutions.py`](../solutions/code/ch12_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★** Classify $Au_{xx}+Bu_{xy}+Cu_{yy}+\dots = 0$ by the sign of $B^2-4AC$ for: (a) $u_{xx}+4u_{xy}+3u_{yy}$; (b) $u_{xx}+2u_{xy}+u_{yy}$; (c) $u_{xx}+u_{xy}+u_{yy}$; (d) the Tricomi equation $yu_{xx}+u_{yy} = 0$, which changes type across $y = 0$. For (a), find the characteristic directions.

**A2 ★★** (a) Show that the 5-point Laplacian $\frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{ij}}{h^2}$ has truncation error $\frac{h^2}{12}(u_{xxxx}+u_{yyyy})+O(h^4)$. Deduce that it is exact for polynomials of degree $\le3$. (b) Prove the *discrete maximum principle*: if the discrete Laplacian of $w$ is $\ge0$ at every interior node, then $\max w$ is attained on the boundary.

**A3 ★★** (von Neumann analysis for $u_t = \alpha^2u_{xx}$.) Substitute $w_j^n = g^ne^{ij\theta}$ and show that the amplification factors are
FTCS: $g = 1-4\lambda s$; BTCS: $g = \frac1{1+4\lambda s}$; Crank–Nicolson: $g = \frac{1-2\lambda s}{1+2\lambda s}$, where $s = \sin^2\frac\theta2$ and $\lambda = \alpha^2k/h^2$. Derive the stability conditions. Show that CN has $g\to-1$ as $\lambda s\to\infty$, and explain the consequence for non-smooth initial data.

**A4 ★★** For the explicit wave scheme, show that $g$ satisfies $g^2-2(1-2\lambda^2s)g+1 = 0$ and that $|g| = 1$ for all $\theta$ iff $\lambda = \alpha k/h\le1$. Interpret the CFL condition with domains of dependence.

**A5 ★★** *(Energy estimates.)* For $u_t = u_{xx}$ with $u = 0$ at $x = 0,1$, show $\frac{d}{dt}\int_0^1u^2dx = -2\int_0^1u_x^2dx\le0$. For $u_{tt} = c^2u_{xx}$, show that $E(t) = \frac12\int(u_t^2+c^2u_x^2)dx$ is conserved. Which of FTCS, BTCS and CN reproduce the discrete analogue of the first estimate unconditionally?

## B. Hand computation

**B1 ★** Solve $\nabla^2u = 4$ on the unit square with $u = x^2+y^2$ on the boundary, using $h = \frac13$ (four unknowns). Explain why the numerical solution is exact.

**B2 ★★** Solve Laplace's equation on the unit square with $u(x,1) = \sin\pi x$ and $u = 0$ on the other sides, using $h = \frac14$ (9 unknowns; use the symmetry $x\leftrightarrow1-x$ to reduce to 6). Compare with $u = \frac{\sinh\pi y\sin\pi x}{\sinh\pi}$, then refine to $h = \frac18,\frac1{16},\frac1{32}$.

**B3 ★★** Apply FTCS to $u_t = u_{xx}$, $u(x,0) = \sin\pi x$, with $h = 0.25$, for two steps with $k = 0.025$ and with $k = 0.05$. What is $\lambda$ in each case? With $\lambda = 0.8$, add a perturbation of size $10^{-3}$ in the highest mode and take 20 steps.

**B4 ★** Take one step of BTCS and one step of Crank–Nicolson with $h = 0.25$, $k = 0.05$ for B3. Compare the amplification of the mode $\sin\pi x$ with the exact factor $e^{-\pi^2k}$.

**B5 ★★** For $u_{tt} = 4u_{xx}$ on $[0,1]$ with $u(x,0) = \sin\pi x$, $u_t(x,0) = 0$, take two steps of the explicit scheme with $h = 0.25$, $k = 0.1$. Check the CFL condition and compare with $u = \sin\pi x\cos2\pi t$.

**B6 ★** Tabulate $g$ for the highest mode ($\theta = \pi$) for FTCS, BTCS and CN at $\lambda = 0.4, 0.6, 5$.

## C. Programming

**C1 ★★** Solve $\nabla^2u = (1-\pi^2)e^x\sin\pi y$ on $[0,2]\times[0,1]$ with $u = e^x\sin\pi y$ on the boundary, using $n\times m = 8\times4,\dots,128\times64$ grids. Confirm $O(h^2)$.

**C2 ★★★** Heat equation with $u(x,0) = \sin\pi x+\frac12\sin3\pi x$, at $t = 0.1$:
(a) with $k = 0.4h^2$, compare the errors of FTCS, BTCS and CN as $h$ decreases;
(b) with $h = \frac1{400}$ fixed and $k = 0.02,\dots,0.0025$, show the $O(k)$ and $O(k^2)$ time errors of BTCS and CN;
(c) with step-function initial data and $\lambda = 50$, compare CN with CN started by two pairs of half-step backward Euler steps (Rannacher smoothing).

**C3 ★★** Solve the 2-D heat equation $u_t = \nabla^2u$ on the unit square ($63\times63$ interior grid) with $u_0 = \sin\pi x\sin2\pi y$ up to $t = 0.05$. Use (i) the explicit scheme at its stability limit and (ii) CN with one sparse LU factorisation reused every step. Compare accuracy and cost.

**C4 ★★** Advect a Gaussian and a square pulse once around a periodic domain ($u_t+u_x = 0$, $h = 0.01$, CFL number $0.8$) with first-order upwind and Lax–Wendroff. Discuss numerical diffusion and dispersion (Gibbs-like overshoots).

## D. Data-science applications

**D1 ★★★** *Option pricing.* The Black–Scholes PDE $V_t+\frac12\sigma^2S^2V_{SS}+rSV_S-rV = 0$. For a put with $K = 100$, $r = 0.05$, $\sigma = 0.25$, $T = 1$ on $S\in[0,300]$: (a) price the European put with CN on three grids and compare the value, Delta and Gamma at $S = 100$ with the closed form; (b) price the American put by projecting onto the payoff after each step, compare with a 2000-step CRR binomial tree, and locate the early-exercise boundary.

**D2 ★★** *Image denoising.* Build a $128\times128$ piecewise-constant image, add noise with $\sigma = 0.2$, and compare linear heat diffusion with Perona–Malik diffusion ($g(d) = 1/(1+(d/K)^2)$, $K = 0.15$) for 5–40 explicit steps. Report PSNR and edge sharpness.

**D3 ★★** *Fokker–Planck equation.* The density of the Ornstein–Uhlenbeck process $dX = -\theta X\,dt+\sigma\,dW$ ($\theta = 1$, $\sigma = 0.8$) solves $p_t = (\theta xp)_x+\frac{\sigma^2}2p_{xx}$. Start from $N(2,0.1^2)$ and solve with a flux-conservative FD scheme and CN on $[-4,4]$ with zero-flux boundaries. Check mass conservation, and compare the mean and variance at $t = 0.5, 1, 3$ with theory and with Monte Carlo.
