# Chapter 12 — Worked Examples: Partial Differential Equations

Companion script: [`ch12_examples.py`](ch12_examples.py) reproduces every number below. · Lecture: [Chapter 12](../lectures/ch12-partial-differential-equations.md)

---

## Example 12.1 — A room with a heater (Poisson's equation with a source)

**Problem.** A square room $[0,1]^2$ has walls held at temperature 0 and a heater of strength 500 on $[0.2,0.4]^2$. Solve $-\nabla^2u = f$ on a $49\times49$ interior grid.

**Solution.** The 5-point scheme gives 2401 unknowns. A sparse direct solve takes 0.009 s. Conjugate gradients converges in 176 iterations, also in about 0.009 s, and the two agree to $2\times10^{-11}$. The hottest point is the heater centre $(0.30, 0.30)$ with $u = 4.87$. The room centre has $u = 1.68$, and the far corner region at $(0.8,0.8)$ has $u = 0.22$.

**Take-away.** Elliptic problems give large, sparse SPD systems. At this size both direct and iterative solvers are instant. In 3-D, or with $10^6$ unknowns, preconditioned CG or multigrid is necessary.

---

## Example 12.2 — When has the rod cooled?

**Problem.** A rod with $u_t = u_{xx}$, $u = 0$ at both ends and $u(x,0) = \sin\pi x+0.3\sin3\pi x$ is cooling. When does the centre temperature fall below 10% of its initial value? The exact answer, from the Fourier solution by root finding, is $t^\ast = 0.26944$.

**Solution.** Integrate until the event occurs, then interpolate linearly within the last step:

| method | $k$ | $\lambda = k/h^2$ | $t^\ast$ |
|---|---|---|---|
| FTCS | 0.001 | 0.40 | 0.26866 |
| FTCS | 0.0012 | 0.48 | 0.26840 |
| CN | 0.01 | 4.0 | 0.26979 |
| CN | 0.002 | 0.8 | 0.26999 |

**Take-away.** Each estimate is within about 0.3% of the exact value. With $h = 1/20$ the *spatial* error, not the time step, dominates, so refining $k$ alone does not help: balance $k$ and $h$. Crank–Nicolson reaches this accuracy with 5–10 times fewer steps than FTCS, which is capped at $\lambda\le\frac12$. Event detection, finding when a threshold is crossed, is a root-finding problem in time.

---

## Example 12.3 — Advection–diffusion of a pollutant

**Problem.** A pollutant pulse at $x = 0.2$ is carried by a river flowing at speed $v = 1$ and diffuses with $D = 0.002$. Simulate to $t = 0.5$ with $h = 0.01$ and $k = 0.4h$, using central and upwind differences for the advection term.

**Solution.** The exact solution is a Gaussian centred at 0.7 with peak 0.620. The cell Péclet number is $\frac{vh}{2D} = 2.5>1$.

| scheme | peak | position | minimum | max error |
|---|---|---|---|---|
| central | 0.938 | 0.69 | $-0.068$ | 0.34 |
| upwind | 0.447 | 0.70 | 0.000 | 0.17 |

**Take-away.** With cell Péclet number above 1, central differences produce negative concentrations and a spurious growth of the peak; at this step size the scheme is also at its stability limit. Upwinding is monotone, but it adds numerical diffusion $\frac{vh}2 = 0.005$, more than twice the physical $D$, so the pulse is smeared out. Resolving the physics requires $h<2D/v = 0.004$, or higher-order upwind (TVD) schemes.

---

## Example 12.4 — The sound of a plucked string

**Problem.** Pluck a string ($c = 1$, $L = 1$) at $x = 0.2$ and record the displacement at $x = 0.9$ (the explicit scheme with $\lambda = 1$, $h = 0.005$, 4000 steps). Which frequencies are present?

**Solution.** The FFT of the recorded signal (Hann window) shows peaks at $0.5, 1, 1.5, 2, (2.5), 3$, i.e. $nc/2L$. The **fifth harmonic is missing**: $\sin(5\pi\cdot0.2) = 0$, because the string was plucked at a node of that mode. The measured relative amplitudes, $1, 0.769, 0.471, 0.192, 0, 0.086$, match the theory $\frac{|\sin(n\pi\,0.2)\sin(n\pi\,0.9)|}{n^2}$ (normalised) to three digits.

**Take-away.** With $\lambda = 1$ the explicit scheme is exact for the 1-D wave equation, and combining it with the FFT (Chapter 8) turns a PDE solver into a spectral analyser. This is why the timbre of a guitar depends on where it is plucked.

---

## Example 12.5 — Black–Scholes in log-price coordinates

**Problem.** Price a European call ($K = 100$, $r = 0.03$, $\sigma = 0.3$, $T = 0.5$). Use $x = \ln S$, which turns the PDE into a constant-coefficient convection–diffusion equation $V_\tau = \frac{\sigma^2}2V_{xx}+(r-\frac{\sigma^2}2)V_x-rV$, and solve with Crank–Nicolson on $x\in[\ln K-3, \ln K+3]$.

**Solution.**

| $M$ (space) | $V(S = 100)$ | error |
|---|---|---|
| 100 | 9.065527 | $8.4\times10^{-2}$ |
| 200 | 9.128680 | $2.1\times10^{-2}$ |
| 400 | 9.144234 | $5.2\times10^{-3}$ |

The exact value is $9.149399$. The error drops by 4 each time $M$ doubles (with $N = M/2$ time steps): second order.

**Take-away.** The logarithmic grid is uniform in *returns*, so it puts points densely near the strike and sparsely at large $S$, and the constant coefficients simplify the stability analysis. Choosing good coordinates is half of numerical PDE work.

---

## Example 12.6 — Heat diffusion on a graph

**Problem.** Place a unit of "heat" (e.g. a label or a rumour) on node 0 of a 9-node graph made of three triangles connected in a chain. Compute $\mathbf f(t) = e^{-tL}\mathbf f_0$ with the graph Laplacian $L$, and compare with implicit Euler (20 steps).

**Solution.**

| $t$ | heat on nodes 0–8 |
|---|---|
| 0.5 | $0.474, 0.251, 0.214, 0.044, 0.008, 0.007, 0.001, 0, 0$ |
| 2 | $0.257, 0.255, 0.215, 0.109, 0.069, 0.058, 0.020, 0.009, 0.009$ |
| 10 | $0.134, 0.134, 0.130, 0.116, 0.111, 0.106, 0.093, 0.088, 0.088$ |

The total heat stays 1 (conservation) and tends to $1/9$ at every node. Implicit Euler agrees to 3–8$\times10^{-3}$.

**Take-away.** The graph heat equation $\mathbf f' = -L\mathbf f$ is the discrete analogue of $u_t = u_{xx}$. It spreads information first within a cluster (the first triangle) and then across bottlenecks. Heat kernels define node similarities and underlie diffusion-based semi-supervised learning and graph neural networks.
