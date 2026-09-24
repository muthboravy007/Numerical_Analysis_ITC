# Chapter 11 — Worked Examples: Boundary-Value Problems

Companion script: [`ch11_examples.py`](ch11_examples.py) reproduces every number below. · Lecture: [Chapter 11](../lectures/ch11-boundary-value-problems.md)

---

## Example 11.1 — The shape of a hanging cable

**Problem.** A cable hangs between $(-1,3)$ and $(1,3)$. Its shape satisfies $y'' = \frac1a\sqrt{1+(y')^2}$ with $a = 2$. Find the sag by nonlinear shooting, and compare with the catenary $y = a\cosh(x/a)+C$.

**Solution.** Newton's method on the initial slope $t = y'(-1)$ (RK4, 40 steps), using the variational equation $z'' = f_{y'}z'$, generates $t = 0, -0.4621, -0.52037, -0.521095, -0.521095$. The exact slope is $-\sinh(1/2) = -0.521095$. The sag at the centre is $0.2552519$ (exact $2\cosh\frac12-2 = 0.2552519$), and the maximum error is $1.6\times10^{-10}$.

**Take-away.** Nonlinear shooting converges quadratically when the IVP is well behaved. The problem is symmetric, so the shooting slope must equal minus the slope at the far end, which is a good sanity check.

---

## Example 11.2 — Diffusion and reaction in a catalyst pellet

**Problem.** The reactant concentration in a slab pellet satisfies $y'' = \phi^2y$, $y'(0) = 0$ (symmetry) and $y(1) = 1$. The *effectiveness factor* $\eta = y'(1)/\phi^2 = \tanh\phi/\phi$ measures how much of the pellet is used. Solve by finite differences, handling the Neumann condition with a ghost point $y_{-1} = y_1$.

**Solution.** The ghost point turns the first equation into $(2+h^2\phi^2)y_0-2y_1 = 0$, and the matrix stays tridiagonal. Estimate $y'(1)$ with the 3-point one-sided formula.

| $\phi$ | $h$ | $y(0)$ | exact | $\eta$ | exact |
|---|---|---|---|---|---|
| 1 | 1/10 | 0.648260 | 0.648054 | 0.75881 | 0.761594 |
| 1 | 1/40 | 0.648067 | | 0.76141 | |
| 5 | 1/10 | 0.014176 | 0.013475 | 0.18662 | 0.199982 |
| 5 | 1/40 | 0.013519 | | 0.19890 | |

**Take-away.** Ghost points keep second-order accuracy for derivative boundary conditions. When the reaction is fast ($\phi = 5$) the solution has a boundary layer, only 20% of the pellet is effective, and a finer grid is needed near $x = 1$.

---

## Example 11.3 — Optimal control is a boundary-value problem

**Problem.** Steer $x' = u$ from $x(0) = 1$ so as to minimise $J = \int_0^2(x^2+u^2)\,dt$, with the final state free.

**Solution.** The Euler–Lagrange (Pontryagin) conditions give $x'' = x$, $x(0) = 1$ and $x'(2) = 0$ (transversality), with solution $x^\ast = \frac{\cosh(2-t)}{\cosh2}$. Finite differences with $N = 200$ and a ghost point at $t = 2$ reproduce it to $2\times10^{-6}$. The optimal cost is $0.964025$ (exact $\tanh2 = 0.964028$), and the initial control is $u(0)\approx-0.96$.

| strategy | cost |
|---|---|
| do nothing ($u = 0$) | 2.0 |
| linear decay to 0 | 1.167 |
| optimal | **0.964** |

**Take-away.** Initial-value problems run forward in time. Optimal-control, and more generally planning, problems couple an initial condition with a terminal condition, which makes them BVPs. The LQR controllers used in robotics and economics come from this structure.

---

## Example 11.4 — Heat through a composite wall: averaging coefficients

**Problem.** A wall has conductivity $k = 1$ on $[0,0.5)$ and $k = 10$ on $(0.5,1]$, with $T(0) = 100$ and $T(1) = 0$. We know $k$ only at the grid nodes, and the interface lies midway between two nodes. The flux scheme $-(k_{i+1/2}(T_{i+1}-T_i)-k_{i-1/2}(T_i-T_{i-1})) = 0$ needs $k$ at the half points. Compare the arithmetic and harmonic means of neighbouring node values.

**Solution.** The exact heat flux is $\frac{100}{0.5/1+0.5/10} = 181.82$ (series resistances).

| $N$ | arithmetic mean: max error | flux | harmonic mean: max error | flux |
|---|---|---|---|---|
| 10 | 5.4 | 193.6 | $2\times10^{-14}$ | 181.82 |
| 20 | 2.9 | 187.8 | $2\times10^{-14}$ | 181.82 |
| 40 | 1.5 | 184.8 | $2\times10^{-14}$ | 181.82 |

**Take-away.** Across a jump, conductances add like resistors in series, so the harmonic mean is exact, while the arithmetic mean gives an $O(h)$ error that overestimates the flux. The same rule applies to porous-media flow, diffusion in heterogeneous tissue, and graph Laplacians with edge weights.

---

## Example 11.5 — Eigenvalues of a vibrating string

**Problem.** The frequencies of a string with fixed ends come from $-y'' = \lambda y$, $y(0) = y(1) = 0$, with exact eigenvalues $\lambda_k = (k\pi)^2$. Approximate them with the FD matrix $\frac1{h^2}\operatorname{tridiag}(-1,2,-1)$.

**Solution.**

| $N$ | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ | relative error of $\lambda_1$ | relative error of the highest mode |
|---|---|---|---|---|---|
| 10 | 9.8027 | 38.417 | 83.524 | $6.8\times10^{-3}$ | 0.52 |
| 20 | 9.8512 | 39.185 | 87.346 | $1.9\times10^{-3}$ | 0.56 |
| 40 | 9.8648 | 39.401 | 88.436 | $4.9\times10^{-4}$ | 0.58 |
| 80 | 9.8684 | 39.459 | 88.726 | $1.3\times10^{-4}$ | 0.59 |

The exact values are $9.8696, 39.478, 88.826$.

**Take-away.** Low modes converge at $O(h^2)$, but the highest discrete modes are always about 50% wrong, because they are not resolved by the grid. Only trust the lowest part of a discrete spectrum. The same matrix, as a graph Laplacian of a path, gives the frequencies used in spectral clustering and graph signal processing.
