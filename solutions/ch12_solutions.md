# Chapter 12 — Solutions

> Exercises: [`exercises/ch12_exercises.md`](../exercises/ch12_exercises.md) · Numbers from [`solutions/code/ch12_solutions.py`](code/ch12_solutions.py).

## A. Theory and proofs

**A1.** (a) $B^2-4AC = 16-12 = 4>0$: hyperbolic. The characteristics satisfy $A\,dy^2-B\,dx\,dy+C\,dx^2 = 0$, i.e. $\frac{dy}{dx} = \frac{B\pm\sqrt{B^2-4AC}}{2A} = 3$ or $1$. So they are the lines $y-3x = $ const and $y-x = $ const. (b) $4-4 = 0$: parabolic. (c) $1-4 = -3<0$: elliptic. (d) $B^2-4AC = -4y$: elliptic for $y>0$, parabolic on $y = 0$ and hyperbolic for $y<0$ (the model of transonic flow).

**A2.** (a) $u(x\pm h,y) = u\pm hu_x+\frac{h^2}2u_{xx}\pm\frac{h^3}6u_{xxx}+\frac{h^4}{24}u_{xxxx}+\dots$. Adding the two expansions and dividing by $h^2$ gives $u_{xx}+\frac{h^2}{12}u_{xxxx}+O(h^4)$, and similarly in $y$. The truncation error vanishes whenever $u_{xxxx} = u_{yyyy} = 0$, e.g. for all cubics (and for $x^2y^2$). (b) If the maximum $M$ were attained at an interior node $P$ and not on the boundary, then $4w_P\le\sum_{\text{nbrs}}w\le4M = 4w_P$. So all four neighbours equal $M$. Propagating this to the boundary gives a contradiction.

**A3.** Substituting $w_j^n = g^ne^{ij\theta}$ into FTCS gives $g = 1+\lambda(e^{i\theta}-2+e^{-i\theta}) = 1-4\lambda s$. Stability requires $|g|\le1$ for all $s\in[0,1]$, i.e. $\lambda\le\frac12$. BTCS gives $g(1+4\lambda s) = 1$, so $0<g\le1$ always. For CN, $g(1+2\lambda s) = 1-2\lambda s$, so $|g|\le1$ always. As $\lambda s\to\infty$, $g_{CN}\to-1$: high-frequency components are *not* damped, and they flip sign every step. With a step or kink in the data the error persists as slowly decaying oscillations (C2(c)). BTCS ($g\to0$) is L-stable and smooths them immediately.

**A4.** Substituting $w_j^n = g^ne^{ij\theta}$ into $w^{n+1} = 2(1-\lambda^2)w^n+\lambda^2(w_{j+1}^n+w_{j-1}^n)-w^{n-1}$ gives $g^2-2(1-2\lambda^2s)g+1 = 0$. The product of the roots is $1$. If $|1-2\lambda^2s|\le1$ the roots are complex conjugates with $|g| = 1$. Otherwise they are real and one has $|g|>1$. So stability requires $2\lambda^2s\le2$ for $s\le1$, i.e. $\lambda\le1$. Domain of dependence: $u(x,t)$ depends on the data in $[x-\alpha t,x+\alpha t]$, while the explicit scheme's numerical domain is $[x-\frac hkt,x+\frac hkt]$. If $\alpha k>h$, the numerical domain misses part of the true one, so the scheme cannot converge.

**A5.** $\frac{d}{dt}\int u^2 = 2\int uu_{xx} = [2uu_x]_0^1-2\int u_x^2 = -2\int u_x^2$. For the wave equation, $\frac{dE}{dt} = \int(u_tu_{tt}+c^2u_xu_{xt}) = \int u_t(u_{tt}-c^2u_{xx})+[c^2u_xu_t]_0^1 = 0$. BTCS and CN give $\|w^{n+1}\|_2\le\|w^n\|_2$ for every $\lambda$, since their iteration matrices are symmetric with eigenvalues $|g|\le1$. FTCS does so only for $\lambda\le\frac12$.

## B. Hand computation

**B1.** The unknowns are $w_{11}, w_{21}, w_{12}, w_{22}$, and the result is $(0.222222, 0.555556, 0.555556, 0.888889)$, the exact values $x^2+y^2$ at $(\frac13,\frac13)$, $(\frac23,\frac13)$, $(\frac13,\frac23)$ and $(\frac23,\frac23)$. The 5-point formula is exact for quadratics (A2), so the discrete equations are satisfied by the exact solution. By uniqueness, the discrete solution *is* the exact one at the nodes.

**B2.** With $h = \frac14$ the values at $x = 0.5$ are $0.082524, 0.213388, 0.469253$ for $y = 0.25, 0.5, 0.75$ (exact $0.075218, 0.199268, 0.452688$). The values at $x = 0.25$ and $x = 0.75$ are equal, which is the symmetry.

| $h$ | max error | $u(0.5,0.5)$ |
|---|---|---|
| 1/4 | $1.66\times10^{-2}$ | 0.213388 |
| 1/8 | $4.32\times10^{-3}$ | 0.202915 |
| 1/16 | $1.11\times10^{-3}$ | 0.200188 |
| 1/32 | $2.78\times10^{-4}$ | 0.199499 |

The exact value is $u(0.5,0.5) = 0.199268$. The error ratio is about 3.9–4, i.e. $O(h^2)$.

**B3.**
- $k = 0.025$ ($\lambda = 0.4$): after two steps $(0.414558, 0.586274, 0.414558)$, against the exact $(0.431687, 0.610498, 0.431687)$.
- $k = 0.05$ ($\lambda = 0.8>\frac12$): after two steps $(0.199655, 0.282355, 0.199655)$, against the exact $(0.263544, 0.372708, 0.263544)$. Twenty steps still look harmless, $\approx(2,3,2)\times10^{-6}$, because the data contain only the mode $\sin\pi x$, whose factor $1-4(0.8)\sin^2\frac\pi8 = 0.531$ is stable.
- Adding a $10^{-3}$ perturbation in the highest mode ($g = 1-4(0.8)\sin^2\frac{3\pi}8 = -1.73$) gives values of $(50, -71, 50)$ after 20 steps. In practice round-off plays the role of the perturbation.

**B4.** After one step, BTCS gives $(0.481474, 0.680907, 0.481474)$ and CN gives $(0.438641, 0.620332, 0.438641)$; the exact values are $(0.431687, 0.610498, 0.431687)$. The amplification factors of the mode $\sin\pi x$, with $s = \sin^2\frac\pi8$ and $\lambda = 0.8$, are:

| method | factor |
|---|---|
| FTCS | 0.5314 |
| BTCS | 0.6809 |
| CN | 0.6203 |
| exact $e^{-\pi^2k}$ | 0.6105 |

CN is the closest ($O(k^2)$).

**B5.** $\lambda = \frac{2(0.1)}{0.25} = 0.8\le1$, so the CFL condition holds.

| $t$ | numerical | exact |
|---|---|---|
| 0.1 | $(0.574558, 0.812548, 0.574558)$ | $(0.572061, 0.809017, 0.572061)$ |
| 0.2 | $(0.226606, 0.320470, 0.226606)$ | $(0.218508, 0.309017, 0.218508)$ |

**B6.**

| $\lambda$ | FTCS | BTCS | CN |
|---|---|---|---|
| 0.4 | $-0.600$ | 0.385 | 0.111 |
| 0.6 | $-1.400$ (unstable) | 0.294 | $-0.091$ |
| 5 | $-19.0$ | 0.048 | $-0.818$ (weakly damped, oscillating) |

## C. Programming (key results)

**C1.** The maximum errors are $0.169$, $0.0424$, $0.0106$, $2.66\times10^{-3}$ and $6.64\times10^{-4}$ for 21 to 8001 unknowns. The ratio is 3.99–4.00, i.e. $O(h^2+k^2)$, with the $h\ne k$ grid handled by the parameter $(h/k)^2$.

**C2.**

(a) $k = 0.4h^2$:

| $m$ | FTCS | BTCS | CN |
|---|---|---|---|
| 10 | $4.3\times10^{-3}$ | $9.8\times10^{-3}$ | $2.9\times10^{-3}$ |
| 20 | $1.05\times10^{-3}$ | $2.5\times10^{-3}$ | $7.4\times10^{-4}$ |
| 40 | $2.6\times10^{-4}$ | $6.3\times10^{-4}$ | $1.9\times10^{-4}$ |
| 80 | $6.5\times10^{-5}$ | $1.6\times10^{-4}$ | $4.7\times10^{-5}$ |

All three are $O(h^2)$ when $k\propto h^2$. FTCS beats BTCS because its time error partly cancels the space error.

(b) $h = \frac1{400}$:

| $k$ | $\lambda$ | BTCS | CN |
|---|---|---|---|
| 0.02 | 3200 | $3.1\times10^{-2}$ | $1.1\times10^{-3}$ |
| 0.01 | 1600 | $1.7\times10^{-2}$ | $2.6\times10^{-4}$ |
| 0.005 | 800 | $8.6\times10^{-3}$ | $6.4\times10^{-5}$ |
| 0.0025 | 400 | $4.4\times10^{-3}$ | $1.5\times10^{-5}$ |

The ratios are 2 for BTCS ($O(k)$) and 4 for CN ($O(k^2)$). Both are stable at $\lambda$ in the thousands.

(c) Step-function data, $\lambda = 50$. Maximum errors after 1, 2 and 5 steps:

| | 1 step | 2 steps | 5 steps |
|---|---|---|---|
| plain CN | 0.44 | 0.40 | 0.34 |
| Rannacher start | 0.051 | 0.031 | 0.010 |

Plain CN barely decays because of the $g\approx-0.96$ modes at the jumps. Rannacher smoothing damps the high frequencies first and then keeps second order. This is standard for option payoffs with kinks.

**C3.**

| method | steps | max error | time |
|---|---|---|---|
| explicit ($k = 6.1\times10^{-5}\le h^2/4$) | 820 | $1.7\times10^{-4}$ | 0.029 s |
| CN, $k = 0.005$ | 10 | $9.2\times10^{-4}$ | 0.021 s |
| CN, $k = 0.001$ | 50 | $1.0\times10^{-4}$ | 0.053 s |

The CN runs include the factorisation (sparse LU fill $2.1\times10^5$ nonzeros, factored once). At this size explicit and implicit cost about the same. As $h\to0$ the explicit step count grows like $h^{-2}$, whereas CN can take $k\propto h$. So implicit methods win for fine grids, and even more in 3-D, where ADI or multigrid replace the sparse LU.

**C4.**

| pulse | upwind: error / peak / min | Lax–Wendroff: error / peak / min |
|---|---|---|
| Gaussian | 0.255 / 0.745 / 0.000 | 0.064 / 0.978 / $-0.005$ |
| square | 0.464 / 1.000 / 0.000 | 0.560 / 1.174 / $-0.174$ |

Upwind is monotone but diffusive, with effective viscosity $\frac h2(1-\nu)$: the peak drops 25%. Lax–Wendroff is second order and excellent for smooth data. At discontinuities it is dispersive, producing 17% over- and undershoots (Godunov's theorem: linear monotone schemes are at most first order). Flux limiters (TVD schemes) combine the two behaviours.

## D. Data-science applications

**D1.** (a) European put, exact value $7.45894$, $\Delta = -0.37259$, $\Gamma = 0.015137$:

| grid $M\times N$ | $V(100)$ | $\Delta$ | $\Gamma$ |
|---|---|---|---|
| $150\times50$ | 7.45130 | $-0.37276$ | 0.015150 |
| $300\times100$ | 7.45703 | $-0.37263$ | 0.015140 |
| $600\times200$ | 7.45846 | $-0.37260$ | 0.015138 |

The price error falls $7.6\to1.9\to0.48\times10^{-3}$, i.e. $O(h^2+k^2)$. The Greeks come for free from the grid. (b) The American put is $7.97104$ by CN with projection and $7.97396$ by CRR with 2000 steps. The early-exercise premium is $0.512$, and the exercise boundary at $t = 0$ is at $S^\ast\approx76$. Projection is a first-order-in-time approximation of the linear complementarity problem; PSOR or penalty methods do better.

**D2.** The noisy image has PSNR 14.0 dB.

| steps ($\Delta t = 0.2$) | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| heat PSNR (dB) | 21.1 | 20.2 | 18.9 | 17.4 |
| Perona–Malik PSNR (dB) | 19.9 | 25.6 | **29.1** | 27.2 |

The largest one-pixel jump across a true unit edge after 20 steps is $0.14$ for heat and $0.85$ for Perona–Malik. Linear diffusion is Gaussian blurring: it removes noise and edges alike, so more steps are soon worse. Perona–Malik's conductivity $g$ is small where $|\nabla u|$ is large, so it diffuses inside regions and not across edges. It is a nonlinear PDE and the basis of edge-preserving smoothing.

**D3.** Mass is conserved to $10^{-6}$, since the flux form is exactly conservative. The mean and variance are:

| $t$ | mean (FD) | mean (theory) | variance (FD) | variance (theory) |
|---|---|---|---|---|
| 0.5 | 1.2131 | 1.2131 | 0.2059 | 0.2060 |
| 1 | 0.7358 | 0.7358 | 0.2779 | 0.2780 |
| 3 | 0.0996 | 0.0996 | 0.3191 | 0.3192 |

The theory values are mean $= 2e^{-\theta t}$ and variance $= 0.01e^{-2\theta t}+\frac{\sigma^2}{2\theta}(1-e^{-2\theta t})$. Monte Carlo at $t = 3$ gives mean $0.0998$ and variance $0.3205$. The stationary distribution is $N(0, 0.32)$. The PDE and the SDE describe the same process: the PDE gives the whole density deterministically in 1-D, while simulation scales to high dimensions.
