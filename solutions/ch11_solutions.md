# Chapter 11 — Solutions

> Exercises: [`exercises/ch11_exercises.md`](../exercises/ch11_exercises.md) · Numbers from [`solutions/code/ch11_solutions.py`](code/ch11_solutions.py).

## A. Theory and proofs

**A1.** Suppose $u$ has a positive maximum at an interior point $x^\ast$. Then $u'(x^\ast) = 0$ and $u''(x^\ast)\le0$. But the equation gives $u''(x^\ast) = q(x^\ast)u(x^\ast)>0$, a contradiction. A negative minimum is excluded in the same way. So $\max u\le0\le\min u$, i.e. $u\equiv0$. If $y$ and $\tilde y$ both solve the BVP, then $u = y-\tilde y$ solves the homogeneous problem with zero boundary values, so $u\equiv0$.

**A2.** (a) Taylor expansion gives $\frac{y(x+h)-2y(x)+y(x-h)}{h^2} = y''+\frac{h^2}{12}y^{(4)}(\xi)$ and $\frac{y(x+h)-y(x-h)}{2h} = y'+\frac{h^2}6y'''(\eta)$. Substituting into $y''-py'-qy-r = 0$ leaves $\tau = \frac{h^2}{12}(y^{(4)}-2py''')$. (b) Row $i$ of the matrix is $(-1-\frac h2p_i,\ 2+h^2q_i,\ -1+\frac h2p_i)$. If $\frac h2|p_i|<1$, both off-diagonal entries are negative and their absolute values sum to exactly $2\le2+h^2q_i$. The first and last rows drop one neighbour, so they are strictly diagonally dominant. The matrix is irreducible (tridiagonal with nonzero off-diagonals), and an irreducibly diagonally dominant matrix is nonsingular.

**A3.** $y = y_1+cy_2$ satisfies the ODE by linearity (the $r$ term enters only through $y_1$). It satisfies $y(a) = \alpha$, and $y(b) = y_1(b)+cy_2(b) = \beta$ by the choice of $c$. If $y_2(b) = 0$, then $y_2$ would solve the homogeneous BVP with zero boundary values and hence vanish by A1, contradicting $y_2'(a) = 1$.

**A4.** (a) For $v\in H_0^1$, $I[y+\epsilon v] = I[y]+2\epsilon\bigl[\int(py'v'+qyv-fv)\bigr]+\epsilon^2\int(pv'^2+qv^2)$. Integrating by parts, the bracket is $\int(-(py')'+qy-f)v = 0$. The $\epsilon^2$ term is $\ge0$, so $y$ minimises $I$. (b) Both the exact and the Galerkin solutions satisfy $a(\cdot,v) = (f,v)$ for $v\in V_h$. Subtracting gives $a(y-u_h,v) = 0$. (c) For any $v_h\in V_h$, $\|y-v_h\|_a^2 = \|y-u_h\|_a^2+\|u_h-v_h\|_a^2+2a(y-u_h,u_h-v_h)$. The cross term vanishes by (b), so $\|y-u_h\|_a\le\|y-v_h\|_a$.

**A5.** $y_2'' = k^2y_2$ with $y_2(0) = 0$, $y_2'(0) = 1$ gives $y_2 = \sinh(kx)/k$. The shooting coefficient is $c = (e^{-k}-y_1(1))/y_2(1)$, where $y_1(1) = \cosh k$. Both $y_1(1)$ and $c\,y_2(1)$ are about $e^k/2$, but their sum must equal the tiny number $e^{-k}$. The cancellation loses about $\log_{10}(e^k/e^{-k}) = 2k\log_{10}e$ digits: 8.7 for $k = 10$, 17 for $k = 20$ (everything), and 35 for $k = 40$. This is intrinsic to single shooting, not to RK4. Finite differences solve a well-conditioned global system.

## B. Hand computation

**B1.** $c = \frac{2-\cosh2}{\sinh2/2} = -0.971747$, so $y = \cosh2x-0.971747\cdot\frac12\sinh2x$ and $y'(0) = c = -0.971747$. With RK4 and $h = 0.25$, the values at $x = 0.25, 0.5, 0.75$ are $0.87471, 0.97243, 1.31813$, against the exact $0.87444, 0.97208, 1.31785$. The maximum error is $3.5\times10^{-4}$.

**B2.** The equations are $-w_{i-1}+(2-h^2)w_i-w_{i+1} = -h^2x_i$ with $2-h^2 = 1.9375$:
$\begin{pmatrix}1.9375&-1&0\\-1&1.9375&-1\\0&-1&1.9375\end{pmatrix}\mathbf w = \begin{pmatrix}-0.015625\\-0.03125\\-0.046875+2\end{pmatrix}$.
The solution is $\mathbf w = (0.544274, 1.070156, 1.560403)$, against the exact $(0.544014, 1.069747, 1.560056)$, with maximum error $4.09\times10^{-4}$. For $h = 1/8, 1/16, 1/32$ the maximum errors are $1.02\times10^{-4}$, $2.58\times10^{-5}$ and $6.45\times10^{-6}$. Each halving of $h$ divides the error by 4.

**B3.** Here $p = -2$, so the subdiagonal is $-1-\frac h2p = -0.75$ and the superdiagonal is $-1+\frac h2p = -1.25$. The solution is $\mathbf w = (0.459559, 0.735294, 0.900735)$, against the exact $(0.455054, 0.731059, 0.898464)$. The maximum error is $4.5\times10^{-3}$, which is larger than in B2 because of the $p\,y'''$ term.

**B4.** The exact solution $\frac{e^{50x}-1}{e^{50}-1}$ is almost $0$ except in a layer of width about $0.02$ at $x = 1$.

| $h$ | $h\lvert p\rvert/2$ | centred | upwind |
|---|---|---|---|
| 0.1 | 2.5 | oscillates: last nodes $0.034, -0.079, 0.184, -0.429$; 8 sign changes; max error $0.44$ | monotone: $\dots, 0.0046, 0.028, 0.167$; max error $0.16$ |
| 0.02 | 0.5 | monotone: $0.012, 0.037, 0.111, 0.333$ (exact $0.018, 0.050, 0.135, 0.368$); max error $0.035$ | monotone; max error $0.13$ |

With $h|p|/2>1$ the centred scheme's off-diagonal entries have mixed signs, the maximum principle fails, and the solution oscillates. Upwinding restores monotonicity at any $h$ but is only $O(h)$ accurate, because it adds artificial diffusion $\frac{hp}{2}y''$. With a resolved layer the centred scheme is more accurate.

**B5.** Integrate $y'' = -(y')^2$ with $y'(0) = t$: $y' = \frac{t}{1+tx}$, so $y(1;t) = \ln(1+t)$ and $z(1;t) = \frac1{1+t}$. From $t_0 = \ln2 = 0.693147$: $y(1) = 0.526589$, $z(1) = 0.590616$, and $t_1 = t_0-\frac{0.526589-0.693147}{0.590616} = 0.975155$. The numerical Newton iterates are $0.693147, 0.975156, 0.999848, 1.000003$, converging to the exact slope $1$ up to the RK4 error ($h = 0.1$). The maximum error is $5.4\times10^{-7}$.

**B6.** $\mathbf w^{(0)} = \ln2\cdot(0.25, 0.5, 0.75) = (0.173287, 0.346574, 0.519860)$. Every centred slope equals $\ln2$, so $F_i = -h^2(\ln2)^2 = -0.030028$ for all $i$. The Jacobian is
$J = \begin{pmatrix}2&-1.173287&0\\-0.826713&2&-1.173287\\0&-0.826713&2\end{pmatrix}$,
with off-diagonal entries $-1\mp\frac h2f_{y'} = -1\pm\frac h2\cdot2y'$. One Newton step gives $\mathbf w^{(1)} = (0.222506, 0.404879, 0.558976)$. After 4 iterations $\mathbf w = (0.223457, 0.405755, 0.559775)$, against the exact $(0.223144, 0.405465, 0.559616)$, with error $3.1\times10^{-4}$. For $h = 1/8, 1/16, 1/32$ the maximum errors are $8.1\times10^{-5}$, $2.0\times10^{-5}$ and $5.1\times10^{-6}$, i.e. $O(h^2)$.

**B7.** The system is $K = \begin{pmatrix}8.166667&-3.958333&0\\-3.958333&8.166667&-3.958333\\0&-3.958333&8.166667\end{pmatrix}$, $\mathbf b = (0.25,0.25,0.25)$. Its solution is $\mathbf c = (0.085731, 0.113719, 0.085731)$, against the exact $(0.085323, 0.113181, 0.085323)$. The nodal errors are $5.4\times10^{-4}$, $1.3\times10^{-4}$ and $3.3\times10^{-5}$ for $n = 3, 7, 15$, i.e. $O(h^2)$. Nodal values are not exact here (unlike $-y'' = f$) because of the mass term.

## C. Programming (key results)

**C1.**

| $h$ | shooting (RK4) | finite differences |
|---|---|---|
| 1/4 | $3.0\times10^{-6}$ | $4.1\times10^{-4}$ |
| 1/8 | $2.6\times10^{-7}$ | $1.0\times10^{-4}$ |
| 1/16 | $1.8\times10^{-8}$ | $2.6\times10^{-5}$ |
| 1/32 | $1.2\times10^{-9}$ | $6.5\times10^{-6}$ |
| 1/64 | $7.7\times10^{-11}$ | $1.6\times10^{-6}$ |

The shooting error falls by about 16 per halving ($O(h^4)$) and the FD error by 4 ($O(h^2)$). Richardson extrapolation at $x = 0.5$: the FD values $1.070156, 1.069849, 1.069772$ have errors $4\times10^{-4}$ down to $2.5\times10^{-5}$. The first extrapolation level has errors $9\times10^{-7}$ and $6\times10^{-8}$, and the second level has error $6\times10^{-10}$. Extrapolation works because the FD error has an expansion in even powers of $h$.

**C2.** (a) $\lambda(\theta) = \frac{\theta^2}{2\cosh^2(\theta/4)}$ has its maximum $\lambda_c = 3.513830719$ at $\theta = 4.7987$, where $\max y = 1.1868$. For $\lambda>\lambda_c$ there is no solution. (b)

| $\lambda$ | lower $\max y$ (exact / FD) | Newton iterations | upper $\max y$ (exact / FD) | Newton iterations |
|---|---|---|---|---|
| 1 | 0.140539 / 0.140540 | 4 | 4.091467 / 4.091459 | 9 |
| 2 | 0.328952 / 0.328955 | 5 | 2.895531 / 2.895504 | 10 |
| 3 | 0.640147 / 0.640159 | 6 | 1.975267 / 1.975222 | 11 |
| 3.5 | 1.085159 / 1.085314 | 8 | 1.294585 / 1.294394 | 15 |

As $\lambda\to\lambda_c$ the two branches merge at a fold. The Jacobian becomes nearly singular there, so Newton slows down and the basins shrink. Beyond the fold, pseudo-arclength continuation is needed to follow the branch.

**C3.**

| $h$ | nodal max | $L^2$ | $H^1$-seminorm |
|---|---|---|---|
| 1/8 | $2.42\times10^{-3}$ | $8.68\times10^{-3}$ | $0.251$ |
| 1/16 | $6.02\times10^{-4}$ | $2.17\times10^{-3}$ | $0.126$ |
| 1/32 | $1.50\times10^{-4}$ | $5.43\times10^{-4}$ | $0.063$ |
| 1/128 | $9.4\times10^{-6}$ | $3.4\times10^{-5}$ | $0.0157$ |

The rates are $2.00$ (nodal), $2.00$ ($L^2$) and $1.00$ ($H^1$), exactly as theory predicts for P1 elements: the energy error is $O(h)$, and the Aubin–Nitsche duality argument gives $O(h^2)$ in $L^2$.

**C4.**

| $k$ | $y_2(1)$ | shooting max error | shooting relative error at $x = 1$ | FD max error |
|---|---|---|---|---|
| 10 | $1.1\times10^3$ | $3\times10^{-11}$ | $1.4\times10^{-8}$ | $1.5\times10^{-6}$ |
| 20 | $1.2\times10^7$ | $6\times10^{-8}$ | **1.0** (100%) | $6.1\times10^{-6}$ |
| 40 | $2.9\times10^{15}$ | **32** | $4\times10^{18}$ | $2.5\times10^{-5}$ |

The forecast of A5 holds exactly: at $k = 20$ the tiny boundary value $e^{-20}$ is lost completely, and at $k = 40$ the whole solution is garbage. FD stays accurate. Multiple shooting (restarting the IVPs on subintervals) is the shooting-type remedy.

## D. Data-science applications

**D1.**

| $\lambda$ | edf | GCV | RMSE vs truth |
|---|---|---|---|
| $10^{-1}$ | 213.1 | 0.1037 | 0.193 |
| $10^{1}$ | 62.8 | 0.0750 | 0.105 |
| $10^{3}$ | 19.9 | 0.0680 | 0.053 |
| $\mathbf{10^4}$ | **11.6** | **0.0676** | **0.039** |
| $10^{5}$ | 7.0 | 0.0758 | 0.092 |
| $10^{6}$ | 4.4 | 0.1141 | 0.210 |

GCV picks $\lambda = 10^4$, which is also the RMSE-optimal value on this grid. It does so without knowing the truth or the noise level. Connection to BVPs: $D_2^TD_2$ is the discrete fourth-difference operator $h^4\frac{d^4}{dx^4}$, so $\mathbf z$ solves the discretised $z+\lambda h^{-4}z'''' = y$ with natural boundary conditions. In the continuous limit this is the smoothing-spline Euler–Lagrange equation.

**D2.** The fit in $(\log D,\log k)$ converges to $D = 2\times10^{-9}$, $k = 3\times10^{-8}$ from one start and to $D = 1.3\times10^5$, $k = 2.0\times10^6$ from another. Both have the same cost ($5.10\times10^{-5}$) and the same ratio $\sqrt{k/D} = 3.9855$. The Jacobian's singular values are $(0.528, 0)$: exactly rank-deficient, because the model depends on $(D,k)$ only through $m = \sqrt{k/D}$. Refitting in $m$ gives $\hat m = 3.986\pm0.019$ (true 4). Lesson: a flat direction in the least-squares Jacobian means the data cannot identify that combination of parameters. Reparametrise, fix one parameter, or add different experiments (e.g. transient data that involve $D$ alone).

**D3.** With $p = 2\mu/\sigma^2 = 0.6$, the ruin probability is $u(x) = \frac{e^{-px}-e^{-pL}}{1-e^{-pL}}$ and the exit time is $T(x) = \frac1\mu\bigl[L\frac{1-e^{-px}}{1-e^{-pL}}-x\bigr]$. FD with $h = 0.01$ gives $u(2) = 0.264579$ (exact $0.264580$) and $T(2) = 5.59035$ (exact $5.59034$).

| Monte Carlo (20 000 paths) | ruin probability | $E[T]$ |
|---|---|---|
| $\Delta t = 10^{-2}$ | $0.2535\pm0.0031$ | $5.851\pm0.032$ |
| $\Delta t = 10^{-3}$ | $0.2581\pm0.0031$ | $5.704\pm0.031$ |
| $\Delta t = 10^{-2}$ + Brownian bridge | $\mathbf{0.2650}\pm0.0031$ | $\mathbf{5.627}\pm0.030$ |

Discrete monitoring misses crossings that happen between time steps. That biases the ruin probability down and the exit time up, with an $O(\sqrt{\Delta t})$ error. The bridge correction, $P(\text{cross}) = e^{-2x_kx_{k+1}/(\sigma^2\Delta t)}$, removes most of the bias. The deterministic BVP is both faster and more accurate; Monte Carlo becomes attractive only in high dimension.
