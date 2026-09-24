# Semester 1 Midterm — Solutions and Marking Scheme

> All numbers are verified by [`code/verify_exams.py`](code/verify_exams.py).

**Q1.** (a) $\sqrt{501}\to22.38$ and $\sqrt{500}\to22.36$ (chopped). Their difference is $0.02$, and $500\times0.02 = 10.00$. The relative error is $\frac{|10.00-11.1748|}{11.1748} = 0.105$, i.e. **10.5%**; only one significant digit survives. [4]
(b) $f(x) = \frac{x}{\sqrt{x+1}+\sqrt x}$. Then $22.38+22.36 = 44.74$ and $500/44.74 = 11.17$ (chopped), with relative error $4.3\times10^{-4}$, which is full 4-digit accuracy. [5]
(c) The relative condition number of $a-b$ is $\frac{|a|+|b|}{|a-b|}\approx\frac{44.74}{0.02}\approx2200$. Rounding errors of relative size $10^{-3}$ in $a$ and $b$ are amplified about 2000-fold. The subtraction is exact, but it exposes the earlier rounding (catastrophic cancellation). [3]

**Q2.** (a) $f(1) = -2<0<7 = f(2)$ and $f'(x) = 3x^2+2>0$, so by the IVT and monotonicity there is exactly one root. [3]
(b) [4]

| step | $p$ | $f(p)$ | new interval |
|---|---|---|---|
| 1 | 1.5 | 1.375 | $[1,1.5]$ |
| 2 | 1.25 | $-0.546875$ | $[1.25,1.5]$ |
| 3 | 1.375 | 0.349609 | $[1.25,1.375]$ |

(c) $\frac{2-1}{2^n}<10^{-6}$ requires $n>\log_2 10^6 = 19.93$, so $n = 20$. [3]
(d) $p_1 = 1.5-\frac{1.375}{8.75} = 1.342857$ and $p_2 = 1.328384$, against the root $p = 1.3282689$. The errors are $1.7\times10^{-1}\to1.5\times10^{-2}\to1.2\times10^{-4}$, i.e. about 1, then 2, then 4 correct digits. Since $f'(p)\ne0$, $e_{n+1}\approx\frac{f''(p)}{2f'(p)}e_n^2$, with $\frac{f''}{2f'}\approx0.55$, so the number of correct digits roughly doubles. [6]

**Q3.** (a) [8]

| $g$ | $g'(p)$ | local convergence? |
|---|---|---|
| $g_1$ | $-\frac32p^2 = -2.646$ | no, $\lvert g'\rvert>1$ (repelling) |
| $g_2$ | $-\frac23(5-2p)^{-2/3} = -0.378$ | yes |
| $g_3$ | $-\frac{10p}{(p^2+2)^2} = -0.937$ | yes, but very slowly |

(b) One extra digit needs about $\frac{1}{-\log_{10}|g'(p)|}$ iterations: $2.4$ for $g_2$ and $36$ for $g_3$. The iterates of $g_2$ from 1 are $1.4423, 1.2837, 1.3449, 1.3220, \dots$, which oscillate because $g'<0$. [3]
(c) The fixed-point (contraction mapping) theorem: if $g\in C[a,b]$, $g([a,b])\subset[a,b]$ and $|g'(x)|\le k<1$ on $(a,b)$, then the iteration converges to the unique fixed point for every $p_0\in[a,b]$, with $|p_n-p|\le k^n\max(p_0-a,b-p_0)$. [3]

**Q4.** (a) [7]

| $x$ | $f$ | 1st | 2nd | 3rd |
|---|---|---|---|---|
| 0 | 1 | | | |
| 1 | 2 | 1 | | |
| 2 | 4 | 2 | 0.5 | |
| 4 | 16 | 6 | 1.3333 | 0.208333 |

$P_3(x) = 1+x+0.5x(x-1)+0.208333\,x(x-1)(x-2)$.
(b) $P_3(3) = 1+3+3+1.25 = 8.25$, against the true value 8. The error is $0.25$. [3]
(c) $|f(3)-P_3(3)| = \frac{|f^{(4)}(\xi)|}{4!}|3\cdot2\cdot1\cdot(-1)|$ with $f^{(4)} = (\ln2)^42^\xi\in[0.231, 3.693]$ for $\xi\in[0,4]$. So the error is at most $\frac{3.693}{24}\cdot6 = 0.923$ (and at least $0.058$). The actual $0.25$ lies within these bounds. [6]

**Q5.** (a) The function values at $0, 0.25, \dots, 1$ are $1, 0.939413, 0.778801, 0.569783, 0.367879$.
- $T_2 = \frac12\bigl(\frac{1}{2}+0.778801+\frac{0.367879}{2}\bigr) = 0.731370$
- $T_4 = 0.742984$ (error $3.8\times10^{-3}$)
- $S_4 = \frac{0.25}{3}(1+4(0.939413)+2(0.778801)+4(0.569783)+0.367879) = 0.746855$ (error $3.1\times10^{-5}$) [9]

(b) $|E_T|\le\frac{b-a}{12}h^2\max|f''|$. With $f'' = (4x^2-2)e^{-x^2}$, $\max_{[0,1]}|f''| = 2$ (at $x = 0$), so $|E_T|\le\frac{1}{12}(0.0625)(2) = 0.0104$. [4]
(c) $\frac{4T_4-T_2}{3} = 0.746855 = S_4$ exactly. Richardson extrapolation of the trapezoidal rule *is* Simpson's rule: this is the first Romberg column. It removes the $h^2$ term of the Euler–Maclaurin expansion. [5]

**Q6.** The exact value is $y(0.2) = e^{-0.04} = 0.960789$. [16]

| method | $y_1$ | error | order |
|---|---|---|---|
| Euler | $1+0.2\cdot f(0,1) = 1.000000$ | $3.9\times10^{-2}$ | 1 |
| Heun | $k_1 = 0$, $k_2 = f(0.2,1) = -0.4$, so $1+0.1(-0.4) = 0.960000$ | $7.9\times10^{-4}$ | 2 |
| RK4 | $K_1 = 0$, $K_2 = -0.04$, $K_3 = -0.0392$, $K_4 = -0.076864$, so $1+\frac{0-0.08-0.0784-0.076864}{6} = 0.960789$ | $1.1\times10^{-7}$ | 4 |

The one-step (local) errors are $O(h^2)$, $O(h^3)$ and $O(h^5)$: roughly $0.04$, $0.008\cdot0.1$ and $0.0003\cdot10^{-3}$.

**Q7.** (a) The amplification factor is $1-50h$, so we need $|1-50h|\le1$, i.e. $h\le0.04$. [4]
(b) A problem is stiff when it contains decay rates on very different time scales (large $|\lambda|$ alongside slow dynamics). Explicit methods are then limited by *stability*, not accuracy, and need $h\sim1/|\lambda_{\max}|$. A-stable implicit methods (backward Euler, trapezoid, BDF) can take steps suited to the slow dynamics. Examples: chemical or pharmacokinetic models with fast and slow compartments; gradient flow on ill-conditioned losses, where the learning-rate limit is $2/L$. [4]
