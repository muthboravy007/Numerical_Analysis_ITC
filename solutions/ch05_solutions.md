# Chapter 5 — Solutions

> Exercises: [`exercises/ch05_exercises.md`](../exercises/ch05_exercises.md) · Numbers from [`solutions/code/ch05_solutions.py`](code/ch05_solutions.py).

## A. Theory and proofs

**A1.** $|t^2\sin y_1 - t^2\sin y_2|\le t^2|\cos\xi||y_1-y_2|\le4|y_1-y_2|$, so $L = 4$. $\sqrt{|y|}$ is not Lipschitz near 0: $\frac{|\sqrt{|y|}-0|}{|y-0|} = |y|^{-1/2}\to\infty$ (and $y' = \sqrt{|y|}$, $y(0) = 0$ indeed has non-unique solutions).

**A2.** By induction $a_{i+1}\le(1+s)^{i+1}a_0 + t\sum_{j=0}^{i}(1+s)^j = (1+s)^{i+1}a_0 + \frac ts[(1+s)^{i+1}-1]$; with $1+s\le e^s$ the claim follows. For Euler, $e_{i+1}\le(1+hL)e_i + \frac{h^2M}{2}$; apply the lemma with $s = hL$, $t = \frac{h^2M}{2}$, $a_0 = 0$: $e_{i+1}\le\frac{hM}{2L}(e^{(i+1)hL}-1) = \frac{hM}{2L}(e^{L(t_{i+1}-a)}-1)$.

**A3.** Expand $a_1f + a_2f(t+\alpha_2h, y+\delta_2hf) = (a_1+a_2)f + a_2h(\alpha_2f_t + \delta_2ff_y) + O(h^2)$ and match with $T^{(2)} = f + \frac h2(f_t + ff_y)$: $a_1+a_2 = 1$, $a_2\alpha_2 = a_2\delta_2 = \frac12$. Midpoint: $a_1 = 0$, $a_2 = 1$, $\alpha_2 = \delta_2 = \frac12$. Modified Euler: $a_1 = a_2 = \frac12$, $\alpha_2 = \delta_2 = 1$.

**A4.** Interpolate $f$ linearly at $t_{i-1}, t_i$: $P(t) = f_i + \frac{f_i - f_{i-1}}{h}(t-t_i)$; $\int_{t_i}^{t_{i+1}}P = hf_i + \frac h2(f_i - f_{i-1})$. The interpolation error $\frac{y'''(\xi)}{2}(t-t_i)(t-t_{i-1})$ integrates (weighted MVT, the factor has constant sign) to $\frac{y'''(\mu)}{2}\int_0^h s(s+h)\,ds = \frac{5}{12}h^3y'''(\mu)$; dividing by $h$ gives $\tau = \frac5{12}h^2y'''$.

**A5.** (a) $Q = 1+z$; (b) $Q = \frac1{1-z}$; (c) $Q = \frac{1+z/2}{1-z/2}$; (d) $Q = 1+z+\frac{z^2}2+\frac{z^3}6+\frac{z^4}{24}$. (b) and (c) satisfy $|Q|<1$ for all $\operatorname{Re}z<0$ (A-stable); (a) and (d) have bounded stability regions. For (c), $Q(z)\to-1$ as $z\to-\infty$: very stiff components are not damped but flip sign each step (slowly decaying oscillations, cf. Example 5.9.4); backward Euler has $Q\to0$ (L-stable).

## B. Hand computation

**B1.** $h = 0.5$: $w_1 = 0$, $w_2 = 0 + 0.5(0.5e^{-0.5} - 0) = 0.151633$ (exact $y(0.5) = 0.075816$, $y(1) = 0.183940$). $h = 0.25$: $w = 0, 0, 0.048675, 0.112323, 0.172811$ at $t = 0, 0.25, 0.5, 0.75, 1$ (exact $0, 0.024338, 0.075816, 0.132853, 0.183940$). Error at $t = 1$: $0.0322$ ($h = 0.5$), $0.0111$ ($h = 0.25$).

**B2.** $f_y = -1$ ⇒ $L = 1$. $y'' = e^{-t}(1 - 2t + \frac{t^2}2)$, $\max_{[0,1]}|y''| = 1$ (at $t = 0$). Bound $\frac{0.25\cdot1}{2}(e^1 - 1) = 0.2148$; actual error $0.0111$ — valid but ~20× pessimistic.

**B3.** $f' = f_t + f_yf = -\frac y{t^2} + 2t + \frac1t\bigl(\frac yt + t^2\bigr) = 3t$. So $w_{i+1} = w_i + h\bigl(\frac{w_i}{t_i} + t_i^2\bigr) + \frac{h^2}{2}\cdot3t_i$. Results: $1, 1.59375, 2.42031, 3.52682, 4.96034$ at $t = 1, 1.25, \dots, 2$ (exact $1, 1.60156, 2.4375, 3.55469, 5$).

**B4.** $k_1 = 0.25f(1,1) = 0.5$; $k_2 = 0.25f(1.125, 1.25) = 0.594184$; $k_3 = 0.25f(1.125, 1.297092) = 0.604649$; $k_4 = 0.25f(1.25, 1.604649) = 0.711555$. $w_1 = 1 + \frac16(0.5 + 2(0.594184) + 2(0.604649) + 0.711555) = 1.601537$ (exact $1.601563$). Full: $w(2) = 4.999910$ (error $9.0\times10^{-5}$).

**B5.** Midpoint $w(2) = 4.962114$ (error $0.038$); modified Euler $4.966830$ (error $0.033$).

**B6.** $h = 0.2$, exact $y(1.8) = 3.816$, $y(2) = 5$. AB4: $3.815971$, $4.999969$; predictor–corrector: $3.815972$, $4.999969$ — both $\approx3\times10^{-5}$ accurate. (Starting values by RK4: $1.463991, 2.071983, 2.847975$.)

**B7.**

| $h$ | Euler $1+h\lambda$ | RK4 $Q(h\lambda)$ | backward Euler $\frac1{1-h\lambda}$ |
|---|---|---|---|
| 0.05 | 0.00 (stable) | 0.375 | 0.500 |
| 0.10 | −1.00 (borderline) | 0.333 | 0.333 |
| 0.15 | −2.00 (unstable) | 1.375 (unstable) | 0.250 |

Euler needs $h<0.1$, RK4 $h<2.785/20 = 0.139$; backward Euler is stable for all $h$.

**B8.** (a) $\lambda^2 - 1 = 0$: roots $\pm1$ — weakly stable (the parasitic root $-1$ causes growing oscillations, Example 5.8.4). (b) $\lambda^2 - \frac43\lambda + \frac13 = (\lambda-1)(\lambda-\frac13)$: roots $1, \frac13$ — strongly stable (BDF2 is also A-stable).

## C. Programming (key results)

**C1.** Errors at $t = 1$ for $h = 0.1\to0.0125$: Euler $3.6\times10^{-3}\to3.9\times10^{-4}$ (orders $1.11, 1.05, 1.03$); Heun $3.5\times10^{-4}\to5.6\times10^{-6}$ (orders $1.98, 1.99, 2.00$); RK4 $3.4\times10^{-8}\to5.1\times10^{-12}$ (orders $4.38, 4.22, 4.12$).

**C2.** 7 accepted steps with $h\in[0.019, 0.232]$; maximum error $2.0\times10^{-7}$ (below the local tolerance per unit step).

**C3.** Periods: $\theta_0 = 0.1$: $2.00732$ s; $1.0$: $2.13914$ s; $2.5$: $3.29593$ s; small-angle $2.00607$ s. Large-amplitude pendulums are slower (the exact period involves an elliptic integral $K(\sin\frac{\theta_0}{2})$).

**C4.** Function evaluations to $t = 40$: RK45 242 066; LSODA 414; BDF 366 (all agree: $y_1(40) = 0.715827$). The explicit method is limited by stability, not accuracy.

## D. Data-science applications

**D1.** $R_0 = \beta/\gamma = 5$. Baseline: peak on day 47.7 with $I = 30.5\%$; final attack rate $99.3\%$.

| contact reduction | peak day | peak $I$ | attack rate |
|---|---|---|---|
| 0% | 47.7 | 0.305 | 0.993 |
| 30% | 64.3 | 0.232 | 0.966 |
| 50% | 89.9 | 0.154 | 0.893 |

Reducing contacts delays and lowers the peak much more than it reduces the final size (with $R_0$ still well above 1) — protecting healthcare capacity is the main benefit of "flattening the curve".

**D2.** Least squares (solving $C' = -kC$ numerically inside each residual evaluation) gives $\hat k = 0.2493$ h$^{-1}$, $\hat V = 9.956$ L (true $0.25$, $10$); half-life $\ln2/\hat k = 2.78$ h.

**D3.** Largest eigenvalue of the Hessian $L = 25$ ⇒ stability requires $\alpha<2/L = 0.08$. After 200 steps: $\alpha = 0.01$: $\|\mathbf w\| = 0.134$ (slow in the flat direction); $0.07$: $5.0\times10^{-7}$; $0.079$: $6.3\times10^{-3}$ (stable but the stiff component oscillates with factor $|1-25\alpha| = 0.975$); $0.081$: $140$ (diverged). A stiff solver (Radau) for the gradient flow gives $\mathbf w(2) = (0.135335, 1.93\times10^{-22}) = (e^{-2}, e^{-50})$ — the continuous flow has no step-size limit; the learning-rate limit is purely a stability property of the discretisation.
