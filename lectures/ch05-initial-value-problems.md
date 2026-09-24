# Chapter 5 — Initial-Value Problems for Ordinary Differential Equations

> **Reference:** Burden & Faires, Chapter 5 (§5.1–5.6, 5.9–5.11) plus §5.10 for data science.
> **Code:** [`numlib/ode.py`](../numlib/ode.py) · **All numbers reproduced by** [`lectures/code/ch05.py`](code/ch05.py)
> **Worked problems:** [`examples/ch05`](../examples/ch05_examples.md) · **Homework:** [`exercises/ch05`](../exercises/ch05_exercises.md)

## Learning outcomes

1. State the Lipschitz condition and the existence–uniqueness and well-posedness theorems for IVPs.
2. Apply and analyse Euler's method (local/global truncation error, round-off).
3. Derive Taylor and Runge–Kutta methods and verify their order.
4. Control the step size adaptively with the Runge–Kutta–Fehlberg method.
5. Derive and use Adams–Bashforth/Adams–Moulton multistep and predictor–corrector methods.
6. Reduce higher-order equations to first-order systems and solve them.
7. Analyse consistency, convergence and stability (root condition, absolute stability regions) and handle stiff problems with implicit methods.
8. Use ODE models in data science: epidemics, growth curves, gradient flow, momentum and neural ODEs.

Throughout, the **model problem** is
$$
y' = y - t^2 + 1,\quad 0\le t\le2,\quad y(0) = 0.5,\qquad y(t) = (t+1)^2 - \tfrac12e^t,
$$
and we write $w_i\approx y(t_i)$, $t_i = a + ih$.

---

## 5.1 The Elementary Theory of Initial-Value Problems

We consider $y' = f(t,y)$, $a\le t\le b$, $y(a) = \alpha$.

**Definition 5.1 (Lipschitz condition).** $f$ satisfies a Lipschitz condition in $y$ on $D\subset\mathbb R^2$ with constant $L$ if
$$
|f(t,y_1) - f(t,y_2)|\le L|y_1-y_2|\qquad\text{for all }(t,y_1),(t,y_2)\in D.
$$
A sufficient condition on a convex $D$: $\bigl|\frac{\partial f}{\partial y}(t,y)\bigr|\le L$ (by the MVT).

**Theorem 5.4 (existence and uniqueness).** If $f$ is continuous and Lipschitz in $y$ on $D = \{a\le t\le b,\ -\infty<y<\infty\}$, the IVP has a unique solution $y(t)$ on $[a,b]$.

**Definition 5.5 (well-posed).** The IVP is **well-posed** if a unique solution exists, and for any $\varepsilon>0$ there is $k(\varepsilon)$ such that whenever $|\varepsilon_0|<\varepsilon$ and $|\delta(t)|<\varepsilon$, the perturbed problem $z' = f(t,z)+\delta(t)$, $z(a) = \alpha+\varepsilon_0$ has a unique solution with $|z(t)-y(t)|<k(\varepsilon)\varepsilon$.

**Theorem 5.6.** Under the hypotheses of Theorem 5.4 the IVP is well-posed.

Well-posedness matters because every numerical method solves a *perturbed* problem (round-off, truncation); if small perturbations produced large changes, no method could succeed.

### Examples for §5.1

**Example 5.1.1 (a Lipschitz constant).** $f(t,y) = t|y|$ on $D = \{1\le t\le2,\ y\in\mathbb R\}$: $|t|y_1| - t|y_2||\le t|y_1-y_2|\le2|y_1-y_2|$. So $L = 2$ (although $f$ is not differentiable at $y = 0$).

**Example 5.1.2 (the model problem is well-posed).** $f = y - t^2+1$ has $\partial f/\partial y = 1$, so $L = 1$ on $\{0\le t\le2\}\times\mathbb R$; Theorems 5.4 and 5.6 apply.

**Example 5.1.3 (non-uniqueness without Lipschitz).** $y' = y^{1/3}$, $y(0) = 0$ has the solutions $y\equiv0$ **and** $y = (2t/3)^{3/2}$ (e.g. $y(1) = 0.5443$). $\partial f/\partial y = \frac13y^{-2/3}$ is unbounded near $y = 0$: no Lipschitz condition, no uniqueness. A numerical method started at $y_0 = 0$ will follow $y\equiv0$ — the method cannot tell which solution you "meant".

**Example 5.1.4 (finite-time blow-up).** $y' = y^2$, $y(0) = 1$ has $y = \frac{1}{1-t}$: $y(0.9) = 10$, $y(0.99) = 100$, $y\to\infty$ as $t\to1$. $f$ is Lipschitz only on bounded $y$-sets, so the solution exists only locally. Adaptive solvers report failure as the step size collapses near $t = 1$.

**Example 5.1.5 (sensitivity to initial data).** For the model problem, $z(t) - y(t)$ satisfies $e' = e$, so $|z-y| = |\varepsilon_0|e^t$. Perturbing $y(0)$ by $10^{-3}$ changes $y(2)$ by $7.39\times10^{-3}$ — amplification $e^{L(b-a)} = e^2$, exactly the factor that appears in the error bounds below.

---

## 5.2 Euler's Method

From Taylor, $y(t_{i+1}) = y(t_i) + hf(t_i,y(t_i)) + \frac{h^2}{2}y''(\xi_i)$. Dropping the remainder:
$$
\boxed{w_0 = \alpha,\qquad w_{i+1} = w_i + hf(t_i,w_i).}
$$

**Theorem 5.9 (error bound).** If $f$ is continuous, Lipschitz with constant $L$ on $D$ and $|y''(t)|\le M$, then
$$
|y(t_i) - w_i|\le\frac{hM}{2L}\bigl[e^{L(t_i-a)} - 1\bigr].
$$
*Proof idea.* The error $e_i = y(t_i)-w_i$ satisfies $|e_{i+1}|\le(1+hL)|e_i| + \frac{h^2M}{2}$; unrolling and using $1+hL\le e^{hL}$ gives the result (Lemma 5.8). ∎

**Local vs global error.** The *local truncation error* $\tau_{i+1}(h) = \frac{y_{i+1}-y_i}{h} - f(t_i,y_i) = \frac h2y''(\xi_i)$ is $O(h)$; the *global* error is also $O(h)$. Euler's method is first order.

**Round-off.** If each step has round-off $\le\delta$ (and $|u_0 - \alpha|\le\delta_0$),
$$
|y(t_i)-u_i|\le\frac1L\Bigl(\frac{hM}{2}+\frac\delta h\Bigr)\bigl[e^{L(t_i-a)}-1\bigr] + |\delta_0|e^{L(t_i-a)},
$$
minimised at $h = \sqrt{2\delta/M}$. Below that, reducing $h$ increases the error.

### Examples for §5.2

**Example 5.2.1 (Euler on the model problem, $h = 0.2$).** $w_1 = 0.5 + 0.2(0.5 - 0 + 1) = 0.8$, $w_2 = 0.8 + 0.2(0.8 - 0.04 + 1) = 1.152$, …

| $t_i$ | $w_i$ | $y(t_i)$ | error | bound $0.1(0.5e^2-2)(e^{t_i}-1)$ |
|---|---|---|---|---|
| 0.2 | 0.8000000 | 0.8292986 | 0.0292986 | 0.0375173 |
| 0.6 | 1.5504000 | 1.6489406 | 0.0985406 | 0.1393103 |
| 1.0 | 2.4581760 | 2.6408591 | 0.1826831 | 0.2911677 |
| 1.4 | 3.4517734 | 3.7324000 | 0.2806266 | 0.5177122 |
| 2.0 | 4.8657845 | 5.3054720 | 0.4396874 | 1.0826435 |

Here $L = 1$ and $M = \max|y''| = \max|2 - \frac12e^t| = 0.5e^2-2 = 1.6945$. The bound holds but grows faster than the actual error.

**Example 5.2.2 (a problem with a Gaussian solution).** $y' = -2ty$, $y(0) = 1$, exact $y = e^{-t^2}$. Euler with $h = 0.1$ gives $w_{10} = 0.3817067$ vs $e^{-1} = 0.3678794$ (error $1.38\times10^{-2}$).

**Example 5.2.3 (observed order).** Same problem: $h = 0.1, 0.05, 0.025$ give errors $1.38\times10^{-2},\ 6.50\times10^{-3},\ 3.16\times10^{-3}$ — halving $h$ halves the error: first order.

**Example 5.2.4 (optimal step with round-off).** If each step of the model problem carries round-off $\delta = 5\times10^{-7}$ (6-digit arithmetic), $h_{\text{opt}} = \sqrt{2\delta/M} = \sqrt{10^{-6}/1.6945} = 7.7\times10^{-4}$. Smaller steps only add round-off.

**Example 5.2.5 (logistic growth).** $y' = 0.8y(1-y/100)$, $y(0) = 1$, exact $y = \frac{100}{1+99e^{-0.8t}}$:

| $h$ | $w(5)$ | $w(10)$ |
|---|---|---|
| 1.0 | 17.14 | 94.23 |
| 0.5 | 23.98 | 95.91 |
| 0.1 | 32.78 | 96.64 |
| exact | 35.55 | 96.79 |

Euler systematically lags the steep phase of the S-curve — forecasting a product launch this way would under-predict mid-term adoption badly.

**Example 5.2.6 (gradient descent is Euler's method).** For $F(w) = (w-3)^2$ the gradient flow is $w' = -F'(w) = -2(w-3)$, and Euler with step $\alpha$ is $w_{k+1} = w_k - 2\alpha(w_k-3)$ — gradient descent with learning rate $\alpha$. From $w_0 = 0$:
* $\alpha = 0.1$: $0.6,\ 1.08,\ 1.464,\ 1.771,\ldots$ (slow monotone convergence);
* $\alpha = 0.5$: $3,3,3,\ldots$ (one step: $\alpha = 1/L$ with $L = F'' = 2$);
* $\alpha = 0.9$: $5.4,\ 1.08,\ 4.536,\ 1.771,\ldots$ (oscillating convergence);
* $\alpha = 1.1$: $6.6,\ -1.32,\ 8.18,\ -3.22,\ldots$ (divergence).
The stability limit $|1-2\alpha|<1\iff\alpha<1 = 2/L$ is Euler's stability condition $|1+h\lambda|<1$ with $\lambda = -2$ (§5.8).

---

## 5.3 Higher-Order Taylor Methods

**Definition 5.11.** A difference method $w_{i+1} = w_i + h\phi(t_i,w_i)$ has **local truncation error**
$$
\tau_{i+1}(h) = \frac{y_{i+1} - y_i}{h} - \phi(t_i,y_i).
$$

**Taylor method of order $n$.** Keep $n$ terms of the Taylor series, expressing $y^{(k)}$ through derivatives of $f$ (with $y'' = f' = f_t + f_yf$, etc.):
$$
w_{i+1} = w_i + hT^{(n)}(t_i,w_i),\qquad T^{(n)} = f + \frac h2f' + \cdots + \frac{h^{n-1}}{n!}f^{(n-1)}.
$$
**Theorem 5.12.** If $y\in C^{n+1}[a,b]$, the local truncation error of the Taylor method of order $n$ is $O(h^n)$.

Drawback: requires derivatives of $f$ — tedious by hand but automatic with symbolic or **automatic differentiation** (the basis of modern Taylor-series integrators).

### Examples for §5.3

**Example 5.3.1 (Taylor order 2).** For $f = y - t^2+1$: $f' = \frac{d}{dt}f = y' - 2t = y - t^2 + 1 - 2t$. So
$$
w_{i+1} = w_i + h\Bigl[\bigl(1+\tfrac h2\bigr)(w_i - t_i^2 + 1) - ht_i\Bigr].
$$
With $h = 0.2$: $w_1 = 0.83$, $w_2 = 1.2158$, …, $w_5 = 2.6486459$, $w_{10} = 5.3476843$ (error $4.2\times10^{-2}$ vs Euler's $0.44$).

**Example 5.3.2 (Taylor order 4).** $f'' = y - t^2 - 2t - 1$ and $f''' = f''$ (for this $f$), giving
$$
w_{i+1} = w_i + hf + \tfrac{h^2}{2}f' + \tfrac{h^3}{6}f'' + \tfrac{h^4}{24}f''' .
$$
With $h = 0.2$:

| $t_i$ | 0.2 | 0.6 | 1.0 | 1.4 | 2.0 |
|---|---|---|---|---|---|
| $w_i$ | 0.8293000 | 1.6489468 | 2.6408744 | 3.7324321 | 5.3055554 |
| error | $1.4\times10^{-6}$ | $6.2\times10^{-6}$ | $1.5\times10^{-5}$ | $3.2\times10^{-5}$ | $8.3\times10^{-5}$ |

**Example 5.3.3 (Taylor order 2 for $y' = -2ty$).** $f' = f_t + f_yf = -2y + (-2t)(-2ty) = (4t^2-2)y$. With $h = 0.1$: $w(1) = 0.3652608$, error $2.6\times10^{-3}$ (Euler: $1.4\times10^{-2}$).

**Example 5.3.4 (observed orders).** Errors at $t = 2$ for the model problem:

| $h$ | Taylor 2 | Taylor 4 |
|---|---|---|
| 0.2 | $4.22\times10^{-2}$ | $8.34\times10^{-5}$ |
| 0.1 | $1.14\times10^{-2}$ | $5.67\times10^{-6}$ |
| 0.05 | $2.96\times10^{-3}$ | $3.69\times10^{-7}$ |

Ratios ≈ 3.8 (order 2) and ≈ 15 (order 4).

**Example 5.3.5 (local vs global).** One Taylor-4 step from the exact $y(0)$ has error $1.38\times10^{-6} = O(h^5)$ (so $\tau = O(h^4)$), while after 10 steps the global error is $8.3\times10^{-5} = O(h^4)$: the global error is roughly (number of steps) × (local error) × (amplification).

---

## 5.4 Runge–Kutta Methods

RK methods achieve Taylor-method accuracy using only evaluations of $f$ — by matching the Taylor expansion with a combination of slopes.

**Theorem 5.13 (Taylor in two variables)** underlies the derivation: matching $a_1f(t,y) + a_2f(t+\alpha_2h, y+\delta_2hf)$ with $f + \frac h2(f_t+f_yf) + O(h^2)$ requires
$$
a_1 + a_2 = 1,\qquad a_2\alpha_2 = \tfrac12,\qquad a_2\delta_2 = \tfrac12 .
$$
One free parameter gives a family of second-order methods:

| Method | $a_2$ | formula |
|---|---|---|
| **Midpoint** | 1 | $w_{i+1} = w_i + hf\bigl(t_i+\frac h2,\ w_i+\frac h2f(t_i,w_i)\bigr)$ |
| **Modified Euler** (trapezoid P–C) | $\frac12$ | $w_{i+1} = w_i + \frac h2\bigl[f(t_i,w_i) + f(t_{i+1}, w_i + hf(t_i,w_i))\bigr]$ |
| Ralston | $\frac34$ | minimises the error constant |

**Heun's third-order method (B&F):** $w_{i+1} = w_i + \frac h4\bigl[f(t_i,w_i) + 3f\bigl(t_i+\frac{2h}3, w_i+\frac{2h}{3}f(t_i+\frac h3, w_i+\frac h3f(t_i,w_i))\bigr)\bigr]$.

**Classical Runge–Kutta of order four (RK4):**
$$
\begin{aligned}
k_1 &= hf(t_i,w_i), & k_2 &= hf(t_i+\tfrac h2,\ w_i+\tfrac12k_1),\\
k_3 &= hf(t_i+\tfrac h2,\ w_i+\tfrac12k_2), & k_4 &= hf(t_{i+1},\ w_i+k_3),\\
w_{i+1} &= w_i + \tfrac16(k_1+2k_2+2k_3+k_4). &&
\end{aligned}
$$
Local truncation error $O(h^4)$ if $y\in C^5$.

**Evaluations per step vs attainable order:** $1,2,3,4\to1,2,3,4$; but 5 evaluations still give only order 4, and order 5 needs 6 — which is why RK4 is the sweet spot.

### Examples for §5.4

**Example 5.4.1 (comparing second/third/fourth-order methods, $h = 0.2$).**

| method | $w(1)$ | error | $w(2)$ | error |
|---|---|---|---|---|
| midpoint | 2.6331668 | $7.7\times10^{-3}$ | 5.2903695 | $1.5\times10^{-2}$ |
| modified Euler | 2.6176876 | $2.3\times10^{-2}$ | 5.2330546 | $7.2\times10^{-2}$ |
| Heun (order 3) | 2.6405555 | $3.0\times10^{-4}$ | 5.3050072 | $4.7\times10^{-4}$ |
| RK4 | 2.6408227 | $3.6\times10^{-5}$ | 5.3053630 | $1.1\times10^{-4}$ |

**Example 5.4.2 (RK4 by hand, first step).** $t_0 = 0$, $w_0 = 0.5$, $h = 0.2$:
$k_1 = 0.2f(0,0.5) = 0.2(1.5) = 0.3$;
$k_2 = 0.2f(0.1, 0.65) = 0.2(1.64) = 0.328$;
$k_3 = 0.2f(0.1, 0.664) = 0.2(1.654) = 0.3308$;
$k_4 = 0.2f(0.2, 0.8308) = 0.2(1.7908) = 0.35816$;
$w_1 = 0.5 + \frac16(0.3 + 0.656 + 0.6616 + 0.35816) = 0.8292933$ (exact $0.8292986$, error $5.3\times10^{-6}$).
Full table: errors grow from $5.3\times10^{-6}$ ($t = 0.2$) to $1.09\times10^{-4}$ ($t = 2$).

**Example 5.4.3 (equal work).** 80 function evaluations on $[0,2]$:

| method | $h$ | error at $t=2$ |
|---|---|---|
| Euler | 0.025 | $6.6\times10^{-2}$ |
| modified Euler | 0.05 | $4.8\times10^{-3}$ |
| RK4 | 0.1 | $7.0\times10^{-6}$ |

For the same cost, RK4 is four orders of magnitude more accurate.

**Example 5.4.4 (RK4 order check).** $y' = -2ty$: errors at $t=1$ for $h = 0.2, 0.1, 0.05$ are $2.42\times10^{-5},\ 1.63\times10^{-6},\ 1.03\times10^{-7}$ — ratios 14.9 and 15.9 → order 4.

**Example 5.4.5 (deriving a 2-stage method).** Choosing $a_2 = \frac34$ in the family above gives $a_1 = \frac14$, $\alpha_2 = \delta_2 = \frac{1}{2a_2} = \frac23$ and **Ralston's method**
$w_{i+1} = w_i + \frac h4\bigl[f(t_i,w_i) + 3f(t_i + \frac23h,\ w_i + \frac23hf(t_i,w_i))\bigr]$. Any member of the family is second order; they differ only in the $h^2$ error constant (Ralston's choice minimises a bound on it).

---

## 5.5 Error Control and the Runge–Kutta–Fehlberg Method

Choose $h$ so that the **local** error per step stays below a tolerance. Given an order-$n$ method $w$ and an order-$(n+1)$ method $\tilde w$ computed from the same data, $\tau_{i+1}(h)\approx\frac1h(\tilde w_{i+1}-w_{i+1})$. Since $\tau(qh)\approx q^n\tau(h)$, the step that achieves $|\tau|\approx\varepsilon$ is
$$
q\approx\Bigl(\frac{\varepsilon h}{|\tilde w_{i+1}-w_{i+1}|}\Bigr)^{1/n}.
$$

**RKF45** (B&F Algorithm 5.3) uses six evaluations to produce a 4th-order $w$ and 5th-order $\tilde w$:
$$
w_{i+1} = w_i + \tfrac{25}{216}k_1 + \tfrac{1408}{2565}k_3 + \tfrac{2197}{4104}k_4 - \tfrac15k_5,\quad
\tilde w_{i+1} = w_i + \tfrac{16}{135}k_1 + \tfrac{6656}{12825}k_3 + \tfrac{28561}{56430}k_4 - \tfrac{9}{50}k_5 + \tfrac{2}{55}k_6,
$$
accepts the step if $R = |\tilde w-w|/h\le\text{TOL}$, and updates $h\leftarrow qh$ with the conservative $q = 0.84(\text{TOL}/R)^{1/4}$, limited to $[0.1, 4]$ and to $[h_{\min},h_{\max}]$. SciPy's `solve_ivp(method="RK45")` uses the related Dormand–Prince pair.

### Examples for §5.5

**Example 5.5.1 (RKF45 on the model problem, TOL $=10^{-6}$, $h_{\max} = 0.5$).**

| $t_i$ | $h_i$ | $w_i$ | error |
|---|---|---|---|
| 0.1365436 | 0.1365436 | 0.7185790 | $7\times10^{-8}$ |
| 0.4012266 | 0.1332144 | 1.2166084 | $2\times10^{-7}$ |
| 0.9630853 | 0.1465906 | 2.5438211 | $7\times10^{-7}$ |
| 1.4678780 | 0.1864963 | 3.9204152 | $1.3\times10^{-6}$ |
| 2.0000000 | 0.0322931 | 5.3054738 | $1.9\times10^{-6}$ |

14 accepted steps; the step sizes adjust automatically (the last step is shortened to land on $t = 2$).

**Example 5.5.2 (steps adapt to the solution).** $y' = 10y(1-y)$, $y(0) = 10^{-3}$: a sharp S-curve around $t\approx0.69$. RKF45 (TOL $10^{-6}$, $h_{\max} = 0.2$) takes 68 steps: $h$ shrinks to $0.0168$ near $t\approx0.92$ where curvature is largest and grows to $0.108$ in the flat parts. Maximum error $8.4\times10^{-6}$.

**Example 5.5.3 (one step in detail).** Model problem, $h = 0.25$ from $t = 0$: $w_1 = 0.9204886$, $\tilde w_1 = 0.9204870$, $R = |\tilde w_1-w_1|/h = 6.2\times10^{-6}>10^{-6}$ → reject. $q = 0.84(10^{-6}/6.2\times10^{-6})^{1/4} = 0.532$; retry with $h = 0.133$. (The true local error of $w_1$ was $1.3\times10^{-6}$ — the estimate is of the right size.)

**Example 5.5.4 (library comparison).** `solve_ivp(RK45, rtol=1e-6, atol=1e-9)` on Example 5.5.2 needs 44 steps / 272 evaluations and returns $y(2) = 0.99999783$ (exact $0.99999794$).

**Example 5.5.5 (why $0.84$?).** $0.84\approx(1/2)^{1/4}$: aiming for half the tolerance leaves a safety margin so that the next step is rarely rejected (a rejected step wastes 6 evaluations).

---

## 5.6 Multistep Methods

Use several previous values. An **$m$-step method**:
$$
w_{i+1} = a_{m-1}w_i + \cdots + a_0w_{i+1-m} + h\bigl[b_mf_{i+1} + b_{m-1}f_i + \cdots + b_0f_{i+1-m}\bigr].
$$
Explicit if $b_m = 0$, implicit otherwise. Derivation: integrate $y' = f$ over $[t_i,t_{i+1}]$ and replace $f$ by its interpolating polynomial through previous points (Adams–Bashforth) or including $t_{i+1}$ (Adams–Moulton).

| method | formula | local truncation error |
|---|---|---|
| AB2 | $w_{i+1} = w_i + \frac h2(3f_i - f_{i-1})$ | $\frac{5}{12}y'''h^2$ |
| AB4 | $w_{i+1} = w_i + \frac{h}{24}(55f_i - 59f_{i-1} + 37f_{i-2} - 9f_{i-3})$ | $\frac{251}{720}y^{(5)}h^4$ |
| AM2 (trapezoid) | $w_{i+1} = w_i + \frac h2(f_{i+1}+f_i)$ | $-\frac1{12}y'''h^2$ |
| AM3 | $w_{i+1} = w_i + \frac{h}{24}(9f_{i+1} + 19f_i - 5f_{i-1} + f_{i-2})$ | $-\frac{19}{720}y^{(5)}h^4$ |

**Predictor–corrector (Adams fourth order, B&F Algorithm 5.4):** start with RK4 for $w_1,w_2,w_3$; then predict with AB4, evaluate $f$ at the prediction, correct with AM3. Two evaluations per step, error constant $19/720$ instead of $251/720$.

Multistep methods are cheaper per step than RK (1–2 new evaluations) but need starting values and are awkward to use with variable step size.

### Examples for §5.6

**Example 5.6.1–5.6.2 (AB4 and predictor–corrector, $h = 0.2$).** Starting values from RK4:

| $t_i$ | AB4 | error | PC4 | error |
|---|---|---|---|---|
| 0.8 | 2.1272892 | $6.0\times10^{-5}$ | 2.1272056 | $2.4\times10^{-5}$ |
| 1.2 | 3.1803141 | $3.7\times10^{-4}$ | 3.1799026 | $3.9\times10^{-5}$ |
| 1.6 | 4.2844424 | $9.6\times10^{-4}$ | 4.2834208 | $6.3\times10^{-5}$ |
| 2.0 | 5.3075082 | $2.0\times10^{-3}$ | 5.3053707 | $1.0\times10^{-4}$ |

The corrector cuts the error by ≈20 for one extra evaluation per step; PC4 is as accurate as RK4 at half the cost.

**Example 5.6.3 (deriving AB2).** Interpolate $f$ linearly through $(t_{i-1},f_{i-1})$, $(t_i,f_i)$ and integrate over $[t_i,t_{i+1}]$: $\int_{t_i}^{t_{i+1}}\bigl[f_i + \frac{f_i-f_{i-1}}{h}(t-t_i)\bigr]dt = hf_i + \frac h2(f_i - f_{i-1}) = \frac h2(3f_i - f_{i-1})$. On the model problem ($h=0.2$, exact $w_1$): $w(2) = 5.3992379$, error $0.094$ (second order, larger constant than modified Euler).

**Example 5.6.4 (an implicit method solved exactly).** For linear $y' = \lambda y$ the trapezoid (AM2) equation $w_{i+1} = w_i + \frac h2\lambda(w_{i+1}+w_i)$ can be solved: $w_{i+1} = \frac{1+h\lambda/2}{1-h\lambda/2}w_i$. For $\lambda = -2$, $h = 0.1$: $w(1) = (0.9/1.1)^{10} = 0.1344306$ vs $e^{-2} = 0.1353353$.

**Example 5.6.5 (orders of AB4 and PC4).** Errors at $t = 2$:

| $h$ | AB4 | PC4 |
|---|---|---|
| 0.2 | $2.04\times10^{-3}$ | $1.01\times10^{-4}$ |
| 0.1 | $1.85\times10^{-4}$ | $1.09\times10^{-5}$ |
| 0.05 | $1.37\times10^{-5}$ | $9.13\times10^{-7}$ |

Ratios ≈ 11–13 (approaching 16 as $h\to0$).

---

## 5.7 Higher-Order Equations and Systems of Differential Equations

An $m$-th order system $u_j' = f_j(t,u_1,\dots,u_m)$ is solved by applying any method **componentwise** (vectors instead of scalars). A higher-order equation $y^{(m)} = f(t,y,y',\dots,y^{(m-1)})$ becomes a system with $u_1 = y$, $u_2 = y'$, …, $u_m = y^{(m-1)}$:
$$
u_1' = u_2,\ \ \dots,\ \ u_{m-1}' = u_m,\ \ u_m' = f(t,u_1,\dots,u_m).
$$
Theorem 5.17 (existence/uniqueness) carries over with a Lipschitz condition in all $u_j$.

### Examples for §5.7

**Example 5.7.1 (damped oscillator).** $y'' + 0.5y' + 4y = 0$, $y(0) = 1$, $y'(0) = 0$ ⇒ $u' = (u_2,\ -4u_1 - 0.5u_2)$. Exact $y = e^{-t/4}\bigl(\cos\omega t + \frac{0.25}{\omega}\sin\omega t\bigr)$, $\omega = \sqrt{3.9375}$. RK4 with $h = 0.1$: $y(5) = -0.2691040$ vs exact $-0.2690750$.

**Example 5.7.2 (predator–prey).** Lotka–Volterra $x' = x - 0.1xz$, $z' = -1.5z + 0.075xz$, $x(0) = 10$, $z(0) = 5$. RK4 ($h = 0.01$, $t\in[0,15]$): prey oscillates between $7.93$ and $40.60$, predators between $3.09$ and $23.28$. The conserved quantity $V = 0.075x - 1.5\ln x + 0.1z - \ln z$ drifts by only $1.4\times10^{-10}$.

**Example 5.7.3 (SIR epidemic).** $S' = -\beta SI$, $I' = \beta SI-\gamma I$, $R' = \gamma I$ with $\beta = 0.3$, $\gamma = 0.1$ ($R_0 = 3$), $S(0) = 0.99$, $I(0) = 0.01$. RK4 ($h = 0.5$): infections peak on **day 26.5** at $I = 30.4\%$; finally $S_\infty = 0.0588$. Check: the final-size relation $S_\infty = S_0e^{-R_0(1-S_\infty)}$ (solved by root finding, Chapter 2) gives $0.0587974$.

**Example 5.7.4 (long-time behaviour).** Harmonic oscillator $y'' = -y$ over 10 periods ($t = 20\pi$) with $h = 0.1$: energy $\frac12(y^2+y'^2)$ should stay $0.5$. Euler: $257.8$ (energy grows by the factor $(1+h^2)$ per step). RK4: $0.499996$. For very long simulations (e.g. Hamiltonian Monte Carlo) one uses *symplectic* methods such as leapfrog.

**Example 5.7.5 (third-order equation).** $y''' = 6$, $y(0) = 1$, $y'(0) = y''(0) = 0$ (exact $1+t^3$) becomes $u' = (u_2, u_3, 6)$; RK4 gives $y(1) = 2.0000000$ (exact, since the solution is a cubic).

---

## 5.8 Stability

**Definitions.** A one-step method is **consistent** if $\lim_{h\to0}\max_i|\tau_i(h)| = 0$, **convergent** if $\lim_{h\to0}\max_i|y(t_i)-w_i| = 0$, and **stable** if small changes in initial data produce small changes in the result.

**Theorem 5.20.** A one-step method $w_{i+1} = w_i + h\phi(t_i,w_i,h)$ with $\phi$ Lipschitz in $w$ is stable, and it is convergent iff it is consistent ($\phi(t,y,0) = f(t,y)$).

**Multistep methods — the root condition.** The **characteristic polynomial** of the method is $P(\lambda) = \lambda^m - a_{m-1}\lambda^{m-1} - \cdots - a_0$. Consistency implies $P(1) = 0$.
* **Root condition:** all roots satisfy $|\lambda_i|\le1$, and roots with $|\lambda_i| = 1$ are simple.
* **Strongly stable:** $\lambda = 1$ is the only root of modulus 1. **Weakly stable:** other simple roots of modulus 1.
* **Theorem 5.24 (Dahlquist):** a consistent multistep method is convergent **iff** it satisfies the root condition.

**Absolute stability.** Apply a method to the test equation $y' = \lambda y$, $\operatorname{Re}\lambda<0$ (whose solution decays). A one-step method gives $w_{i+1} = Q(h\lambda)w_i$; the **region of absolute stability** is $\{z\in\mathbb C: |Q(z)|<1\}$:
* Euler: $Q = 1+z$ (disc of radius 1 centred at $-1$; real interval $(-2,0)$);
* RK4: $Q = 1+z+\frac{z^2}2+\frac{z^3}6+\frac{z^4}{24}$ (real interval $(-2.785, 0)$);
* implicit Euler: $Q = \frac1{1-z}$ (entire left half-plane: **A-stable**);
* trapezoid: $Q = \frac{1+z/2}{1-z/2}$ (A-stable).

### Examples for §5.8

**Example 5.8.1 (consistency of Euler).** $\phi(t,w,h) = f(t,w)$, so $\phi(t,y,0) = f(t,y)$: consistent; and $\phi$ is Lipschitz with the same $L$ as $f$: stable. Hence convergent (Theorem 5.20).

**Example 5.8.2 (AB4 is strongly stable).** $w_{i+1} = w_i + \ldots$ has $P(\lambda) = \lambda^4 - \lambda^3 = \lambda^3(\lambda-1)$: roots $1,0,0,0$. Strongly stable.

**Example 5.8.3 (a consistent but unstable method).** The explicit two-step method
$$
w_{i+1} = -4w_i + 5w_{i-1} + h(4f_i + 2f_{i-1})
$$
has local truncation error $O(h^3)$ — *more* accurate than AB2! But $P(\lambda) = \lambda^2+4\lambda-5 = (\lambda-1)(\lambda+5)$ has a root $-5$. For $y' = -y$, $h = 0.1$ with exact starting values: $w(0.5) = 0.6082$ (exact $0.6065$), $w(1) = -6.68$, $w(2) = -1.2\times10^8$, $w(3) = -2.2\times10^{15}$. Errors are multiplied by $\approx-5$ every step. **Accuracy without stability is worthless.**

**Example 5.8.4 (weak stability: the leapfrog/Milne method).** $w_{i+1} = w_{i-1} + 2hf_i$ has $P = \lambda^2 - 1$ (roots $\pm1$: weakly stable). For $y' = -y$, $h = 0.1$ the roots of $\lambda^2 + 2h\lambda - 1$ are $0.905$ (≈$e^{-h}$) and $-1.105$ (parasitic). Results: $w(5) = 0.0178$ (exact $0.0067$), $w(10) = 1.62$, $w(20) = 3.5\times10^4$ (exact $2\times10^{-9}$). The parasitic root eventually dominates.

**Example 5.8.5 (stability intervals).** On the negative real axis: Euler and modified Euler are stable for $h\lambda\in(-2,0)$; RK4 for $h\lambda\in(-2.785, 0)$. So for $y' = -100y$, RK4 needs $h<0.0279$ just for stability, regardless of accuracy.

---

## 5.9 Stiff Differential Equations

A problem is **stiff** when its Jacobian has eigenvalues with large negative real parts (fast transients) alongside the slow dynamics of interest. The **stiffness ratio** $\max|\operatorname{Re}\lambda|/\min|\operatorname{Re}\lambda|$ can be $10^3$–$10^{10}$ (chemical kinetics, circuits, pharmacokinetics, epidemic models with fast and slow compartments). Explicit methods are then limited by *stability*, not accuracy.

**Implicit (backward) Euler:** $w_{i+1} = w_i + hf(t_{i+1},w_{i+1})$ — A-stable, first order. Each step solves a nonlinear equation by Newton's method (Chapter 2/10) with Jacobian $I - h\,\partial f/\partial y$.
**Trapezoidal method:** $w_{i+1} = w_i + \frac h2[f(t_{i+1},w_{i+1}) + f(t_i,w_i)]$ — A-stable, second order, but $Q(z)\to-1$ as $z\to-\infty$ (fast components are damped slowly and oscillate; not "L-stable").
Production codes: BDF (Gear) methods, Radau IIA, LSODA (automatic stiffness switching).

### Examples for §5.9

**Example 5.9.1 (the prototype).** $y' = -50y$, $y(0) = 1$, $h = 0.05$ ($h\lambda = -2.5$):
Euler $w_i = (-1.5)^i$: $1,\ -1.5,\ 2.25,\ -3.375,\ 5.06,\ -7.59,\ldots$ (explodes);
implicit Euler $w_i = (1/3.5)^i$: $1,\ 0.2857,\ 0.0816,\ 0.0233,\ 0.0067,\ldots$ (correct decay).

**Example 5.9.2 (a stiff problem with a smooth solution).** $y' = -1000(y - \cos t) - \sin t$, $y(0) = 1$, exact $y = \cos t$ — the solution is as smooth as can be, yet:

| method | $h$ | result |
|---|---|---|
| RK4 | 0.0025 | $y(0.2) = 0.980064$ ✓ |
| RK4 | 0.003 | $y(0.2) = -3073$ ✗ |
| RK4 | 0.01 | $y(0.2) = -6.8\times10^{43}$ ✗ |
| implicit Euler | 0.01 | $y(1) = 0.540300$ (error $2.7\times10^{-6}$) |
| implicit Euler | 0.1 | $y(1) = 0.540274$ (error $2.8\times10^{-5}$) |

The RK4 threshold is $h<2.785/1000 = 0.0028$, exactly as predicted by §5.8.

**Example 5.9.3 (Robertson's chemical kinetics).** $y_1' = -0.04y_1 + 10^4y_2y_3$, $y_2' = 0.04y_1 - 10^4y_2y_3 - 3\times10^7y_2^2$, $y_3' = 3\times10^7y_2^2$ on $[0,100]$:

| solver | function evaluations | steps |
|---|---|---|
| RK45 (explicit) | 729 668 | 104 142 |
| BDF (implicit) | 431 | 163 |
| Radau (implicit) | 726 | 88 |

All give $y_1(100) = 0.617235$; the explicit solver is ~1000× more expensive.

**Example 5.9.4 (A-stable is not the end of the story).** $\mathbf y' = \operatorname{diag}(-1,-1000)\,\mathbf y$, $\mathbf y(0) = (1,1)$, integrated to $t=2$:
* Euler, $h = 0.1$: second component $8\times10^{39}$ (unstable);
* trapezoid, $h = 0.1$: $(0.13511, 0.4493)$ — stable, but the fast component, which should be $\approx0$, is damped only by $|\frac{1-50}{1+50}| = 0.96$ per step and oscillates in sign;
* trapezoid, $h=0.5$: $(0.1296, 0.9685)$.
Implicit Euler ($Q(z)\to0$ as $z\to-\infty$) kills such components immediately — this property (L-stability) is why BDF methods dominate stiff computation.

**Example 5.9.5 (stiffness ratio).** For the system above the ratio is $1000/1 = 1000$: Euler needs $h<0.002$ for stability, while accuracy for the slow component $e^{-t}$ would allow $h\approx0.1$.

---

## 5.10 Differential Equations in Data Science *(data-science extension)*

* **Mechanistic models + data:** fit ODE parameters by nonlinear least squares (Chapter 13): each residual evaluation solves an IVP.
* **Optimisation as ODE discretisation:** gradient descent = Euler on $w' = -\nabla F(w)$; momentum = discretised damped oscillator $w'' + \gamma w' + \nabla F(w) = 0$; stability limits = learning-rate limits.
* **Neural ODEs / ResNets:** $h_{k+1} = h_k + \Delta t\,f_\theta(h_k)$ is Euler's method; neural ODEs replace it with adaptive solvers.
* **Diffusion models / score-based generative models** integrate reverse-time SDEs/ODEs; **Hamiltonian Monte Carlo** integrates Hamilton's equations with leapfrog.

### Examples for §5.10

**Example 5.10.1 (fitting an SIR model).** Synthetic prevalence data every 3 days (true $\beta = 0.35$, $\gamma = 0.12$, 5% noise). Minimising $\sum_k(I_{\beta,\gamma}(t_k) - I_k^{\text{obs}})^2$ with `scipy.optimize.least_squares` (each evaluation solves the SIR system) gives $\hat\beta = 0.3557$, $\hat\gamma = 0.1232$, $\hat R_0 = 2.887$ (true $2.917$).

**Example 5.10.2 (learning-rate limit = stability limit).** Gradient descent on $F = \frac12(w_1^2+10w_2^2)$ is Euler on $w' = -\operatorname{diag}(1,10)w$, stable iff $\alpha<2/10 = 0.2$. After 50 steps from $(1,1)$: $\alpha = 0.05$: $\|w\| = 0.077$; $\alpha = 0.19$: $0.0052$; $\alpha = 0.21$: $117$ (diverged).

**Example 5.10.3 (growth-curve forecasting).** Cumulative adoptions over 13 weeks fitted by the logistic solution $y = K/(1+(K/y_0-1)e^{-rt})$ ($y_0 = 20$): $\hat K = 992.6$, $\hat r = 0.596$; forecast at week 20: $992$ (saturation).

**Example 5.10.4 (ResNet = Euler).** Ten residual blocks $x_{k+1} = x_k + 0.1\tanh(Wx_k)$ with $W = \begin{pmatrix}0&-1\\1&0\end{pmatrix}$ map $(1,0)\mapsto(0.6886, 0.7201)$; the underlying ODE $x' = \tanh(Wx)$ integrated by RK4 to $t = 1$ gives $(0.6612, 0.7036)$. The network is a (first-order) discretisation of the flow.

**Example 5.10.5 (momentum as a damped oscillator).** On $F = \frac12(w_1^2+10w_2^2)$ with $\alpha = 0.02$, 100 iterations from $(1,1)$: plain GD reaches $\|w\| = 0.133$; heavy-ball momentum ($\beta = 0.9$) reaches $0.0051$. Momentum discretises $w'' + \gamma w' + \nabla F = 0$, whose solution decays at a rate governed by $\sqrt{\lambda_{\min}}$ rather than $\lambda_{\min}$.

---

## Chapter summary

| Method | Order | $f$-evals/step | Stability | Use |
|---|---|---|---|---|
| Euler | 1 | 1 | $(-2,0)$ | teaching, GD |
| Taylor $n$ | $n$ | derivatives | — | with AD |
| Midpoint / modified Euler | 2 | 2 | $(-2,0)$ | cheap |
| RK4 | 4 | 4 | $(-2.785,0)$ | default fixed-step |
| RKF45 / DOPRI | 4(5) | 6 | similar | default adaptive |
| AB4 / Adams PC | 4 | 1 / 2 | small | smooth, long runs |
| Implicit Euler, BDF | 1–5 | Newton solve | A-/L-stable | stiff |
| Trapezoid | 2 | Newton solve | A-stable | stiff, linear |

## Further reading

Burden & Faires Ch. 5 · Hairer, Nørsett & Wanner, *Solving ODEs I & II* · Chen et al., "Neural ordinary differential equations", NeurIPS 2018 · Su, Boyd & Candès, "A differential equation for modeling Nesterov's accelerated gradient method", JMLR 2016.
