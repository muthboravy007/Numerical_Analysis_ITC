# Chapter 3 — Interpolation and Polynomial Approximation

> **Reference:** Burden & Faires, Chapter 3 (§3.1–3.6) plus §3.7 for data science.
> **Code:** [`numlib/interpolation.py`](../numlib/interpolation.py) · **All numbers reproduced by** [`lectures/code/ch03.py`](code/ch03.py)
> **Worked problems:** [`examples/ch03`](../examples/ch03_examples.md) · **Homework:** [`exercises/ch03`](../exercises/ch03_exercises.md)

## Learning outcomes

1. Construct the interpolating polynomial in Lagrange, Neville, Newton (divided-difference) and barycentric forms, and state its existence, uniqueness and error formula.
2. Use forward/backward difference formulas for equally spaced data.
3. Construct Hermite (osculating) polynomials that match derivatives.
4. Derive and solve the tridiagonal systems for natural and clamped cubic splines, and state their error bounds.
5. Build parametric curves (Hermite and Bézier) for 2-D data.
6. Explain Runge's phenomenon, Chebyshev nodes and the Lebesgue constant; decide between interpolation, splines and regression for real data.

**Why polynomials?** They are easy to evaluate, differentiate and integrate, and they can approximate any continuous function:

**Theorem 3.1 (Weierstrass Approximation Theorem).** If $f\in C[a,b]$, then for every $\varepsilon>0$ there is a polynomial $P$ with $|f(x)-P(x)|<\varepsilon$ for all $x\in[a,b]$.

The theorem says *some* polynomial works; this chapter is about *constructing* good ones from data.

---

## 3.1 Interpolation and the Lagrange Polynomial

Taylor polynomials are accurate only near one point. Interpolation instead uses information spread over the interval.

**Problem.** Given $n+1$ distinct nodes $x_0,\dots,x_n$ and values $f(x_k)$, find $P\in\mathcal P_n$ (degree $\le n$) with $P(x_k) = f(x_k)$.

**Lagrange basis polynomials.**
$$
L_{n,k}(x) = \prod_{i\neq k}\frac{x-x_i}{x_k-x_i},\qquad L_{n,k}(x_j) = \delta_{jk}.
$$

**Theorem 3.2 (existence and uniqueness).** The unique $P\in\mathcal P_n$ interpolating $f$ at $x_0,\dots,x_n$ is
$$
P(x) = \sum_{k=0}^n f(x_k)L_{n,k}(x).
$$
*Uniqueness proof.* If $P,Q$ both interpolate, $P-Q\in\mathcal P_n$ has $n+1$ zeros, so $P-Q\equiv0$. ∎

**Theorem 3.3 (interpolation error).** If $x_0,\dots,x_n\in[a,b]$ are distinct and $f\in C^{n+1}[a,b]$, then for each $x\in[a,b]$ there is $\xi(x)\in(a,b)$ with
$$
f(x) - P(x) = \frac{f^{(n+1)}(\xi(x))}{(n+1)!}\,(x-x_0)(x-x_1)\cdots(x-x_n).
$$
*Proof.* For $x$ not a node, define $g(t) = f(t)-P(t) - [f(x)-P(x)]\prod_i\frac{t-x_i}{x-x_i}$. Then $g$ vanishes at the $n+2$ points $x_0,\dots,x_n,x$. By the generalised Rolle theorem, $g^{(n+1)}(\xi)=0$ for some $\xi$. Since $P^{(n+1)}\equiv0$ and the product is a polynomial with leading term $t^{n+1}/\prod(x-x_i)$,
$0 = f^{(n+1)}(\xi) - [f(x)-P(x)]\frac{(n+1)!}{\prod_i(x-x_i)}$. ∎

The error has two factors: the smoothness of $f$ and the **nodal polynomial** $\omega(x) = \prod(x-x_i)$, which depends only on where the nodes are.

### Examples for §3.1

**Example 3.1.1 (Taylor polynomials can fail badly).** For $f(x) = 1/x$ about $x_0 = 1$, $P_n(x) = \sum_{k=0}^n(-1)^k(x-1)^k$. At $x=3$ the value $f(3) = 1/3$ but
$P_0(3) = 1$, $P_1(3) = -1$, $P_2(3) = 3$, $P_3(3) = -5$, $P_4(3) = 11$, $P_5(3) = -21$, $P_6(3) = 43$, $P_7(3) = -85$ — the approximations **diverge** because $x=3$ is outside the radius of convergence ($|x-1|<1$). Interpolation uses information from the whole interval instead.

**Example 3.1.2 (linear interpolation).** Through $(2,4)$ and $(5,1)$:
$P(x) = 4\frac{x-5}{2-5} + 1\frac{x-2}{5-2} = -\frac43(x-5) + \frac13(x-2) = -x+6$. So $P(3) = 3$.

**Example 3.1.3 (quadratic interpolation).** Interpolate $f(x) = 1/x$ at $x_0 = 2$, $x_1 = 2.75$, $x_2 = 4$:
$$
L_0 = \frac{(x-2.75)(x-4)}{(2-2.75)(2-4)} = \tfrac23(x-2.75)(x-4),\quad
L_1 = -\tfrac{16}{15}(x-2)(x-4),\quad
L_2 = \tfrac25(x-2)(x-2.75).
$$
$P(x) = \tfrac12L_0 + \tfrac{4}{11}L_1 + \tfrac14L_2 = \tfrac{1}{22}x^2 - \tfrac{35}{88}x + \tfrac{49}{44}$.
$P(3) = 0.329545$ versus $1/3 = 0.333333$: error $3.79\times10^{-3}$.

**Example 3.1.4 (using the error formula).** Bound the error of Example 3.1.3 on $[2,4]$. $f'''(x) = -6x^{-4}$, so $\max|f'''| = 6/2^4 = 0.375$. The nodal polynomial $\omega(x) = (x-2)(x-2.75)(x-4)$ has $\max_{[2,4]}|\omega| = 0.5625$ (at $x = 3.5$). Hence
$$
|f(x)-P(x)|\le\frac{0.375}{3!}\times0.5625 = 0.0352 .
$$
The actual maximum error is $0.00736$: the bound is valid but pessimistic (it uses the worst $\xi$).

**Example 3.1.5 (designing a table).** A table of $e^x$ on $[0,1]$ with step $h$ is used with linear interpolation. How small must $h$ be for an error $\le10^{-6}$?
For $x\in[x_j,x_{j+1}]$: $|f-P|\le\frac{\max|f''|}{2}\max|(x-x_j)(x-x_{j+1})| = \frac{e}{2}\cdot\frac{h^2}{4} = \frac{eh^2}{8}$. Requiring $\frac{eh^2}{8}\le10^{-6}$ gives $h\le1.72\times10^{-3}$; e.g. $h = 0.001$ (1001 entries).

**Example 3.1.6 (node placement matters).** Approximate $\ln3$ ($=1.098612$) by quadratic interpolation.
* Nodes $1,2,4$: $P(3) = 1.155245$ (error $0.057$).
* Nodes $2, 2.5, 4$: $P(3) = 1.106196$ (error $0.0076$).
Nodes closer to the evaluation point make $|\omega(3)|$ smaller ($|\omega| = 2$ vs $0.5$) and put $\xi$ where $|f'''| = 2/x^3$ is smaller.

---

## 3.2 Data Approximation and Neville's Method

Often we want $P(x)$ at a *single* point, and we do not know in advance which degree is appropriate. **Neville's method** builds interpolants of increasing degree recursively.

**Notation.** $P_{m_1,\dots,m_k}$ is the polynomial interpolating $f$ at $x_{m_1},\dots,x_{m_k}$.

**Theorem 3.5.** For distinct nodes and $i\neq j$,
$$
P(x) = \frac{(x-x_j)\,P_{0,\dots,j-1,j+1,\dots,k}(x) - (x-x_i)\,P_{0,\dots,i-1,i+1,\dots,k}(x)}{x_i-x_j}
$$
interpolates $f$ at $x_0,\dots,x_k$.

Writing $Q_{i,j} = P_{i-j,\dots,i}$ (degree $j$ on the $j+1$ nodes ending at $x_i$):
$$
Q_{i,0} = f(x_i),\qquad Q_{i,j} = \frac{(x-x_{i-j})Q_{i,j-1} - (x-x_i)Q_{i-1,j-1}}{x_i-x_{i-j}}.
$$
Stop when two successive diagonal entries agree to the desired tolerance.

### Examples for §3.2

**Example 3.2.1 (Neville table).** Approximate $\ln2.5$ from a 7-digit table:

| $x_i$ | $Q_{i,0}$ | $Q_{i,1}$ | $Q_{i,2}$ | $Q_{i,3}$ | $Q_{i,4}$ |
|---|---|---|---|---|---|
| 2.0 | 0.6931472 | | | | |
| 2.2 | 0.7884574 | 0.9314227 | | | |
| 2.4 | 0.8754687 | 0.9189744 | 0.9158623 | | |
| 2.6 | 0.9555114 | 0.9154901 | 0.9163611 | 0.9162780 | |
| 2.8 | 1.0296194 | 0.9184574 | 0.9162319 | 0.9162965 | **0.9162896** |

True $\ln2.5 = 0.9162907$. The degree-4 value has error $1.1\times10^{-6}$ — about the size of the table's rounding error, so a higher degree would not help.

**Example 3.2.2 (order the nodes by distance).** Re-ordering the nodes as $2.4, 2.6, 2.2, 2.8, 2.0$ (closest first) gives diagonal entries $0.8754687,\ 0.9154901,\ 0.9163611,\ 0.9162965,\ 0.9162896$: already the linear interpolant $0.91549$ and the quadratic $0.91636$ are good, and the convergence of the diagonal can be used as a stopping test.

**Example 3.2.3 (reconstructing a missing value).** Suppose $f(0) = 1$, $f(1) = 2$ and the quadratic interpolant through $x = 0,1,2$ satisfies $P(1.5) = 3$. Find $f(2)$.
The basis values at $1.5$ are $L_0 = -0.125$, $L_1 = 0.75$, $L_2 = 0.375$. So $3 = -0.125(1) + 0.75(2) + 0.375f(2)$ and $f(2) = 13/3 = 4.3333$.

**Example 3.2.4 (sensor data).** Temperatures ($^\circ$C) at hours $0,3,6,9,12$: $24.1, 22.8, 25.3, 30.2, 33.0$. Estimate $T(7.5)$ with nodes ordered $6,9,3,12,0$:
$Q$: $25.3$; $27.75$ (linear); $27.45$ (quadratic); $27.73$ (cubic); $27.66$ (quartic).
The estimates fluctuate by $\pm0.3^\circ$C, the scale of real measurement noise — a sign that high-degree interpolation of noisy data is not meaningful (see §3.7).

**Example 3.2.5 (inverse interpolation — root finding from a table).** To solve $g(x) = x - e^{-x} = 0$ from the table $x = 0.5, 0.55, 0.6, 0.65$ with $g = -0.10653, -0.02695, 0.05119, 0.12795$, interpolate $x$ **as a function of $g$** and evaluate at $g = 0$ using Neville:
$0.5669323$ (linear), $0.5671435$ (quadratic), $0.5671433$ (cubic). True root $0.5671432904$. This works because $g$ is monotone (so $g^{-1}$ exists and is smooth).

---

## 3.3 Divided Differences

The **Newton form** is ideal when nodes are added one at a time:
$$
P_n(x) = f[x_0] + \sum_{k=1}^n f[x_0,\dots,x_k]\,(x-x_0)\cdots(x-x_{k-1}),
$$
with divided differences $f[x_i] = f(x_i)$ and
$$
f[x_i,\dots,x_{i+k}] = \frac{f[x_{i+1},\dots,x_{i+k}] - f[x_i,\dots,x_{i+k-1}]}{x_{i+k}-x_i}.
$$
Adding a node $x_{n+1}$ just adds one term. Evaluation uses nested multiplication (like Horner) in $O(n)$.

**Theorem 3.6.** If $f\in C^n[a,b]$ and $x_0,\dots,x_n\in[a,b]$ are distinct, there is $\xi\in(a,b)$ with
$$
f[x_0,\dots,x_n] = \frac{f^{(n)}(\xi)}{n!}.
$$
(Divided differences are scaled derivatives — the basis of numerical differentiation.)

**Equally spaced nodes.** With $x_i = x_0 + ih$ and $x = x_0 + sh$, forward differences $\Delta f_i = f_{i+1}-f_i$, $\Delta^kf_i = \Delta(\Delta^{k-1}f_i)$ satisfy $f[x_0,\dots,x_k] = \frac{\Delta^kf_0}{k!h^k}$, giving the **Newton forward-difference formula**
$$
P_n(x_0+sh) = \sum_{k=0}^n\binom{s}{k}\Delta^kf_0,\qquad \binom sk = \frac{s(s-1)\cdots(s-k+1)}{k!}.
$$
Near the end of the table use the **backward-difference formula** with $\nabla f_i = f_i - f_{i-1}$ and $x = x_n + sh$:
$$
P_n(x_n+sh) = \sum_{k=0}^n(-1)^k\binom{-s}{k}\nabla^kf_n = f_n + s\nabla f_n + \frac{s(s+1)}{2}\nabla^2f_n + \cdots
$$

### Examples for §3.3

**Example 3.3.1 (recovering a cubic).** $x = 0,1,2,3$, $f = 1,2,9,28$:

| $x_i$ | $f[x_i]$ | first | second | third |
|---|---|---|---|---|
| 0 | **1** | | | |
| | | **1** | | |
| 1 | 2 | | **3** | |
| | | 7 | | **1** |
| 2 | 9 | | 6 | |
| | | 19 | | |
| 3 | 28 | | | |

$P(x) = 1 + x + 3x(x-1) + x(x-1)(x-2) = x^3+1$. By Theorem 3.6, the third divided difference equals $f'''/3! = 6/6 = 1$ exactly.

**Example 3.3.2.** Data $x = 1,2,3,4$, $f = 2,3,5,4$: the top diagonal is $2,\ 1,\ 0.5,\ -2/3$, so
$P(x) = 2 + (x-1) + 0.5(x-1)(x-2) - \frac23(x-1)(x-2)(x-3)$ and $P(2.5) = 2 + 1.5 + 0.5(1.5)(0.5) - \frac23(1.5)(0.5)(-0.5) = 4.125$.

**Example 3.3.3 (forward-difference formula).** $f = \sin x$ at $0, 0.2, 0.4, 0.6$ ($h = 0.2$): $\Delta f_0 = 0.1986693$, $\Delta^2f_0 = -0.0079203$, $\Delta^3f_0 = -0.0076046$. For $x = 0.1$, $s = 0.5$:
$$
P(0.1) = 0 + 0.5(0.1986693) + \tfrac{0.5(-0.5)}{2}(-0.0079203) + \tfrac{0.5(-0.5)(-1.5)}{6}(-0.0076046) = 0.0998494,
$$
error $1.6\times10^{-5}$ (true $\sin0.1 = 0.0998334$).

**Example 3.3.4 (backward-difference formula).** Same table, $x = 0.55 = 0.6 + sh$ with $s = -0.25$: using $\nabla f_3 = 0.1752241$, $\nabla^2f_3 = -0.0155249$, $\nabla^3f_3 = -0.0076046$, $P(0.55) = 0.5227078$ (true $0.5226872$, error $2.1\times10^{-5}$).

**Example 3.3.5 (adding a data point).** Append $(5,6)$ to Example 3.3.2. Only one new coefficient is needed: $f[x_0,\dots,x_4] = 0.4166667$, so
$P_4(x) = P_3(x) + 0.4166667(x-1)(x-2)(x-3)(x-4)$ and $P_4(2.5) = 4.125 + 0.4166667(1.5)(0.5)(-0.5)(-1.5) = 4.359375$.

**Example 3.3.6 (divided difference ≈ derivative).** For $e^x$ at $0, 0.1, 0.2, 0.3$: $f[x_0,\dots,x_3] = 0.1938812$. Theorem 3.6 says this equals $e^\xi/6$ for some $\xi\in(0,0.3)$; indeed $e^{\xi}/6 = 0.1938812$ gives $\xi = 0.1512$ — near the middle of the interval.

---

## 3.4 Hermite Interpolation

Sometimes we know derivatives as well as values (e.g. position *and* velocity). An **osculating polynomial** matches $f$ and its derivatives; the most important case matches $f$ and $f'$ at each node.

**Theorem 3.9.** If $f\in C^1[a,b]$ and $x_0,\dots,x_n$ are distinct, the unique polynomial of degree $\le2n+1$ with $H(x_i) = f(x_i)$ and $H'(x_i) = f'(x_i)$ is
$$
H_{2n+1}(x) = \sum_{j=0}^n f(x_j)H_{n,j}(x) + \sum_{j=0}^nf'(x_j)\hat H_{n,j}(x),
$$
$H_{n,j}(x) = [1-2(x-x_j)L'_{n,j}(x_j)]L_{n,j}(x)^2$, $\hat H_{n,j}(x) = (x-x_j)L_{n,j}(x)^2$. If $f\in C^{2n+2}$,
$$
f(x) - H_{2n+1}(x) = \frac{f^{(2n+2)}(\xi)}{(2n+2)!}(x-x_0)^2\cdots(x-x_n)^2 .
$$

**Divided-difference construction (the practical way).** List each node twice, $z_{2i} = z_{2i+1} = x_i$, and use $f[z_{2i},z_{2i+1}] := f'(x_i)$ in place of the undefined $0/0$. Then
$$
H_{2n+1}(x) = f[z_0] + \sum_{k=1}^{2n+1}f[z_0,\dots,z_k](x-z_0)\cdots(x-z_{k-1}).
$$

### Examples for §3.4

**Example 3.4.1 (a cubic Hermite polynomial).** Data: $f(0) = 1$, $f'(0) = 0$, $f(1) = 2$, $f'(1) = 3$. With $z = 0,0,1,1$:
$f[z_0,z_1] = 0$, $f[z_1,z_2] = 1$, $f[z_2,z_3] = 3$; $f[z_0,z_1,z_2] = 1$, $f[z_1,z_2,z_3] = 2$; $f[z_0,\dots,z_3] = 1$.
$H_3(x) = 1 + 0\cdot x + 1\cdot x^2 + 1\cdot x^2(x-1) = x^3+1$.

**Example 3.4.2 (Hermite for $\sin$).** Match $\sin$ and $\cos$ at $0$ and $\pi/2$ ($f = 0, 1$; $f' = 1, 0$). The coefficients are $0,\ 1,\ -0.231335,\ -0.110740$:
$H_3(x) = x - 0.231335x^2 - 0.110740x^2(x-\tfrac\pi2)$.
$H_3(\pi/4) = 0.696350$ vs $\sin(\pi/4) = 0.707107$ (error $0.0108$). Error bound: $|f^{(4)}|\le1$ and $\max x^2(x-\frac\pi2)^2 = (\pi/4)^4$, so $|\text{error}|\le\frac{(\pi/4)^4}{24} = 0.0159$; the actual maximum error on $[0,\pi/2]$ is $0.0108$.

**Example 3.4.3 (degree-5 Hermite by divided differences).** $f = \ln x$ at $1, 1.5, 2$ with $f' = 1/x$:

| $z$ | $f[z]$ | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|---|
| 1.0 | 0 | | | | | |
| 1.0 | 0 | **1** | | | | |
| 1.5 | 0.4054651 | 0.8109302 | −0.3781396 | | | |
| 1.5 | 0.4054651 | **0.6666667** | −0.2885271 | 0.1792249 | | |
| 2.0 | 0.6931472 | 0.5753641 | −0.1826050 | 0.1059221 | −0.0733029 | |
| 2.0 | 0.6931472 | **0.5** | −0.1507283 | 0.0637535 | −0.0421685 | 0.0311343 |

(bold = derivative data). $H_5(1.25) = 0.2231883$ vs $\ln1.25 = 0.2231436$: error $4.5\times10^{-5}$. The quadratic Lagrange interpolant through the same three points gives $0.2174554$ (error $5.7\times10^{-3}$): the derivative information is worth two orders of magnitude.

**Example 3.4.4 (error bound for Example 3.4.3).** $f^{(6)}(x) = -120/x^6$, $\max_{[1,2]}|f^{(6)}| = 120$. At $x = 1.25$:
$|f-H_5|\le\frac{120}{720}\bigl[(0.25)(0.25)(0.75)\bigr]^2 = 3.66\times10^{-4}$. The actual error $4.5\times10^{-5}$ is well inside.

**Example 3.4.5 (position and velocity data).** A car's position $d$ (m) and speed $v$ (m/s) are logged at $t = 0, 5, 10$ s: $d = 0, 90, 220$, $v = 15, 20, 28$. The degree-5 Hermite interpolant predicts $d(8) = 161.78$ m and, by differentiating $H_5$, a speed of $28.18$ m/s at $t=8$. (Note: $28.18>v(10) = 28$ — polynomial interpolants can overshoot; a physically constrained model or monotone spline may be preferable.)

---

## 3.5 Cubic Spline Interpolation

High-degree polynomials oscillate. **Piecewise** polynomials avoid this. The most popular is the cubic spline.

**Definition 3.10.** Given $a = x_0<x_1<\dots<x_n = b$, a **cubic spline interpolant** $S$ of $f$ satisfies:
(a) on each $[x_j,x_{j+1}]$, $S = S_j$ is a cubic;
(b) $S_j(x_j) = f(x_j)$ and $S_j(x_{j+1}) = f(x_{j+1})$;
(c) $S_{j+1}'(x_{j+1}) = S_j'(x_{j+1})$; (d) $S_{j+1}''(x_{j+1}) = S_j''(x_{j+1})$;
(e) one set of boundary conditions:
* **natural (free):** $S''(x_0) = S''(x_n) = 0$;
* **clamped:** $S'(x_0) = f'(x_0)$, $S'(x_n) = f'(x_n)$.

**Counting.** $4n$ coefficients; (b) gives $2n$ equations, (c)–(d) give $2(n-1)$, (e) gives 2. Total $4n$. ✓

**Derivation (second-derivative form).** Let $M_j = S''(x_j)$ and $h_j = x_{j+1}-x_j$. $S''$ is linear on each interval, so integrating twice and imposing (b):
$$
S_j(x) = M_j\frac{(x_{j+1}-x)^3}{6h_j} + M_{j+1}\frac{(x-x_j)^3}{6h_j} + \Bigl(\frac{f_j}{h_j}-\frac{M_jh_j}{6}\Bigr)(x_{j+1}-x) + \Bigl(\frac{f_{j+1}}{h_j}-\frac{M_{j+1}h_j}{6}\Bigr)(x-x_j).
$$
Continuity of $S'$ at interior nodes gives, for $j = 1,\dots,n-1$,
$$
h_{j-1}M_{j-1} + 2(h_{j-1}+h_j)M_j + h_jM_{j+1} = 6\Bigl(\frac{f_{j+1}-f_j}{h_j} - \frac{f_j-f_{j-1}}{h_{j-1}}\Bigr).
$$
* Natural: add $M_0 = M_n = 0$.
* Clamped: add $2h_0M_0 + h_0M_1 = 6\bigl(\frac{f_1-f_0}{h_0} - f'(x_0)\bigr)$ and $h_{n-1}M_{n-1} + 2h_{n-1}M_n = 6\bigl(f'(x_n) - \frac{f_n-f_{n-1}}{h_{n-1}}\bigr)$.

**Theorem 3.11/3.12.** Both systems are **strictly diagonally dominant** and tridiagonal, hence have unique solutions computable in $O(n)$ operations (Thomas/Crout algorithm, Chapter 6).

B&F write $S_j(x) = a_j + b_j(x-x_j) + c_j(x-x_j)^2 + d_j(x-x_j)^3$; the conversion is
$a_j = f_j$, $c_j = M_j/2$, $d_j = \frac{M_{j+1}-M_j}{6h_j}$, $b_j = \frac{f_{j+1}-f_j}{h_j} - \frac{h_j(2M_j+M_{j+1})}{6}$.

**Theorem 3.13 (error of the clamped spline).** If $f\in C^4[a,b]$ with $|f^{(4)}|\le M$ and $h = \max h_j$, then
$$
\max_{a\le x\le b}|f(x)-S(x)|\le\frac{5M}{384}h^4 .
$$
**Theorem (minimum curvature).** Among all $C^2$ interpolants $g$ of the data, the natural spline minimises $\int_a^b[g''(x)]^2dx$ — it is the "smoothest" interpolant (the physical draughtsman's spline).

### Examples for §3.5

**Example 3.5.1 (a natural spline by hand).** Data $(0,0),(1,1),(2,0),(3,1)$, $h = 1$. Interior equations:
$$
\begin{aligned}
4M_1 + M_2 &= 6[(0-1) - (1-0)] = -12,\\
M_1 + 4M_2 &= 6[(1-0) - (0-1)] = 12,
\end{aligned}
$$
so $M_1 = -4$, $M_2 = 4$ ($M_0 = M_3 = 0$). In B&F form:

| $j$ | $a_j$ | $b_j$ | $c_j$ | $d_j$ |
|---|---|---|---|---|
| 0 | 0 | 5/3 | 0 | −2/3 |
| 1 | 1 | −1/3 | −2 | 4/3 |
| 2 | 0 | −1/3 | 2 | −2/3 |

$S(0.5) = 0.75$, $S(1.5) = 0.5$.

**Example 3.5.2 (natural vs clamped).** $f = \cos x$ on $[0,\pi]$ with 5 equally spaced nodes. The natural spline forces $S''(0) = S''(\pi) = 0$, but $\cos''(0) = -1\neq0$; the clamped spline uses the correct slopes $f'(0) = f'(\pi) = 0$.
* $M$ natural: $0,\ -1.0072,\ 0,\ 1.0072,\ 0$; max error $0.0334$.
* $M$ clamped: $-1.0524,\ -0.7441,\ 0,\ 0.7441,\ 1.0524$ (close to $-\cos x_j$); max error $0.00107$.
Wrong boundary conditions pollute accuracy near the ends. For $\sin$ on the same nodes (where $\sin'' = 0$ at both ends, so natural conditions are *correct*), integrating the spline gives $\int_0^\pi S = 1.99869$ (natural) and $1.99893$ (clamped) vs the exact $2$.

**Example 3.5.3 (splines vs polynomials on Runge's function).** $f(x) = \frac{1}{1+x^2}$ on $[-5,5]$, 11 equally spaced nodes:

| interpolant | max error |
|---|---|
| degree-10 polynomial | 1.916 |
| piecewise linear | 0.0674 |
| natural cubic spline | 0.0220 |
| clamped cubic spline | 0.0220 |

**Example 3.5.4 (verifying the $O(h^4)$ rate).** Clamped spline for $\sin$ on $[0,\pi]$:

| $n$ | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| max error | $4.34\times10^{-4}$ | $2.57\times10^{-5}$ | $1.59\times10^{-6}$ | $9.92\times10^{-8}$ |
| ratio | | 16.9 | 16.1 | 16.0 |

Halving $h$ divides the error by $2^4 = 16$, as Theorem 3.13 predicts (bound for $n=10$: $\frac{5}{384}(\pi/10)^4 = 1.27\times10^{-4}$).

**Example 3.5.5 (filling gaps in a time series).** Daily values $20 + 5\sin(2\pi d/14) + 0.5d$ for $d = 0,\dots,14$ with days $4,5,6,10$ missing. Reconstructed values:

| day | true | linear | natural spline |
|---|---|---|---|
| 4 | 26.875 | 25.656 | 26.777 |
| 5 | 26.409 | 24.937 | 26.261 |
| 6 | 25.169 | 24.219 | 25.083 |
| 10 | 20.125 | 20.608 | 20.140 |

RMSE: linear $1.094$, spline $0.099$. This is `pandas.Series.interpolate(method="cubicspline")`.

---

## 3.6 Parametric Curves

To represent curves that are not graphs of functions (loops, vertical tangents, handwriting, GPS tracks), interpolate $x(t)$ and $y(t)$ separately in a parameter $t$.

**Cubic Hermite segment with guidepoints.** Endpoints $(x_0,y_0),(x_1,y_1)$ and guidepoints $(x_0+\alpha_0, y_0+\beta_0)$, $(x_1-\alpha_1,y_1-\beta_1)$ define
$x(t) = [2(x_0-x_1)+3(\alpha_0+\alpha_1)]t^3 + [3(x_1-x_0) - 3(\alpha_1+2\alpha_0)]t^2 + 3\alpha_0t + x_0$ (and similarly $y$), which is exactly a **cubic Bézier curve** with control points $P_0 = (x_0,y_0)$, $P_1 = P_0 + (\alpha_0,\beta_0)$, $P_2 = P_3 - (\alpha_1,\beta_1)$, $P_3 = (x_1,y_1)$:
$$
B(t) = (1-t)^3P_0 + 3t(1-t)^2P_1 + 3t^2(1-t)P_2 + t^3P_3,\qquad 0\le t\le1.
$$
Properties: $B(0) = P_0$, $B(1) = P_3$, $B'(0) = 3(P_1-P_0)$, $B'(1) = 3(P_3-P_2)$, and the curve lies in the convex hull of the control points.

**de Casteljau's algorithm** evaluates $B(t)$ stably by repeated linear interpolation between control points.

### Examples for §3.6

**Example 3.6.1 (evaluating a Bézier curve).** Control points $P_0 = (0,0)$, $P_1 = (1,2)$, $P_2 = (3,3)$, $P_3 = (4,0)$:
$B(0.5) = \frac18P_0 + \frac38P_1 + \frac38P_2 + \frac18P_3 = (2, 1.875)$, and $B(0.25) = (0.90625, 1.265625)$.

**Example 3.6.2 (de Casteljau).** At $t = \frac12$: level 1 midpoints $(0.5,1), (2,2.5), (3.5,1.5)$; level 2: $(1.25,1.75), (2.75,2)$; level 3: $(2, 1.875) = B(0.5)$ ✓. The level-1..3 points also give the control points of the two halves (subdivision), used in font rendering.

**Example 3.6.3 (power form).** Expanding Example 3.6.1: $x(t) = -2t^3 + 3t^2 + 3t$, $y(t) = -3t^3 - 3t^2 + 6t$. Check: $x'(0) = 3 = 3(P_1-P_0)_x$ ✓; $y'(1) = -9-6+6 = -9 = 3(P_3 - P_2)_y$ ✓.

**Example 3.6.4 (closed curves).** Interpolating 5 points on the unit circle ($\theta = 0,\frac\pi2,\pi,\frac{3\pi}2,2\pi$) by *periodic* splines $x(\theta), y(\theta)$ gives a closed smooth curve whose radius deviates from $1$ by at most $0.028$.

**Example 3.6.5 (a GPS path).** Points $(0,0),(1,1.5),(2.5,1.8),(3,0.5),(4.2,0.9)$. Using **chord-length parametrisation** ($t_0 = 0$, $t_{k} = t_{k-1} + \|P_k-P_{k-1}\|$: $0,\ 1.803,\ 3.332,\ 4.725,\ 5.990$) and natural splines for $x(t)$, $y(t)$ gives a smooth path of length $6.207$ (the polyline has length $5.990$). Chord-length parameters avoid the loops that uniform $t$ can produce when points are unevenly spaced.

---

## 3.7 Interpolation in Practice: Runge, Chebyshev and Noisy Data *(data-science extension)*

**Runge's phenomenon.** For $f(x) = \frac{1}{1+25x^2}$ on $[-1,1]$, equally spaced interpolation *diverges* as $n\to\infty$, although $f$ is analytic on $[-1,1]$. The culprit is the nodal polynomial, which is huge near the endpoints.

**Chebyshev nodes.** $x_k = \cos\frac{(2k+1)\pi}{2(n+1)}$, $k = 0,\dots,n$, are the zeros of $T_{n+1}$ and minimise $\max_{[-1,1]}|\omega(x)|$, attaining $2^{-n}$ (Chapter 8). For $f$ analytic near $[-1,1]$, Chebyshev interpolation converges exponentially.

**Lebesgue constant.** $\Lambda_n = \max_x\sum_k|L_{n,k}(x)|$ bounds how much interpolation amplifies data errors: $\|P - P^\ast\|_\infty\le\Lambda_n\|f-f^\ast\|_\infty$. For equispaced nodes $\Lambda_n\sim\frac{2^{n+1}}{en\ln n}$ (exponential); for Chebyshev nodes $\Lambda_n\le\frac2\pi\ln(n+1)+1$ (logarithmic).

**Barycentric formula (the stable way to evaluate).** With weights $w_k = 1/\prod_{j\neq k}(x_k-x_j)$,
$$
P(x) = \frac{\sum_k\frac{w_k}{x-x_k}f_k}{\sum_k\frac{w_k}{x-x_k}},
$$
$O(n)$ per point and numerically stable (`scipy.interpolate.BarycentricInterpolator`).

**Interpolation vs regression.** Interpolation reproduces the data exactly — including its noise. For measured data, use least-squares fitting (Chapters 8 and 13) or smoothing splines.

### Examples for §3.7

**Example 3.7.1 (Runge's phenomenon).**

| $n$ | 5 | 10 | 15 | 20 |
|---|---|---|---|---|
| equispaced max error | 0.433 | 1.92 | 2.11 | 59.8 |
| Chebyshev max error | 0.556 | 0.109 | 0.0831 | 0.0153 |

**Example 3.7.2 (the nodal polynomial).** $\max_{[-1,1]}|\omega(x)|$:

| $n$ | equispaced | Chebyshev | $2^{-n}$ |
|---|---|---|---|
| 4 | $1.14\times10^{-1}$ | $6.25\times10^{-2}$ | $6.25\times10^{-2}$ |
| 8 | $1.88\times10^{-2}$ | $3.91\times10^{-3}$ | $3.91\times10^{-3}$ |
| 16 | $9.43\times10^{-4}$ | $1.53\times10^{-5}$ | $1.53\times10^{-5}$ |

Chebyshev nodes attain the theoretical minimum $2^{-n}$ exactly.

**Example 3.7.3 (barycentric evaluation).** Nodes $0,1,3$, values $1,3,2$. Weights: $w_0 = \frac{1}{(0-1)(0-3)} = \frac13$, $w_1 = \frac{1}{(1-0)(1-3)} = -\frac12$, $w_2 = \frac{1}{(3-0)(3-1)} = \frac16$. At $x = 2$:
$$
P(2) = \frac{\frac{1/3}{2}(1) + \frac{-1/2}{1}(3) + \frac{1/6}{-1}(2)}{\frac{1/3}{2} + \frac{-1/2}{1} + \frac{1/6}{-1}} = \frac{\frac16 - \frac32 - \frac13}{\frac16-\frac12-\frac16} = \frac{-5/3}{-1/2} = \frac{10}{3}.
$$

**Example 3.7.4 (Lebesgue constants).**

| $n$ | 5 | 10 | 20 |
|---|---|---|---|
| equispaced $\Lambda_n$ | 3.11 | 29.9 | $1.1\times10^4$ |
| Chebyshev $\Lambda_n$ | 2.10 | 2.49 | 2.90 |

With equispaced degree-20 interpolation, data errors of $10^{-3}$ can become errors of $10$ in $P$.

**Example 3.7.5 (don't interpolate noise).** Eleven equispaced samples of $\sin(2\pi x)$ with noise $\sigma = 0.1$. The degree-10 interpolant deviates from the true curve by up to $0.61$ (it fits the noise); a least-squares cubic deviates by only $0.15$. Interpolation has zero *training* error but large *generalisation* error — overfitting in numerical form.

**Example 3.7.6 (bilinear interpolation — image resizing).** On the unit square with corner values $f(0,0) = 10$, $f(1,0) = 14$, $f(0,1) = 12$, $f(1,1) = 20$:
$$
f(x,y)\approx f_{00}(1-x)(1-y) + f_{10}x(1-y) + f_{01}(1-x)y + f_{11}xy .
$$
At $(0.3,0.6)$: $10(0.7)(0.4) + 14(0.3)(0.4) + 12(0.7)(0.6) + 20(0.3)(0.6) = 13.12$. Bilinear/bicubic interpolation is used by every image library when resizing images for a CNN.

---

## Chapter summary

| Method | Data used | Cost | Best for |
|---|---|---|---|
| Lagrange / barycentric | $f(x_i)$ | $O(n^2)$ setup, $O(n)$ eval | theory; stable evaluation |
| Neville | $f(x_i)$ | $O(n^2)$ per point | one point, unknown degree |
| Newton divided differences | $f(x_i)$ | $O(n^2)$ setup | adding points; derivatives |
| Hermite | $f, f'$ | $O(n^2)$ | position + velocity data |
| Cubic spline | $f(x_i)$ (+ end slopes) | $O(n)$ | general data, $O(h^4)$ |
| Bézier / parametric | control points | $O(1)$ per segment | curves, graphics |

## Further reading

Burden & Faires Ch. 3 · Trefethen, *Approximation Theory and Approximation Practice* (SIAM 2013) · de Boor, *A Practical Guide to Splines* · Berrut & Trefethen, "Barycentric Lagrange interpolation", *SIAM Review* 46 (2004).
