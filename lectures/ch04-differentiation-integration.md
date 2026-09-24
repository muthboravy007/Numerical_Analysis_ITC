# Chapter 4 — Numerical Differentiation and Integration

> **Reference:** Burden & Faires, Chapter 4 (§4.1–4.9) plus §4.10 (Monte Carlo) for data science.
> **Code:** [`numlib/integration.py`](../numlib/integration.py) · **All numbers reproduced by** [`lectures/code/ch04.py`](code/ch04.py)
> **Worked problems:** [`examples/ch04`](../examples/ch04_examples.md) · **Homework:** [`exercises/ch04`](../exercises/ch04_exercises.md)

## Learning outcomes

1. Derive finite-difference formulas (two-, three-, five-point; second derivative) with error terms from Lagrange interpolation or Taylor series.
2. Analyse the truncation/round-off trade-off and choose an optimal step size.
3. Apply Richardson extrapolation to any quantity with a known error expansion.
4. Derive Newton–Cotes rules, their degree of precision and error terms; apply composite rules and prove their error bounds.
5. Use Romberg integration, adaptive quadrature and Gaussian quadrature (Legendre, Hermite, Laguerre).
6. Approximate double integrals and improper integrals.
7. Use Monte Carlo and quasi-Monte Carlo integration with error estimates and variance reduction.

---

## 4.1 Numerical Differentiation

**Derivation from interpolation.** If $x_0,\dots,x_n$ are distinct and $P$ interpolates $f$, then $f = P + \frac{f^{(n+1)}(\xi(x))}{(n+1)!}\prod(x-x_k)$. Differentiating and evaluating at a node $x_j$ (where the product vanishes, which removes the unknown derivative of $\xi(x)$) gives the **$(n+1)$-point formula**
$$
f'(x_j) = \sum_{k=0}^nf(x_k)L_k'(x_j) + \frac{f^{(n+1)}(\xi_j)}{(n+1)!}\prod_{k\ne j}(x_j-x_k).
$$

**Standard formulas** (step $h$, $\xi$ in the span of the nodes):

| Name | Formula | Error |
|---|---|---|
| two-point (forward/backward) | $\dfrac{f(x_0+h)-f(x_0)}{h}$ | $-\dfrac h2f''(\xi)$ |
| three-point **endpoint** | $\dfrac{-3f(x_0)+4f(x_0+h)-f(x_0+2h)}{2h}$ | $+\dfrac{h^2}{3}f'''(\xi)$ |
| three-point **midpoint** | $\dfrac{f(x_0+h)-f(x_0-h)}{2h}$ | $-\dfrac{h^2}{6}f'''(\xi)$ |
| five-point midpoint | $\dfrac{f(x_0-2h)-8f(x_0-h)+8f(x_0+h)-f(x_0+2h)}{12h}$ | $+\dfrac{h^4}{30}f^{(5)}(\xi)$ |
| second-derivative midpoint | $\dfrac{f(x_0-h)-2f(x_0)+f(x_0+h)}{h^2}$ | $-\dfrac{h^2}{12}f^{(4)}(\xi)$ |

(Negative $h$ in the endpoint formula gives the backward version.)

**Round-off instability.** If each $f$-value has error $|e|\le\varepsilon$, the three-point midpoint formula has total error
$$
E(h)\le\frac{\varepsilon}{h} + \frac{h^2}{6}M,\qquad M = \max|f'''|,
$$
minimised at $h^\ast = (3\varepsilon/M)^{1/3}$. **Numerical differentiation is unstable:** small $h$ reduces truncation error but amplifies round-off like $1/h$ (or $1/h^2$ for second derivatives).

### Examples for §4.1

**Example 4.1.1 (forward difference and its bound).** $f(x) = \ln x$ at $x_0 = 2$ ($f'(2) = 0.5$). Error bound $\frac h2\max|f''| = \frac h2\cdot\frac{1}{2^2} = \frac h8$:

| $h$ | $\frac{f(2+h)-f(2)}{h}$ | error | bound $h/8$ |
|---|---|---|---|
| 0.1 | 0.4879016 | $1.21\times10^{-2}$ | $1.25\times10^{-2}$ |
| 0.05 | 0.4938523 | $6.15\times10^{-3}$ | $6.25\times10^{-3}$ |
| 0.01 | 0.4987542 | $1.25\times10^{-3}$ | $1.25\times10^{-3}$ |

The error halves with $h$: first order.

**Example 4.1.2 (comparing formulas).** $f(x) = e^x$, $x_0 = 1$, $h = 0.1$ (true $f'(1) = 2.7182818$):

| formula | value | error |
|---|---|---|
| two-point forward | 2.8588420 | $1.4\times10^{-1}$ |
| three-point endpoint | 2.7085084 | $9.8\times10^{-3}$ |
| three-point midpoint | 2.7228146 | $4.5\times10^{-3}$ |
| five-point midpoint | 2.7182728 | $9.1\times10^{-6}$ |

The midpoint formula's error is about half the endpoint's — compare the constants $\frac16$ and $\frac13$ in the table.

**Example 4.1.3 (tabulated data).** From a 6-decimal table of $f(x) = xe^x$ at $x = 1.8, 1.9, 2.0, 2.1, 2.2$ ($10.889365, 12.703199, 14.778112, 17.148957, 19.855030$), approximate $f'(2) = 3e^2 = 22.167168$ and $f''(2) = 4e^2 = 29.556224$:
* three-point midpoint ($h = 0.1$): $\frac{17.148957-12.703199}{0.2} = 22.228790$ (error $0.062$);
* three-point endpoint ($h=0.1$): $\frac{-3(14.778112)+4(17.148957)-19.855030}{0.2} = 22.032310$ (error $0.135$);
* five-point: $\frac{10.889365-8(12.703199)+8(17.148957)-19.855030}{1.2} = 22.166999$ (error $1.7\times10^{-4}$, matching $\frac{h^4}{30}|f^{(5)}|\approx\frac{10^{-4}}{30}\cdot7e^2 = 1.7\times10^{-4}$);
* second derivative: $\frac{12.703199-2(14.778112)+17.148957}{0.01} = 29.593200$ (error $0.037$).

**Example 4.1.4 (round-off for the second derivative).** $f = \sin$, $x_0 = 1$, $f''(1) = -0.8414709848$:

| $h$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-7}$ | $10^{-8}$ |
|---|---|---|---|---|---|---|---|---|
| error | $7.0\times10^{-4}$ | $7.0\times10^{-6}$ | $7.0\times10^{-8}$ | $3.0\times10^{-9}$ | $7.6\times10^{-7}$ | $7.8\times10^{-5}$ | $8.8\times10^{-3}$ | $0.27$ |

Truncation $\frac{h^2}{12}|f^{(4)}|$ dominates for $h\ge10^{-4}$; round-off $\frac{4\varepsilon|f|}{h^2}$ dominates below. The theory predicts $h^\ast = (48u|f|/|f^{(4)}|)^{1/4}\approx3\times10^{-4}$, consistent with the table.

**Example 4.1.5 (rates from cumulative counts).** Cumulative cases on days $0,\dots,8$: $0,12,30,51,74,98,120,138,151$. `np.gradient` (central differences inside, one-sided at the ends) gives daily rates $12,\ 15,\ 19.5,\ 22,\ 23.5,\ 23,\ 20,\ 15.5,\ 13$. At day 4 the central rate is $\frac{98-51}{2} = 23.5$ and the second difference $98 - 2(74) + 51 = 1$ — the epidemic is near its inflection point (maximum daily rate). Differencing noisy data amplifies the noise, so in practice one smooths first (splines, §3.5; regression, Chapter 13).

---

## 4.2 Richardson Extrapolation

Suppose $M = N_1(h) + K_1h + K_2h^2 + K_3h^3+\cdots$. Then $M = N_1(h/2) + K_1\frac h2 + \cdots$, and $2\times$(second) $-$ (first) eliminates the $h$ term:
$$
N_2(h) = N_1(\tfrac h2) + \bigl[N_1(\tfrac h2) - N_1(h)\bigr],\qquad M = N_2(h) + O(h^2).
$$
In general
$$
N_j(h) = N_{j-1}(\tfrac h2) + \frac{N_{j-1}(h/2)-N_{j-1}(h)}{2^{j-1}-1}.
$$
If only **even** powers appear, $M = N_1(h) + K_1h^2 + K_2h^4+\cdots$ (e.g. central differences, trapezoid rule), use
$$
N_j(h) = N_{j-1}(\tfrac h2) + \frac{N_{j-1}(h/2)-N_{j-1}(h)}{4^{j-1}-1},
$$
gaining **two** orders per level. Richardson extrapolation turns any low-order method with a known error expansion into a high-order one — for the price of a few more evaluations.

### Examples for §4.2

**Example 4.2.1 (central differences, even powers).** $N_1(h) = \frac{e^{1+h}-e^{1-h}}{2h}$ with $h = 0.2, 0.1, 0.05, 0.025$:

| $N_1$ ($O(h^2)$) | $N_2$ ($O(h^4)$) | $N_3$ ($O(h^6)$) | $N_4$ ($O(h^8)$) |
|---|---|---|---|
| 2.736439986 | | | |
| 2.722814564 | 2.718272757 | | |
| 2.719414587 | 2.718281262 | 2.718281829 | |
| 2.718564992 | 2.718281793 | 2.718281828 | 2.718281828 |

Diagonal errors: $1.8\times10^{-2}$, $9.1\times10^{-6}$, $5.4\times10^{-10}$, $4.9\times10^{-15}$.

**Example 4.2.2 (forward differences, all powers).** $N_1(h) = \frac{e^{1+h}-e}{h}$ has an expansion in *all* powers of $h$, so we use the $2^{j-1}-1$ denominators. From $h = 0.2,\ 0.1,\ 0.05,\ 0.025$: last row $2.752545,\ 2.717705,\ 2.718296,\ 2.7182812$ — error $6.0\times10^{-7}$ from first-order data.

**Example 4.2.3 (Archimedes' $\pi$).** A regular $n$-gon inscribed in the unit circle has half-perimeter $p_n = n\sin(\pi/n) = \pi - \frac{\pi^3}{6n^2} + \frac{\pi^5}{120n^4}-\cdots$ — even powers of $h = 1/n$. With $n = 6, 12, 24$: $p = 3.0000000,\ 3.1058285,\ 3.1326286$. One extrapolation: $3.1411047,\ 3.1415620$; two: $\mathbf{3.1415925}$ (error $2\times10^{-7}$). Archimedes needed a 96-gon for $3.14$.

**Example 4.2.4 (extrapolating a limit).** $F(h) = (1+h)^{1/h}\to e$ with $F(h) = e - \frac e2h + O(h^2)$. From $h = 0.04, 0.02, 0.01$: $2.665836,\ 2.691588,\ 2.704814$ (error $1.3\times10^{-2}$). First extrapolation ($2N(h/2) - N(h)$): $2.717340,\ 2.718040$; second ($\frac{4N_2(h/2)-N_2(h)}{3}$): $2.718273$ (error $9\times10^{-6}$).

**Example 4.2.5 (extrapolating the trapezoid rule = Simpson).** For $\int_0^\pi\sin x\,dx = 2$: $T_8 = 1.9742316$, $T_{16} = 1.9935703$. $\frac{4T_{16}-T_8}{3} = 2.0000166$, which is exactly Simpson's rule $S_{16}$. Repeating the idea is Romberg integration (§4.5).

---

## 4.3 Elements of Numerical Integration

**Numerical quadrature:** $\int_a^bf(x)\,dx\approx\sum_{i=0}^na_if(x_i)$, with weights $a_i = \int_a^bL_i(x)\,dx$ obtained by integrating the interpolating polynomial.

**Trapezoidal rule** ($n=1$, $h = b-a$) and **Simpson's rule** ($n=2$, $h = \frac{b-a}{2}$):
$$
\int_{x_0}^{x_1}f = \frac h2[f(x_0)+f(x_1)] - \frac{h^3}{12}f''(\xi),\qquad
\int_{x_0}^{x_2}f = \frac h3[f(x_0)+4f(x_1)+f(x_2)] - \frac{h^5}{90}f^{(4)}(\xi).
$$
(The Simpson error comes from a Taylor expansion to third order; the $h^4$ term vanishes by symmetry, which is why Simpson is exact for cubics.)

**Definition 4.1 (degree of precision).** The largest $n$ such that the formula is exact for $x^k$, $k = 0,1,\dots,n$.

**Closed Newton–Cotes** ($x_i = x_0+ih$, $h = \frac{b-a}{n}$):

| $n$ | rule | formula | error | precision |
|---|---|---|---|---|
| 1 | trapezoid | $\frac h2(f_0+f_1)$ | $-\frac{h^3}{12}f''$ | 1 |
| 2 | Simpson | $\frac h3(f_0+4f_1+f_2)$ | $-\frac{h^5}{90}f^{(4)}$ | 3 |
| 3 | Simpson 3/8 | $\frac{3h}{8}(f_0+3f_1+3f_2+f_3)$ | $-\frac{3h^5}{80}f^{(4)}$ | 3 |
| 4 | Boole | $\frac{2h}{45}(7f_0+32f_1+12f_2+32f_3+7f_4)$ | $-\frac{8h^7}{945}f^{(6)}$ | 5 |

**Open Newton–Cotes** (no endpoints; $h = \frac{b-a}{n+2}$, $x_i = a + (i+1)h$):

| $n$ | formula | error |
|---|---|---|
| 0 (midpoint) | $2hf_0$ | $+\frac{h^3}{3}f''$ |
| 1 | $\frac{3h}{2}(f_0+f_1)$ | $+\frac{3h^3}{4}f''$ |
| 2 | $\frac{4h}{3}(2f_0-f_1+2f_2)$ | $+\frac{14h^5}{45}f^{(4)}$ |
| 3 | $\frac{5h}{24}(11f_0+f_1+f_2+11f_3)$ | $+\frac{95h^5}{144}f^{(4)}$ |

Even $n$ gains one extra degree of precision. High-order Newton–Cotes rules eventually get negative weights and are unstable — so in practice we use *composite* low-order rules.

### Examples for §4.3

**Example 4.3.1.** $I = \int_0^1e^{-x^2}dx = 0.7468241$: trapezoid $0.6839397$, midpoint $0.7788008$, Simpson $0.7471804$, Simpson 3/8 $0.7469923$. Note that the midpoint error ($-0.032$) is about half the trapezoid error ($+0.063$) with opposite sign — compare $+\frac{h^3}{24}$ and $-\frac{h^3}{12}$ for the same interval. This observation leads to Simpson $= \frac23M + \frac13T$.

**Example 4.3.2 (degree of precision).** On $[0,2]$:

| $f$ | exact | trapezoid | Simpson | 3/8 |
|---|---|---|---|---|
| $1$ | 2 | 2 | 2 | 2 |
| $x$ | 2 | 2 | 2 | 2 |
| $x^2$ | 2.6667 | 4 | 2.6667 | 2.6667 |
| $x^3$ | 4 | 8 | 4 | 4 |
| $x^4$ | 6.4 | 16 | 6.6667 | 6.5185 |

Trapezoid: precision 1. Simpson and 3/8: precision 3.

**Example 4.3.3 (using the error term).** For Simpson on $\int_0^1e^{-x^2}$: $h = 0.5$, $f^{(4)}(x) = (16x^4-48x^2+12)e^{-x^2}$ has $\max_{[0,1]}|f^{(4)}| = 12$ (at $x=0$). Bound: $\frac{(0.5)^5}{90}\cdot12 = 4.2\times10^{-3}$; actual error $3.6\times10^{-4}$.

**Example 4.3.4 (closed vs open rules).** $\int_0^2\frac{dx}{1+x} = \ln3 = 1.0986123$.

| $n$ | closed | open |
|---|---|---|
| 0 | — | 1.0000000 |
| 1 | 1.3333333 | 1.0285714 |
| 2 | 1.1111111 | 1.0888889 |
| 3 | 1.1047619 | 1.0915011 |
| 4 | 1.0992593 | — |

Open rules are used when $f$ cannot be evaluated at an endpoint (singularities, §4.9) and as predictors in multistep ODE methods (Chapter 5).

**Example 4.3.5 (midpoint rule precision).** On $[0,2]$ the midpoint rule $2f(1)$ gives $2, 2, 2, 2$ for $1, x, x^2, x^3$; the exact values are $2, 2, 2.667, 4$. It is exact for degree $\le1$ although it uses **one** point — foreshadowing Gaussian quadrature, where node placement buys precision.

**Example 4.3.6 (method of undetermined coefficients).** Find $a_0,a_1,a_2$ so that $\int_0^3f\approx a_0f(0)+a_1f(1)+a_2f(3)$ is exact for degree $\le2$:
$$
a_0+a_1+a_2 = 3,\quad a_1+3a_2 = \tfrac92,\quad a_1+9a_2 = 9 \ \Rightarrow\ a_0 = 0,\ a_1 = \tfrac94,\ a_2 = \tfrac34.
$$
For $x^3$ it gives $\frac94+\frac34\cdot27 = 22.5\ne\frac{81}{4}$: precision exactly 2.

---

## 4.4 Composite Numerical Integration

Divide $[a,b]$ into $n$ subintervals of width $h = (b-a)/n$, $x_j = a+jh$, and apply a low-order rule on each.

**Theorem 4.4 (Composite Simpson).** For $f\in C^4[a,b]$ and even $n$,
$$
\int_a^bf = \frac h3\Bigl[f(a) + 2\sum_{j=1}^{n/2-1}f(x_{2j}) + 4\sum_{j=1}^{n/2}f(x_{2j-1}) + f(b)\Bigr] - \frac{b-a}{180}h^4f^{(4)}(\mu).
$$
**Theorem 4.5 (Composite trapezoid).** For $f\in C^2[a,b]$,
$$
\int_a^bf = \frac h2\Bigl[f(a) + 2\sum_{j=1}^{n-1}f(x_j) + f(b)\Bigr] - \frac{b-a}{12}h^2f''(\mu).
$$
**Composite midpoint** ($n$ subintervals): $h\sum_{j=1}^nf\bigl(a+(j-\frac12)h\bigr) + \frac{b-a}{24}h^2f''(\mu)$.

*Proof idea (Simpson).* Sum the local errors $-\frac{h^5}{90}f^{(4)}(\xi_j)$ over $n/2$ panels; the average $\frac{2}{n}\sum f^{(4)}(\xi_j)$ lies between $\min f^{(4)}$ and $\max f^{(4)}$, so by the IVT equals $f^{(4)}(\mu)$; then $\frac n2\cdot\frac{h^5}{90} = \frac{(b-a)h^4}{180}$. ∎

**Round-off stability.** If each $f(x_i)$ has error $\le\varepsilon$, the composite Simpson rule's round-off error is $\le(b-a)\varepsilon$, **independent of $n$**. Unlike differentiation, integration is stable.

### Examples for §4.4

**Example 4.4.1 (convergence rates).** $\int_0^1e^{-x^2}dx$:

| $n$ | midpoint error | trapezoid error | Simpson error |
|---|---|---|---|
| 2 | $7.8\times10^{-3}$ | $1.6\times10^{-2}$ | $3.6\times10^{-4}$ |
| 4 | $1.9\times10^{-3}$ | $3.8\times10^{-3}$ | $3.1\times10^{-5}$ |
| 8 | $4.8\times10^{-4}$ | $9.6\times10^{-4}$ | $2.0\times10^{-6}$ |
| 16 | $1.2\times10^{-4}$ | $2.4\times10^{-4}$ | $1.3\times10^{-7}$ |
| 32 | $3.0\times10^{-5}$ | $6.0\times10^{-5}$ | $7.8\times10^{-9}$ |

Trapezoid/midpoint errors drop ×4 per halving ($O(h^2)$); Simpson ×16 ($O(h^4)$).

**Example 4.4.2 (choosing $n$ in advance).** For accuracy $10^{-6}$ on $\int_0^1e^{-x^2}$:
* trapezoid: $\frac{1}{12}h^2\cdot2\le10^{-6}$ ($\max|f''| = 2$) $\Rightarrow n\ge408.2$, so $n = 409$ (actual error $3.7\times10^{-7}$);
* Simpson: $\frac{1}{180}h^4\cdot12\le10^{-6}$ $\Rightarrow n\ge16.07$, so $n = 18$ (even) (actual error $7.8\times10^{-8}$).
Simpson needs 19 evaluations; the trapezoid rule 410.

**Example 4.4.3 ($\pi$ as an integral).** $\pi = \int_0^1\frac{4}{1+x^2}dx$. With $n = 8$: $T_8 = 3.1389885$ (error $2.6\times10^{-3}$), $S_8 = 3.1415925$ (error $1.5\times10^{-7}$).

**Example 4.4.4 (periodic integrands — exponential convergence).** $\int_0^{2\pi}e^{\cos x}dx = 2\pi I_0(1) = 7.954926521$. Trapezoid: $n = 4$: error $3.4\times10^{-2}$; $n = 8$: $1.3\times10^{-6}$; $n = 16$: $1.8\times10^{-15}$. For smooth periodic functions all Euler–Maclaurin correction terms cancel, so the trapezoid rule beats every Newton–Cotes rule. (This is why the discrete Fourier transform, Chapter 8, is so accurate.)

**Example 4.4.5 (ROC AUC).** A classifier's ROC curve has points (FPR, TPR): $(0,0),(0,0.4),(0.1,0.4),(0.1,0.7),(0.3,0.7),(0.3,0.9),(0.6,1),(1,1)$. The trapezoid rule on these unequally spaced data gives $\text{AUC} = 0.865$ — exactly what `sklearn.metrics.auc` computes.

**Example 4.4.6 (distance from speed data).** Speed (m/s) at $t = 0,10,\dots,60$ s: $0,12,21,27,30,31,31.5$. Distance: trapezoid $1367.5$ m, composite Simpson $1378.3$ m. The difference ($\approx0.8\%$) indicates the uncertainty due to discretisation.

---

## 4.5 Romberg Integration

The composite trapezoid rule has the **Euler–Maclaurin** error expansion (for smooth $f$)
$$
\int_a^bf - T(h) = K_1h^2 + K_2h^4 + K_3h^6 + \cdots,
$$
so Richardson extrapolation with even powers applies. With $h_k = \frac{b-a}{2^{k-1}}$:
$$
R_{1,1} = \frac{b-a}{2}[f(a)+f(b)],\qquad R_{k,1} = \frac12\Bigl[R_{k-1,1} + h_{k-1}\sum_{i=1}^{2^{k-2}}f\bigl(a+(2i-1)h_k\bigr)\Bigr],
$$
$$
R_{k,j} = R_{k,j-1} + \frac{R_{k,j-1}-R_{k-1,j-1}}{4^{j-1}-1}.
$$
Each new row re-uses all previous function values; $R_{k,2}$ is composite Simpson and $R_{k,3}$ composite Boole. $R_{k,k}$ has error $O(h_k^{2k})$ if $f$ is smooth enough.

### Examples for §4.5

**Example 4.5.1.** $\int_0^1e^{-x^2}dx$:

| $R_{k,1}$ | $R_{k,2}$ | $R_{k,3}$ | $R_{k,4}$ | $R_{k,5}$ |
|---|---|---|---|---|
| 0.683939721 | | | | |
| 0.731370252 | 0.747180429 | | | |
| 0.742984098 | 0.746855380 | 0.746833710 | | |
| 0.745865615 | 0.746826121 | 0.746824170 | 0.746824018 | |
| 0.746584597 | 0.746824257 | 0.746824133 | 0.746824133 | **0.746824133** |

$R_{5,5}$ has error $2.8\times10^{-10}$ using 17 function evaluations.

**Example 4.5.2.** $\int_0^\pi\sin x\,dx = 2$. The first column is $0,\ 1.5708,\ 1.8961,\ 1.9742,\ 1.9936$ — slow — but $R_{5,5} = 1.999999995$ (error $5.4\times10^{-9}$).

**Example 4.5.3 (when Romberg disappoints).** $\int_0^1\sqrt x\,dx = \frac23$. $\sqrt x$ is not differentiable at 0, so there is no $h^2, h^4,\dots$ expansion (the error behaves like $h^{3/2}$). Diagonal errors: $0.167,\ 0.029,\ 0.0089,\ 0.0031,\ 0.0011,\ 0.00038$ — barely better than the first column. *Extrapolation needs smoothness*; for such integrals use a substitution ($x = u^2$) or adaptive/Gauss–Jacobi quadrature.

**Example 4.5.4.** $\int_1^3\frac{dx}{x} = \ln3 = 1.0986123$: diagonal $1.3333,\ 1.1111,\ 1.0992593,\ 1.0986305,\ 1.0986125$ (error $2.3\times10^{-7}$).

**Example 4.5.5 (cost).** $R_{n,n}$ requires $2^{n-1}+1$ evaluations: $n = 5$ needs 17, $n = 10$ needs 513. Stop when $|R_{n,n}-R_{n-1,n-1}|<\text{TOL}$.

---

## 4.6 Adaptive Quadrature Methods

Use small steps only where $f$ varies rapidly. Let $S(a,b)$ denote Simpson's rule on $[a,b]$ and $m = \frac{a+b}2$. From the error terms,
$$
\Bigl|\int_a^bf - S(a,m) - S(m,b)\Bigr|\approx\frac{1}{15}\bigl|S(a,b) - S(a,m) - S(m,b)\bigr|.
$$
**Adaptive Simpson:** if $|S(a,b)-S(a,m)-S(m,b)|<15\,\text{TOL}$, accept $S(a,m)+S(m,b)$ (optionally plus the correction $\frac{1}{15}(\ldots)$); otherwise apply the procedure recursively to $[a,m]$ and $[m,b]$ with tolerance $\text{TOL}/2$ each. (Many codes use $10\,\text{TOL}$ for safety.) `scipy.integrate.quad` (QUADPACK) uses the same idea with Gauss–Kronrod pairs.

### Examples for §4.6

**Example 4.6.1 (an oscillatory integrand).** $\int_1^3\frac{100}{x^2}\sin\frac{10}{x}\,dx = -1.4260247563$. Adaptive Simpson with $\text{TOL} = 10^{-4}$ uses 89 evaluations and has error $6.8\times10^{-8}$. Composite Simpson errors: $n = 8$: $0.27$; $n=32$: $9.7\times10^{-3}$; $n = 128$: $3.9\times10^{-5}$ — adaptivity puts the points near $x = 1$ where the oscillation is fastest.

**Example 4.6.2 (the error estimate works).** One step on $\int_0^1e^{-x^2}$: $S(0,1) = 0.7471804$, $S(0,\frac12)+S(\frac12,1) = 0.7468554$. Estimated error $\frac1{15}|0.7471804-0.7468554| = 2.2\times10^{-5}$; true error $3.1\times10^{-5}$ — the right order of magnitude.

**Example 4.6.3 (a kink).** $\int_0^1\sqrt{|x-0.3|}\,dx = \frac23(0.3^{3/2}+0.7^{3/2})$: adaptive Simpson (TOL $10^{-8}$) error $2.5\times10^{-11}$ with 561 evaluations, whereas composite Simpson with 513 points has error $1.2\times10^{-5}$ — uniform refinement wastes effort far from the kink.

**Example 4.6.4 (a peak).** $\int_0^1e^{-50(x-0.5)^2}dx = 0.2506627$: adaptive (TOL $10^{-10}$) error $2\times10^{-14}$, concentrating points near $0.5$.

**Example 4.6.5 (library routine).** `scipy.integrate.quad` on Example 4.6.1 returns $-1.426024756346$ with an error estimate $1.7\times10^{-8}$ using 63 evaluations (21-point Gauss–Kronrod).

---

## 4.7 Gaussian Quadrature

Newton–Cotes fixes the nodes; **Gaussian quadrature** chooses nodes *and* weights: $2n$ free parameters, so we can hope for precision $2n-1$.

**Legendre polynomials** $P_0 = 1$, $P_1 = x$, $P_2 = x^2-\frac13$, $P_3 = x^3-\frac35x$, … (monic form) are orthogonal on $[-1,1]$: $\int_{-1}^1P_iP_j\,dx = 0$ for $i\ne j$.

**Theorem 4.7.** If $x_1,\dots,x_n$ are the roots of $P_n$ and $c_i = \int_{-1}^1\prod_{j\ne i}\frac{x-x_j}{x_i-x_j}dx$, then $\int_{-1}^1P(x)\,dx = \sum c_iP(x_i)$ for every polynomial of degree $\le2n-1$.
*Proof sketch.* Divide $P$ by $P_n$: $P = QP_n + R$ with $\deg Q,R<n$. Then $\int QP_n = 0$ (orthogonality) and $P(x_i) = R(x_i)$; the rule is exact for $R$ (interpolatory). ∎

The weights are all **positive** (so the rule is stable), and the error is $\frac{2^{2n+1}(n!)^4}{(2n+1)[(2n)!]^3}f^{(2n)}(\xi)$.

**Arbitrary interval:** $x = \frac{(b-a)t + a + b}{2}$,
$$
\int_a^bf(x)\,dx = \frac{b-a}{2}\int_{-1}^1f\Bigl(\frac{(b-a)t+b+a}{2}\Bigr)dt .
$$

**Other weights** (same theory with other orthogonal polynomials):

| weight / interval | family | typical DS use |
|---|---|---|
| $1$ on $[-1,1]$ | Gauss–Legendre | finite integrals |
| $e^{-x^2}$ on $\mathbb R$ | Gauss–Hermite | expectations under a normal ($E[g(X)]$) |
| $e^{-x}$ on $[0,\infty)$ | Gauss–Laguerre | exponential/Gamma models, survival |
| $(1-x^2)^{-1/2}$ on $[-1,1]$ | Gauss–Chebyshev | spectral methods |

For $X\sim N(\mu,\sigma^2)$: $E[g(X)] = \frac{1}{\sqrt\pi}\int e^{-t^2}g(\mu+\sqrt2\sigma t)\,dt\approx\frac{1}{\sqrt\pi}\sum w_ig(\mu+\sqrt2\sigma t_i)$.

### Examples for §4.7

**Example 4.7.1 (nodes and weights).**

| $n$ | nodes $x_i$ | weights $c_i$ |
|---|---|---|
| 2 | $\pm0.5773502692 = \pm1/\sqrt3$ | $1,\ 1$ |
| 3 | $0,\ \pm0.7745966692 = \pm\sqrt{3/5}$ | $0.8888888889 = \frac89,\ 0.5555555556 = \frac59$ |
| 4 | $\pm0.3399810436,\ \pm0.8611363116$ | $0.6521451549,\ 0.3478548451$ |
| 5 | $0,\ \pm0.5384693101,\ \pm0.9061798459$ | $0.5688888889,\ 0.4786286705,\ 0.2369268851$ |

**Example 4.7.2.** $\int_0^1e^{-x^2}dx$ by Gauss–Legendre:
$n=2$: $0.746594688$ (error $2.3\times10^{-4}$); $n=3$: $0.746814584$ ($9.5\times10^{-6}$); $n=4$: $0.746824468$ ($3.4\times10^{-7}$); $n = 5$: $0.746824127$ ($6.1\times10^{-9}$). Five evaluations beat composite Simpson with 33.

**Example 4.7.3 (precision $2n-1$).** Two-point Gauss on $[-1,1]$: $f(-\frac{1}{\sqrt3}) + f(\frac1{\sqrt3})$. For $f = x^3+2x^2-1$: $2\cdot\frac23 - 2 = -\frac23$, exact. For $x^4$: $2\cdot\frac19 = 0.2222$ vs exact $0.4$ — precision is exactly 3 with only 2 points.

**Example 4.7.4 (change of interval).** $\int_0^2x^2e^{-x}dx = 2-10e^{-2} = 0.6466472$. With $x = t+1$: $\int_{-1}^1(t+1)^2e^{-(t+1)}dt\approx\frac59g(-0.7746)+\frac89g(0)+\frac59g(0.7746) = 0.6461734$ (error $4.7\times10^{-4}$).

**Example 4.7.5 (expectations with Gauss–Hermite).** For $X\sim N(1, 2^2)$, $E[X^4] = \mu^4+6\mu^2\sigma^2+3\sigma^4 = 73$. Gauss–Hermite with $n = 3$ gives $73.0000000$ (exact, since $x^4$ has degree $\le 2n-1 = 5$). For the logistic-normal integral $E[\sigma(X)]$ (the probit/logit approximation used in Bayesian logistic regression): $n=3$: $0.66521$; $n=5$: $0.65200$; $n=10$: $0.64758$ — converging as $n$ grows.

**Example 4.7.6 (Gauss–Laguerre).** $\int_0^\infty e^{-x}x^3dx = 3! = 6$: the 4-point rule gives $6.000000$ exactly (degree $3\le7$). For $\int_0^\infty\frac{e^{-x}}{1+x}dx = 0.5963474$ the 4-point rule gives $0.5933014$ (error $3\times10^{-3}$; $1/(1+x)$ is not polynomial-like on $[0,\infty)$).

---

## 4.8 Multiple Integrals

For $\iint_Rf(x,y)\,dA$ over $R = \{a\le x\le b,\ c(x)\le y\le d(x)\}$, apply a 1-D rule in $y$ for each $x$-node, then a 1-D rule in $x$:
$$
\int_a^b\underbrace{\int_{c(x)}^{d(x)}f(x,y)\,dy}_{g(x)}\,dx\approx\sum_iw_i\,g(x_i),\qquad g(x_i)\approx\sum_jv_{ij}f(x_i,y_{ij}).
$$
Composite Simpson in both directions (B&F Algorithm 4.4) has error $O(h^4 + k^4)$; the **Gauss product rule** with $n\times n$ points is exact for $x^py^q$, $p,q\le2n-1$.

**Curse of dimensionality.** A product rule with $m$ points per axis in $d$ dimensions needs $m^d$ evaluations and has error $O(N^{-p/d})$ in terms of $N = m^d$. Beyond $d\approx4$ Monte Carlo (§4.10) is usually preferable.

### Examples for §4.8

**Example 4.8.1 (a rectangle).** $\int_0^1\int_0^2\ln(1+x+y)\,dy\,dx = 1.7603052$. Composite Simpson with $n = m = 2$: $1.7577031$ (error $2.6\times10^{-3}$); $n = m = 4$: $1.7600870$ ($2.2\times10^{-4}$); Gauss $3\times3$ (9 points): $1.7603610$ ($5.6\times10^{-5}$).

**Example 4.8.2 (a non-rectangular region).** $\int_0^1\int_{x^2}^xxe^y\,dy\,dx = \int_0^1x(e^x-e^{x^2})dx = 1 - \frac{e-1}{2} = 0.1408591$. Simpson with $n=m=2$: $0.1215654$; $n=m=4$: $0.1391719$; $n = m = 10$: $0.1408106$.

**Example 4.8.3 (area as an integral).** Area of the quarter disc $= \int_0^1\int_0^{\sqrt{1-x^2}}1\,dy\,dx$. Simpson with $n = m = 20$: $0.7841118$ vs $\pi/4 = 0.7853982$. Convergence is slow because $\sqrt{1-x^2}$ has an infinite derivative at $x = 1$.

**Example 4.8.4 (a bivariate normal probability).** For standard normals with correlation $\rho = 0.5$, $P(X\le0,Y\le0) = \frac14 + \frac{\arcsin\rho}{2\pi} = \frac13$. A $20\times20$ Gauss product rule on $[-6,0]^2$ gives $0.33333333$ — this is how `scipy.stats.multivariate_normal.cdf` works in low dimensions.

**Example 4.8.5 (cost growth).** Composite Simpson with 11 points per axis: $d = 1$: 11 evaluations; $d=2$: 121; $d=3$: 1331; $d=5$: $1.6\times10^5$; $d=10$: $2.6\times10^{10}$. A posterior over 10 parameters cannot be integrated this way.

---

## 4.9 Improper Integrals

**Singularity at an endpoint.** For $\int_a^b\frac{g(x)}{(x-a)^p}dx$, $0<p<1$, subtract the Taylor polynomial $P_4$ of $g$ at $a$:
$$
\int_a^b\frac{g(x)}{(x-a)^p}dx = \underbrace{\int_a^b\frac{P_4(x)}{(x-a)^p}dx}_{\text{exact}} + \int_a^b\underbrace{\frac{g(x)-P_4(x)}{(x-a)^p}}_{G(x),\ G(a):=0}dx,
$$
where $G\in C^4$ now and Simpson converges normally.

**Infinite interval.** $\int_a^\infty f(x)\,dx$ with $t = 1/x$ becomes $\int_0^{1/a}t^{-2}f(1/t)\,dt$, a (possibly singular) finite integral. Alternatives: truncate the interval where the tail is negligible; use a mapping such as $x = \tan t$; or Gauss–Laguerre/Hermite.

### Examples for §4.9

**Example 4.9.1 (endpoint singularity).** $\int_0^1\frac{\cos x}{x^{1/4}}dx$. With $P_4(x) = 1-\frac{x^2}2+\frac{x^4}{24}$:
$$
\int_0^1x^{-1/4}P_4\,dx = \frac43 - \frac12\cdot\frac{4}{11} + \frac1{24}\cdot\frac{4}{19} = 1.1602871 .
$$
The remainder $G(x) = \frac{\cos x - P_4(x)}{x^{1/4}}$ is tiny and smooth; Simpson with $n = 4$ gives $-0.0002058$. Total $1.1600813$; exact $1.1600841$ (error $2.8\times10^{-6}$ with 5 evaluations).

**Example 4.9.2 (infinite interval by $t = 1/x$).** $\int_1^\infty x^{-4}\sin x\,dx = \int_0^1t^2\sin(1/t)\,dt = 0.2865295$. The transformed integrand oscillates infinitely often near $t = 0$ (but is bounded by $t^2$). Composite Simpson: $n = 8$: $0.29009$; $n = 32$: $0.28628$; $n = 128$: $0.2865395$. Adaptive Simpson (TOL $10^{-8}$): $0.28652954$ (1613 evaluations).

**Example 4.9.3 (Gaussian tail).** $\int_1^\infty e^{-x^2}dx = \frac{\sqrt\pi}2\operatorname{erfc}(1) = 0.1394028$. With $t = 1/x$: $\int_0^1t^{-2}e^{-1/t^2}dt$, whose integrand is smooth (all derivatives vanish at $t = 0$); Simpson with $n = 16$: $0.1394051$.

**Example 4.9.4 (the $\tan$ map).** $\int_0^\infty e^{-x^2}dx = \frac{\sqrt\pi}2 = 0.8862269$. With $x = \tan t$: $\int_0^{\pi/2}e^{-\tan^2t}\sec^2t\,dt$; 20-point Gauss–Legendre gives $0.8862262$.

**Example 4.9.5 (expectations as improper integrals).** For $X\sim\text{Exp}(\lambda = 0.5)$, $E[X] = \int_0^\infty x\lambda e^{-\lambda x}dx = 2$. Substituting $u = \lambda x$ gives $\frac1\lambda\int_0^\infty ue^{-u}du$; 2-point Gauss–Laguerre is exact: $2.000$.

**Example 4.9.6 (don't ignore singularities).** $\int_0^1x^{-1/2}dx = 2$. 10-point Gauss–Legendre (which never evaluates at 0) gives $1.9171$ — a 4% error because the integrand is unbounded. Substituting $x = u^2$ gives $\int_0^12\,du$, integrated exactly by any rule.

---

## 4.10 Monte Carlo Integration *(data-science extension)*

Write the integral as an expectation. For $X\sim\text{Uniform}(\Omega)$ with volume $|\Omega|$:
$$
I = \int_\Omega f = |\Omega|\,E[f(X)],\qquad \hat I_N = \frac{|\Omega|}{N}\sum_{i=1}^Nf(X_i).
$$
* **Unbiased:** $E[\hat I_N] = I$.
* **Error:** by the CLT $\hat I_N\approx N(I, \sigma^2/N)$ with $\sigma^2 = |\Omega|^2\operatorname{Var}f(X)$, so the standard error is $\hat\sigma/\sqrt N$ — **$O(N^{-1/2})$ in every dimension**. Report $\hat I_N\pm1.96\,\hat\sigma/\sqrt N$.

**Variance reduction.**
* *Antithetic variates:* average $f(U)$ and $f(1-U)$ (negatively correlated when $f$ is monotone).
* *Control variates:* use $f(X) - c\,(h(X) - E[h])$ with known $E[h]$; optimal $c = \operatorname{Cov}(f,h)/\operatorname{Var}(h)$.
* *Importance sampling:* sample $Y\sim q$ and average $f(Y)p(Y)/q(Y)$ — essential for rare events.
* *Quasi-Monte Carlo:* low-discrepancy points (Sobol, Halton) give error close to $O(N^{-1}(\log N)^d)$.

### Examples for §4.10

**Example 4.10.1 ($\pi$ from random points).** Fraction of uniform points in $[-1,1]^2$ inside the unit disc, times 4:

| $N$ | $10^2$ | $10^4$ | $10^6$ |
|---|---|---|---|
| estimate | 2.92 | 3.1636 | 3.14152 |
| std. error | 0.18 | 0.016 | 0.0016 |

100× more samples → 10× smaller error.

**Example 4.10.2 (dimension-independence).** $\int_{[0,1]^d}\prod_{i=1}^d\cos x_i\,d\mathbf x = (\sin1)^d$ with $N = 10^5$:

| $d$ | MC estimate | std. error | exact |
|---|---|---|---|
| 2 | 0.708324 | $5.3\times10^{-4}$ | 0.708073 |
| 5 | 0.421925 | $5.1\times10^{-4}$ | 0.421887 |
| 10 | 0.177999 | $3.1\times10^{-4}$ | 0.177988 |

The error does not grow with $d$ (compare Example 4.8.5).

**Example 4.10.3 (variance reduction).** $\int_0^1e^x\,dx = e-1 = 1.7182818$ with $N = 10^4$:
* plain MC: $1.71953\pm0.0050$;
* antithetic $\frac12[e^U+e^{1-U}]$: $1.71947\pm0.00063$;
* control variate $e^U - c(U-\frac12)$ with $c = \operatorname{Cov}(e^U,U)/\operatorname{Var}U = 1.6903$: $1.71950\pm0.00063$.
Both reductions cut the standard error by ≈8 — equivalent to 64× more samples.

**Example 4.10.4 (importance sampling for a rare event).** $P(Z>4) = 3.167\times10^{-5}$ for $Z\sim N(0,1)$. Plain MC with $10^5$ samples sees only 3 exceedances: estimate $3.0\times10^{-5}$ with ~60% relative error. Sampling $Y\sim N(4,1)$ and weighting by $\frac{\varphi(y)}{\varphi(y-4)} = e^{-4y+8}$ gives $3.170\times10^{-5}\pm0.021\times10^{-5}$ (0.7% relative error).

**Example 4.10.5 (quasi-Monte Carlo).** RMS error of the $\pi$ estimate over 20 runs:

| $N$ | $2^8$ | $2^{10}$ | $2^{12}$ | $2^{14}$ |
|---|---|---|---|---|
| random MC | $1.1\times10^{-1}$ | $5.2\times10^{-2}$ | $2.4\times10^{-2}$ | $1.1\times10^{-2}$ |
| scrambled Sobol | $2.2\times10^{-2}$ | $8.7\times10^{-3}$ | $3.3\times10^{-3}$ | $1.2\times10^{-3}$ |

MC error halves per 4× samples ($N^{-1/2}$); QMC error drops faster (≈$N^{-0.7}$ here, limited by the disc's boundary).

**Example 4.10.6 (a Bayesian posterior predictive).** If a conversion probability has posterior $\theta\sim N(0.3, 0.1^2)$, the probability of at least one conversion in 5 trials is $E[1-(1-\theta)^5]$. Monte Carlo with $5\times10^4$ draws: $0.7978\pm0.0006$; Gauss–Hermite with 20 nodes: $0.79658$ (exact for this degree-5 polynomial). In one dimension quadrature wins; in many dimensions (hierarchical models), MCMC/Monte Carlo is the only option.

---

## Chapter summary

| Task | Method | Error |
|---|---|---|
| $f'(x)$ from values | central difference / five-point | $O(h^2)$ / $O(h^4)$, unstable as $h\to0$ |
| improve any $N(h)$ | Richardson | +1 or +2 orders per level |
| $\int_a^bf$, smooth | composite Simpson, Romberg, Gauss | $O(h^4)$, $O(h^{2k})$, precision $2n-1$ |
| localised features | adaptive quadrature | tolerance-driven |
| periodic | trapezoid | exponential |
| singular / infinite | subtract singularity, $t = 1/x$, Gauss–Laguerre | — |
| $d\ge5$ dimensions | Monte Carlo / QMC | $O(N^{-1/2})$ / ≈$O(N^{-1})$ |

## Further reading

Burden & Faires Ch. 4 · Davis & Rabinowitz, *Methods of Numerical Integration* · A. B. Owen, *Monte Carlo Theory, Methods and Examples* (online) · Baydin et al., "Automatic differentiation in machine learning: a survey", *JMLR* 18 (2018).
