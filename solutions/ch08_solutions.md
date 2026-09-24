# Chapter 8 — Solutions

> Exercises: [`exercises/ch08_exercises.md`](../exercises/ch08_exercises.md) · Numbers from [`solutions/code/ch08_solutions.py`](code/ch08_solutions.py).

## A. Theory and proofs

**A1.** (a) $E(\mathbf a) = \mathbf y^T\mathbf y - 2\mathbf a^TA^T\mathbf y + \mathbf a^TA^TA\mathbf a$ has gradient $2(A^TA\mathbf a-A^T\mathbf y)$ and Hessian $2A^TA\succeq0$. So $E$ is convex, and its minimisers are exactly the solutions of the normal equations. (b) $\mathbf a^TA^TA\mathbf a = \|A\mathbf a\|^2 = \sum_i p(x_i)^2$, where $p(x) = \sum a_jx^j$. If this is $0$, the polynomial $p$ of degree $\le n$ has at least $n+1$ distinct zeros, so $p\equiv0$ and $\mathbf a = \mathbf 0$. Hence $A^TA$ is positive definite and therefore nonsingular. (c) $A^T(\mathbf y-A\mathbf a) = \mathbf 0$ is the normal equation itself.

**A2.** (a) $x\phi_k$ is a polynomial of degree $k+1$ with leading coefficient 1, so $x\phi_k = \phi_{k+1}+\sum_{j\le k}c_j\phi_j$ with $c_j = \langle x\phi_k,\phi_j\rangle/\|\phi_j\|^2 = \langle\phi_k,x\phi_j\rangle/\|\phi_j\|^2$. For $j\le k-2$, $x\phi_j$ has degree $<k$, so $c_j = 0$. What remains is $c_k = B_{k+1} = \frac{\langle x\phi_k,\phi_k\rangle}{\langle\phi_k,\phi_k\rangle}$ and $c_{k-1} = C_{k+1} = \frac{\langle x\phi_k,\phi_{k-1}\rangle}{\langle\phi_{k-1},\phi_{k-1}\rangle}$. (b) Let $z_1<\dots<z_r$ be the points in $(a,b)$ where $\phi_n$ changes sign. Suppose $r<n$. Then $q = \prod(x-z_i)$ has degree $<n$, so $\langle\phi_n,q\rangle = 0$. But $\phi_nq$ has constant sign on $(a,b)$ and is not identically zero, so $\int w\phi_nq\ne0$, a contradiction. Hence $r = n$: all $n$ zeros are real, simple and lie in $(a,b)$.

**A3.** (a) $\cos(n+1)\theta+\cos(n-1)\theta = 2\cos\theta\cos n\theta$ gives the recurrence. The zeros are $\bar x_k = \cos\frac{(2k-1)\pi}{2n}$, $k = 1..n$. The extrema are $\bar x_k' = \cos\frac{k\pi}{n}$, $k = 0..n$, where $T_n = (-1)^k$. (b) Suppose a monic $P_n$ has $\max|P_n|<2^{1-n}$. Then $Q = \tilde T_n-P_n$ has degree $\le n-1$. At the $n+1$ extreme points, $Q(\bar x_k') = (-1)^k2^{1-n}-P_n(\bar x_k')$ alternates in sign, so $Q$ has $n$ zeros. Hence $Q\equiv0$, which contradicts $\max|P_n|<2^{1-n} = \max|\tilde T_n|$.

**A4.** (a) The normal equations are diagonal in an orthogonal basis: $a_k\langle\phi_k,\phi_k\rangle = \langle f,\phi_k\rangle$. No other $a_j$ appears, so $a_k$ is the same for every $n\ge k$. (b) $f-P_n^\ast\perp\phi_k$ for $k\le n$, so by Pythagoras $\|f\|^2 = \|f-P_n^\ast\|^2+\|P_n^\ast\|^2$ and $\|P_n^\ast\|^2 = \sum a_k^2\|\phi_k\|^2$. Since $\|f-P_n^\ast\|^2\ge0$, $\sum_{k=0}^\infty a_k^2\|\phi_k\|^2\le\|f\|^2$.

**A5.** An $N$-point transform consists of two $N/2$-point transforms plus $N/2$ twiddle multiplications $\omega^k\cdot O_k$. Each twiddle product is used twice, in $E_k\pm\omega^kO_k$. So $C(N) = 2C(N/2)+N/2$. With $N = 2^p$ and $c_p = C(2^p)/2^p$ we get $c_p = c_{p-1}+\frac12$, so $c_p = \frac p2$ and $C(N) = \frac N2\log_2N$. (A naive DFT needs $N^2$.)

**A6.** (a) $(F^\ast F)_{jk} = \sum_l\bar\omega^{lj}\omega^{lk} = \sum_l\omega^{l(k-j)}$. This sum is $N$ if $j = k$. Otherwise it is a geometric series with ratio $\omega^{k-j}\ne1$ and $\omega^{N(k-j)} = 1$, so it is $0$. (b) $\sum_j(\mathbf x\circledast\mathbf y)_j\omega^{jk} = \sum_j\sum_lx_ly_{j-l}\omega^{lk}\omega^{(j-l)k} = X_kY_k$, using periodicity of the indices mod $N$. (c) $\|F\mathbf x\|^2 = \mathbf x^\ast F^\ast F\mathbf x = N\|\mathbf x\|^2$.

## B. Hand computation

**B1.** $m = 6$, $\sum x = 15$, $\sum y = 27.3$, $\sum x^2 = 55$, $\sum xy = 86.1$. Then $a_1 = \frac{6(86.1)-15(27.3)}{6(55)-225} = \frac{107.1}{105} = 1.02$ and $a_0 = \frac{27.3-15.3}{6} = 2.0$. So $y = 2.0+1.02x$, $E = 0.128$, $R^2 = 0.9930$.

**B2.** $\sum x^k = (5,0,10,0,34)$, $\sum y = 10.2$, $\sum xy = -0.7$, $\sum x^2y = 34.1$. The normal equations decouple: $10a_1 = -0.7$, and $\{5a_0+10a_2 = 10.2,\ 10a_0+34a_2 = 34.1\}$. Hence $a_1 = -0.07$, $a_2 = 0.978571$, $a_0 = 0.082857$, and $E = 0.016571$.

**B3.** Linearised fit: $\ln b = $ intercept and $a$ = slope of $\ln y$ on $x$, giving $b = 2.98010$, $a = 0.40265$, SSE $= 0.005797$. Nonlinear least squares gives $b = 2.98401$, $a = 0.40227$, SSE $= 0.005737$, which is necessarily smaller. The linearised fit minimises squared errors in $\ln y$, i.e. *relative* errors. That weights the small $y$ values more heavily than the original criterion does.

**B4.** (a) The Gram matrix is $\begin{pmatrix}1&\frac12\\\frac12&\frac13\end{pmatrix}$ and the right-hand side is $(\frac23,\frac25)$. The line is $P(x) = \frac4{15}+\frac45x$. The $L^2$ error is $\sqrt{1/450} = 0.0471$ (squared error $0.00222$). The maximum error, $0.267$, occurs at $x = 0$, where $\sqrt x$ has infinite slope. (b) $a_0 = \frac12$, $a_2 = \frac58$, $a_4 = -\frac3{16}$, and the odd coefficients are $0$. So $P_4 = \frac12+\frac58P_2(x)-\frac3{16}P_4(x) = 0.117188+1.640625x^2-0.820313x^4$.

| degree | $L^2$ error | max error |
|---|---|---|
| 0 | 0.408248 | 0.500 |
| 2 | 0.102062 | 0.1875 |
| 4 | 0.051031 | 0.1172 |

The maximum error occurs at the kink $x = 0$ and decays only like $1/n$.

**B5.** For $w = 1$ on $[0,1]$: $\phi_0 = 1$, $\phi_1 = x-\frac12$, $\phi_2 = x^2-x+\frac16$, $\phi_3 = x^3-\frac32x^2+\frac35x-\frac1{20}$ (monic shifted Legendre polynomials). For $w = x$: $\phi_1 = x-\frac23$ and $\phi_2 = x^2-\frac65x+\frac3{10}$.

**B6.** (a) The nodes are $\pm\frac{\sqrt3}{2}, 0$ and the interpolant is $P_2(x) = 0.693147+0.535318x-0.138426x^2$. The bound is $\frac{\max|f'''|}{3!\,2^2} = \frac{2}{24} = 0.0833$, since $f''' = 2/(x+2)^3\le2$. The actual maximum error is $0.0194$; with equispaced nodes it is $0.0248$. (b) Subtract $\frac1{120}\tilde T_5 = \frac1{120}\cdot\frac{T_5}{16}$ from $P_5 = x-\frac{x^3}{6}+\frac{x^5}{120}$. This gives $0.997396x-0.15625x^3$, adding at most $\frac1{1920} = 5.2\times10^{-4}$ to the error.

| degree-3 polynomial | max error on $[-1,1]$ |
|---|---|
| economised | $5.7\times10^{-4}$ |
| degree-3 Taylor | $8.1\times10^{-3}$ |
| degree-5 Taylor (for reference) | $2.0\times10^{-4}$ |

**B7.** (a) $r(x) = \dfrac{1-\frac5{12}x^2}{1+\frac1{12}x^2}$, so $r(1) = \frac7{13} = 0.538462$ against $\cos1 = 0.540302$ (error $1.8\times10^{-3}$). The Taylor polynomial of degree 4 does slightly better here (error $1.4\times10^{-3}$). Padé is not automatically better for entire functions near the origin. (b) $r(x) = \frac{x}{1+x/2} = \frac{2x}{2+x}$.

| | $x = 0.5$ | $x = 1$ |
|---|---|---|
| exact $\ln(1+x)$ | 0.405465 | 0.693147 |
| Padé $[1/1]$ | 0.4 (error 0.0055) | 0.6667 (error 0.026) |
| Taylor degree 2 | 0.375 (error 0.030) | 0.5 (error 0.19) |

Here Padé is much better, because the rational form mimics the singularity at $x = -1$.

**B8.** $b = (1.896119, -0.785398, 0.325323)$, against the Fourier values $(2,-1,0.667)$. The coefficients are $a_k = -\frac\pi4(-1)^k$, i.e. $a_0 = -0.785$, $a_1 = 0.785$, $a_2 = -0.785$, $a_3 = 0.785$. The grid includes $x_0 = -\pi$ but not $+\pi$. The sample there is the value of the periodic extension at its jump, and it has no symmetric partner, so the discrete data are not odd. Only the $j = 0$ term survives in $a_k = \frac14\sum x_j\cos kx_j$. Aliasing also shrinks the $b_k$. (Sampling the jump point at the average value $0$ restores oddness.)

**B9.** $\omega = -i$.
- $(2,1,0,1)\mapsto(4,2,0,2)$ is real, because the input is real and even ($x_j = x_{N-j}$).
- $(0,1,0,-1)\mapsto(0,-2i,0,2i)$ is purely imaginary, because the input is real and odd.
- $(1,2,3,4)\mapsto(10,-2+2i,-2,-2-2i)$ is conjugate-symmetric ($X_{N-k} = \bar X_k$), as for every real input.

## C. Programming (key results)

**C1.**

| $n$ | cond$(V)$ | cond$(V^TV)$ | cond(Chebyshev) | max error: normal equations | max error: QR | max error: Chebyshev |
|---|---|---|---|---|---|---|
| 5 | 43 | $1.8\times10^3$ | 2.8 | 0.332 | 0.332 | 0.332 |
| 10 | $3.0\times10^3$ | $8.9\times10^6$ | 3.2 | 0.102 | 0.102 | 0.102 |
| 20 | $1.7\times10^7$ | $2.9\times10^{14}$ | 3.5 | 0.0140 | 0.0140 | 0.0140 |
| 30 | $1.1\times10^{11}$ | $6.3\times10^{18}$ | 3.8 | **0.0106** | 0.0019 | 0.0019 |

By degree 30 the normal equations have lost all accuracy (cond $>1/u$). QR still works, but with cond $10^{11}$ it keeps only about 5 digits in the coefficients. The Chebyshev basis is perfectly conditioned. Least squares at 200 points avoids Runge's divergence, unlike interpolation.

**C2.** The radix-2 FFT agrees with numpy to $10^{-13}$.

| $N$ | radix-2 (ours) | numpy | naive DFT |
|---|---|---|---|
| $2^8$ | 2.2 ms | 0.06 ms | 3.8 ms |
| $2^{10}$ | 8.8 ms | 0.08 ms | 0.21 s |
| $2^{12}$ | 34 ms | 0.15 ms | 3.6 s |
| $2^{16}$ | 0.61 s | 2.5 ms | — (≈15 min extrapolated) |

The naive DFT grows ×16 per ×4 in $N$ ($N^2$); the FFT grows ×4.3 ($N\log N$).

**C3.** $\max(|c_k|,|c_{k+1}|)$:

| $f$ | $k = 10$ | $k = 50$ | $k = 200$ | last $k$ above $10^{-12}$ |
|---|---|---|---|---|
| $e^x$ (entire) | $5.5\times10^{-10}$ | $10^{-18}$ | $10^{-18}$ | 11 |
| Runge (poles at $\pm i/5$) | $5.4\times10^{-2}$ | $1.9\times10^{-5}$ | $10^{-18}$ | 140 |
| $\tanh50x$ (poles at $\pm i\pi/100$) | 0.11 | 0.017 | $1.4\times10^{-4}$ | 791 |
| $|x|$ (kink) | $1.3\times10^{-2}$ | $5.1\times10^{-4}$ | $3.2\times10^{-5}$ | $>2048$ |

- The entire function has superexponential decay.
- Analytic functions decay geometrically at rate $\rho^{-k}$. For the Runge function, $|c_{k+2}/c_k| = 0.672078$, which matches $\rho^{-2} = (0.2+\sqrt{1.04})^{-2}$ to 10 digits. $\tanh50x$ has poles very close to $[-1,1]$, so $\rho\approx1.03$ and the decay is slow.
- The kink gives only algebraic decay, $O(k^{-2})$.

**C4.**

| | Padé max error | Taylor max error (degree $2n$) |
|---|---|---|
| $[2/2]$ | $4.0\times10^{-3}$ | $9.9\times10^{-3}$ |
| $[3/3]$ | $2.8\times10^{-5}$ | $2.3\times10^{-4}$ |
| $[4/4]$ | $1.1\times10^{-7}$ | $3.1\times10^{-6}$ |

Padé is 2.5–28× better for the same number of coefficients, and its advantage grows with $n$. Its error is largest at $x = 1$, where $e^x$ is largest; at $x = -1$ it is about 7× smaller.

## D. Data-science applications

**D1.**

| degree | train MSE | CV MSE | AIC | BIC | true MSE |
|---|---|---|---|---|---|
| 2 | 0.968 | 1.061 | 4.1 | 10.4 | 0.999 |
| 4 | 0.116 | 0.146 | −119.1 | −108.6 | 0.122 |
| 6 | 0.079 | 0.100 | −138.0 | **−123.3** | **0.011** |
| 7 | 0.076 | **0.098** | −138.5 | −121.8 | 0.014 |
| 8 | 0.073 | 0.146 | **−138.7** | −119.8 | 0.059 |
| 12 | 0.071 | 0.138 | −133.2 | −105.9 | 0.018 |

The training MSE decreases monotonically. BIC picks degree 6, which is the true optimum. CV picks 7, which is close. AIC picks 8, because its penalty $2k$ is weaker than BIC's $k\ln n$, so it tends to overfit. The CV curve is noisy (0.146 vs 0.098 between neighbouring degrees): with $n = 60$, repeated CV or the one-standard-error rule is advisable.

**D2.** The series has 526 monthly means. The periodogram peaks at $1.004$ and $2.008$ cycles/yr (periods 11.95 and 5.98 months), i.e. the annual cycle and its first harmonic. The third peak is leftover trend.

| $K$ | train RMSE | test RMSE (last 5 years) |
|---|---|---|
| 0 | 2.140 | 2.493 |
| 1 | 0.890 | 1.512 |
| 2 | 0.728 | 1.395 |
| 3 | 0.725 | 1.390 |
| 4 | 0.724 | 1.389 |

The annual amplitude is $2.75$ ppm. Two harmonics capture the seasonal shape, and more add nothing. The remaining test error comes from extrapolating the quadratic trend, not from seasonality: the Fourier part is excellent, and a better trend model (piecewise or local) is where to improve.

**D3.**

| coefficients kept | energy retained | PSNR |
|---|---|---|
| 1% | 0.9798 | 20.6 dB |
| 5% | 0.9875 | 22.7 dB |
| 10% | 0.9912 | 24.3 dB |
| 25% | 0.9962 | 27.9 dB |

Parseval holds: $\sum\text{pixel}^2 = \sum\text{coef}^2 = 7.610\times10^9$. A global DCT spreads the edges of every object over all frequencies, so the coefficients decay slowly (like $|x|$ in C3). JPEG's $8\times8$ blocks localise edges, so most blocks are smooth and need only a few coefficients. The block transform is also $O(N)$, and quantisation can adapt to local content.
