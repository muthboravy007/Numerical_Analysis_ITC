# Semester 1 Final — Solutions and Marking Scheme

> All numbers are verified by [`code/verify_exams.py`](code/verify_exams.py).

**Q1.** (a) The largest entry in column 1 is 4, so swap rows 1 and 2. The multipliers are $m_{21} = \frac14$ and $m_{31} = \frac24$. They give row 2 $= (0, 1.25, -1.25)$ and row 3 $= (0, 0.5, 2.5)$. The pivot in column 2 is $1.25>0.5$, so no swap is needed, and $m_{32} = 0.4$ gives $u_{33} = 2.5+0.4(1.25) = 3$. [8]
$P = \begin{pmatrix}0&1&0\\1&0&0\\0&0&1\end{pmatrix}$, $L = \begin{pmatrix}1&0&0\\0.25&1&0\\0.5&0.4&1\end{pmatrix}$, $U = \begin{pmatrix}4&3&1\\0&1.25&-1.25\\0&0&3\end{pmatrix}$.
(b) $P\mathbf b = (12,2,13)$. Forward substitution gives $\mathbf y = (12, -1, 7.4)$, and back substitution gives $\mathbf x = (17/15, 5/3, 37/15) = (1.1333, 1.6667, 2.4667)$. [5]
(c) $\det A = \det P^{-1}\cdot\prod u_{ii} = (-1)(4)(1.25)(3) = -15$. [3]

**Q2.** (a) The leading minors are $4$, $36$ and $144$, all positive, so $S$ is PD. [4]
(b) $\ell_{11} = 2$, $\ell_{21} = 1$, $\ell_{31} = 1$, $\ell_{22} = \sqrt{10-1} = 3$, $\ell_{32} = (4-1)/3 = 1$ and $\ell_{33} = \sqrt{6-1-1} = 2$. So $L = \begin{pmatrix}2&0&0\\1&3&0\\1&1&2\end{pmatrix}$. Solving $L\mathbf y = (8,16,12)$ gives $\mathbf y = (4,4,4)$, and solving $L^T\mathbf x = \mathbf y$ gives $\mathbf x = (1,1,1)$. [8]

**Q3.** (a) $A$ is strictly diagonally dominant by rows ($5>3$, $6>3$, $7>3$), so both methods converge (Theorem 7.21). [3]
(b) [8]

| $k$ | Jacobi | Gauss–Seidel |
|---|---|---|
| 1 | $(1.2, 0.8333, 1.1429)$ | $(1.2, 0.6333, 0.8905)$ |
| 2 | $(0.9095, 1.0143, 0.9190)$ | $(0.9705, 0.9684, 1.0039)$ |

(c) $\|T_j\|_\infty = \max(\frac35,\frac36,\frac37) = 0.6$ and $\|\mathbf x^{(1)}-\mathbf x^{(0)}\|_\infty = 1.2$. We need $\frac{0.6^k}{0.4}(1.2)\le10^{-6}$, i.e. $k\ge\frac{\ln(3.33\times10^{-7})}{\ln0.6} = 29.2$, so $k = 30$. The actual rate is $\rho(T_j) = 0.39$, so fewer iterations are needed in practice. [3]

**Q4.** (a) $\det A = 0.02$ and $A^{-1} = \begin{pmatrix}50.5&-50\\-100&100\end{pmatrix}$. Then $\|A\|_\infty = 3.01$ and $\|A^{-1}\|_\infty = 200$, so $K_\infty = 602$. [5]
(b) $\mathbf x = (1,1)$ and $\tilde{\mathbf x} = (0.5, 2)$. The relative change in $\mathbf x$ is $\frac{1}{1} = 1$ (100%). The relative change in $\mathbf b$ is $\frac{0.01}{3.01} = 3.3\times10^{-3}$, so the bound is $602\times3.3\times10^{-3} = 2.0$ and the actual change lies within it. [5]
(c) The columns $(2,2)$ and $(1,1.01)$ are nearly parallel. A tiny change in the data swings the coefficients wildly while the fitted values hardly change: the coefficients have huge standard errors (high VIF). Remedies are to drop or combine the predictors, or to regularise (ridge). [2]

**Q5.** (a) $m = 5$, $\sum x = 10$, $\sum y = 15$, $\sum x^2 = 30$, $\sum xy = 39.7$. The normal equations are $5a_0+10a_1 = 15$ and $10a_0+30a_1 = 39.7$. Their solution is $a_1 = 0.97$, $a_0 = 1.06$. Then SSE $= 0.0910$ and $R^2 = 0.9904$. [9]
(b) The drop in SSE is $0.0910-0.0903 = 0.0007$. The test statistic is $F = \frac{0.0007/1}{0.0903/2} = 0.016\ll F_{1,2,0.95} = 18.5$. The quadratic term is not justified: it spends a degree of freedom to fit noise. [4]
(c) $\kappa(X^TX) = \kappa(X)^2$. Forming $X^TX$ can lose twice as many digits, while QR works with $X$ directly and is backward stable (cf. the Longley data). [3]

**Q6.** (a) $x_k = 1+\cos\frac{(2k-1)\pi}{6}$, i.e. $1.8660$, $1$ and $0.1340$. [4]
(b) Map $[0,2]$ to $[-1,1]$ by $t = x-1$. The product is the monic $\tilde T_3(t) = t^3-\frac34t$, whose maximum is $2^{-2} = 0.25$. For the equispaced nodes, $\max|x(x-1)(x-2)| = \frac{2}{3\sqrt3} = 0.385$, attained at $x = 1\pm\frac1{\sqrt3}$. [6]
(c) $|e^x-P_2(x)|\le\frac{\max e^\xi}{3!}(0.25) = \frac{e^2}{6}(0.25) = 0.308$. The actual maximum error is $0.153$. [4]

**Q7.** (a) With $\omega = e^{-i\pi/2} = -i$ and $X_k = \sum x_j(-i)^{jk}$:
- $X_0 = 1+0-1+2 = 2$
- $X_1 = 1-(-1)+2i = 2+2i$, using $(-i)^2 = -1$ and $(-i)^3 = i$
- $X_2 = 1+(-1)(1)+2(-1) = -2$
- $X_3 = 2-2i$

[6]

(b) $\sum|x_j|^2 = 1+0+1+4 = 6$ and $\frac14\sum|X_k|^2 = \frac14(4+8+4+8) = 6$. ✓ [3]
(c) The FFT needs $\frac N2\log_2N = 2^{19}\cdot20\approx1.05\times10^7$, against $N^2\approx1.1\times10^{12}$ for the direct DFT: a speed-up of about $10^5$. [3]

**Q8.** (a) $g(x) = e^x-3x$ has $g(0) = 1>0$ and $g(1) = e-3<0$, so there is a root by the IVT. [3]
(b) $x_{n+1} = x_n-\frac{e^{x_n}-3x_n}{e^{x_n}-3}$ gives $0.6100597$, $0.6189968$, $0.6190613$. [6]
(c) $g'(\ln3) = 3-3 = 0$: the tangent is horizontal and the Newton step is undefined. $\ln3$ is the minimiser of $g$. [3]

**Q9.** [3 each]
(a) A zero or tiny pivot causes division by zero or huge multipliers, which amplify rounding errors (see the example $0.003x+59.14y$). Pivoting keeps $|m_{ij}|\le1$.
(b) It measures $\max|u_{ij}|/\max|a_{ij}|$, the growth of entries during elimination, which bounds the backward error. The worst case is $2^{n-1}$, but it is typically $O(\sqrt n)$ in practice.
(c) CG converges in about $\sqrt\kappa$ iterations against about $\kappa$ for Jacobi. It needs no parameters, and it uses only matrix–vector products, so it exploits sparsity just as well.
(d) By the Euler–Maclaurin formula all boundary correction terms cancel for periodic functions. Equivalently, the trapezoidal rule integrates the Fourier modes $e^{ikx}$ exactly for $|k|<N$, so the error decays like the Fourier coefficients: exponentially for analytic functions.
