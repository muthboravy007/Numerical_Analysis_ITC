# Chapter 8 — Exercises: Approximation Theory

> Lecture: [Chapter 8](../lectures/ch08-approximation-theory.md) · Solutions: [`solutions/ch08_solutions.md`](../solutions/ch08_solutions.md) · Solution code: [`solutions/code/ch08_solutions.py`](../solutions/code/ch08_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★★** Let $A\in\mathbb R^{m\times(n+1)}$ be the Vandermonde matrix $a_{ij} = x_i^{j}$ of the data points. (a) Show that the least-squares error $E(\mathbf a) = \|\mathbf y-A\mathbf a\|_2^2$ is minimised exactly when $A^TA\mathbf a = A^T\mathbf y$. (b) Prove that $A^TA$ is nonsingular if the $x_i$ contain at least $n+1$ distinct values. (c) Show that the fitted residual is orthogonal to every column of $A$.

**A2 ★★** Let $\{\phi_k\}$ be the monic orthogonal polynomials for a weight $w>0$ on $[a,b]$. (a) Prove the three-term recurrence $\phi_{k+1} = (x-B_{k+1})\phi_k - C_{k+1}\phi_{k-1}$. (b) Prove that $\phi_n$ has $n$ distinct real zeros, all in $(a,b)$. (Hint: consider the points where $\phi_n$ changes sign and use orthogonality to $\prod(x-z_i)$.)

**A3 ★★** (a) Using $T_n(\cos\theta) = \cos n\theta$ derive the recurrence $T_{n+1} = 2xT_n-T_{n-1}$, the zeros and the extrema of $T_n$ on $[-1,1]$. (b) Prove the minimax property: among all monic polynomials of degree $n$, $\tilde T_n = 2^{1-n}T_n$ has the smallest maximum absolute value on $[-1,1]$, namely $2^{1-n}$.

**A4 ★★** Let $P_n^\ast = \sum_{k=0}^na_k\phi_k$ be the continuous least-squares approximation of $f$ with respect to an orthogonal basis. (a) Show that $a_k = \langle f,\phi_k\rangle/\langle\phi_k,\phi_k\rangle$ does not depend on $n$. (b) Prove $\|f-P_n^\ast\|^2 = \|f\|^2 - \sum_{k\le n}a_k^2\|\phi_k\|^2$ and deduce Bessel's inequality.

**A5 ★★** Let $C(N)$ be the number of complex multiplications of the radix-2 FFT. Show that $C(N) = 2C(N/2)+N/2$ with $C(1) = 0$, and solve the recurrence to get $C(N) = \frac N2\log_2N$.

**A6 ★★** Let $F$ be the DFT matrix $F_{jk} = \omega^{jk}$, $\omega = e^{-2\pi i/N}$. (a) Prove $F^{\ast}F = NI$. (b) Prove the circular convolution theorem: $\mathcal F(\mathbf x\circledast\mathbf y) = \mathcal F(\mathbf x)\odot\mathcal F(\mathbf y)$. (c) Deduce Parseval's identity $\sum|x_j|^2 = \frac1N\sum|X_k|^2$.

## B. Hand computation

**B1 ★** Fit a least-squares line to $x = 0,1,2,3,4,5$, $y = 2.1, 2.9, 4.2, 4.8, 6.1, 7.2$. Report $E$ and $R^2$.

**B2 ★** Fit a least-squares quadratic to $x = -2,-1,0,1,2$, $y = 4.1, 1.2, 0.1, 0.9, 3.9$. (The symmetric $x$ values make half the normal-equation entries vanish.)

**B3 ★★** Fit $y = be^{ax}$ to $x = 0,\dots,4$, $y = 3.0, 4.4, 6.7, 10.0, 14.9$ by linearisation. Then solve the true nonlinear least-squares problem numerically and compare the two sums of squared errors. Why do they differ?

**B4 ★★** (a) Find the continuous least-squares line for $\sqrt x$ on $[0,1]$ and its $L^2$ error. (b) Find the Legendre least-squares approximations of degrees 0, 2 and 4 to $|x|$ on $[-1,1]$, and tabulate their $L^2$ and maximum errors.

**B5 ★★** Use the three-term recurrence to construct the monic orthogonal polynomials of degrees 0–3 on $[0,1]$ with $w\equiv1$, and of degrees 0–2 with $w(x) = x$.

**B6 ★★** (a) Interpolate $f(x) = \ln(x+2)$ by a quadratic at the zeros of $T_3$. Give the error bound and the actual maximum error, and compare with equispaced nodes $-1,0,1$. (b) Economise the degree-5 Maclaurin polynomial of $\sin x$ to degree 3 on $[-1,1]$. Compare its error with the degree-3 Taylor polynomial.

**B7 ★★** (a) Find the $[2/2]$ Padé approximant of $\cos x$ and compare it with the degree-4 Taylor polynomial at $x = 1$. (b) Find the $[1/1]$ Padé approximant of $\ln(1+x)$ and compare it with the degree-2 Taylor polynomial at $x = 0.5$ and $x = 1$.

**B8 ★★** Compute the discrete least-squares trigonometric polynomial $S_3$ of $f(x) = x$ at the 8 points $x_j = -\pi+j\pi/4$. Compare the $b_k$ with the continuous Fourier coefficients $2(-1)^{k+1}/k$. Why are the $a_k$ nonzero although $f$ is odd?

**B9 ★** Compute by hand the 4-point DFTs of $(2,1,0,1)$, $(0,1,0,-1)$ and $(1,2,3,4)$. Explain the symmetry properties of the first two results.

## C. Programming

**C1 ★★** Fit polynomials of degree $5,10,15,20,30$ to 200 samples of the Runge function $1/(1+25x^2)$ on $[-1,1]$ in three ways: normal equations in the monomial basis, QR (`lstsq`) in the monomial basis, and the Chebyshev basis. Tabulate the condition numbers and the maximum errors.

**C2 ★** Implement the recursive radix-2 FFT and verify it against `numpy.fft.fft`. Time it for $N = 2^8,\dots,2^{16}$ against the naive $O(N^2)$ DFT and numpy.

**C3 ★★★** Compute Chebyshev coefficients from values at Chebyshev extreme points with a DCT-I, for $e^x$, the Runge function, $|x|$ and $\tanh(50x)$. Relate the decay rates to smoothness. For the Runge function, verify that $|c_{k+2}/c_k|\to\rho^{-2}$, where $\rho = 0.2+\sqrt{1.04}$ is the parameter of the Bernstein ellipse through the poles $\pm i/5$.

**C4 ★★** Compare the $[n/n]$ Padé approximants of $e^x$ ($n = 2,3,4$) with the Taylor polynomial of degree $2n$ on $[-1,1]$. Which is better, and where is the Padé error largest?

## D. Data-science applications

**D1 ★★** *Model selection.* Draw 60 noisy samples ($\sigma = 0.3$) of $f(x) = e^x\sin3\pi x$ on $[0,1]$. For degrees 0–12 (Chebyshev basis) compute the training MSE, 5-fold CV MSE, AIC, BIC and the true MSE against $f$. Which criterion picks the best degree?

**D2 ★★** *Seasonality in Mauna Loa CO₂* (`statsmodels.datasets.co2`, monthly means 1958–2001). Remove a quadratic trend, locate the dominant periods with the FFT periodogram, and fit trend + $K$ harmonics ($K = 0,\dots,4$) on all but the last 5 years. Report the training and test RMSE and the annual amplitude.

**D3 ★★** *Transform coding.* Take the grayscale `china.jpg` from `sklearn.datasets.load_sample_image`, compute its orthonormal 2-D DCT, keep the largest 1%, 5%, 10% or 25% of the coefficients, and report the retained energy and the PSNR. Verify Parseval's identity numerically. Why does JPEG use $8\times8$ blocks instead of a global DCT?
