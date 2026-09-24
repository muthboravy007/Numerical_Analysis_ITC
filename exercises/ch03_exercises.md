# Chapter 3 — Exercises: Interpolation and Polynomial Approximation

> Lecture: [Chapter 3](../lectures/ch03-interpolation.md) · Solutions: [`solutions/ch03_solutions.md`](../solutions/ch03_solutions.md) · Solution code: [`solutions/code/ch03_solutions.py`](../solutions/code/ch03_solutions.py)
>
> ★ routine · ★★ standard · ★★★ challenging.

## A. Theory and proofs

**A1 ★** Prove that the interpolating polynomial of degree $\le n$ through $n+1$ distinct nodes is unique, and show that $\sum_{k=0}^nL_{n,k}(x)\equiv1$.

**A2 ★★** For linear interpolation on $[x_0,x_1]$, $h = x_1-x_0$, prove $|f(x)-P_1(x)|\le\frac{h^2}{8}\max_{[x_0,x_1]}|f''|$.

**A3 ★★** Prove that $f[x_0,\dots,x_n]$ is a symmetric function of its arguments, and that $f[x_0,\dots,x_n] = \frac{f^{(n)}(\xi)}{n!}$ for some $\xi$ in the span of the nodes. Deduce that the $n$-th divided difference of a polynomial of degree $n$ is its leading coefficient.

**A4 ★★★** Let $S$ be the natural cubic spline interpolant of data $(x_i, y_i)$ and $g\in C^2[a,b]$ any other interpolant. Show $\int_a^b[S''(x)]^2dx\le\int_a^b[g''(x)]^2dx$. (Hint: write $g = S + e$, integrate $\int S''e''$ by parts twice on each subinterval.)

**A5 ★★** Using $T_n(\cos\theta) = \cos n\theta$, derive $T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)$, the zeros and extrema of $T_n$, and show that $\tilde T_n = 2^{1-n}T_n$ alternates $n+1$ times between $\pm2^{1-n}$.

## B. Hand computation

**B1 ★** Find the Lagrange polynomial through $(0,1)$, $(1,3)$, $(3,2)$ and evaluate it at $x = 2$.

**B2 ★★** Use Neville's method with $f(x) = \sqrt x$ and nodes ordered $4, 9, 1, 16$ to approximate $\sqrt5$. Which entry of the table is most accurate, and why?

**B3 ★★** Build the divided-difference table for $x = -1, 0, 1, 2$ and $f = -1, 1, 1, 5$, write the Newton form, and then add the point $(3, 20)$. What does the new coefficient tell you?

**B4 ★** A table of $\ln x$ (6 decimals) is given at $x = 1.0, 1.1, 1.2, 1.3$. Use the Newton forward-difference formula of degree 3 to approximate $\ln1.05$ and state the error.

**B5 ★★** Construct the cubic Hermite interpolant of $f(x) = 1/x$ at $x_0 = 1$, $x_1 = 2$, use it to approximate $f(1.5)$, and compare the actual error with the Hermite error bound.

**B6 ★★** Find the natural cubic spline through $(0,1)$, $(1,2)$, $(2,0)$: give the $M_i$, the coefficients $a_j,b_j,c_j,d_j$, and $S(0.5)$, $S(1.5)$.

**B7 ★★** Repeat B6 for the clamped spline with $S'(0) = S'(2) = 0$. Compare the two splines.

**B8 ★** Evaluate the cubic Bézier curve with control points $(0,0), (1,3), (4,3), (5,0)$ at $t = \frac13$ by de Casteljau's algorithm.

## C. Programming

**C1 ★★** Implement barycentric interpolation. For Runge's function $f(x) = 1/(1+25x^2)$ on $[-1,1]$ compare the maximum error with equispaced and Chebyshev nodes for $n = 5, 10, 20, 40$.

**C2 ★★** Compute the Lebesgue constant $\Lambda_n = \max_x\sum_k|L_{n,k}(x)|$ numerically for equispaced and Chebyshev nodes ($n = 4, 8, 12, 16$) and compare with $\frac2\pi\ln(n+1)+1$.

**C3 ★★** Implement the clamped cubic spline using the Thomas algorithm (Chapter 6). For $\cos x$ on $[0,2\pi]$ with $n = 8, 16, 32, 64$ subintervals verify the $O(h^4)$ rate, and check your result against `scipy.interpolate.CubicSpline(bc_type="clamped")`.

**C4 ★★** A 60-day series $50 + 10\sin(2\pi t/30) + 0.2t$ has 15 random missing days. Fill the gaps by linear interpolation, cubic spline and PCHIP, and compare RMSEs.

## D. Data-science applications

**D1 ★★** Cumulative downloads are recorded at irregular times $t = 0, 1.5, 2, 5, 5.5, 9, 10$ with values $0, 5, 30, 32, 60, 61, 90$. Resample to a fine regular grid with a cubic spline and with PCHIP. Which one produces *decreasing* cumulative counts (impossible), and where? Recommend a method.

**D2 ★** Upscale the $4\times4$ image $\begin{pmatrix}0&0&1&1\\0&1&2&1\\1&2&3&2\\1&1&2&2\end{pmatrix}$ to $8\times8$ with nearest-neighbour and bilinear interpolation (e.g. `scipy.ndimage.zoom` with `order=0/1`). Compare one row and discuss the visual difference.

**D3 ★★★** Validation accuracy as a function of $\log_{10}$(learning rate) is modelled by $a(\ell) = 0.9 - 0.02(\ell+2)^2 + 0.01\sin3\ell$, $\ell\in[-5,1]$ (pretend each evaluation costs an hour of training). Build a surrogate by interpolating 9 Chebyshev-node evaluations, maximise the surrogate, and compare with the true maximiser. Discuss this as the simplest form of Bayesian-optimisation-style hyperparameter search.
