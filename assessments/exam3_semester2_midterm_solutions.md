# Semester 2 Midterm — Solutions and Marking Scheme

> All numbers are verified by [`code/verify_exams.py`](code/verify_exams.py).

**Q1.** (a) [8]

| $k$ | $A\mathbf x^{(k-1)}$ | $\mu^{(k)}$ | $\mathbf x^{(k)}$ | Rayleigh quotient |
|---|---|---|---|---|
| 1 | $(4,5,4)$ | 5 | $(0.8,1,0.8)$ | 4.403509 |
| 2 | $(3.4,4.6,3.4)$ | 4.6 | $(0.73913,1,0.73913)$ | 4.412827 |
| 3 | $(3.21739,4.47826,3.21739)$ | 4.478261 | $(0.718447,1,0.718447)$ | 4.414035 |

The iteration converges to $\lambda_1 = 3+\sqrt2 = 4.414214$ with $\mathbf v = (1/\sqrt2, 1, 1/\sqrt2)$.
(b) The initial vector is symmetric, so it has no component along the eigenvector $(1,0,-1)$ of $\lambda = 3$. The relevant ratio is therefore $\frac{3-\sqrt2}{3+\sqrt2} = 0.359$ for $\mu$, and its square, $0.129$, for the Rayleigh quotient (symmetric $A$). For a general start the ratio would be $\frac{3}{4.414} = 0.68$. [4]
(c) Use inverse iteration with a shift near $1.586$, e.g. $q = 1.5$. The factor is $\frac{|1.586-1.5|}{|3-1.5|} = 0.057$ per step for the vector, and its square for the Rayleigh quotient. The Rayleigh quotients are $1.5974, 1.585819, 1.585787$, against $3-\sqrt2 = 1.585786$. [6]

**Q2.** (a) $\|\mathbf x\| = 5$, and the stable choice is $\alpha = -\operatorname{sign}(x_1)\|\mathbf x\| = -5$. Then $\mathbf w = \frac{\mathbf x-\alpha\mathbf e_1}{\|\cdot\|} = \frac{(8,4)}{\sqrt{80}} = (0.8944, 0.4472)$, $H = \begin{pmatrix}-0.6&-0.8\\-0.8&0.6\end{pmatrix}$ and $H\mathbf x = (-5,0)$. [6]
(b) $Q = \frac1{\sqrt5}\begin{pmatrix}2&-1\\1&2\end{pmatrix}$ (up to signs) and $R = \begin{pmatrix}\sqrt5&4/\sqrt5\\0&3/\sqrt5\end{pmatrix} = \begin{pmatrix}2.2361&1.7889\\0&1.3416\end{pmatrix}$. Then $RQ = \begin{pmatrix}2.8&0.6\\0.6&1.2\end{pmatrix}$; the off-diagonal sign depends on the sign convention. The eigenvalues are 3 and 1, and the off-diagonal entry decays by the factor $\lambda_2/\lambda_1 = \frac13$ per step ($1\to0.6\to0.2\dots$), so the diagonal converges to $(3,1)$. [8]

**Q3.** (a) $A^TA = \begin{pmatrix}3&1\\1&3\end{pmatrix}$ has eigenvalues $4$ and $2$, so $\sigma_1 = 2$ and $\sigma_2 = \sqrt2$, with $\mathbf v_1 = \frac{(1,1)}{\sqrt2}$ and $\mathbf v_2 = \frac{(1,-1)}{\sqrt2}$. [6]
(b) $\mathbf u_1 = A\mathbf v_1/2 = \frac{(1,1,0)}{\sqrt2}$ and $\mathbf u_2 = A\mathbf v_2/\sqrt2 = (0,0,1)$. The condition number is $\kappa_2 = \sqrt2$. The best rank-1 approximation is $\sigma_1\mathbf u_1\mathbf v_1^T = \begin{pmatrix}1&1\\1&1\\0&0\end{pmatrix}$, with 2-norm error $\sqrt2$. [6]
(c) $A^+ = V\Sigma^{-1}U^T = \begin{pmatrix}0.25&0.25&0.5\\0.25&0.25&-0.5\end{pmatrix}$, and $\mathbf x = A^+(1,2,3) = (2.25, -0.75)$. [4]

**Q4.** (a) $F(1.5,1) = (0.25, -0.5)$ and $J = \begin{pmatrix}2x&-1\\1&2y\end{pmatrix} = \begin{pmatrix}3&-1\\1&2\end{pmatrix}$. Solving $J\mathbf s = -F$ gives $\mathbf s = (0, 0.25)$, so $\mathbf x^{(1)} = (1.5, 1.25)$ and $F(\mathbf x^{(1)}) = (0, 0.0625)$. The root is $(1.49257, 1.22777)$. [8]
(b) $\mathbf y = F(\mathbf x^{(1)})-F(\mathbf x^{(0)}) = (-0.25, 0.5625)$, and $A_1 = A_0+\frac{(\mathbf y-A_0\mathbf s)\mathbf s^T}{\mathbf s^T\mathbf s} = \begin{pmatrix}3&-1\\1&2.25\end{pmatrix}$. Compare $J(\mathbf x^{(1)}) = \begin{pmatrix}3&-1\\1&2.5\end{pmatrix}$: only the column along $\mathbf s = \mathbf e_2$ changed, and it is the secant slope, the average of $2y$ over the step. [6]
(c) Advantage: no Jacobian evaluations, and $O(n^2)$ updates (Sherman–Morrison). Disadvantage: only superlinear convergence, and it needs a good initial $A_0$. [2]

**Q5.** (a) $-w_{i-1}+(2+h^2)w_i-w_{i+1} = h^2x_i$ with $2+h^2 = 2.0625$:
$\begin{pmatrix}2.0625&-1&0\\-1&2.0625&-1\\0&-1&2.0625\end{pmatrix}\mathbf w = \begin{pmatrix}0.015625\\0.03125\\0.046875+2\end{pmatrix}$.
The solution is $\mathbf w = (0.465115, 0.943674, 1.449963)$, against the exact $(0.464952, 0.943409, 1.449724)$. The errors are about $2.5\times10^{-4}$. [10]
(b) The method is $O(h^2)$. Richardson extrapolation of results with $h$ and $h/2$, $\frac{4w_{h/2}-w_h}{3}$, gives $O(h^4)$. [3]
(c) The IVP solutions grow like $e^{20x}\approx5\times10^8$, and the combination $y_1+cy_2$ cancels catastrophically. The shooting problem is ill-conditioned. Finite differences or multiple shooting avoid this. [3]

**Q6.** (a) $k = 0.5(0.0625) = 0.03125$. Then $w_j^1 = 0.5(w_{j-1}+w_{j+1})$, giving $(0.5, 0.707107, 0.5)$. The exact values are $e^{-\pi^2(0.03125)}\sin\pi x = (0.519443, 0.734603, 0.519443)$. [6]
(b) $k = 0.0625$. The system is $(1+\lambda)w_j^1-\frac\lambda2(w_{j-1}^1+w_{j+1}^1) = (1-\lambda)w_j^0+\frac\lambda2(w_{j-1}^0+w_{j+1}^0)$, i.e. $2w_j^1-\frac12(w_{j-1}^1+w_{j+1}^1) = \frac12(w_{j-1}^0+w_{j+1}^0)$. The right-hand side is $(0.5, 0.707107, 0.5)$. Solving gives $\mathbf w^1 = (0.386730, 0.546918, 0.386730)$, against the exact $(0.381584, 0.539641, 0.381584)$. [8]
(c) FTCS has $g = 1-4\lambda\sin^2\frac\theta2$, and $|g|\le1$ for all $\theta$ iff $\lambda\le\frac12$. CN has $g = \frac{1-2\lambda s}{1+2\lambda s}$, and $|g|\le1$ for all $\lambda>0$. [6]
