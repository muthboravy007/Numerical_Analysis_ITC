# Semester 2 — Midterm Examination (Chapters 9–12)

**Time:** 2 hours · **Total:** 100 points · A scientific calculator is allowed, and one handwritten A4 formula sheet. Show all working.

Solutions: [`exam3_semester2_midterm_solutions.md`](exam3_semester2_midterm_solutions.md)

---

**Question 1 (18 points) — Power and inverse power methods.**
$A = \begin{pmatrix}3&1&0\\1&3&1\\0&1&3\end{pmatrix}$ has eigenvalues $3-\sqrt2$, $3$ and $3+\sqrt2$.
(a) Perform three iterations of the power method with $\infty$-norm scaling from $\mathbf x^{(0)} = (1,1,1)^T$. Give $\mu^{(k)}$ and the Rayleigh quotient at each step. (8 pts)
(b) Predict the convergence factor of $\mu^{(k)}$ and of the Rayleigh quotient. (4 pts)
(c) Which shift $q$ and which method would you use to find $3-\sqrt2$ quickly? Predict the convergence factor of the inverse power method with $q = 1.5$. (6 pts)

**Question 2 (14 points) — Householder and QR.**
(a) Construct the Householder matrix $H = I-2\mathbf w\mathbf w^T$ that maps $(3,4)^T$ to a multiple of $\mathbf e_1$, choosing the numerically stable sign. (6 pts)
(b) Perform one step of the unshifted QR algorithm on $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ (compute $Q$, $R$ and $RQ$). Why do the diagonal entries move towards $3$ and $1$? (8 pts)

**Question 3 (16 points) — Singular value decomposition.**
$A = \begin{pmatrix}1&1\\1&1\\1&-1\end{pmatrix}$.
(a) Compute the singular values and right singular vectors from $A^TA$. (6 pts)
(b) Compute the left singular vectors $\mathbf u_1,\mathbf u_2$, the condition number $\kappa_2(A)$, and the best rank-1 approximation. (6 pts)
(c) Compute $A^+$ and the least-squares solution of $A\mathbf x = (1,2,3)^T$. (4 pts)

**Question 4 (16 points) — Nonlinear systems.**
$F(x,y) = (x^2-y-1,\ x+y^2-3)$.
(a) Perform one Newton step from $(1.5, 1.0)$. (8 pts)
(b) Perform the Broyden update of $A_0 = J(\mathbf x^{(0)})$, and compare $A_1$ with $J(\mathbf x^{(1)})$. (6 pts)
(c) State one advantage and one disadvantage of Broyden's method compared with Newton's. (2 pts)

**Question 5 (16 points) — Boundary-value problem.**
$y'' = y-x$ on $[0,1]$, $y(0) = 0$, $y(1) = 2$ (exact $y = x+\frac{\sinh x}{\sinh1}$).
(a) Write the finite-difference equations for $h = 0.25$ as a $3\times3$ linear system and solve it. (10 pts)
(b) What is the order of the method, and how would you improve the accuracy without refining the grid? (3 pts)
(c) Why is linear shooting unreliable for $y'' = 400y$ on $[0,1]$? (3 pts)

**Question 6 (20 points) — Heat equation.**
$u_t = u_{xx}$ on $[0,1]$, $u(0,t) = u(1,t) = 0$, $u(x,0) = \sin\pi x$, $h = 0.25$.
(a) Take one step of the forward-difference (FTCS) method with $\lambda = k/h^2 = 0.5$ and compare with the exact solution $e^{-\pi^2t}\sin\pi x$. (6 pts)
(b) Take one step of Crank–Nicolson with $\lambda = 1$. (8 pts)
(c) Use von Neumann analysis to explain why FTCS requires $\lambda\le\frac12$ while CN is unconditionally stable. (6 pts)
