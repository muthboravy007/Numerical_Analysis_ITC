# Semester 1 — Final Examination (Chapters 1–8, emphasis on 6–8)

**Time:** 3 hours · **Total:** 120 points · A scientific calculator is allowed, and two handwritten A4 formula sheets. Show all working.

Solutions: [`exam2_semester1_final_solutions.md`](exam2_semester1_final_solutions.md)

---

**Question 1 (16 points) — LU with partial pivoting.**
$A = \begin{pmatrix}1&2&-1\\4&3&1\\2&2&3\end{pmatrix}$, $\mathbf b = (2,12,13)^T$.
(a) Compute the factorisation $PA = LU$ with partial pivoting. (8 pts)
(b) Solve $A\mathbf x = \mathbf b$ using the factors. (5 pts)
(c) Compute $\det A$ from the factorisation. (3 pts)

**Question 2 (12 points) — Cholesky.**
$S = \begin{pmatrix}4&2&2\\2&10&4\\2&4&6\end{pmatrix}$.
(a) Show that $S$ is positive definite, using Sylvester's criterion. (4 pts)
(b) Compute the Cholesky factor $L$ and solve $S\mathbf x = (8,16,12)^T$. (8 pts)

**Question 3 (14 points) — Iterative methods.**
$A = \begin{pmatrix}5&-1&2\\1&6&-2\\2&-1&7\end{pmatrix}$, $\mathbf b = (6,5,8)^T$ (solution $(1,1,1)$).
(a) Explain why both the Jacobi and Gauss–Seidel methods converge for any starting vector. (3 pts)
(b) Perform two iterations of each from $\mathbf x^{(0)} = \mathbf 0$. (8 pts)
(c) Compute $\|T_j\|_\infty$ and use it to bound the number of Jacobi iterations needed for an error $\le10^{-6}$ (in the $\infty$-norm). (3 pts)

**Question 4 (12 points) — Conditioning.**
$A = \begin{pmatrix}2&1\\2&1.01\end{pmatrix}$, $\mathbf b = (3, 3.01)^T$.
(a) Compute $A^{-1}$ and $K_\infty(A)$. (5 pts)
(b) Solve $A\mathbf x = \mathbf b$ and $A\tilde{\mathbf x} = (3, 3.02)^T$. Compare the relative change in the solution with the bound $K_\infty\frac{\|\delta\mathbf b\|}{\|\mathbf b\|}$. (5 pts)
(c) Interpret the result for a regression problem with two nearly collinear predictors. (2 pts)

**Question 5 (16 points) — Least squares.**
The data $x = 0,1,2,3,4$, $y = 1.1, 1.9, 3.2, 3.8, 5.0$.
(a) Derive the normal equations for the least-squares line and solve them. Give the SSE and $R^2$. (9 pts)
(b) The least-squares quadratic has SSE $= 0.0903$. Is the quadratic term worth including? Justify your answer with an $F$-test or an argument about degrees of freedom. (4 pts)
(c) Why do statistical packages use QR instead of the normal equations? (3 pts)

**Question 6 (14 points) — Chebyshev interpolation.**
(a) Find the three Chebyshev nodes for quadratic interpolation on $[0,2]$. (4 pts)
(b) Show that $\max_{[0,2]}|(x-x_0)(x-x_1)(x-x_2)|$ equals $0.25$ for these nodes, compared with $0.385$ for equispaced nodes $0,1,2$. (6 pts)
(c) Bound the interpolation error for $f(x) = e^x$ with the Chebyshev nodes. (4 pts)

**Question 7 (12 points) — Discrete Fourier transform.**
(a) Compute the 4-point DFT of $\mathbf x = (1,0,-1,2)$ by hand. (6 pts)
(b) Verify Parseval's identity for this example. (3 pts)
(c) How many complex multiplications does the radix-2 FFT need for $N = 2^{20}$, compared with the direct DFT? (3 pts)

**Question 8 (12 points) — Nonlinear equations (cumulative).**
(a) Show that $e^x = 3x$ has a root in $[0,1]$. (3 pts)
(b) Perform three Newton iterations from $x_0 = 0.5$. (6 pts)
(c) Explain why Newton's method would fail if started at $x_0 = \ln3$. (3 pts)

**Question 9 (12 points) — Short answers.** (3 points each)
(a) Why is partial pivoting needed even when $A$ is nonsingular?
(b) What does the growth factor measure, and what is its worst-case value for partial pivoting?
(c) Why is conjugate gradients preferred to Jacobi for large sparse SPD systems?
(d) Why does the trapezoidal rule converge exponentially fast for smooth periodic functions?
