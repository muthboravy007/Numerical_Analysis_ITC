# Semester 1 — Midterm Examination (Chapters 1–5)

**Time:** 2 hours · **Total:** 100 points · A non-programmable scientific calculator is allowed, and one handwritten A4 formula sheet (both sides). Show all working: a correct answer without justification earns at most half the marks.

Solutions: [`exam1_semester1_midterm_solutions.md`](exam1_semester1_midterm_solutions.md)

---

**Question 1 (12 points) — Floating-point arithmetic.**
Let $f(x) = x\bigl(\sqrt{x+1}-\sqrt x\bigr)$.
(a) Evaluate $f(500)$ using four-digit **chopping** arithmetic, and give the relative error. The true value is $11.174755\ldots$ (4 pts)
(b) Rewrite $f$ in an algebraically equivalent form that avoids cancellation, and evaluate it again in four-digit chopping arithmetic. (5 pts)
(c) Explain, using the condition number of subtraction, why (a) fails. (3 pts)

**Question 2 (16 points) — Root finding.**
Let $f(x) = x^3+2x-5$.
(a) Show that $f$ has exactly one root $p$ in $[1,2]$. (3 pts)
(b) Perform three steps of the bisection method on $[1,2]$. (4 pts)
(c) How many bisection steps guarantee $|p_n-p|<10^{-6}$? (3 pts)
(d) Perform two steps of Newton's method from $p_0 = 1.5$. Estimate the number of correct digits after each step, and explain why the growth is quadratic. (6 pts)

**Question 3 (14 points) — Fixed-point iteration.**
Three rearrangements of $x^3+2x-5 = 0$ are
$g_1(x) = \frac{5-x^3}{2}$, $g_2(x) = (5-2x)^{1/3}$, $g_3(x) = \frac5{x^2+2}$.
(a) For each, compute $|g_i'(p)|$ with $p\approx1.3283$, and state whether fixed-point iteration converges locally. (8 pts)
(b) For the convergent one(s), estimate how many iterations are needed per extra correct decimal digit. (3 pts)
(c) Which theorem guarantees convergence for *every* starting point in an interval? State its hypotheses. (3 pts)

**Question 4 (16 points) — Interpolation.**
The function $f(x) = 2^x$ is sampled at $x = 0, 1, 2, 4$.
(a) Build the divided-difference table and write the Newton form of the interpolating polynomial $P_3$. (7 pts)
(b) Estimate $f(3)$ with $P_3$ and compare with the true value. (3 pts)
(c) Use the interpolation error formula to bound $|f(3)-P_3(3)|$, and check that the actual error satisfies the bound. (6 pts)

**Question 5 (18 points) — Numerical integration.**
Let $I = \int_0^1e^{-x^2}dx$ (true value $0.746824$).
(a) Approximate $I$ by the composite trapezoidal rule with $n = 2$ and $n = 4$, and by composite Simpson's rule with $n = 4$. (9 pts)
(b) Give an upper bound for the trapezoidal error with $n = 4$. (4 pts)
(c) Apply one Richardson extrapolation to the two trapezoidal values. What do you notice when you compare the result with Simpson's rule? Explain. (5 pts)

**Question 6 (16 points) — Initial-value problems.**
Let $y' = -2ty$, $y(0) = 1$, with exact solution $y = e^{-t^2}$. With $h = 0.2$, compute $y(0.2)$ by
(a) Euler's method, (b) the modified Euler (Heun) method, and (c) the classical RK4 method.
Give the error of each, and relate the errors to the orders of the methods. (16 pts)

**Question 7 (8 points) — Stability.**
(a) For the test equation $y' = -50y$, find the largest step size for which Euler's method is absolutely stable. (4 pts)
(b) Explain what "stiffness" means and why implicit methods are preferred for stiff problems. Give an example from data science or epidemiology. (4 pts)
