# Numerical Analysis for Data Science — Course Syllabus

**Level:** Year 3, Data Science major · **Length:** two semesters × 15 weeks (Numerical Analysis I and II) · **Credits:** 4 + 4
**Contact hours per week:** 3 h lectures, 2 h tutorial (hand computation and proofs), 2 h computer lab (Python)

## Course description
This course develops the numerical methods on which modern data science rests: floating-point computation, root finding, interpolation, integration, differential equations, direct and iterative linear algebra, approximation and Fourier analysis, eigenvalues and the SVD, nonlinear systems, BVPs and PDEs. It then applies them in depth to regression, PCA and optimisation. For every method we study
- **why it works:** derivations, convergence theorems, error bounds;
- **when it fails:** conditioning, stability, cost;
- **how it is used on real data.**

The spine of the course follows R. L. Burden & J. D. Faires, *Numerical Analysis* (Chapters 1–12). Three further chapters (13–15) cover regression, symmetric matrices/SVD/PCA and numerical optimisation.

## Prerequisites
Calculus I–III, linear algebra (matrices, eigenvalues, orthogonality), probability and statistics I, and programming in Python (numpy basics). Differential equations are helpful but not required: Chapter 5 is self-contained.

## Learning outcomes
By the end of the course, students can:
1. analyse the rounding error, conditioning and stability of an algorithm, and choose stable formulations;
2. derive the classical numerical methods and prove their convergence orders and error bounds;
3. implement them in vectorised Python, verify them with convergence studies, and compare them with library routines;
4. choose an appropriate method (direct or iterative, explicit or implicit, Newton or quasi-Newton, QR or SVD) given the size, structure and accuracy requirements of a problem;
5. formulate data-science tasks (regression, dimensionality reduction, ranking, model calibration, training) as numerical problems and solve them reliably;
6. communicate numerical results honestly, with error estimates and uncertainty.

## Materials
- **Main text:** Burden, Faires & Burden, *Numerical Analysis*, 10th ed., Cengage (2016).
- **Supplementary:**
  - Trefethen & Bau, *Numerical Linear Algebra* (SIAM);
  - Nocedal & Wright, *Numerical Optimization* (Springer);
  - Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online);
  - Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM).
- **Course repository (this one):**
  - lecture notes in [`lectures/`](lectures/), with 554 worked examples and at least 5 per section;
  - companion scripts in [`lectures/code/`](lectures/code/);
  - extended worked examples in [`examples/`](examples/);
  - homework in [`exercises/`](exercises/), with solutions in [`solutions/`](solutions/);
  - projects in [`projects/`](projects/) and practice exams in [`assessments/`](assessments/);
  - the teaching library [`numlib/`](numlib/), with tests in [`tests/`](tests/).
- **Software:** Python ≥ 3.10 with the packages in [`requirements.txt`](requirements.txt). Jupyter or VS Code.

## Semester 1 — Numerical Analysis I (Chapters 1–8)

| Week | Topic | Reading | Lab | Due |
|---|---|---|---|---|
| 1 | Review of calculus (Taylor, MVT, IVT); floating-point arithmetic, IEEE 754 | Ch 1.1–1.2 | Python/numpy refresher; machine epsilon | — |
| 2 | Errors, conditioning, stability; algorithms and convergence; stable statistics (variance, log-sum-exp) | Ch 1.3–1.4 | Cancellation experiments | HW1 (Ch 1) |
| 3 | Bisection, fixed-point iteration, contraction mapping theorem | Ch 2.1–2.2 | Root-finding toolkit | — |
| 4 | Newton, secant, false position; error analysis and order; multiple roots; Aitken/Steffensen; polynomial zeros (Horner, Müller) | Ch 2.3–2.6 | MLE by Newton; IRR | HW2 (Ch 2) |
| 5 | Lagrange and Newton interpolation, error formula, Runge phenomenon, Chebyshev nodes; Hermite | Ch 3.1–3.4 | Interpolation lab | — |
| 6 | Cubic splines (natural, clamped); parametric curves; splines in regression | Ch 3.5–3.6 | Spline smoothing of data | HW3 (Ch 3) |
| 7 | Numerical differentiation, Richardson extrapolation; Newton–Cotes, composite rules | Ch 4.1–4.4 | Optimal step size; AUC | — |
| 8 | Romberg, adaptive quadrature, Gaussian quadrature; multiple integrals; improper integrals; Monte Carlo | Ch 4.5–4.9 | Expectations by quadrature vs MC | **Midterm (Ch 1–5 up to 5.4)**; HW4 (Ch 4) |
| 9 | IVPs: well-posedness, Euler, Taylor, Runge–Kutta, RKF45 | Ch 5.1–5.5 | Epidemic (SIR) simulation | — |
| 10 | Multistep methods, predictor–corrector; systems and higher order; stability; stiffness | Ch 5.6–5.11 | Stiff solvers; GD as Euler | HW5 (Ch 5) |
| 11 | Gaussian elimination, pivoting, LU, PA = LU; operation counts | Ch 6.1–6.5 | Timing vs LAPACK | Project proposal |
| 12 | Special matrices: SPD, Cholesky, LDLᵀ, banded/tridiagonal; Gaussian processes | Ch 6.6 | Portfolio simulation, GP regression | HW6 (Ch 6) |
| 13 | Norms, eigenvalues; Jacobi, Gauss–Seidel, SOR; condition numbers and error bounds; CG | Ch 7.1–7.6 | Sparse Poisson solvers; ridge by CG | — |
| 14 | Discrete and continuous least squares; orthogonal polynomials; Chebyshev economisation; Padé | Ch 8.1–8.4 | Polynomial regression & model selection | HW7 (Ch 7) |
| 15 | Trigonometric approximation, DFT and FFT; review | Ch 8.5–8.7 | Signal/seasonality lab | HW8 (Ch 8); project progress report |
| — | **Final exam (Ch 1–8)** | | | |

## Semester 2 — Numerical Analysis II (Chapters 9–15)

| Week | Topic | Reading | Lab | Due |
|---|---|---|---|---|
| 1 | Eigenvalues: Gershgorin, orthogonal similarity; power, inverse and Rayleigh-quotient iteration; deflation | Ch 9.1–9.3 | PageRank, Markov chains | — |
| 2 | Householder reduction; QR algorithm with shifts; SVD (theory and computation) | Ch 9.4–9.7 | Spectral clustering | HW9 (Ch 9) |
| 3 | Nonlinear systems: fixed points, Newton, Broyden (quasi-Newton) | Ch 10.1–10.3 | MLE for multi-parameter models | — |
| 4 | Steepest descent, homotopy and continuation; applications (equilibria, EM) | Ch 10.4–10.6 | Market equilibrium, logistic regression by Newton | HW10 (Ch 10) |
| 5 | BVPs: linear and nonlinear shooting; finite differences | Ch 11.1–11.4 | Heat in a rod; Bratu problem | — |
| 6 | Rayleigh–Ritz / finite elements; smoothing and diffusion on graphs | Ch 11.5–11.6 | FEM; Whittaker smoother | HW11 (Ch 11) |
| 7 | Elliptic PDEs (5-point scheme); parabolic PDEs (FTCS, BTCS, Crank–Nicolson, stability) | Ch 12.1–12.2 | Heat equation, image denoising | — |
| 8 | Hyperbolic PDEs, CFL; PDEs in data science (Black–Scholes, Fokker–Planck) | Ch 12.3–12.4 | Option pricing | **Midterm (Ch 9–12)**; HW12 (Ch 12) |
| 9 | Simple and multiple regression: estimation, inference, Gauss–Markov, diagnostics | Ch 13.1–13.2 | Regression with statsmodels vs own QR | — |
| 10 | Numerical least squares (QR, SVD, conditioning); polynomial/spline/feature regression; ridge and lasso | Ch 13.3–13.5 | Diabetes: OLS vs ridge vs lasso | — |
| 11 | WLS, GLS, robust regression; nonlinear LS (Gauss–Newton, LM); GLMs by IRLS | Ch 13.6–13.8 | Poisson/logistic GLMs | HW13 (Ch 13) |
| 12 | Symmetric matrices, spectral theorem, PD matrices, SVD, low-rank approximation | Ch 14.1–14.4 | Image compression, LSA | — |
| 13 | PCA: theory, computation, interpretation; SVD applications (recommenders, whitening, TLS) | Ch 14.5–14.6 | PCA pipeline; randomised SVD | HW14 (Ch 14) |
| 14 | Optimisation: line searches, GD and acceleration, Newton/BFGS/L-BFGS, SGD/Adam | Ch 15.1–15.5 | Training logistic regression and an MLP | — |
| 15 | Constrained optimisation (KKT, projection, proximal methods); automatic differentiation; review | Ch 15.6–15.7 | Autodiff from scratch | HW15 (Ch 15); **project presentations** |
| — | **Final exam (Ch 9–15, cumulative emphasis on 13–15)** | | | |

## Assessment (each semester)

| Component | Weight | Notes |
|---|---|---|
| Homework (8 sets in Semester 1, 7 in Semester 2) | 20% | Sections A–D of the exercise sheets; the lowest score is dropped |
| Lab reports / quizzes | 10% | Short weekly checks; code + 1 paragraph of interpretation |
| Project (teams of 2–3; see [`projects/`](projects/)) | 20% | Proposal 2%, progress report 3%, final report + code 12%, talk 3%. Runs over the second half of Semester 1 or of Semester 2 (one project per student per year is also possible, weighted 20% in Semester 2 only) |
| Midterm exam | 20% | Closed book, calculator, 1 formula sheet |
| Final exam | 30% | Closed book, calculator, 2 formula sheets |

**Grading scale:** A ≥ 85, B ≥ 70, C ≥ 55, D ≥ 45, F < 45 (adjust to institutional policy).

## Homework guidelines
- Section A (theory and proofs) and Section B (hand computation) are submitted handwritten or typeset. Sections C and D (programming and data science) are submitted as code plus a short report with figures.
- Every numerical claim must be backed by output from your code. Report convergence rates on log–log plots.
- Collaboration is encouraged for discussion. Written solutions and code must be your own. Cite any sources.
- Late policy: 10% per day, up to 3 days.

## Use of AI tools
You may use AI assistants to understand concepts or to debug code. You must disclose any such use, and you are responsible for the correctness of everything you submit. Exams test your own understanding without such tools.

## Suggested pacing for a one-semester version
Use Chapters 1, 2, 3.1–3.5, 4.1–4.4, 5.1–5.4, 6, 7.1–7.4, 7.6, 9.1–9.3, 13.1–13.5, 14.3–14.5 and 15.1–15.5. That leaves out BVPs, PDEs and most of Chapters 8 and 10. Assign projects P2, P3, P6 or P7.
