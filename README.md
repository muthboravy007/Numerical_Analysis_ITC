# Numerical Analysis for Data Science (Year 3)

A complete two-semester course: lecture notes, worked examples, homework with full solutions, data-science projects and practice exams. It follows the chapter structure of **Burden & Faires, *Numerical Analysis*** (Chapters 1–12) and extends it with three chapters central to data science: **regression**, **symmetric matrices / SVD / PCA**, and **optimisation**.

Every numerical result in the notes, solutions and exams is produced by a Python script in this repository, so every number can be checked and reproduced.

- 📅 [**Syllabus**](SYLLABUS.md): weekly schedule, assessment, policies.
- 🧪 [**Projects**](projects/README.md): eight 6–8-week data-science projects with rubrics and starter code.
- 📝 [**Assessments**](assessments/README.md): four practice exams with verified solutions.

## Contents by chapter

| Ch | Topic | Lecture notes | Code | Worked examples | Exercises | Solutions |
|---|---|---|---|---|---|---|
| 1 | Mathematical preliminaries and error analysis | [notes](lectures/ch01-mathematical-preliminaries.md) | [py](lectures/code/ch01.py) | [examples](examples/ch01_examples.md) | [ex](exercises/ch01_exercises.md) | [sol](solutions/ch01_solutions.md) |
| 2 | Solutions of equations in one variable | [notes](lectures/ch02-equations-in-one-variable.md) | [py](lectures/code/ch02.py) | [examples](examples/ch02_examples.md) | [ex](exercises/ch02_exercises.md) | [sol](solutions/ch02_solutions.md) |
| 3 | Interpolation and polynomial approximation | [notes](lectures/ch03-interpolation.md) | [py](lectures/code/ch03.py) | [examples](examples/ch03_examples.md) | [ex](exercises/ch03_exercises.md) | [sol](solutions/ch03_solutions.md) |
| 4 | Numerical differentiation and integration | [notes](lectures/ch04-differentiation-integration.md) | [py](lectures/code/ch04.py) | [examples](examples/ch04_examples.md) | [ex](exercises/ch04_exercises.md) | [sol](solutions/ch04_solutions.md) |
| 5 | Initial-value problems for ODEs | [notes](lectures/ch05-initial-value-problems.md) | [py](lectures/code/ch05.py) | [examples](examples/ch05_examples.md) | [ex](exercises/ch05_exercises.md) | [sol](solutions/ch05_solutions.md) |
| 6 | Direct methods for linear systems | [notes](lectures/ch06-direct-linear-systems.md) | [py](lectures/code/ch06.py) | [examples](examples/ch06_examples.md) | [ex](exercises/ch06_exercises.md) | [sol](solutions/ch06_solutions.md) |
| 7 | Iterative techniques in matrix algebra | [notes](lectures/ch07-iterative-techniques.md) | [py](lectures/code/ch07.py) | [examples](examples/ch07_examples.md) | [ex](exercises/ch07_exercises.md) | [sol](solutions/ch07_solutions.md) |
| 8 | Approximation theory, data fitting, FFT | [notes](lectures/ch08-approximation-theory.md) | [py](lectures/code/ch08.py) | [examples](examples/ch08_examples.md) | [ex](exercises/ch08_exercises.md) | [sol](solutions/ch08_solutions.md) |
| 9 | Approximating eigenvalues, SVD | [notes](lectures/ch09-approximating-eigenvalues.md) | [py](lectures/code/ch09.py) | [examples](examples/ch09_examples.md) | [ex](exercises/ch09_exercises.md) | [sol](solutions/ch09_solutions.md) |
| 10 | Nonlinear systems of equations | [notes](lectures/ch10-nonlinear-systems.md) | [py](lectures/code/ch10.py) | [examples](examples/ch10_examples.md) | [ex](exercises/ch10_exercises.md) | [sol](solutions/ch10_solutions.md) |
| 11 | Boundary-value problems | [notes](lectures/ch11-boundary-value-problems.md) | [py](lectures/code/ch11.py) | [examples](examples/ch11_examples.md) | [ex](exercises/ch11_exercises.md) | [sol](solutions/ch11_solutions.md) |
| 12 | Partial differential equations | [notes](lectures/ch12-partial-differential-equations.md) | [py](lectures/code/ch12.py) | [examples](examples/ch12_examples.md) | [ex](exercises/ch12_exercises.md) | [sol](solutions/ch12_solutions.md) |
| 13 | **Regression and least squares** (simple, multiple, polynomial, ridge, lasso, WLS, robust, nonlinear, GLM) | [notes](lectures/ch13-regression-least-squares.md) | [py](lectures/code/ch13.py) | [examples](examples/ch13_examples.md) | [ex](exercises/ch13_exercises.md) | [sol](solutions/ch13_solutions.md) |
| 14 | **Symmetric matrices, SVD and PCA** | [notes](lectures/ch14-symmetric-matrices-svd-pca.md) | [py](lectures/code/ch14.py) | [examples](examples/ch14_examples.md) | [ex](exercises/ch14_exercises.md) | [sol](solutions/ch14_solutions.md) |
| 15 | **Numerical optimisation** (GD, Newton, BFGS, SGD/Adam, KKT, proximal methods, autodiff) | [notes](lectures/ch15-numerical-optimization.md) | [py](lectures/code/ch15.py) | [examples](examples/ch15_examples.md) | [ex](exercises/ch15_exercises.md) | [sol](solutions/ch15_solutions.md) |

Solution code for every exercise set is in [`solutions/code/`](solutions/code/). The **worked examples** files are separate from the lectures. Each holds 6–9 extended, data-science-flavoured problems with full solutions, verified by `examples/chXX_examples.py`.

## What each chapter contains

- **Lecture notes.** Mathematically detailed notes with definitions, theorems and proofs or proof sketches, algorithms, and **at least five worked examples in every section** (554 in total). Each chapter ends with a data-science section that connects the numerical method to practice, e.g. PageRank, Gaussian processes, epidemic models, option pricing, image denoising or model training.
- **Exercises.** Four sections, each problem rated ★ to ★★★:
  - A — theory and proofs;
  - B — hand computation;
  - C — programming;
  - D — data-science applications on real or realistic data.
- **Solutions.** Complete worked solutions. All numbers are generated by the solution scripts, and the discussions explain *why* results come out as they do, including failures such as instability, ill-conditioning and non-identifiability.

## Projects (1–2 months, teams of 2–3)

| # | Project |
|---|---|
| P1 | Recommender systems by low-rank matrix completion (ALS, SGD, soft-impute) |
| P2 | A regression engine from scratch: stable least squares, regularisation, inference (NIST benchmarks) |
| P3 | Dimensionality reduction for images: PCA, randomised SVD, classification |
| P4 | Epidemic modelling: ODE simulation, calibration with sensitivity equations, forecasting |
| P5 | Seasonality and forecasting with Fourier analysis and splines |
| P6 | Graph analytics: PageRank, spectral clustering, label propagation |
| P7 | Optimisation algorithms for machine learning (with your own autodiff) |
| P8 | Computational finance: option pricing with PDEs, Monte Carlo and root finding |

## The `numlib` teaching library

[`numlib/`](numlib/) holds readable reference implementations of every algorithm in the course: floating-point tools, roots, interpolation, quadrature, ODE solvers, direct and iterative linear algebra, least squares, eigenvalue methods, approximation and FFT, nonlinear systems, BVPs, PDEs, regression and optimisation. Students use it to check their own implementations. It is deliberately simple rather than fast; in production, use numpy, scipy, scikit-learn and statsmodels.

## Getting started

```bash
pip install -r requirements.txt
pytest -q tests                     # unit tests of numlib
python lectures/code/ch13.py        # reproduce the regression chapter's examples
python solutions/code/ch09_solutions.py
python projects/starter/p6_graph_analytics.py
python assessments/code/verify_exams.py
```

All scripts run offline. The only data used are generated on the fly or bundled with scikit-learn and statsmodels. Projects optionally download public data sets.

## Reference
R. L. Burden, J. D. Faires & A. M. Burden, *Numerical Analysis*, 10th ed., Cengage Learning, 2016. The notes follow its chapter structure and notation, while all examples, exercises and data applications in this repository are original.
