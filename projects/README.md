# Course Projects — Numerical Analysis for Data Science

Each project is designed for **teams of 2–3 students** over **6–8 weeks** (one to two months). All projects follow the same arc:

1. derive the mathematics;
2. implement the core numerical method yourself;
3. validate it against a library and against theory (convergence rates, error bounds);
4. apply it to real data;
5. write it up.

Choose one project. You may also propose your own; see the end of this page.

| # | Project | Main chapters | Difficulty | Data |
|---|---|---|---|---|
| P1 | [Recommender systems by low-rank matrix completion](P1_recommender_matrix_completion.md) | 9, 13, 14, 15 | ★★★ | MovieLens 100K / synthetic |
| P2 | [A regression engine from scratch: stable least squares, regularisation and inference](P2_regression_engine.md) | 6, 7, 13 | ★★ | diabetes, Longley, Ames housing |
| P3 | [Dimensionality reduction for images: PCA, randomised SVD and classification](P3_pca_images.md) | 9, 14 | ★★ | digits, Olivetti faces |
| P4 | [Epidemic modelling: ODE simulation, calibration and forecasting](P4_epidemic_calibration.md) | 5, 10, 13 | ★★★ | Our World in Data / synthetic |
| P5 | [Seasonality and forecasting with Fourier analysis and splines](P5_fourier_forecasting.md) | 3, 8, 13 | ★★ | Mauna Loa CO₂, electricity load |
| P6 | [Graph analytics: PageRank, spectral clustering and label propagation](P6_graph_analytics.md) | 7, 9, 14 | ★★ | SNAP networks, Zachary karate club |
| P7 | [Optimisation algorithms for machine learning](P7_optimization_for_ml.md) | 10, 15 | ★★★ | digits, breast cancer, two moons |
| P8 | [Computational finance: option pricing with PDEs, Monte Carlo and root finding](P8_option_pricing.md) | 2, 4, 5, 12 | ★★★ | Simulated / public option chains |

Starter code for every project is in [`projects/starter/`](starter/). It runs out of the box, produces a baseline result, and marks the places you must extend with `TODO`.

## Common timeline (8 weeks; compress to 6 by merging weeks 6–7)

| Week | Milestone | Checkpoint |
|---|---|---|
| 1 | Read the project brief and background; run the starter code; team contract | 1-page **proposal** (goals, division of work, risks) |
| 2 | Mathematical derivations; toy problems with known answers | Derivation notes checked in |
| 3 | Core implementation v1 with unit tests | Tests pass in CI / `pytest` |
| 4 | Validation: convergence orders, comparison with library, timing | **Progress report** (2 pages) + 10-min meeting |
| 5 | Real data: cleaning, exploratory analysis, first results | Notebook with first results |
| 6 | Extensions (see "going further" in each brief) | — |
| 7 | Sensitivity analysis, uncertainty quantification, ablations | Draft report |
| 8 | Final report, reproducible repository, presentation | **Final submission + 15-min talk** |

## Deliverables

1. **Report** (10–15 pages, PDF, LaTeX or Markdown). Sections: problem and motivation; mathematical background with derivations; algorithms (pseudocode, complexity); validation experiments; data analysis; discussion of limitations; references.
2. **Code repository.** A package or module with docstrings, `pytest` tests, a `requirements.txt`, and one script or notebook that regenerates *every* figure and table. Set random seeds.
3. **Presentation.** 15 minutes plus 5 minutes of questions. Every team member presents and must be able to answer questions about any part.
4. **Individual contribution statement.** Half a page per student.

## Common grading rubric (100 points)

| Criterion | Points | Excellent (full marks) |
|---|---|---|
| Mathematical correctness and depth | 25 | Complete, correct derivations; assumptions stated; theorems cited and *used* (e.g. to predict convergence rates) |
| Implementation quality | 20 | Correct, vectorised, numerically stable; tests cover edge cases; clean API |
| Validation | 20 | Convergence orders and error bounds verified experimentally; agreement with reference libraries quantified; failure modes shown |
| Data analysis and interpretation | 15 | Sensible pre-processing; honest evaluation (held-out data, CV); conclusions supported by evidence |
| Communication | 15 | Clear report and figures (labelled axes, log scales where appropriate); a talk that tells a story |
| Reproducibility | 5 | One command regenerates all results |

The project-specific rubrics in each brief refine the first four rows.

**Academic integrity.** You may use libraries (numpy, scipy, scikit-learn, statsmodels) for *validation* and for peripheral tasks. The core algorithm named in each brief must be your own implementation. Generative-AI tools may be used for coding help, but you must disclose how you used them, and you must be able to explain every line.

## Proposing your own project

Write a one-page proposal. It must contain:
- (a) a data-science question;
- (b) at least two numerical methods from the course that you will implement and compare;
- (c) a validation plan with a known-answer test;
- (d) a data source.

Examples: Kalman filtering for GPS data (Chapters 5, 6); sparse PCA for genomics (Chapters 13–15); physics-informed regression of a heat-transfer coefficient (Chapters 11, 13); topic models via non-negative matrix factorisation (Chapters 14, 15).
