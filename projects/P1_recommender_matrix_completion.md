# P1 — Recommender Systems by Low-Rank Matrix Completion

**Chapters:** 9 (SVD), 13 (regularised least squares), 14 (low-rank approximation), 15 (SGD, alternating minimisation) · **Difficulty:** ★★★ · **Duration:** 6–8 weeks

## Question
Given a sparse matrix of user ratings $M\in\mathbb R^{m\times n}$ with only $|\Omega|\ll mn$ observed entries, predict the missing ratings. Which numerical method gives the best accuracy for a given computational budget, and why?

## Mathematical background
- **Low-rank model:** $M\approx UV^T$ with $U\in\mathbb R^{m\times k}$, $V\in\mathbb R^{n\times k}$, $k\ll\min(m,n)$. Eckart–Young (Chapter 14) gives the best rank-$k$ approximation of a *fully observed* matrix. With missing entries, the problem is non-convex.
- **Regularised factorisation:**
$$\min_{U,V}\ \tfrac12\sum_{(i,j)\in\Omega}(m_{ij}-\mathbf u_i^T\mathbf v_j)^2+\tfrac\lambda2(\|U\|_F^2+\|V\|_F^2).$$
With $V$ fixed, each row $\mathbf u_i$ solves a ridge-regression problem, which gives **alternating least squares (ALS)**. SGD updates one observed entry at a time.
- **Convex relaxation:** $\min_Z\frac12\|P_\Omega(M-Z)\|_F^2+\lambda\|Z\|_*$ with the nuclear norm $\|Z\|_* = \sum\sigma_i$. Its proximal operator is singular-value soft-thresholding. This gives **soft-impute** (proximal gradient, Chapter 15).
- Useful facts to prove in the report:
  - (i) the ALS subproblems are strictly convex for $\lambda>0$;
  - (ii) the objective is non-increasing under ALS;
  - (iii) $\operatorname{prox}_{\lambda\|\cdot\|_*}(Y) = U\operatorname{diag}((\sigma_i-\lambda)_+)V^T$;
  - (iv) the identity $\|Z\|_* = \min_{UV^T = Z}\frac12(\|U\|_F^2+\|V\|_F^2)$, which links the two formulations.

## Required tasks
1. **Baselines:** the global mean; user and item biases $\hat m_{ij} = \mu+b_i+c_j$, fitted by regularised least squares.
2. **Implement three solvers:** ALS, SGD (with learning-rate schedule and shuffling), and soft-impute (with a warm-started $\lambda$ path). Use sparse data structures: do not form dense $m\times n$ matrices, except for soft-impute on small data, where you should explain the "sparse + low-rank" trick.
3. **Synthetic validation:** generate $M = UV^T+$ noise with known rank. Show the recovery error as a function of the sampling fraction, the rank and the noise. Compare with the rough information-theoretic threshold $|\Omega|\gtrsim k(m+n)$.
4. **Real data (MovieLens 100K):** use a random 80/10/10 split. Tune $k$ and $\lambda$ on the validation set and report test RMSE for every method. Plot RMSE against wall-clock time.
5. **Analysis:** convergence curves; sensitivity to initialisation (non-convexity); the singular-value spectrum of the completed matrix; a qualitative look at the latent factors (which movies load on factor 1?).

## Going further (choose at least one)
- Implicit feedback and weighted ALS.
- The cold-start problem: side information (genres) as extra regression features.
- Randomised SVD (Chapter 14 C1) inside soft-impute for larger data (MovieLens 1M).
- Fairness or popularity bias: is the error uniform across users with few and many ratings?

## Weekly milestones
| Week | Goal |
|---|---|
| 1 | Run starter; load MovieLens; baseline RMSE; proposal |
| 2 | Derive ALS normal equations, SGD updates, soft-thresholding prox; prove (i)–(iii) |
| 3 | ALS + SGD on synthetic data with unit tests (exact recovery in the noiseless full-rank-sampling case) |
| 4 | Soft-impute; recovery-vs-sampling phase diagram; progress report |
| 5 | MovieLens experiments, hyper-parameter tuning |
| 6 | Extension |
| 7 | Timing/accuracy trade-offs, factor interpretation, robustness |
| 8 | Report + talk |

## Data
- MovieLens 100K: <https://grouplens.org/datasets/movielens/100k/> (943 users, 1682 movies, 100 000 ratings; file `u.data`).
- The starter script falls back to a synthetic generator when no network connection is available.

## Project-specific rubric additions
- The derivations of ALS and soft-impute are complete, including a proof of monotone decrease.
- The phase diagram demonstrates when recovery succeeds.
- The comparison of methods on MovieLens is fair: the same split, and tuning only on the validation set.

## Starter code
[`starter/p1_matrix_completion.py`](starter/p1_matrix_completion.py) contains a data loader (MovieLens or synthetic), a bias baseline, and a reference ALS implementation that you must extend (SGD and soft-impute are TODOs).

## References
- Koren, Bell & Volinsky, "Matrix factorization techniques for recommender systems", *IEEE Computer* (2009).
- Mazumder, Hastie & Tibshirani, "Spectral regularization algorithms for learning large incomplete matrices", *JMLR* 11 (2010).
- Candès & Recht, "Exact matrix completion via convex optimization", *Found. Comput. Math.* (2009).
