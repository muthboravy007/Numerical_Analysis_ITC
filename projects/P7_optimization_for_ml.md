# P7 — Optimisation Algorithms for Machine Learning

**Chapters:** 10 (Newton, quasi-Newton), 15 (GD, momentum, SGD, Adam, line searches, AD) · **Difficulty:** ★★★ · **Duration:** 6–8 weeks

## Question
Training a model means minimising a loss. Which optimisers work best for convex models (logistic or softmax regression) and for non-convex neural networks? How well do classical convergence theorems predict what we observe?

## Mathematical background
- **Convex, smooth theory:**
  - GD rates: $O(1/k)$, and linear at rate $(1-\mu/L)^k$ under strong convexity;
  - Nesterov acceleration: $O(1/k^2)$ and $(1-\sqrt{\mu/L})^k$;
  - Newton: locally quadratic convergence, with damping for global convergence;
  - BFGS and L-BFGS: superlinear convergence under Wolfe line searches.
- **Stochastic methods:**
  - SGD with diminishing steps: $O(1/\sqrt k)$ for convex problems, $O(1/k)$ for strongly convex ones;
  - mini-batch variance scaling $\sigma^2/b$;
  - variance reduction (SVRG): linear convergence for finite sums;
  - Adam and its known failure cases.
- **Automatic differentiation:** the reverse mode and its cost. Gradient checking.
- **Conditioning:** the effect of feature scaling (standardisation) on $\kappa$ and hence on the GD rate. Preconditioning with a diagonal Hessian approximation.

## Required tasks
1. Build a small reverse-mode AD library (scalar `Var` or array-based) sufficient for a multilayer perceptron. Gradient-check every operation.
2. Implement GD with backtracking, Nesterov, Newton (with Cholesky and damping), BFGS and L-BFGS with a strong-Wolfe line search, SGD, SGD with momentum, SVRG and Adam.
3. **Convex study:** softmax regression on digits (or MNIST), with ridge penalty $\lambda$. Plot the suboptimality $f-f^\ast$ against iterations, gradient evaluations and wall-clock time, on log scales. Verify the predicted rates and their dependence on $\kappa$ (vary $\lambda$ and the feature scaling).
4. **Non-convex study:** an MLP on two moons and on digits. Compare the optimisers' test accuracy and their sensitivity to the learning rate (learning-rate sweeps) and the batch size. Discuss the generalisation of sharp and flat minima.
5. **Reproducibility:** at least 5 random seeds per configuration, reporting mean ± SD.

## Going further
- Second-order methods for deep learning: Hessian-vector products via AD, and Newton–CG.
- Learning-rate schedules (cosine, warm restarts) and the "edge of stability" ($\lambda_{\max}(\nabla^2f)\approx2/\eta$).
- Constrained learning, e.g. a fairness constraint handled with an augmented Lagrangian.

## Weekly milestones
| Week | Goal |
|---|---|
| 1 | Proposal; starter: logistic regression with GD vs L-BFGS |
| 2 | Derivations: rates for GD/Nesterov, SGD variance, softmax gradient/Hessian |
| 3 | AD library + gradient checks |
| 4 | Deterministic optimisers + convex experiments; progress report |
| 5 | Stochastic optimisers, SVRG |
| 6 | MLP experiments, learning-rate sweeps |
| 7 | Extension; multi-seed statistics |
| 8 | Report + talk |

## Data
- `sklearn.datasets.load_digits`, `load_breast_cancer` and `make_moons` (all offline).
- MNIST (`fetch_openml("mnist_784")`) for the larger runs.

## Project-specific rubric additions
- The AD is correct: all gradient checks pass at a relative error below $10^{-7}$.
- The observed convergence rates are compared *quantitatively* with theory (fitted slopes on log plots).
- The comparisons are fair: equal budgets, tuned hyper-parameters, and multiple seeds.

## Starter code
[`starter/p7_optimization.py`](starter/p7_optimization.py) contains the logistic loss and gradient, GD, and SciPy L-BFGS as a reference, a suboptimality plot, and TODOs for AD, Nesterov, SVRG and Adam.

## References
- Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer (2006).
- Bottou, Curtis & Nocedal, "Optimization methods for large-scale machine learning", *SIAM Review* 60 (2018).
- Baydin et al., "Automatic differentiation in machine learning: a survey", *JMLR* 18 (2018).
