# P3 — Dimensionality Reduction for Images: PCA, Randomised SVD and Classification

**Chapters:** 9 (power/QR methods, SVD), 14 (spectral theorem, Eckart–Young, PCA) · **Difficulty:** ★★ · **Duration:** 6–8 weeks

## Question
How much of an image data set is captured by a few principal components? How do the classic eigenvalue algorithms compare with modern randomised methods at scale? Does dimensionality reduction help or hurt a downstream classifier?

## Mathematical background
- PCA as the eigen-decomposition of the sample covariance, and the equivalent SVD of the centred data matrix (Chapter 14, A5). Variance maximisation is equivalent to minimising reconstruction error.
- **Algorithms:**
  - power iteration with deflation;
  - subspace (block power) iteration;
  - the Lanczos process (via `scipy.sparse.linalg.eigsh`, as a reference);
  - randomised range finder with power iterations (Halko–Martinsson–Tropp). Its error bound is $E\|A-QQ^TA\|\le\bigl(1+\sqrt{\tfrac{k}{p-1}}\bigr)\sigma_{k+1}+\dots$
- **Convergence:** subspace iteration converges at rate $(\sigma_{k+1}/\sigma_k)^{2q+1}$ per block iteration. Davis–Kahan explains unstable individual components when eigenvalues are close.
- **Whitening and Mahalanobis distance;** the "eigenfaces" interpretation.

## Required tasks
1. Implement block power (subspace) iteration and randomised SVD. Verify both against `np.linalg.svd` on the digits data ($1797\times64$): errors of the singular values and angles between the subspaces.
2. **Scaling study:** use Olivetti faces ($400\times4096$), or synthetic matrices up to $20000\times2000$. Plot time and accuracy against $k$, the oversampling $p$ and the power iterations $q$, and compare with the theoretical bound.
3. **Explained variance:** scree plots; reconstruction of faces with $k = 5, 20, 50, 100$; verify Eckart–Young numerically.
4. **Classification pipeline:** logistic regression (or $k$-NN) on the first $k$ PCs. Plot CV accuracy against $k$ and discuss the bias–variance trade-off. Report the confusion matrix.
5. **Stability:** bootstrap the data and measure the variability of each principal direction. Relate it to the eigengaps (Davis–Kahan).

## Going further
- Kernel PCA with an RBF kernel on a non-linear data set (swiss roll, moons). Compare with linear PCA.
- Robust PCA (low-rank + sparse) for images with occlusions, using ADMM.
- Incremental or streaming PCA (Oja's rule as SGD on the Rayleigh quotient).

## Weekly milestones
| Week | Goal |
|---|---|
| 1 | Proposal; starter: PCA on digits, scree plot |
| 2 | Derivations: PCA ⇔ SVD, power/subspace iteration rates, RSVD bound |
| 3 | Implement subspace iteration + RSVD with tests |
| 4 | Scaling study; progress report |
| 5 | Faces: eigenfaces, reconstructions |
| 6 | Classification pipeline, CV |
| 7 | Stability/bootstrap; extension |
| 8 | Report + talk |

## Data
- `sklearn.datasets.load_digits` (offline).
- `sklearn.datasets.fetch_olivetti_faces` (downloads 4 MB once).
- Optional: MNIST (`fetch_openml("mnist_784")`) for the scaling study.

## Project-specific rubric additions
- The convergence rates of subspace iteration are verified against $(\sigma_{k+1}/\sigma_k)^{2q+1}$.
- The RSVD error is compared with the theoretical bound and with the optimal $\sigma_{k+1}$.
- There is a thoughtful discussion of when PCA helps classification and when it hurts.

## Starter code
[`starter/p3_pca_images.py`](starter/p3_pca_images.py) contains digits PCA via SVD, a scree plot, a reference randomised SVD, and TODOs for subspace iteration and the scaling study.

## References
- Halko, Martinsson & Tropp, "Finding structure with randomness", *SIAM Review* 53 (2011).
- Turk & Pentland, "Eigenfaces for recognition", *J. Cognitive Neuroscience* (1991).
- Jolliffe & Cadima, "Principal component analysis: a review", *Phil. Trans. R. Soc. A* (2016).
