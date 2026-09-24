# Chapter 9 — Worked Examples: Approximating Eigenvalues

Companion script: [`ch09_examples.py`](ch09_examples.py) reproduces every number below. · Lecture: [Chapter 9](../lectures/ch09-approximating-eigenvalues.md)

---

## Example 9.1 — Is a covariance matrix positive definite? Gershgorin says yes

**Problem.** Without computing any eigenvalues, show that
$S = \begin{pmatrix}2&0.3&0.2&0.1\\0.3&1.5&0.4&0\\0.2&0.4&1&0.2\\0.1&0&0.2&0.5\end{pmatrix}$ is positive definite.

**Solution.** The Gershgorin discs are $[1.4,2.6]$, $[0.8,2.2]$, $[0.2,1.8]$ and $[0.2,0.8]$. $S$ is symmetric, so its eigenvalues are real and lie in the union $[0.2, 2.6]$, which is strictly positive. Hence $S$ is PD. The actual eigenvalues are $0.408, 0.858, 1.476, 2.258$.

**Take-away.** Diagonal dominance with positive diagonal gives a free positive-definiteness certificate, and lower and upper bounds on the spectrum, e.g. for step-size selection $\alpha<2/\lambda_{\max}\le2/2.6$.

---

## Example 9.2 — Long-run growth of a population (Leslie model)

**Problem.** A population has four age classes. The fecundities are $(0, 1.2, 1.1, 0.3)$ and the survival rates are $(0.8, 0.7, 0.5)$. Find the long-run growth rate and the stable age distribution.

**Solution.** The power method on the Leslie matrix converges in 36 iterations to $\lambda_1 = 1.231161$, i.e. **23.1% growth per period**. The stable age distribution (normalised eigenvector) is $(0.461, 0.300, 0.170, 0.069)$. The other eigenvalues are $-0.520\pm0.294i$ and $-0.191$, so $|\lambda_2/\lambda_1| = 0.485$. Reaching $10^{-12}$ takes about $\frac{\ln10^{-12}}{\ln0.485} = 38$ iterations, in line with the 36 observed.

**Take-away.** For non-negative irreducible matrices (Perron–Frobenius), the power method finds the positive dominant eigenpair. That eigenpair governs population models, Markov chains, PageRank and input–output economics.

---

## Example 9.3 — Targeting an interior eigenvalue

**Problem.** The matrix $\operatorname{tridiag}(-1,2,-1)$ of size 100 is a discrete Laplacian with eigenvalues $2-2\cos\frac{k\pi}{101}$. Find the eigenvalue closest to $1$.

**Solution.** Inverse iteration with shift $q = 1$ converges to $1.0180118381$ (exact) in 21 iterations. The rate is $\frac{|\lambda_k-q|}{|\lambda_{\text{next}}-q|} = 0.505$ per iteration: the neighbouring eigenvalue on the other side of $q$ is almost as close. Rayleigh-quotient iteration from a nearby vector needs **4** iterations, with cubic convergence.

**Take-away.** Shift-and-invert turns any eigenvalue into the dominant one of $(A-qI)^{-1}$. Each iteration costs one linear solve, so factor $A-qI$ once. This is how `eigsh(..., sigma=q)` finds vibration modes or the small Laplacian eigenvalues used in spectral clustering.

---

## Example 9.4 — Householder + shifted QR in action

**Problem.** Compute all eigenvalues of a random symmetric $6\times6$ matrix via tridiagonalisation and QR steps with the Wilkinson shift.

**Solution.** The Householder reduction produces a tridiagonal $T$ with the same eigenvalues (checked). For the first eigenvalue the last subdiagonal entry evolves as $0.109\to4.3\times10^{-3}\to6.3\times10^{-8}\to$ below $10^{-14}$. The convergence is at least quadratic, and the exponent rises at each step. The numbers of QR steps per deflation are $4, 2, 2, 3, 1$, i.e. 12 steps for 6 eigenvalues. The eigenvalues $-4.155, -2.471, -1.706, 0.335, 0.632, 1.542$ agree with `eigvalsh`.

**Take-away.** About 2 QR iterations per eigenvalue after an $O(n^3)$ reduction: this is why dense symmetric eigenproblems cost only a few times as much as a matrix multiplication.

---

## Example 9.5 — How fast does a Markov chain forget its start?

**Problem.** A web surfer moves among 4 pages with transition matrix
$P = \begin{pmatrix}0.1&0.6&0.3&0\\0.4&0.1&0.4&0.1\\0.3&0.3&0.2&0.2\\0.5&0&0&0.5\end{pmatrix}$.
Find the stationary distribution, and the speed of convergence from page 4.

**Solution.** The left eigenvector for eigenvalue 1 gives $\pi = (0.300, 0.285, 0.255, 0.159)$. The eigenvalue moduli are $1, 0.366, 0.276, 0.010$. The total-variation distance after $1, 2, 5, 10, 20$ steps is $0.54, 0.11, 4.1\times10^{-3}, 1.9\times10^{-5}, \approx0$, and the observed per-step ratio is $0.36\approx|\lambda_2|$.

**Take-away.** The spectral gap $1-|\lambda_2|$ sets the mixing time. It matters for MCMC (how long to run burn-in) and for PageRank (the damping factor bounds $|\lambda_2|\le\alpha$).

---

## Example 9.6 — Denoising a low-rank matrix with the SVD

**Problem.** A $200\times50$ data matrix is rank 3 plus Gaussian noise with $\sigma = 0.5$. How many singular components should be kept?

**Solution.** The singular values are $112.1, 83.8, 71.6$, then $10.4, 10.0, 9.7, \dots$. There is a clear gap after 3. The noise "bulk edge" $\sigma(\sqrt m+\sqrt n) = 10.6$ matches the start of the flat part.

| rank kept | 1 | **3** | 5 | 10 | 50 (all) |
|---|---|---|---|---|---|
| relative error vs truth | 0.70 | **0.087** | 0.127 | 0.184 | 0.319 |

**Take-away.** Truncating at the gap removes most of the noise: the error drops to 27% of that of the raw data. Keeping more components adds back noise. Random-matrix theory predicts the noise edge, a principled threshold for choosing the rank (cf. Gavish–Donoho).
