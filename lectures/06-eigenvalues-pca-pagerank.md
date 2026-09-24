# Lecture 6 — Eigenvalue Problems: Power Method, QR Algorithm, PCA and PageRank

> **Week 9** · Code: [`numlib/eigen.py`](../numlib/eigen.py) · Examples: [`examples/ch06`](../examples/ch06_examples.md) · Exercises: [`exercises/ch06`](../exercises/ch06_exercises.md)

## Learning objectives

1. Recall eigenvalues, eigenvectors, diagonalisation and the spectral theorem for symmetric matrices.
2. Locate eigenvalues with Gershgorin discs and understand eigenvalue conditioning.
3. Implement the power method, inverse iteration, shifted inverse iteration and Rayleigh quotient iteration, and state their convergence rates.
4. Describe the QR algorithm as the standard dense eigen-solver.
5. Derive PCA from the SVD / covariance eigen-decomposition and PageRank as a dominant eigenvector problem.

---

## 6.1 Review

$\lambda$ is an eigenvalue of $A\in\mathbb R^{n\times n}$ with eigenvector $\mathbf v\ne\mathbf 0$ if $A\mathbf v = \lambda\mathbf v$. The eigenvalues are the roots of $\det(A-\lambda I)$, but **computing them via the characteristic polynomial is a numerical disaster** (polynomial roots are extremely ill-conditioned — Wilkinson's example). In fact the reverse is done: `np.roots` finds polynomial roots as eigenvalues of a companion matrix.

**Spectral theorem.** A real symmetric matrix has real eigenvalues and an orthonormal basis of eigenvectors: $A = Q\Lambda Q^\top$. Symmetric eigenproblems (covariance matrices, graph Laplacians, Hessians) are well-conditioned: perturbing $A$ by $E$ moves each eigenvalue by at most $\|E\|_2$ (Weyl's theorem).

**Gershgorin circle theorem.** Every eigenvalue of $A$ lies in at least one disc
$$
D_i = \Bigl\{z\in\mathbb C : |z - a_{ii}| \le \sum_{j\ne i}|a_{ij}|\Bigr\}.
$$
Example: $A = \begin{pmatrix}4&1&0\\1&3&1\\0&1&2\end{pmatrix}$ gives discs centred at $4, 3, 2$ with radii $1, 2, 1$; the true eigenvalues are $1.27, 3.00, 4.73$, all in $[1, 5]$.

---

## 6.2 The power method

Repeatedly multiply by $A$ and normalise:
$$
\mathbf x_{k+1} = \frac{A\mathbf x_k}{\|A\mathbf x_k\|},\qquad \lambda^{(k)} = \frac{\mathbf x_k^\top A\mathbf x_k}{\mathbf x_k^\top\mathbf x_k}\ \ (\text{Rayleigh quotient}).
$$

**Why it works.** Expand $\mathbf x_0 = \sum c_i\mathbf v_i$. Then
$$
A^k\mathbf x_0 = \lambda_1^k\Bigl(c_1\mathbf v_1 + \sum_{i\ge2}c_i\bigl(\tfrac{\lambda_i}{\lambda_1}\bigr)^k\mathbf v_i\Bigr).
$$

**Theorem.** If $|\lambda_1| > |\lambda_2| \ge \cdots$ and $c_1 \ne 0$, the direction of $\mathbf x_k$ converges to $\mathbf v_1$ linearly with ratio $|\lambda_2/\lambda_1|$. For symmetric $A$ the Rayleigh quotient converges twice as fast: error $O(|\lambda_2/\lambda_1|^{2k})$.

* Needs only matrix–vector products → works on huge sparse matrices.
* Slow when $|\lambda_2|\approx|\lambda_1|$.

**Deflation.** After finding $(\lambda_1,\mathbf v_1)$ for symmetric $A$, apply the power method to $A - \lambda_1\mathbf v_1\mathbf v_1^\top$ to get $\lambda_2$ (Hotelling deflation). Errors accumulate, so only use it for a few eigenpairs.

---

## 6.3 Inverse and shifted iterations

The eigenvalues of $(A - \sigma I)^{-1}$ are $1/(\lambda_i - \sigma)$ with the same eigenvectors. Applying the power method to it (**shifted inverse iteration**) converges to the eigenvalue **closest to $\sigma$** with ratio
$$
\frac{|\lambda_{\text{closest}} - \sigma|}{|\lambda_{\text{second closest}} - \sigma|}.
$$
Factor $A-\sigma I = LU$ once; each iteration is then two triangular solves.

**Rayleigh quotient iteration** updates the shift every step, $\sigma_k = \lambda^{(k)}$. It converges **cubically** for symmetric matrices — typically 3–4 iterations to machine precision — but you can't control which eigenvalue it finds.

---

## 6.4 The QR algorithm

The standard method for *all* eigenvalues of a dense matrix (`np.linalg.eig`, LAPACK):

```text
A_0 = A
for k = 0, 1, 2, ...
    Q_k R_k = A_k - mu_k I      (QR factorisation)
    A_{k+1} = R_k Q_k + mu_k I
```

Each $A_{k+1} = Q_k^\top A_kQ_k$ is **orthogonally similar** to $A$ (same eigenvalues, backward stable). Under mild conditions $A_k$ converges to (quasi-)upper-triangular Schur form, whose diagonal holds the eigenvalues. Practical implementations

1. first reduce $A$ to **Hessenberg** form (tridiagonal if symmetric) at cost $O(n^3)$, making each QR step $O(n^2)$ (or $O(n)$);
2. use **shifts** (Wilkinson shift) for fast, typically cubic, convergence;
3. **deflate** when a sub-diagonal entry becomes negligible.

Total cost ≈ $10n^3$ for eigenvalues (unsymmetric), $\tfrac43 n^3$ for symmetric eigenvalues only.

For large sparse problems where only a few eigenpairs are needed, use **Lanczos** (symmetric) or **Arnoldi** (general) — `scipy.sparse.linalg.eigsh` / `eigs`.

---

## 6.5 Principal Component Analysis (PCA)

Given centred data $X_c\in\mathbb R^{m\times n}$ ($m$ samples, $n$ features), the sample covariance is $S = \frac{1}{m-1}X_c^\top X_c$. PCA finds orthogonal directions of maximal variance:

$$
\mathbf w_1 = \arg\max_{\|\mathbf w\|=1}\ \mathbf w^\top S\mathbf w \quad\Longrightarrow\quad S\mathbf w_1 = \lambda_1\mathbf w_1 .
$$

The variance along $\mathbf w_1$ is $\lambda_1$ (a Rayleigh quotient argument). Successive components are the next eigenvectors.

**Compute it with the SVD, not the covariance matrix.** If $X_c = U\Sigma V^\top$ then
$$
S = V\frac{\Sigma^2}{m-1}V^\top,
$$
so the principal directions are the columns of $V$, the variances are $\sigma_i^2/(m-1)$, and the **scores** are $X_cV = U\Sigma$. Avoiding $X_c^\top X_c$ avoids squaring the condition number. `sklearn.decomposition.PCA` uses the SVD (randomised SVD for large data).

**Explained variance ratio** of the first $k$ components: $\sum_{i\le k}\sigma_i^2 / \sum_i\sigma_i^2$. Choose $k$ from a scree plot or a threshold (e.g. 95%).

**Standardise first** if features have different units — otherwise PCA just finds the feature with the biggest numbers.

---

## 6.6 PageRank

Model a random surfer on a web graph with $n$ pages. With probability $d$ (≈ 0.85) they follow a random outgoing link; otherwise they jump to a uniformly random page. The transition matrix
$$
G = dP + \frac{1-d}{n}\mathbf 1\mathbf 1^\top,
\qquad P_{ij} = \begin{cases}1/\text{outdeg}(j) & j\to i\\ 0&\text{otherwise}\end{cases}
$$
is column-stochastic (dangling pages with no out-links are treated as linking to all pages). The PageRank vector is the stationary distribution: $G\mathbf r = \mathbf r$, $\sum r_i = 1$ — the eigenvector for eigenvalue $1$.

**Perron–Frobenius.** Because $G$ is positive, $\lambda_1 = 1$ is simple and all other eigenvalues satisfy $|\lambda_i|\le d$. Therefore the **power method converges with rate $d$**: about $\log(10^{-8})/\log(0.85) \approx 113$ iterations for 8 digits — *independently of $n$*. Each iteration is a sparse mat-vec:
$$
\mathbf r_{k+1} = dP\mathbf r_k + \frac{1-d}{n}\mathbf 1 .
$$

The same machinery powers eigenvector centrality in network analysis, spectral clustering (eigenvectors of the graph Laplacian) and the stationary distribution of any ergodic Markov chain.

---

## 6.7 Summary

| Task | Method | Convergence |
|---|---|---|
| Largest $|\lambda|$, sparse | Power method | linear, $|\lambda_2/\lambda_1|$ |
| Eigenvalue near $\sigma$ | Shifted inverse iteration | linear, fast if $\sigma$ good |
| Refine a pair (symmetric) | Rayleigh quotient iteration | cubic |
| All eigenvalues, dense | Shifted QR algorithm | cubic (typ.), $O(n^3)$ |
| Few eigenpairs, sparse | Lanczos / Arnoldi | `eigsh` / `eigs` |

## Further reading

* Trefethen & Bau, Lectures 24–29.
* Langville & Meyer, *Google's PageRank and Beyond*, Princeton, 2006.
* Jolliffe & Cadima, *Principal component analysis: a review and recent developments*, Phil. Trans. R. Soc. A, 2016.
