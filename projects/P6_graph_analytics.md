# P6 — Graph Analytics: PageRank, Spectral Clustering and Label Propagation

**Chapters:** 7 (Jacobi, Gauss–Seidel, CG), 9 (power method, eigenvalues), 14 (symmetric matrices, Rayleigh quotients) · **Difficulty:** ★★ · **Duration:** 6–8 weeks

## Question
Networks (the web, social networks, citations) are sparse matrices. How do the three core tasks of ranking, community detection and semi-supervised classification reduce to linear algebra? Which iterative solvers make them scale to millions of nodes?

## Mathematical background
- **PageRank:** $\mathbf x = \alpha P^T\mathbf x+(1-\alpha)\mathbf v$, where $P$ is row-stochastic and dangling nodes need a fix. Two views:
  - the power method on the Google matrix, whose second eigenvalue satisfies $|\lambda_2|\le\alpha$ (Haveliwala–Kamvar);
  - the linear system $(I-\alpha P^T)\mathbf x = (1-\alpha)\mathbf v$, an M-matrix, solved by Jacobi, Gauss–Seidel or GMRES.

  Also study the sensitivity $\|\mathbf x(\alpha)'\|$ as $\alpha\to1$.
- **Spectral clustering:** the graph Laplacian $L = D-W$ is PSD, with $\mathbf f^TL\mathbf f = \frac12\sum w_{ij}(f_i-f_j)^2$. The multiplicity of the eigenvalue $0$ equals the number of connected components. Ratio cut and normalised cut have relaxations whose solutions are the bottom eigenvectors (Courant–Fischer). The Cheeger inequality relates $\lambda_2$ to the conductance.
- **Label propagation:** the harmonic solution $L_{uu}\mathbf f_u = -L_{ul}\mathbf f_l$ (a discrete Dirichlet problem, like Chapter 12's Laplace equation). It has a random-walk interpretation, and it is solved by CG, since $L_{uu}$ is SPD when every component contains a labelled node.

## Required tasks
1. Implement PageRank by the power method, by Jacobi and by Gauss–Seidel on sparse CSR matrices. Compare iterations and time on the web-Stanford or web-Google graph (SNAP) for $\alpha\in\{0.5,0.85,0.95,0.99\}$, and relate the results to theory.
2. Study the sensitivity: how rankings change with $\alpha$ (Kendall's $\tau$ between top-100 lists).
3. **Spectral clustering:** the Zachary karate club (verify the known split); a stochastic block model with a known ground truth (plot ARI against the signal-to-noise ratio, and find the detectability threshold); and a real network (e.g. email-Eu-core, which has department labels).
4. Implement Lanczos yourself, or compare `eigsh` with your own subspace iteration for the bottom eigenvectors of $L_{sym}$. Discuss shift-invert.
5. **Label propagation** with CG on a citation network (Cora) or an email network: accuracy against the fraction of labelled nodes, compared with a feature-only classifier.

## Going further
- Personalised PageRank for recommendation, and local push algorithms.
- HITS (hubs and authorities) via the SVD.
- Graph signal processing: low-pass filtering with Laplacian eigenvectors, or Chebyshev polynomial filters (Chapter 8).

## Weekly milestones
| Week | Goal |
|---|---|
| 1 | Proposal; starter: karate club spectral split + PageRank |
| 2 | Derivations: PageRank convergence, Laplacian properties, relaxation of Ncut, harmonic solution |
| 3 | Sparse PageRank solvers with tests |
| 4 | Large-graph experiments; progress report |
| 5 | Spectral clustering + SBM study |
| 6 | Label propagation with CG |
| 7 | Extension, sensitivity analysis |
| 8 | Report + talk |

## Data
- SNAP: <https://snap.stanford.edu/data/> (web-Stanford, web-Google, email-Eu-core).
- Cora citation network: <https://linqs.org/datasets/>.
- Zachary karate club: `networkx.karate_club_graph()`, or hard-coded in the starter.

## Project-specific rubric additions
- The iteration counts of the PageRank solvers are explained by spectral theory, including $\alpha\to1$.
- The spectral-clustering relaxation is derived and the SBM threshold is shown.
- All matrices are sparse, and memory and time scaling are reported.

## Starter code
[`starter/p6_graph_analytics.py`](starter/p6_graph_analytics.py) contains the karate club graph (edge list included), sparse PageRank by the power method, a Fiedler-vector split, and TODOs for Gauss–Seidel, the SBM study and label propagation.

## References
- Langville & Meyer, *Google's PageRank and Beyond*, Princeton (2006).
- von Luxburg, "A tutorial on spectral clustering", *Statistics and Computing* 17 (2007).
- Zhu, Ghahramani & Lafferty, "Semi-supervised learning using Gaussian fields and harmonic functions", ICML (2003).
