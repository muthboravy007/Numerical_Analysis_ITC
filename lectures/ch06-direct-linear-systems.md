# Chapter 6 — Direct Methods for Solving Linear Systems

> **Reference:** Burden & Faires, Chapter 6 (§6.1–6.6) plus §6.7 for data science.
> **Code:** [`numlib/linalg_direct.py`](../numlib/linalg_direct.py) · **All numbers reproduced by** [`lectures/code/ch06.py`](code/ch06.py)
> **Worked problems:** [`examples/ch06`](../examples/ch06_examples.md) · **Homework:** [`exercises/ch06`](../exercises/ch06_exercises.md)

## Learning outcomes

1. Solve $A\mathbf x = \mathbf b$ by Gaussian elimination with backward substitution and count its operations.
2. Explain why pivoting is needed; apply partial, scaled partial and complete pivoting.
3. Use matrix algebra, inverses and determinants correctly — and know when *not* to compute them.
4. Factor $A = LU$ and $PA = LU$; reuse a factorisation for many right-hand sides.
5. Recognise diagonally dominant, positive definite, symmetric and banded matrices and apply Cholesky, $LDL^T$ and Crout/Thomas factorisations.
6. Apply these tools to multivariate Gaussians, Gaussian processes, Markov chains and regression.

Notation: $A = (a_{ij})\in\mathbb R^{n\times n}$, the **augmented matrix** $[A\,|\,\mathbf b]$, rows $E_1,\dots,E_n$.

---

## 6.1 Linear Systems of Equations

**Elementary row operations** (which do not change the solution set): (1) $(\lambda E_i)\to(E_i)$, $\lambda\ne0$; (2) $(E_i+\lambda E_j)\to(E_i)$; (3) $(E_i)\leftrightarrow(E_j)$.

**Gaussian elimination with backward substitution (B&F Algorithm 6.1).** For $i = 1,\dots,n-1$: choose a pivot row $p\ge i$ with $a_{pi}\ne0$ (swap if $p\ne i$); for $j = i+1,\dots,n$, compute the **multiplier** $m_{ji} = a_{ji}/a_{ii}$ and perform $(E_j - m_{ji}E_i)\to(E_j)$. Then back-substitute:
$$
x_n = \frac{a_{n,n+1}}{a_{nn}},\qquad x_i = \frac{a_{i,n+1} - \sum_{j=i+1}^na_{ij}x_j}{a_{ii}},\quad i = n-1,\dots,1.
$$
If no nonzero pivot can be found, the system has no unique solution.

**Operation count.**
$$
\text{multiplications/divisions: }\frac{n^3}{3} + n^2 - \frac n3,\qquad \text{additions/subtractions: }\frac{n^3}{3} + \frac{n^2}{2} - \frac{5n}{6}.
$$
The cost is $O(n^3)$ — doubling $n$ multiplies the work by 8.

### Examples for §6.1

**Example 6.1.1 (3×3 by hand).**
$$
\left[\begin{array}{rrr|r}2&1&1&5\\4&-6&0&-2\\-2&7&2&9\end{array}\right]
\xrightarrow[m_{31} = -1]{m_{21} = 2}
\left[\begin{array}{rrr|r}2&1&1&5\\0&-8&-2&-12\\0&8&3&14\end{array}\right]
\xrightarrow{m_{32} = -1}
\left[\begin{array}{rrr|r}2&1&1&5\\0&-8&-2&-12\\0&0&1&2\end{array}\right].
$$
Back substitution: $x_3 = 2$, $x_2 = \frac{-12+2(2)}{-8} = 1$, $x_1 = \frac{5 - 1 - 2}{2} = 1$. So $\mathbf x = (1,1,2)^T$.

**Example 6.1.2 (4×4).** $A = \begin{pmatrix}4&-1&1&2\\2&5&-1&1\\1&1&6&-2\\-1&2&1&7\end{pmatrix}$, $\mathbf b = (9,-4,10,6)^T$. After the three elimination steps the augmented matrix is
$$
\left[\begin{array}{rrrr|r}4&-1&1&2&9\\0&5.5&-1.5&0&-8.5\\0&0&6.0909&-2.5&9.6818\\0&0&0&8.2090&8.2090\end{array}\right],
$$
giving $x_4 = 1$, $x_3 = 2$, $x_2 = -1$, $x_1 = 1$.

**Example 6.1.3 (no unique solution).** For $A = \begin{pmatrix}1&1&1\\2&2&1\\3&3&2\end{pmatrix}$ the first two columns are equal: after step 1 the (2,2) and (3,2) entries are both zero, so no pivot exists in column 2. Indeed $\det A = 0$ and $\operatorname{rank}A = 2$: depending on $\mathbf b$ there are no solutions or infinitely many.

**Example 6.1.4 (operation counts).**

| $n$ | mult/div | add/sub |
|---|---|---|
| 3 | 17 | 11 |
| 10 | 430 | 375 |
| 100 | $3.43\times10^5$ | $3.38\times10^5$ |
| 1000 | $3.34\times10^8$ | $3.34\times10^8$ |

At $10^{10}$ operations per second, $n = 10^4$ ($\approx7\times10^{11}$ operations) takes about a minute, while $n = 10^6$ ($\approx7\times10^{17}$) would take over two years — dense elimination is out of reach for large sparse problems, motivating Chapter 7.

**Example 6.1.5 (a zero pivot forces a row interchange).** For $A = \begin{pmatrix}1&2&-1&1\\2&1&1&-1\\1&-1&2&1\\3&1&-1&2\end{pmatrix}$ and $\mathbf x = (1,-1,2,1)^T$, after step 1 rows 2 and 3 have the same entry $-3$ in column 2, so step 2 produces a **zero pivot** $a_{33}^{(3)} = 0$ while $a_{43}^{(3)} = -3\ne0$. Swapping $E_3\leftrightarrow E_4$ lets elimination finish and gives $\mathbf x = (1,-1,2,1)^T$. The matrix is nonsingular; only the *order* of the equations was bad.

---

## 6.2 Pivoting Strategies

Even a nonzero but **small** pivot is dangerous: the multiplier $m_{ji} = a_{ji}/a_{ii}$ is large, and round-off in row $i$ is magnified.

* **Partial pivoting (Algorithm 6.2):** at step $i$ choose $p\ge i$ with $|a_{pi}| = \max_{k\ge i}|a_{ki}|$. Then $|m_{ji}|\le1$.
* **Scaled partial pivoting (Algorithm 6.3):** with scale factors $s_k = \max_j|a_{kj}|$ (computed once), choose $p$ maximising $|a_{pi}|/s_p$. This makes the choice independent of how each equation is scaled.
* **Complete (maximal) pivoting:** choose the largest $|a_{kl}|$ over the whole remaining submatrix (row and column swaps). Most robust, but costs $O(n^3)$ extra comparisons; rarely used.

**Growth factor.** Gaussian elimination with partial pivoting (GEPP) is backward stable provided the entries do not grow much: $\rho_n = \max|u_{ij}|/\max|a_{ij}|\le2^{n-1}$. The worst case is attainable but essentially never occurs in practice.

### Examples for §6.2

**Example 6.2.1 (a tiny pivot, 4-digit rounding).** The system
$$
0.0004x_1 + 1.402x_2 = 1.406,\qquad 0.4003x_1 - 1.502x_2 = 2.501
$$
has solution $x_1 = 10$, $x_2 = 1$. **Without pivoting:** $m_{21} = 0.4003/0.0004 = 1001$ (1000.75 rounded), $a_{22} = -1.502 - 1001(1.402) = -1405$, $b_2 = 2.501 - 1001(1.406) = -1404$, so $x_2 = 0.9993$ and $x_1 = \frac{1.406 - 1.402(0.9993)}{0.0004} = 12.5$ — a 25% error in $x_1$. The small error in $x_2$ is multiplied by $1.402/0.0004 = 3505$. **With partial pivoting** (swap rows): $x = (10.00, 1.000)$ exactly.

**Example 6.2.2 (partial pivoting is fooled by scaling).** Multiply the first equation by $10^4$: $4x_1 + 14020x_2 = 14060$. Now $|4|>|0.4003|$, so partial pivoting does **not** swap and again gives $(12.5, 0.9993)$. Scaled partial pivoting compares $4/14020 = 0.000285$ with $0.4003/1.502 = 0.267$ and chooses row 2: $(10.00, 1.000)$.

**Example 6.2.3 (3×3, 4-digit arithmetic).**
$$
\begin{pmatrix}30&59140&1200\\5.291&-6.13&1.5\\2.2&3.1&-4.4\end{pmatrix}\mathbf x = \begin{pmatrix}121910\\-2.469\\-4.8\end{pmatrix},\qquad\mathbf x = (1,2,3)^T.
$$
Scale factors $s = (59140,\ 6.13,\ 4.4)$; ratios in column 1: $0.000507,\ 0.863,\ 0.500$.

| strategy | computed $\mathbf x$ |
|---|---|
| no pivoting | $(2.433,\ 1.985,\ 3.689)$ |
| partial pivoting (picks row 1, $\lvert30\rvert$ largest) | $(2.433,\ 1.985,\ 3.689)$ |
| scaled partial pivoting (picks row 2) | $(1.002,\ 2.002,\ 3.001)$ |

**Example 6.2.4 (complete pivoting).** For $\begin{pmatrix}1&2&3\\4&5&6\\7&8&10\end{pmatrix}$ the first pivot is $\max|a_{ij}| = 10$ at position $(3,3)$: swap rows 1↔3 and columns 1↔3 (remembering that column swaps permute the unknowns). With $\mathbf b = (1,1,1)^T$ the solution is $(-1,1,0)^T$.

**Example 6.2.5 (worst-case growth).** For Wilkinson's matrix ($1$ on the diagonal and in the last column, $-1$ below the diagonal) GEPP has growth factor exactly $2^{n-1}$: $16$ ($n = 5$), $512$ ($n=10$), $5.2\times10^5$ ($n = 20$), $5.5\times10^{11}$ ($n = 40$). For 200 random $50\times50$ matrices the growth factor had median $3.2$ and maximum $6.5$ — in practice GEPP is extremely reliable.

---

## 6.3 Linear Algebra and Matrix Inversion

Matrix product $(AB)_{ij} = \sum_ka_{ik}b_{kj}$ — associative, distributive, **not commutative**. $A$ is **nonsingular (invertible)** if there is $A^{-1}$ with $AA^{-1} = A^{-1}A = I$. Properties: $(A^{-1})^{-1} = A$, $(AB)^{-1} = B^{-1}A^{-1}$, $(A^T)^{-1} = (A^{-1})^T$.

**Theorem 6.17 (equivalences).** For $A\in\mathbb R^{n\times n}$ the following are equivalent: (i) $A\mathbf x = \mathbf 0$ has only $\mathbf x = \mathbf 0$; (ii) $A\mathbf x = \mathbf b$ has a unique solution for every $\mathbf b$; (iii) $A$ is nonsingular; (iv) $\det A\neq0$; (v) Gaussian elimination with row interchanges can be performed on $A\mathbf x = \mathbf b$ for any $\mathbf b$.

**Computing $A^{-1}$.** Solve $A\mathbf x_j = \mathbf e_j$ for $j = 1,\dots,n$ (Gauss–Jordan on $[A\,|\,I]$). Cost $\approx n^3$ multiplications — about three times the cost of one solve, and less accurate. **Golden rule: to solve $A\mathbf x = \mathbf b$, never form $A^{-1}$.**

### Examples for §6.3

**Example 6.3.1 (non-commutativity).** $A = \begin{pmatrix}1&2\\3&4\end{pmatrix}$, $B = \begin{pmatrix}0&1\\1&0\end{pmatrix}$: $AB = \begin{pmatrix}2&1\\4&3\end{pmatrix}$ (columns swapped), $BA = \begin{pmatrix}3&4\\1&2\end{pmatrix}$ (rows swapped).

**Example 6.3.2 (Gauss–Jordan inverse).** For the SPD tridiagonal $A = \begin{pmatrix}2&1&0\\1&2&1\\0&1&2\end{pmatrix}$, reducing $[A\,|\,I]$ to $[I\,|\,A^{-1}]$ gives
$$
A^{-1} = \frac14\begin{pmatrix}3&-2&1\\-2&4&-2\\1&-2&3\end{pmatrix}.
$$
Note that $A^{-1}$ is **dense** although $A$ is tridiagonal: inverting destroys sparsity.

**Example 6.3.3 (why not invert).** For a random $500\times500$ system: `solve` took 0.007 s with residual $8.7\times10^{-12}$; `inv(A) @ b` took 0.014 s with residual $5.5\times10^{-11}$. For the ill-conditioned Hilbert matrix $H_{10}$ with $\mathbf x = \mathbf 1$: `solve` error $4.4\times10^{-5}$, residual $4.7\times10^{-16}$; via the inverse: error $8.1\times10^{-3}$, residual $2.0\times10^{-4}$. Inversion is slower *and* less accurate.

**Example 6.3.4 (inverse of a product).** With $A$ as above and $B = \begin{pmatrix}2&0\\1&1\end{pmatrix}$: $(AB)^{-1} = B^{-1}A^{-1} = \begin{pmatrix}-1&0.5\\2.5&-1\end{pmatrix}$ (not $A^{-1}B^{-1}$).

**Example 6.3.5 (several right-hand sides at once).** $AX = B$ with $A = \begin{pmatrix}2&1&1\\4&-6&0\\-2&7&2\end{pmatrix}$ and $B = [\mathbf b\ \mathbf e_1\ \mathbf e_2]$ (with $\mathbf b = (5,-2,9)^T$) is solved by one elimination on $[A\,|\,B]$: $X = \begin{pmatrix}1&0.75&-0.3125\\1&0.5&-0.375\\2&-1&1\end{pmatrix}$. The last two columns are the first two columns of $A^{-1}$.

---

## 6.4 The Determinant of a Matrix

$\det A$ can be defined by cofactor expansion: $\det A = \sum_{j}(-1)^{i+j}a_{ij}M_{ij}$. Key properties (Theorem 6.16):
1. a row interchange changes the sign;
2. multiplying a row by $\lambda$ multiplies $\det$ by $\lambda$;
3. adding a multiple of one row to another leaves $\det$ unchanged;
4. $\det(AB) = \det A\det B$, $\det A^T = \det A$, $\det A^{-1} = 1/\det A$;
5. $\det$ of a triangular matrix is the product of its diagonal.

Hence, from elimination: $\det A = (-1)^{\#\text{swaps}}\prod_iu_{ii}$ — $O(n^3)$ work, versus $O(n!)$ for cofactor expansion.

### Examples for §6.4

**Example 6.4.1.** For $A$ of Example 6.1.1, no row swaps were needed and the pivots are $2, -8, 1$: $\det A = 2(-8)(1) = -16$.

**Example 6.4.2 (4×4).** For the matrix of Example 6.1.2 the pivots are $4, 5.5, 6.0909, 8.2090$: $\det = 4\times5.5\times6.0909\times8.2090 = 1100$.

**Example 6.4.3 (properties).** Swapping rows 1 and 2 of $A$ gives $\det = +16$; multiplying row 1 by 3 gives $-48$.

**Example 6.4.4 (cost).** Cofactor expansion needs ~$n!$ operations: $n = 10$: $3.6\times10^6$; $n = 20$: $2.4\times10^{18}$ (≈ 77 years at $10^9$/s); $n=25$: $1.6\times10^{25}$. Elimination: $n^3/3 = 2667$ for $n = 20$.

**Example 6.4.5 (log-determinants in statistics).** A $200\times200$ sample covariance matrix has $\det = 1.9\times10^{-87}$ — close to underflow; for $n = 1000$ it would underflow to 0. The Gaussian log-likelihood needs $\log\det\Sigma$, computed as $\sum\log|u_{ii}|$ (`np.linalg.slogdet`) or, for SPD matrices, $2\sum\log\ell_{ii}$ from Cholesky: both give $-199.686$.

---

## 6.5 Matrix Factorizations

Gaussian elimination without row swaps is equivalent to $A = LU$: $U$ is the final upper-triangular matrix and $L$ is unit lower-triangular holding the multipliers $m_{ji}$ below the diagonal.

**Theorem 6.19.** If Gaussian elimination can be performed on $A\mathbf x = \mathbf b$ without row interchanges (equivalently, all leading principal submatrices of $A$ are nonsingular), then $A = LU$ with $L$ unit lower triangular and $U$ upper triangular.

**With pivoting:** $PA = LU$, $P$ a permutation matrix. Every nonsingular $A$ has such a factorisation. Then $A\mathbf x = \mathbf b\iff L\mathbf y = P\mathbf b$, $U\mathbf x = \mathbf y$.

**Cost.** Factorisation $\approx\frac{n^3}{3}$ multiplications; each subsequent solve $\approx n^2$. *Factor once, solve many times.*

**Doolittle / Crout / Cholesky** (Algorithm 6.4) are the choices $\ell_{ii} = 1$, $u_{ii} = 1$, $\ell_{ii} = u_{ii}$ respectively.

### Examples for §6.5

**Example 6.5.1 ($LU$ and two triangular solves).** From Example 6.1.1:
$$
L = \begin{pmatrix}1&0&0\\2&1&0\\-1&-1&1\end{pmatrix},\qquad U = \begin{pmatrix}2&1&1\\0&-8&-2\\0&0&1\end{pmatrix}.
$$
$L\mathbf y = (5,-2,9)^T$: $y = (5,-12,2)$; $U\mathbf x = \mathbf y$: $\mathbf x = (1,1,2)$.

**Example 6.5.2 ($PA = LU$ with partial pivoting).** For the same $A$, row 2 has the largest first entry: $P = \begin{pmatrix}0&1&0\\1&0&0\\0&0&1\end{pmatrix}$,
$$
L = \begin{pmatrix}1&0&0\\0.5&1&0\\-0.5&1&1\end{pmatrix},\qquad U = \begin{pmatrix}4&-6&0\\0&4&1\\0&0&1\end{pmatrix}.
$$
All multipliers satisfy $|\ell_{ij}|\le1$.

**Example 6.5.3 (4×4 and reuse).** For the matrix of Example 6.1.2:
$$
L = \begin{pmatrix}1&&&\\0.5&1&&\\0.25&0.2273&1&\\-0.25&0.3182&0.2836&1\end{pmatrix},\quad U = \begin{pmatrix}4&-1&1&2\\&5.5&-1.5&0\\&&6.0909&-2.5\\&&&8.2090\end{pmatrix}.
$$
With these factors, $\mathbf b = (9,-4,10,6)^T$ gives $(1,-1,2,1)$ and $\mathbf e_1$ gives the first column of $A^{-1}$: $(0.2, -0.0909, 0, 0.0545)$, each for $O(n^2)$ work.

**Example 6.5.4 (no $LU$ without pivoting).** $A = \begin{pmatrix}0&1\\1&1\end{pmatrix}$ is nonsingular but $a_{11} = 0$: no $A = LU$ exists. With $P$ = swap: $PA = \begin{pmatrix}1&1\\0&1\end{pmatrix} = I\cdot U$.

**Example 6.5.5 (factor once).** For $n = 1000$ and 50 right-hand sides (e.g. 50 Newton iterations with a frozen Jacobian, or 50 time steps of an implicit ODE method): one factorisation plus 50 solves ≈ $7.7\times10^8$ flops; 50 separate eliminations ≈ $3.3\times10^{10}$ — 43× more.

---

## 6.6 Special Types of Matrices

**Strictly diagonally dominant (SDD):** $|a_{ii}|>\sum_{j\ne i}|a_{ij}|$ for every row.
**Theorem 6.21.** An SDD matrix is nonsingular, and Gaussian elimination can be performed **without row interchanges**, with stable growth.

**Positive definite (PD):** $A = A^T$ and $\mathbf x^TA\mathbf x>0$ for all $\mathbf x\neq\mathbf 0$. (B&F define PD for symmetric matrices.)
**Theorem 6.23.** If $A$ is PD then $A$ is nonsingular, $a_{ii}>0$, $\max|a_{kj}|\le\max|a_{ii}|$, and $a_{ij}^2<a_{ii}a_{jj}$.
**Theorem 6.25.** A symmetric $A$ is PD iff every leading principal submatrix has positive determinant.
**Theorem 6.26 ($LDL^T$).** Symmetric $A$ is PD iff $A = LDL^T$ with $L$ unit lower triangular and $D$ diagonal with positive entries.
**Corollary 6.27 (Cholesky).** Symmetric $A$ is PD iff $A = LL^T$ with $L$ lower triangular with positive diagonal:
$$
\ell_{jj} = \sqrt{a_{jj} - \sum_{k<j}\ell_{jk}^2},\qquad \ell_{ij} = \frac{a_{ij}-\sum_{k<j}\ell_{ik}\ell_{jk}}{\ell_{jj}}\ (i>j).
$$
Cost $\frac{n^3}{6}$ multiplications (half of LU); no pivoting needed; a negative radicand **proves** $A$ is not PD.

**Band matrices.** $a_{ij} = 0$ for $|i-j|>p$. Tridiagonal ($p = 1$) systems are solved by **Crout factorisation** (Algorithm 6.7), a.k.a. the Thomas algorithm, with $5n-4$ multiplications and $3n-3$ additions:
$$
\ell_{11} = a_{11},\ u_{12} = \frac{a_{12}}{\ell_{11}},\quad \ell_{ii} = a_{ii} - a_{i,i-1}u_{i-1,i},\ u_{i,i+1} = \frac{a_{i,i+1}}{\ell_{ii}},
$$
then $z_i = \frac{b_i - a_{i,i-1}z_{i-1}}{\ell_{ii}}$ and $x_n = z_n$, $x_i = z_i - u_{i,i+1}x_{i+1}$.

### Examples for §6.6

**Example 6.6.1 (diagonal dominance).** $A = \begin{pmatrix}7&2&0\\3&5&-1\\0&5&-6\end{pmatrix}$ is SDD ($7>2$, $5>4$, $6>5$). $B = \begin{pmatrix}6&4&-3\\4&-2&0\\-3&0&1\end{pmatrix}$ is symmetric but not SDD ($|-2|<4$), and neither is $B^T$. (SDD is not required to be symmetric; PD is.)

**Example 6.6.2 (Cholesky).** $A = \begin{pmatrix}4&12&-16\\12&37&-43\\-16&-43&98\end{pmatrix}$. Leading minors $4, 4, 36>0$ ⇒ PD.
$\ell_{11} = 2$; $\ell_{21} = 12/2 = 6$; $\ell_{31} = -16/2 = -8$; $\ell_{22} = \sqrt{37-36} = 1$; $\ell_{32} = (-43 - (-8)(6))/1 = 5$; $\ell_{33} = \sqrt{98 - 64 - 25} = 3$:
$$
L = \begin{pmatrix}2&0&0\\6&1&0\\-8&5&3\end{pmatrix}.
$$
Solving $A\mathbf x = (1,2,3)^T$: $L\mathbf y = \mathbf b\Rightarrow\mathbf y = (0.5,-1,4)$; $L^T\mathbf x = \mathbf y\Rightarrow\mathbf x = (28.583, -7.667, 1.333)$.

**Example 6.6.3 ($LDL^T$).** Same $A$: $L = \begin{pmatrix}1&0&0\\3&1&0\\-4&5&1\end{pmatrix}$, $D = \operatorname{diag}(4,1,9)$ — no square roots needed; and $L_{\text{Chol}} = L\,D^{1/2}$.

**Example 6.6.4 (Crout for a tridiagonal system).** $\begin{pmatrix}2&-1&&\\-1&2&-1&\\&-1&2&-1\\&&-1&2\end{pmatrix}\mathbf x = \begin{pmatrix}1\\0\\0\\1\end{pmatrix}$:
$\ell = (2,\ 1.5,\ 1.3333,\ 1.25)$, $u = (-0.5,\ -0.6667,\ -0.75)$, $z = (0.5,\ 0.3333,\ 0.25,\ 1)$, and back substitution gives $\mathbf x = (1,1,1,1)^T$.

**Example 6.6.5 (Cholesky as a PD test).** $\begin{pmatrix}1&2\\2&1\end{pmatrix}$: $\ell_{11} = 1$, $\ell_{21} = 2$, $\ell_{22}^2 = 1 - 4 = -3<0$ — **not PD** (eigenvalues $-1, 3$). In ML this detects an invalid covariance/kernel matrix (e.g. due to round-off; the fix is adding "jitter" $\varepsilon I$).

**Example 6.6.6 (band structure pays).** Dense LU vs tridiagonal solve: $n = 10^3$: $6.7\times10^8$ vs $8\times10^3$ flops; $n = 10^6$: $6.7\times10^{17}$ vs $8\times10^6$. Cubic splines (Chapter 3), implicit heat-equation steps (Chapter 12) and 1-D smoothing all lead to tridiagonal systems.

---

## 6.7 Direct Methods in Data Science *(data-science extension)*

| Task | Factorisation | What it gives |
|---|---|---|
| sample $\mathbf x\sim N(\boldsymbol\mu,\Sigma)$ | $\Sigma = LL^T$ | $\mathbf x = \boldsymbol\mu + L\mathbf z$, $\mathbf z\sim N(0,I)$ |
| Gaussian log-density | Cholesky | $\log\det\Sigma = 2\sum\log\ell_{ii}$, $\|L^{-1}(\mathbf x-\boldsymbol\mu)\|^2$ |
| Gaussian-process regression | $K+\sigma^2I = LL^T$ | posterior mean $\mathbf k_*^T\boldsymbol\alpha$, variance $k_{**} - \|L^{-1}\mathbf k_*\|^2$ |
| absorbing Markov chains | $LU$ of $I-Q$ | absorption probabilities, expected times |
| least squares (small $p$) | Cholesky of $X^TX$ | $\hat{\boldsymbol\beta}$ (but see Chapter 13 for conditioning) |

### Examples for §6.7

**Example 6.7.1 (sampling a correlated Gaussian).** $\Sigma = \begin{pmatrix}4&2.4\\2.4&9\end{pmatrix}$ (correlation $0.4$), $\boldsymbol\mu = (1,-2)$. $L = \begin{pmatrix}2&0\\1.2&2.7495\end{pmatrix}$. With $10^5$ draws $\mathbf x = \boldsymbol\mu + L\mathbf z$: sample mean $(1.000, -2.014)$, sample covariance $\begin{pmatrix}3.99&2.39\\2.39&8.98\end{pmatrix}$ ✓.

**Example 6.7.2 (Gaussian log-likelihood without inverses).** For $\mathbf x = (3,1)$: solve $L\mathbf z = \mathbf x - \boldsymbol\mu = (2,3)$ ⇒ Mahalanobis distance $\|\mathbf z\|^2 = 1.428571$; $\log\det\Sigma = 2(\ln2 + \ln2.7495) = 3.409166$;
$\log p(\mathbf x) = -\frac12(1.428571) - \frac12(3.409166) - \ln(2\pi) = -4.256746$ — identical to `scipy.stats.multivariate_normal.logpdf`.

**Example 6.7.3 (Gaussian-process regression).** Training inputs $0,1,2,3$, targets $\sin x$, squared-exponential kernel ($\ell = 1$), noise $\sigma = 0.1$. Cholesky of $K+0.01I$ gives $\boldsymbol\alpha = (K+\sigma^2I)^{-1}\mathbf y = (-0.5556, 0.7613, 0.7683, -0.4176)$. At $x_* = 1.5$: posterior mean $\mathbf k_*^T\boldsymbol\alpha = 1.0340$ (true $\sin1.5 = 0.9975$), posterior s.d. $0.132$.

**Example 6.7.4 (absorbing Markov chain — gambler's ruin).** States $1,2,3$ transient, $0$ and $4$ absorbing, fair coin. $Q = \begin{pmatrix}0&0.5&0\\0.5&0&0.5\\0&0.5&0\end{pmatrix}$, $R = \begin{pmatrix}0.5&0\\0&0\\0&0.5\end{pmatrix}$. Fundamental matrix $N = (I-Q)^{-1} = \begin{pmatrix}1.5&1&0.5\\1&2&1\\0.5&1&1.5\end{pmatrix}$; absorption probabilities $NR = \begin{pmatrix}0.75&0.25\\0.5&0.5\\0.25&0.75\end{pmatrix}$; expected steps to absorption $N\mathbf 1 = (3,4,3)$. (Compute $N\mathbf 1$ by *solving* $(I-Q)\mathbf t = \mathbf 1$.)

**Example 6.7.5 (normal equations by Cholesky).** Data $x = 1,\dots,5$, $y = 2.2, 2.8, 3.6, 4.5, 5.1$, model $y = \beta_0+\beta_1x$: $X^TX = \begin{pmatrix}5&15\\15&55\end{pmatrix}$, $X^T\mathbf y = (18.2, 62.1)$, $L = \begin{pmatrix}2.2361&0\\6.7082&3.1623\end{pmatrix}$, $\hat{\boldsymbol\beta} = (1.39, 0.75)$. (Fine here; for ill-conditioned designs use QR — Chapter 13.)

---

## Chapter summary

| Matrix type | Method | Cost (mult.) | Pivoting |
|---|---|---|---|
| general | $PA = LU$ (GEPP) | $n^3/3$ | partial (scaled if rows badly scaled) |
| SDD | $LU$ | $n^3/3$ | not needed |
| symmetric PD | Cholesky $LL^T$ / $LDL^T$ | $n^3/6$ | not needed |
| tridiagonal | Crout / Thomas | $5n$ | not needed if SDD |
| many right-hand sides | factor once | $+n^2$ per solve | — |

## Further reading

Burden & Faires Ch. 6 · Golub & Van Loan, *Matrix Computations*, Ch. 3–4 · Trefethen & Bau, *Numerical Linear Algebra*, Lectures 20–23 · Rasmussen & Williams, *Gaussian Processes for Machine Learning*, Algorithm 2.1.
