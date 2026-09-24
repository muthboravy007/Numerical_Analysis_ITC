# Chapter 1 — Exercises: Mathematical Preliminaries and Error Analysis

> Lecture: [Chapter 1](../lectures/ch01-mathematical-preliminaries.md) · Solutions: [`solutions/ch01_solutions.md`](../solutions/ch01_solutions.md) · Solution code: [`solutions/code/ch01_solutions.py`](../solutions/code/ch01_solutions.py)
>
> **Levels:** ★ routine · ★★ standard · ★★★ challenging. Programming exercises should be submitted as a Python script or notebook using NumPy; do not use a library routine that solves the whole exercise unless the exercise says so.

## A. Theory and proofs

**A1 ★★** Let $y = 0.d_1d_2\ldots\times10^n$ with $d_1\ne0$. Prove that $k$-digit chopping satisfies $\frac{|y - fl(y)|}{|y|}\le10^{-k+1}$ and $k$-digit rounding satisfies $\frac{|y-fl(y)|}{|y|}\le\frac12\times10^{-k+1}$.

**A2 ★** Use the Mean Value Theorem to prove $|\sin a - \sin b|\le|a-b|$ for all real $a,b$. Deduce that the fixed-point iteration $x_{n+1} = \frac12\sin x_n$ contracts distances by at least a factor $\frac12$.

**A3 ★★** Compute the relative condition number $\kappa_f(x) = |xf'(x)/f(x)|$ for (a) $f(x) = x^n$; (b) $f(x) = 1/x$; (c) $f(x) = \sqrt{1+x}-1$; (d) $f(x) = e^x$; (e) $f(x) = \cos x$. For which $x$ is each problem ill-conditioned? Explain why (c) is well-conditioned near $x = 0$ although its naive evaluation is inaccurate there.

**A4 ★★★** Let $\hat s_k = fl(\hat s_{k-1} + x_k)$ with $\hat s_1 = x_1$ (naive left-to-right summation) and assume $fl(a+b) = (a+b)(1+\delta)$, $|\delta|\le u$. Show that, to first order in $u$,
$$
|\hat s_n - s_n|\le(n-1)\,u\sum_{i=1}^n|x_i|.
$$
Why does sorting the numbers by increasing magnitude usually help?

**A5 ★★** Let $y_n = \int_0^1\frac{x^n}{x+5}dx$. (a) Show $y_n + 5y_{n-1} = \frac1n$ and $y_0 = \ln\frac65$. (b) Show that the forward recurrence $y_n = \frac1n - 5y_{n-1}$ multiplies any error in $y_{n-1}$ by $-5$, and predict how large an initial error of $10^{-17}$ becomes at $n = 25$. (c) Propose a stable algorithm and justify it.

## B. Hand computation

**B1 ★** Find the third Maclaurin polynomial $P_3$ of $f(x) = e^{-x}$, use it to approximate $e^{-0.5}$, and bound the error with the Taylor remainder. Compare with the actual error.

**B2 ★** How many terms of $\frac\pi4 = 1 - \frac13 + \frac15 - \cdots$ guarantee an error below $10^{-3}$ (use the alternating-series bound)? What does this say about using this series to compute $\pi$?

**B3 ★★** Let $x = 0.54617$, $y = 0.54601$, $z = 0.001234$. Compute $(x - y)/z$ using four-digit rounding arithmetic and find the relative error. Which operation caused the loss of accuracy?

**B4 ★★** Rewrite each expression to avoid loss of significance for the indicated $x$, and evaluate both forms in double precision:
(a) $\frac{1-\cos x}{\sin x}$, $x = 10^{-8}$; (b) $\ln\bigl(x - \sqrt{x^2-1}\bigr)$, $x = 10^8$; (c) $e^x - e^{-x}$, $x = 10^{-10}$; (d) $x - \sin x$, $x = 10^{-5}$.

**B5 ★** Evaluate $P(t) = 1.01t^3 - 4.62t^2 - 3.11t + 12.2$ at $t = 1.72$ using three-digit rounding (i) directly and (ii) in nested form. Compare with the exact value and count the multiplications in each method.

**B6 ★** Using your formulas from A3, evaluate $\kappa$ for: $x^5$; $1/x$; $\sqrt{1+x}-1$ at $x = 10^{-3}$; $e^x$ at $x = 50$; $\cos x$ at $x = \frac\pi2 - 10^{-3}$. How many significant digits might be lost in each case?

**B7 ★★** (a) Write $0.625$ exactly in binary. (b) Give the first 12 bits after the binary point of $0.1$ and explain why $0.1$ is not a machine number. (c) Which decimal number does the IEEE-754 double `1 10000000001 1010 000…0` represent?

## C. Programming

**C1 ★** Write a function computing machine epsilon by repeated halving for `np.float16`, `np.float32` and `np.float64`, and compare with `np.finfo(dtype).eps`.

**C2 ★★** Compute $\sum_{k=1}^n1/k$ in `float32` for $n = 10^6$ and $10^7$ by (i) a naive running sum, (ii) `np.sum` (pairwise), (iii) Kahan summation. Compare with `math.fsum` in double precision. Explain the size of each error.

**C3 ★★** For $f(x) = \sin x$ at $x = 1$ compute the forward-difference error for $h = 10^{-1},\dots,10^{-15}$. Plot on log–log axes, identify the optimal $h$ and explain the "V" shape quantitatively.

**C4 ★** Implement numerically stable `logsumexp` and `softmax`. Test them on $(-1000,-1001)$ and $(10^4, 10^4+1)$, where the naive formulas fail.

**C5 ★★** Generate 1000 values $10^6 + N(0,1)$. Compute the sample variance with the textbook one-pass formula, the two-pass formula and Welford's algorithm, in `float32` and `float64`. Explain every discrepancy.

## D. Data-science applications

**D1 ★★** Simulate $n = 10^5$ Bernoulli($0.7$) observations. Compute the likelihood $\prod p(y_i)$ and the log-likelihood $\sum\log p(y_i)$ at $p = 0.7$. Explain the result and relate the average log-likelihood to $0.7\ln0.7 + 0.3\ln0.3$.

**D2 ★★★** Mixed-precision training stores gradients in `float16`. (a) What is the smallest positive (subnormal) `float16` number? (b) Show that a gradient of $10^{-8}$ becomes $0$ in `float16`. (c) Explain **loss scaling**: multiply the loss by $S = 1024$ before back-propagation, cast to `float16`, then divide by $S$ in `float32`. Demonstrate numerically that the small gradient survives.
