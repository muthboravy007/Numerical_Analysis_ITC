"""Chapter 1 — Floating-point arithmetic and error analysis.

Library equivalents: ``np.finfo``, ``math.fsum``, ``np.spacing``.
"""

import math

import numpy as np


def machine_epsilon(dtype=float):
    """Smallest eps of the form 2**-k with fl(1 + eps) > 1.

    >>> machine_epsilon() == np.finfo(float).eps
    True
    """
    one = dtype(1)
    eps = dtype(1)
    while one + eps / dtype(2) > one:
        eps = eps / dtype(2)
    return eps


def absolute_error(approx, exact):
    return abs(approx - exact)


def relative_error(approx, exact):
    if exact == 0:
        raise ValueError("relative error undefined for exact == 0")
    return abs(approx - exact) / abs(exact)


def significant_digits(approx, exact):
    """Approximate number of correct significant decimal digits."""
    rel = relative_error(approx, exact)
    if rel == 0:
        return float("inf")
    return -math.log10(2 * rel)


def naive_sum(xs):
    s = 0.0
    for x in xs:
        s += x
    return s


def kahan_sum(xs):
    """Compensated (Kahan) summation: error O(eps) instead of O(n eps)."""
    s = 0.0
    c = 0.0  # running compensation for lost low-order bits
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def pairwise_sum(xs):
    """Recursive pairwise summation (what NumPy's ``np.sum`` uses)."""
    xs = list(xs)
    n = len(xs)
    if n <= 8:
        return naive_sum(xs)
    mid = n // 2
    return pairwise_sum(xs[:mid]) + pairwise_sum(xs[mid:])


def quadratic_roots_naive(a, b, c):
    d = math.sqrt(b * b - 4 * a * c)
    return (-b + d) / (2 * a), (-b - d) / (2 * a)


def quadratic_roots_stable(a, b, c):
    """Avoid cancellation: compute the large-magnitude root first,
    then use Vieta's formula x1 * x2 = c / a for the other."""
    d = math.sqrt(b * b - 4 * a * c)
    q = -0.5 * (b + math.copysign(d, b))
    x1 = q / a
    x2 = c / q if q != 0 else 0.0
    return x1, x2


def logsumexp(x):
    """Numerically stable log(sum(exp(x))) — used everywhere in ML."""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(x - m)))


def softmax(x):
    x = np.asarray(x, dtype=float)
    z = np.exp(x - np.max(x))
    return z / z.sum()


def variance_two_pass(xs):
    xs = np.asarray(xs, dtype=float)
    mu = xs.mean()
    return np.sum((xs - mu) ** 2) / (len(xs) - 1)


def variance_textbook(xs):
    """The unstable one-pass formula (sum x^2 - n mean^2)/(n-1)."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    return (np.sum(xs ** 2) - n * xs.mean() ** 2) / (n - 1)


def variance_welford(xs):
    """Welford's stable streaming algorithm."""
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in xs:
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    return m2 / (n - 1)


def fl(x, digits, mode="round"):
    """k-digit decimal machine number of x (B&F finite-digit arithmetic).
    mode: 'round' or 'chop'."""
    if x == 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x))) + 1  # x = 0.d1d2... * 10^e
    m = x / 10 ** e
    scale = 10 ** digits
    if mode == "chop":
        m = math.trunc(m * scale) / scale
    else:
        m = math.floor(abs(m) * scale + 0.5) / scale * (1 if m > 0 else -1)
    return float(f"{m * 10 ** e:.{digits}g}") if m else 0.0


def ieee_double_decode(bits):
    """Decode a 64-character string of 0/1 as an IEEE-754 double."""
    bits = bits.replace(" ", "")
    s = int(bits[0])
    c = int(bits[1:12], 2)
    f = int(bits[12:], 2) / 2 ** 52
    return (-1) ** s * 2.0 ** (c - 1023) * (1 + f)
