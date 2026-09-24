"""P8 starter: Black-Scholes closed form, Crank-Nicolson PDE, Monte Carlo, implied volatility.

Run: python p8_option_pricing.py
"""
import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm


def bs_price(S, K, r, sigma, T, kind="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def cn_european_put(K, r, sigma, T, Smax=300.0, M=300, N=150):
    S = np.linspace(0, Smax, M + 1); dt = T / N; i = np.arange(1, M)
    a = 0.25 * dt * (sigma ** 2 * i ** 2 - r * i)
    b = -0.5 * dt * (sigma ** 2 * i ** 2 + r)
    c = 0.25 * dt * (sigma ** 2 * i ** 2 + r * i)
    V = np.maximum(K - S, 0.0)
    ab = np.zeros((3, M - 1)); ab[0, 1:] = -c[:-1]; ab[1] = 1 - b; ab[2, :-1] = -a[1:]
    for n in range(N):
        tau = (n + 1) * dt
        rhs = a * V[:-2] + (1 + b) * V[1:-1] + c * V[2:]
        rhs[0] += a[0] * K * np.exp(-r * tau)  # new boundary value (the old one is inside a*V[:-2])
        V = np.concatenate([[K * np.exp(-r * tau)], solve_banded((1, 1), ab, rhs), [0.0]])
    return S, V


def mc_european(S0, K, r, sigma, T, n, kind="put", seed=0, antithetic=False):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    if antithetic:
        Z = np.concatenate([Z[: n // 2], -Z[: n // 2]])
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    pay = np.maximum(K - ST, 0) if kind == "put" else np.maximum(ST - K, 0)
    disc = np.exp(-r * T) * pay
    if antithetic:
        pairs = 0.5 * (disc[: n // 2] + disc[n // 2:])
        return pairs.mean(), pairs.std(ddof=1) / np.sqrt(len(pairs))
    return disc.mean(), disc.std(ddof=1) / np.sqrt(n)


def implied_vol(price, S, K, r, T, kind="call", tol=1e-12):
    """Newton on sigma with a bisection safeguard on [1e-4, 5]."""
    lo, hi, s = 1e-4, 5.0, 0.2
    for _ in range(100):
        f = bs_price(S, K, r, s, T, kind) - price
        if abs(f) < tol:
            return s
        lo, hi = (s, hi) if f < 0 else (lo, s)
        v = bs_vega(S, K, r, s, T)
        s_new = s - f / v if v > 1e-12 else 0.5 * (lo + hi)
        s = s_new if lo < s_new < hi else 0.5 * (lo + hi)
    return s


def psor_american_put(*args, **kwargs):
    """TODO: projected SOR for the American put linear complementarity problem."""
    raise NotImplementedError


if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.25, 1.0
    exact = bs_price(S0, K, r, sigma, T, "put")
    print(f"closed-form put {exact:.6f}")
    for M, N in ((100, 50), (200, 100), (400, 200)):
        S, V = cn_european_put(K, r, sigma, T, M=M, N=N)
        print(f"CN M={M} N={N}: {np.interp(S0, S, V):.6f} (error {abs(np.interp(S0, S, V) - exact):.2e})")
    for n in (10 ** 4, 10 ** 5, 10 ** 6):
        p, se = mc_european(S0, K, r, sigma, T, n)
        pa, sea = mc_european(S0, K, r, sigma, T, n, antithetic=True)
        print(f"MC n={n:>7d}: {p:.4f} +- {se:.4f}   antithetic {pa:.4f} +- {sea:.4f}")
    call = bs_price(S0, 120.0, r, 0.3, T)
    print(f"implied vol recovered from a 0.30-vol call price: {implied_vol(call, S0, 120.0, r, T):.10f}")
    # TODO: Rannacher start, American put (PSOR, penalty, CRR), Asian MC with control variate, smile calibration
