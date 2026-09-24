"""P5 starter: periodogram + harmonic regression forecast of Mauna Loa CO2.

Run: python p5_fourier_forecasting.py
"""
import numpy as np
import statsmodels.api as sm


def periodogram(x, dt=1.0, window=True):
    x = np.asarray(x, float) - np.mean(x)
    w = np.hanning(len(x)) if window else np.ones(len(x))
    F = np.fft.rfft(x * w)  # TODO: replace with your own FFT
    return np.fft.rfftfreq(len(x), d=dt), np.abs(F) ** 2 / np.sum(w ** 2)


def design(t, K, trend_degree=2):
    cols = [t ** d for d in range(trend_degree + 1)]
    for k in range(1, K + 1):
        cols += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
    return np.column_stack(cols)


if __name__ == "__main__":
    co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
    y = co2.values
    t = (np.arange(len(y)) / 12.0)
    tc = t - t.mean()
    detr = y - np.polyval(np.polyfit(tc, y, 2), tc)
    f, P = periodogram(detr, dt=1 / 12)
    top = np.argsort(-P)[:3]
    print("dominant frequencies (cycles/year):", f[top].round(3))
    h = 60
    for K in (1, 2, 3, 4):
        X = design(tc, K)
        beta = np.linalg.lstsq(X[:-h], y[:-h], rcond=None)[0]
        pred = X[-h:] @ beta
        print(f"K={K}: 5-year holdout RMSE {np.sqrt(np.mean((pred - y[-h:]) ** 2)):.3f} ppm")
    naive = np.concatenate([y[-h - 12:-h]] * (h // 12))  # seasonal naive: repeat the last observed year
    print(f"seasonal naive RMSE {np.sqrt(np.mean((naive - y[-h:]) ** 2)):.3f} ppm")
    # TODO: own FFT, spline trend (B-splines), Whittaker smoother + GCV, rolling-origin CV, data set B
