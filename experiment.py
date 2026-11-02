"""Educational electricity-price conformal forecasting and spread backtest."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["hour_sin", "hour_cos", "weekday", "load_forecast",
            "wind_forecast", "da_lag_48", "rt_lag_48"]


def make_demo(days=180, seed=42):
    """Generate artificial prices, forecast covariates, spikes, and regime drift."""
    if days < 30:
        raise ValueError("Use at least 30 days for chronological evaluation.")
    rng = np.random.default_rng(seed)
    n = days * 24
    time = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    hour = time.hour.to_numpy()
    load = 50 + 12 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 3, n)
    wind = np.clip(18 + 8 * np.sin(np.arange(n) / 100) + rng.normal(0, 3, n), 0, None)
    scale = np.where(np.arange(n) < n * .8, 1., 1.8)
    da = 30 + 1.4 * load - wind + rng.normal(0, 5, n) * scale
    spikes = (rng.random(n) < .025) * rng.exponential(60, n)
    rt = da + .65 * (load - 50) - .4 * (wind - 18) + rng.normal(0, 12, n) * scale + spikes
    return pd.DataFrame({"timestamp": time, "load_forecast": load,
                         "wind_forecast": wind, "price_da": da, "price_rt": rt})


def prepare(raw):
    """Use conservative 48-hour lags; never fill missing prices from the future."""
    required = ["timestamp", "load_forecast", "wind_forecast", "price_da", "price_rt"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = raw[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].isna().any() or df["timestamp"].duplicated().any():
        raise ValueError("Timestamps must be nonmissing and unique.")
    if not df["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Input must be a regular hourly UTC series; resolve gaps explicitly.")
    values = df[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Numeric input must contain only finite values.")
    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["da_lag_48"] = df["price_da"].shift(48)
    df["rt_lag_48"] = df["price_rt"].shift(48)
    df["spread"] = df["price_rt"] - df["price_da"]
    return df.iloc[48:].reset_index(drop=True)


def conformal_radius(scores, alpha):
    """Return the ceil((n+1)(1-alpha))-th absolute-residual order statistic."""
    scores = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or scores.ndim != 1 or not len(scores):
        raise ValueError("Require 0 < alpha < 1 and nonempty one-dimensional scores.")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Scores must be finite and nonnegative.")
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.partition(scores, k - 1)[k - 1]) if k <= len(scores) else np.inf


def daily_intervals(pred, actual, timestamps, calibration_scores, alpha, window):
    """Issue all intervals for a day before adding that day's realized residuals.

    This assumes yesterday's settlement prices are available at today's issue time.
    Adapt the update delay to the actual market publication schedule.
    """
    return daily_calibrated_bounds(pred, pred, actual, timestamps,
                                   calibration_scores, alpha, window)


def nonconformity(lower, upper, actual):
    """Use expansion-only CQR scores; an already covered label scores zero.

    Clipping at zero is conservative relative to signed CQR and prevents
    calibrated intervals from shrinking into empty sets.
    """
    return np.maximum(np.maximum(lower - actual, actual - upper), 0.)


def daily_calibrated_bounds(base_lower, base_upper, actual, timestamps,
                            calibration_scores, alpha, window):
    """Calibrate arbitrary base intervals using only completed prior days."""
    if window < 1:
        raise ValueError("Window must be positive.")
    history = deque(np.asarray(calibration_scores)[-window:], maxlen=window)
    lower, upper = np.empty(len(actual)), np.empty(len(actual))
    days = pd.DatetimeIndex(timestamps).floor("D")
    for day in days.unique():
        mask = days == day
        q = conformal_radius(list(history), alpha)
        lower[mask], upper[mask] = base_lower[mask] - q, base_upper[mask] + q
        history.extend(nonconformity(base_lower[mask], base_upper[mask], actual[mask]))
    return lower, upper


def quantile_bounds(train, frame, target, alpha):
    """Fit lower/upper quantile models on training data only, then sort crossings."""
    endpoints = []
    for quantile in (alpha / 2, 1 - alpha / 2):
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, max_iter=100,
            max_leaf_nodes=15, random_state=42)
        model.fit(train[FEATURES], train[target])
        endpoints.append(model.predict(frame[FEATURES]))
    return np.minimum(*endpoints), np.maximum(*endpoints)


def interval_metrics(y, pred, lower, upper, alpha):
    covered = (y >= lower) & (y <= upper)
    score = upper - lower + 2 / alpha * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    return {"mae": float(np.mean(np.abs(y - pred))),
            "coverage": float(covered.mean()), "mean_width": float(np.mean(upper - lower)),
            "interval_score": float(np.mean(score)), "n": len(y)}


def trading_signals(lower, upper, cost):
    """Size virtual spread positions in MWh: +1 long RT-minus-DA, -1 short."""
    if cost < 0:
        raise ValueError("Trading cost must be nonnegative.")
    return np.where(lower > cost, 1, np.where(upper < -cost, -1, 0))


def run(raw, output, alpha=.1, window=24 * 14, cost=2.):
    if not 0 < alpha < 1 or window < 1 or not np.isfinite(cost) or cost < 0:
        raise ValueError("Require 0 < alpha < 1, a positive window and finite nonnegative cost.")
    df = prepare(raw)
    # Split on whole days so a test day never overlaps calibration.
    days = df["timestamp"].dt.floor("D").drop_duplicates().to_list()
    if len(days) < 25:
        raise ValueError("Need at least 25 complete/partial days after lag construction.")
    a, b = days[int(len(days) * .6)], days[int(len(days) * .8)]
    train = df[df.timestamp < a]
    cal = df[(df.timestamp >= a) & (df.timestamp < b)]
    test = df[df.timestamp >= b].copy()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, predictions, spread_bounds = [], test[["timestamp", "price_da", "price_rt", "spread"]].copy(), {}
    for target in ["price_da", "price_rt", "spread"]:
        model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=42)
        model.fit(train[FEATURES], train[target])
        residuals = np.abs(cal[target].to_numpy() - model.predict(cal[FEATURES]))
        pred, y = model.predict(test[FEATURES]), test[target].to_numpy()
        q = conformal_radius(residuals, alpha)
        bounds = {"fixed": (pred - q, pred + q),
                  "rolling": daily_intervals(pred, y, test.timestamp, residuals, alpha, window)}
        # Predict calibration and test endpoints with the same train-only models.
        quantile_lower, quantile_upper = quantile_bounds(
            train, pd.concat([cal, test]), target, alpha)
        cal_lower, lower = quantile_lower[:len(cal)], quantile_lower[len(cal):]
        cal_upper, upper = quantile_upper[:len(cal)], quantile_upper[len(cal):]
        scores = nonconformity(cal_lower, cal_upper, cal[target].to_numpy())
        correction = conformal_radius(scores, alpha)
        bounds.update({
            "quantile": (lower, upper),
            "cqr_fixed": (lower - correction, upper + correction),
            "cqr_rolling": daily_calibrated_bounds(
                lower, upper, y, test.timestamp, scores, alpha, window),
        })
        predictions[f"{target}_prediction"] = pred
        for method, (lower, upper) in bounds.items():
            rows.append({"target": target, "method": method,
                         **interval_metrics(y, pred, lower, upper, alpha)})
            predictions[f"{target}_{method}_lower"] = lower
            predictions[f"{target}_{method}_upper"] = upper
            if target == "spread":
                spread_bounds[method] = (lower, upper)
        if target == "spread":
            spread_pred = pred
    metrics = pd.DataFrame(rows)
    trades, trade_rows = test[["timestamp", "spread"]].copy(), []
    # Point-only and no-trade baselines share exactly the same test observations.
    signals = {"no_trade": np.zeros(len(test), dtype=int),
               "point": trading_signals(spread_pred, spread_pred, cost)}
    signals.update({name: trading_signals(lo, hi, cost) for name, (lo, hi) in spread_bounds.items()})
    for name, position in signals.items():
        pnl = position * test.spread.to_numpy() - cost * np.abs(position)
        equity = np.r_[0., np.cumsum(pnl)]
        trades[f"{name}_position"] = position
        trades[f"{name}_pnl"] = pnl
        active = position != 0
        trade_rows.append({"strategy": name, "trades": int(active.sum()),
                           "total_pnl": float(pnl.sum()),
                           "pnl_per_trade": float(pnl[active].mean()) if active.any() else 0.,
                           "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity))})
    trading = pd.DataFrame(trade_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    trading.to_csv(output / "trading_summary.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    trades.to_csv(output / "trades.csv", index=False)
    metadata = {"alpha": alpha, "window_hours": window, "cost_per_mwh": cost,
                "train_rows": len(train), "calibration_rows": len(cal), "test_rows": len(test),
                "test_start": str(test.timestamp.iloc[0]), "test_end": str(test.timestamp.iloc[-1])}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    count = min(168, len(test))
    t = test.timestamp.iloc[:count]
    lo, hi = spread_bounds["cqr_rolling"]
    axes[0].plot(t, test.spread.iloc[:count], label="Realized RT - DA", linewidth=1)
    axes[0].plot(t, spread_pred[:count], label="Point forecast", linewidth=1)
    axes[0].fill_between(t, lo[:count], hi[:count], alpha=.25, label="Rolling CQR interval")
    axes[0].set_ylabel("Price spread (currency/MWh)")
    axes[0].legend()
    for name in signals:
        axes[1].plot(test.timestamp, trades[f"{name}_pnl"].cumsum(), label=name)
    axes[1].set_ylabel("Cumulative virtual P&L (currency)")
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "results.png", dpi=150)
    plt.close(fig)
    print(metrics.to_string(index=False))
    print(trading.to_string(index=False))
    return metrics, trading


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Hourly CSV; otherwise use synthetic demonstration data.")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--window", type=int, default=336, help="Rolling calibration window in hours.")
    parser.add_argument("--cost", type=float, default=2., help="All-in round-trip cost per MWh traded.")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = pd.read_csv(args.csv) if args.csv else make_demo(args.days, args.seed)
    run(raw, args.output, args.alpha, args.window, args.cost)
