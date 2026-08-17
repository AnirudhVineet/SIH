"""
SARIMAX benchmark (CLAUDE.md Phase 2, step 2) -- this is what DoCA uses
today, so it's the number LightGBM has to visibly beat.

Order (1,1,1), no seasonal term. Timed at ~0.1s/fit on ~11 years of daily
data (see PROGRESS.md), so refitting per (series, origin) in the
walk-forward loop is cheap -- no need to cache across origins or shrink the
training window. This isn't meant to be a hand-tuned SARIMAX; it's the
"linear, mostly univariate" baseline CLAUDE.md describes DoCA's current
ARIMA setup as, i.e. the fair floor to clear.

Trained per (commodity, centre) on its full history up to the origin,
reindexed to daily frequency so forecast step k always means "k calendar
days ahead" regardless of reporting gaps (missing days become NaN;
statsmodels' state-space SARIMAX handles NaN observations natively via the
Kalman filter, no imputation needed here).
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import polars as pl
from statsmodels.tsa.statespace.sarimax import SARIMAX

from harness import HORIZONS, TARGET

ORDER = (1, 1, 1)
MIN_OBS = 60  # fewer real (non-NaN) observations than this and a fit isn't trustworthy

# Cache is keyed by (origin, commodity, centre) and lives for the process
# lifetime: harness.eval_at_origin calls predict_fn once per horizon for the
# same origin, and the fit (and its whole 14-day-ahead forecast curve) is
# identical for both calls, so this avoids refitting on the second call.
_forecast_cache: dict[tuple[dt.date, str, str], pd.Series | None] = {}


def is_plausible_forecast(forecast: pd.Series, recent_level: float, max_multiple: float = 8.0) -> bool:
    """enforce_stationarity=False occasionally lets the optimizer land on an
    explosive AR root (e.g. ar.L1 > 1) even when it reports convergence; for
    a d=1 model that means forecasts blow up exponentially within a handful
    of steps. A commodity price forecast has no business going negative or
    landing an order of magnitude off the recent trading range, so treat
    that as a failed fit rather than an outlier. Split out from
    _fit_and_forecast so it's unit-testable without a real SARIMAX fit."""
    return bool((forecast > 0).all() and (forecast <= max_multiple * recent_level).all())


def _fit_and_forecast(series: pd.Series, steps: int) -> pd.Series | None:
    observed = series.dropna()
    if observed.shape[0] < MIN_OBS:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                series, order=ORDER, enforce_stationarity=False, enforce_invertibility=False
            )
            res = model.fit(disp=False, maxiter=50)
            if not res.mle_retvals.get("converged", True):
                return None
            forecast = res.get_forecast(steps=steps).predicted_mean
        except Exception:
            return None

    recent_level = observed.iloc[-30:].mean()
    if not is_plausible_forecast(forecast, recent_level):
        return None
    return forecast


def _forecasts_for_origin(train: pl.DataFrame, origin: dt.date) -> dict:
    max_h = max(HORIZONS)
    series_keys = train.select(["commodity", "centre"]).unique().rows()

    for commodity, centre in series_keys:
        cache_key = (origin, commodity, centre)
        if cache_key in _forecast_cache:
            continue
        sub = (
            train.filter((pl.col("commodity") == commodity) & (pl.col("centre") == centre))
            .select(["date", TARGET])
            .sort("date")
            .to_pandas()
        )
        sub["date"] = pd.to_datetime(sub["date"])
        endog = sub.set_index("date")[TARGET].asfreq("D")
        _forecast_cache[cache_key] = _fit_and_forecast(endog, max_h)

    return {(c, ce): _forecast_cache[(origin, c, ce)] for c, ce in series_keys}


def sarimax_predict(train: pl.DataFrame, origin: dt.date, horizon: int) -> pl.DataFrame:
    forecasts = _forecasts_for_origin(train, origin)
    target_date = pd.Timestamp(origin) + pd.Timedelta(days=horizon)

    rows = []
    for (commodity, centre), fc in forecasts.items():
        if fc is None or target_date not in fc.index:
            continue
        rows.append({"commodity": commodity, "centre": centre, "pred": float(fc.loc[target_date])})

    if not rows:
        return pl.DataFrame(
            schema={"commodity": pl.String, "centre": pl.String, "pred": pl.Float64}
        )
    return pl.DataFrame(rows)
