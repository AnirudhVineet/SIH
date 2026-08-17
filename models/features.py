"""
Date-safe price feature engineering for LightGBM.

modelling_frame.parquet's pre-built price_lag_*/price_roll_*/price_ewma_*/
momentum_30 columns were computed with row-positional shift/rolling over a
series that has real reporting gaps -- confirmed wrong immediately after
any gap (see QUESTIONS.md #4, PROGRESS.md). This module recomputes the same
family of features directly from (commodity, centre, date, wholesale_price),
calendar-correct, and is meant to replace those columns for modelling.
Everything else in the parquet (weather, festivals, calendar,
cross-commodity prices, months_since_harvest) was checked and is
date-correct, so it's reused as-is via TRUSTED_COLUMNS.

Correctness approach, all per (commodity, centre) group sorted by date:
  - lags: pandas `Series.shift(n, freq="D")` shifts the *index* by n days
    (not n rows), so reindexing back onto the original dates yields "value
    n calendar days ago" and is naturally NaN wherever that date has no
    data -- exactly what a gappy series needs.
  - rolling mean/std: pandas `.rolling("Nd")` on a DatetimeIndex is a
    calendar-time window (whatever points actually fall in the last N
    days), not an N-row window.
  - EWMA: pandas `.ewm(halflife="Nd", times=...)` decays by elapsed
    calendar time rather than row count. Parameterized directly by
    halflife in days rather than "span", since span-based decay has no
    well-defined meaning for irregularly-spaced data.

Only `train` (data already filtered to date <= origin by the harness) is
ever passed in, so features naturally never see future information.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from harness import TARGET

TRUSTED_COLUMNS = [
    "rainfall_mm",
    "temp_max",
    "temp_min",
    "rainfall_7d_sum",
    "rainfall_30d_sum",
    "rainfall_dev_from_normal",
    "month",
    "dow",
    "doy_sin",
    "doy_cos",
    "festival_diwali",
    "festival_navratri",
    "festival_eid",
    "festival_onam",
    "months_since_harvest",
    "price_potato_same_state",
    "price_onion_same_state",
    "price_tur_same_state",
]

LAG_DAYS = [1, 7, 14, 30, 90]
ROLL_WINDOWS = [7, 30, 90]
EWMA_HALFLIVES = [7, 30]
MOMENTUM_DAYS = 30

ENGINEERED_COLUMNS = (
    [f"price_lag_{n}" for n in LAG_DAYS]
    + [f"price_roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"price_roll_std_{w}" for w in ROLL_WINDOWS]
    + [f"price_ewma_{s}" for s in EWMA_HALFLIVES]
    + ["momentum_30"]
)

FEATURE_COLUMNS = TRUSTED_COLUMNS + ENGINEERED_COLUMNS
CATEGORICAL_COLUMNS = ["commodity", "centre"]
MODEL_COLUMNS = CATEGORICAL_COLUMNS + FEATURE_COLUMNS


def _engineer_series(g: pd.DataFrame) -> pd.DataFrame:
    commodity, centre = g.name
    g = g.sort_values("date")
    idx = pd.DatetimeIndex(g["date"])
    s = pd.Series(g[TARGET].to_numpy(), index=idx)

    out = {}
    for n in LAG_DAYS:
        out[f"price_lag_{n}"] = s.shift(n, freq="D").reindex(idx).to_numpy()
    for w in ROLL_WINDOWS:
        out[f"price_roll_mean_{w}"] = s.rolling(f"{w}D").mean().to_numpy()
        out[f"price_roll_std_{w}"] = s.rolling(f"{w}D").std().to_numpy()
    for h in EWMA_HALFLIVES:
        out[f"price_ewma_{h}"] = s.ewm(halflife=f"{h}D", times=idx).mean().to_numpy()

    result = g.copy()
    result["commodity"] = commodity
    result["centre"] = centre
    for col, values in out.items():
        result[col] = values
    result["momentum_30"] = result[TARGET] / result[f"price_lag_{MOMENTUM_DAYS}"] - 1
    return result


def build_features(train: pl.DataFrame) -> pl.DataFrame:
    """train -> same rows, with ENGINEERED_COLUMNS added (calendar-safe),
    keeping TRUSTED_COLUMNS untouched. Safe to call repeatedly on growing
    training windows as the walk-forward origin advances."""
    pdf = train.select(
        ["commodity", "centre", "date", TARGET] + TRUSTED_COLUMNS
    ).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])

    engineered = pdf.groupby(["commodity", "centre"], group_keys=False).apply(
        _engineer_series, include_groups=False
    )
    engineered["date"] = engineered["date"].dt.date

    return pl.from_pandas(engineered)
