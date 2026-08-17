"""
Naive and seasonal-naive baselines -- the backtest floor (CLAUDE.md Phase 2,
step 1). Both plug into harness.run_backtest via the shared
predict_fn(train, origin, horizon) -> [commodity, centre, pred] interface.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from harness import TARGET


def naive_predict(train: pl.DataFrame, origin: dt.date, horizon: int) -> pl.DataFrame:
    """Flat forecast: repeat the last known price, for every horizon.
    Requires a row exactly at `origin` -- if a series has a reporting gap
    there, it has no last-known price to forecast from and is skipped
    (same requirement every other model in this harness has, so all models
    are compared on the same evaluated population)."""
    return train.filter(pl.col("date") == origin).select(
        ["commodity", "centre", pl.col(TARGET).alias("pred")]
    )


def seasonal_naive_predict(train: pl.DataFrame, origin: dt.date, horizon: int) -> pl.DataFrame:
    """Forecast = actual price 365 days before the target date (same
    calendar date last year). Falls back to the naive flat forecast for any
    series where that date doesn't exist in training history (gap, or not
    enough history yet)."""
    target_date = origin + dt.timedelta(days=horizon)
    season_date = target_date - dt.timedelta(days=365)

    seasonal = train.filter(pl.col("date") == season_date).select(
        ["commodity", "centre", pl.col(TARGET).alias("seasonal_pred")]
    )
    flat = naive_predict(train, origin, horizon).rename({"pred": "flat_pred"})

    merged = flat.join(seasonal, on=["commodity", "centre"], how="left")
    return merged.with_columns(pred=pl.coalesce(["seasonal_pred", "flat_pred"])).select(
        ["commodity", "centre", "pred"]
    )
