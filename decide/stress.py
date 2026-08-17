"""
Price Stress Index, 0-100, per (commodity, centre).

Three components, all from real fitted-model output -- no sourced-data
inputs are required, which is why this survives the arrivals/retail gaps
that block the rest of the Phase 4 spec:

  1. Forecast level   -- P50 forecast vs that centre's own trailing 1-year
                         median price. "Is this heading above its own normal?"
                         Using the centre's own history rather than a national
                         figure means a Rs 900/qtl onion centre and a Rs 1,600
                         one are judged on the same footing.
  2. Band width       -- (P90 - P10) as a share of current price. Wide band =
                         the model itself is unsure = more reason for an
                         officer to look.
  3. Spike probability-- classifier P(>8% rise within 14 days), used directly.

Each is clipped to 0-1, weighted, and scaled to 0-100. Weights and full-scale
points are module constants so they are auditable rather than buried.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

# Forecast level carries the most weight (it is the actual price signal);
# spike probability is the forward-looking early warning; band width is a
# confidence modifier rather than a signal in its own right.
W_LEVEL, W_SPIKE, W_BAND = 0.45, 0.35, 0.20

# Full-scale points -- where each component saturates at 1.0.
# +25% above a centre's own 1-year median is a serious move for these
# commodities; observed 80% band widths run ~9-75% of price (median ~39%).
LEVEL_FULL_SCALE = 0.25
BAND_FULL_SCALE = 0.60

BANDS = [(60, "High"), (45, "Elevated"), (30, "Moderate"), (0, "Low")]


def trailing_median(
    history: pl.DataFrame, as_of: dt.date, window_days: int = 365
) -> pl.DataFrame:
    """Median real (non-imputed) price per (commodity, centre) over the
    trailing window. Imputed rows are excluded so a run of forward-filled
    values cannot drag the baseline."""
    start = as_of - dt.timedelta(days=window_days)
    return (
        history.filter(
            (pl.col("date") > start)
            & (pl.col("date") <= as_of)
            & (~pl.col("is_imputed"))
        )
        .group_by(["commodity", "centre"])
        .agg(median_1yr=pl.col("wholesale_price").median())
    )


def score(forecasts: pl.DataFrame, medians: pl.DataFrame) -> pl.DataFrame:
    """forecasts (one horizon) + trailing medians -> stress components and
    a 0-100 score per (commodity, centre)."""
    joined = forecasts.join(medians, on=["commodity", "centre"], how="left")

    return (
        joined.with_columns(
            level_component=(
                (pl.col("p50") / pl.col("median_1yr") - 1) / LEVEL_FULL_SCALE
            ).clip(0, 1),
            band_component=(
                (pl.col("band_width_pct") / 100) / BAND_FULL_SCALE
            ).clip(0, 1),
            spike_component=pl.col("spike_proba").fill_null(0).clip(0, 1),
        )
        .with_columns(
            # A missing 1-year median (a centre with no real history in the
            # window) must not silently score 0 on the level term -- fall back
            # to the other two components reweighted.
            level_component=pl.when(pl.col("median_1yr").is_null())
            .then(None)
            .otherwise(pl.col("level_component"))
        )
        .with_columns(
            stress=pl.when(pl.col("level_component").is_null())
            .then(
                (
                    W_SPIKE * pl.col("spike_component")
                    + W_BAND * pl.col("band_component")
                )
                / (W_SPIKE + W_BAND)
                * 100
            )
            .otherwise(
                (
                    W_LEVEL * pl.col("level_component")
                    + W_SPIKE * pl.col("spike_component")
                    + W_BAND * pl.col("band_component")
                )
                * 100
            )
        )
        .with_columns(stress=pl.col("stress").round(1))
    )


def band_of(value: float) -> str:
    for threshold, name in BANDS:
        if value >= threshold:
            return name
    return "Low"


def add_band(df: pl.DataFrame, column: str = "stress") -> pl.DataFrame:
    return df.with_columns(
        band=pl.col(column).map_elements(band_of, return_dtype=pl.String)
    )
