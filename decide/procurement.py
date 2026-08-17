"""
Procurement (buy-side) signal -- CLAUDE.md Phase 4: "also flag when to buy
(harvest lows) to rebuild the buffer. Closes the loop, shows you understand
the whole PSF cycle rather than half of it."

Two real, data-grounded ingredients, no sourced procurement feed required:

  1. Harvest proximity -- `months_since_harvest`, a real Phase 1 feature
     (harvest-calendar-derived, already trusted and passed through to the
     forecast model). 0 = just harvested, when supply-driven price troughs
     are expected; the component decays to 0 by month 3.
  2. Price discount -- current price vs that centre's own trailing 1-year
     median, using the *same* `stress.trailing_median` baseline the
     sell-side stress score uses. Sharing the baseline matters: a centre
     cannot show High release-stress and an open procurement window off two
     different definitions of "normal" at once.

Combined **multiplicatively**, not additively: `score = 100 * discount *
(floor + (1-floor) * harvest)`. An additive blend was tried first and
rejected -- it let a large price discount alone clear the "Open" threshold
even ten months from harvest, which is a demand-side or oversupply signal,
not the "harvest low" the spec asks this to detect. Multiplying means a
discount far from harvest is capped at a low ceiling (`HARVEST_FLOOR` of the
discount score) and can only reach "Open" when both a real discount and
genuine harvest proximity are present together.

Mirrors decide/stress.py's shape (weighted 0-1 components -> 0-100 score ->
named band) so release and procurement read as one coherent system rather
than two unrelated calculators.
"""

from __future__ import annotations

import polars as pl

# A price 12% below the centre's own 1-year median is a strong procurement
# opportunity; the discount component saturates there.
DISCOUNT_FULL_SCALE = 0.12

# Within this many months of harvest counts as the "harvest window"; the
# component decays linearly from 1.0 (just harvested) to 0 at this many
# months out.
HARVEST_FULL_WINDOW_MONTHS = 3.0

# A discount with zero harvest proximity can reach at most this fraction of
# the full score -- keeps a genuine off-season discount visible (still worth
# a look) without letting it register as an "Open" harvest-low window.
HARVEST_FLOOR = 0.15

BANDS = [(60, "Open"), (35, "Watch"), (0, "Closed")]


def score(forecasts: pl.DataFrame, medians: pl.DataFrame) -> pl.DataFrame:
    """forecasts (current_price, months_since_harvest, one row per
    commodity/centre) + trailing medians -> procurement components and a
    0-100 score."""
    joined = forecasts.join(medians, on=["commodity", "centre"], how="left")

    return (
        joined.with_columns(
            discount_component=(
                (pl.col("median_1yr") / pl.col("current_price") - 1)
                / DISCOUNT_FULL_SCALE
            ).clip(0, 1),
            harvest_component=(
                1 - pl.col("months_since_harvest") / HARVEST_FULL_WINDOW_MONTHS
            ).clip(0, 1),
            months_to_next_harvest=(
                -pl.col("months_since_harvest") % 12
            ),
        )
        .with_columns(
            # No trailing median (a centre with no real history in the
            # window) -> no discount baseline to score against, don't
            # silently score 0.
            procurement_score=pl.when(pl.col("median_1yr").is_null())
            .then(None)
            .otherwise(
                100
                * pl.col("discount_component")
                * (HARVEST_FLOOR + (1 - HARVEST_FLOOR) * pl.col("harvest_component"))
            )
        )
        .with_columns(procurement_score=pl.col("procurement_score").round(1))
    )


def band_of(value: float | None) -> str:
    if value is None:
        return "Closed"
    for threshold, name in BANDS:
        if value >= threshold:
            return name
    return "Closed"


def add_band(df: pl.DataFrame, column: str = "procurement_score") -> pl.DataFrame:
    return df.with_columns(
        band=pl.col(column).map_elements(band_of, return_dtype=pl.String)
    )
