"""
Walk-forward backtest harness shared by every Phase 2 model.

Rolling-origin evaluation per CLAUDE.md: train on everything up to origin
date T, forecast T+h, step T forward, repeat across the last two years of
data. Horizons are narrowed to {7, 14} days for this run (no 1d/30d).

Ground truth for scoring is restricted to non-imputed rows (see
QUESTIONS.md #3 / PROGRESS.md): ~26% of rows are forward-filled or
state-median imputed, and scoring against a fabricated placeholder would
misrepresent real accuracy. Models may still *train* on imputed rows -- only
evaluation targets are filtered.

Every model plugs in via one function shape:

    predict_fn(train: pl.DataFrame, origin: date, horizon: int) -> pl.DataFrame
        columns: commodity, centre, pred

`run_backtest` handles origin/horizon looping, actual-value lookup, and
long-format result assembly; each model only needs to implement how it turns
training history into a prediction per series.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
FRAME_PATH = ROOT / "data" / "processed" / "modelling_frame.parquet"

TARGET = "wholesale_price"
HORIZONS = [7, 14]
STEP_DAYS = 30
BACKTEST_YEARS = 2
MIN_TRAIN_OBS = 90  # minimum observed (non-gap) rows a series needs before we trust a fit to it

PredictFn = Callable[[pl.DataFrame, dt.date, int], pl.DataFrame]


def load_frame() -> pl.DataFrame:
    return pl.read_parquet(FRAME_PATH).sort(["commodity", "centre", "date"])


def generate_origins(
    frame: pl.DataFrame,
    step_days: int = STEP_DAYS,
    backtest_years: int = BACKTEST_YEARS,
    horizons: list[int] = HORIZONS,
) -> list[dt.date]:
    """Origins spaced step_days apart, covering the last backtest_years of
    the available range, each leaving room for the largest horizon to land
    inside the data."""
    max_date: dt.date = frame["date"].max()
    max_horizon = max(horizons)
    last_origin = max_date - dt.timedelta(days=max_horizon)
    start = last_origin - dt.timedelta(days=365 * backtest_years)

    origins = []
    d = start
    while d <= last_origin:
        origins.append(d)
        d += dt.timedelta(days=step_days)
    return origins


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def series_obs_count(frame: pl.DataFrame, origin: dt.date) -> pl.DataFrame:
    """Non-imputed observation count per (commodity, centre) up to origin --
    used by models to decide whether a series has enough real history to fit."""
    return (
        frame.filter((pl.col("date") <= origin) & (~pl.col("is_imputed")))
        .group_by(["commodity", "centre"])
        .agg(pl.len().alias("n_obs"))
    )


def eval_at_origin(
    frame: pl.DataFrame, origin: dt.date, horizon: int, predict_fn: PredictFn
) -> pl.DataFrame:
    """Runs one model at one (origin, horizon) and returns long-format rows:
    commodity, centre, origin, horizon, actual, pred."""
    train = frame.filter(pl.col("date") <= origin)
    target_date = origin + dt.timedelta(days=horizon)

    actual = frame.filter((pl.col("date") == target_date) & (~pl.col("is_imputed"))).select(
        ["commodity", "centre", pl.col(TARGET).alias("actual")]
    )
    if actual.height == 0:
        return actual.with_columns(
            pred=pl.lit(None, dtype=pl.Float64),
            origin=pl.lit(origin),
            horizon=pl.lit(horizon, dtype=pl.Int64),
        ).select(["commodity", "centre", "origin", "horizon", "actual", "pred"])

    preds = predict_fn(train, origin, horizon)
    joined = actual.join(preds, on=["commodity", "centre"], how="inner").with_columns(
        origin=pl.lit(origin), horizon=pl.lit(horizon, dtype=pl.Int64)
    )
    return joined.select(["commodity", "centre", "origin", "horizon", "actual", "pred"])


def run_backtest(
    frame: pl.DataFrame,
    predict_fn: PredictFn,
    model_name: str,
    origins: list[dt.date] | None = None,
    horizons: list[int] = HORIZONS,
    verbose: bool = True,
) -> pl.DataFrame:
    """Runs predict_fn across every (origin, horizon), returns one long
    DataFrame with a `model` column added."""
    if origins is None:
        origins = generate_origins(frame, horizons=horizons)

    rows = []
    for i, origin in enumerate(origins):
        for h in horizons:
            r = eval_at_origin(frame, origin, h, predict_fn)
            rows.append(r)
        if verbose:
            print(f"  [{model_name}] origin {i + 1}/{len(origins)} ({origin}) done", flush=True)

    out = pl.concat(rows, how="vertical").with_columns(model=pl.lit(model_name))
    return out.drop_nulls(subset=["actual", "pred"])


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """Long-format results -> per (model, commodity, horizon) MAPE/RMSE.
    Vectorized (not the scalar mape()/rmse() helpers above) so it stays a
    plain polars aggregation -- no UDF return-type inference issues."""
    return (
        results.filter(pl.col("actual") != 0)
        .with_columns(
            abs_pct_err=((pl.col("actual") - pl.col("pred")).abs() / pl.col("actual").abs()) * 100,
            sq_err=(pl.col("actual") - pl.col("pred")) ** 2,
        )
        .group_by(["model", "commodity", "horizon"])
        .agg(mape=pl.col("abs_pct_err").mean(), rmse=pl.col("sq_err").mean().sqrt(), n=pl.len())
        .sort(["commodity", "horizon", "model"])
    )


def make_supervised(frame: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Adds `target` = TARGET at date+horizon (calendar days) for the same
    (commodity, centre), for direct multi-horizon training.

    Deliberately a date-based join, not a row-positional `.shift(-horizon)`:
    series have real reporting gaps (see PROGRESS.md), so the row `horizon`
    positions ahead can be weeks or years later in calendar time, not
    `horizon` days later. A positional shift would silently mislabel targets
    across those gaps.

    Rows with no exact date+horizon match (series gap, or tail of history)
    get a null target and should be dropped by the caller.
    """
    future = frame.select(
        ["commodity", "centre", "date", pl.col(TARGET).alias("target")]
    ).with_columns((pl.col("date") - pl.duration(days=horizon)).alias("origin_date"))

    return frame.join(
        future.select(["commodity", "centre", "origin_date", "target"]),
        left_on=["commodity", "centre", "date"],
        right_on=["commodity", "centre", "origin_date"],
        how="left",
    )
