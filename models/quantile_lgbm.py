"""
Quantile LightGBM (CLAUDE.md Phase 2, step 4): alpha in {0.1, 0.5, 0.9} at
horizons {7, 14}. Gives the uncertainty band the officer-facing sentence
needs ("80% chance tur crosses X"), not just a point estimate.

Same pooled-across-series design and feature set as lgbm_model.py (see its
docstring for the rationale) -- just three LightGBM regressors per horizon
instead of one, each trained with objective="quantile" at its own alpha,
rather than a new architecture.
"""

from __future__ import annotations

import datetime as dt

import lightgbm as lgb
import polars as pl

from features import CATEGORICAL_COLUMNS, build_features, to_model_frame
from harness import make_supervised

ALPHAS = [0.1, 0.5, 0.9]

QUANTILE_PARAMS = dict(
    objective="quantile",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    verbosity=-1,
)

# Separate from lgbm_model's cache (different module, different dict), same
# per-origin reuse rationale: harness calls a predict_fn once per horizon.
_cache: dict[dt.date, dict] = {}


def _fit_quantiles(engineered: pl.DataFrame, horizon: int) -> dict[float, lgb.LGBMRegressor]:
    sup = make_supervised(engineered, horizon).drop_nulls(subset=["target"])
    X = to_model_frame(sup)
    y = sup["target"].to_pandas()

    models = {}
    for alpha in ALPHAS:
        model = lgb.LGBMRegressor(alpha=alpha, **QUANTILE_PARAMS)
        model.fit(X, y, categorical_feature=CATEGORICAL_COLUMNS)
        models[alpha] = model
    return models


def _state_for_origin(train: pl.DataFrame, origin: dt.date) -> dict:
    if origin not in _cache:
        _cache[origin] = {"engineered": build_features(train), "models": {}}
    return _cache[origin]


def quantile_predict_band(train: pl.DataFrame, origin: dt.date, horizon: int) -> pl.DataFrame:
    """commodity, centre, p10, p50, p90 -- the full band. Not the predict_fn
    shape harness.run_backtest wants; see quantile_median_predict for that."""
    state = _state_for_origin(train, origin)
    if horizon not in state["models"]:
        state["models"][horizon] = _fit_quantiles(state["engineered"], horizon)
    models = state["models"][horizon]

    today = state["engineered"].filter(pl.col("date") == origin)
    if today.height == 0:
        return pl.DataFrame(
            schema={
                "commodity": pl.String,
                "centre": pl.String,
                "p10": pl.Float64,
                "p50": pl.Float64,
                "p90": pl.Float64,
            }
        )

    X = to_model_frame(today)
    preds = {a: models[a].predict(X) for a in ALPHAS}
    return today.select(["commodity", "centre"]).with_columns(
        p10=pl.Series(preds[0.1]), p50=pl.Series(preds[0.5]), p90=pl.Series(preds[0.9])
    )


def quantile_median_predict(train: pl.DataFrame, origin: dt.date, horizon: int) -> pl.DataFrame:
    """predict_fn-shaped wrapper (median only) so this plugs into
    harness.run_backtest exactly like every other model."""
    band = quantile_predict_band(train, origin, horizon)
    return band.select(["commodity", "centre", pl.col("p50").alias("pred")])


def coverage_at_origin(
    frame: pl.DataFrame, origin: dt.date, horizon: int
) -> pl.DataFrame:
    """actual vs. [p10, p90] band for one origin/horizon -- the raw rows
    behind the 80% prediction-interval coverage check. `frame` is the full
    (unfiltered) dataset so the actual at origin+horizon can be looked up;
    training only ever uses frame filtered to <= origin, same as everywhere
    else in this harness."""
    train = frame.filter(pl.col("date") <= origin)
    band = quantile_predict_band(train, origin, horizon)

    target_date = origin + dt.timedelta(days=horizon)
    actual = frame.filter((pl.col("date") == target_date) & (~pl.col("is_imputed"))).select(
        ["commodity", "centre", pl.col("wholesale_price").alias("actual")]
    )
    return actual.join(band, on=["commodity", "centre"], how="inner").with_columns(
        origin=pl.lit(origin),
        horizon=pl.lit(horizon, dtype=pl.Int64),
        inside_band=(pl.col("actual") >= pl.col("p10")) & (pl.col("actual") <= pl.col("p90")),
        band_width=pl.col("p90") - pl.col("p10"),
    )
