"""
LightGBM, direct multi-horizon (CLAUDE.md Phase 2, step 3), narrowed to
horizons {7, 14} for this run -- the workhorse model.

One pooled model per (origin, horizon): trained across ALL series
(commodity, centre) at once, with commodity/centre as categorical
features, rather than 30 separate per-series models. Deliberate choice,
matching the "one global model, not 12,100 models" scaling story in
CLAUDE.md's Q&A prep -- it also lets each horizon's model borrow strength
across series (e.g. a rainfall signal is more informative pooled across
centres than fit on any single centre's few thousand rows alone).

Features: models/features.py's date-safe engineered lags/rolling/EWMA/
momentum, plus the pre-built columns confirmed date-correct (weather,
festivals, calendar, cross-commodity, months_since_harvest). Target: actual
wholesale_price at date+horizon, built with the same date-based join
harness.make_supervised uses (not a row-positional shift, for the same
reporting-gap reason).
"""

from __future__ import annotations

import datetime as dt

import lightgbm as lgb
import polars as pl

from features import CATEGORICAL_COLUMNS, build_features, to_model_frame
from harness import make_supervised

LGBM_PARAMS = dict(
    objective="regression",
    metric="mae",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    verbosity=-1,
)

# Cache engineered features + fitted models per origin: harness calls
# predict_fn once per horizon for the same origin, and feature engineering
# (and the horizon-7 model) shouldn't be redone for the horizon-14 call.
_cache: dict[dt.date, dict] = {}


def _fit_for_horizon(engineered: pl.DataFrame, horizon: int) -> lgb.LGBMRegressor:
    sup = make_supervised(engineered, horizon).drop_nulls(subset=["target"])
    X = to_model_frame(sup)
    y = sup["target"].to_pandas()

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y, categorical_feature=CATEGORICAL_COLUMNS)
    return model


def _state_for_origin(train: pl.DataFrame, origin: dt.date) -> dict:
    if origin not in _cache:
        _cache[origin] = {"engineered": build_features(train), "models": {}}
    return _cache[origin]


def lgbm_predict(train: pl.DataFrame, origin: dt.date, horizon: int) -> pl.DataFrame:
    state = _state_for_origin(train, origin)
    if horizon not in state["models"]:
        state["models"][horizon] = _fit_for_horizon(state["engineered"], horizon)
    model = state["models"][horizon]

    today = state["engineered"].filter(pl.col("date") == origin)
    if today.height == 0:
        return pl.DataFrame(
            schema={"commodity": pl.String, "centre": pl.String, "pred": pl.Float64}
        )

    preds = model.predict(to_model_frame(today))
    return today.select(["commodity", "centre"]).with_columns(pred=pl.Series(preds))
