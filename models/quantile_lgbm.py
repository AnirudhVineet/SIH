"""
Quantile LightGBM (CLAUDE.md Phase 2, step 4): alpha in {0.1, 0.5, 0.9} at
horizons {7, 14}. Gives the uncertainty band the officer-facing sentence
needs ("80% chance tur crosses X"), not just a point estimate.

Same pooled-across-series design and feature set as lgbm_model.py (see its
docstring for the rationale) -- just three LightGBM regressors per horizon
instead of one, each trained with objective="quantile" at its own alpha,
rather than a new architecture.

**Conformal calibration (CQR).** Raw quantile-LGBM bands were measurably
over-confident -- 71.3%/74.1% empirical coverage against an 80% target,
and only 63-65% for onion (QUESTIONS.md #6). Since these bands go on
screen in front of officers, the band is now calibrated with split
conformal prediction (Conformalized Quantile Regression, Romano et al.
2019):

  1. Split the supervised training rows by DATE into proper-train
     (everything older) and calibration (the most recent
     CALIBRATION_WINDOW_DAYS). A time-based split, not random -- random
     would leak future regime information into calibration and report
     optimistic coverage.
  2. Fit the quantile models on proper-train only.
  3. On the held-out calibration rows, score each with the CQR conformity
     score E = max(q_lo - y, y - q_hi): how far outside the band the truth
     actually fell (negative when comfortably inside).
  4. Take the ceil((n+1)*(1-alpha))/n empirical quantile of E -- the finite-
     sample-corrected level that gives CQR its coverage guarantee.
  5. At predict time widen the band to [q_lo - Q, q_hi + Q].

Q is computed **per commodity**, because the miscoverage was very uneven
across commodities -- one pooled correction would over-widen tur/potato to
fix onion. Commodities with no calibration rows fall back to the pooled Q.
"""

from __future__ import annotations

import datetime as dt
import math

import lightgbm as lgb
import numpy as np
import polars as pl

from features import CATEGORICAL_COLUMNS, build_features, to_model_frame
from harness import make_supervised

ALPHAS = [0.1, 0.5, 0.9]
LOW_ALPHA, HIGH_ALPHA = 0.1, 0.9
TARGET_COVERAGE = HIGH_ALPHA - LOW_ALPHA  # 0.8

# Most recent slice of training data held out to calibrate the band. 180
# days keeps a few thousand calibration rows per commodity (ample for an
# 80% quantile of the conformity score) while leaving the bulk of the
# 10-year history for fitting.
CALIBRATION_WINDOW_DAYS = 180

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


def rearrange_quantiles(band: pl.DataFrame) -> pl.DataFrame:
    """Sorts p10/p50/p90 within each row so the band is always monotonic.

    The three alphas are fitted as independent LightGBM models, so nothing
    ties them together and they can cross -- measured at ~1.9% of rows on
    the real backtest, e.g. a p50 landing above its own p90. A crossed band
    is nonsense to display (an inverted uncertainty interval on screen) and
    would break the P10-P90 shading in the dashboard.

    Sorting is the standard fix (monotone rearrangement, Chernozhukov et
    al. 2010) and is not a fudge: rearranging a set of estimated quantile
    curves weakly reduces estimation error, so the sorted band is never a
    worse estimate than the crossed one.
    """
    lo = pl.min_horizontal("p10", "p50", "p90")
    hi = pl.max_horizontal("p10", "p50", "p90")
    mid = pl.col("p10") + pl.col("p50") + pl.col("p90") - lo - hi
    return band.with_columns(p10=lo, p50=mid, p90=hi)


def conformal_quantile(scores: np.ndarray, target_coverage: float = TARGET_COVERAGE) -> float:
    """The split-conformal correction: the ceil((n+1)*target)/n empirical
    quantile of the conformity scores. The (n+1) is the finite-sample
    correction that makes the coverage guarantee hold for small n -- with
    plain `np.quantile(scores, target)` the band systematically
    under-covers on small calibration sets. Returns 0.0 for an empty score
    set (no calibration data -> no correction applied)."""
    n = len(scores)
    if n == 0:
        return 0.0
    level = math.ceil((n + 1) * target_coverage) / n
    if level > 1.0:  # too few points to reach the target level at all
        return float(np.max(scores))
    return float(np.quantile(scores, level, method="higher"))


def _fit_quantiles(engineered: pl.DataFrame, horizon: int) -> dict:
    """Fits the three quantile models on a time-based proper-train split and
    derives the per-commodity conformal correction on the held-out recent
    calibration slice. Returns {"models": ..., "q_by_commodity": ...,
    "q_pooled": ...}."""
    sup = make_supervised(engineered, horizon).drop_nulls(subset=["target"]).sort("date")

    split_date = sup["date"].max() - dt.timedelta(days=CALIBRATION_WINDOW_DAYS)
    proper = sup.filter(pl.col("date") <= split_date)
    calib = sup.filter(pl.col("date") > split_date)

    # Degenerate case (very short history): fall back to fitting on
    # everything and applying no correction, rather than fitting on nothing.
    if proper.height == 0 or calib.height == 0:
        proper, calib = sup, sup.head(0)

    X_proper = to_model_frame(proper)
    y_proper = proper["target"].to_pandas()

    models = {}
    for alpha in ALPHAS:
        model = lgb.LGBMRegressor(alpha=alpha, **QUANTILE_PARAMS)
        model.fit(X_proper, y_proper, categorical_feature=CATEGORICAL_COLUMNS)
        models[alpha] = model

    q_by_commodity: dict[str, float] = {}
    q_pooled = 0.0
    if calib.height > 0:
        X_calib = to_model_frame(calib)
        lo = models[LOW_ALPHA].predict(X_calib)
        hi = models[HIGH_ALPHA].predict(X_calib)
        y_calib = calib["target"].to_numpy()
        # CQR conformity score: how far outside the band the truth fell
        # (negative when comfortably inside).
        scores = np.maximum(lo - y_calib, y_calib - hi)

        scored = calib.select("commodity").with_columns(score=pl.Series(scores))
        q_pooled = conformal_quantile(scores)
        for (commodity,), group in scored.group_by(["commodity"]):
            q_by_commodity[commodity] = conformal_quantile(group["score"].to_numpy())

    return {"models": models, "q_by_commodity": q_by_commodity, "q_pooled": q_pooled}


def _state_for_origin(train: pl.DataFrame, origin: dt.date) -> dict:
    if origin not in _cache:
        _cache[origin] = {"engineered": build_features(train), "models": {}}
    return _cache[origin]


def fitted_for(train: pl.DataFrame, origin: dt.date, horizon: int) -> dict:
    """The fitted quantile models + conformal corrections for this
    (origin, horizon), fitting and caching on first use. Public so callers
    that need to score rows other than the origin row (e.g. the dashboard
    artifact builder, which scores each centre at its own latest reading)
    can reuse the same fit."""
    state = _state_for_origin(train, origin)
    if horizon not in state["models"]:
        state["models"][horizon] = _fit_quantiles(state["engineered"], horizon)
    return state["models"][horizon]


def band_from_fitted(
    fitted: dict, rows: pl.DataFrame, conformal: bool = True
) -> pl.DataFrame:
    """Applies an already-fitted quantile set to an arbitrary engineered
    feature frame -> commodity, centre, p10, p50, p90 (monotonic, and
    conformally widened unless conformal=False)."""
    if rows.height == 0:
        return pl.DataFrame(
            schema={
                "commodity": pl.String,
                "centre": pl.String,
                "p10": pl.Float64,
                "p50": pl.Float64,
                "p90": pl.Float64,
            }
        )

    models = fitted["models"]
    X = to_model_frame(rows)
    preds = {a: models[a].predict(X) for a in ALPHAS}

    band = rows.select(["commodity", "centre"]).with_columns(
        p10=pl.Series(preds[LOW_ALPHA]),
        p50=pl.Series(preds[0.5]),
        p90=pl.Series(preds[HIGH_ALPHA]),
    )
    band = rearrange_quantiles(band)
    if not conformal:
        return band

    q = (
        band["commodity"]
        .replace_strict(
            fitted["q_by_commodity"], default=fitted["q_pooled"], return_dtype=pl.Float64
        )
        .to_numpy()
    )
    return band.with_columns(
        p10=pl.col("p10") - pl.Series(q), p90=pl.col("p90") + pl.Series(q)
    )


def quantile_predict_band(
    train: pl.DataFrame, origin: dt.date, horizon: int, conformal: bool = True
) -> pl.DataFrame:
    """commodity, centre, p10, p50, p90 -- the full band, conformally
    widened by default. Not the predict_fn shape harness.run_backtest
    wants; see quantile_median_predict for that.

    conformal=False returns the raw quantile-model band, which is what the
    before/after coverage comparison in PROGRESS.md is measured against."""
    fitted = fitted_for(train, origin, horizon)
    today = _cache[origin]["engineered"].filter(pl.col("date") == origin)
    return band_from_fitted(fitted, today, conformal=conformal)


def quantile_median_predict(train: pl.DataFrame, origin: dt.date, horizon: int) -> pl.DataFrame:
    """predict_fn-shaped wrapper (median only) so this plugs into
    harness.run_backtest exactly like every other model."""
    band = quantile_predict_band(train, origin, horizon)
    return band.select(["commodity", "centre", pl.col("p50").alias("pred")])


def coverage_at_origin(
    frame: pl.DataFrame, origin: dt.date, horizon: int, conformal: bool = True
) -> pl.DataFrame:
    """actual vs. [p10, p90] band for one origin/horizon -- the raw rows
    behind the 80% prediction-interval coverage check. `frame` is the full
    (unfiltered) dataset so the actual at origin+horizon can be looked up;
    training only ever uses frame filtered to <= origin, same as everywhere
    else in this harness."""
    train = frame.filter(pl.col("date") <= origin)
    band = quantile_predict_band(train, origin, horizon, conformal=conformal)

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
