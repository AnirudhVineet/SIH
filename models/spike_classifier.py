"""
Spike classifier (CLAUDE.md Phase 2, step 5): binary -- will this centre's
price rise more than SPIKE_THRESHOLD (8%) at any point in the next
SPIKE_WINDOW_DAYS (14) days?

**The headline metric is median lead time in days, not MAPE.** A forecast
that is accurate but late is useless to a buffer-stock officer; what
matters is how many days of warning they get before the price actually
crosses the threshold.

Optimised for precision/recall rather than squared error, per the spec.

Evaluation design differs from the point-forecast models on purpose. The
other models are scored at 25 walk-forward origins spaced 30 days apart,
which would give only 25 lead-time measurements per series -- far too
sparse for a median lead time to mean anything. Here, each origin's model
instead predicts **every day** in the 30-day span until the next origin.
That mirrors the real deployment pattern (retrain periodically, score
daily) and yields a dense, genuinely out-of-sample daily alert series
across the whole 2-year backtest window.

Labels use only non-imputed prices. ~26% of rows are forward-filled or
state-median imputed; a flat forward-fill can't produce an 8% jump, and
imputed values re-entering the label would invent or mask spikes. The
anchor price and the forward window are both restricted to real reported
observations.
"""

from __future__ import annotations

import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from features import CATEGORICAL_COLUMNS, build_features, to_model_frame
from harness import TARGET

SPIKE_THRESHOLD = 0.08  # >8% rise
SPIKE_WINDOW_DAYS = 14
# Probability cutoff for firing an alert. 0.30 is the empirical max-F1 point
# on the walk-forward sweep (see models/spike_results.csv); it also gives the
# best episode recall / lead time among the points at or above peak F1.
# Lower it toward 0.20 for a recall-first posture (89.8% -> catches more, at
# ~45% precision); raise it toward 0.50 to cut false alarms.
DECISION_THRESHOLD = 0.30

SPIKE_PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    verbosity=-1,
)


def add_spike_labels(
    frame: pl.DataFrame,
    threshold: float = SPIKE_THRESHOLD,
    window_days: int = SPIKE_WINDOW_DAYS,
) -> pl.DataFrame:
    """Adds `spike` (bool) and `spike_day` (date of first threshold crossing,
    null if none) per (commodity, centre) row.

    Both the anchor price and the forward window come from non-imputed rows
    only. Rows whose own price is imputed, or that have no real observation
    anywhere in the forward window, get a null label and should be dropped
    by the caller -- they are genuinely unlabellable, not negatives.

    Implemented as `window_days` date-shifted self-joins (one per day
    offset) rather than a positional rolling max: series have real
    reporting gaps, so "the next 14 rows" is not "the next 14 days".
    """
    real = frame.filter(~pl.col("is_imputed")).select(
        ["commodity", "centre", "date", TARGET]
    )

    anchors = real.rename({TARGET: "anchor_price"})
    best_ratio = None
    first_cross = None

    for offset in range(1, window_days + 1):
        shifted = real.select(
            "commodity",
            "centre",
            (pl.col("date") - pl.duration(days=offset)).alias("date"),
            pl.col(TARGET).alias("future_price"),
        )
        joined = anchors.join(shifted, on=["commodity", "centre", "date"], how="left")
        ratio = (pl.col("future_price") / pl.col("anchor_price") - 1).alias(f"r{offset}")
        anchors = joined.with_columns(ratio).drop("future_price")

        crossed = pl.col(f"r{offset}") > threshold
        cross_day = (
            pl.when(crossed)
            .then(pl.col("date") + pl.duration(days=offset))
            .otherwise(None)
        )
        first_cross = cross_day if first_cross is None else pl.coalesce([first_cross, cross_day])
        best_ratio = (
            pl.col(f"r{offset}")
            if best_ratio is None
            else pl.max_horizontal([best_ratio, pl.col(f"r{offset}")])
        )

    ratio_cols = [f"r{o}" for o in range(1, window_days + 1)]
    labelled = anchors.with_columns(
        max_fwd_ratio=best_ratio, spike_day=first_cross
    ).with_columns(
        # null (unlabellable) when the whole forward window is missing;
        # otherwise a real True/False
        spike=pl.when(pl.col("max_fwd_ratio").is_null())
        .then(None)
        .otherwise(pl.col("max_fwd_ratio") > threshold)
    )
    labelled = labelled.select(
        ["commodity", "centre", "date", "spike", "spike_day", "max_fwd_ratio"]
    )

    return frame.join(labelled, on=["commodity", "centre", "date"], how="left")


def _fit_spike_model(engineered_labelled: pl.DataFrame) -> lgb.LGBMClassifier | None:
    train = engineered_labelled.drop_nulls(subset=["spike"])
    if train.height == 0 or train["spike"].n_unique() < 2:
        return None  # single-class training window -- nothing to learn
    X = to_model_frame(train)
    y = train["spike"].to_pandas().astype(int)
    model = lgb.LGBMClassifier(**SPIKE_PARAMS)
    model.fit(X, y, categorical_feature=CATEGORICAL_COLUMNS)
    return model


def run_spike_backtest(
    frame: pl.DataFrame,
    origins: list[dt.date],
    threshold: float = SPIKE_THRESHOLD,
    window_days: int = SPIKE_WINDOW_DAYS,
    decision_threshold: float = DECISION_THRESHOLD,
    verbose: bool = True,
) -> pl.DataFrame:
    """Walk-forward: train at each origin, then score every day up to the
    next origin. Returns one row per (commodity, centre, date) scored,
    with the predicted probability, the alert flag, the true label, and
    the realised spike_day (for lead-time computation)."""
    labelled = add_spike_labels(frame, threshold, window_days)

    out = []
    for i, origin in enumerate(origins):
        # Labels are only known once the full forward window has elapsed,
        # so training rows must end window_days before the origin --
        # otherwise the model trains on labels it could not have known yet.
        label_cutoff = origin - dt.timedelta(days=window_days)
        train_raw = frame.filter(pl.col("date") <= origin)
        engineered = build_features(train_raw)

        train_labelled = engineered.join(
            labelled.select(["commodity", "centre", "date", "spike"]),
            on=["commodity", "centre", "date"],
            how="left",
        ).filter(pl.col("date") <= label_cutoff)

        model = _fit_spike_model(train_labelled)
        if model is None:
            continue

        # Score every day from this origin up to the next one.
        next_origin = origins[i + 1] if i + 1 < len(origins) else frame["date"].max()
        score_rows = engineered.filter(pl.col("date") == origin)
        # engineered only covers <= origin; for the days after the origin we
        # need features built from data available at that day, so rebuild on
        # the full frame truncated at each scoring date. Doing that per day
        # would be slow, so build once on everything up to next_origin and
        # slice -- features at date d only use data <= d by construction
        # (lags, rolling windows and EWMA are all backward-looking).
        span = build_features(frame.filter(pl.col("date") <= next_origin)).filter(
            (pl.col("date") > origin) & (pl.col("date") <= next_origin)
        )
        score_rows = pl.concat([score_rows, span], how="vertical")

        if score_rows.height == 0:
            continue
        proba = model.predict_proba(to_model_frame(score_rows))[:, 1]
        scored = score_rows.select(["commodity", "centre", "date"]).with_columns(
            proba=pl.Series(proba), origin=pl.lit(origin)
        )
        out.append(scored)

        if verbose:
            print(f"  [spike] origin {i + 1}/{len(origins)} ({origin})", flush=True)

    if not out:
        return pl.DataFrame()

    # Each origin scores [origin, next_origin], so the boundary day is
    # scored by two consecutive models; keep="last" gives it to the fresher
    # (later-origin) one.
    scored = pl.concat(out, how="vertical").unique(
        subset=["commodity", "centre", "date"], keep="last"
    )
    truth = labelled.select(["commodity", "centre", "date", "spike", "spike_day"])
    return (
        scored.join(truth, on=["commodity", "centre", "date"], how="inner")
        .drop_nulls(subset=["spike"])
        .with_columns(alert=pl.col("proba") >= decision_threshold)
    )


def label_episodes(scored: pl.DataFrame, max_gap_days: int = 7) -> pl.DataFrame:
    """Groups each series' spike days into episodes.

    A sustained run-up produces a crossing on many consecutive days, and
    each of those is a separate `spike_day`. Treating them as separate
    events would make "how early did we catch it" collapse to roughly the
    per-alert number, since most events would be one day long. Spike days
    within `max_gap_days` of the previous one are therefore folded into a
    single episode, and the episode is credited to its FIRST crossing date
    -- the day the price actually began breaching the threshold.
    """
    spike_days = (
        scored.filter(pl.col("spike"))
        .select(["commodity", "centre", "spike_day"])
        .unique()
        .sort(["commodity", "centre", "spike_day"])
    )
    if spike_days.height == 0:
        return spike_days.with_columns(episode_start=pl.lit(None, dtype=pl.Date))

    gap = (
        pl.col("spike_day") - pl.col("spike_day").shift(1).over(["commodity", "centre"])
    ).dt.total_days()
    return spike_days.with_columns(
        episode_start=pl.when(gap.is_null() | (gap > max_gap_days))
        .then(pl.col("spike_day"))
        .otherwise(None)
        .forward_fill()
        .over(["commodity", "centre"])
    )


def evaluate_spikes(scored: pl.DataFrame, max_gap_days: int = 7) -> dict:
    """Precision / recall / F1 plus the headline lead-time numbers.

    Two lead-time views, both in days:
      * per-alert   -- over every correct alert, days from the alert to the
                       crossing it predicted. What an officer experiences on
                       any given day the system warns them.
      * per-episode -- over each distinct spike episode (consecutive
                       crossings folded together, see label_episodes), days
                       from the EARLIEST correct alert to the start of the
                       episode. "How early did we first catch this run-up."
                       This is the headline number.
    """
    tp = scored.filter(pl.col("alert") & pl.col("spike")).height
    fp = scored.filter(pl.col("alert") & ~pl.col("spike")).height
    fn = scored.filter(~pl.col("alert") & pl.col("spike")).height
    tn = scored.filter(~pl.col("alert") & ~pl.col("spike")).height

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else float("nan")
    )

    hits = scored.filter(pl.col("alert") & pl.col("spike")).with_columns(
        lead_days=(pl.col("spike_day") - pl.col("date")).dt.total_days()
    )
    per_alert_median = float(hits["lead_days"].median()) if hits.height else float("nan")

    episodes = label_episodes(scored, max_gap_days)
    # An episode is a (series, episode_start) pair -- counting distinct
    # episode_start dates alone would merge unrelated episodes that happen
    # to begin on the same day in different centres.
    n_episodes = (
        episodes.select(["commodity", "centre", "episode_start"]).unique().height
        if episodes.height
        else 0
    )

    caught = (
        hits.join(episodes, on=["commodity", "centre", "spike_day"], how="inner")
        .group_by(["commodity", "centre", "episode_start"])
        .agg(alert_date=pl.col("date").min())
        .with_columns(
            lead_days=(pl.col("episode_start") - pl.col("alert_date")).dt.total_days()
        )
    )
    per_episode_median = float(caught["lead_days"].median()) if caught.height else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_lead_days_per_alert": per_alert_median,
        "median_lead_days_per_episode": per_episode_median,
        "n_scored": scored.height,
        "n_episodes": n_episodes,
        "n_episodes_caught": caught.height,
        "episode_recall": caught.height / n_episodes if n_episodes else float("nan"),
        "base_rate": float(scored["spike"].mean()),
    }
