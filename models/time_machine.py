"""
Time machine: replay the Aug-Dec 2023 onion price crisis and measure how much
warning the system would actually have given.

Retrospective validation, not a backtest metric: it takes the spike classifier
that already exists and asks the question a judge or an officer asks -- "would
this have helped, on the event everyone remembers?"

**Strictly no hindsight.** The model is retrained at each walk-forward origin
on data up to that origin only, and scored on the days that follow. Nothing
from after an alert date is available to the model that produced it. That is
the whole point: a replay that leaked would prove nothing.

Run: `python models/time_machine.py`  ->  models/time_machine_onion_2023.csv
"""

from __future__ import annotations

import datetime as dt

import polars as pl

import harness as h
import spike_classifier as sc

COMMODITY = "onion"
CRISIS_START = dt.date(2023, 8, 1)
CRISIS_END = dt.date(2023, 12, 31)

# Replay origins: retrain every 30 days across the run-up and the crisis, with
# a lead-in from earlier in 2023 so the first alerts are not cold-started.
REPLAY_FROM = dt.date(2023, 6, 1)
STEP_DAYS = 30

OUT_PATH = h.ROOT / "models" / "time_machine_onion_2023.csv"


def replay_origins() -> list[dt.date]:
    origins, d = [], REPLAY_FROM
    while d <= CRISIS_END:
        origins.append(d)
        d += dt.timedelta(days=STEP_DAYS)
    return origins


def crisis_peaks(frame: pl.DataFrame) -> pl.DataFrame:
    """Per centre: the pre-crisis baseline, the crisis peak, and when it hit."""
    pre = (
        frame.filter(
            (pl.col("commodity") == COMMODITY)
            & (pl.col("date") >= CRISIS_START - dt.timedelta(days=60))
            & (pl.col("date") < CRISIS_START)
            & (~pl.col("is_imputed"))
        )
        .group_by("centre")
        .agg(baseline=pl.col("wholesale_price").median())
    )
    during = frame.filter(
        (pl.col("commodity") == COMMODITY)
        & (pl.col("date") >= CRISIS_START)
        & (pl.col("date") <= CRISIS_END)
        & (~pl.col("is_imputed"))
    )
    peak = (
        during.sort("wholesale_price", descending=True)
        .group_by("centre")
        .first()
        .select(["centre", pl.col("wholesale_price").alias("peak"), pl.col("date").alias("peak_date")])
    )
    return pre.join(peak, on="centre", how="inner").with_columns(
        peak_rise_pct=(pl.col("peak") / pl.col("baseline") - 1) * 100
    )


def main() -> None:
    frame = h.load_frame()
    origins = replay_origins()
    print(
        f"replaying {COMMODITY} {CRISIS_START} -> {CRISIS_END} "
        f"across {len(origins)} retrain origins (no hindsight)"
    )

    scored = sc.run_spike_backtest(frame, origins, verbose=False)
    scored = scored.filter(
        (pl.col("commodity") == COMMODITY)
        & (pl.col("date") >= CRISIS_START)
        & (pl.col("date") <= CRISIS_END)
    )
    if scored.height == 0:
        print("no scored rows in the crisis window")
        return

    metrics = sc.evaluate_spikes(scored)
    peaks = crisis_peaks(frame)

    # First correct alert per centre, and how far ahead of the episode it fired.
    episodes = sc.label_episodes(scored)
    hits = (
        scored.filter(pl.col("alert") & pl.col("spike"))
        .join(episodes, on=["commodity", "centre", "spike_day"], how="inner")
        .group_by(["centre", "episode_start"])
        .agg(first_alert=pl.col("date").min())
        .with_columns(
            lead_days=(pl.col("episode_start") - pl.col("first_alert")).dt.total_days()
        )
    )
    first_per_centre = (
        hits.sort("episode_start").group_by("centre").first().sort("centre")
    )

    table = (
        peaks.join(first_per_centre, on="centre", how="left")
        .with_columns(
            # Two different lead times, both worth stating:
            #  lead_days     -- alert to the first 8% breach. The classifier's
            #                   own target, and the strict metric.
            #  days_to_peak  -- alert to the crisis peak. The policy-relevant
            #                   one: how much runway an officer actually had
            #                   before the worst of it.
            days_to_peak=(pl.col("peak_date") - pl.col("first_alert")).dt.total_days()
        )
        .select(
            pl.col("centre").alias("centre"),
            pl.col("baseline").round(0),
            pl.col("peak").round(0),
            pl.col("peak_rise_pct").round(1),
            pl.col("peak_date"),
            pl.col("episode_start").alias("first_spike_day"),
            pl.col("first_alert"),
            pl.col("lead_days"),
            pl.col("days_to_peak"),
        )
        .sort("peak_rise_pct", descending=True)
    )
    table.write_csv(OUT_PATH)

    # Price path through the crisis, for the dashboard chart.
    (
        frame.filter(
            (pl.col("commodity") == COMMODITY)
            & (pl.col("date") >= CRISIS_START - dt.timedelta(days=60))
            & (pl.col("date") <= CRISIS_END)
            & (~pl.col("is_imputed"))
        )
        .select(["date", "centre", "wholesale_price"])
        .sort(["centre", "date"])
        .write_parquet(h.ROOT / "app" / "data" / "crisis_2023_onion.parquet")
    )

    with pl.Config(tbl_rows=20, tbl_cols=-1, tbl_width_chars=200):
        print(table)

    caught = table.filter(pl.col("first_alert").is_not_null())
    print(f"\nwrote {OUT_PATH}")
    print(f"centres alerted during the crisis: {caught.height}/{table.height}")
    if caught.height:
        ahead = caught.filter(pl.col("lead_days") >= 0)
        print(
            f"  alerted at or before the first 8% breach: {ahead.height}/{caught.height}"
            f" (median lead {caught['lead_days'].median():.0f} d, "
            f"max {caught['lead_days'].max():.0f} d)"
        )
        print(
            f"  runway before the crisis PEAK: median "
            f"{caught['days_to_peak'].median():.0f} days "
            f"(min {caught['days_to_peak'].min():.0f}, "
            f"max {caught['days_to_peak'].max():.0f})"
        )
    print(
        f"crisis-window precision {metrics['precision']:.0%}, "
        f"recall {metrics['recall']:.0%}, "
        f"episode recall {metrics['episode_recall']:.0%}"
    )


if __name__ == "__main__":
    main()
