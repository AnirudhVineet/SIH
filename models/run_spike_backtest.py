"""
Runs the spike-classifier walk-forward backtest and writes:

  models/spike_results.csv          -- threshold sweep (precision/recall/F1/lead time)
  models/spike_results_by_commodity.csv -- per-commodity metrics at the chosen threshold
  models/spike_scored.parquet       -- raw daily scored alerts, reused by the dashboard

Rerunnable: `python models/run_spike_backtest.py` from the models/ directory.
"""

from __future__ import annotations

import time

import polars as pl

import harness as h
import spike_classifier as sc

SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

OUT_SWEEP = h.ROOT / "models" / "spike_results.csv"
OUT_BY_COMMODITY = h.ROOT / "models" / "spike_results_by_commodity.csv"
OUT_SCORED = h.ROOT / "models" / "spike_scored.parquet"


def sweep_table(scored: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for thr in SWEEP:
        m = sc.evaluate_spikes(scored.with_columns(alert=pl.col("proba") >= thr))
        rows.append(
            {
                "decision_threshold": thr,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "median_lead_days_per_alert": m["median_lead_days_per_alert"],
                "median_lead_days_per_episode": m["median_lead_days_per_episode"],
                "episode_recall": round(m["episode_recall"], 4),
                "n_alerts": m["tp"] + m["fp"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }
        )
    return pl.DataFrame(rows)


def by_commodity_table(scored: pl.DataFrame, threshold: float) -> pl.DataFrame:
    alerted = scored.with_columns(alert=pl.col("proba") >= threshold)
    rows = []
    for (commodity,), group in alerted.group_by(["commodity"]):
        m = sc.evaluate_spikes(group)
        rows.append(
            {
                "commodity": commodity,
                "decision_threshold": threshold,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "median_lead_days_per_alert": m["median_lead_days_per_alert"],
                "median_lead_days_per_episode": m["median_lead_days_per_episode"],
                "episode_recall": round(m["episode_recall"], 4),
                "n_episodes": m["n_episodes"],
                "base_rate": round(m["base_rate"], 4),
            }
        )
    return pl.DataFrame(rows).sort("commodity")


def main() -> None:
    frame = h.load_frame()
    origins = h.generate_origins(frame)
    print(
        f"spike classifier: >{sc.SPIKE_THRESHOLD:.0%} rise within "
        f"{sc.SPIKE_WINDOW_DAYS}d, {len(origins)} origins, scoring daily"
    )

    t0 = time.time()
    scored = sc.run_spike_backtest(frame, origins, verbose=False)
    print(f"  backtest: {time.time() - t0:.1f}s, {scored.height} scored series-days")

    scored.write_parquet(OUT_SCORED)

    sweep = sweep_table(scored)
    sweep.write_csv(OUT_SWEEP)
    by_com = by_commodity_table(scored, sc.DECISION_THRESHOLD)
    by_com.write_csv(OUT_BY_COMMODITY)

    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(f"\nthreshold sweep -> {OUT_SWEEP}")
        print(sweep)
        print(f"\nper-commodity @ threshold {sc.DECISION_THRESHOLD} -> {OUT_BY_COMMODITY}")
        print(by_com)

    chosen = sc.evaluate_spikes(scored)
    print(
        f"\nHEADLINE @ threshold {sc.DECISION_THRESHOLD}: "
        f"median lead time {chosen['median_lead_days_per_episode']:.0f} days per spike episode "
        f"({chosen['median_lead_days_per_alert']:.0f} days per alert); "
        f"precision {chosen['precision']:.1%}, recall {chosen['recall']:.1%}, "
        f"catching {chosen['n_episodes_caught']}/{chosen['n_episodes']} episodes "
        f"({chosen['episode_recall']:.1%})"
    )


if __name__ == "__main__":
    main()
