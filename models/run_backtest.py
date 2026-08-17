"""
Runs the full Phase 2 walk-forward backtest end to end and writes
models/backtest_results.csv.

Rerunnable: `python models/run_backtest.py` (run from anywhere -- paths are
resolved relative to the repo root, same convention as ingest/*.py).

Columns: commodity, horizon_days, n, then MAPE/RMSE for every model
(naive, seasonal_naive, sarimax, lgbm, lgbm_q50), then improvement_pct
(LightGBM's relative MAPE change vs. SARIMAX -- negative means LightGBM
has lower error, matching CLAUDE.md's example table's sign convention).
"""

from __future__ import annotations

import time

import polars as pl

import baselines
import harness as h
import lgbm_model
import quantile_lgbm
import sarimax_model

OUT_PATH = h.ROOT / "models" / "backtest_results.csv"

MODELS = [
    ("naive", baselines.naive_predict),
    ("seasonal_naive", baselines.seasonal_naive_predict),
    ("sarimax", sarimax_model.sarimax_predict),
    ("lgbm", lgbm_model.lgbm_predict),
    ("lgbm_q50", quantile_lgbm.quantile_median_predict),
]


def build_wide_table(summary: pl.DataFrame) -> pl.DataFrame:
    """Long (model, commodity, horizon, mape, rmse, n) -> one row per
    (commodity, horizon) with every model's mape/rmse as its own columns."""
    base = summary.select(["commodity", "horizon"]).unique().sort(["commodity", "horizon"])
    for name, _ in MODELS:
        per_model = summary.filter(pl.col("model") == name).select(
            "commodity",
            "horizon",
            pl.col("mape").round(2).alias(f"{name}_mape"),
            pl.col("rmse").round(1).alias(f"{name}_rmse"),
            pl.col("n").alias(f"{name}_n"),
        )
        base = base.join(per_model, on=["commodity", "horizon"], how="left")

    return base.with_columns(
        (((pl.col("lgbm_mape") - pl.col("sarimax_mape")) / pl.col("sarimax_mape")) * 100)
        .round(1)
        .alias("lgbm_improvement_vs_sarimax_pct")
    ).rename({"horizon": "horizon_days"})


def main() -> None:
    frame = h.load_frame()
    origins = h.generate_origins(frame)
    print(f"walk-forward backtest: {len(origins)} origins x horizons={h.HORIZONS}")

    all_results = []
    for name, fn in MODELS:
        t0 = time.time()
        res = h.run_backtest(frame, fn, name, origins=origins, verbose=False)
        print(f"  {name}: {time.time() - t0:.1f}s, {res.height} scored rows")
        all_results.append(res)

    combined = pl.concat(all_results)
    summary = h.summarize(combined)
    wide = build_wide_table(summary)

    wide.write_csv(OUT_PATH)
    print(f"\nwrote {OUT_PATH}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(wide)

    macro = summary.group_by("model").agg(macro_mape=pl.col("mape").mean()).sort("macro_mape")
    print("\nmacro-average MAPE across all (commodity, horizon) cells:")
    print(macro)

    lgbm_mape = macro.filter(pl.col("model") == "lgbm")["macro_mape"][0]
    sarimax_mape = macro.filter(pl.col("model") == "sarimax")["macro_mape"][0]
    beats = lgbm_mape < sarimax_mape
    verdict = "BEATS" if beats else "DOES NOT BEAT"
    print(
        f"\nLightGBM macro-avg MAPE {lgbm_mape:.2f}% vs SARIMAX {sarimax_mape:.2f}% "
        f"-> LightGBM {verdict} SARIMAX"
    )


if __name__ == "__main__":
    main()
