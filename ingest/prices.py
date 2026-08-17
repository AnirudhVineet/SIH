"""
Builds prices.parquet from the Kaggle mirror of Agmarknet mandi price data
("khandelwalmanas/daily-commodity-prices-india", 2015-2026, per-year parquet files).

IMPORTANT SCOPE NOTE: no working DoCA retail-price-by-centre source was found
(CEDA mirror is DNS-dead, CEDA's own API is Agmarknet-only + JWT-gated, DoCA's
fcainfoweb portal is CAPTCHA-gated, data.gov.in's live resource has no history).
This script produces WHOLESALE prices only, sourced from real Agmarknet mandi
data. retail_price is written as null for every row. "centre" is therefore
defined pragmatically as the highest-volume real market for that commodity in
each chosen state (reference/centres.json), not a true DoCA retail centre.

Rerunnable: raw per-year files are cached to data/raw/prices/<year>.parquet and
reused unless --refresh is passed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import polars as pl
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "prices"
OUT_PATH = ROOT / "data" / "processed" / "prices.parquet"
CENTRES_PATH = ROOT / "reference" / "centres.json"
SECRETS_PATH = ROOT / "reference" / "secrets.json"

YEARS = range(2015, 2027)
DATASET_REF = "khandelwalmanas/daily-commodity-prices-india"
API_BASE = "https://www.kaggle.com/api/v1"

COMMODITY_MAP = {
    "tur": "Arhar (Tur/Red Gram)(Whole)",
    "onion": "Onion",
    "potato": "Potato",
}


def kaggle_token() -> str:
    return json.loads(SECRETS_PATH.read_text())["kaggle_token"]


def download_year(year: int, token: str, refresh: bool) -> Path:
    cache_path = RAW_DIR / f"{year}.parquet"
    if cache_path.exists() and not refresh:
        return cache_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{API_BASE}/datasets/download/{DATASET_REF}/parquet%2F{year}.parquet"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return cache_path


def load_raw(years, token, refresh) -> pl.DataFrame:
    frames = []
    for year in years:
        path = download_year(year, token, refresh)
        frames.append(pl.read_parquet(path))
        print(f"  {year}: cached ({path.stat().st_size / 1e6:.1f} MB)")
    return pl.concat(frames, how="vertical_relaxed")


def pick_benchmark_markets(df: pl.DataFrame, centres: list[dict]) -> pl.DataFrame:
    """For each (state, commodity) pick the single real market with the most
    reported rows across all years -- our best available proxy for a
    consistently-reporting centre."""
    states = [c["state"] for c in centres]
    counts = (
        df.filter(pl.col("State").is_in(states))
        .group_by(["State", "commodity", "Market"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    benchmark = counts.group_by(["State", "commodity"]).first()
    return benchmark.select(
        pl.col("State").alias("state"),
        pl.col("commodity"),
        pl.col("Market").alias("benchmark_market"),
        pl.col("n").alias("benchmark_rows"),
    )


def fill_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """ffill <=3 days -> same-centre trailing 7d median -> drop remaining nulls, flagged."""
    lo, hi = df["date"].min(), df["date"].max()
    all_dates = pl.DataFrame({"date": pl.date_range(lo, hi, interval="1d", eager=True)})
    full_grid = df.select(["commodity", "centre"]).unique().join(all_dates, how="cross")

    out = full_grid.join(df, on=["date", "commodity", "centre"], how="left").sort(
        ["commodity", "centre", "date"]
    )

    out = out.with_columns(
        pl.col("wholesale_price").is_null().alias("was_missing")
    )
    out = out.with_columns(
        pl.col("wholesale_price")
        .forward_fill(limit=3)
        .over(["commodity", "centre"])
        .alias("wholesale_price")
    )
    trailing_median = (
        pl.col("wholesale_price")
        .rolling_median(window_size=7, min_periods=1)
        .over(["commodity", "centre"])
    )
    out = out.with_columns(
        pl.when(pl.col("wholesale_price").is_null())
        .then(trailing_median)
        .otherwise(pl.col("wholesale_price"))
        .alias("wholesale_price")
    )
    out = out.with_columns(
        (pl.col("was_missing") & pl.col("wholesale_price").is_not_null()).alias("is_imputed")
    )

    dropped = out.filter(pl.col("wholesale_price").is_null()).height
    out = out.filter(pl.col("wholesale_price").is_not_null())
    print(f"  dropped {dropped} centre-days with no fillable price (ffill+7d-median exhausted)")

    return out.select(
        "date", "commodity", "centre", "state", "retail_price", "wholesale_price", "is_imputed"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    centres = json.loads(CENTRES_PATH.read_text())
    token = kaggle_token()

    print("downloading/caching Kaggle price years...")
    raw = load_raw(YEARS, token, args.refresh)

    inv_map = {v: k for k, v in COMMODITY_MAP.items()}
    raw = raw.filter(pl.col("Commodity").is_in(list(inv_map.keys()))).with_columns(
        pl.col("Commodity").replace(inv_map).alias("commodity"),
        pl.col("Arrival_Date").str.to_date("%Y-%m-%d").alias("date"),
    )

    print("picking benchmark market per state x commodity...")
    benchmark = pick_benchmark_markets(raw, centres)
    print(benchmark.sort(["state", "commodity"]))

    centres_df = pl.DataFrame(centres)
    bench_with_centre = benchmark.join(centres_df, on="state", how="inner")

    priced = raw.join(
        bench_with_centre.select("state", "commodity", "benchmark_market", "centre"),
        left_on=["State", "commodity", "Market"],
        right_on=["state", "commodity", "benchmark_market"],
        how="inner",
    ).select(
        "date",
        "commodity",
        "centre",
        pl.col("State").alias("state"),
        pl.lit(None, dtype=pl.Float64).alias("retail_price"),
        pl.col("Modal_Price").cast(pl.Float64).alias("wholesale_price"),
    )
    # a market can report >1 row/day (multiple varieties/grades) -- collapse to daily median
    priced = priced.group_by(["date", "commodity", "centre", "state"]).agg(
        pl.col("retail_price").first(),
        pl.col("wholesale_price").median(),
    )

    print("filling gaps...")
    final = fill_gaps(priced).sort(["commodity", "centre", "date"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(OUT_PATH)

    print(f"\nwrote {OUT_PATH} -- {final.height} rows")
    print(f"date range: {final['date'].min()} -> {final['date'].max()}")
    print(f"commodities: {sorted(final['commodity'].unique().to_list())}")
    print(f"centres: {sorted(final['centre'].unique().to_list())}")
    print(f"imputed rows: {final['is_imputed'].sum()} ({final['is_imputed'].mean() * 100:.1f}%)")
    print(f"retail_price non-null: {final['retail_price'].is_not_null().sum()} (expected 0)")


if __name__ == "__main__":
    main()
