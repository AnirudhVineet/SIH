"""
Joins prices.parquet with weather.parquet into the base modelling frame,
keyed on (date, commodity, centre).

Weather is a commodity-level supply signal, not a per-centre one: for each
commodity we average rainfall/temp across that commodity's growing districts
(reference/growing_districts.json) and broadcast the same daily value to every
centre trading that commodity. This mirrors the arrivals design (commodity ->
benchmark mandis) the same way -- see prices.py's header note for why arrivals
itself isn't wired in yet.
"""

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "data" / "processed" / "prices.parquet"
WEATHER_PATH = ROOT / "data" / "processed" / "weather.parquet"
GROWING_DISTRICTS_PATH = ROOT / "reference" / "growing_districts.json"
OUT_PATH = ROOT / "data" / "processed" / "modelling_frame.parquet"


def commodity_weather(weather: pl.DataFrame) -> pl.DataFrame:
    growing = json.loads(GROWING_DISTRICTS_PATH.read_text())
    mapping = pl.DataFrame(
        [
            {"commodity": commodity, "district": d["district"]}
            for commodity, districts in growing.items()
            for d in districts
        ]
    )
    joined = mapping.join(weather, on="district", how="left")
    return joined.group_by(["commodity", "date"]).agg(
        pl.col("rainfall_mm").mean().alias("rainfall_mm"),
        pl.col("temp_max").mean().alias("temp_max"),
        pl.col("temp_min").mean().alias("temp_min"),
    )


def main():
    prices = pl.read_parquet(PRICES_PATH)
    weather = pl.read_parquet(WEATHER_PATH)

    cw = commodity_weather(weather)
    frame = prices.join(cw, on=["commodity", "date"], how="left").sort(
        ["commodity", "centre", "date"]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT_PATH)

    print(f"wrote {OUT_PATH} -- {frame.height} rows, {frame.width} cols")
    print(f"date range: {frame['date'].min()} -> {frame['date'].max()}")
    missing_weather = frame.filter(pl.col("rainfall_mm").is_null()).height
    print(f"rows missing weather join: {missing_weather}")


if __name__ == "__main__":
    main()
