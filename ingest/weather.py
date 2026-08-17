"""
Pulls daily rainfall + temperature for each commodity's growing districts
from the Open-Meteo Archive API (no key required).

Growing districts are commodity-level, not centre-level (see reference/growing_districts.json).
This script writes district-level weather only; the commodity<->district mapping is
applied later at join time in features/build.py.

Rerunnable: raw API responses are cached to data/raw/weather/<district>.json and reused
unless --refresh is passed, so a rerun never re-hits the API for a district it already has.
"""

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "weather"
OUT_PATH = ROOT / "data" / "processed" / "weather.parquet"
DISTRICTS_PATH = ROOT / "reference" / "growing_districts.json"

START_DATE = "2015-01-01"
LAG_DAYS = 1  # verified 2026-08-16: archive API had zero lag (data through "today"); 1-day margin kept as a safety buffer only
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_unique_districts() -> list[dict]:
    growing = json.loads(DISTRICTS_PATH.read_text())
    seen = {}
    for districts in growing.values():
        for d in districts:
            seen[d["district"]] = d  # dedupe by district name
    return list(seen.values())


def fetch_district(district: dict, end_date: str, refresh: bool) -> dict:
    cache_path = RAW_DIR / f"{district['district'].lower()}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())

    params = {
        "latitude": district["lat"],
        "longitude": district["lon"],
        "start_date": START_DATE,
        "end_date": end_date,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Kolkata",
    }
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"[retry {attempt + 1}/4 in {wait}s: {e}]", end=" ", flush=True)
            time.sleep(wait)
    else:
        raise last_err

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def payload_to_rows(district_name: str, payload: dict) -> pl.DataFrame:
    daily = payload["daily"]
    return pl.DataFrame(
        {
            "date": daily["time"],
            "district": district_name,
            "rainfall_mm": daily["precipitation_sum"],
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
        }
    ).with_columns(pl.col("date").str.to_date())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    end_date = (date.today() - timedelta(days=LAG_DAYS)).isoformat()
    districts = load_unique_districts()

    frames = []
    for d in districts:
        print(f"  {d['district']} ({d['state']})...", end=" ", flush=True)
        payload = fetch_district(d, end_date, args.refresh)
        frames.append(payload_to_rows(d["district"], payload))
        print("ok")

    out = pl.concat(frames).sort(["district", "date"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH)

    print(f"\nwrote {OUT_PATH} — {out.height} rows, {out['district'].n_unique()} districts")
    print(f"date range: {out['date'].min()} -> {out['date'].max()}")
    null_counts = out.select(pl.col("rainfall_mm", "temp_max", "temp_min").is_null().sum())
    print(f"nulls: {null_counts.to_dicts()[0]}")


if __name__ == "__main__":
    main()
