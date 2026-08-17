import datetime as dt

import polars as pl
import pytest

import baselines as b


def _frame(rows: list[dict]) -> pl.DataFrame:
    for r in rows:
        r.setdefault("is_imputed", False)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_naive_predict_returns_last_known_price():
    origin = dt.date(2024, 6, 1)
    frame = _frame(
        [
            {"commodity": "tur", "centre": "X", "date": dt.date(2024, 5, 31), "wholesale_price": 90.0},
            {"commodity": "tur", "centre": "X", "date": origin, "wholesale_price": 100.0},
        ]
    )
    out = b.naive_predict(frame, origin, horizon=7)
    assert out.height == 1
    assert out["pred"][0] == pytest.approx(100.0)


def test_naive_predict_skips_series_missing_at_origin():
    origin = dt.date(2024, 6, 1)
    frame = _frame(
        [{"commodity": "tur", "centre": "X", "date": dt.date(2024, 5, 20), "wholesale_price": 90.0}]
    )
    out = b.naive_predict(frame, origin, horizon=7)
    assert out.height == 0


def test_seasonal_naive_uses_price_365_days_before_target():
    origin = dt.date(2024, 6, 1)
    horizon = 7
    target_date = origin + dt.timedelta(days=horizon)  # 2024-06-08
    season_date = target_date - dt.timedelta(days=365)  # 2023-06-09 (2024 is a leap year)
    frame = _frame(
        [
            {"commodity": "tur", "centre": "X", "date": season_date, "wholesale_price": 77.0},
            {"commodity": "tur", "centre": "X", "date": origin, "wholesale_price": 100.0},
        ]
    )
    out = b.seasonal_naive_predict(frame, origin, horizon)
    assert out["pred"][0] == pytest.approx(77.0)


def test_seasonal_naive_falls_back_to_flat_when_season_date_missing():
    origin = dt.date(2024, 6, 1)
    frame = _frame(
        [{"commodity": "tur", "centre": "X", "date": origin, "wholesale_price": 100.0}]
    )
    out = b.seasonal_naive_predict(frame, origin, horizon=7)
    assert out["pred"][0] == pytest.approx(100.0)
