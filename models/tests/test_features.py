import datetime as dt

import polars as pl
import pytest

import features as feat


def _train_frame(rows: list[dict]) -> pl.DataFrame:
    """Minimal frame with every TRUSTED_COLUMNS present (build_features
    selects them), all zeroed except date/price so tests can focus on the
    engineered columns."""
    base = {c: 0.0 for c in feat.TRUSTED_COLUMNS}
    out = []
    for r in rows:
        row = dict(base)
        row.update(r)
        out.append(row)
    return pl.DataFrame(out).with_columns(pl.col("date").cast(pl.Date))


def test_lag_is_null_immediately_after_a_reporting_gap():
    # a real gap: no rows between 2024-01-03 and 2024-02-01
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 1), "wholesale_price": 100.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 2), "wholesale_price": 101.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 3), "wholesale_price": 102.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 2, 1), "wholesale_price": 500.0},
    ]
    eng = feat.build_features(_train_frame(rows))
    row = eng.filter(pl.col("date") == dt.date(2024, 2, 1))

    # a naive row-positional shift(1) would return 102.0 (the previous row);
    # the calendar-correct answer is null, since 2024-01-31 has no data
    assert row["price_lag_1"][0] is None
    assert row["price_lag_7"][0] is None


def test_lag_is_correct_value_when_history_is_contiguous():
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, d), "wholesale_price": 100.0 + d}
        for d in range(1, 10)
    ]
    eng = feat.build_features(_train_frame(rows))
    row = eng.filter(pl.col("date") == dt.date(2024, 1, 9))
    assert row["price_lag_1"][0] == pytest.approx(108.0)  # price on 2024-01-08
    assert row["price_lag_7"][0] == pytest.approx(102.0)  # price on 2024-01-02


def test_rolling_mean_only_counts_points_within_the_calendar_window():
    # one isolated point, then a gap, then another isolated point 60 days
    # later -- a 7-day rolling mean at the second point must be just that
    # point's own value, not an average blended across the gap
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 1), "wholesale_price": 100.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 3, 1), "wholesale_price": 300.0},
    ]
    eng = feat.build_features(_train_frame(rows))
    row = eng.filter(pl.col("date") == dt.date(2024, 3, 1))
    assert row["price_roll_mean_7"][0] == pytest.approx(300.0)


def test_momentum_30_matches_price_over_lag_30_minus_one():
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 1), "wholesale_price": 100.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 31), "wholesale_price": 120.0},
    ]
    eng = feat.build_features(_train_frame(rows))
    row = eng.filter(pl.col("date") == dt.date(2024, 1, 31))
    assert row["momentum_30"][0] == pytest.approx(120.0 / 100.0 - 1)


def test_series_are_kept_independent():
    # same date, two different centres -- lag features must not cross-pollinate
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 1), "wholesale_price": 100.0},
        {"commodity": "tur", "centre": "Y", "date": dt.date(2024, 1, 1), "wholesale_price": 999.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 2), "wholesale_price": 105.0},
    ]
    eng = feat.build_features(_train_frame(rows))
    row = eng.filter((pl.col("centre") == "X") & (pl.col("date") == dt.date(2024, 1, 2)))
    assert row["price_lag_1"][0] == pytest.approx(100.0)


def test_trusted_columns_pass_through_unchanged():
    rows = [
        {
            "commodity": "tur",
            "centre": "X",
            "date": dt.date(2024, 1, 1),
            "wholesale_price": 100.0,
            "rainfall_mm": 12.5,
            "festival_diwali": True,
        }
    ]
    eng = feat.build_features(_train_frame(rows))
    assert eng["rainfall_mm"][0] == pytest.approx(12.5)
    assert eng["festival_diwali"][0] is True
