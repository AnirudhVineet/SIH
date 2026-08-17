import datetime as dt
import random

import pandas as pd
import polars as pl
import pytest

import sarimax_model as sm


# ------------------------------------------------------- is_plausible_forecast


def test_plausible_forecast_accepted():
    forecast = pd.Series([105.0, 108.0, 110.0])
    assert sm.is_plausible_forecast(forecast, recent_level=100.0) is True


def test_negative_forecast_rejected():
    forecast = pd.Series([105.0, -3.0])
    assert sm.is_plausible_forecast(forecast, recent_level=100.0) is False


def test_forecast_beyond_max_multiple_rejected():
    # the real failure mode this guard was written for: an explosive AR
    # root blowing a forecast up to many multiples of the recent price
    forecast = pd.Series([105.0, 900.0])
    assert sm.is_plausible_forecast(forecast, recent_level=100.0, max_multiple=8.0) is False


def test_forecast_exactly_at_max_multiple_accepted():
    forecast = pd.Series([800.0])
    assert sm.is_plausible_forecast(forecast, recent_level=100.0, max_multiple=8.0) is True


# --------------------------------------------------------------- sarimax_predict


def test_sarimax_predict_skips_series_with_too_little_history():
    origin = dt.date(2024, 6, 1)
    # far fewer than MIN_OBS rows
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 5, 20) + dt.timedelta(days=i), "wholesale_price": 100.0 + i}
        for i in range(5)
    ]
    train = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    out = sm.sarimax_predict(train, origin, horizon=7)
    assert out.height == 0


def test_sarimax_predict_produces_a_forecast_for_a_well_behaved_series():
    origin = dt.date(2024, 6, 1)
    start = dt.date(2023, 1, 1)
    n_days = (origin - start).days + 1
    # gentle upward-drifting series with a little noise -- a perfectly
    # noiseless linear ramp is a degenerate (zero-variance-after-
    # differencing) edge case that made the MLE fit fail to converge
    rng = random.Random(0)
    rows = [
        {
            "commodity": "tur",
            "centre": "well_behaved",
            "date": start + dt.timedelta(days=i),
            "wholesale_price": 100.0 + 0.05 * i + rng.uniform(-0.5, 0.5),
        }
        for i in range(n_days)
    ]
    train = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    out = sm.sarimax_predict(train, origin, horizon=7)

    assert out.height == 1
    assert out["commodity"][0] == "tur"
    pred = out["pred"][0]
    assert pred > 0
    # should be in the right ballpark for a slow drift, not wildly off
    assert 100.0 < pred < 200.0
