import datetime as dt

import numpy as np
import polars as pl
import pytest

import harness as h


def _frame(rows: list[dict]) -> pl.DataFrame:
    """Builds a minimal (commodity, centre, date, wholesale_price,
    is_imputed) frame from row dicts, defaulting is_imputed=False."""
    for r in rows:
        r.setdefault("is_imputed", False)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


# ---------------------------------------------------------------- mape/rmse


def test_mape_matches_hand_computed_value():
    actual = np.array([100.0, 200.0, 150.0, 50.0])
    pred = np.array([110.0, 190.0, 140.0, 55.0])
    # |10/100| + |10/200| + |10/150| + |5/50| = .10+.05+.0667+.10 = .3167 -> /4 *100
    assert h.mape(actual, pred) == pytest.approx(7.9166, abs=1e-3)


def test_mape_ignores_zero_actuals():
    actual = np.array([0.0, 100.0])
    pred = np.array([999.0, 110.0])
    assert h.mape(actual, pred) == pytest.approx(10.0)


def test_rmse_matches_hand_computed_value():
    actual = np.array([100.0, 200.0])
    pred = np.array([110.0, 190.0])
    assert h.rmse(actual, pred) == pytest.approx(10.0)


# ------------------------------------------------------------ generate_origins


def test_generate_origins_spacing_and_bounds():
    frame = _frame(
        [{"commodity": "tur", "centre": "X", "date": dt.date(2025, 12, 31), "wholesale_price": 100.0}]
    )
    origins = h.generate_origins(frame, step_days=30, backtest_years=1, horizons=[7, 14])

    max_horizon = 14
    last_allowed = dt.date(2025, 12, 31) - dt.timedelta(days=max_horizon)
    assert all(o <= last_allowed for o in origins)
    # the last origin is within one full step of the boundary (it's the
    # last point of a fixed-size stride that lands at or before the cutoff,
    # not necessarily exactly on it)
    assert (last_allowed - origins[-1]).days < 30
    # strictly increasing, spaced exactly step_days apart
    diffs = [(b - a).days for a, b in zip(origins, origins[1:])]
    assert diffs == [30] * (len(origins) - 1)


# ------------------------------------------------------------- make_supervised


def test_make_supervised_matches_actual_future_price_when_contiguous():
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, d), "wholesale_price": 100.0 + d}
        for d in range(1, 11)
    ]
    frame = _frame(rows)
    sup = h.make_supervised(frame, horizon=3)

    row = sup.filter(pl.col("date") == dt.date(2024, 1, 1))
    # price on 2024-01-04 is 100+4=104
    assert row["target"][0] == pytest.approx(104.0)


def test_make_supervised_target_is_null_across_a_gap():
    # gap: no row for 2024-01-04..2024-01-09, next row is 2024-01-10
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 1), "wholesale_price": 100.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 2), "wholesale_price": 101.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 3), "wholesale_price": 102.0},
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 10), "wholesale_price": 999.0},
    ]
    frame = _frame(rows)
    sup = h.make_supervised(frame, horizon=7)

    # 2024-01-03 + 7 days = 2024-01-10, which DOES exist -> real target
    row_hit = sup.filter(pl.col("date") == dt.date(2024, 1, 3))
    assert row_hit["target"][0] == pytest.approx(999.0)

    # 2024-01-01 + 7 days = 2024-01-08, which does NOT exist -> null, not a
    # positional guess at some other row's value
    row_gap = sup.filter(pl.col("date") == dt.date(2024, 1, 1))
    assert row_gap["target"][0] is None


def test_make_supervised_does_not_leak_across_commodities_or_centres():
    rows = [
        {"commodity": "tur", "centre": "X", "date": dt.date(2024, 1, 1), "wholesale_price": 100.0},
        {"commodity": "onion", "centre": "X", "date": dt.date(2024, 1, 8), "wholesale_price": 500.0},
        {"commodity": "tur", "centre": "Y", "date": dt.date(2024, 1, 8), "wholesale_price": 700.0},
    ]
    frame = _frame(rows)
    sup = h.make_supervised(frame, horizon=7)

    row = sup.filter((pl.col("commodity") == "tur") & (pl.col("centre") == "X"))
    # neither the onion/X row nor the tur/Y row at the same date should be picked up
    assert row["target"][0] is None


# ----------------------------------------------------------------- summarize


def test_summarize_matches_scalar_mape_rmse_helpers():
    results = pl.DataFrame(
        {
            "model": ["m"] * 4,
            "commodity": ["tur"] * 4,
            "horizon": [7] * 4,
            "actual": [100.0, 200.0, 150.0, 50.0],
            "pred": [110.0, 190.0, 140.0, 55.0],
        }
    )
    out = h.summarize(results)
    expected_mape = h.mape(results["actual"].to_numpy(), results["pred"].to_numpy())
    expected_rmse = h.rmse(results["actual"].to_numpy(), results["pred"].to_numpy())
    assert out["mape"][0] == pytest.approx(expected_mape)
    assert out["rmse"][0] == pytest.approx(expected_rmse)
    assert out["n"][0] == 4


# -------------------------------------------------------------- eval_at_origin


def test_eval_at_origin_excludes_imputed_actuals():
    origin = dt.date(2024, 1, 1)
    target_date = dt.date(2024, 1, 8)
    rows = [
        {"commodity": "tur", "centre": "X", "date": origin, "wholesale_price": 100.0},
        {
            "commodity": "tur",
            "centre": "X",
            "date": target_date,
            "wholesale_price": 999.0,
            "is_imputed": True,
        },
    ]
    frame = _frame(rows)

    def flat_predict(train, o, horizon):
        return train.filter(pl.col("date") == o).select(
            ["commodity", "centre", pl.col("wholesale_price").alias("pred")]
        )

    out = h.eval_at_origin(frame, origin, 7, flat_predict)
    assert out.height == 0  # the only actual at the target date is imputed -> dropped


def test_eval_at_origin_joins_prediction_to_real_actual():
    origin = dt.date(2024, 1, 1)
    target_date = dt.date(2024, 1, 8)
    rows = [
        {"commodity": "tur", "centre": "X", "date": origin, "wholesale_price": 100.0},
        {"commodity": "tur", "centre": "X", "date": target_date, "wholesale_price": 105.0},
    ]
    frame = _frame(rows)

    def flat_predict(train, o, horizon):
        return train.filter(pl.col("date") == o).select(
            ["commodity", "centre", pl.col("wholesale_price").alias("pred")]
        )

    out = h.eval_at_origin(frame, origin, 7, flat_predict)
    assert out.height == 1
    assert out["actual"][0] == pytest.approx(105.0)
    assert out["pred"][0] == pytest.approx(100.0)
