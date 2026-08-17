"""
Lightweight integration smoke tests for the LightGBM-family models. Unlike
test_harness/test_baselines/test_features/test_sarimax_model (pure unit
tests on tiny synthetic frames), these train real (tiny) LightGBM models --
building a fully synthetic frame with every TRUSTED_COLUMNS populated
realistically is more boilerplate than it's worth, and a small slice of the
real modelling frame exercises the real column set for free. Kept small (2
commodities, a handful of centres, ~2 years) so this stays fast.
"""

import datetime as dt

import polars as pl
import pytest

import harness as h
import lgbm_model as lm
import quantile_lgbm as q


@pytest.fixture(scope="module")
def small_frame() -> pl.DataFrame:
    frame = h.load_frame()
    centres = frame.select("centre").unique().sort("centre")["centre"].to_list()[:3]
    return frame.filter(
        pl.col("commodity").is_in(["tur", "onion"])
        & pl.col("centre").is_in(centres)
        & (pl.col("date") >= dt.date(2023, 1, 1))
    )


@pytest.fixture(autouse=True)
def clear_caches():
    lm._cache.clear()
    q._cache.clear()
    yield
    lm._cache.clear()
    q._cache.clear()


def _pick_origin(frame: pl.DataFrame) -> dt.date:
    return frame["date"].max() - dt.timedelta(days=20)


def test_lgbm_predict_returns_finite_predictions_for_every_series(small_frame):
    origin = _pick_origin(small_frame)
    train = small_frame.filter(pl.col("date") <= origin)
    out = lm.lgbm_predict(train, origin, horizon=7)

    assert out.height > 0
    assert set(out.columns) == {"commodity", "centre", "pred"}
    assert out["pred"].is_finite().all()
    assert (out["pred"] > 0).all()  # prices can't be negative


def test_lgbm_predict_reuses_cached_model_across_horizons(small_frame):
    origin = _pick_origin(small_frame)
    train = small_frame.filter(pl.col("date") <= origin)

    lm.lgbm_predict(train, origin, horizon=7)
    assert origin in lm._cache
    assert 7 in lm._cache[origin]["models"]
    assert 14 not in lm._cache[origin]["models"]

    engineered_before = lm._cache[origin]["engineered"]
    lm.lgbm_predict(train, origin, horizon=14)
    # same engineered-features object reused, not recomputed, for the 2nd horizon
    assert lm._cache[origin]["engineered"] is engineered_before
    assert 14 in lm._cache[origin]["models"]


def test_quantile_predict_band_is_monotonic_p10_le_p50_le_p90(small_frame):
    origin = _pick_origin(small_frame)
    train = small_frame.filter(pl.col("date") <= origin)
    band = q.quantile_predict_band(train, origin, horizon=7)

    assert band.height > 0
    assert (band["p10"] <= band["p50"] + 1e-6).all()
    assert (band["p50"] <= band["p90"] + 1e-6).all()


def test_quantile_median_predict_matches_band_p50(small_frame):
    origin = _pick_origin(small_frame)
    train = small_frame.filter(pl.col("date") <= origin)

    band = q.quantile_predict_band(train, origin, horizon=7)
    median_only = q.quantile_median_predict(train, origin, horizon=7)

    joined = median_only.join(band, on=["commodity", "centre"])
    assert (joined["pred"] == joined["p50"]).all()
