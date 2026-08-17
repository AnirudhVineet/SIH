"""
Precomputes everything the Phase 3 dashboard renders, from the real models.

The dashboard must not train models on page load (slow, and it would make
deployment fragile), so this script fits the committed Phase 2 models once
at the latest available date and writes plain artifacts the app reads:

  app/data/forecasts.parquet     -- per (commodity, centre, horizon): current
                                    price, conformal P10/P50/P90, % change,
                                    spike probability
  app/data/shap_drivers.parquet  -- per (commodity, centre, horizon): top SHAP
                                    drivers with signed contributions
  app/data/stress.parquet        -- Price Stress Index per (state, commodity)
  app/data/history.parquet       -- trimmed price history for the charts
  app/data/meta.json             -- as-of date, model config, provenance

Rerunnable: `python models/build_dashboard_artifacts.py` from models/.

Every number here comes from a real fitted model on real Agmarknet data.
Backtest accuracy figures shown in the app are read separately, straight
from models/backtest_results.csv.
"""

from __future__ import annotations

import datetime as dt
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
import shap

import harness as h
import lgbm_model as lm
import quantile_lgbm as q
import spike_classifier as sc
from features import CATEGORICAL_COLUMNS, MODEL_COLUMNS, build_features, to_model_frame

sys.path.insert(0, str(h.ROOT / "decide"))
import stress as stress_mod  # noqa: E402
import procurement as procurement_mod  # noqa: E402

OUT_DIR = h.ROOT / "app" / "data"
HISTORY_DAYS = 400  # how much history the commodity chart shows
# Pool of candidate drivers stored per series/horizon (used for the bar chart
# and as the source pool for the sentence's category-balanced pick below).
# Wider than the 3-4 the sentence actually uses so that a real but
# mid-magnitude weather/season signal still has a chance to be selected.
TOP_DRIVERS = 8

# Not every centre reports on the latest date -- reporting is patchy, which
# is the whole reason for the is_imputed flag. Rather than dropping those
# centres from the dashboard, each is scored at its own most recent real
# observation within this lookback, and the app shows how stale that is.
STALENESS_LOOKBACK_DAYS = 45

# Stress index weights and scales now live in decide/stress.py.

# Human-readable phrasing for the driver sentence. Anything not listed
# falls back to a cleaned-up version of the raw column name.
FEATURE_LABELS = {
    "price_lag_1": "yesterday's price",
    "price_lag_7": "last week's price",
    "price_lag_14": "price a fortnight ago",
    "price_lag_30": "price a month ago",
    "price_lag_90": "price a quarter ago",
    "price_roll_mean_7": "7-day average price",
    "price_roll_mean_30": "30-day average price",
    "price_roll_mean_90": "90-day average price",
    "price_roll_std_7": "7-day price volatility",
    "price_roll_std_30": "30-day price volatility",
    "price_roll_std_90": "90-day price volatility",
    "price_ewma_7": "recent price trend (7-day)",
    "price_ewma_30": "recent price trend (30-day)",
    "momentum_30": "30-day price momentum",
    "rainfall_mm": "rainfall in growing districts",
    "rainfall_7d_sum": "7-day cumulative rainfall",
    "rainfall_30d_sum": "30-day cumulative rainfall",
    "rainfall_dev_from_normal": "rainfall vs seasonal normal",
    "temp_max": "daytime temperature in growing districts",
    "temp_min": "night temperature in growing districts",
    "months_since_harvest": "months since last harvest",
    "month": "time of year",
    "doy_sin": "seasonal cycle",
    "doy_cos": "seasonal cycle",
    "dow": "day of week",
    "festival_diwali": "Diwali demand",
    "festival_navratri": "Navratri demand",
    "festival_eid": "Eid demand",
    "festival_onam": "Onam demand",
    "price_tur_same_state": "tur price in the same state",
    "price_onion_same_state": "onion price in the same state",
    "price_potato_same_state": "potato price in the same state",
    "commodity": "commodity",
    "centre": "centre",
}


def human_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def latest_origin(frame: pl.DataFrame) -> dt.date:
    """Most recent date that actually has real (non-imputed) observations --
    the date an officer would be looking at 'today'."""
    return frame.filter(~pl.col("is_imputed"))["date"].max()


def latest_rows_per_series(engineered: pl.DataFrame, origin: dt.date) -> pl.DataFrame:
    """The most recent real observation for each (commodity, centre) within
    STALENESS_LOOKBACK_DAYS of the origin, with how stale it is.

    Scoring every centre at the single global latest date would silently
    drop the ~30% of centres that happened not to report that day. The
    models are trained on everything up to `origin` and applied to each
    centre's latest reading -- correct for a live "as of today" dashboard
    (it is not, and must not be used as, a backtest)."""
    cutoff = origin - dt.timedelta(days=STALENESS_LOOKBACK_DAYS)
    return (
        engineered.filter((pl.col("date") >= cutoff) & (pl.col("date") <= origin))
        .sort("date")
        .group_by(["commodity", "centre"])
        .last()
        .with_columns(staleness_days=(pl.lit(origin) - pl.col("date")).dt.total_days())
    )


def build_forecasts(frame: pl.DataFrame, origin: dt.date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Conformal quantile band + point forecast + SHAP drivers per
    (commodity, centre, horizon), fitted on everything up to `origin`."""
    train = frame.filter(pl.col("date") <= origin)
    engineered = lm.engineered_for(train, origin)
    latest = latest_rows_per_series(engineered, origin)

    # SHAP reference set: the last year of each commodity's own rows, so
    # contributions read as "vs this commodity's recent normal".
    background = engineered.filter(pl.col("date") > origin - dt.timedelta(days=365))

    current = latest.select(
        [
            "commodity",
            "centre",
            pl.col(h.TARGET).alias("current_price"),
            pl.col("date").alias("as_of_date"),
            "staleness_days",
            "months_since_harvest",
        ]
    )

    forecast_rows, driver_rows = [], []
    for horizon in h.HORIZONS:
        band = q.band_from_fitted(
            q.fitted_for(train, origin, horizon), latest, conformal=True
        )
        point = lm.predict_with(lm.fitted_for(train, origin, horizon), latest).rename(
            {"pred": "point_forecast"}
        )

        merged = (
            band.join(point, on=["commodity", "centre"], how="inner")
            .join(current, on=["commodity", "centre"], how="inner")
            .with_columns(
                horizon=pl.lit(horizon, dtype=pl.Int64),
                pct_change=(pl.col("p50") / pl.col("current_price") - 1) * 100,
                band_width_pct=(pl.col("p90") - pl.col("p10")) / pl.col("current_price") * 100,
            )
        )
        forecast_rows.append(merged)
        driver_rows.append(
            _shap_drivers(
                lm.fitted_for(train, origin, horizon), latest, horizon, background
            )
        )

    return pl.concat(forecast_rows), pl.concat(driver_rows)


def _shap_drivers(
    model, today: pl.DataFrame, horizon: int, background: pl.DataFrame
) -> pl.DataFrame:
    """Top signed SHAP contributions for each series' current prediction,
    explained **against that commodity's own recent average** rather than
    against the pooled model's global base value.

    This matters a lot for readability. The point model is pooled across all
    three commodities, so its default base value sits somewhere between
    onion (~Rs 1,500/qtl) and tur (~Rs 7,000/qtl). Explained against that,
    every onion driver comes out strongly negative and every tur driver
    strongly positive, purely because of which commodity it is -- which
    tells an officer nothing about why *this* price is moving.

    Re-referencing is done by subtracting the commodity's mean SHAP vector
    rather than by passing a background set to TreeExplainer (SHAP refuses
    background data on models with categorical splits, which this model
    has). It is exactly equivalent and relies only on SHAP's additivity:

        pred(x)            = base + sum_j shap_j(x)
        mean_pred(commod)  = base + sum_j mean_shap_j(commod)
        => pred(x) - mean_pred(commod) = sum_j [shap_j(x) - mean_shap_j(commod)]

    so the re-referenced values still decompose the quantity we care about
    -- how far this centre's forecast sits above or below what is normal
    for this commodity right now. It also collapses the `commodity` feature
    to ~zero, which is correct: "it is onion" is not a driver of onion
    moving.
    """
    rows = []
    for commodity in today["commodity"].unique().to_list():
        subset = today.filter(pl.col("commodity") == commodity)
        bg = background.filter(pl.col("commodity") == commodity)
        if subset.height == 0 or bg.height == 0:
            continue

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(to_model_frame(subset))
        bg_values = explainer.shap_values(to_model_frame(bg))
        baseline = bg_values.mean(axis=0)

        X_subset = to_model_frame(subset)
        # commodity/centre are identity, not drivers -- "it is Delhi" is not
        # an explanation an officer can act on, so keep them out of the panel.
        candidates = [
            j for j, f in enumerate(MODEL_COLUMNS) if f not in CATEGORICAL_COLUMNS
        ]
        keys = subset.select(["commodity", "centre"]).rows()
        for i, (_, centre) in enumerate(keys):
            contribs = values[i] - baseline
            order = sorted(candidates, key=lambda j: abs(contribs[j]), reverse=True)[
                :TOP_DRIVERS
            ]
            for rank, j in enumerate(order):
                feature = MODEL_COLUMNS[j]
                raw = X_subset.iloc[i, j]
                rows.append(
                    {
                        "commodity": commodity,
                        "centre": centre,
                        "horizon": horizon,
                        "rank": rank + 1,
                        "feature": feature,
                        "label": human_label(feature),
                        "shap_value": float(contribs[j]),
                        "feature_value": format_feature_value(feature, raw),
                        "raw_value": None if raw is None or _is_na(raw) else str(raw),
                    }
                )
    return pl.DataFrame(rows)


def _is_na(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def format_feature_value(feature: str, raw) -> str | None:
    """Display string for a driver's current value.

    Raw repr is unreadable on screen -- momentum arrives as
    0.032258064516129004 and an EWMA as 1538.4286528487282. Formats by what
    the feature actually is: rates as percentages, prices and rainfall to
    sensible precision, flags as Yes/No.
    """
    if raw is None or _is_na(raw):
        return None
    if isinstance(raw, (bool, np.bool_)):
        return "Yes" if raw else "No"

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return str(raw)

    if feature.startswith("festival_"):
        return "Yes" if value else "No"
    if feature == "momentum_30":
        return f"{value * 100:+.1f}%"
    if feature.startswith("price_") or feature in ("temp_max", "temp_min"):
        return f"{value:,.0f}"
    if feature.startswith("rainfall"):
        return f"{value:,.1f} mm"
    if feature in ("month", "dow", "months_since_harvest"):
        return f"{value:.0f}"
    return f"{value:,.2f}"


def describe_driver(feature: str, value: str | None, shap_value: float) -> str:
    """One human-readable driver clause, e.g.
    'below-normal rainfall in growing districts' or '30-day momentum up 10.7%'.

    Phrases the *feature's actual value* where that reads naturally, rather
    than just naming the feature, since 'rainfall vs seasonal normal' alone
    tells an officer nothing about which direction it went.
    """
    label = human_label(feature)
    # Phrased against the commodity's recent norm, because that is exactly
    # what the re-referenced SHAP value measures (see _shap_drivers). Saying
    # "pushing prices up" here would be a different baseline from the
    # headline % change (which is vs today's price) and the two can point
    # opposite ways in the same sentence -- a price can sit below its
    # commodity's recent norm and still be forecast to tick up.
    direction = "above" if shap_value > 0 else "below"

    try:
        numeric = float(value) if value is not None else None
    except (TypeError, ValueError):
        numeric = None

    if feature == "momentum_30" and numeric is not None:
        return f"30-day momentum {numeric * 100:+.1f}%"
    if feature == "rainfall_dev_from_normal" and numeric is not None:
        word = "above" if numeric > 0 else "below"
        return f"rainfall {word} seasonal normal in growing districts"
    if feature in ("rainfall_7d_sum", "rainfall_30d_sum") and numeric is not None:
        window = "7-day" if "7d" in feature else "30-day"
        return f"{window} rainfall {numeric:.0f} mm in growing districts"
    if feature == "months_since_harvest" and numeric is not None:
        return f"{numeric:.0f} months since last harvest"
    if feature.startswith("festival_") and value in ("true", "True"):
        return f"{label} in window"
    if feature.startswith("price_roll_std"):
        return f"{label} elevated" if shap_value > 0 else f"{label} subdued"

    return f"{label} {direction} recent norm"


# Groupings used only to keep the driver sentence balanced across what the
# model actually knows about -- price history, weather, and the calendar --
# instead of it being whatever N features happen to have the largest raw
# SHAP magnitude, which in practice tends to be three flavours of "price
# trend" (lag/rolling-mean/EWMA are correlated and often rank together).
# Deliberately does not include a "market arrivals" or "production estimate"
# category: no arrivals or yield/sowing series exists anywhere in this
# pipeline (see the Why panel's own disclosure in app/dashboard.py), so
# inventing a category for it would either always be empty or, worse,
# tempt a future edit to fill it with a proxy that isn't really that thing.
TREND_FEATURES = {
    "momentum_30",
    *(f"price_lag_{n}" for n in (1, 7, 14, 30, 90)),
    *(f"price_roll_mean_{w}" for w in (7, 30, 90)),
    *(f"price_roll_std_{w}" for w in (7, 30, 90)),
    *(f"price_ewma_{s}" for s in (7, 30)),
}
WEATHER_FEATURES = {
    "rainfall_mm",
    "rainfall_7d_sum",
    "rainfall_30d_sum",
    "rainfall_dev_from_normal",
    "temp_max",
    "temp_min",
}
SEASON_FEATURES = {
    "month",
    "doy_sin",
    "doy_cos",
    "dow",
    "months_since_harvest",
    "festival_diwali",
    "festival_navratri",
    "festival_eid",
    "festival_onam",
}

# Order in which categories get a guaranteed slot in the sentence, if the
# data actually supports one (see _select_sentence_drivers).
SENTENCE_CATEGORY_PRIORITY = ("trend", "weather", "season")
SENTENCE_DRIVER_SLOTS = 4


def driver_category(feature: str) -> str:
    if feature in TREND_FEATURES:
        return "trend"
    if feature in WEATHER_FEATURES:
        return "weather"
    if feature in SEASON_FEATURES:
        return "season"
    return "other"  # cross-commodity prices, currently the only "other"


def _select_sentence_drivers(top_drivers: list[dict]) -> list[dict]:
    """Picks up to SENTENCE_DRIVER_SLOTS drivers for the sentence: the
    strongest real contributor from each of trend/weather/season first (in
    that priority order, skipping any category with no representative among
    this row's top SHAP drivers -- never invented), then fills any remaining
    slots by raw SHAP magnitude. `top_drivers` must already be sorted by
    rank (magnitude, descending).

    Without this, a naive top-N by magnitude can silently be e.g. three
    price-lag variants -- technically the largest SHAP values, but useless
    to an officer who already knows the price moved and wants to know why,
    and who has no way to tell that real weather/season signal for this row
    was simply outranked and dropped.
    """
    by_category: dict[str, dict] = {}
    for d in top_drivers:
        by_category.setdefault(driver_category(d["feature"]), d)

    chosen: list[dict] = []
    seen: set[str] = set()
    for category in SENTENCE_CATEGORY_PRIORITY:
        d = by_category.get(category)
        if d is not None:
            chosen.append(d)
            seen.add(d["feature"])

    for d in top_drivers:
        if len(chosen) >= SENTENCE_DRIVER_SLOTS:
            break
        if d["feature"] not in seen:
            chosen.append(d)
            seen.add(d["feature"])

    chosen.sort(key=lambda d: d["rank"])
    return chosen


def build_sentences(forecasts: pl.DataFrame, drivers: pl.DataFrame) -> pl.DataFrame:
    """The officer-facing sentence per (commodity, centre, horizon).

    CLAUDE.md is explicit that this sentence, not the SHAP plot, is the
    product: 'Bureaucrats don't read SHAP plots.' Wording is kept literal
    about what the model actually claims -- a forecast band, not a promise.
    """
    rows = []
    for f in forecasts.iter_rows(named=True):
        top = (
            drivers.filter(
                (pl.col("commodity") == f["commodity"])
                & (pl.col("centre") == f["centre"])
                & (pl.col("horizon") == f["horizon"])
            )
            .sort("rank")
        )
        selected = _select_sentence_drivers(list(top.iter_rows(named=True)))
        # raw_value, not the display-formatted feature_value -- describe_driver
        # parses numerics to phrase them ("momentum +10.7%").
        clauses = [
            describe_driver(d["feature"], d["raw_value"], d["shap_value"])
            for d in selected
        ]

        direction = "rise" if f["pct_change"] >= 0 else "ease"
        sentence = (
            f"{f['commodity'].title()} in {f['centre']} forecast to {direction} "
            f"{abs(f['pct_change']):.1f}% over {f['horizon']} days "
            f"(₹{f['current_price']:,.0f} → ₹{f['p50']:,.0f}/quintal, "
            f"80% band ₹{f['p10']:,.0f}–₹{f['p90']:,.0f})."
        )
        if clauses:
            sentence += " Main drivers vs this commodity's recent norm: " + ", ".join(clauses) + "."
        if f["spike_proba"] is not None:
            sentence += (
                f" Probability of a >{sc.SPIKE_THRESHOLD:.0%} rise within "
                f"{sc.SPIKE_WINDOW_DAYS} days: {f['spike_proba']:.0%}."
            )

        rows.append(
            {
                "commodity": f["commodity"],
                "centre": f["centre"],
                "horizon": f["horizon"],
                "sentence": sentence,
            }
        )
    return pl.DataFrame(rows)


def build_spike_probabilities(frame: pl.DataFrame, origin: dt.date) -> pl.DataFrame:
    """Current probability that each series spikes >8% within 14 days."""
    labelled = sc.add_spike_labels(frame)
    train_raw = frame.filter(pl.col("date") <= origin)
    engineered = build_features(train_raw)

    label_cutoff = origin - dt.timedelta(days=sc.SPIKE_WINDOW_DAYS)
    train_labelled = engineered.join(
        labelled.select(["commodity", "centre", "date", "spike"]),
        on=["commodity", "centre", "date"],
        how="left",
    ).filter(pl.col("date") <= label_cutoff)

    model = sc._fit_spike_model(train_labelled)
    today = latest_rows_per_series(engineered, origin)
    if model is None or today.height == 0:
        return today.select(["commodity", "centre"]).with_columns(
            spike_proba=pl.lit(None, dtype=pl.Float64)
        )

    proba = model.predict_proba(to_model_frame(today))[:, 1]
    return today.select(["commodity", "centre"]).with_columns(spike_proba=pl.Series(proba))


def build_stress(
    forecasts: pl.DataFrame, frame: pl.DataFrame, origin: dt.date
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Price Stress Index per (commodity, centre), plus a state-level roll-up
    for the map. Scoring lives in decide/stress.py -- see that module for the
    component definitions and weights.

    Only three of CLAUDE.md's five Phase 4 signals are computable from the
    data in this repo: retail-wholesale spread needs retail prices (100%
    null), arrivals decline needs an arrivals series (absent from the source
    entirely, verified against all 12 raw yearly files), and stock cover ratio
    needs a stock feed. Those three are omitted rather than mocked.

    Also returns the trailing medians used as the baseline, so
    build_procurement can score against the exact same "normal" rather than
    recomputing it -- the buy-side and sell-side signals must agree on what a
    centre's normal price is.
    """
    medians = stress_mod.trailing_median(frame, origin)
    per_centre = stress_mod.score(forecasts.filter(pl.col("horizon") == 14), medians)
    per_centre = stress_mod.add_band(per_centre)

    by_state = (
        per_centre.group_by(["state", "commodity"])
        .agg(
            stress=pl.col("stress").mean().round(1),
            level_component=pl.col("level_component").mean().round(3),
            spike_component=pl.col("spike_component").mean().round(3),
            band_component=pl.col("band_component").mean().round(3),
            pct_change=pl.col("pct_change").mean().round(2),
            n_centres=pl.len(),
        )
        .sort(["state", "commodity"])
    )
    return per_centre, by_state, medians


def build_procurement(
    forecasts: pl.DataFrame, medians: pl.DataFrame
) -> pl.DataFrame:
    """Procurement (buy-side) score per (commodity, centre) -- the other half
    of the PSF cycle. Scoring lives in decide/procurement.py; see that module
    for the component definitions. Reuses the same trailing-median baseline
    as the sell-side stress score (`medians`, computed once in build_stress)
    so the two halves cannot disagree about what "normal" is for a centre.
    """
    per_centre = procurement_mod.score(forecasts.filter(pl.col("horizon") == 14), medians)
    return procurement_mod.add_band(per_centre)


def main() -> None:
    # Windows' default console codepage (cp1252) can't encode the rupee
    # sign or box-drawing characters in polars' printed tables below.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = h.load_frame()
    origin = latest_origin(frame)
    print(f"building dashboard artifacts as of {origin}")

    forecasts, drivers = build_forecasts(frame, origin)
    print(f"  forecasts: {forecasts.height} rows")
    print(f"  shap drivers: {drivers.height} rows")

    spikes = build_spike_probabilities(frame, origin)
    forecasts = forecasts.join(spikes, on=["commodity", "centre"], how="left")
    print(f"  spike probabilities: {spikes.height} rows")

    # centre -> state, taken from the real data (state is null on imputed
    # rows, so take the first non-null per centre)
    states = (
        frame.filter(pl.col("state").is_not_null())
        .group_by("centre")
        .agg(state=pl.col("state").first())
    )
    forecasts = forecasts.join(states, on="centre", how="left")

    stress_by_centre, stress, medians = build_stress(forecasts, frame, origin)
    print(
        f"  stress index: {stress_by_centre.height} (commodity, centre) rows"
        f" -> {stress.height} (state, commodity) rows"
    )

    procurement = build_procurement(forecasts, medians)
    print(f"  procurement score: {procurement.height} (commodity, centre) rows")

    history = frame.filter(
        pl.col("date") > origin - dt.timedelta(days=HISTORY_DAYS)
    ).select(["date", "commodity", "centre", "state", h.TARGET, "is_imputed"])

    sentences = build_sentences(forecasts, drivers)
    print(f"  driver sentences: {sentences.height} rows")

    forecasts.write_parquet(OUT_DIR / "forecasts.parquet")
    drivers.write_parquet(OUT_DIR / "shap_drivers.parquet")
    sentences.write_parquet(OUT_DIR / "sentences.parquet")
    stress.write_parquet(OUT_DIR / "stress.parquet")
    stress_by_centre.write_parquet(OUT_DIR / "stress_by_centre.parquet")
    procurement.write_parquet(OUT_DIR / "procurement.parquet")
    history.write_parquet(OUT_DIR / "history.parquet")

    meta = {
        "as_of": str(origin),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "horizons": h.HORIZONS,
        "target": h.TARGET,
        "price_units": "INR per quintal (100 kg), wholesale mandi modal price",
        "spike_threshold_pct": sc.SPIKE_THRESHOLD * 100,
        "spike_window_days": sc.SPIKE_WINDOW_DAYS,
        "spike_decision_threshold": sc.DECISION_THRESHOLD,
        "conformal_calibration_window_days": q.CALIBRATION_WINDOW_DAYS,
        "stress_weights": {
            "forecast_level": stress_mod.W_LEVEL,
            "spike": stress_mod.W_SPIKE,
            "band_width": stress_mod.W_BAND,
        },
        "stress_full_scale": {
            "level_vs_1yr_median": stress_mod.LEVEL_FULL_SCALE,
            "band_width_pct_of_price": stress_mod.BAND_FULL_SCALE,
        },
        "procurement_full_scale": {
            "discount_vs_1yr_median": procurement_mod.DISCOUNT_FULL_SCALE,
            "harvest_window_months": procurement_mod.HARVEST_FULL_WINDOW_MONTHS,
            "off_season_discount_ceiling": procurement_mod.HARVEST_FLOOR,
        },
        "n_centres": int(frame["centre"].n_unique()),
        "n_commodities": int(frame["commodity"].n_unique()),
        "data_span": [str(frame["date"].min()), str(frame["date"].max())],
        "source": "Agmarknet daily mandi prices (Kaggle mirror) + Open-Meteo archive",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote artifacts to {OUT_DIR}")
    with pl.Config(tbl_rows=20, tbl_cols=-1, tbl_width_chars=200):
        print(stress.sort("stress", descending=True).head(10))


if __name__ == "__main__":
    main()
