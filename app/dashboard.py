"""
PSS01 -- AI Price Intelligence & Buffer Stock Decision Support (Phase 3).

Four screens: national stress map, commodity forecast view, why-panel, and
the Phase 4 action placeholder.

Every number rendered here is real. Forecasts, bands, SHAP drivers and spike
probabilities come from `app/data/*` (written by
models/build_dashboard_artifacts.py from the fitted Phase 2 models);
accuracy figures come straight from models/backtest_results.csv and
models/spike_results*.csv. There is no mock data anywhere in this file --
where a number genuinely does not exist yet (buffer stock levels, arrivals,
retail prices) the app says so instead of inventing it.

Run:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import theme as T

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "app" / "data"
ASSETS = ROOT / "app" / "assets"
MODELS = ROOT / "models"

# The GeoJSON names Delhi "Delhi"; our price data calls the state "NCT of
# Delhi". Every other state name matches exactly.
STATE_TO_GEO = {"NCT of Delhi": "Delhi"}

st.set_page_config(
    page_title="PSS01 · Price Intelligence & Buffer Stock Support",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(T.CSS, unsafe_allow_html=True)


# --------------------------------------------------------------- data loading


@st.cache_data
def load() -> dict:
    d = {
        "forecasts": pl.read_parquet(DATA / "forecasts.parquet"),
        "drivers": pl.read_parquet(DATA / "shap_drivers.parquet"),
        "sentences": pl.read_parquet(DATA / "sentences.parquet"),
        "stress": pl.read_parquet(DATA / "stress.parquet"),
        "history": pl.read_parquet(DATA / "history.parquet"),
        "meta": json.loads((DATA / "meta.json").read_text()),
        "backtest": pl.read_csv(MODELS / "backtest_results.csv"),
        "spike_sweep": pl.read_csv(MODELS / "spike_results.csv"),
        "spike_by_commodity": pl.read_csv(MODELS / "spike_results_by_commodity.csv"),
        "coverage": pl.read_csv(MODELS / "quantile_coverage.csv"),
    }
    d["geojson"] = json.loads((ASSETS / "india_states.geojson").read_text())
    return d


D = load()
META = D["meta"]


def tile(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div>'
        f'<div class="tile-note">{note}</div></div>'
    )


def band_pill(band: str) -> str:
    colour = T.STATUS[band]
    return (
        f'<span class="pill" style="background:{colour}22;color:{colour};'
        f'border:1px solid {colour}66">{band}</span>'
    )


# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("### PSS01")
    st.markdown(
        f'<div style="color:{T.TEXT_MUTED};font-size:0.8rem;line-height:1.5">'
        f"Price Monitoring Division<br/>Dept. of Consumer Affairs"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    screen = st.radio(
        "Screen",
        ["National map", "Commodity view", "Why", "Action"],
        label_visibility="collapsed",
    )

    st.divider()
    commodities = sorted(D["forecasts"]["commodity"].unique().to_list())
    commodity = st.selectbox("Commodity", commodities, index=commodities.index("onion"))

    centres = sorted(
        D["forecasts"].filter(pl.col("commodity") == commodity)["centre"].unique().to_list()
    )
    centre = st.selectbox("Centre", centres)
    horizon = st.radio("Forecast horizon", META["horizons"], format_func=lambda h: f"{h} days",
                       horizontal=True)

    st.divider()
    st.markdown(
        f'<div style="font-size:0.75rem;color:{T.TEXT_MUTED};line-height:1.6">'
        f'<b style="color:{T.TEXT_SECONDARY}">Data as of</b><br/>{META["as_of"]}<br/><br/>'
        f'<b style="color:{T.TEXT_SECONDARY}">Coverage</b><br/>'
        f'{META["n_commodities"]} commodities · {META["n_centres"]} centres<br/><br/>'
        f'<b style="color:{T.TEXT_SECONDARY}">Prices</b><br/>'
        f"Wholesale mandi modal,<br/>₹ per quintal (100 kg)<br/><br/>"
        f'<b style="color:{T.TEXT_SECONDARY}">Sources</b><br/>{META["source"]}'
        f"</div>",
        unsafe_allow_html=True,
    )


def selected_forecast() -> dict | None:
    rows = D["forecasts"].filter(
        (pl.col("commodity") == commodity)
        & (pl.col("centre") == centre)
        & (pl.col("horizon") == horizon)
    )
    return rows.to_dicts()[0] if rows.height else None


# --------------------------------------------------------------- 1. the map


def screen_map() -> None:
    st.markdown('<div class="hero-sub">National overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero">Price Stress Index</div>', unsafe_allow_html=True)
    st.caption(
        "Composite 0–100 per state × commodity, built on the 14-day outlook · "
        f"as of {META['as_of']}"
    )

    stress = D["stress"].filter(pl.col("commodity") == commodity).with_columns(
        geo_state=pl.col("state").replace(STATE_TO_GEO)
    )

    national = D["stress"]
    worst = national.sort("stress", descending=True).head(1).to_dicts()[0]
    n_high = national.filter(pl.col("stress") >= 60).height
    mean_stress = national["stress"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        tile("Highest stress now", f"{worst['stress']:.0f}",
             f"{worst['commodity'].title()} · {worst['state']}"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        tile("State×commodity in High band", f"{n_high}",
             f"of {national.height} tracked"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        tile("Mean stress", f"{mean_stress:.0f}", "across all tracked pairs"),
        unsafe_allow_html=True,
    )
    spike_hdr = D["spike_sweep"].filter(
        pl.col("decision_threshold") == META["spike_decision_threshold"]
    ).to_dicts()[0]
    c4.markdown(
        tile("Spike early warning",
             f"{spike_hdr['median_lead_days_per_episode']:.0f} days",
             "median lead time, backtested"),
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1.35, 1])

    with left:
        all_states = [ft["properties"]["ST_NM"] for ft in D["geojson"]["features"]]
        fig = go.Figure()
        # Base layer: every Indian state in a flat neutral, so the country
        # reads as India rather than as a handful of floating shapes. Only
        # the 9 states with a reporting centre for this commodity get colour.
        fig.add_trace(
            go.Choropleth(
                geojson=D["geojson"],
                featureidkey="properties.ST_NM",
                locations=all_states,
                z=[0] * len(all_states),
                colorscale=[[0, T.SURFACE_RAISED], [1, T.SURFACE_RAISED]],
                showscale=False,
                marker_line_color=T.BORDER,
                marker_line_width=0.7,
                hovertemplate="<b>%{location}</b><br>No reporting centre<extra></extra>",
            )
        )
        fig.add_trace(
            go.Choropleth(
                geojson=D["geojson"],
                featureidkey="properties.ST_NM",
                locations=stress["geo_state"].to_list(),
                z=stress["stress"].to_list(),
                zmin=0,
                zmax=100,
                colorscale=T.STRESS_SCALE,
                marker_line_color=T.SURFACE,
                marker_line_width=1.1,
                colorbar=dict(
                    title=dict(text="Stress", font=dict(color=T.TEXT_MUTED, size=11)),
                    tickfont=dict(color=T.TEXT_MUTED, size=10),
                    thickness=12,
                    len=0.65,
                    y=0.5,
                    outlinewidth=0,
                ),
                customdata=stress.select(["state", "pct_change", "n_centres"]).to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Stress %{z:.0f}/100"
                    "<br>14d forecast %{customdata[1]:+.1f}%"
                    "<br>%{customdata[2]} centre(s)<extra></extra>"
                ),
            )
        )
        fig.update_geos(
            visible=False,
            bgcolor="rgba(0,0,0,0)",
            projection_type="mercator",
            center=dict(lat=22.5, lon=82.5),
            lataxis_range=[5, 37],
            lonaxis_range=[67, 98],
        )
        fig.update_layout(
            height=620,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            font=dict(color=T.TEXT_SECONDARY),
            dragmode=False,
            hoverlabel=dict(bgcolor=T.SURFACE_RAISED, bordercolor=T.BORDER,
                            font=dict(color=T.TEXT_PRIMARY)),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            f"Grey states have no reporting centre for {commodity} in this pilot — "
            "10 benchmark centres, not all 550."
        )

    with right:
        st.markdown("##### Ranked by stress")
        ranked = (
            D["stress"]
            .sort("stress", descending=True)
            .with_columns(
                band=pl.col("stress").map_elements(T.stress_band, return_dtype=pl.String)
            )
            .select(
                pl.col("state").alias("State"),
                pl.col("commodity").str.to_titlecase().alias("Commodity"),
                pl.col("stress").alias("Stress"),
                pl.col("band").alias("Band"),
                pl.col("pct_change").alias("Fcst %"),
            )
        )
        st.dataframe(
            ranked.to_pandas(),
            use_container_width=True,
            hide_index=True,
            height=430,
            column_config={
                "Stress": st.column_config.ProgressColumn(
                    "Stress", min_value=0, max_value=100, format="%.0f"
                ),
                "Fcst %": st.column_config.NumberColumn("Fcst %", format="%+.1f%%"),
            },
        )

    with st.expander("How the Price Stress Index is built"):
        w = META["stress_weights"]
        st.markdown(
            f"""
A weighted composite of three signals, each produced by a fitted model and
scaled 0–1 before weighting:

| Component | Weight | Source |
|---|---|---|
| Forecast pressure | {w['forecast']:.0%} | LightGBM 14-day forecast vs today's price; saturates at +10% |
| Spike risk | {w['spike']:.0%} | Classifier probability of a >8% rise within 14 days |
| Forecast uncertainty | {w['uncertainty']:.0%} | Conformal P10–P90 band width as a share of price; saturates at 60% |

**What this index does not yet include.** CLAUDE.md's full Phase 4 index adds
retail–wholesale spread widening, arrivals decline vs seasonal norm, and buffer
stock cover ratio. None of the three can be computed from the data in this repo:
retail prices are entirely null, no arrivals series exists in the source at all,
and there is no stock-level feed. They are omitted rather than mocked, so this is
an honest partial index — Phase 4 completes it.
            """
        )


# ------------------------------------------------- 2. commodity forecast view


def screen_commodity() -> None:
    f = selected_forecast()
    st.markdown('<div class="hero-sub">Commodity view</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero">{commodity.title()} · {centre}</div>', unsafe_allow_html=True
    )

    if f is None:
        st.warning(f"No current forecast for {commodity} in {centre}.")
        return

    st.caption(
        f"Wholesale modal price, ₹/quintal · latest reading {f['as_of_date']}"
        + (f" ({f['staleness_days']} days stale)" if f["staleness_days"] else "")
    )

    band = T.stress_band(
        D["stress"]
        .filter((pl.col("commodity") == commodity) & (pl.col("state") == f["state"]))["stress"]
        .to_list()[0]
    )
    direction = "▲" if f["pct_change"] >= 0 else "▼"
    colour = T.NEGATIVE if f["pct_change"] >= 0 else T.STATUS["Low"]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(tile("Current price", f"₹{f['current_price']:,.0f}", "per quintal"),
                unsafe_allow_html=True)
    c2.markdown(
        tile(f"{horizon}-day forecast (P50)",
             f"<span style='color:{colour}'>₹{f['p50']:,.0f}</span>",
             f"{direction} {abs(f['pct_change']):.1f}% vs today"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        tile("80% band", f"₹{f['p10']:,.0f}–{f['p90']:,.0f}",
             f"width {f['band_width_pct']:.0f}% of price"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        tile("Spike probability",
             f"{f['spike_proba']:.0%}" if f["spike_proba"] is not None else "—",
             f">{META['spike_threshold_pct']:.0f}% rise within "
             f"{META['spike_window_days']}d · state {band}"),
        unsafe_allow_html=True,
    )

    st.write("")

    hist_all = (
        D["history"]
        .filter((pl.col("commodity") == commodity) & (pl.col("centre") == centre))
        .sort("date")
    )
    # A 7-day forecast against 400 days of history is invisible. Default to a
    # window where the band is actually legible; full history stays available.
    window = st.radio(
        "History window",
        [90, 180, 400],
        index=0,
        horizontal=True,
        format_func=lambda d: f"{d} days",
        label_visibility="collapsed",
    )
    hist = hist_all.filter(
        pl.col("date") > f["as_of_date"] - dt.timedelta(days=window)
    )
    fig = go.Figure()

    # Uncertainty band first so the lines sit on top of it.
    last_date = f["as_of_date"]
    target_date = last_date + dt.timedelta(days=horizon)
    fig.add_trace(
        go.Scatter(
            x=[last_date, target_date, target_date, last_date],
            y=[f["current_price"], f["p90"], f["p10"], f["current_price"]],
            fill="toself",
            fillcolor=T.ACCENT_SOFT,
            line=dict(width=0),
            hoverinfo="skip",
            name="80% band (P10–P90)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["date"].to_list(),
            y=hist[META["target"]].to_list(),
            mode="lines",
            line=dict(color=T.TEXT_SECONDARY, width=2),
            name="Observed price",
            hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}/qtl<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_date, target_date],
            y=[f["current_price"], f["p50"]],
            mode="lines+markers",
            line=dict(color=T.ACCENT, width=2.5),
            marker=dict(size=9, color=T.ACCENT,
                        line=dict(color=T.SURFACE, width=2)),
            name=f"Forecast P50 ({horizon}d)",
            hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}/qtl<extra></extra>",
        )
    )
    # Direct-label the endpoint rather than every point.
    fig.add_annotation(
        x=target_date, y=f["p50"], text=f"  ₹{f['p50']:,.0f}", showarrow=False,
        xanchor="left", font=dict(color=T.ACCENT, size=13),
    )
    fig.update_yaxes(title_text="₹ per quintal")
    T.style_fig(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)

    imputed = hist.filter(pl.col("is_imputed")).height
    st.caption(
        f"{hist.height} days shown · {imputed} ({imputed / max(hist.height, 1):.0%}) "
        "are gap-filled readings, carried forward or state-median imputed. "
        "Model accuracy is only ever scored against genuine reported prices."
    )

    acc = D["backtest"].filter(
        (pl.col("commodity") == commodity) & (pl.col("horizon_days") == horizon)
    ).to_dicts()
    if acc:
        a = acc[0]
        st.markdown("##### Backtested accuracy for this commodity & horizon")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(tile("LightGBM MAPE", f"{a['lgbm_mape']:.1f}%", "our model"),
                    unsafe_allow_html=True)
        m2.markdown(tile("SARIMAX MAPE", f"{a['sarimax_mape']:.1f}%", "DoCA's current approach"),
                    unsafe_allow_html=True)
        m3.markdown(tile("Naive MAPE", f"{a['naive_mape']:.1f}%", "persistence floor"),
                    unsafe_allow_html=True)
        delta = a["lgbm_improvement_vs_sarimax_pct"]
        m4.markdown(
            tile("vs SARIMAX", f"{delta:+.1f}%",
                 "lower is better" if delta < 0 else "SARIMAX ahead here"),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Walk-forward backtest, {a['lgbm_n']} scored origin-days over the last "
            "two years. Full table: models/backtest_results.csv"
        )


# ------------------------------------------------------------------ 3. why


def screen_why() -> None:
    f = selected_forecast()
    st.markdown('<div class="hero-sub">Explanation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero">Why this forecast</div>', unsafe_allow_html=True)

    if f is None:
        st.warning(f"No current forecast for {commodity} in {centre}.")
        return

    sentence = D["sentences"].filter(
        (pl.col("commodity") == commodity)
        & (pl.col("centre") == centre)
        & (pl.col("horizon") == horizon)
    )["sentence"].to_list()
    if sentence:
        st.markdown(f'<div class="sentence">{sentence[0]}</div>', unsafe_allow_html=True)
    st.write("")

    drivers = (
        D["drivers"]
        .filter(
            (pl.col("commodity") == commodity)
            & (pl.col("centre") == centre)
            & (pl.col("horizon") == horizon)
        )
        .sort("rank", descending=True)
    )

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("##### Top model drivers")
        fig = go.Figure(
            go.Bar(
                x=drivers["shap_value"].to_list(),
                y=drivers["label"].to_list(),
                orientation="h",
                marker=dict(
                    color=[
                        T.POSITIVE if v > 0 else T.NEGATIVE
                        for v in drivers["shap_value"].to_list()
                    ],
                    line=dict(color=T.SURFACE, width=2),
                ),
                hovertemplate="%{y}<br>%{x:+,.0f} ₹/qtl vs norm<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1, line_color=T.TEXT_MUTED)
        fig.update_xaxes(title_text="Contribution vs this commodity's recent norm (₹/quintal)")
        T.style_fig(fig, height=360, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<span style="color:{T.POSITIVE}">■</span> above recent norm &nbsp;&nbsp; '
            f'<span style="color:{T.NEGATIVE}">■</span> below recent norm',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("##### Driver values")
        st.dataframe(
            drivers.sort("rank")
            .select(
                pl.col("rank").alias("#"),
                pl.col("label").alias("Driver"),
                pl.col("feature_value").alias("Current value"),
                pl.col("shap_value").round(0).alias("₹ vs norm"),
            )
            .to_pandas(),
            use_container_width=True,
            hide_index=True,
            height=280,
        )
        cov = D["coverage"].filter(
            (pl.col("commodity") == commodity) & (pl.col("horizon") == horizon)
        ).to_dicts()
        if cov:
            c = cov[0]
            st.markdown(
                f'<div class="caveat">Band reliability: this commodity/horizon\'s 80% '
                f'band actually contained the outcome <b>{c["conformal_coverage"]:.0%}</b> '
                f'of the time in backtest (target 80%, was {c["raw_coverage"]:.0%} before '
                f'conformal calibration). Read the band accordingly.</div>',
                unsafe_allow_html=True,
            )

    with st.expander("How to read this panel"):
        st.markdown(
            """
Bars are SHAP contributions from the LightGBM forecast model, re-referenced
against **this commodity's own recent average** rather than the pooled model's
global baseline. Without that re-referencing every onion driver would read
strongly negative and every tur driver strongly positive purely because onion
trades near ₹1,500/qtl and tur near ₹7,000/qtl — an artifact of commodity
identity, not a statement about price pressure.

So a positive bar means *this centre sits above what is normal for this
commodity right now*. That is a different baseline from the headline % change
(which is against today's price), and the two can legitimately point in
opposite directions: a price can sit below its commodity's recent norm and
still be forecast to tick up.

Commodity and centre identity are excluded from the bars — "it is Delhi" is not
a driver an officer can act on.

**Missing driver, stated plainly:** mandi arrivals are the strongest known
leading indicator for these commodities and are *not* in this model, because no
arrivals series exists in the source data (verified against all 12 raw yearly
files). Every driver shown is real; the set is not complete.
            """
        )


# --------------------------------------------------------------- 4. action


def screen_action() -> None:
    st.markdown('<div class="hero-sub">Decision support</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero">Recommended action</div>', unsafe_allow_html=True)
    st.caption("Phase 4 — release optimiser, what-if simulator, PDF brief")

    st.markdown(
        '<div class="caveat"><b>This screen is a deliberate placeholder.</b> '
        "The release optimiser is Phase 4 and is not built yet. Rather than show a "
        "mocked-up recommendation table that looks signable but is not backed by an "
        "optimiser, this screen lists the real model inputs already available to feed "
        "it, and states exactly what is still missing.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns(2)

    with left:
        st.markdown("##### Ready — real model outputs available now")
        ready = D["stress"].sort("stress", descending=True).head(6).with_columns(
            band=pl.col("stress").map_elements(T.stress_band, return_dtype=pl.String)
        )
        st.dataframe(
            ready.select(
                pl.col("state").alias("State"),
                pl.col("commodity").str.to_titlecase().alias("Commodity"),
                pl.col("stress").alias("Stress"),
                pl.col("band").alias("Band"),
                pl.col("pct_change").alias("14d fcst %"),
            ).to_pandas(),
            use_container_width=True,
            hide_index=True,
            column_config={
                "14d fcst %": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )
        st.markdown(
            """
These feed the optimiser's objective directly:

- forecast price path with calibrated P10/P50/P90 per centre
- probability of a >8% spike within 14 days, per centre
- median **7 days** of early warning before a spike episode begins
- per-state stress ranking to prioritise candidate release locations
            """
        )

    with right:
        st.markdown("##### Blocked — inputs that do not exist yet")
        st.markdown(
            """
The LP in CLAUDE.md Phase 4 is

```
minimize  Σ (price deviation from target)
          + λ · transport cost
s.t.      Σ release_i ≤ available_stock
          0 ≤ release_i
            ≤ state_absorption_capacity_i
```

Three of its inputs have no data source in this repo:

| Input | Status |
|---|---|
| `available_stock` | No NAFED/NCCF buffer stock feed ingested |
| `state_absorption_capacity_i` | No capacity reference data |
| transport cost matrix | Not sourced |

Two further Phase 4 signals are also unavailable and are the reason the stress
index on screen 1 is a partial composite: **retail–wholesale spread** (retail
prices are 100% null in the pipeline) and **arrivals decline vs seasonal norm**
(no arrivals series exists in the source at all).

Wiring any of these is a Phase 1 ingest task. Until then a release
recommendation would be a guess wearing a table's clothing, which is precisely
what a decision-support tool for a ₹10,000 crore fund should not do.
            """
        )

    st.divider()
    st.markdown("##### Backtested spike early-warning performance")
    sweep = D["spike_sweep"]
    chosen = META["spike_decision_threshold"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sweep["recall"].to_list(),
            y=sweep["precision"].to_list(),
            mode="lines+markers",
            line=dict(color=T.ACCENT, width=2),
            marker=dict(size=8, color=T.ACCENT, line=dict(color=T.SURFACE, width=2)),
            customdata=sweep.select(
                ["decision_threshold", "median_lead_days_per_episode"]
            ).to_numpy(),
            name="Operating points",
            hovertemplate=(
                "threshold %{customdata[0]:.2f}<br>precision %{y:.1%}"
                "<br>recall %{x:.1%}<br>lead %{customdata[1]:.0f}d<extra></extra>"
            ),
        )
    )
    sel = sweep.filter(pl.col("decision_threshold") == chosen).to_dicts()[0]
    fig.add_trace(
        go.Scatter(
            x=[sel["recall"]], y=[sel["precision"]], mode="markers",
            marker=dict(size=15, color=T.STATUS["Moderate"],
                        line=dict(color=T.SURFACE, width=2)),
            name=f"Selected (threshold {chosen})",
            hovertemplate=f"selected · lead {sel['median_lead_days_per_episode']:.0f}d<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Recall", tickformat=".0%")
    fig.update_yaxes(title_text="Precision", tickformat=".0%")
    T.style_fig(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The alert threshold is a policy choice, not a modelling one — it trades false "
        "alarms against missed spikes. Full sweep: models/spike_results.csv"
    )


SCREENS = {
    "National map": screen_map,
    "Commodity view": screen_commodity,
    "Why": screen_why,
    "Action": screen_action,
}
SCREENS[screen]()
