# PSS01 — AI Price Intelligence & Buffer Stock Decision Support

Forecasts wholesale mandi prices for **tur, onion and potato** across 10
benchmark centres, explains each forecast in plain English, flags price
spikes early enough to act on, and recommends how much buffer stock to
release (and when to buy to rebuild it).

Built for the Price Monitoring Division, Department of Consumer Affairs.

**Live demo:** https://hackunamatata-sih.streamlit.app/

---

## The one-sentence product

An officer opens a map, clicks "Tur dal, Maharashtra," sees the 7/14-day
price forecast, sees **why**, and gets told **how much buffer stock to
release and when**. Today's DoCA workflow (SARIMAX/ARIMA) stops at the
number; this stops at the decision.

Four stages: **Data → Forecast → Explain → Decide**.

## Headline numbers

All figures are from walk-forward backtests over the last two years of real
data (25 rolling origins, 30-day steps), not in-sample fits.

| | |
|---|---|
| **Forecast accuracy** | LightGBM macro-avg MAPE **9.82%** vs SARIMAX **10.20%** (DoCA's current approach) and naive **13.24%** |
| **Spike early warning** | median **7 days** of lead time before a spike episode begins; **87.4%** of episodes caught (505/578) at 50% precision / 78% recall |
| **Uncertainty bands** | conformally calibrated — 80% band coverage **77.8%** overall (was 70.5% raw), 80.7% at the 14-day horizon |
| **Retrospective validation** | replayed against the real Aug–Dec 2023 onion crisis (₹819 → ₹6,000/qtl): every one of 9 tracked centres alerted in the first days of August, a median **88 days** before the late-October peak |

Full tables: `models/backtest_results.csv`, `models/spike_results.csv`,
`models/quantile_coverage.csv`. Full feature/model/screen documentation:
**`FEATURES.md`**.

## The five dashboard screens

1. **National map** — India shaded by Price Stress Index (0–100), red/amber/green.
2. **Commodity view** — historical price line + forecast with a shaded P10–P90 band, 7/14-day horizon toggle.
3. **Why** — horizontal SHAP driver bars plus the plain-English generated sentence (the actual product — see `FEATURES.md`).
4. **Action** — the buffer-stock release plan (LP-optimised, signable table), a procurement/buy-side flag, a live what-if slider bending the price path against a do-nothing baseline, and one-click PDF export.
5. **Time machine** — replays the 2023 onion crisis with no hindsight, showing how many days earlier this system would have flagged it.

## Run the dashboard

```bash
pip install -r requirements.txt      # full dev/modelling env
streamlit run app/dashboard.py
```

Serving only needs `app/requirements.txt` / `requirements-app.txt` (no
lightgbm/shap/statsmodels — the app reads pre-built artifacts and never fits
a model at request time).

## Deploy

**Live on Streamlit Community Cloud** (free, no Docker build, deploys
straight from this GitHub repo — connect at share.streamlit.io, main file
`app/dashboard.py`, it auto-detects `app/requirements.txt`).

```bash
# Any Docker host (Render, Railway, Fly, Cloud Run, self-hosted):
docker build -t pss01 . && docker run -p 7860:7860 pss01
```

Note: Hugging Face Spaces' free `cpu-basic` tier now gates the Docker and
Gradio SDKs behind a PRO subscription (a recent HF policy change, confirmed
2026-08-17) — `deploy/deploy_hf.py` still works for accounts with that access,
but Streamlit Community Cloud is the actual free path in use. See
`deploy/README.md` for Render/Railway settings.

## Regenerate everything

```bash
cd models
python run_backtest.py              # -> backtest_results.csv
python run_spike_backtest.py        # -> spike_results*.csv, spike_scored.parquet
python time_machine.py              # -> time_machine_onion_2023.csv
python build_dashboard_artifacts.py # -> app/data/*
```

Tests: `python -m pytest models/tests/` (30 tests, ~6s).

## Layout

```
ingest/     one script per source, idempotent          (Phase 1)
features/   feature builder                            (Phase 1)
models/     harness, baselines, SARIMAX, LightGBM,
            quantile + conformal, spike classifier,
            time machine replay                        (Phase 2 + 5)
app/        Streamlit dashboard + prebuilt artifacts    (Phase 3)
decide/     stress index, release optimiser, procurement
            signal, what-if simulator, PDF brief        (Phase 4)
deploy/     Docker + Hugging Face Spaces / Streamlit Cloud deploy
api/        FastAPI                                     (not built)
```

## Modelling notes

- **Target is `wholesale_price`** (₹/quintal, Agmarknet modal). Retail prices
  are not available — see `QUESTIONS.md` #2.
- **One pooled global model per horizon**, not one model per series, with
  commodity and centre as categorical features. This is the scaling story:
  22 commodities × 550 centres is one model, not 12,100.
- **Reporting is patchy.** ~26% of rows are gap-filled. Lag/rolling/EWMA
  features are computed on a **calendar basis, not row positions**, so a
  "7-day lag" is never silently a 671-day lag across a reporting gap.
  Accuracy is only ever scored against genuine reported prices.
- **Forecasts are the no-intervention counterfactual.** If a release is made
  and the price falls, the forecast was not wrong — evaluation is against the
  policy-adjusted outcome.
- **Decision support, never automation.** Probabilistic outputs, explicit
  confidence, and an officer signs — not the model.
- **The what-if price path and the LP's capacity/transport inputs are
  explicitly labelled illustrative estimates**, not fitted/sourced figures —
  no source this project ingests logs historical buffer-stock releases to
  calibrate against. The UI and PDF carry these labels wherever the numbers
  appear. See `FEATURES.md` and `decide/reference_data.py`.

## Known gaps

Stated plainly rather than papered over; details in `QUESTIONS.md`.

- **No mandi arrivals data.** The strongest known leading indicator is absent
  from the source — verified against all raw yearly files, which carry
  prices only (`Arrival_Date` is a date, not a tonnage). This is the single
  biggest available accuracy improvement.
- **No retail prices**, so no retail–wholesale spread and no hoarding signal.
- **No buffer stock / capacity / transport data**, which is what the Phase 4
  release optimiser works around with labelled estimates instead.
- LightGBM beats SARIMAX on average but not in every commodity × horizon
  cell; tur's short-horizon baselines are very strong (naive MAPE 4.3%).
- The 7-day uncertainty band still under-covers (75.1% vs 80% target).
- **Forecast horizons are 7/14 days only** (the original 1/7/14/30 spec was
  narrowed); there is no recursive/autoregressive mechanism to forecast
  further out, so the dashboard's forecasts extend from the data's real last
  date, not from today.
- **The underlying price series currently ends 2025-11-05**, not the present
  day. Freshest raw data (a rerunnable Kaggle-mirrored Agmarknet source) is
  actually cached through April 2026; the gap is a benchmark-market
  selection artifact in `ingest/prices.py`, not a source limitation — a
  known, fixable follow-up.

`PROGRESS.md` is the running engineering log; `QUESTIONS.md` holds open
decisions for the team; `FEATURES.md` documents every feature, model and
screen in detail.
