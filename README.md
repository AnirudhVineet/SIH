# PSS01 — AI Price Intelligence & Buffer Stock Decision Support

Forecasts wholesale mandi prices for **tur, onion and potato** across 10
benchmark centres, explains each forecast in plain English, and flags price
spikes early enough to act on.

Built for the Price Monitoring Division, Department of Consumer Affairs.

---

## Headline numbers

All figures are from walk-forward backtests over the last two years of real
data (25 rolling origins, 30-day steps), not in-sample fits.

| | |
|---|---|
| **Forecast accuracy** | LightGBM macro-avg MAPE **9.82%** vs SARIMAX **10.20%** (DoCA's current approach) and naive **13.24%** |
| **Spike early warning** | median **7 days** of lead time before a spike episode begins; **87.4%** of episodes caught (505/578) at 50% precision / 78% recall |
| **Uncertainty bands** | conformally calibrated — 80% band coverage **77.8%** overall (was 70.5% raw), 80.7% at the 14-day horizon |

Full tables: `models/backtest_results.csv`, `models/spike_results.csv`,
`models/quantile_coverage.csv`.

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The app reads pre-built artifacts from `app/data/` and never trains on page
load. Four screens: national stress map, commodity forecast view, why-panel
(SHAP + generated sentence), and the Phase 4 action placeholder.

## Regenerate everything

```bash
cd models
python run_backtest.py              # -> backtest_results.csv
python run_spike_backtest.py        # -> spike_results*.csv, spike_scored.parquet
python build_dashboard_artifacts.py # -> app/data/*
```

Tests: `python -m pytest models/tests/` (30 tests, ~6s).

## Layout

```
ingest/     one script per source, idempotent          (Phase 1)
features/   feature builder                            (Phase 1)
models/     harness, baselines, SARIMAX, LightGBM,
            quantile + conformal, spike classifier     (Phase 2)
app/        Streamlit dashboard + prebuilt artifacts   (Phase 3)
decide/     release optimiser                          (Phase 4 — not built)
api/        FastAPI                                    (not built)
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

## Known gaps

Stated plainly rather than papered over; details in `QUESTIONS.md`.

- **No mandi arrivals data.** The strongest known leading indicator is absent
  from the source — verified against all 12 raw yearly files, which carry
  prices only (`Arrival_Date` is a date, not a tonnage). This is the single
  biggest available accuracy improvement.
- **No retail prices**, so no retail–wholesale spread and no hoarding signal.
- **No buffer stock / capacity / transport data**, which is what blocks the
  Phase 4 release optimiser.
- LightGBM beats SARIMAX on average but not in every commodity × horizon
  cell; tur's short-horizon baselines are very strong (naive MAPE 4.3%).
- The 7-day uncertainty band still under-covers (75.1% vs 80% target).

`PROGRESS.md` is the running engineering log; `QUESTIONS.md` holds open
decisions for the team.
