# Progress log — Phase 2 (models)

Unattended run, scope locked to Phase 2 only (backtest harness, naive/seasonal
naive, SARIMAX, LightGBM direct multi-horizon @ 7d/14d, quantile LightGBM).
Not touching Phase 3/4 or ingest.

## 2026-08-17 — session start

- Surveyed repo state. `data/processed/modelling_frame.parquet` exists
  (39 cols, 99,988 rows, 3 commodities x 10 centres, 2015-01-01 to
  2025-11-05) with a real feature set already built: price lags
  (1/7/14/30/90), rolling mean/std (7/30/90), EWMA (7/30), momentum_30,
  rainfall (raw + 7d/30d cumulative + deviation from normal), temp,
  festival flags (Diwali/Navratri/Eid/Onam), months_since_harvest,
  cross-commodity same-state prices, calendar features (month/dow/doy sin-cos).
- Found and logged three scope issues to QUESTIONS.md before writing any
  model code: (1) no arrivals data exists anywhere in the repo despite the
  task's debug playbook assuming an arrivals join to check, (2) retail_price
  is 100% null so wholesale_price is the only usable modelling target, (3)
  the committed features/build.py doesn't reproduce the committed
  modelling_frame.parquet (script drift — parquet has ~4x the columns the
  current script would produce). Proceeding using the parquet as-is per the
  "keep going on something else" rule.
- Environment check: Python 3.10.11, polars 1.37.1, lightgbm 4.7.0,
  statsmodels 0.14.6, shap 0.49.1, scikit-learn 1.7.2, PuLP 3.3.0 all
  present. 24 CPUs available — plan to parallelize the SARIMAX walk-forward
  fits (720 of them: 24 origins x 30 series) with joblib if a single-fit
  timing test says it's needed.
