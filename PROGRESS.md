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
- **Discovered series have real reporting gaps, not a fully-filled daily
  grid**: e.g. tur/Delhi spans 3650 days but only has 2033 rows (a 1618-day
  hole). This matches prices.py's fill_gaps docstring ("ffill <=3 days ->
  state-median impute -> drop remaining nulls") -- rows that couldn't be
  imputed are dropped outright, not kept as nulls. Consequence: building
  direct multi-horizon labels with a row-positional `.shift(-horizon)`
  would silently grab the wrong date across a gap. `models/harness.py`'s
  `make_supervised()` instead joins each row to the actual row at
  `date + horizon` (calendar days), which is correct regardless of gaps and
  naturally nulls out the target when that date doesn't exist.
- Built `models/harness.py`: origin generation (30-day steps over the last
  2 years, 25 origins from 2023-10-23 to 2025-10-12), MAPE/RMSE, the
  `predict_fn(train, origin, horizon) -> [commodity, centre, pred]`
  interface every model plugs into, `run_backtest`/`summarize` for
  aggregation, and `make_supervised` for direct-horizon training frames.
  Evaluation targets are restricted to non-imputed rows (`~is_imputed`) so
  the backtest scores against real reported prices, not fabricated fill
  values -- models may still train on imputed rows. Smoke-tested against
  the real parquet: origins generate correctly, the date-based join gives
  correct 7-day-ahead targets on a spot-checked series, `summarize()`
  aggregates correctly on a synthetic frame (first version used
  `pl.map_groups` for MAPE/RMSE and hit a polars UDF return-type inference
  error -- replaced with plain vectorized polars expressions).
- **Found a real bug in the pre-built lag/rolling features**: confirmed
  `price_lag_1`/`price_lag_7`/`price_roll_mean_7`/etc. in
  `modelling_frame.parquet` were computed with row-positional
  shift/rolling over the gappy series, not date-aware -- e.g. right after a
  671-day gap in tur/Delhi, `price_lag_1 == price_lag_7` == the price from
  671 days earlier. Full writeup + spot-check numbers in QUESTIONS.md #4.
  Cross-commodity/calendar/weather/festival columns checked out fine
  (date-derived or same-date-joined, not shifted). Decision: LGBM will
  recompute its own lag/rolling/EWMA/momentum features date-safe from raw
  `wholesale_price`, reusing everything else from the parquet as-is.
- Built and tested `models/baselines.py` (naive flat forecast, seasonal
  naive = same calendar date last year with a flat-forecast fallback when
  that date is missing). Ran both through the full harness across all 25
  origins x {7,14}d: completes in well under a second. Results are sane
  and match domain expectations -- naive clearly beats seasonal_naive at
  these short horizons (onion/potato/tur are volatile enough that
  short-term persistence beats a 365-day-ago echo):

  | commodity | h  | naive MAPE | seasonal_naive MAPE |
  |-----------|----|-----------:|---------------------:|
  | onion     | 7  | 13.4%      | 48.1%                |
  | onion     | 14 | 14.2%      | 49.4%                |
  | potato    | 7  | ~20.6%     | ~40%+                |
  | tur       | 7  | 4.3%       | 31.7%                |
  | tur       | 14 | 4.7%       | 30.2%                |

  N per (commodity,horizon) cell is 116-159 out of a 250 max (10 centres x
  25 origins) -- coverage loss is expected and correct: it comes from
  requiring a real (non-imputed) actual at the target date and a real row
  at the origin date, which real reporting gaps sometimes fail.
