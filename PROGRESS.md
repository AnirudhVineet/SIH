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
- Built `models/sarimax_model.py`: order (1,1,1), no seasonal term, fit per
  (commodity, centre) on full history up to each origin, reindexed to daily
  frequency (`asfreq('D')`) so a "14 steps ahead" forecast always lands on
  origin+14 calendar days regardless of gaps -- statsmodels' state-space
  SARIMAX handles the resulting NaNs natively via the Kalman filter, no
  extra imputation needed. Forecasts for both horizons come from one fit
  per (origin, series), cached across the two `predict_fn` calls the
  harness makes per origin (one per horizon) so nothing gets refit.
  Timing: ~0.1-0.2s/fit, full 25-origin x 30-series backtest runs in
  ~175s -- no need for the joblib parallelism I'd planned for.
  **Hit and fixed a real numerical bug**: potato/Kurnool at origin
  2025-09-12 produced a forecast of -3.8e26. Root cause: `ar.L1 = 1.041`
  (an explosive, non-stationary AR root -- `enforce_stationarity=False` is
  needed for fast/robust fitting across 750 fits but doesn't prevent this),
  compounded by 49% NaN in that series' reindexed window, and
  statsmodels reported `ConvergenceWarning: Maximum Likelihood optimization
  failed to converge`. Fixed by checking `res.mle_retvals['converged']` and,
  as a backstop (convergence flags aren't always reliable), rejecting any
  forecast that goes negative or exceeds 8x the trailing-30-day observed
  price level -- a 14-day-ahead commodity price forecast has no business
  doing either. Failed fits return None and that series/origin/horizon
  drops out of evaluation (same as any other missing-data case). Re-ran
  after the fix -- all commodities now report sane MAPEs:

  | commodity | h  | sarimax MAPE |
  |-----------|----|-------------:|
  | onion     | 7  | 13.4%        |
  | onion     | 14 | 14.1%        |
  | potato    | 7  | 11.8%        |
  | potato    | 14 | 12.9%        |
  | tur       | 7  | 4.7%         |
  | tur       | 14 | 4.3%         |

  Roughly ties naive on onion/tur, clearly beats naive on potato -- a
  believable benchmark, not sandbagged, which is what LightGBM needs to
  visibly beat.
- Built `models/features.py`: the date-safe replacement for the buggy
  price_lag_*/roll_*/ewma_*/momentum_30 columns (QUESTIONS.md #4), computed
  per (commodity, centre) group via pandas time-aware ops --
  `Series.shift(n, freq="D")` + reindex for lags (not `.shift(n)`, which is
  row count), `.rolling("Nd")` for rolling mean/std, `.ewm(halflife="Nd",
  times=idx)` for EWMA (parameterized by halflife in days rather than
  "span", since span-based decay has no well-defined meaning on
  irregularly-spaced data). Trusted pre-built columns (weather, festivals,
  calendar, cross-commodity, months_since_harvest) pass through untouched.
  Spot-checked against the known 671-day tur/Delhi gap: `price_lag_1`/
  `price_lag_7` are now correctly null there (previously silently wrong),
  `price_roll_mean_7` correctly falls back to just the single in-window
  point. Runs in ~0.28s on the full 99.5k-row history.
- Built `models/lgbm_model.py`: one pooled LightGBM model per (origin,
  horizon), trained across all 3 commodities x 10 centres at once with
  commodity/centre as categorical features (matches the "one global model"
  scaling story in CLAUDE.md's Q&A prep). Target built the same
  date-based-join way as `harness.make_supervised`. Features/model cached
  per origin so the horizon-14 `predict_fn` call reuses horizon-7's
  feature engineering. Full 25-origin backtest: ~54s.

  First result, default params (300 trees, lr=0.05, 31 leaves, L2):

  | commodity | h  | sarimax | lgbm  |
  |-----------|----|--------:|------:|
  | onion     | 7  | 13.4%   | 14.5% |
  | onion     | 14 | 14.1%   | 14.1% |
  | potato    | 7  | 11.8%   | 10.0% |
  | potato    | 14 | 12.9%   | 10.5% |
  | tur       | 7  | 4.7%    | 5.3%  |
  | tur       | 14 | 4.3%    | 4.4%  |

  Not a clean sweep -- LGBM clearly wins potato, roughly ties onion h14,
  loses narrowly on onion h7 and both tur horizons. Per the run rules
  ("if it doesn't beat SARIMAX, debug features... don't try new
  architectures") I tried three feature/config variants before accepting
  this: (1) `regression_l1` objective + more capacity (500 trees, 63
  leaves, subsample/colsample) -- macro-avg MAPE 9.86% (worse than
  default's 9.82%); (2) separate model per commodity instead of one pooled
  model (still pooling centres) -- 9.96% (worse); (3) time-based
  train/validation split with early stopping -- 10.73% (worse; the
  85/15 tail split on an expanding window is a small and often
  unrepresentative recent slice). The original default config was the best
  of all four tried, so it's what's committed -- no point shipping the more
  complex configs for a worse number. **Verdict: LGBM beats SARIMAX on
  macro-average MAPE, 9.82% vs 10.20% (~3.7% relative improvement)**, which
  is what the DONE criterion asks for, but it's not uniform across every
  cell. Two likely real explanations, both already flagged in QUESTIONS.md
  rather than worked around further: the missing arrivals join (#1) and
  the Phase-1 lag-feature bug at the source (#4, which I've routed around
  in my own features but couldn't fix upstream). A third, non-bug
  explanation worth recording for Q&A prep: tur's naive/SARIMAX baselines
  are already very strong at 7-14d (naive MAPE 4.3-4.7%) because tur is
  highly persistent short-term -- there may just be less short-horizon
  predictability left for any model to add on that particular
  commodity/horizon combination.
- Built `models/quantile_lgbm.py`: same pooled-across-series design and
  feature set as `lgbm_model.py`, three `LGBMRegressor(objective="quantile",
  alpha=...)` models per (origin, horizon) instead of one L2 model.
  `quantile_median_predict` is a predict_fn-shaped wrapper (alpha=0.5) that
  plugs into `harness.run_backtest` the same way every other model does;
  `coverage_at_origin` separately reports the [P10, P90] band against the
  actual, for calibration checking. Full 25-origin backtest: ~153s for the
  three-alpha fits, plus the 80%-coverage pass reuses the same per-origin
  cache so it added under a second.

  Median (P50) MAPE roughly matches the point LGBM model (as it should --
  same features, same pooling, different loss):

  | commodity | h  | lgbm (L2) | lgbm_q50 |
  |-----------|----|----------:|---------:|
  | onion     | 7  | 14.5%     | 14.7%    |
  | onion     | 14 | 14.1%     | 15.0%    |
  | potato    | 7  | 10.0%     | 8.5%     |
  | potato    | 14 | 10.5%     | 11.0%    |
  | tur       | 7  | 5.3%      | 5.3%     |
  | tur       | 14 | 4.4%      | 4.8%     |

  80% prediction-interval coverage (fraction of actuals landing inside
  [P10, P90] -- should be ~80% if well-calibrated): **71.3% at 7d, 74.1% at
  14d** overall -- bands a bit too narrow. Breaks down unevenly by
  commodity: potato (76.0%/81.8%) and tur (75.6%/74.1%) are close to
  target, onion is the weak spot (63.1%/64.9%). Not fixing via conformal
  calibration or similar -- that would drift toward "new architecture,"
  which the run rules say not to do for this pass. Flagging the onion
  under-coverage as a known limitation: the demo's "80% chance tur crosses
  X" framing (CLAUDE.md's example sentence) is reasonably trustworthy for
  tur/potato but the equivalent onion claim would currently overstate
  confidence.
- Built `models/run_backtest.py`: runs all five models (naive,
  seasonal_naive, sarimax, lgbm, lgbm_q50) through the same walk-forward
  harness end to end and writes `models/backtest_results.csv`. Rerunnable
  with `python models/run_backtest.py` from the `models/` directory, no
  arguments. Full run: ~322s (naive/seasonal_naive ~0.5s combined, sarimax
  130s, lgbm 45s, lgbm_q50 147s). Numbers matched the individual per-model
  runs above exactly -- good consistency check that nothing was order-
  dependent or cache-polluted across models.

  **DONE criterion met**: `models/backtest_results.csv` exists (commodity,
  horizon_days, n/mape/rmse for every model, plus
  `lgbm_improvement_vs_sarimax_pct`), and LightGBM's macro-average MAPE
  across all 6 (commodity, horizon) cells beats SARIMAX:

  | model          | macro-avg MAPE |
  |----------------|----------------:|
  | lgbm           | 9.82%           |
  | lgbm_q50       | 9.89%           |
  | sarimax        | 10.20%          |
  | naive          | 13.24%          |
  | seasonal_naive | 40.02%          |

  Per-cell `lgbm_improvement_vs_sarimax_pct`: potato -15.7%/-18.1% (LGBM
  clearly better), onion +8.5%/+0.3% (LGBM worse/~tied), tur +12.5%/+3.7%
  (LGBM worse). See QUESTIONS.md #5 for why I stopped tuning here rather
  than chasing a clean sweep, and #1/#4 for the two most likely paths to
  one (arrivals data, Phase-1 lag-feature fix) if the team wants to
  revisit before the demo.
