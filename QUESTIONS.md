# Open questions for the team (Phase 2 work)

Logged by the unattended Phase-2 agent. Not decided unilaterally — flagging and
continuing on other work per the run rules.

---

## 1. No arrivals data exists — but the debug playbook assumes it does

`ingest/prices.py`'s own docstring says arrivals was never wired in (no
working DoCA retail source was found; the script pulls Agmarknet **wholesale**
prices only). `data/processed/modelling_frame.parquet` has zero
arrivals-related columns. `features/build.py` as currently committed only
joins prices + weather — no arrivals join exists anywhere in the repo to
"check."

The Phase-2 task instructions I'm operating under say: *"If it doesn't beat
SARIMAX, debug features (check the arrivals join), don't try new
architectures."* There is no arrivals join to check. Ingest is closed for
this run, so I can't add the source myself.

**Impact:** if LightGBM underperforms SARIMAX, my only lever is the features
that do exist (lags, rolling stats, EWMA, momentum, weather, festivals,
cross-commodity prices, months-since-harvest) — arrivals (the single
strongest leading indicator per CLAUDE.md) is off the table until Phase 1
adds it.

**Question for the team:** is a Phase-1 follow-up to wire in an arrivals
source (Agmarknet arrivals-in-tonnes, data.gov.in mandi API) planned before
the demo? If LightGBM needs it to convincingly beat SARIMAX, that's a
blocker worth prioritizing.

## 2. `retail_price` is entirely null

Every row in `prices.parquet` / `modelling_frame.parquet` has
`retail_price = null` (documented in `ingest/prices.py`'s header: no working
retail-by-centre source was found, so the pipeline pulls wholesale mandi
prices only, keyed by a "benchmark market" proxy for each state/commodity
rather than a true DoCA retail centre).

**Decision I'm making to keep moving (flagging, not asking permission,
since it's a mechanical necessity to have any target at all):** all Phase 2
models are trained/evaluated against **`wholesale_price`**, not
`retail_price`. This should be called out explicitly in the demo — the
"₹158 → ₹171" style sentences in CLAUDE.md's example are retail prices, but
today's real pipeline output is wholesale-mandi-price forecasts. Someone
should decide whether that's acceptable to present as-is, or whether the
narrative needs to say "wholesale" throughout, before Monday's demo prep.

## 3. `features/build.py` as committed doesn't match `modelling_frame.parquet` as committed

The committed `features/build.py` (61 lines) only does a prices+weather join
and produces ~10 columns. The committed `modelling_frame.parquet` has 39
columns, including lags, rolling stats, EWMA, momentum, rainfall
deviation-from-normal, festival flags, months-since-harvest, and
cross-commodity prices — none of which the current script computes. Both
were part of the same initial commit, so the parquet was evidently built by
an earlier/fuller version of the feature code that isn't in the repo now.

**I'm treating the parquet as the source of truth and building Phase 2 on
top of it as-is**, since regenerating it is Phase 1/ingest territory and out
of scope for this run. Flagging so whoever owns Phase 1 next knows the
checked-in `build.py` needs to be reconciled with what actually produced the
data on disk — right now a clean `git clone` + rerun of the pipeline would
NOT reproduce `modelling_frame.parquet`.

## 4. Found the likely real reason LGBM might underperform SARIMAX -- and it isn't arrivals

While building the walk-forward harness I found that `modelling_frame.parquet`
series have real reporting gaps (not a filled daily grid -- see PROGRESS.md).
Checking how the pre-built `price_lag_*`, `price_roll_mean/std_*`,
`price_ewma_*`, and `momentum_30` columns behave across a gap:

```
tur / Delhi, gap of 671 days (2020-03-03 -> 2022-01-03):
  wholesale_price on 2022-01-03 = 3000.0
  price_lag_1  = 4331.0   <- identical to price_lag_7 below, and to the
  price_lag_7  = 4331.0      last price actually observed on 2020-03-03,
                              671 days earlier, not "yesterday"
  price_roll_mean_7 = 4140.86  <- a "7-day" window that actually spans
                                   hundreds of real days across the gap
```

These columns were evidently computed with row-positional
`.shift(n)`/`.rolling_mean(n)` over the gappy series (n rows back), not a
date-aware `n days back`. Every one of these price-derived columns is
silently wrong immediately after any gap, large or small. By contrast,
`price_onion_same_state` / `price_potato_same_state` (cross-commodity,
same-date join) checked out correct against the actual same-day price, and
the calendar/festival/weather columns are date-derived, not
row-shifted, so they're unaffected.

**This is likely the real reason a naive LightGBM would look weak against
SARIMAX** -- lag/rolling features are exactly what LightGBM leans on most,
and roughly a quarter of rows are imputed short gaps (plus a handful of
long real gaps on top of that), so a meaningful share of training rows have
corrupted lag inputs.

**What I did about it (Phase 2 territory, not touching ingest):** rather
than use the pre-built lag/rolling/EWMA/momentum columns, `models/lgbm_model.py`
recomputes them itself, directly from `wholesale_price` + `date`, using the
same date-based-join technique as `harness.make_supervised` (see
`models/features.py`). Everything else (weather, festivals, calendar,
cross-commodity, months_since_harvest) is taken from the parquet as-is.

**Question for the team:** worth fixing at the source
(`features/build.py`) once Phase 1 picks back up, so every future consumer
of `modelling_frame.parquet` doesn't need to route around it the way I just
did. Flagging rather than fixing there myself since that file is Phase 1 /
ingest territory for this run.

## 5. `months_since_harvest` granularity vs. spec

CLAUDE.md's feature list asks for "days since last harvest." The committed
column is `months_since_harvest` (integer, monthly granularity). Not fixing
this myself (Phase 1 territory) — noting in case it matters for feature
quality; day-level granularity would give the model a smoother signal near
harvest transitions.
