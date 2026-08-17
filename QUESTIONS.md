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

## 4. `months_since_harvest` granularity vs. spec

CLAUDE.md's feature list asks for "days since last harvest." The committed
column is `months_since_harvest` (integer, monthly granularity). Not fixing
this myself (Phase 1 territory) — noting in case it matters for feature
quality; day-level granularity would give the model a smoother signal near
harvest transitions.
