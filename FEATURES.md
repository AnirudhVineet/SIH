# FEATURES.md — everything the system is built on and does

Reference doc: what data it's trained on, every input feature and why it's
there, what each model predicts, how the decision layer turns a forecast
into a recommendation, and what each dashboard screen shows. `README.md` is
the pitch; this is the specification behind it.

---

## 1. Scope

- **3 commodities:** tur (dal), onion, potato.
- **10 benchmark centres**, one per state: Delhi, Mumbai, Nagpur, Lucknow,
  Kolkata, Bengaluru, Patna, Ahmedabad, Bhopal, Kurnool. "Centre" means the
  highest-volume real Agmarknet market for that commodity in each state
  (there is no true DoCA retail-centre feed — see §2).
- **History:** 2015-01-01 to 2025-11-05 (~100k rows across all
  commodity × centre series). Freshest cached raw data actually reaches
  April 2026; the processed series currently stops in Nov 2025 due to a
  benchmark-market selection quirk, not a source limit (see `README.md`
  Known gaps).
- **Forecast horizons:** 7 and 14 days ahead. (The original spec's 1/7/14/30
  was narrowed; there is no 30-day or recursive/autoregressive model.)
- **Prediction target:** `wholesale_price` — ₹ per quintal (100 kg), Agmarknet
  modal price. Retail price is not available in any source this project
  could reach (all rows null), so wholesale is the only trained/evaluated
  target throughout.

## 2. Data sources

| Source | What it provides | Notes |
|---|---|---|
| **Kaggle mirror of Agmarknet** (`khandelwalmanas/daily-commodity-prices-india`) | Daily wholesale mandi prices (min/max/modal), 2015–2026, per state/market/commodity | Used because CEDA's mirror is DNS-dead, CEDA's own API is JWT-gated Agmarknet-only, DoCA's fcainfoweb is CAPTCHA-gated, and data.gov.in's live resource has no history. Cached per-year in `data/raw/prices/`, rerunnable with `--refresh`. |
| **Open-Meteo Archive API** | Daily rainfall (mm) and min/max temperature for each commodity's growing districts, no key required | Cached per-district in `data/raw/weather/`. |
| **Census 2011 state populations** | Used to derive absorption-capacity estimates for the release optimizer | Last full published census; a scaling constant (not the population itself) is the assumption. |
| **Great-circle centre coordinates** | Genuine lat/lon for the 10 centres and a single assumed Delhi depot | Used to derive transport-cost estimates. |

**Growing districts** (weather is pulled here, mapped to commodity not centre):
- **Tur:** Kalaburagi (KA), Latur, Akola (MH), Vidisha (MP)
- **Onion:** Nashik (MH), Kurnool (AP), Rajkot (GJ)
- **Potato:** Agra (UP), Hooghly (WB), Patna (BR)

**Not available anywhere in the pipeline** (checked directly against the raw
files, not assumed): mandi arrivals in tonnes (`Arrival_Date` in the Kaggle
mirror is a date string, not a quantity), retail prices, buffer-stock
levels, and any historical log of past PSF releases. These absences are
handled by omission or by an explicitly labelled estimate — never by
inventing a number. See `QUESTIONS.md`.

## 3. Missing-data handling

~26% of rows are gap-filled before modelling: forward-fill up to 3 days,
then state-level median imputation, with an `is_imputed` flag carried
through. Backtests and spike labels are always scored against **real,
non-imputed** prices only — a model may train on imputed rows, but accuracy
numbers never credit or blame it for a fabricated value.

Lag/rolling/EWMA features are computed **calendar-aware**, not by row
position — `Series.shift(n, freq="D")`, `.rolling("Nd")`,
`.ewm(halflife="Nd", times=...)` in `models/features.py`. This matters
because reporting gaps are real and sometimes long (one tur/Delhi gap is 671
days); a naive row-positional shift would silently treat "671 days ago" as
"yesterday." This was found as a live bug in an earlier pre-built feature
set and fixed — see `QUESTIONS.md` #4.

## 4. Input features

### Passed through as-is (already date-correct)

| Feature | What it is |
|---|---|
| `rainfall_mm`, `temp_max`, `temp_min` | Daily weather at the commodity's growing districts |
| `rainfall_7d_sum`, `rainfall_30d_sum` | Cumulative rainfall over the trailing 7/30 days |
| `rainfall_dev_from_normal` | Deviation from the day-of-year's historical normal rainfall |
| `month`, `dow`, `doy_sin`, `doy_cos` | Calendar position (month, day-of-week, and sin/cos-encoded day-of-year so the model sees seasonality as continuous, not a hard year-boundary jump) |
| `festival_diwali`, `festival_navratri`, `festival_eid`, `festival_onam` | ±14-day window flags around major demand-driving festivals |
| `months_since_harvest` | Months since that commodity's last harvest (monthly granularity — spec asked for daily; see `QUESTIONS.md` #7) |
| `price_tur_same_state`, `price_onion_same_state`, `price_potato_same_state` | Same-day price of the *other* two commodities in the same state (captures substitution effects, e.g. tur ↑ driving chana/onion demand shifts) |

### Engineered fresh per training window (date-safe, recomputed from raw price + date)

| Feature | What it is |
|---|---|
| `price_lag_1/7/14/30/90` | Price exactly N calendar days ago (null if that date has no real report) |
| `price_roll_mean_7/30/90`, `price_roll_std_7/30/90` | Rolling mean/std over a true N-calendar-day window |
| `price_ewma_7/30` | Exponentially-weighted moving average, decaying by elapsed calendar time (halflife N days), not by row count |
| `momentum_30` | `price / price_lag_30 − 1` — 30-day % change |

### Categorical

`commodity`, `centre` — every model is one pooled model across all
commodities and centres (not one model per series), with these as
categorical features. This is the scaling argument for going from this
prototype to 22 commodities × 550 centres: one model, not 12,100.

### Deliberately absent

Mandi arrivals (tonnes), retail price, retail–wholesale spread — the
strongest candidate leading indicators per the original brief — are not in
any reachable source, so they are not features. This is flagged, not hidden;
it's the single biggest available accuracy lever if a source is found.

## 5. Models

Built in order, each required to beat the one before it on the same
walk-forward harness (25 rolling origins, 30-day steps, over the last two
years of real data):

| Model | What it predicts | How | Macro-avg MAPE |
|---|---|---|---|
| **Naive** | Flat-forward last price | Floor baseline | 13.24% |
| **Seasonal naive** | Same calendar date last year | Floor baseline | 40.02% (loses to naive — these series are volatile enough that short-term persistence beats a year-old echo) |
| **SARIMAX** | Point forecast, order (1,1,1), no seasonal term | Per (commodity, centre), fit fresh at each origin — this is DoCA's current real-world method, so it's the benchmark to beat, not a strawman | 10.20% |
| **LightGBM** | Point forecast (P50-equivalent) | One pooled model per horizon (7d, 14d), direct multi-horizon (not recursive), trained on the engineered features above | **9.82%** |
| **Quantile LightGBM** | P10/P50/P90 forecast band | Three `objective="quantile"` fits per horizon, monotone-rearranged so bands never invert, then **conformally calibrated** (split conformal / CQR, per-commodity, calibrated on a held-out trailing 180-day window) so the stated 80% interval actually covers ~80% of real outcomes | P50 ≈ LightGBM; 80% band coverage 77.8% overall (was 70.5% before calibration) |
| **Spike classifier** | Binary: will this series rise >8% within the next 14 days? | Binary LightGBM, decision threshold 0.30 (empirical max-F1) | 50% precision / 78% recall; **median 7-day lead time**, 87.4% of spike episodes (505/578) caught |

LightGBM beats SARIMAX on the macro-average but not in every cell — it wins
potato clearly, roughly ties onion at 14d, and loses narrowly on onion 7d
and both tur horizons. The two most likely fixable causes (no arrivals data,
a lag-feature bug in an earlier pipeline stage routed around but not fixed
at the source) are documented in `QUESTIONS.md` #1 and #4, alongside a third,
non-bug explanation: tur is highly persistent short-term, so its naive/
SARIMAX baselines are already very strong (4.3–4.7% MAPE) and there may
genuinely be little predictability left on the table there.

### SHAP driver explanations

Every LightGBM point forecast carries its top SHAP drivers (pool of 8),
re-referenced against that commodity's own recent mean (not a global
background set) so "driver above/below norm" means something concrete.
`describe_driver()` in `models/build_dashboard_artifacts.py` turns each into
a human clause (e.g. "30-day momentum +10.7%", "rainfall below seasonal
normal in growing districts"), and `build_sentences()` assembles the
officer-facing sentence from up to 4 of them, picked by
`_select_sentence_drivers()` to guarantee one each from price trend,
weather, and season/festival where the data actually supports it — instead
of naively taking the largest-magnitude 3, which in practice tends to
surface three correlated price-trend variants and drop real weather/season
signal. (Market arrivals and production estimates are not a category here
because no such series exists anywhere in this pipeline — see "Deliberately
absent" above.)

> "Onion in Nashik forecast to rise 8.4% over 14 days (₹1,580 → ₹1,713/quintal,
> 80% band ₹1,540–₹1,890). Main drivers vs this commodity's recent norm:
> 30-day momentum +10.7%, rainfall below seasonal normal in growing
> districts, festival demand in window. Probability of a >8% rise within
> 14 days: 62%."

CLAUDE.md's framing is taken literally here: this sentence, not the SHAP
plot, is the actual product — "bureaucrats don't read SHAP plots."

## 6. Decision layer (`decide/`)

Everything below is built from **real fitted-model output only** — no
sourced buffer-stock, capacity, or transport feed exists, so every input
that isn't a model output is an explicit, labelled estimate (never silently
invented), carried through to the UI and the PDF wherever it's shown.

### Price Stress Index (0–100), `decide/stress.py`

Per (commodity, centre), three weighted components:

| Component | Weight | What it measures |
|---|---|---|
| Forecast level | 45% | 14-day P50 vs that centre's **own** trailing 1-year median (real, non-imputed prices only) — comparable across centres trading at very different price levels |
| Spike probability | 35% | Classifier P(>8% rise within 14 days), used directly |
| Band width | 20% | (P90−P10) as % of price — a wide band means the model itself is unsure, which is reason for an officer to look regardless of the point forecast |

Bands: **High** ≥60, **Elevated** ≥45, **Moderate** ≥30, **Low** below.
Feeds the map colours and the release optimizer's candidate list.

### Procurement (buy-side) signal, `decide/procurement.py`

Per (commodity, centre), 0–100, answering "is now a harvest-low buying
window to rebuild the buffer?" — **multiplicative**, not additive, of:

- **Discount component:** current price vs the same trailing 1-year median
  the stress index uses (shared baseline, so a centre can't show High
  release-stress and an open procurement window off two different
  definitions of "normal").
- **Harvest-proximity component:** decays from 1.0 (just harvested) to 0 by
  3 months out, using the real `months_since_harvest` feature.

Multiplying (rather than adding) means a large discount alone, ten months
from harvest, can't clear the "Open" threshold — that pattern is a
demand/oversupply signal, not a harvest low, and the multiplicative form
caps its ceiling at 15% of the full score. Bands: **Open** ≥60, **Watch**
≥35, **Closed** below.

### Release optimizer, `decide/optimizer.py` (PuLP linear program)

```
maximise   Σ (stress_i − λ · cost_i / cost_max) · release_i
subject to Σ release_i ≤ available_stock
           0 ≤ release_i ≤ absorption_capacity_i
```

Only centres at stress ≥30 qualify. Each tonne is worth its centre's stress
score, discounted by transport distance (both normalised to the same
stress-point scale so λ is directly interpretable as "how many stress points
the longest haul is worth giving up"). Absorption capacity is split across
multiple centres sharing one state so the LP can't spend a state's capacity
twice. Output is a signable table: centre, state, stress, release tonnes,
% of capacity, distance, transport cost, and a release date (`as_of` +
an operator-set dispatch-lead slider, 0–14 days).

**Estimate labels carried through the UI and PDF:** available stock is an
operator input (not sourced); absorption capacity is a Census-2011
population-derived estimate; transport cost is real great-circle distance ×
a flat assumed per-tonne-km rate from a single assumed Delhi depot. The
distances are genuine; the rate and single-depot assumption are not.

### What-if simulator, `decide/whatif.py`

Sliders: release quantity (defaults to the LP's own allocation for the
selected centre), state/centre, and dispatch timing (the release-date
slider above) — the three CLAUDE.md asks for. The do-nothing baseline is a
straight-line interpolation through the three real forecast points (today,
7-day P50, 14-day P50) — not illustrative. The with-release line is:

```
impact_pct(t) = 6% × min(release/capacity, 1) × ramp(t − release_day)
```

ramping from 0 to full strength over 5 days after dispatch. This impact
model is an **explicit, labelled illustrative assumption** — no source this
project ingests logs historical PSF releases, so there is nothing to fit a
real price-elasticity against. It exists to show the *shape* of the
intended effect (more release, or release into a thinner market, bends
price down more; the effect isn't instant) so an officer can reason about
timing and quantity — it is not a rupee forecast.

### PDF brief, `decide/report.py` (ReportLab)

One-page officer-signable brief: current price, forecast + band, top SHAP
drivers and the generated sentence, the release recommendation table with
release date, the procurement window status, and every estimate-provenance
caveat printed on the page (not just shown in the UI), so they travel with
the document. ReportLab was chosen over WeasyPrint to avoid a GTK/Pango
system-library dependency in the deploy image.

## 7. Time machine — 2023 onion crisis replay (`models/time_machine.py`)

Retrospective validation, not a backtest metric: replays the real Aug–Dec
2023 onion price crisis (₹819 → ₹6,000/quintal) with the spike classifier
**retrained at each 30-day origin on only the data available at that date** —
strictly no hindsight. Result: all 9 tracked onion centres alerted in the
first days of August 2023, a median **88 days** before the late-October
peak. Two lead times are reported because they answer different questions:
median 3 days ahead of each centre's own first 8% breach (the classifier's
literal target), vs median 88 days of runway before the actual peak (the
operationally useful figure for an officer). Stated caveat: one centre
(Lucknow) fired one day *after* its own breach, and precision inside the
crisis window was 52%.

## 8. Dashboard screens (`app/dashboard.py`)

1. **National map** — India shaded by Price Stress Index, red/amber/green;
   click a state to drill in.
2. **Commodity view** — historical price line + forecast extending forward,
   shaded P10–P90 band, 7/14-day horizon toggle, configurable history
   window (default 90 days, so a short forecast isn't an invisible sliver
   against a multi-year history).
3. **Why** — horizontal SHAP driver bars (signed, blue/red diverging) with
   the plain-English generated sentence above them.
4. **Action** — the release optimizer's signable table, the procurement
   (buy-side) panel with months-to-next-harvest, the live what-if sliders
   and bending price-path chart, and one-click PDF export.
5. **Time machine** — the 2023 onion crisis replay, shown as alert dates vs
   the real price path and the peak.

Design: dark theme, one accent colour (`#3987e5`), no default Streamlit
grey, no dual-axis charts (arrivals — the natural second axis — don't exist
in this dataset, and a second y-scale would invent a correlation that isn't
there).

## 9. Known limitations

See `README.md`'s "Known gaps" and `QUESTIONS.md` for the full list with
evidence. Summary: no arrivals data, no retail prices, no sourced
buffer-stock/capacity/transport data (worked around with labelled
estimates), forecasts capped at 14 days with no recursive extension, the 7-day
uncertainty band still under-covers slightly (75.1% vs 80% target), and the
live price series currently runs through 2025-11-05 rather than the present
day (a fixable data-pipeline artifact, not a hard source limit — see
README).
