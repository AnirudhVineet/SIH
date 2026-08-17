# BUILD SPEC — PSS01: AI Price Intelligence & Buffer Stock Decision Support

> Paste this into Claude Code / Cursor as the project brief. Build in phase order. Stop at Phase 4 if time runs out — Phase 5 is optional.

---

## Context

Client: **Price Monitoring Division (PMD), Department of Consumer Affairs, Govt. of India.**

They collect daily retail + wholesale prices of 22 essential food commodities from 550 reporting centres across 34 states/UTs. When prices spike, they release buffer stock (pulses: gram/tur/urad/moong/masur, plus onion) through NAFED/NCCF, funded by the ₹10,000 crore Price Stabilization Fund.

Today they use ARIMA. It's linear, mostly univariate, ignores rainfall/arrivals/imports, gives no lead time on spikes, and stops at a number — it never says what to do.

**We build the thing that fixes all four of those.**

Deadline: demo Tuesday morning. Two days. Scope accordingly.

---

## The one-sentence product

An officer opens a map, clicks "Tur dal, Maharashtra," sees the 30-day price forecast, sees **why**, and gets told **how much buffer stock to release and when**.

---

## Hard constraints

- **3 commodities only:** tur dal, onion, potato. Do not attempt 22.
- **10–15 centres max.** Pick major ones (Delhi, Mumbai, Nagpur, Lucknow, Kolkata, Bengaluru, Patna, Ahmedabad).
- **Real data only.** No synthetic/generated series. Judges ask where data came from.
- **Must be deployed to a public URL** by Monday night. Do not demo from localhost.
- **Record a screen-capture video of the working demo** as backup against venue Wi-Fi failure.

---

## Data sources

| Source | URL | Pull | Priority |
|---|---|---|---|
| DoCA daily prices (via CEDA Ashoka) | `dca.ceda.ashoka.edu.in` | Daily retail + wholesale, per centre, 2015→now | **P0 — start here** |
| Agmarknet mandi arrivals | `agmarknet.gov.in` | Daily arrivals in tonnes per mandi | **P0 — strongest leading indicator** |
| data.gov.in mandi prices API | `data.gov.in` (free API key) | Daily min/max/modal wholesale by mandi | P1 |
| Open-Meteo Archive API | `archive-api.open-meteo.com` | Daily rainfall + temp for growing districts. No key needed | **P0** |
| Agriculture Ministry | sowing progress, Advance Estimates | Weekly kharif/rabi sown area | P2 |
| PIB press releases | `pib.gov.in` | Export bans, stock limits, MSP changes → event flags | P2 |
| DGCIS / Dept of Commerce | import volumes | Tur from Myanmar/Mozambique, urad from Myanmar | P2 |

Growing districts for weather pulls:
- **Tur:** Kalaburagi (KA), Latur/Akola (MH), Vidisha (MP)
- **Onion:** Nashik (MH), Kurnool (AP), Rajkot (GJ)
- **Potato:** Agra (UP), Hooghly (WB), Patna (BR)

---

## Repo structure

```
/data          raw + processed parquet
/ingest        one script per source, all idempotent
/features      feature builder
/models        baselines, lgbm, backtest harness
/decide        stress index + release optimizer
/api           FastAPI
/app           dashboard
/notebooks     exploration only, nothing production
```

---

## PHASE 1 — Data pipeline

Build `ingest/` scripts that each write a tidy parquet:

**`prices.parquet`** → `date, commodity, centre, state, retail_price, wholesale_price`
**`arrivals.parquet`** → `date, commodity, mandi, state, arrivals_tonnes, modal_price`
**`weather.parquet`** → `date, district, rainfall_mm, temp_max, temp_min`

Then `features/build.py` joins them into a single modelling frame keyed on `(date, commodity, centre)`.

**Features to compute:**

- Price lags: 1, 7, 14, 30, 90 days
- Rolling mean + std over 7, 30, 90 day windows
- EWMA (spans 7, 30)
- Momentum: `price / price.shift(30) - 1`
- **Arrivals**, arrivals lags (1/7/14), arrivals 30d rolling mean, arrivals YoY deviation
- Rainfall: 7d and 30d cumulative, deviation from 10-year normal for that day-of-year
- Retail–wholesale spread, and spread change over 7 days
- Day-of-year sin/cos, month, day-of-week
- Festival flags: Diwali, Navratri, Ramzan, Eid, Onam (±14 day windows)
- Cross-commodity: other commodities' prices same day (tur ↑ drives chana substitution)
- Days since last harvest for that commodity's calendar

**Non-negotiable:** handle missing centre-days. Reporting is patchy. Forward-fill up to 3 days, then impute from state-level median, and add an `is_imputed` flag column.

---

## PHASE 2 — Models

Build a **walk-forward backtest harness first**, before any model. Rolling origin: train on everything up to date *T*, predict *T+1 … T+30*, step forward 30 days, repeat across the last 2 years. Report MAPE and RMSE per commodity per horizon.

Then, in order:

1. **Naive + seasonal naive** — the floor
2. **SARIMAX** — this is what DoCA uses. This is your benchmark. You must beat it visibly.
3. **LightGBM, direct multi-horizon** — separate model per horizon h ∈ {1, 7, 14, 30}. This is the workhorse.
4. **Quantile LightGBM** — fit at α = 0.1, 0.5, 0.9 → gives the uncertainty band. Officers need "80% chance tur crosses ₹165," not a point estimate.
5. **Spike classifier** — binary: will price rise >8% in the next 14 days? Optimise precision/recall, not RMSE. **Report median lead time in days** — that's the headline metric.

**Output artifact:** a results table, saved as CSV and rendered on slide 3.

```
commodity | horizon | naive | sarimax | lgbm | improvement
tur       | 7d      | 8.1%  | 6.2%    | 3.9% | -37%
tur       | 14d     | 12.4% | 9.8%    | 6.1% | -38%
...
```

**If your LightGBM doesn't beat SARIMAX, the problem is almost always your features, not your model.** Check that arrivals actually joined correctly.

6. **SHAP** on the LightGBM. Then write a template that turns the top 3 SHAP values into a sentence:

> "Tur dal in Nagpur forecast to rise 8.4% over 14 days (₹158 → ₹171, 80% CI ₹164–₹179). Drivers: mandi arrivals down 34% YoY (+4.1%), below-normal August rainfall in Vidarbha (+2.2%), festival demand (+1.6%)."

That sentence is the product. Bureaucrats don't read SHAP plots.

---

## PHASE 3 — Dashboard

Four screens. Streamlit is fine and fast; Next.js only if you have a strong frontend person with spare time. **A finished Streamlit app beats a half-built React app.**

**1. Map** — India, states shaded red/amber/green by Price Stress Index. Click a state → drill in.

**2. Commodity view** — historical price line, forecast line extending forward, shaded P10–P90 band. Toggle horizon 7/14/30d. Overlay mandi arrivals on a second axis.

**3. Why panel** — horizontal bar chart of top SHAP drivers + the generated sentence above it.

**4. Action panel** — the recommendation, plus a what-if slider.

Design notes: dark or neutral background, one accent colour, real numbers everywhere. Two hours of visual polish disproportionately affects perceived quality. Do not ship default Streamlit grey.

---

## PHASE 4 — Decision layer (this is the differentiator)

Most teams stop at a forecast chart. Do not stop there — the PS title says *Buffer Stock Decision Support*.

**Price Stress Index (0–100)** per commodity × state. Weighted composite of:
- forecast price vs. trigger threshold
- forecast volatility (P90 − P10 spread)
- retail–wholesale spread widening vs. peer centres
- arrivals decline vs. 3-year seasonal norm
- current buffer stock cover ratio

**Release optimizer.** Small LP in PuLP or OR-Tools:

```
minimize:  Σ (expected price deviation from target)  +  λ · (transport cost)
subject to: Σ release_i ≤ available_stock
            release_i ≥ 0
            release_i ≤ state_absorption_capacity_i
```

Output a table an officer could sign: *state, quantity, release date, expected price impact*.

**Procurement side** — also flag when to *buy* (harvest lows) to rebuild the buffer. Closes the loop, shows you understand the whole PSF cycle rather than half of it.

**What-if simulator** — sliders for quantity, state, and timing. Show the counterfactual price path bending against the do-nothing baseline. **This is the demo climax. Rehearse it.**

**PDF report generator** — one button, WeasyPrint or ReportLab, produces a one-page brief: current prices, forecast, drivers, recommendation. This is Expected Solution #5, stated verbatim in the PS.

---

## PHASE 5 — Optional. Pick at most two.

- **Time machine** — replay the system over the Aug–Dec 2023 onion crisis or the 2023–24 tur run-up. Show: *"would have flagged this N days earlier, recommended release on date X."* Cheapest high-impact thing on this list. **Do this one.**
- **Hoarding detector** — flag centres where retail–wholesale spread widens abnormally vs. peer centres. ~15 lines of code, maps directly onto a real government lever.
- **Contagion map** — animate a Nashik onion shock propagating to Mumbai (~5 days) and Delhi (~9 days). Lagged cross-correlation is enough; no GNN needed.
- **News event scraper** — parse PIB for export bans / stock limits → binary event features.
- **Multilingual WhatsApp alerts** to state officials.

---

## Stack

```
python 3.11, polars or pandas, duckdb
statsmodels (SARIMAX), lightgbm, shap
pulp or ortools
fastapi
streamlit  (or next.js + tailwind + recharts + leaflet)
weasyprint
docker → render / railway / hf spaces
```

---

## Definition of done

- [ ] Real data from ≥3 sources, ingest scripts rerunnable end-to-end
- [ ] Walk-forward backtest table, LightGBM clearly beating SARIMAX
- [ ] Forecasts with P10/P50/P90 bands for 3 commodities
- [ ] Spike classifier with a stated lead time in days
- [ ] Plain-English driver explanation generating correctly
- [ ] All four dashboard screens working
- [ ] Release recommendation producing a signable table
- [ ] What-if slider visibly moving the price path
- [ ] PDF export working
- [ ] Deployed to a public URL
- [ ] Backup demo video recorded

---

## Do not

- Build a generic price predictor with no buffer-stock layer. The words are in the PS title.
- Use synthetic data.
- Skip the SARIMAX baseline. An unbenchmarked model is an unproven model.
- Promise a transformer and demo a linear regression.
- Show only SHAP plots without the sentence.
- Ship default Streamlit styling.
- Run over 10 minutes. Time it with a stopwatch, three times.

---

## Demo script (10 min, rehearse Monday night)

| Time | Beat |
|---|---|
| 0:00–1:00 | The problem. DoCA uses ARIMA. It misses spikes and never says what to do. |
| 1:00–2:00 | Map. Red states. "Tur is under stress in Maharashtra right now." |
| 2:00–4:00 | Drill in. Forecast with uncertainty band. State your MAPE vs. ARIMA out loud. |
| 4:00–5:30 | Why panel. Read the generated sentence aloud. |
| 5:30–7:00 | Recommendation table. Then the what-if slider. **Slow down here.** |
| 7:00–8:30 | Time machine — the 2023 onion crisis, caught N days early. |
| 8:30–10:00 | Scale story: global model, 22 commodities × 550 centres. PDF export. Close. |

---

## Q&A prep

Have crisp answers ready. The starred one is what separates teams that thought about this from teams that didn't.

- *Why beat ARIMA?* → Point at the table. Never answer this qualitatively.
- *What's your MAPE, at what horizon?* → Know it cold, per commodity.
- *Where's the data from?* → Name the portals. You already pulled them.
- *550 centres × 22 commodities = 12,100 series. Scale?* → One global model, not 12,100 models. Hierarchical reconciliation so state/national forecasts add up.
- *Missing reports from centres?* → Forward-fill → state-median impute → confidence flag on the forecast.
- ***If you recommend a release and the price falls, was your forecast wrong?*** → No. We forecast the **no-intervention counterfactual**. The intervention enters as a treatment variable. Evaluation is against the policy-adjusted outcome.
- *What if the model is wrong and stock is released unnecessarily?* → Human-in-the-loop. Decision **support**, never automation. Probabilistic outputs with explicit confidence. The officer signs, not the model.
- *Running cost?* → Modest. No GPU at inference. Retrain weekly on CPU.

---

## PPT — 6 slides, template already provided, export to PDF, delete slide 7

| # | Slide | Content |
|---|---|---|
| 1 | Title | PSS01 · full PS title · Theme: Agriculture, FoodTech & Rural Development · Software · team ID + name |
| 2 | Idea | One diagram: **Data → Forecast → Explain → Decide**. Call out the innovation explicitly: the decision + optimizer layer, and retrospective validation. |
| 3 | Technical Approach | Stack, tiered model diagram, prototype screenshots, **and the backtest table**. Real numbers here separate you from the field. |
| 4 | Feasibility & Viability | Every source free and government-published — list them. Risks: patchy centre reporting, regime breaks after policy shocks, sparse history for new commodities. Mitigations for each. |
| 5 | Impact & Benefits | 22 commodities ≈ 26% of CPI basket. ₹10,000cr PSF deployed better. Consumer welfare, farmer price realisation on the procurement side, less wastage from better-timed releases. |
| 6 | References | DoCA PMD methodology, PSF guidelines, CEDA portal, Agmarknet, forecasting papers. |

Template rules: max 6 slides including title, no paragraphs — points/diagrams/infographics only, don't alter the template's idea-detail pointers, submit as PDF.

---

## Team split (6 people)

| Role | Owns |
|---|---|
| Data × 2 | Phase 1 + 2. One on ingest, one on features + models. |
| Frontend × 1 | Phase 3. Starts Sunday evening against fake JSON, swaps in the real API Monday. |
| Decision × 1 | Phase 4 + the time machine case study. |
| Deck × 1 | PPT, narrative, references. Also writes the driver-sentence templates. |
| Integrator × 1 | Deployment, the demo script, runs rehearsals, owns the backup video. |

---

## Timeline

**Sunday:** submit the PS form (closes 8pm). Lock scope. Phase 1 complete. One commodity through Phase 2 with a number you trust.

**Monday AM:** Phase 2 complete across all three commodities. SHAP + sentences working.

**Monday PM:** Phase 3 + 4. Time machine. Deploy.

**Monday night:** freeze code. Record backup video. Rehearse three times with a stopwatch.

**Tuesday 9:30am:** registration, Room 1105, New Building Seminar Hall. Team leader must attend.
