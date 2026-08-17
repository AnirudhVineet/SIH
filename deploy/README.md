# Deploying the dashboard

The app serves pre-built artifacts from `app/data/` and never fits a model at
request time, so the deploy image is small (~1.2 MB of app payload) and starts
in seconds. No GPU, no model training on the host.

## Option A — Hugging Face Spaces (fastest, no Docker locally)

```bash
hf auth login                 # paste a WRITE token from hf.co/settings/tokens
python deploy/deploy_hf.py
```

Non-interactive instead of `hf auth login`:

```bash
export HF_TOKEN=hf_xxx        # write-scoped
python deploy/deploy_hf.py
```

Creates/updates a Space using the **Docker SDK** — the same `Dockerfile` any
other host would use, so there is only one deploy definition to keep working.
First push builds the image; allow 2–4 minutes. Prints the public URL when done.

Custom target:

```bash
python deploy/deploy_hf.py --space yourname/pss01 --private
```

## Option B — Render / Railway (connect the GitHub repo)

Both auto-detect the `Dockerfile`. In the web UI:

| Setting | Value |
|---|---|
| Repository | `AnirudhVineet/SIH` |
| Environment | Docker |
| Dockerfile path | `./Dockerfile` |
| Port | `7860` (or leave blank — the image reads `$PORT`) |
| Health check path | `/_stcore/health` |
| Instance | Free tier is sufficient |

No environment variables or secrets are required — the app reads only committed
artifacts. `reference/secrets.json` is gitignored and is **not** needed at
runtime (it is only used by the Phase 1 ingest scripts).

## Option C — any Docker host

```bash
docker build -t pss01 .
docker run -p 7860:7860 pss01
# http://localhost:7860
```

## What gets deployed

`app/` (dashboard + prebuilt artifacts + India GeoJSON), `decide/` (stress
index, LP optimizer, PDF report), `.streamlit/config.toml`, the four result
CSVs the app quotes accuracy from, `Dockerfile`, `requirements-app.txt`.

Deliberately **excluded**: `models/` code, `ingest/`, `features/`, `data/raw/`,
the 100k-row modelling frame, and the modelling stack itself
(lightgbm/shap/statsmodels/scikit-learn). Those are needed to *regenerate*
artifacts, not to serve them, and including them would roughly triple the image
for no runtime benefit.

## Refreshing the data on a deployed instance

Artifacts are baked into the image, so a data refresh is a rebuild:

```bash
cd models
python run_backtest.py
python run_spike_backtest.py
python build_dashboard_artifacts.py
cd .. && git commit -am "refresh artifacts" && python deploy/deploy_hf.py
```
