# Deploy image for the PSS01 dashboard.
#
# The app reads pre-built artifacts from app/data/ and never trains a model at
# runtime, so this image needs no build step beyond pip install and starts in
# seconds. Works on Render, Railway, Fly, Cloud Run, or a Hugging Face Space
# with sdk: docker.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Only the runtime deps -- the modelling stack (lightgbm/shap/statsmodels) is
# needed to *regenerate* artifacts, not to serve them, and pulling it in would
# roughly triple the image for no runtime benefit.
COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY .streamlit/ ./.streamlit/
COPY app/ ./app/
COPY decide/ ./decide/
COPY models/backtest_results.csv models/spike_results.csv \
     models/spike_results_by_commodity.csv models/quantile_coverage.csv ./models/

# Most PaaS hosts inject $PORT; HF Spaces expects 7860. Default to 7860 and let
# the platform override.
ENV PORT=7860
EXPOSE 7860

CMD streamlit run app/dashboard.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
