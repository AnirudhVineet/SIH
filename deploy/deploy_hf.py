"""
One-command deploy of the dashboard to a Hugging Face Space.

    python deploy/deploy_hf.py                      # uses cached login or $HF_TOKEN
    python deploy/deploy_hf.py --space you/pss01    # explicit target

Auth, in order of preference:
    hf auth login            (interactive, token cached; nothing lands in shell history)
    export HF_TOKEN=...      (non-interactive/CI)

Uses the Docker SDK so the Space runs the exact same image definition as
Render/Railway/Fly would -- one deploy path to keep working, not two.

Only the *serving* surface is uploaded (app/, decide/, .streamlit/, the four
result CSVs, Dockerfile, requirements-app.txt). The modelling code, raw data
and the 100k-row modelling frame stay out: the dashboard reads pre-built
artifacts from app/data/ and never fits a model at request time.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, whoami

ROOT = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: PSS01 Price Intelligence & Buffer Stock Support
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PSS01 — AI Price Intelligence & Buffer Stock Decision Support

Forecasts wholesale mandi prices for tur, onion and potato across 10 benchmark
centres, explains each forecast in plain English, flags price spikes with a
median **7 days** of lead time, and produces a buffer-stock release plan.

Built for the Price Monitoring Division, Department of Consumer Affairs.

**Accuracy (walk-forward backtest, 25 rolling origins over 2 years):**
LightGBM macro-avg MAPE **9.82%** vs SARIMAX **10.20%** (the current ARIMA-class
approach) and naive **13.24%**.

Source: https://github.com/AnirudhVineet/SIH
"""

# Paths uploaded to the Space, relative to repo root.
INCLUDE_FILES = [
    "Dockerfile",
    "requirements-app.txt",
    ".streamlit/config.toml",
    "models/backtest_results.csv",
    "models/spike_results.csv",
    "models/spike_results_by_commodity.csv",
    "models/quantile_coverage.csv",
]
INCLUDE_DIRS = ["app", "decide"]
SKIP_SUFFIXES = (".pyc",)
SKIP_DIRS = {"__pycache__", "tests"}


def stage(tmp: Path) -> None:
    (tmp / "README.md").write_text(SPACE_README, encoding="utf-8")

    for rel in INCLUDE_FILES:
        src = ROOT / rel
        if not src.exists():
            sys.exit(f"missing required file: {rel}")
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for d in INCLUDE_DIRS:
        for src in (ROOT / d).rglob("*"):
            if src.is_dir() or src.suffix in SKIP_SUFFIXES:
                continue
            if SKIP_DIRS & set(src.relative_to(ROOT).parts):
                continue
            dst = tmp / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="target Space id, e.g. username/pss01-price-intelligence")
    ap.add_argument("--name", default="pss01-price-intelligence")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    try:
        user = whoami(token=token)["name"]
    except Exception:
        sys.exit(
            "Not authenticated.\n"
            "  Interactive:     hf auth login\n"
            "  Non-interactive: set HF_TOKEN to a write token from\n"
            "                   https://huggingface.co/settings/tokens"
        )

    space_id = args.space or f"{user}/{args.name}"
    api = HfApi(token=token)

    print(f"deploying to Space: {space_id} (docker sdk, {'private' if args.private else 'public'})")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stage(tmp)
        n = sum(1 for _ in tmp.rglob("*") if _.is_file())
        size = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file())
        print(f"  staged {n} files, {size / 1e6:.1f} MB")
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(tmp),
            commit_message="Deploy PSS01 dashboard",
        )

    url = f"https://huggingface.co/spaces/{space_id}"
    print(f"\ndone: {url}")
    print("The Space builds the Docker image on first push; give it ~2-4 minutes.")


if __name__ == "__main__":
    main()
