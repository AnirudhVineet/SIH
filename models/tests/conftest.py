"""
Every models/*.py module uses bare imports (`from harness import TARGET`,
not `from models.harness import TARGET`), matching how they're invoked
elsewhere in this repo (e.g. run_backtest.py is run from inside models/).
Put models/ on sys.path so the same imports work under pytest regardless
of the directory pytest is invoked from.
"""

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
