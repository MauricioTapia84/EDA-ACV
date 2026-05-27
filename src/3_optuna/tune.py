"""Phase 3 hyperparameter tuning orchestrator wrapper."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tune import run_tuning


if __name__ == "__main__":
    run_tuning()
