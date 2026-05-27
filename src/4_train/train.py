"""Phase 4 training orchestrator wrapper."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import run_training


if __name__ == "__main__":
    run_training()
