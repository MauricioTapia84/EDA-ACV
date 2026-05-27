"""Phase 2 unsupervised orchestrator wrapper."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsupervised import run_unsupervised


if __name__ == "__main__":
    run_unsupervised()
