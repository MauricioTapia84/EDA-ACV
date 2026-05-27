"""Factory functions for supervised learning models used in the project.

The module centralizes the instantiation of baseline classifiers so notebooks
and scripts can reuse the same seeded configurations in a reproducible way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:  # Optional dependency in the current environment.
    from xgboost import XGBClassifier  # type: ignore

    _XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional import
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False


class ModelSpec(NamedTuple):
    """Lightweight metadata for a model factory entry."""

    name: str
    estimator: object
    description: str


def build_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """Create a logistic regression classifier with imbalance-friendly defaults."""
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create a random forest classifier with reproducible defaults."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svc(random_state: int = 42) -> SVC:
    """Create an SVC classifier configured for imbalanced binary classification."""
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )


def build_xgboost(
    random_state: int = 42,
    *,
    scale_pos_weight: Optional[float] = None,
):
    """Create an XGBoost classifier if the optional dependency is installed.

    Parameters
    ----------
    random_state:
        Seed for reproducibility.
    scale_pos_weight:
        Optional imbalance factor. When omitted, a conservative default of 1 is
        used. Callers can tune this externally after inspecting the data.
    """
    if not _XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not installed in the current environment. "
            "Install it to use build_xgboost()."
        )

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=1.0 if scale_pos_weight is None else scale_pos_weight,
    )


def get_model_registry(random_state: int = 42) -> Dict[str, object]:
    """Return the baseline model registry used across the notebooks.

    The registry contains robust classical classifiers that work well with the
    project's imbalance-aware evaluation strategy. XGBoost is included only if
    the dependency is available in the environment.
    """
    registry: Dict[str, object] = {
        "logistic_regression": build_logistic_regression(random_state=random_state),
        "random_forest": build_random_forest(random_state=random_state),
        "svc": build_svc(random_state=random_state),
    }

    if _XGBOOST_AVAILABLE:
        registry["xgboost"] = build_xgboost(random_state=random_state)

    return registry


def get_model_specifications(random_state: int = 42) -> Dict[str, ModelSpec]:
    """Return a metadata-rich mapping of models and their descriptions."""
    specs = {
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            estimator=build_logistic_regression(random_state=random_state),
            description="Linear baseline with class balancing and calibrated probabilities.",
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=build_random_forest(random_state=random_state),
            description="Non-linear ensemble baseline robust to mixed feature interactions.",
        ),
        "svc": ModelSpec(
            name="svc",
            estimator=build_svc(random_state=random_state),
            description="Kernel-based classifier with probability estimates for ROC-AUC.",
        ),
    }

    if _XGBOOST_AVAILABLE:
        specs["xgboost"] = ModelSpec(
            name="xgboost",
            estimator=build_xgboost(random_state=random_state),
            description="Gradient-boosted tree ensemble, included when the package is installed.",
        )

    return specs


__all__ = [
    "ModelSpec",
    "build_logistic_regression",
    "build_random_forest",
    "build_svc",
    "build_xgboost",
    "get_model_registry",
    "get_model_specifications",
]


def run_training(random_state: int = 42) -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train_path = processed_dir / "X_train.csv"
    y_train_path = processed_dir / "y_train.csv"
    best_params_path = models_dir / "best_params.json"

    if not x_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError("Missing train data artifacts. Run preprocess first.")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]

    registry = get_model_registry(random_state=random_state)

    selected_model_name = "logistic_regression"
    selected_params: dict = {}
    if best_params_path.exists():
        with best_params_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        selected_model_name = payload.get("best_model", selected_model_name)
        selected_params = payload.get("best_params", {})

    if selected_model_name not in registry:
        selected_model_name = "logistic_regression"
        selected_params = {}

    model = registry[selected_model_name]
    # Remove pipeline prefix from tuned params when present.
    cleaned_params = {}
    for key, value in selected_params.items():
        cleaned_key = key.replace("classifier__", "")
        cleaned_params[cleaned_key] = value

    if cleaned_params:
        model.set_params(**cleaned_params)

    model.fit(X_train, y_train)
    joblib.dump(model, models_dir / "final_model.pkl")
    joblib.dump(model, models_dir / "best_model.pkl")

    print(f"Training complete. Model saved to {models_dir / 'final_model.pkl'}")


if __name__ == "__main__":
    run_training()
