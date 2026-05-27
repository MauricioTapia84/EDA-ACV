"""Hyperparameter tuning utilities for the supervised ML workflows.

The project standardizes on ``GridSearchCV`` and ``RandomizedSearchCV`` with
stratified cross-validation to preserve the 4.9% positive-class rate in every
fold. The tuning strategy prioritizes recall and F1, especially for logistic
regression, while still exploring the rest of the classifiers in a controlled
way.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

try:
    from xgboost import XGBClassifier  # type: ignore

    _XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False

from .model_evaluation import CV_SCORING, build_stratified_kfold
from .model_training import get_model_registry

SCORING = CV_SCORING


def _validate_inputs(X, y) -> None:
    """Validate tuning inputs before running search."""
    if X is None or y is None:
        raise ValueError("X and y cannot be None.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")


def _validate_pipeline(estimator) -> None:
    """Ensure the estimator is a pipeline with the expected classifier step."""
    if not hasattr(estimator, "fit") or not hasattr(estimator, "set_params"):
        raise TypeError("Expected a scikit-learn estimator or Pipeline.")
    if not hasattr(estimator, "named_steps"):
        raise ValueError(
            "Hyperparameter tuning expects a Pipeline with a 'classifier' step."
        )
    if "classifier" not in estimator.named_steps:
        raise ValueError(
            "The supplied Pipeline must expose the final estimator as 'classifier'."
        )


def get_default_search_plan(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """Return the default tuning plan for the project's model registry.

    Logistic regression is tuned with a compact grid search and refit on
    recall, because recall is the main objective for the minority class.
    Random forest and SVC are tuned with randomized search to explore a wider
    parameter space efficiently. XGBoost is included only if the package is
    available in the environment.
    """
    plan: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "search_type": "grid",
            "param_grid": {
                "classifier__solver": ["liblinear"],
                "classifier__penalty": ["l2"],
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
            "refit": "recall",
            "n_iter": None,
        },
        "random_forest": {
            "search_type": "random",
            "param_distributions": {
                "classifier__n_estimators": [200, 300, 500, 700, 900],
                "classifier__max_depth": [None, 3, 5, 8, 12, 16],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__bootstrap": [True, False],
                "classifier__class_weight": [None, "balanced", "balanced_subsample"],
                "classifier__criterion": ["gini", "entropy"],
            },
            "refit": "f1",
            "n_iter": 25,
        },
        "svc": {
            "search_type": "random",
            "param_distributions": [
                {
                    "classifier__kernel": ["linear"],
                    "classifier__C": [0.1, 0.3, 1, 3, 10, 30, 100],
                    "classifier__class_weight": [None, "balanced"],
                },
                {
                    "classifier__kernel": ["rbf"],
                    "classifier__C": [0.1, 0.3, 1, 3, 10, 30, 100],
                    "classifier__gamma": ["scale", "auto"],
                    "classifier__class_weight": [None, "balanced"],
                },
            ],
            "refit": "f1",
            "n_iter": 20,
        },
    }

    if _XGBOOST_AVAILABLE:
        plan["xgboost"] = {
            "search_type": "random",
            "param_distributions": {
                "classifier__n_estimators": [150, 250, 400, 600],
                "classifier__max_depth": [3, 4, 5, 6, 8],
                "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "classifier__min_child_weight": [1, 3, 5],
                "classifier__gamma": [0.0, 0.1, 0.2, 0.5],
                "classifier__reg_alpha": [0.0, 0.1, 1.0],
                "classifier__reg_lambda": [1.0, 5.0, 10.0],
                "classifier__scale_pos_weight": [1.0, 5.0, 10.0, 15.0],
            },
            "refit": "f1",
            "n_iter": 25,
        }

    return plan


def build_search_object(
    model_name: str,
    estimator,
    *,
    cv=None,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
    refit_metric: Optional[str] = None,
    n_iter: Optional[int] = None,
):
    """Create the configured search object for a given pipeline.

    Parameters
    ----------
    model_name:
        Key of the model in the project registry.
    estimator:
        A scikit-learn Pipeline whose final step is named ``classifier``.
    cv:
        Optional prebuilt cross-validator. When omitted, the project default
        ``StratifiedKFold`` is used.
    """
    _validate_pipeline(estimator)

    search_plan = get_default_search_plan(random_state=random_state)
    if model_name not in search_plan:
        raise KeyError(f"No tuning plan available for model '{model_name}'.")

    plan = search_plan[model_name]
    splitter = cv or build_stratified_kfold(random_state=random_state)
    refit = refit_metric or plan["refit"]
    search_type = plan["search_type"]

    if search_type == "grid":
        return GridSearchCV(
            estimator=estimator,
            param_grid=plan["param_grid"],
            scoring=SCORING,
            refit=refit,
            cv=splitter,
            n_jobs=n_jobs,
            verbose=verbose,
            error_score=np.nan,
        )

    if search_type == "random":
        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=plan["param_distributions"],
            n_iter=n_iter or plan["n_iter"],
            scoring=SCORING,
            refit=refit,
            cv=splitter,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            error_score=np.nan,
        )

    raise ValueError(f"Unsupported search type '{search_type}' for model '{model_name}'.")


def tune_model(
    model_name: str,
    estimator,
    X,
    y,
    *,
    cv=None,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
    refit_metric: Optional[str] = None,
    n_iter: Optional[int] = None,
):
    """Fit the configured hyperparameter search for a single model."""
    _validate_inputs(X, y)
    search = build_search_object(
        model_name=model_name,
        estimator=estimator,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit_metric=refit_metric,
        n_iter=n_iter,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            search.fit(X, y)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise RuntimeError(
            f"Hyperparameter tuning failed for model '{model_name}': {exc}"
        ) from exc

    return search


def tune_all_models(
    model_pipelines: Dict[str, Any],
    X,
    y,
    *,
    cv=None,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
    refit_overrides: Optional[Dict[str, str]] = None,
    n_iter_overrides: Optional[Dict[str, int]] = None,
):
    """Tune all known model pipelines and return both searches and summary.

    Returns
    -------
    searches:
        Mapping of model name to the fitted search object.
    summary:
        DataFrame with the best score and fold metrics for each model.
    """
    _validate_inputs(X, y)
    if not model_pipelines:
        raise ValueError("model_pipelines cannot be empty.")

    searches: Dict[str, Any] = {}
    rows = []

    for model_name, estimator in model_pipelines.items():
        search = tune_model(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            refit_metric=(refit_overrides or {}).get(model_name),
            n_iter=(n_iter_overrides or {}).get(model_name),
        )
        searches[model_name] = search

        idx = search.best_index_
        row = {
            "model": model_name,
            "search_type": type(search).__name__,
            "refit_metric": search.refit,
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
            "precision_mean": float(search.cv_results_["mean_test_precision"][idx]),
            "recall_mean": float(search.cv_results_["mean_test_recall"][idx]),
            "f1_mean": float(search.cv_results_["mean_test_f1"][idx]),
            "roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][idx]),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        by=["recall_mean", "f1_mean", "roc_auc_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return searches, summary


def print_tuning_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Print a compact tuning summary prioritizing recall and F1."""
    if summary.empty:
        raise ValueError("summary cannot be empty.")

    display_cols = [
        "model",
        "search_type",
        "refit_metric",
        "best_score",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
    ]
    display_df = summary[display_cols].copy()
    print("\n=== Hyperparameter Tuning Summary ===")
    print(display_df.to_string(index=False))
    return display_df


def get_best_estimators(searches: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the best estimator from each fitted search object."""
    if not searches:
        raise ValueError("searches cannot be empty.")
    return {name: search.best_estimator_ for name, search in searches.items()}


def get_tuned_model_registry(random_state: int = 42) -> Dict[str, Any]:
    """Build the registry of base estimators to be tuned.

    This helper keeps the tuning layer aligned with the current project models.
    """
    return get_model_registry(random_state=random_state)


__all__ = [
    "SCORING",
    "get_default_search_plan",
    "build_search_object",
    "tune_model",
    "tune_all_models",
    "print_tuning_summary",
    "get_best_estimators",
    "get_tuned_model_registry",
]
