"""Hyperparameter tuning utilities for the supervised ML workflows.

The project standardizes on ``GridSearchCV`` and ``RandomizedSearchCV`` with
stratified cross-validation to preserve the 4.9% positive-class rate in every
fold. The tuning strategy prioritizes recall and F1, especially for logistic
regression, while still exploring the rest of the classifiers in a controlled
way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.trial import TrialState

    _OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore
    TPESampler = None  # type: ignore
    TrialState = None  # type: ignore
    _OPTUNA_AVAILABLE = False

try:
    from xgboost import XGBClassifier  # type: ignore

    _XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False

try:
    from .model_evaluation import CV_SCORING, build_stratified_kfold
    from .model_training import get_model_registry
except ImportError:  # pragma: no cover - script execution fallback
    from src.model_evaluation import CV_SCORING, build_stratified_kfold  # type: ignore
    from src.model_training import get_model_registry  # type: ignore

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


def _cross_val_metric_summary(estimator, X, y, cv) -> Dict[str, float]:
    """Compute CV metric summary using the project scoring dictionary."""
    results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "precision_mean": float(np.mean(results["test_precision"])),
        "recall_mean": float(np.mean(results["test_recall"])),
        "f1_mean": float(np.mean(results["test_f1"])),
        "roc_auc_mean": float(np.mean(results["test_roc_auc"])),
    }


def tune_logistic_with_optuna(
    X,
    y,
    *,
    cv=None,
    random_state: int = 42,
    n_trials: int = 20,
    preprocessing_pipeline=None,
) -> tuple[optuna.Study, pd.DataFrame, pd.DataFrame]:
    """Optimize logistic regression hyperparameters with Optuna.

    The objective prioritizes recall to align with ACV false-negative minimization.
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna no esta disponible en el entorno actual. "
            "Instala las dependencias o usa el tuning basado en Grid/Random search."
        )
    _validate_inputs(X, y)
    splitter = cv or build_stratified_kfold(random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        c_value = trial.suggest_float("C", 1e-3, 100.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        steps = []
        if preprocessing_pipeline is not None:
            steps.append(("preprocessing", clone(preprocessing_pipeline)))
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=random_state,
                ),
            )
        )
        estimator = Pipeline(steps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            metrics = _cross_val_metric_summary(estimator, X, y, splitter)

        for name, value in metrics.items():
            trial.set_user_attr(name, value)

        return float(metrics["recall_mean"])

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        study_name="acv_logistic_optuna",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = {
        "classifier__solver": "liblinear",
        "classifier__penalty": "l2",
        "classifier__C": float(best_trial.params["C"]),
        "classifier__class_weight": best_trial.params["class_weight"],
    }

    optuna_summary = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "search_type": "OptunaStudy",
                "refit_metric": "recall",
                "best_score": float(best_trial.value),
                "best_params": best_params,
                "precision_mean": float(best_trial.user_attrs.get("precision_mean", np.nan)),
                "recall_mean": float(best_trial.user_attrs.get("recall_mean", np.nan)),
                "f1_mean": float(best_trial.user_attrs.get("f1_mean", np.nan)),
                "roc_auc_mean": float(best_trial.user_attrs.get("roc_auc_mean", np.nan)),
            }
        ]
    )

    trial_rows = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        trial_rows.append(
            {
                "trial_number": trial.number,
                "state": str(trial.state),
                "value": float(trial.value),
                "C": float(trial.params.get("C", np.nan)),
                "class_weight": trial.params.get("class_weight"),
                "precision_mean": float(trial.user_attrs.get("precision_mean", np.nan)),
                "recall_mean": float(trial.user_attrs.get("recall_mean", np.nan)),
                "f1_mean": float(trial.user_attrs.get("f1_mean", np.nan)),
                "roc_auc_mean": float(trial.user_attrs.get("roc_auc_mean", np.nan)),
            }
        )

    optuna_trials_df = pd.DataFrame(trial_rows).sort_values(
        by=["recall_mean", "f1_mean", "roc_auc_mean"],
        ascending=[False, False, False],
    )

    return study, optuna_summary, optuna_trials_df


__all__ = [
    "SCORING",
    "get_default_search_plan",
    "build_search_object",
    "tune_model",
    "tune_all_models",
    "print_tuning_summary",
    "get_best_estimators",
    "get_tuned_model_registry",
    "tune_logistic_with_optuna",
]


def run_tuning(random_state: int = 42) -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train_path = processed_dir / "X_train.csv"
    y_train_path = processed_dir / "y_train.csv"
    if not x_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(
            "Missing train artifacts (X_train.csv/y_train.csv). Run preprocess phase first."
        )

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]

    # Data is already transformed in preprocess phase.
    logistic_pipeline = Pipeline(
        [
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                ),
            )
        ]
    )

    cv = build_stratified_kfold(n_splits=3, random_state=random_state)
    search = tune_model(
        model_name="logistic_regression",
        estimator=logistic_pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
        n_iter=6,
    )

    idx = search.best_index_
    baseline_summary = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "search_type": type(search).__name__,
                "refit_metric": search.refit,
                "best_score": float(search.best_score_),
                "best_params": search.best_params_,
                "precision_mean": float(search.cv_results_["mean_test_precision"][idx]),
                "recall_mean": float(search.cv_results_["mean_test_recall"][idx]),
                "f1_mean": float(search.cv_results_["mean_test_f1"][idx]),
                "roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][idx]),
            }
        ]
    )

    summary_parts = [baseline_summary]
    if _OPTUNA_AVAILABLE:
        _, optuna_summary, optuna_trials_df = tune_logistic_with_optuna(
            X_train,
            y_train,
            cv=cv,
            random_state=random_state,
            n_trials=20,
        )
        summary_parts.append(optuna_summary)
        optuna_trials_df.to_csv(models_dir / "optuna_study.csv", index=False)
    else:
        print("Optuna no esta instalado; se omite la optimizacion adicional y se conserva el tuning base.")

    summary = pd.concat(summary_parts, ignore_index=True).sort_values(
        by=["recall_mean", "f1_mean", "roc_auc_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_row = summary.iloc[0]
    best_payload = {
        "best_model": best_row["model"],
        "best_params": best_row["best_params"],
        "best_recall": float(best_row["recall_mean"]),
        "best_f1": float(best_row["f1_mean"]),
        "search_method": best_row["search_type"],
    }

    with (models_dir / "best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(best_payload, fh, indent=2)

    summary.to_csv(models_dir / "tuning_comparison.csv", index=False)
    print(f"Tuning complete. Artifacts written to {models_dir}")


if __name__ == "__main__":
    run_tuning()
