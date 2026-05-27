"""Optimizacion de hiperparametros para los flujos supervisados del proyecto."""

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
except Exception:  # pragma: no cover - dependencia opcional
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False

from .model_evaluation import CV_SCORING, build_stratified_kfold
from .model_training import get_model_registry

SCORING = CV_SCORING


def _validate_inputs(X, y) -> None:
    """Valida las entradas antes de ejecutar la busqueda."""
    if X is None or y is None:
        raise ValueError("X y no pueden ser None.")
    if len(X) != len(y):
        raise ValueError("X y deben tener la misma cantidad de muestras.")


def _validate_pipeline(estimator) -> None:
    """Asegura que el estimador sea una Pipeline con el paso classifier."""
    if not hasattr(estimator, "fit") or not hasattr(estimator, "set_params"):
        raise TypeError("Se esperaba un estimador o Pipeline de scikit-learn.")
    if not hasattr(estimator, "named_steps"):
        raise ValueError(
            "La busqueda de hiperparametros espera una Pipeline con el paso 'classifier'."
        )
    if "classifier" not in estimator.named_steps:
        raise ValueError(
            "La Pipeline entregada debe exponer el estimador final como 'classifier'."
        )


def get_default_search_plan(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """Retorna el plan de tuning por defecto para el catalogo de modelos."""
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
                "classifier__n_estimators": [200, 300, 500],
                "classifier__max_depth": [None, 3, 5, 8],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2"],
                "classifier__bootstrap": [True],
                "classifier__class_weight": [None, "balanced", "balanced_subsample"],
                "classifier__criterion": ["gini"],
            },
            "refit": "f1",
            "n_iter": 8,
        },
        "svc": {
            "search_type": "random",
            "param_distributions": [
                {
                    "classifier__kernel": ["linear"],
                    "classifier__C": [0.1, 1, 10, 100],
                    "classifier__class_weight": [None, "balanced"],
                },
                {
                    "classifier__kernel": ["rbf"],
                    "classifier__C": [0.1, 1, 10, 100],
                    "classifier__gamma": ["scale", "auto"],
                    "classifier__class_weight": [None, "balanced"],
                },
            ],
            "refit": "f1",
            "n_iter": 8,
        },
    }

    if _XGBOOST_AVAILABLE:
        plan["xgboost"] = {
            "search_type": "random",
            "param_distributions": {
                "classifier__n_estimators": [150, 300, 500],
                "classifier__max_depth": [3, 4, 6],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__subsample": [0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.8, 0.9, 1.0],
                "classifier__min_child_weight": [1, 3],
                "classifier__gamma": [0.0, 0.1, 0.2],
                "classifier__reg_alpha": [0.0, 0.1],
                "classifier__reg_lambda": [1.0, 5.0],
                "classifier__scale_pos_weight": [1.0, 5.0, 10.0],
            },
            "refit": "f1",
            "n_iter": 8,
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
    """Crea el objeto de busqueda configurado para una Pipeline dada."""
    _validate_pipeline(estimator)

    search_plan = get_default_search_plan(random_state=random_state)
    if model_name not in search_plan:
        raise KeyError(f"No existe un plan de tuning para el modelo '{model_name}'.")

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

    raise ValueError(f"Tipo de busqueda no soportado '{search_type}' para el modelo '{model_name}'.")


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
    """Ajusta la busqueda de hiperparametros para un solo modelo."""
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
    except Exception as exc:  # pragma: no cover - envoltorio defensivo
        raise RuntimeError(
            f"La optimizacion de hiperparametros fallo para el modelo '{model_name}': {exc}"
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
    """Ajusta todos los modelos conocidos y devuelve busquedas y resumen."""
    _validate_inputs(X, y)
    if not model_pipelines:
        raise ValueError("model_pipelines no puede estar vacio.")

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
    """Imprime un resumen compacto del tuning priorizando Recall y F1."""
    if summary.empty:
        raise ValueError("summary no puede estar vacio.")

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
    print("\n=== Resumen de optimizacion de hiperparametros ===")
    print(display_df.to_string(index=False))
    return display_df


def get_best_estimators(searches: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae el mejor estimador de cada objeto de busqueda ajustado."""
    if not searches:
        raise ValueError("searches no puede estar vacio.")
    return {name: search.best_estimator_ for name, search in searches.items()}


def get_tuned_model_registry(random_state: int = 42) -> Dict[str, Any]:
    """Construye el catalogo de estimadores base a optimizar."""
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
