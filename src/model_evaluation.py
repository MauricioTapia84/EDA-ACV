"""Funciones de evaluacion para modelos supervisados con datos desbalanceados."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


CV_SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def _validate_inputs(X, y) -> None:
    if X is None or y is None:
        raise ValueError("X y no pueden ser None.")
    if len(X) != len(y):
        raise ValueError("X y deben tener la misma cantidad de filas o muestras.")


def build_stratified_kfold(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedKFold:
    """Crea el separador de validacion cruzada usado en todo el proyecto."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def evaluate_model_cv(
    estimator,
    X,
    y,
    *,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evalua un modelo con validacion cruzada y retorna metricas resumen."""
    _validate_inputs(X, y)
    splitter = cv or build_stratified_kfold(random_state=random_state)

    results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=splitter,
        scoring=CV_SCORING,
        n_jobs=1,
        return_train_score=False,
    )

    summary = {
        "precision_mean": float(np.mean(results["test_precision"])),
        "precision_std": float(np.std(results["test_precision"], ddof=1)),
        "recall_mean": float(np.mean(results["test_recall"])),
        "recall_std": float(np.std(results["test_recall"], ddof=1)),
        "f1_mean": float(np.mean(results["test_f1"])),
        "f1_std": float(np.std(results["test_f1"], ddof=1)),
        "roc_auc_mean": float(np.mean(results["test_roc_auc"])),
        "roc_auc_std": float(np.std(results["test_roc_auc"], ddof=1)),
    }
    return summary


def build_cross_validation_report(
    estimator,
    X,
    y,
    *,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Retorna una tabla de metricas por fold para el estimador indicado."""
    _validate_inputs(X, y)
    splitter = cv or build_stratified_kfold(random_state=random_state)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]

        estimator_fold = clone(estimator)
        estimator_fold.fit(X_train, y_train)
        y_pred = estimator_fold.predict(X_test)
        y_score = _get_score_vector(estimator_fold, X_test)

        rows.append(
            {
                "fold": fold_idx,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score),
            }
        )

    return pd.DataFrame(rows)


def print_classification_cv_report(
    model_name: str,
    estimator,
    X,
    y,
    *,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Imprime y retorna un reporte completo de validacion cruzada."""
    summary = evaluate_model_cv(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        random_state=random_state,
    )
    fold_table = build_cross_validation_report(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        random_state=random_state,
    )

    print(f"\n=== {model_name} ===")
    print("Metricas cruzadas (media +/- desviacion estandar)")
    print(f"Precision:      {summary['precision_mean']:.4f} +/- {summary['precision_std']:.4f}")
    print(f"Exhaustividad:  {summary['recall_mean']:.4f} +/- {summary['recall_std']:.4f}")
    print(f"F1:             {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    print(f"ROC-AUC:        {summary['roc_auc_mean']:.4f} +/- {summary['roc_auc_std']:.4f}")
    print("\nDetalle por fold:")
    print(fold_table.to_string(index=False))

    return fold_table


def print_model_comparison_report(
    models: Dict[str, object],
    X,
    y,
    *,
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evalua varios modelos e imprime una tabla comparativa."""
    rows = []
    for name, estimator in models.items():
        summary = evaluate_model_cv(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            random_state=random_state,
        )
        rows.append({"model": name, **summary})

    report = pd.DataFrame(rows).sort_values(
        by=["recall_mean", "f1_mean", "roc_auc_mean", "precision_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    print("\n=== Comparacion de modelos ===")
    print(report.to_string(index=False))
    return report


def _get_score_vector(estimator, X):
    """Extrae un vector continuo de puntuaciones para calcular ROC-AUC."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return np.asarray(scores)
    raise AttributeError(
        "El estimador debe implementar predict_proba o decision_function para calcular ROC-AUC."
    )


def confusion_matrix_report(y_true, y_pred) -> pd.DataFrame:
    """Retorna la matriz de confusion en un DataFrame con etiquetas."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )


def roc_curve_points(y_true, y_score) -> pd.DataFrame:
    """Retorna las coordenadas de la curva ROC como DataFrame."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


__all__ = [
    "build_stratified_kfold",
    "CV_SCORING",
    "evaluate_model_cv",
    "build_cross_validation_report",
    "print_classification_cv_report",
    "print_model_comparison_report",
    "confusion_matrix_report",
    "roc_curve_points",
]
