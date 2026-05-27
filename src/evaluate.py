"""Evaluation utilities for supervised models with imbalanced data.

This module provides a compact, reusable cross-validation workflow based on
``StratifiedKFold`` so every fold preserves the minority class proportion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone


CV_SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def _validate_inputs(X, y) -> None:
    if X is None or y is None:
        raise ValueError("X and y must not be None.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows/samples.")


def build_stratified_kfold(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedKFold:
    """Create the cross-validation splitter used across the project."""
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
    """Evaluate a model with cross-validation and return summary metrics.

    The function computes Precision, Recall, F1, and ROC-AUC using a
    stratified split, which is appropriate for the project's 4.9% positive
    class rate.
    """
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
    """Return a fold-level metric table for the given estimator."""
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
    """Print and return a complete cross-validation report."""
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
    print("Cross-validated metrics (mean ± std)")
    print(f"Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    print(f"Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    print(f"F1-score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"ROC-AUC:   {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")
    print("\nFold-level detail:")
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
    """Evaluate several models and print a comparison table."""
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
    print("\n=== Model Comparison ===")
    print(report.to_string(index=False))
    return report


def _get_score_vector(estimator, X):
    """Extract a continuous score vector for ROC-AUC computation."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return np.asarray(scores)
    raise AttributeError(
        "Estimator must implement predict_proba or decision_function to compute ROC-AUC."
    )


def confusion_matrix_report(y_true, y_pred) -> pd.DataFrame:
    """Return the confusion matrix in a labeled dataframe."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )


def roc_curve_points(y_true, y_score) -> pd.DataFrame:
    """Return ROC curve coordinates as a dataframe."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def _extract_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Build a model-agnostic importance table for interpretation."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        # Fallback when the estimator does not expose importances.
        values = np.zeros(len(feature_names), dtype=float)

    if len(values) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(values))]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


def _optional_shap_summary(model, X: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return mean absolute SHAP values when shap is installed, else None."""
    try:
        import shap  # type: ignore
    except Exception:
        return None

    sample = X.head(min(200, len(X))).copy()
    if sample.empty:
        return None

    try:
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)
        values = shap_values.values
        if values.ndim == 3:
            # Binary classifier can return shape (n_samples, n_features, n_classes).
            values = values[:, :, -1]
        mean_abs = np.abs(values).mean(axis=0)
        shap_df = pd.DataFrame(
            {
                "feature": list(sample.columns),
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        return shap_df.reset_index(drop=True)
    except Exception:
        return None


def run_evaluation() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    x_test_path = processed_dir / "X_test.csv"
    y_test_path = processed_dir / "y_test.csv"
    model_path = models_dir / "final_model.pkl"

    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("Missing test artifacts. Run preprocessing first.")
    if not model_path.exists():
        raise FileNotFoundError("Missing models/final_model.pkl. Run training first.")

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_score = _get_score_vector(model, X_test)

    feature_names = [str(col) for col in X_test.columns]
    importance_df = _extract_feature_importance(model, feature_names)
    importance_df.to_csv(reports_dir / "feature_importance.csv", index=False)

    shap_df = _optional_shap_summary(model, X_test)
    if shap_df is not None:
        shap_df.to_csv(reports_dir / "shap_summary.csv", index=False)

    report_text = classification_report(y_test, y_pred, digits=4)
    cm_df = confusion_matrix_report(y_test, y_pred)
    roc_auc = float(roc_auc_score(y_test, y_score))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    with (reports_dir / "classification_report.txt").open("w", encoding="utf-8") as fh:
        fh.write(report_text)

    md_content = "\n".join(
        [
            "# Evaluation Results",
            "",
            "## Metrics",
            f"- precision: {precision:.4f}",
            f"- recall: {recall:.4f}",
            f"- f1: {f1:.4f}",
            f"- roc_auc: {roc_auc:.4f}",
            "",
            "## Interpretability",
            "",
            "Top 10 feature importances:",
            "",
            "| feature | importance |",
            "|---|---:|",
            *[
                f"| {row.feature} | {row.importance:.6f} |"
                for row in importance_df.head(10).itertuples(index=False)
            ],
            "",
            (
                "SHAP summary exported to `reports/shap_summary.csv`."
                if shap_df is not None
                else "SHAP no disponible en el entorno: se omite resumen SHAP."
            ),
            "",
            "## Confusion Matrix",
            "",
            "|       | pred_0 | pred_1 |",
            "|-------|--------|--------|",
            f"| actual_0 | {int(cm_df.loc['actual_0', 'pred_0'])} | {int(cm_df.loc['actual_0', 'pred_1'])} |",
            f"| actual_1 | {int(cm_df.loc['actual_1', 'pred_0'])} | {int(cm_df.loc['actual_1', 'pred_1'])} |",
            "",
            "## Classification Report",
            "",
            "```text",
            report_text,
            "```",
        ]
    )

    with (reports_dir / "evaluation_results.md").open("w", encoding="utf-8") as fh:
        fh.write(md_content)

    print(f"Evaluation complete. Artifacts written to {reports_dir}")


__all__ = [
    "build_stratified_kfold",
    "CV_SCORING",
    "evaluate_model_cv",
    "build_cross_validation_report",
    "print_classification_cv_report",
    "print_model_comparison_report",
    "confusion_matrix_report",
    "roc_curve_points",
    "_extract_feature_importance",
    "run_evaluation",
]


if __name__ == "__main__":
    run_evaluation()
