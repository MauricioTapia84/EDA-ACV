"""Public exports for the project's source package."""

from .preprocess import OutlierCapper, SmartImputer, UnknownToNaN
from .evaluate import (
    build_cross_validation_report,
    build_stratified_kfold,
    confusion_matrix_report,
    evaluate_model_cv,
    print_classification_cv_report,
    print_model_comparison_report,
    roc_curve_points,
)
from .train import (
    ModelSpec,
    build_logistic_regression,
    build_random_forest,
    build_svc,
    build_xgboost,
    get_model_registry,
    get_model_specifications,
)

__all__ = [
    "UnknownToNaN",
    "SmartImputer",
    "OutlierCapper",
    "ModelSpec",
    "build_logistic_regression",
    "build_random_forest",
    "build_svc",
    "build_xgboost",
    "get_model_registry",
    "get_model_specifications",
    "build_stratified_kfold",
    "evaluate_model_cv",
    "build_cross_validation_report",
    "print_classification_cv_report",
    "print_model_comparison_report",
    "confusion_matrix_report",
    "roc_curve_points",
]
