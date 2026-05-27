"""Exportaciones publicas del paquete ``src`` del proyecto.

El paquete usa importacion diferida para evitar que dependencias opcionales
como `optuna` bloqueen fases tempranas del pipeline cuando todavia no se
necesitan.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    "UnknownToNaN": ("src.data_preprocessing", "UnknownToNaN"),
    "SmartImputer": ("src.data_preprocessing", "SmartImputer"),
    "OutlierCapper": ("src.data_preprocessing", "OutlierCapper"),
    "load_raw_dataset": ("src.data_preprocessing", "load_raw_dataset"),
    "split_features_target": ("src.data_preprocessing", "split_features_target"),
    "build_feature_preprocessor": ("src.data_preprocessing", "build_feature_preprocessor"),
    "build_unsupervised_matrix": ("src.data_preprocessing", "build_unsupervised_matrix"),
    "ModelSpec": ("src.model_training", "ModelSpec"),
    "build_logistic_regression": ("src.model_training", "build_logistic_regression"),
    "build_random_forest": ("src.model_training", "build_random_forest"),
    "build_svc": ("src.model_training", "build_svc"),
    "build_xgboost": ("src.model_training", "build_xgboost"),
    "get_model_registry": ("src.model_training", "get_model_registry"),
    "get_model_specifications": ("src.model_training", "get_model_specifications"),
    "build_model_pipelines": ("src.model_training", "build_model_pipelines"),
    "serialize_trained_model": ("src.model_training", "serialize_trained_model"),
    "fit_and_serialize_model": ("src.model_training", "fit_and_serialize_model"),
    "CV_SCORING": ("src.model_evaluation", "CV_SCORING"),
    "build_stratified_kfold": ("src.model_evaluation", "build_stratified_kfold"),
    "evaluate_model_cv": ("src.model_evaluation", "evaluate_model_cv"),
    "build_cross_validation_report": ("src.model_evaluation", "build_cross_validation_report"),
    "print_classification_cv_report": ("src.model_evaluation", "print_classification_cv_report"),
    "print_model_comparison_report": ("src.model_evaluation", "print_model_comparison_report"),
    "confusion_matrix_report": ("src.model_evaluation", "confusion_matrix_report"),
    "roc_curve_points": ("src.model_evaluation", "roc_curve_points"),
    "get_default_search_plan": ("src.hyperparameter_tuning", "get_default_search_plan"),
    "build_search_object": ("src.hyperparameter_tuning", "build_search_object"),
    "tune_model": ("src.hyperparameter_tuning", "tune_model"),
    "tune_all_models": ("src.hyperparameter_tuning", "tune_all_models"),
    "print_tuning_summary": ("src.hyperparameter_tuning", "print_tuning_summary"),
    "get_best_estimators": ("src.hyperparameter_tuning", "get_best_estimators"),
    "get_tuned_model_registry": ("src.hyperparameter_tuning", "get_tuned_model_registry"),
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'src' has no attribute '{name}'")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
