"""Compatibility wrapper for tuning utilities."""

from .tune import (
    SCORING,
    get_default_search_plan,
    build_search_object,
    tune_model,
    tune_all_models,
    tune_logistic_with_optuna,
    print_tuning_summary,
    get_best_estimators,
    get_tuned_model_registry,
)

__all__ = [
    "SCORING",
    "get_default_search_plan",
    "build_search_object",
    "tune_model",
    "tune_all_models",
    "tune_logistic_with_optuna",
    "print_tuning_summary",
    "get_best_estimators",
    "get_tuned_model_registry",
]
