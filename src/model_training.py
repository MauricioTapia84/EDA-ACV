"""Compatibility wrapper for model factory functions."""

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
    "ModelSpec",
    "build_logistic_regression",
    "build_random_forest",
    "build_svc",
    "build_xgboost",
    "get_model_registry",
    "get_model_specifications",
]
