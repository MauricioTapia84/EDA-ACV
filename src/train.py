"""Compatibilidad hacia atras para el modulo oficial de entrenamiento."""

from .model_training import (
    ModelSpec,
    build_logistic_regression,
    build_model_pipelines,
    build_random_forest,
    build_svc,
    build_xgboost,
    fit_and_serialize_model,
    get_model_registry,
    get_model_specifications,
    serialize_trained_model,
)

__all__ = [
    "ModelSpec",
    "build_logistic_regression",
    "build_random_forest",
    "build_svc",
    "build_xgboost",
    "get_model_registry",
    "get_model_specifications",
    "build_model_pipelines",
    "serialize_trained_model",
    "fit_and_serialize_model",
]
