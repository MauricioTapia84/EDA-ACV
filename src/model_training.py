"""Definicion, construccion y serializacion de modelos del proyecto."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:  # Dependencia opcional en el entorno actual.
    from xgboost import XGBClassifier  # type: ignore

    _XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - importacion opcional
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False


class ModelSpec(NamedTuple):
    """Metadatos ligeros para una entrada del catalogo de modelos."""

    name: str
    estimator: object
    description: str


def build_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """Crea una regresion logistica con valores base aptos para desbalance."""
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Crea un Random Forest con parametros reproducibles."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svc(random_state: int = 42) -> SVC:
    """Crea un SVC configurado para clasificacion binaria desbalanceada."""
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )


def build_xgboost(
    random_state: int = 42,
    *,
    scale_pos_weight: Optional[float] = None,
):
    """Crea un XGBoost si la dependencia opcional esta instalada."""
    if not _XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost no esta instalado en el entorno actual. "
            "Instalalo para usar build_xgboost()."
        )

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=1.0 if scale_pos_weight is None else scale_pos_weight,
    )


def get_model_registry(random_state: int = 42) -> Dict[str, object]:
    """Retorna el catalogo de modelos base usado en los notebooks."""
    registry: Dict[str, object] = {
        "logistic_regression": build_logistic_regression(random_state=random_state),
        "random_forest": build_random_forest(random_state=random_state),
        "svc": build_svc(random_state=random_state),
    }

    if _XGBOOST_AVAILABLE:
        registry["xgboost"] = build_xgboost(random_state=random_state)

    return registry


def get_model_specifications(random_state: int = 42) -> Dict[str, ModelSpec]:
    """Retorna un mapeo enriquecido con los modelos y su descripcion."""
    specs = {
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            estimator=build_logistic_regression(random_state=random_state),
            description="Linea base lineal con balance de clases y probabilidades calibradas.",
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=build_random_forest(random_state=random_state),
            description="Linea base no lineal robusta a interacciones mixtas entre variables.",
        ),
        "svc": ModelSpec(
            name="svc",
            estimator=build_svc(random_state=random_state),
            description="Clasificador basado en kernels con probabilidades para ROC-AUC.",
        ),
    }

    if _XGBOOST_AVAILABLE:
        specs["xgboost"] = ModelSpec(
            name="xgboost",
            estimator=build_xgboost(random_state=random_state),
            description="Ensamble de arboles con boosting gradiente, incluido cuando el paquete esta instalado.",
        )

    return specs


def build_model_pipelines(feature_preprocessor, random_state: int = 42) -> Dict[str, Pipeline]:
    """Construye una Pipeline completa por cada modelo base del proyecto."""
    model_registry = get_model_registry(random_state=random_state)
    return {
        name: Pipeline(
            [
                ("preprocessing", feature_preprocessor),
                ("classifier", estimator),
            ]
        )
        for name, estimator in model_registry.items()
    }


def serialize_trained_model(
    estimator,
    output_path: str | Path,
    *,
    metadata: Optional[dict] = None,
    metadata_path: str | Path | None = None,
) -> Path:
    """Serializa un estimador y, opcionalmente, su metadata asociada."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)

    if metadata is not None:
        metadata_target = Path(metadata_path) if metadata_path is not None else path.with_suffix(".json")
        metadata_target.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    return path


def fit_and_serialize_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: str | Path,
    *,
    metadata: Optional[dict] = None,
    metadata_path: str | Path | None = None,
) -> Path:
    """Entrena una pipeline y guarda el modelo serializado en disco."""
    pipeline.fit(X_train, y_train)
    final_metadata = {"model_name": model_name}
    if metadata:
        final_metadata.update(metadata)
    return serialize_trained_model(
        pipeline,
        output_path,
        metadata=final_metadata,
        metadata_path=metadata_path,
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
