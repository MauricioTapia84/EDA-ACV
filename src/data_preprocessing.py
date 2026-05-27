"""Preprocesamiento y preparacion de datos para el proyecto de ACV.

Este modulo concentra la logica oficial de limpieza, transformacion y
codificacion usada por notebooks y scripts. Los transformadores son compatibles
con ``Pipeline`` y ``ColumnTransformer`` de scikit-learn.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _is_categorical_series(series: pd.Series) -> bool:
    """Retorna True cuando la serie de pandas usa un tipo categorico."""
    return isinstance(series.dtype, pd.CategoricalDtype)


class UnknownToNaN(BaseEstimator, TransformerMixin):
    """Reemplaza variantes textuales de valores faltantes por ``NaN``.

    El informe indica que en campos categoricos pueden aparecer cadenas
    explicitas como ``"Unknown"`` que no son detectadas por ``isnull()``.
    """

    def __init__(
        self,
        columns: Optional[Iterable[str]] = None,
        variants: Optional[Iterable[str]] = None,
    ) -> None:
        self.columns = columns
        self.variants = variants

    def fit(self, X, y=None):  # noqa: D401
        """Guarda las columnas a procesar y valida la entrada."""
        X_df = self._validate_input(X)
        self.columns_ = self._resolve_columns(X_df)
        self.variants_ = {
            "unknown",
            "na",
            "n/a",
            "null",
            "none",
            "",
        }
        if self.variants is not None:
            self.variants_.update(str(value).strip().lower() for value in self.variants)
        return self

    def transform(self, X):  # noqa: D401
        """Reemplaza las variantes textuales configuradas por ``NaN``."""
        self._ensure_fitted()
        X_df = self._validate_input(X)
        X_out = X_df.copy()

        for column in self.columns_:
            if column not in X_out.columns:
                continue
            series = X_out[column]
            if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
                continue

            normalized = series.astype("string").str.strip().str.lower()
            mask = normalized.isin(self.variants_)
            X_out.loc[mask, column] = np.nan

        return X_out

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [col for col in self.columns if col in X.columns]

        return [
            col
            for col in X.columns
            if pd.api.types.is_object_dtype(X[col])
            or pd.api.types.is_string_dtype(X[col])
            or _is_categorical_series(X[col])
        ]

    @staticmethod
    def _validate_input(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("UnknownToNaN espera un DataFrame de pandas como entrada.")
        return X

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "columns_"):
            raise AttributeError("Este transformador aun no fue ajustado. Llama primero a 'fit'.")


class SmartImputer(BaseEstimator, TransformerMixin):
    """Imputa valores faltantes segun el porcentaje de ausencias."""

    def __init__(
        self,
        numeric_threshold_low: float = 0.10,
        numeric_threshold_high: float = 0.50,
        categorical_missing_label: str = "missing",
    ) -> None:
        self.numeric_threshold_low = numeric_threshold_low
        self.numeric_threshold_high = numeric_threshold_high
        self.categorical_missing_label = categorical_missing_label

    def fit(self, X, y=None):  # noqa: D401
        """Aprende la estrategia de imputacion por columna."""
        X_df = self._validate_input(X)
        self.feature_names_in_ = list(X_df.columns)
        self.impute_strategies_ = {}

        for column in self.feature_names_in_:
            series = X_df[column]
            missing_rate = series.isna().mean()
            is_numeric = pd.api.types.is_numeric_dtype(series)

            if is_numeric:
                if missing_rate <= self.numeric_threshold_high:
                    self.impute_strategies_[column] = {
                        "type": "numeric",
                        "strategy": "median",
                        "fill_value": float(series.median(skipna=True)),
                    }
                else:
                    warnings.warn(
                        f"La columna '{column}' tiene mas del 50% de valores faltantes; "
                        "se deja sin cambios segun las reglas del proyecto.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self.impute_strategies_[column] = {
                        "type": "numeric",
                        "strategy": "none",
                        "fill_value": np.nan,
                    }
            else:
                if missing_rate < self.numeric_threshold_low:
                    mode = series.mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else self.categorical_missing_label
                    self.impute_strategies_[column] = {
                        "type": "categorical",
                        "strategy": "mode",
                        "fill_value": fill_value,
                    }
                elif missing_rate <= self.numeric_threshold_high:
                    self.impute_strategies_[column] = {
                        "type": "categorical",
                        "strategy": "missing",
                        "fill_value": self.categorical_missing_label,
                    }
                else:
                    warnings.warn(
                        f"La columna '{column}' tiene mas del 50% de valores faltantes; "
                        "se deja sin cambios segun las reglas del proyecto.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self.impute_strategies_[column] = {
                        "type": "categorical",
                        "strategy": "none",
                        "fill_value": np.nan,
                    }

        return self

    def transform(self, X):  # noqa: D401
        """Imputa valores usando la estrategia aprendida en ``fit``."""
        self._ensure_fitted()
        X_df = self._validate_input(X)
        X_out = X_df.copy()

        for column, config in self.impute_strategies_.items():
            if column not in X_out.columns:
                continue

            fill_value = config["fill_value"]
            strategy = config["strategy"]
            if strategy == "none":
                continue

            if config["type"] == "categorical":
                if not _is_categorical_series(X_out[column]):
                    X_out[column] = X_out[column].astype("object")
                X_out[column] = X_out[column].fillna(fill_value)
            else:
                X_out[column] = X_out[column].fillna(fill_value)

        return X_out

    @staticmethod
    def _validate_input(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SmartImputer espera un DataFrame de pandas como entrada.")
        return X

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "impute_strategies_"):
            raise AttributeError("Este transformador aun no fue ajustado. Llama primero a 'fit'.")


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Recorta valores atipicos numericos usando la regla del IQR."""

    def __init__(self, columns: Optional[Iterable[str]] = None, factor: float = 1.5) -> None:
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):  # noqa: D401
        """Calcula los limites del IQR a partir de los datos de entrenamiento."""
        X_df = self._validate_input(X)
        self.columns_ = self._resolve_columns(X_df)
        self.bounds_ = {}

        for column in self.columns_:
            series = pd.to_numeric(X_df[column], errors="coerce")
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
            self.bounds_[column] = {"lower": lower, "upper": upper}

        return self

    def transform(self, X):  # noqa: D401
        """Acota columnas numericas usando los limites aprendidos del IQR."""
        self._ensure_fitted()
        X_df = self._validate_input(X)
        X_out = X_df.copy()

        for column, bounds in self.bounds_.items():
            if column not in X_out.columns:
                continue
            X_out[column] = pd.to_numeric(X_out[column], errors="coerce").clip(
                lower=bounds["lower"],
                upper=bounds["upper"],
            )

        return X_out

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [col for col in self.columns if col in X.columns]
        return [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]

    @staticmethod
    def _validate_input(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OutlierCapper espera un DataFrame de pandas como entrada.")
        return X

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "bounds_"):
            raise AttributeError("Este transformador aun no fue ajustado. Llama primero a 'fit'.")


def load_raw_dataset(data_path: str | Path) -> pd.DataFrame:
    """Carga el dataset crudo y valida su existencia minima."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el dataset en: {path}")
    return pd.read_csv(path)


def split_features_target(
    df_raw: pd.DataFrame,
    target: str = "stroke",
    drop_columns: Sequence[str] = ("id",),
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y objetivo desde el dataset original."""
    if target not in df_raw.columns:
        raise KeyError(f"La columna objetivo '{target}' no existe en el dataset.")

    removable = [column for column in drop_columns if column in df_raw.columns and column != target]
    X_raw = df_raw.drop(columns=[target, *removable])
    y_raw = df_raw[target].copy()
    return X_raw, y_raw


def build_feature_preprocessor(X_raw: pd.DataFrame) -> Pipeline:
    """Construye el preprocesador compartido para modelado supervisado."""
    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError("build_feature_preprocessor espera un DataFrame de pandas.")

    numeric_features = X_raw.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_features = X_raw.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    return Pipeline(
        [
            ("unknown_to_nan", UnknownToNaN(columns=categorical_features)),
            ("smart_imputer", SmartImputer()),
            ("outlier_capper", OutlierCapper(columns=numeric_features)),
            (
                "feature_encoding",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline([("scaler", StandardScaler())]),
                            numeric_features,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "onehot",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            sparse_output=False,
                                        ),
                                    )
                                ]
                            ),
                            categorical_features,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )


def build_unsupervised_matrix(
    df_raw: pd.DataFrame,
    target: str = "stroke",
    drop_columns: Sequence[str] = ("id",),
):
    """Prepara una matriz numerica coherente con el flujo del proyecto."""
    X_raw, _ = split_features_target(df_raw, target=target, drop_columns=drop_columns)
    preprocessor = build_feature_preprocessor(X_raw)
    X_matrix = preprocessor.fit_transform(X_raw)
    return X_raw, X_matrix, preprocessor


__all__ = [
    "UnknownToNaN",
    "SmartImputer",
    "OutlierCapper",
    "load_raw_dataset",
    "split_features_target",
    "build_feature_preprocessor",
    "build_unsupervised_matrix",
]
