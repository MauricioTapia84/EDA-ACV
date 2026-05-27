"""Custom preprocessing transformers for the ACV prediction pipeline.

This module implements the preprocessing logic documented in the project
report. The transformers are designed to be compatible with scikit-learn
``Pipeline`` and ``ColumnTransformer`` workflows.
"""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _is_categorical_series(series: pd.Series) -> bool:
    """Return True when the pandas series uses a categorical dtype."""
    return isinstance(series.dtype, pd.CategoricalDtype)


class UnknownToNaN(BaseEstimator, TransformerMixin):
    """Replace textual missing-value variants with ``NaN``.

    The report specifies that categorical fields may contain explicit strings
    such as ``"Unknown"`` that are not detected by ``isnull()``. This
    transformer normalizes those variants into actual missing values so they
    can be handled by downstream imputers.

    Parameters
    ----------
    columns:
        Optional iterable with the columns to inspect. When omitted, all
        object, string, and categorical columns are processed.
    variants:
        Iterable of textual variants that should be treated as missing values.
    """

    def __init__(
        self,
        columns: Optional[Iterable[str]] = None,
        variants: Optional[Iterable[str]] = None,
    ) -> None:
        self.columns = columns
        self.variants = variants

    def fit(self, X, y=None):  # noqa: D401
        """Store the columns to process and validate the input."""
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
        """Replace configured textual variants with ``NaN``."""
        self._ensure_fitted()
        X_df = self._validate_input(X)
        X_out = X_df.copy()

        for column in self.columns_:
            if column not in X_out.columns:
                continue
            series = X_out[column]
            if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
                # Preserve non-string columns unchanged.
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
            raise TypeError("UnknownToNaN expects a pandas DataFrame as input.")
        return X

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "columns_"):
            raise AttributeError("This transformer is not fitted yet. Call 'fit' first.")


class SmartImputer(BaseEstimator, TransformerMixin):
    """Impute missing values according to the percentage of missing data.

    Logic extracted from the report:

    - Numeric columns
      - Missingness < 10%: median imputation
      - Missingness between 10% and 50%: median imputation
      - Missingness > 50%: warning, no automatic imputation
    - Categorical columns
      - Missingness < 10%: mode imputation
      - Missingness between 10% and 50%: fill with ``"missing"``
      - Missingness > 50%: warning, no automatic imputation

    Parameters
    ----------
    numeric_threshold_low:
        Lower threshold for the 10% range.
    numeric_threshold_high:
        Upper threshold for the 50% range.
    categorical_missing_label:
        Label used when the categorical missingness is in the 10%-50% range.
    """

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
        """Learn the per-column imputation strategy from the training data."""
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
                        f"Column '{column}' has more than 50% missing values; "
                        "leaving values unchanged per project rules.",
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
                        f"Column '{column}' has more than 50% missing values; "
                        "leaving values unchanged per project rules.",
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
        """Impute values using the strategy learned in ``fit``."""
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
                    # Keep string/object columns aligned with the report logic.
                    X_out[column] = X_out[column].astype("object")
                X_out[column] = X_out[column].fillna(fill_value)
            else:
                X_out[column] = X_out[column].fillna(fill_value)

        return X_out

    @staticmethod
    def _validate_input(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SmartImputer expects a pandas DataFrame as input.")
        return X

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "impute_strategies_"):
            raise AttributeError("This transformer is not fitted yet. Call 'fit' first.")


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers using the IQR rule.

    The report specifies the standard IQR-based capping rule:

    ``lower = Q1 - 1.5 * IQR``
    ``upper = Q3 + 1.5 * IQR``

    where ``IQR = Q3 - Q1``. Values outside the interval are clipped to the
    nearest boundary, preserving clinically relevant extremes while reducing
    the influence of leverage points.

    Parameters
    ----------
    columns:
        Optional iterable with the numeric columns to cap. When omitted, all
        numeric columns are processed.
    factor:
        IQR multiplier. The project uses the standard ``1.5`` factor.
    """

    def __init__(self, columns: Optional[Iterable[str]] = None, factor: float = 1.5) -> None:
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):  # noqa: D401
        """Compute the IQR bounds from the training data."""
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
        """Clip numeric columns to the learned IQR bounds."""
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
            raise TypeError("OutlierCapper expects a pandas DataFrame as input.")
        return X

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "bounds_"):
            raise AttributeError("This transformer is not fitted yet. Call 'fit' first.")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_raw_csv(data_raw_dir: Path) -> Path:
    csv_files = sorted(data_raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_raw_dir}")
    return csv_files[0]


def _build_feature_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    return Pipeline(
        steps=[
            ("unknown_to_nan", UnknownToNaN(columns=categorical_features)),
            ("smart_imputer", SmartImputer()),
            ("outlier_capper", OutlierCapper(columns=numeric_features)),
            (
                "feature_encoding",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "onehot",
                                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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


def run_preprocessing(test_size: float = 0.2, random_state: int = 42) -> None:
    root = _project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _find_raw_csv(raw_dir)
    df = pd.read_csv(csv_path)

    if "stroke" not in df.columns:
        raise ValueError("Target column 'stroke' was not found in raw dataset.")

    y = df["stroke"].copy()
    X = df.drop(columns=[col for col in ["stroke", "id"] if col in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = _build_feature_pipeline(X_train)
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t = pipeline.transform(X_test)

    train_features = pd.DataFrame(X_train_t)
    test_features = pd.DataFrame(X_test_t)

    train_features.to_csv(processed_dir / "X_train.csv", index=False)
    test_features.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    pd.concat([train_features, y_train.reset_index(drop=True)], axis=1).to_csv(
        processed_dir / "train.csv", index=False
    )
    pd.concat([test_features, y_test.reset_index(drop=True)], axis=1).to_csv(
        processed_dir / "test.csv", index=False
    )

    audited = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    audited.to_csv(processed_dir / "data_audited.csv", index=False)
    audited.to_csv(processed_dir / "processed_data.csv", index=False)

    print(f"Preprocessing complete. Artifacts written to: {processed_dir}")


if __name__ == "__main__":
    run_preprocessing()


__all__ = ["UnknownToNaN", "SmartImputer", "OutlierCapper", "run_preprocessing"]
