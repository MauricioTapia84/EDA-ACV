"""Live smoke test for the ACV project.

This entry point validates the current project structure by:
1. Loading the raw dataset.
2. Building the shared preprocessing pipeline.
3. Evaluating the baseline supervised models with stratified cross-validation.
4. Reporting the best model by Recall/F1.

The notebooks remain the primary narrative layer, but this script gives us a
single executable check that the modular pieces still work together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import OutlierCapper, SmartImputer, UnknownToNaN
from src.model_evaluation import build_stratified_kfold, print_model_comparison_report
from src.model_training import get_model_registry


def build_feature_preprocessor(X_raw: pd.DataFrame) -> Pipeline:
    """Create the shared preprocessing pipeline used by the notebooks."""
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


def main() -> int:
    """Run a live validation of the supervised modeling pipeline."""
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "raw" / "healthcare-dataset-stroke-data.csv"

    if not data_path.exists():
        print(f"Missing dataset: {data_path}")
        return 1

    df_raw = pd.read_csv(data_path)
    print("Dataset loaded")
    print(f"Rows: {df_raw.shape[0]}, Columns: {df_raw.shape[1]}")

    target = "stroke"
    X_raw = df_raw.drop(columns=[target, "id"])
    y_raw = df_raw[target]

    feature_preprocessor = build_feature_preprocessor(X_raw)
    model_registry = get_model_registry(random_state=42)
    model_pipelines = {
        name: Pipeline(
            [
                ("preprocessing", feature_preprocessor),
                ("classifier", estimator),
            ]
        )
        for name, estimator in model_registry.items()
    }

    cv = build_stratified_kfold(n_splits=5, random_state=42)
    summary = print_model_comparison_report(
        models=model_pipelines,
        X=X_raw,
        y=y_raw,
        cv=cv,
        random_state=42,
    )

    best_model = summary.iloc[0]
    print("\nBest supervised candidate")
    print(best_model.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
