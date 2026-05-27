"""Orquestador liviano del proyecto ACV.

Modos:
- ``--status``: muestra el contexto actual del proyecto leyendo ``progress_log.md``.
- ``--compat``: valida dependencias minimas y muestra el flujo seguro de sincronizacion.
- ``--smoke-test``: ejecuta una validacion supervisada y serializa el mejor modelo base.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.data_preprocessing import (
    build_feature_preprocessor,
    load_raw_dataset,
    split_features_target,
)
from src.model_evaluation import build_stratified_kfold, print_model_comparison_report
from src.model_training import build_model_pipelines, serialize_trained_model


PROJECT_ROOT = Path(__file__).resolve().parent
PROGRESS_LOG = PROJECT_ROOT / "progress_log.md"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"


@dataclass(frozen=True)
class WorkItem:
    name: str
    owner: str
    path: str
    status: str


WORK_ITEMS = (
    WorkItem("EDA y no supervisado", "@data-visualizer", "notebooks/01_exploratory_analysis.ipynb", "operativo"),
    WorkItem("Modelado supervisado", "@stats-modeler", "notebooks/02_supervised_modeling.ipynb", "operativo"),
    WorkItem("Evaluacion comparativa", "@data-reporter", "notebooks/3_model_evaluation.ipynb", "operativo"),
    WorkItem("Optimizacion", "@stats-modeler", "notebooks/04_hyperparameter_optimization.ipynb", "operativo"),
    WorkItem("Analisis final", "@data-reporter", "notebooks/05_final_analysis.ipynb", "pendiente"),
    WorkItem("Preprocesamiento modular", "@data-cleaner", "src/data_preprocessing.py", "operativo"),
    WorkItem("Entrenamiento y serializacion", "@stats-modeler", "src/model_training.py", "operativo"),
    WorkItem("Evaluacion modular", "@data-reporter", "src/model_evaluation.py", "operativo"),
    WorkItem("Tuning modular", "@stats-modeler", "src/hyperparameter_tuning.py", "operativo"),
    WorkItem("No supervisado por script", "@data-visualizer", "src/unsupervised.py", "operativo"),
)

REQUIRED_MIN_VERSIONS = (
    ("pandas", "2.0.0"),
    ("numpy", "1.24.0"),
    ("scikit-learn", "1.3.0"),
    ("matplotlib", "3.8.0"),
    ("seaborn", "0.13.0"),
)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_progress_log() -> None:
    if PROGRESS_LOG.exists():
        return
    PROGRESS_LOG.write_text(
        "# Registro de Progreso - Proyecto ACV (Fase 2)\n\n"
        "## Hitos completados\n\n"
        "## Estado actual\n\n"
        "## Siguiente paso recomendado\n",
        encoding="utf-8",
    )


def append_activity(message: str) -> None:
    ensure_progress_log()
    with PROGRESS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- [{timestamp()}] {message}\n")


def _normalize_version(raw: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", raw)
    return tuple(int(part) for part in parts[:3]) if parts else (0,)


def _has_minimum_version(installed: str, minimum: str) -> bool:
    return _normalize_version(installed) >= _normalize_version(minimum)


def print_status() -> int:
    ensure_progress_log()
    print("Llamando a `@data-orchestrator`: leyendo contexto del proyecto desde progress_log.md.\n")
    print("Estado actual del proyecto\n")
    print(f"- Dataset base: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- Log de progreso: {PROGRESS_LOG.relative_to(PROJECT_ROOT)}")
    print(f"- Modelos serializados: models/trained_models/")

    print("\nRuta de trabajo")
    for item in WORK_ITEMS:
        print(f"- {item.name} | {item.owner} | {item.path} | {item.status}")

    print("\nResumen de progress_log.md")
    lines = PROGRESS_LOG.read_text(encoding="utf-8").splitlines()[:40]
    for line in lines:
        print(line)

    print("\nSiguiente paso sugerido: completar notebooks/05_final_analysis.ipynb")
    return 0


def compatibility_report() -> int:
    print("Compatibilidad de entorno")
    errors = 0

    for package, minimum in REQUIRED_MIN_VERSIONS:
        try:
            installed = version(package)
        except PackageNotFoundError:
            print(f"- {package}: NO INSTALADO (requerido >= {minimum})")
            errors += 1
            continue

        if _has_minimum_version(installed, minimum):
            print(f"- {package}: OK ({installed} >= {minimum})")
        else:
            print(f"- {package}: INCOMPATIBLE ({installed} < {minimum})")
            errors += 1

    print("\nFlujo seguro para traer cambios remotos sin perder trabajo local")
    print("1) git stash push -u -m 'wip-sync-acv'")
    print("2) git pull --rebase origin main")
    print("3) git stash pop")
    print("4) resolver conflictos y validar con: python main.py --status")
    return 1 if errors else 0


def smoke_test() -> int:
    """Ejecuta la validacion supervisada y serializa el mejor modelo base."""
    print("Llamando a `@stats-modeler`: ejecutando smoke test supervisado con pipelines oficiales.")
    try:
        df_raw = load_raw_dataset(DATASET_PATH)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print("Dataset cargado")
    print(f"Filas: {df_raw.shape[0]}, Columnas: {df_raw.shape[1]}")

    X_raw, y_raw = split_features_target(df_raw, target="stroke", drop_columns=("id",))
    feature_preprocessor = build_feature_preprocessor(X_raw)
    model_pipelines = build_model_pipelines(feature_preprocessor, random_state=42)

    cv = build_stratified_kfold(n_splits=5, random_state=42)
    summary = print_model_comparison_report(
        models=model_pipelines,
        X=X_raw,
        y=y_raw,
        cv=cv,
        random_state=42,
    )

    best_model_name = str(summary.iloc[0]["model"])
    best_pipeline = model_pipelines[best_model_name]
    best_pipeline.fit(X_raw, y_raw)

    artifact_dir = PROJECT_ROOT / "models" / "trained_models"
    output_path = artifact_dir / f"{best_model_name}_baseline_pipeline.joblib"
    metadata_path = artifact_dir / f"{best_model_name}_baseline_pipeline.json"
    serialize_trained_model(
        best_pipeline,
        output_path,
        metadata={
            "model_name": best_model_name,
            "source": "main.py --smoke-test",
            "selection_criteria": ["recall_mean", "f1_mean", "roc_auc_mean", "precision_mean"],
            "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        },
        metadata_path=metadata_path,
    )

    append_activity(
        f"@data-orchestrator: smoke test supervisado completado; mejor baseline {best_model_name}; "
        f"artefacto {output_path.relative_to(PROJECT_ROOT)}"
    )

    print("\nMejor candidato supervisado")
    print(summary.iloc[0].to_string())
    print(f"\nModelo serializado en: {output_path.relative_to(PROJECT_ROOT)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orquestador del proyecto ACV")
    parser.add_argument("--status", action="store_true", help="Muestra contexto y estado actual del proyecto")
    parser.add_argument("--compat", action="store_true", help="Valida compatibilidad y flujo seguro de sync")
    parser.add_argument("--smoke-test", action="store_true", help="Ejecuta un chequeo supervisado y serializa el mejor modelo base")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.compat:
        return compatibility_report()

    if args.smoke_test:
        return smoke_test()

    return print_status()


if __name__ == "__main__":
    raise SystemExit(main())
