<<<<<<< Updated upstream
"""Fase 2 orchestrator and compatibility entrypoint for EDA-ACV.

Modes:
- --status: Show current roadmap context and delegated phases.
- --compat: Validate environment versions and show safe git sync flow.
- --run: Execute the delegated phase scripts.
- --smoke-test: Run an end-to-end supervised smoke test using teammate modules.
=======
"""Punto de entrada del proyecto ACV para estado, compatibilidad y smoke test.

Modos:
- ``--status``: muestra contexto del proyecto y ruta de trabajo recomendada.
- ``--compat``: valida versiones de librerias y muestra flujo seguro de sync.
- ``--smoke-test``: ejecuta una validacion supervisada y serializa el mejor modelo base.
>>>>>>> Stashed changes
"""

from __future__ import annotations

import argparse
import re
<<<<<<< Updated upstream
import subprocess
=======
>>>>>>> Stashed changes
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Optional

<<<<<<< Updated upstream
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
PROGRESS_LOG = PROJECT_ROOT / "progress_log.md"


@dataclass(frozen=True)
class PhaseTask:
    phase: int
    title: str
    agent: str
    script_candidates: tuple[Path, ...]
    expected_artifact_candidates: tuple[Path, ...]
    required: bool = True


TASKS: tuple[PhaseTask, ...] = (
    PhaseTask(
        phase=1,
        title="Auditoria y optimizacion de datos",
        agent="@data-cleaner",
        script_candidates=(
            PROJECT_ROOT / "src" / "audit.py",
            PROJECT_ROOT / "src" / "preprocess.py",
        ),
        expected_artifact_candidates=(
            PROJECT_ROOT / "data" / "processed" / "data_audited.csv",
            PROJECT_ROOT / "data" / "processed" / "processed_data.csv",
        ),
    ),
    PhaseTask(
        phase=2,
        title="Preprocesamiento y split Train/Test",
        agent="@data-cleaner",
        script_candidates=(
            PROJECT_ROOT / "src" / "preprocessing.py",
            PROJECT_ROOT / "src" / "preprocess.py",
        ),
        expected_artifact_candidates=(
            PROJECT_ROOT / "data" / "processed" / "train.csv",
            PROJECT_ROOT / "data" / "processed" / "test.csv",
            PROJECT_ROOT / "data" / "processed" / "X_train.csv",
            PROJECT_ROOT / "data" / "processed" / "X_test.csv",
        ),
        required=False,
    ),
    PhaseTask(
        phase=3,
        title="Analisis no supervisado (PCA y clustering)",
        agent="@data-visualizer",
        script_candidates=(PROJECT_ROOT / "src" / "unsupervised.py",),
        expected_artifact_candidates=(
            PROJECT_ROOT / "reports" / "figures" / "pca_clusters.png",
            PROJECT_ROOT / "reports" / "figures" / "clustering_summary.png",
        ),
        required=False,
    ),
    PhaseTask(
        phase=4,
        title="Ajuste de hiperparametros",
        agent="@stats-modeler",
        script_candidates=(PROJECT_ROOT / "src" / "tune.py",),
        expected_artifact_candidates=(
            PROJECT_ROOT / "models" / "best_params.json",
            PROJECT_ROOT / "models" / "optuna_study.csv",
        ),
    ),
    PhaseTask(
        phase=5,
        title="Entrenamiento final con Train",
        agent="@stats-modeler",
        script_candidates=(PROJECT_ROOT / "src" / "train.py",),
        expected_artifact_candidates=(
            PROJECT_ROOT / "models" / "final_model.pkl",
            PROJECT_ROOT / "models" / "best_model.pkl",
        ),
    ),
    PhaseTask(
        phase=6,
        title="Evaluacion final con Test",
        agent="@data-visualizer",
        script_candidates=(PROJECT_ROOT / "src" / "evaluate.py",),
        expected_artifact_candidates=(
            PROJECT_ROOT / "reports" / "evaluation_results.md",
            PROJECT_ROOT / "reports" / "classification_report.txt",
        ),
    ),
)


REQUIRED_MIN_VERSIONS: tuple[tuple[str, str], ...] = (
    ("pandas", "2.0.0"),
    ("numpy", "1.24.0"),
    ("scikit-learn", "1.3.0"),
    ("optuna", "3.6.0"),
)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_progress_log() -> None:
    if PROGRESS_LOG.exists():
        return
    PROGRESS_LOG.write_text(
        "# Bitacora de Progreso - Fase 2\n\n"
        "## Registro de Actividad de Agentes\n\n",
        encoding="utf-8",
    )


def read_context_lines() -> list[str]:
    ensure_progress_log()
    return PROGRESS_LOG.read_text(encoding="utf-8").splitlines()[:40]


def append_activity(message: str) -> None:
    ensure_progress_log()
    with PROGRESS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- [{timestamp()}] {message}\n")


def _normalize_version(v: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", v)
    return tuple(int(p) for p in parts[:3]) if parts else (0,)


def _has_minimum_version(installed: str, minimum: str) -> bool:
    return _normalize_version(installed) >= _normalize_version(minimum)


def compatibility_report() -> int:
    print("Compatibilidad de entorno:")
    errors = 0
    for pkg, min_version in REQUIRED_MIN_VERSIONS:
        try:
            installed = version(pkg)
        except PackageNotFoundError:
            print(f"- {pkg}: NO INSTALADO (requerido >= {min_version})")
            errors += 1
            continue

        if _has_minimum_version(installed, min_version):
            print(f"- {pkg}: OK ({installed} >= {min_version})")
        else:
            print(f"- {pkg}: INCOMPATIBLE ({installed} < {min_version})")
            errors += 1

    print("\nFlujo recomendado para traer cambios del companero sin perder trabajo local:")
    print("1) git stash push -u -m 'wip-compat-orchestrator-progress'")
    print("2) git pull --rebase origin main")
    print("3) git stash pop")
    print("4) Resolver conflictos y validar con: python3 main.py --status")
    return 1 if errors else 0


def resolve_script(task: PhaseTask) -> Optional[Path]:
    for candidate in task.script_candidates:
        if candidate.exists():
            return candidate
    return None


def notify_delegation(task: PhaseTask) -> None:
    script = resolve_script(task)
    script_text = script.relative_to(PROJECT_ROOT) if script else "<sin script>"
    delegation = (
        f"@data-orchestrator -> {task.agent}: Fase {task.phase} "
        f"{task.title} en {script_text}"
    )
    print(f"\nDelegacion: {delegation}")
    append_activity(delegation)


def run_task_script(task: PhaseTask) -> bool:
    script = resolve_script(task)
    if script is None:
        msg = f"Fase {task.phase} bloqueada: no se encontro script en candidatos configurados."
        print(msg)
        append_activity(f"@data-orchestrator: {msg}")
        return not task.required

    print(f"Ejecutando Fase {task.phase}: {task.title}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), check=False)

    if result.returncode != 0:
        msg = f"Fase {task.phase} fallo con codigo {result.returncode}."
        print(msg)
        append_activity(f"@data-orchestrator: {msg}")
        return not task.required

    has_any_artifact = (
        any(path.exists() for path in task.expected_artifact_candidates)
        if task.expected_artifact_candidates
        else True
    )
    if not has_any_artifact:
        missing_txt = ", ".join(str(m.relative_to(PROJECT_ROOT)) for m in task.expected_artifact_candidates)
        msg = f"Fase {task.phase} sin artefactos esperados: {missing_txt}."
        print(msg)
        append_activity(f"@data-orchestrator: {msg}")
        return not task.required

    ok_msg = f"@data-orchestrator: Fase {task.phase} completada correctamente."
    append_activity(ok_msg)
    print(ok_msg)
    return True


def print_status(tasks: Iterable[PhaseTask]) -> None:
    print("Contexto actual (inicio de progress_log.md):")
    for line in read_context_lines():
        print(line)

    print("\nPlan de delegacion:")
    for task in tasks:
        script = resolve_script(task)
        script_text = script.relative_to(PROJECT_ROOT) if script else "NO_ENCONTRADO"
        print(
            f"- Fase {task.phase}: {task.title} | Agente {task.agent} | "
            f"Script {script_text}"
        )


def execute_pipeline(tasks: Iterable[PhaseTask]) -> int:
    append_activity("@data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.")
    for task in tasks:
        notify_delegation(task)
        if not run_task_script(task):
            append_activity("@data-orchestrator: Pipeline detenido por error de fase.")
            return 1

    append_activity("@data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).")
    print("\nPipeline finalizado. Siguiente paso: solicitar commit a @github-git-agent con confirmacion Y/n.")
    return 0


def build_feature_preprocessor(X_raw: pd.DataFrame) -> Pipeline:
    """Create the preprocessing pipeline used by smoke-test mode."""
    try:
        from src.data_preprocessing import OutlierCapper, SmartImputer, UnknownToNaN
    except Exception:
        from src.preprocess import OutlierCapper, SmartImputer, UnknownToNaN

    numeric_features = X_raw.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X_raw.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    return Pipeline(
        [
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


def smoke_test() -> int:
    """Run teammate-compatible supervised smoke test."""
    try:
        from src.model_evaluation import build_stratified_kfold, print_model_comparison_report
        from src.model_training import get_model_registry
    except Exception:
        print("No se pudieron importar src.model_evaluation o src.model_training.")
        return 1

    data_path = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
    if not data_path.exists():
        print(f"Missing dataset: {data_path}")
        return 1

    df_raw = pd.read_csv(data_path)
    target = "stroke"
    if target not in df_raw.columns:
        print("No existe la columna objetivo 'stroke' en el dataset.")
        return 1

    X_raw = df_raw.drop(columns=[c for c in [target, "id"] if c in df_raw.columns])
    y_raw = df_raw[target]
=======
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
        "# Registro de Progreso - Proyecto ACV\n\n"
        "## Actividad reciente\n\n",
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
    print("Estado actual del proyecto\n")
    print(f"- Dataset base: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- Log de progreso: {PROGRESS_LOG.relative_to(PROJECT_ROOT)}")
    print(f"- Modelos serializados: models/trained_models/")
    print("\nRuta de trabajo")
    for item in WORK_ITEMS:
        print(f"- {item.name} | {item.owner} | {item.path} | {item.status}")

    print("\nResumen de progress_log.md")
    lines = PROGRESS_LOG.read_text(encoding="utf-8").splitlines()[:20]
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
    try:
        df_raw = load_raw_dataset(DATASET_PATH)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print("Dataset cargado")
    print(f"Filas: {df_raw.shape[0]}, Columnas: {df_raw.shape[1]}")
>>>>>>> Stashed changes

    X_raw, y_raw = split_features_target(df_raw, target="stroke", drop_columns=("id",))
    feature_preprocessor = build_feature_preprocessor(X_raw)
<<<<<<< Updated upstream
    model_registry = get_model_registry(random_state=42)
    model_pipelines = {
        name: Pipeline([
            ("preprocessing", feature_preprocessor),
            ("classifier", estimator),
        ])
        for name, estimator in model_registry.items()
    }
=======
    model_pipelines = build_model_pipelines(feature_preprocessor, random_state=42)
>>>>>>> Stashed changes

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
        f"Smoke test supervisado ejecutado. Mejor modelo base: {best_model_name}. "
        f"Artefacto: {output_path.relative_to(PROJECT_ROOT)}"
    )

    print("\nMejor candidato supervisado")
    print(summary.iloc[0].to_string())
    print(f"\nModelo serializado en: {output_path.relative_to(PROJECT_ROOT)}")
    return 0


def parse_args() -> argparse.Namespace:
<<<<<<< Updated upstream
    parser = argparse.ArgumentParser(description="Orquestador Fase 2 + compatibilidad de equipo")
    parser.add_argument("--run", action="store_true", help="Ejecuta scripts de fases y actualiza progress_log.md")
    parser.add_argument("--status", action="store_true", help="Muestra contexto actual y plan de delegacion")
    parser.add_argument("--compat", action="store_true", help="Valida versiones y muestra flujo seguro para pull/rebase")
    parser.add_argument("--smoke-test", action="store_true", help="Ejecuta smoke test supervisado compatible con modulo del companero")
=======
    parser = argparse.ArgumentParser(description="Entrada principal del proyecto ACV")
    parser.add_argument("--status", action="store_true", help="Muestra contexto y estado actual del proyecto")
    parser.add_argument("--compat", action="store_true", help="Valida compatibilidad y flujo seguro de sync")
    parser.add_argument("--smoke-test", action="store_true", help="Ejecuta un chequeo supervisado y serializa el mejor modelo base")
>>>>>>> Stashed changes
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.compat:
        return compatibility_report()

    if args.smoke_test:
        return smoke_test()

<<<<<<< Updated upstream
    if args.status or not args.run:
        print_status(TASKS)

    if not args.run:
        print("\nModo solo lectura. Usa --run para ejecutar el pipeline.")
        return 0

    return execute_pipeline(TASKS)
=======
    return print_status()
>>>>>>> Stashed changes


if __name__ == "__main__":
    raise SystemExit(main())
