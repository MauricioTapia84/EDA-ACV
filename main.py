"""Fase 2 orchestrator and compatibility entrypoint for EDA-ACV.

Modes:
- --status: Show current roadmap context and delegated phases.
- --compat: Validate environment versions and show safe git sync flow.
- --run: Execute the delegated phase scripts.
- --smoke-test: Run an end-to-end supervised smoke test using teammate modules.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Optional

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
            PROJECT_ROOT / "src" / "0_audit" / "audit.py",
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
            PROJECT_ROOT / "src" / "1_prep" / "preprocess.py",
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
        script_candidates=(
            PROJECT_ROOT / "src" / "2_unsupervised" / "unsupervised.py",
            PROJECT_ROOT / "src" / "unsupervised.py",
        ),
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
        script_candidates=(
            PROJECT_ROOT / "src" / "3_optuna" / "tune.py",
            PROJECT_ROOT / "src" / "tune.py",
        ),
        expected_artifact_candidates=(
            PROJECT_ROOT / "models" / "best_params.json",
            PROJECT_ROOT / "models" / "optuna_study.csv",
        ),
    ),
    PhaseTask(
        phase=5,
        title="Entrenamiento final con Train",
        agent="@stats-modeler",
        script_candidates=(
            PROJECT_ROOT / "src" / "4_train" / "train.py",
            PROJECT_ROOT / "src" / "train.py",
        ),
        expected_artifact_candidates=(
            PROJECT_ROOT / "models" / "final_model.pkl",
            PROJECT_ROOT / "models" / "best_model.pkl",
        ),
    ),
    PhaseTask(
        phase=6,
        title="Evaluacion final con Test",
        agent="@data-visualizer",
        script_candidates=(
            PROJECT_ROOT / "src" / "5_report" / "evaluate.py",
            PROJECT_ROOT / "src" / "evaluate.py",
        ),
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
    if script.parent.name == "src" and script.suffix == ".py":
        module_name = f"src.{script.stem}"
        cmd = [sys.executable, "-m", module_name]
    else:
        cmd = [sys.executable, str(script)]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)

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

    feature_preprocessor = build_feature_preprocessor(X_raw)
    model_registry = get_model_registry(random_state=42)
    model_pipelines = {
        name: Pipeline([
            ("preprocessing", feature_preprocessor),
            ("classifier", estimator),
        ])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orquestador Fase 2 + compatibilidad de equipo")
    parser.add_argument("--run", action="store_true", help="Ejecuta scripts de fases y actualiza progress_log.md")
    parser.add_argument("--status", action="store_true", help="Muestra contexto actual y plan de delegacion")
    parser.add_argument("--compat", action="store_true", help="Valida versiones y muestra flujo seguro para pull/rebase")
    parser.add_argument("--smoke-test", action="store_true", help="Ejecuta smoke test supervisado compatible con modulo del companero")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.compat:
        return compatibility_report()

    if args.smoke_test:
        return smoke_test()

    if args.status or not args.run:
        print_status(TASKS)

    if not args.run:
        print("\nModo solo lectura. Usa --run para ejecutar el pipeline.")
        return 0

    return execute_pipeline(TASKS)


if __name__ == "__main__":
    raise SystemExit(main())
