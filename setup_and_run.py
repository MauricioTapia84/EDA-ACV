"""Bootstrap unificado para preparar y ejecutar el proyecto ACV.

Este script:
1. Verifica la estructura principal exigida por la rubrica y el progress log.
2. Crea o reutiliza un entorno virtual.
3. Instala dependencias desde ``requirements.txt``.
4. Ejecuta ``main.py`` en modo status, compat, run o smoke-test.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
DEFAULT_VENV_NAME = ".venv"

REQUIRED_FOLDERS = (
    "data/raw",
    "data/processed",
    "docs",
    "models/trained_models",
    "notebooks",
    "results/metrics",
    "results/plots",
    "results/reports",
    "src",
)


@dataclass
class StepResult:
    name: str
    ok: bool
    details: str
    return_code: int = 0


def _venv_python(venv_path: Path) -> Path:
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _run_cmd(command: list[str], cwd: Path, name: str) -> StepResult:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        return StepResult(name=name, ok=False, details=f"Exception: {exc}", return_code=1)

    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    output = output.strip() or "(sin salida)"
    return StepResult(name=name, ok=(proc.returncode == 0), details=output, return_code=proc.returncode)


def ensure_structure() -> StepResult:
    created = 0
    for rel in REQUIRED_FOLDERS:
        folder = PROJECT_ROOT / rel
        existed = folder.exists()
        folder.mkdir(parents=True, exist_ok=True)
        if not existed:
            created += 1

    return StepResult(
        name="Estructura",
        ok=True,
        details=f"Estructura verificada. Directorios nuevos creados: {created}.",
    )


def ensure_venv(venv_path: Path) -> StepResult:
    python_bin = _venv_python(venv_path)
    if python_bin.exists():
        return StepResult(
            name="Entorno virtual",
            ok=True,
            details=f"Entorno existente detectado: {venv_path.name}",
        )

    result = _run_cmd([sys.executable, "-m", "venv", str(venv_path)], PROJECT_ROOT, "Crear venv")
    if not result.ok:
        return StepResult(
            name="Entorno virtual",
            ok=False,
            details=f"No se pudo crear el entorno:\n{result.details}",
            return_code=result.return_code,
        )

    return StepResult(
        name="Entorno virtual",
        ok=True,
        details=f"Entorno creado correctamente: {venv_path.name}",
    )


def install_dependencies(venv_path: Path) -> StepResult:
    python_bin = _venv_python(venv_path)
    if not REQUIREMENTS_FILE.exists():
        return StepResult(
            name="Dependencias",
            ok=False,
            details=f"No se encontro {REQUIREMENTS_FILE.name}",
            return_code=1,
        )

    pip_upgrade = _run_cmd(
        [str(python_bin), "-m", "pip", "install", "--upgrade", "pip"],
        PROJECT_ROOT,
        "Actualizar pip",
    )
    if not pip_upgrade.ok:
        return StepResult(
            name="Dependencias",
            ok=False,
            details=f"Fallo al actualizar pip:\n{pip_upgrade.details}",
            return_code=pip_upgrade.return_code,
        )

    install = _run_cmd(
        [str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        PROJECT_ROOT,
        "Instalar requirements",
    )
    if not install.ok:
        return StepResult(
            name="Dependencias",
            ok=False,
            details=f"Fallo instalando requirements:\n{install.details}",
            return_code=install.return_code,
        )

    return StepResult(name="Dependencias", ok=True, details="Dependencias instaladas correctamente.")


def run_main_mode(venv_path: Path, mode: str) -> StepResult:
    if not MAIN_SCRIPT.exists():
        return StepResult(name="Ejecucion", ok=False, details="No existe main.py", return_code=1)

    python_bin = _venv_python(venv_path)
    mode_flag = {
        "status": "--status",
        "compat": "--compat",
        "run": "--run",
        "smoke-test": "--smoke-test",
    }[mode]

    result = _run_cmd([str(python_bin), str(MAIN_SCRIPT), mode_flag], PROJECT_ROOT, f"main.py {mode_flag}")
    return StepResult(name="Ejecucion", ok=result.ok, details=result.details, return_code=result.return_code)


def collect_observations(mode: str, venv_name: str, steps: list[StepResult]) -> list[str]:
    notes: list[str] = []
    failed = [step for step in steps if not step.ok]

    if failed:
        notes.append("Se detectaron fallas en uno o mas pasos; revisa el bloque de errores.")
    else:
        notes.append("Flujo completado sin errores de setup o ejecucion.")

    if venv_name == ".venv":
        notes.append("Se esta usando '.venv', coherente con el estado estable actual del repo.")
    else:
        notes.append("Si cambias a otro entorno virtual, alinea README y comandos del equipo.")

    if mode == "status":
        notes.append("Modo status: muestra contexto y pendientes sin correr entrenamiento.")
    elif mode == "compat":
        notes.append("Modo compat: util para revisar dependencias y flujo seguro de pull/rebase.")
    elif mode == "run":
        notes.append("Modo run: ejecuta el pipeline completo de fases con artefactos en data/processed, models y reports.")
    elif mode == "smoke-test":
        notes.append("Smoke-test: valida integracion supervisada y deja un modelo serializado.")

    return notes


def print_report(steps: list[StepResult], observations: list[str]) -> None:
    print("\n" + "=" * 72)
    print("RESUMEN SETUP_AND_RUN - PROYECTO ACV")
    print("=" * 72)

    for step in steps:
        status = "OK" if step.ok else "ERROR"
        print(f"[{status}] {step.name}")

    errors = [step for step in steps if not step.ok]
    if errors:
        print("\nERRORES DE EJECUCION")
        print("-" * 72)
        for err in errors:
            print(f"Paso: {err.name}")
            print(f"Codigo retorno: {err.return_code}")
            print(err.details)
            print("-" * 72)

    print("\nOBSERVACIONES GENERALES")
    print("-" * 72)
    for note in observations:
        print(f"- {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup y ejecucion unificada para el proyecto ACV")
    parser.add_argument(
        "--mode",
        choices=["status", "compat", "run", "smoke-test"],
        default="status",
        help="Modo a ejecutar en main.py (default: status).",
    )
    parser.add_argument(
        "--venv-name",
        default=DEFAULT_VENV_NAME,
        help="Nombre del entorno virtual a usar o crear (default: .venv).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Omite la instalacion de dependencias si ya estan disponibles.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    venv_path = PROJECT_ROOT / args.venv_name

    steps: list[StepResult] = []
    steps.append(ensure_structure())
    steps.append(ensure_venv(venv_path))

    if not steps[-1].ok:
        observations = collect_observations(args.mode, args.venv_name, steps)
        print_report(steps, observations)
        return 1

    if not args.skip_install:
        dep_result = install_dependencies(venv_path)
        steps.append(dep_result)
        if not dep_result.ok:
            observations = collect_observations(args.mode, args.venv_name, steps)
            print_report(steps, observations)
            return 1
    else:
        steps.append(StepResult(name="Dependencias", ok=True, details="Instalacion omitida por --skip-install."))

    steps.append(run_main_mode(venv_path, args.mode))

    observations = collect_observations(args.mode, args.venv_name, steps)
    print_report(steps, observations)
    return 0 if all(step.ok for step in steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
