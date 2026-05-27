# 🎯 Reglas de Operación - Fase 2 (Modelado)

## 🧬 Contexto y Referencia
Este proyecto sigue la metodología modular de [codon-classification-pipeline](https://github.com/trigoduoc/codon-classification-pipeline). Todo el desarrollo debe alinearse con la estructura de 6 fases (0-5).

## 🤖 Roles de Agente
Consulta [.github/AGENTS.md](.github/AGENTS.md) para identificar qué agente debe realizar cada tarea. No ignores los límites de responsabilidad de cada rol.

## 🧭 Jerarquía de Instrucciones
- Primero: perfiles especializados en `.github/agents/*.agent.md`.
- Segundo: mapa de roles y carpetas en `.github/AGENTS.md`.
- Tercero: estas reglas globales en `.github/copilot-instructions.md`.
- Si hay conflicto, prima la instrucción más específica al archivo o tarea.

## 📁 Estructura del Código
- **`src/`**: Dividido en subcarpetas numeradas por fase (`0_audit` a `5_report`).
- **Scripts**: Cada fase debe tener un script orquestador o notebook de ejecución.
- **Data**: 
  - `data/raw/`: INMUTABLE. No abrir para escritura.
  - `data/processed/`: Salida de procesos de limpieza y transformación.

## 🛠 Stack Tecnológico
- **Core**: Python 3.x, pandas, scikit-learn.
- **Modelado**: xgboost, lightgbm, optuna.
- **Visualización**: matplotlib, seaborn, yellowbrick.

## 🚀 Comandos Críticos
- Inicializar estructura: `python3 setup_and_run.py`
- Instalación: `pip install -r requirements.txt`

## ✅ Protocolo Para Evaluar Estado Del Proyecto
Cuando el usuario pida "evaluar cómo está el proyecto" o "ejecutarlo para validar estado", usar este orden:

1. Estado general (sin ejecución completa):
  - `python setup_and_run.py --mode status --venv-name venv --skip-install`
2. Compatibilidad del entorno:
  - `python setup_and_run.py --mode compat --venv-name venv --skip-install`
3. Ejecución end-to-end del pipeline:
  - `python setup_and_run.py --mode run --venv-name venv --skip-install`
4. Verificación rápida opcional:
  - `python setup_and_run.py --mode smoke-test --venv-name venv --skip-install`

Reportar resultados con foco en:
- Métricas finales y trade-off recall/precision.
- Artefactos generados en `data/processed/`, `models/` y `reports/`.
- Brechas detectadas respecto de [docs/verificacion_rubrica_pdf.md](../docs/verificacion_rubrica_pdf.md).

## 👥 Definiciones Detalladas
Los perfiles de agentes individuales se encuentran en [.github/agents/](.github/agents/).
