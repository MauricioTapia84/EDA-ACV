# 🎯 Reglas de Operación - Fase 2 (Modelado)

## 🧬 Contexto y Referencia
Este proyecto sigue la metodología modular de [codon-classification-pipeline](https://github.com/trigoduoc/codon-classification-pipeline). Todo el desarrollo debe alinearse con la estructura de 6 fases (0-5).

## 🤖 Roles de Agente
Consulta [.github/AGENTS.md](.github/AGENTS.md) para identificar qué agente debe realizar cada tarea. No ignores los límites de responsabilidad de cada rol.

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

## 👥 Definiciones Detalladas
Los perfiles de agentes individuales se encuentran en [.github/agents/](.github/agents/).
