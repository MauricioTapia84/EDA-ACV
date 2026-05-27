# 🤖 Ecosistema de Agentes - Proyecto EDA-ACV (Fase 2)

Este archivo entrena a los agentes para seguir la estructura de 6 pasos del pipeline de clasificación.

## 👥 Especialidades

| Agente | Rol | Responsabilidad Principal | Carpeta de Trabajo |
| :--- | :--- | :--- | :--- |
| **@env-architect** | Arquitecto | Mantenimiento de estructura, dependencias y `setup_and_run.py`. | Raíz / `src/` |
| **@eda-auditor** | Auditor | Auditoría de datos, chequeo de colinealidad (VIF) y calidad. | `src/0_audit/` |
| **@preprocessor** | Procesador | Limpieza, feature engineering y escalado. | `src/1_prep/` |
| **@unsupervised-modeler** | Modelador No Supervisado | Clustering (K-Means, DBSCAN) y reducción de dimensionalidad (PCA). | `src/2_unsupervised/` |
| **@hyperparameter-optimizer** | Optimizador | Búsqueda de hiperparámetros con Optuna y selección de modelos. | `src/3_optuna/` |
| **@trainer** | Entrenador | Entrenamiento final del pipeline y validación cruzada. | `src/4_train/` |
| **@report-generator** | Generador de Reportes | Visualización, métricas de desempeño y documentación de resultados. | `src/5_report/` / `reports/` |

## 🛠 Comandos Comunes
- **Configuración**: `python3 setup_and_run.py`
- **Carpeta de Datos**: `data/raw/` (Inmutable), `data/processed/` (Generados)
- **Documentación**: `docs/`

## 🔎 Workflow: Evaluar Estado Ejecutando El Proyecto
Usar este flujo cuando el usuario solicite revisar el estado actual del repositorio en ejecución real:

1. `python setup_and_run.py --mode status --venv-name venv --skip-install`
2. `python setup_and_run.py --mode compat --venv-name venv --skip-install`
3. `python setup_and_run.py --mode run --venv-name venv --skip-install`
4. (Opcional) `python setup_and_run.py --mode smoke-test --venv-name venv --skip-install`

Checklist de salida esperada:
- Evidencia de artefactos en `data/processed/`, `models/`, `reports/`.
- Lectura de métricas con prioridad en recall/F1 por desbalance.
- Verificación documental contra [docs/estructura_proyecto.md](../docs/estructura_proyecto.md) y [docs/verificacion_rubrica_pdf.md](../docs/verificacion_rubrica_pdf.md).

## 🧩 Convención de Fases
- El mapeo pedagógico es 0-5 (audit, prep, unsupervised, optuna, train, report).
- Algunas salidas operativas de scripts pueden mostrarse 1-6 por índice interno.
- Para coordinación entre agentes, usar siempre el mapeo 0-5.
