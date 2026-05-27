# Verificacion de Cumplimiento contra la Rubrica PDF

Fecha de verificacion: 2026-05-27
Fuente: `docs/Documentacion/EV PARCIAL 2 SCY1101_ESTUDIANTE.pdf`

## Extractos clave considerados

- Estructura de notebooks por etapas.
- Modulos oficiales de `src/`.
- `models/trained_models/`.
- `results/metrics/`, `results/plots/`, `results/reports/`.
- `README.md`.
- Reproducibilidad del entregable.
- Uso de `GridSearchCV` y `RandomizedSearchCV` para tuning.

## Checklist de cumplimiento

| Criterio de rubrica | Evidencia actual | Estado |
|---|---|---|
| `01_exploratory_analysis.ipynb` | `notebooks/01_exploratory_analysis.ipynb` | Cumple |
| `02_supervised_modeling.ipynb` | `notebooks/02_supervised_modeling.ipynb` | Cumple |
| `03_model_evaluation.ipynb` | `notebooks/03_model_evaluation.ipynb` | Cumple |
| `04_hyperparameter_optimization.ipynb` | `notebooks/04_hyperparameter_optimization.ipynb` | Cumple |
| `05_final_analysis.ipynb` | `notebooks/05_final_analysis.ipynb` | Cumple |
| `src/data_preprocessing.py` | presente y operativo | Cumple |
| `src/model_training.py` | presente y operativo | Cumple |
| `src/model_evaluation.py` | presente y operativo | Cumple |
| `src/hyperparameter_tuning.py` | presente y operativo | Cumple |
| Modelos serializados | `models/trained_models/` y `models/final_model.pkl` | Cumple |
| Evidencia en resultados | `results/metrics/`, `results/plots/`, `results/reports/` | Cumple |
| Reproducibilidad | `setup_and_run.py --mode run --skip-install` validado | Cumple |

## Observaciones de integracion

- La capa narrativa del proyecto vive en `notebooks/`.
- La capa modular oficial exigida por la rubrica vive en `src/data_preprocessing.py`, `src/model_training.py`, `src/model_evaluation.py` y `src/hyperparameter_tuning.py`.
- El proyecto agrega una capa operativa por fases (`src/0_audit` a `src/5_report`) para ejecutar el flujo completo desde consola sin romper la estructura pedida por la pauta.
- La evaluacion final incorpora interpretabilidad mediante `feature_importance` y SHAP opcional cuando la dependencia esta disponible.
- El enfoque metodologico prioriza `Recall` para deteccion de ACV positivo, dado el desbalance severo de la clase objetivo.
