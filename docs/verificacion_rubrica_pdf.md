# Verificacion de Cumplimiento contra PDF de Rubrica

Fecha de verificacion: 2026-05-27
Fuente: `docs/Documentacion/EV PARCIAL 2 SCY1101_ESTUDIANTE.pdf`
Metodo: extraccion automatica de texto con `pypdf` en `.venv` y contraste con evidencia del repo.

## Extractos clave detectados en la rubrica

- "El entregable debe incluir ... estructura de carpetas y archivos".
- "notebooks/01_exploratory_analysis.ipynb".
- "02_supervised_modeling.ipynb".
- "03_model_evaluation.ipynb".
- "04_hyperparameter_optimization.ipynb: Optimizacion ... (GridSearchCV/RandomizedSearchCV)".
- "05_final_analysis.ipynb: Integracion y analisis final".
- "src/data_preprocessing.py, model_training.py, model_evaluation.py, hyperparameter_tuning.py".
- "models/trained_models/".
- "results/metrics/, plots/, reports/".
- "README.md".
- "Entregable debe ser 100% reproducible".

## Checklist de cumplimiento

| Criterio de rubrica | Evidencia en proyecto | Estado |
|---|---|---|
| Estructura de notebooks por etapas | `notebooks/01_exploratory_analysis.ipynb`, `02_supervised_modeling.ipynb`, `04_hyperparameter_optimization.ipynb`, `05_final_analysis.ipynb` | Cumple |
| Notebook de evaluacion comparativa | `notebooks/3_model_evaluation.ipynb` (alias `03_model_evaluation.ipynb` agregado para alineacion nominal) | Cumple |
| Modulos `src` solicitados | `src/data_preprocessing.py`, `src/model_training.py`, `src/model_evaluation.py`, `src/hyperparameter_tuning.py` | Cumple |
| Modelos serializados | `models/trained_models/` y `models/final_model.pkl` | Cumple |
| Resultados en metricas/plots/reports | `results/metrics`, `results/plots`, `reports/` | Cumple |
| README y guia de uso | `README.md` actualizado y `docs/estructura_proyecto.md` creado | Cumple |
| Optimizacion de hiperparametros | `GridSearchCV/RandomizedSearchCV` en notebooks + Optuna en `src/tune.py` | Cumple (extendido) |
| Reproducibilidad | Ejecucion validada con `setup_and_run.py --mode run --skip-install` | Cumple |
| Interpretacion de resultados | `reports/feature_importance.csv` generado en fase de evaluacion; SHAP opcional cuando esta disponible | Cumple |

## Observaciones

- Se agrego interpretabilidad explicita en `src/evaluate.py` mediante feature importance y salida opcional SHAP.
- Se reforzo narrativa de decisiones en notebooks de modelado, evaluacion y analisis final.
- El flujo operativo final se mantiene modular y ejecutable end-to-end desde launcher oficial.
