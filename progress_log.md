# Registro de Progreso - Proyecto ACV (Fase 2)

## Hitos completados

- Se alineo la estructura principal del repositorio con la rubrica: `notebooks/`, `src/`, `models/trained_models/` y `results/`.
- Se consolidaron los modulos oficiales de la rubrica como capa principal:
  - `src/data_preprocessing.py`
  - `src/model_training.py`
  - `src/model_evaluation.py`
  - `src/hyperparameter_tuning.py`
- Los archivos `preprocess.py`, `train.py`, `evaluate.py` y `tune.py` quedaron como compatibilidad hacia atras para no romper imports previos.
- Se implementaron los transformers reutilizables `UnknownToNaN`, `SmartImputer` y `OutlierCapper`.
- Se implemento el catalogo de modelos base con `LogisticRegression`, `RandomForest`, `SVC` y soporte opcional para `XGBoost`.
- Se implemento evaluacion con `StratifiedKFold`, `Precision`, `Recall`, `F1` y `ROC-AUC`.
- Se implemento tuning con `GridSearchCV` y `RandomizedSearchCV`, alineado con la rubrica.
- Se dejaron operativos los notebooks `01` a `04` con EDA, PCA, K-Means, pipelines supervisados, evaluacion comparativa y tuning.
- Se adapto `src/unsupervised.py` al flujo real del proyecto para que use el dataset crudo actual y guarde evidencia en `results/plots/`.
- Se actualizo `README.md` y `docs/estructura_proyecto.md` para que reflejen el estado real del repositorio.
- Se serializo un modelo entrenado real en `models/trained_models/`:
  - `logistic_regression_baseline_pipeline.joblib`
  - `logistic_regression_baseline_pipeline.json`

## Estado actual

- El punto de entrada `main.py` soporta:
  - `--status`
  - `--compat`
  - `--smoke-test`
- `setup_and_run.py` valida estructura, entorno virtual, dependencias y ejecucion por modo.
- `results/metrics/` y `results/plots/` contienen evidencia real generada por los notebooks y scripts.
- Los notebooks `02`, `03` y `04` consumen los modulos oficiales de `src/`:
  - `src.data_preprocessing`
  - `src.model_training`
  - `src.model_evaluation`
  - `src.hyperparameter_tuning`

## Hallazgos tecnicos importantes

- La variable objetivo `stroke` es binaria, por lo que `LogisticRegression` si corresponde como modelo de clasificacion.
- El flujo supervisado no muestra fugas obvias de datos:
  - `StandardScaler` y `OneHotEncoder` viven dentro de `Pipeline` y `ColumnTransformer`.
  - No se detecto uso de `SMOTE`.
  - La validacion cruzada se ejecuta con `StratifiedKFold`.
- El rendimiento no es fuerte en `Precision`, pero si es razonable en `Recall`, lo que es consistente con el desbalance del dataset.
- El mejor baseline actual por criterio de `Recall` sigue siendo `logistic_regression`.

## Siguiente paso recomendado

- Completar `notebooks/05_final_analysis.ipynb` con:
  - resumen de hallazgos del EDA
  - justificacion del mejor modelo
  - comparacion baseline vs tuned
  - interpretacion de `Recall`, `F1` y `ROC-AUC`
  - conclusion tecnica final

## Contexto operativo del orquestador

- Antes de iniciar una tarea, `@data-orchestrator` debe leer este archivo y `.github/copilot-instructions.md`.
- Delegacion sugerida:
  - `@data-cleaner`: mantenimiento de `src/data_preprocessing.py`
  - `@stats-modeler`: mantenimiento de `src/model_training.py` y `src/hyperparameter_tuning.py`
  - `@data-reporter`: mantenimiento de `src/model_evaluation.py` y notebooks de cierre
  - `@data-visualizer`: mantenimiento de `src/unsupervised.py` y graficos
- La trazabilidad de hitos debe quedar registrada aqui cuando haya cambios relevantes.

## Actividad reciente

- [actualizado] Se resolvieron conflictos de merge en `main.py`, `setup_and_run.py` y este `progress_log.md`.
- [actualizado] Se verifico que `main.py --status` lea este archivo como fuente de continuidad.
- [actualizado] Se refactorizaron `02_supervised_modeling.ipynb`, `3_model_evaluation.ipynb` y `04_hyperparameter_optimization.ipynb` para consumir los helpers oficiales de `src/`.
- [actualizado] Se alinearon `data-orchestrator.agent.md` y `SKILLS.md` con la arquitectura actual basada en `data_preprocessing.py`, `model_training.py`, `model_evaluation.py` y `hyperparameter_tuning.py`.
