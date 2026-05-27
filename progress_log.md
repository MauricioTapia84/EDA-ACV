# Bitacora de Progreso - Fase 2: Modelado ACV

Este archivo resume el estado actual del proyecto despues de integrar el
enfoque de defensa clinica del equipo con la estructura modular alineada a la
rubrica.

## Fuente de rubrica

- `docs/Documentacion/EV PARCIAL 2 SCY1101_ESTUDIANTE.pdf`
- `docs/verificacion_rubrica_pdf.md`

## Estado general

- El proyecto mantiene la estructura principal exigida por la pauta:
  - `notebooks/`
  - `src/`
  - `models/trained_models/`
  - `results/metrics/`
  - `results/plots/`
  - `results/reports/`
  - `README.md`
- El proyecto tambien conserva una capa operativa por fases (`src/0_audit` a
  `src/5_report`) para ejecutar el flujo completo desde `setup_and_run.py`.

## Historia analitica del proyecto

1. `01_exploratory_analysis.ipynb`
   - EDA del dataset.
   - Deteccion y estandarizacion de nulos ocultos.
   - PCA y clustering como apoyo no supervisado.

2. `02_supervised_modeling.ipynb`
   - Pipelines supervisados con preprocesamiento reusable.
   - Comparacion de modelos base con validacion cruzada estratificada.
   - Defensa metodologica centrada en `Recall` y `F1`.

3. `03_model_evaluation.ipynb`
   - Curvas ROC, metricas holdout y matrices de confusion.
   - Explicacion del trade-off entre detectar ACV positivos y aumentar falsos positivos.

4. `04_hyperparameter_optimization.ipynb`
   - Tuning con `GridSearchCV` y `RandomizedSearchCV`.
   - Comparacion baseline vs tuned.

5. `05_final_analysis.ipynb`
   - Cierre tecnico del proyecto.
   - Conclusion sobre el mejor modelo y su uso como apoyo de screening.

## Estructura de `src/`

### Modulos oficiales de la pauta

- `data_preprocessing.py`
- `model_training.py`
- `model_evaluation.py`
- `hyperparameter_tuning.py`

### Capa operativa por fases

- `preprocess.py`
- `unsupervised.py`
- `tune.py`
- `train.py`
- `evaluate.py`

Wrappers:

- `0_audit/audit.py`
- `1_prep/preprocess.py`
- `2_unsupervised/unsupervised.py`
- `3_optuna/tune.py`
- `4_train/train.py`
- `5_report/evaluate.py`

## Hallazgos tecnicos importantes

- La variable objetivo `stroke` es binaria, por lo que `LogisticRegression`
  corresponde como clasificador.
- El dataset tiene alrededor de 4.9% de positivos; por eso la lectura correcta
  del problema prioriza `Recall` y `F1` por sobre una comparacion ingenua por
  `Accuracy`.
- El flujo supervisado usa `Pipeline` y `StratifiedKFold`, por lo que no muestra
  fugas de datos evidentes.
- La evaluacion final ya incorpora interpretabilidad via `feature_importance` y
  SHAP opcional.

## Punto de entrada oficial

- `setup_and_run.py`
- Entorno virtual actual: `venv/`

Comandos recomendados:

- `python setup_and_run.py --mode status --venv-name venv --skip-install`
- `python setup_and_run.py --mode run --venv-name venv --skip-install`
- `python setup_and_run.py --mode smoke-test --venv-name venv --skip-install`

## Estado de integracion

- Se mantuvo la organizacion actual del proyecto.
- Se incorporo la defensa clinica del enfoque de deteccion positiva del equipo.
- Se agrego documentacion de verificacion de rubrica.
- Se reforzo la interpretabilidad del reporte final.
- Se valido la ejecucion completa del pipeline desde `setup_and_run.py`.
- [2026-05-27 13:00:00] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 13:00:00] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/0_audit/audit.py
- [2026-05-27 13:00:02] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 13:00:02] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/1_prep/preprocess.py
- [2026-05-27 13:00:03] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 13:00:03] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/2_unsupervised/unsupervised.py
- [2026-05-27 13:00:07] @data-orchestrator: Fase 3 completada correctamente.
- [2026-05-27 13:00:07] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/3_optuna/tune.py
- [2026-05-27 13:00:09] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 13:00:09] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/4_train/train.py
- [2026-05-27 13:00:10] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 13:00:10] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/5_report/evaluate.py
- [2026-05-27 13:00:11] @data-orchestrator: Fase 6 completada correctamente.
- [2026-05-27 13:00:11] @data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).
