# Estructura del Proyecto ACV

Este documento explica como se organiza el repositorio y para que sirve cada
bloque principal.

## 1. `notebooks/`

Es la capa narrativa del proyecto. Aqui se cuenta la historia analitica en el
orden en que se desarrollo:

- `01_exploratory_analysis.ipynb`: EDA, limpieza conceptual inicial, PCA y clustering.
- `02_supervised_modeling.ipynb`: modelos base y validacion cruzada.
- `03_model_evaluation.ipynb`: comparacion holdout, curvas ROC y matrices de confusion.
- `04_hyperparameter_optimization.ipynb`: tuning y comparacion baseline vs tuned.
- `05_final_analysis.ipynb`: integracion final, conclusion tecnica y defensa del modelo.

## 2. `src/`

Contiene la logica reutilizable y los scripts operativos.

### 2.1 Modulos oficiales de la pauta

Estos archivos existen para cumplir la estructura pedida por la rubrica y para
centralizar la logica que consumen los notebooks:

- `data_preprocessing.py`: funciones y transformers de limpieza y preparacion.
- `model_training.py`: catalogo de modelos y serializacion.
- `model_evaluation.py`: metricas y evaluacion comparativa.
- `hyperparameter_tuning.py`: tuning con busquedas de hiperparametros.

### 2.2 Scripts operativos

Estos archivos permiten correr el flujo completo desde consola:

- `preprocess.py`
- `unsupervised.py`
- `tune.py`
- `train.py`
- `evaluate.py`

### 2.3 Wrappers por fase

Para orquestar la ejecucion end-to-end, el repositorio agrega wrappers en:

- `0_audit/audit.py`
- `1_prep/preprocess.py`
- `2_unsupervised/unsupervised.py`
- `3_optuna/tune.py`
- `4_train/train.py`
- `5_report/evaluate.py`

Estos wrappers llaman a los scripts operativos y facilitan que `main.py` y
`setup_and_run.py` ejecuten las fases en orden.

## 3. `results/`

Aqui vive la evidencia generada por los notebooks narrativos:

- `results/metrics/`: tablas `.csv` y `.json` de metricas.
- `results/plots/`: graficos de EDA, clustering, ROC y matrices de confusion.
- `results/reports/`: espacio reservado para reportes finales.

## 4. `models/`

Contiene los artefactos serializados:

- `models/trained_models/`: artefactos del smoke test modular.
- `models/*.pkl`, `models/*.json`, `models/*.csv`: artefactos del flujo completo por fases.

## 5. `setup_and_run.py`

Es el punto de entrada oficial del proyecto. Su rol es:

- verificar la estructura
- reutilizar o crear el entorno virtual
- instalar dependencias si corresponde
- ejecutar `main.py` en modo `status`, `compat`, `run` o `smoke-test`

## 6. `main.py`

Es el orquestador del pipeline por fases. Su rol es:

- mostrar contexto del proyecto
- validar compatibilidad
- ejecutar el pipeline de auditoria, preprocesamiento, tuning, entrenamiento y evaluacion
- servir como smoke test supervisado

## 7. Como interpretar la coexistencia de dos capas

No es un error que existan notebooks y scripts, ni que haya modulos oficiales y
wrappers por fase.

- Los notebooks cuentan la historia academica.
- Los modulos oficiales de `src/` dejan la logica reusable en nombres alineados con la pauta.
- Los wrappers y scripts operativos permiten ejecutar el proyecto completo desde consola.

Mientras las tres capas sean consistentes entre si, la estructura es valida y
reproducible.
