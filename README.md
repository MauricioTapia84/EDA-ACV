# Proyecto ACV - EP2 SCY1101

Proyecto de analisis y modelado predictivo para deteccion de accidentes
cerebrovasculares (ACV), alineado con la rubrica del curso y ejecutable desde
un launcher oficial.

## Objetivo

Construir un flujo reproducible de preprocesamiento, modelado, optimizacion y
evaluacion que priorice la deteccion de casos positivos de ACV. Por la
naturaleza del problema y el fuerte desbalance de clases, la defensa tecnica
prioriza `Recall` y `F1` por sobre una lectura ingenua de `Accuracy`.

## Estructura principal

- `notebooks/`: capa narrativa del proyecto.
- `src/`: logica modular reutilizable.
- `src/0_audit` a `src/5_report`: wrappers por fase para la ejecucion end-to-end.
- `data/raw/`: dataset de entrada original.
- `data/processed/`: artefactos tabulares generados por el pipeline.
- `results/metrics/`: tablas y archivos de metricas.
- `results/plots/`: graficos generados por los notebooks y scripts.
- `results/reports/`: espacio para reportes finales.
- `models/trained_models/`: modelos serializados como evidencia de entrenamiento.
- `models/`: artefactos del flujo por fases (`best_model.pkl`, `final_model.pkl`, `best_params.json`).

La explicacion detallada de cada modulo esta en
[docs/estructura_proyecto.md](docs/estructura_proyecto.md).
La verificacion de cumplimiento frente al PDF de la pauta esta documentada en
[docs/verificacion_rubrica_pdf.md](docs/verificacion_rubrica_pdf.md).

## Flujo recomendado de notebooks

1. Ejecutar `notebooks/01_exploratory_analysis.ipynb`.
2. Ejecutar `notebooks/02_supervised_modeling.ipynb`.
3. Ejecutar `notebooks/03_model_evaluation.ipynb`.
4. Ejecutar `notebooks/04_hyperparameter_optimization.ipynb`.
5. Ejecutar y revisar `notebooks/05_final_analysis.ipynb`.

## Modulos oficiales de `src/`

- `data_preprocessing.py`: limpieza, imputacion, tratamiento de outliers y
  construccion del preprocesador compartido.
- `model_training.py`: catalogo de modelos, armado de pipelines y serializacion.
- `model_evaluation.py`: validacion cruzada estratificada, metricas, ROC y
  matrices de confusion.
- `hyperparameter_tuning.py`: grillas y busquedas con `GridSearchCV` y
  `RandomizedSearchCV`.
- `unsupervised.py`: PCA y clustering coherentes con el dataset actual del
  proyecto.

La capa operativa por fases agrega:

- `preprocess.py`
- `unsupervised.py`
- `tune.py` (incluye Optuna como extension opcional del flujo)
- `train.py`
- `evaluate.py`

## Datos

- Dataset crudo: `data/raw/healthcare-dataset-stroke-data.csv`
- Dataset procesado auxiliar: `data/processed/`

El flujo principal actual trabaja desde el dataset crudo y aplica
preprocesamiento modular desde `src/`. El flujo por fases persiste artefactos
intermedios en `data/processed/` para asegurar reproducibilidad.

## Resultados ya disponibles

En `results/` ya existen evidencias generadas del proyecto:

- comparacion de modelos base
- matrices de confusion
- curvas ROC
- comparacion antes y despues del tuning
- graficos de EDA y clustering
- reportes finales e interpretabilidad de variables

## Modelos serializados

El proyecto deja evidencia en dos niveles:

- `models/trained_models/`: artefactos del smoke test modular.
- `models/`: artefactos del flujo orquestado por fases.

## Ejecucion recomendada

El punto de entrada oficial del proyecto es `setup_and_run.py`.

Revisar el estado del repositorio:

```bash
python setup_and_run.py --mode status --venv-name venv --skip-install
```

Ejecutar el pipeline completo por fases:

```bash
python setup_and_run.py --mode run --venv-name venv --skip-install
```

Ejecutar un chequeo supervisado rapido:

```bash
python setup_and_run.py --mode smoke-test --venv-name venv --skip-install
```

Para abrir los notebooks:

```bash
jupyter notebook
```

## Dependencias

Instalar con:

```bash
pip install -r requirements.txt
```

## Estado actual

El proyecto ya cuenta con:

- EDA con PCA y K-Means
- pipelines supervisados
- evaluacion comparativa con `Recall`, `F1` y `ROC-AUC`
- tuning con `GridSearchCV` y `RandomizedSearchCV`, con Optuna como extension del flujo por fases
- persistencia de metricas, graficos, reportes y modelo entrenado
- interpretabilidad via `feature_importance` y SHAP opcional en evaluacion

La conclusion central del proyecto es que `LogisticRegression` balanceada se
mantiene como el modelo mas defendible para screening inicial de ACV: detecta
la mayoria de los positivos reales, aunque a costa de una `Precision` modesta y
mayor cantidad de falsos positivos.
