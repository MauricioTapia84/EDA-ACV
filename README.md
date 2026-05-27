# Proyecto ACV - EP2 SCY1101

Proyecto de analisis y modelado predictivo sobre accidentes cerebrovasculares
(ACV), organizado segun la estructura pedida en la rubrica del curso.

## Objetivo

Analizar el dataset de ACV, desarrollar modelos supervisados y no supervisados,
comparar su rendimiento con validacion cruzada estratificada y optimizar
hiperparametros con herramientas permitidas por la rubrica.

## Estructura principal

- `notebooks/`: capa narrativa del proyecto.
- `src/`: logica modular reutilizable.
- `results/metrics/`: tablas y archivos de metricas.
- `results/plots/`: graficos generados por los notebooks y scripts.
- `results/reports/`: espacio para reportes finales.
- `models/trained_models/`: modelos serializados como evidencia de entrenamiento.

La explicacion detallada de cada modulo esta en
[docs/estructura_proyecto.md](docs/estructura_proyecto.md).

## Flujo recomendado

1. Ejecutar `notebooks/01_exploratory_analysis.ipynb`.
2. Ejecutar `notebooks/02_supervised_modeling.ipynb`.
3. Ejecutar `notebooks/3_model_evaluation.ipynb`.
4. Ejecutar `notebooks/04_hyperparameter_optimization.ipynb`.
5. Completar `notebooks/05_final_analysis.ipynb`.

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

## Datos

- Dataset crudo: `data/raw/healthcare-dataset-stroke-data.csv`
- Dataset procesado auxiliar: `data/processed/`

El flujo principal actual trabaja desde el dataset crudo y aplica
preprocesamiento modular desde `src/`.

## Resultados ya disponibles

En `results/` ya existen evidencias generadas del proyecto:

- comparacion de modelos base
- matrices de confusion
- curvas ROC
- comparacion antes y despues del tuning
- graficos de EDA y clustering

## Modelo serializado

`main.py` ejecuta un chequeo del proyecto y serializa el mejor modelo base en
`models/trained_models/`.

## Ejecucion rapida

Con el entorno virtual activo:

```bash
python main.py
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
- tuning con metodos permitidos por la rubrica
- persistencia de metricas, graficos y modelo entrenado

La principal tarea pendiente de cierre academico es completar
`05_final_analysis.ipynb`.
