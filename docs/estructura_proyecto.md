# Estructura del Proyecto ACV

Este proyecto se organiza en capas para separar exploración, modelado, evaluación y documentación de resultados.

## 1. `notebooks/`

Los notebooks son la capa narrativa del proyecto. Sirven para ejecutar el análisis en un orden lógico y dejar trazabilidad visual de lo que se hizo.

- `01_exploratory_analysis.ipynb`: exploración inicial, gráficos y análisis no supervisado.
- `02_supervised_modeling.ipynb`: construcción de pipelines supervisados y evaluación base.
- `3_model_evaluation.ipynb`: comparación de modelos, métricas, curvas ROC y matrices de confusión.
- `04_hyperparameter_optimization.ipynb`: tuning con `GridSearchCV` y `RandomizedSearchCV`.
- `05_final_analysis.ipynb`: cierre técnico, conclusiones y recomendación final.

## 2. `src/`

Esta carpeta contiene el código reutilizable. Aquí vive la lógica que los notebooks importan.

- `preprocess.py` y `data_preprocessing.py`: transformers de limpieza y preparación.
- `train.py` y `model_training.py`: instanciación de modelos base.
- `evaluate.py` y `model_evaluation.py`: validación cruzada y métricas.
- `tune.py` y `hyperparameter_tuning.py`: búsqueda de hiperparámetros.

## 3. `results/`

Esta carpeta guarda los resultados generados por los análisis.

- `results/metrics/`: archivos CSV y JSON con métricas, comparaciones y parámetros.
- `results/plots/`: figuras y gráficas generadas en el análisis.
- `results/reports/`: reportes finales o documentos de salida.

Estas carpetas son necesarias porque permiten separar código de evidencias, y facilitan la revisión del proyecto.

## 4. `models/trained_models/`

Aquí se guardan modelos serializados si se decide persistir el mejor clasificador para uso posterior.

## 5. Flujo recomendado

1. Ejecutar `01_exploratory_analysis.ipynb` para entender los datos.
2. Ejecutar `02_supervised_modeling.ipynb` para comparar modelos base.
3. Ejecutar `3_model_evaluation.ipynb` para revisar métricas y curvas.
4. Ejecutar `04_hyperparameter_optimization.ipynb` para mejorar los modelos.
5. Ejecutar `05_final_analysis.ipynb` para cerrar con una recomendación.

## 6. Qué significa el desempeño

En este proyecto importa más `Recall` que `Accuracy`, porque el dataset está desbalanceado y perder un caso positivo de ACV es más grave que marcar falsos positivos.

- `Recall`: cuántos casos positivos reales detecta el modelo.
- `Precision`: cuántos de los casos detectados realmente eran positivos.
- `F1`: balance entre `Precision` y `Recall`.
- `ROC-AUC`: capacidad general para separar clases.
