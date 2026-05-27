# Informe Final Fase 2 - Proyecto ACV

## 1. Introduccion

Este informe presenta la actualizacion de la Fase 2 del proyecto de analisis predictivo de accidentes cerebrovasculares (ACV), integrando modelado supervisado, optimizacion de hiperparametros, evaluacion final en test holdout e interpretabilidad. El desarrollo se alinea con la rubrica academica y con una logica metodologica centrada en deteccion oportuna de casos positivos.

Dado que la variable objetivo `stroke` presenta un desbalance severo (aprox. 4.9% de positivos), la estrategia de evaluacion prioriza `Recall` y `F1` de la clase positiva por sobre `Accuracy`, con foco en minimizar falsos negativos.

## 2. Objetivo de la Fase 2

El objetivo de esta fase fue construir, comparar y validar modelos de clasificacion binaria para estimar riesgo de ACV, cumpliendo los siguientes puntos:

1. Entrenar modelos supervisados en pipeline reproducible.
2. Ejecutar tuning con tecnicas exigidas por pauta (`GridSearchCV` y `RandomizedSearchCV`).
3. Evaluar desempeno final en holdout/test.
4. Entregar evidencia de artefactos, metricas, reportes y modelo serializado.
5. Incorporar interpretabilidad basica para justificar decisiones de modelado.

## 3. Metodologia de modelado supervisado

Se implemento una arquitectura de pipeline con validacion cruzada estratificada para controlar sesgo por desbalance y mantener consistencia en preprocesamiento y entrenamiento. Los modelos evaluados fueron:

- Logistic Regression
- SVC
- Random Forest

La comparacion de modelos se realizo mediante metricas de clasificacion con enfasis en `Recall` positivo, `F1` y `ROC-AUC`.

## 4. Optimizacion de hiperparametros

Para cumplir con los criterios de rubrica, se aplicaron dos enfoques de optimizacion:

1. `GridSearchCV`
2. `RandomizedSearchCV`

Evidencia:
- `results/metrics/tuned_model_comparison.csv`

Como extension metodologica, se incluyo optimizacion con Optuna en la capa operativa por fases, con evidencia en:
- `models/best_params.json`
- `models/optuna_study.csv`

Mejor configuracion registrada:
- Modelo: `logistic_regression`
- Parametros: `solver=liblinear`, `penalty=l2`, `C=0.0010907785690006091`, `class_weight=balanced`
- Metodo de busqueda: `OptunaStudy`

## 5. Resultados cuantitativos (holdout tuned)

Resultados de `results/metrics/tuned_holdout_metrics.csv`:

- Logistic Regression: precision=0.13399, recall=0.82, f1=0.23034, roc_auc=0.84154
- SVC: precision=0.12342, recall=0.78, f1=0.21311, roc_auc=0.82533
- Random Forest: precision=0.12037, recall=0.78, f1=0.20856, roc_auc=0.81907

Interpretacion tecnica:

1. Logistic Regression balanceada presenta el mejor rendimiento global para el objetivo de screening, liderando en recall y AUC.
2. El enfoque favorece deteccion de positivos, aceptando mayor costo en falsos positivos.
3. El comportamiento observado es coherente con el desbalance estructural del dataset.

## 6. Evaluacion final del modelo seleccionado

Reporte final (`reports/evaluation_results.md`):

- Precision (clase positiva): 0.1102
- Recall (clase positiva): 0.8200
- F1 (clase positiva): 0.1943
- ROC-AUC: 0.8369

Matriz de confusion:
- TN=641
- FP=331
- FN=9
- TP=41

Lectura de negocio/clinica:

- El modelo reduce significativamente los falsos negativos (FN=9), que son el error mas critico para un escenario de deteccion temprana.
- El costo es un numero elevado de falsos positivos (FP=331), por lo que el modelo debe utilizarse como apoyo de tamizaje y no como diagnostico unico.

## 7. Interpretabilidad

Se incorporo interpretabilidad basada en importancia de variables:

- Evidencia: `reports/feature_importance.csv`
- Reporte integrado: `reports/evaluation_results.md`

Adicionalmente, se considero SHAP como mecanismo opcional dependiente del entorno. En la corrida validada, SHAP no estuvo disponible y se documento explicitamente.

## 8. Cumplimiento de rubrica

El estado de cumplimiento se encuentra documentado en `docs/verificacion_rubrica_pdf.md`, con evidencia sobre:

1. Notebooks requeridos (`01` a `05`).
2. Modulos oficiales en `src/`:
   - `data_preprocessing.py`
   - `model_training.py`
   - `model_evaluation.py`
   - `hyperparameter_tuning.py`
3. Modelos serializados en `models/trained_models/` y `models/`.
4. Resultados en `results/metrics/`, `results/plots/`, `results/reports/`.
5. Reproducibilidad operativa del flujo.

## 9. Reproducibilidad

Comando recomendado de ejecucion integral:

`python setup_and_run.py --mode run --venv-name .venv --skip-install`

Comandos de validacion de estado:

- `python setup_and_run.py --mode status --venv-name .venv --skip-install`
- `python setup_and_run.py --mode compat --venv-name .venv --skip-install`
- `python setup_and_run.py --mode smoke-test --venv-name .venv --skip-install`

Nota operativa:

Se detecto inconsistencia de permisos en `venv/`. El entorno funcional validado para ejecucion completa fue `.venv/`.

## 10. Conclusiones

1. El proyecto cumple los elementos tecnicos de la Fase 2 exigidos por rubrica.
2. Logistic Regression balanceada es el modelo mas defendible para screening inicial de ACV en este contexto de desbalance.
3. El enfoque maximiza deteccion de positivos (alto recall), sacrificando precision en clase positiva.
4. La solucion es util como apoyo a la priorizacion clinica, pero no reemplaza evaluacion medica ni confirmacion diagnostica.
5. Como siguiente mejora, se recomienda calibrar umbral de decision para optimizar trade-off recall/precision segun criterio operativo.
