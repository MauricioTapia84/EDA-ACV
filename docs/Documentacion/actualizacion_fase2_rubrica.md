# Actualizacion Fase 2 segun Rubrica

Este documento resume que actualizar en:
- docs/Documentacion/Análisis Predictivo en Accidentes Cerebrovasculares (1).docx
- docs/Documentacion/PPT ACV prog. ciencia de datos.pdf

Usa solo evidencia ya generada por el proyecto en `results/`, `models/`, `reports/` y `src/`.

## 1. Diagnostico de estado actual de tus documentos

### 1.1 Informe DOCX
- Estado: bien desarrollado para EDA + limpieza + pipeline de preprocesamiento (Fase 1).
- Brecha principal: faltan secciones de Fase 2 ejecutada (modelado supervisado, tuning, evaluacion final, interpretabilidad, cierre de rubrica).
- Observacion: el documento dice "trabajos futuros" para varias tareas que ya estan implementadas (GridSearchCV, RandomizedSearchCV, interpretabilidad).

### 1.2 Presentacion PDF
- Estado: tambien centrada en diagnostico EDA, nulos, outliers y arquitectura de preprocesamiento.
- Brecha principal: no muestra resultados de comparacion de modelos, tuning ni evaluacion final en test holdout.
- Observacion: en algunas diapositivas se sugiere capping por percentiles, mientras el pipeline consolidado usa enfoque IQR en varias rutas y comparacion formal de rendimiento en Fase 2.

## 2. Contenido minimo que debes agregar para cumplir Fase 2

Basado en la pauta y en `docs/verificacion_rubrica_pdf.md`:

1. Flujo narrativo completo por notebooks 01-05.
2. Evidencia de modulos de `src`:
   - `src/data_preprocessing.py`
   - `src/model_training.py`
   - `src/model_evaluation.py`
   - `src/hyperparameter_tuning.py`
3. Evidencia de tuning con GridSearchCV y RandomizedSearchCV.
4. Evidencia de artefactos:
   - `models/trained_models/`
   - `results/metrics/`, `results/plots/`, `results/reports/`
   - `models/final_model.pkl` y/o `models/best_model.pkl`
5. Evaluacion final del modelo seleccionado en holdout/test.
6. Interpretabilidad basica (feature importance; SHAP opcional si entorno lo permite).
7. Reproducibilidad con comando de ejecucion.

## 3. Texto sugerido para insertar en el informe DOCX

## 3.1 Nueva seccion: Modelado Supervisado (Fase 2)

"En la Fase 2 se entrenaron y compararon modelos supervisados con validacion cruzada estratificada sobre un pipeline reproducible. Dado el desbalance de clases de la variable `stroke` (~4.9% positivos), la seleccion de modelo priorizo `Recall` de la clase positiva y `F1`, en lugar de accuracy como metrica principal."

## 3.2 Nueva seccion: Optimizacion de Hiperparametros

"Se aplicaron estrategias de tuning exigidas por la pauta: `GridSearchCV` y `RandomizedSearchCV` (evidencia en `results/metrics/tuned_model_comparison.csv`). Adicionalmente, en la capa operativa por fases se integro `Optuna` para contrastar busqueda clasica vs busqueda bayesiana en el modelo logistico (evidencia en `models/best_params.json` y `models/optuna_study.csv`)."

## 3.3 Nueva seccion: Resultados cuantitativos Fase 2

Tabla recomendada (usar estos valores):

- Holdout tuned (`results/metrics/tuned_holdout_metrics.csv`):
  - logistic_regression: precision=0.13399, recall=0.82, f1=0.23034, roc_auc=0.84154
  - svc: precision=0.12342, recall=0.78, f1=0.21311, roc_auc=0.82533
  - random_forest: precision=0.12037, recall=0.78, f1=0.20856, roc_auc=0.81907

Conclusiones tecnicas sugeridas:
- "Logistic Regression balanceada se mantiene como mejor compromiso para screening: mayor recall y mejor AUC en holdout."
- "El costo operacional es una precision baja en clase positiva, con incremento de falsos positivos, coherente con objetivo clinico de minimizar falsos negativos."

## 3.4 Nueva seccion: Evaluacion final e interpretabilidad

Usar reporte final (`reports/evaluation_results.md`):
- precision (clase positiva): 0.1102
- recall (clase positiva): 0.8200
- f1 (clase positiva): 0.1943
- roc_auc: 0.8369
- matriz de confusion: TN=641, FP=331, FN=9, TP=41

Texto sugerido:
"La evaluacion final confirma una configuracion orientada a sensibilidad para deteccion de ACV, reduciendo falsos negativos (FN=9) a costa de una mayor tasa de falsos positivos (FP=331). Por ello, el modelo se propone como apoyo de tamizaje y no como diagnostico unico."

Interpretabilidad:
"Se incorporo `feature_importance` en `reports/feature_importance.csv` como mecanismo de trazabilidad del modelo. SHAP queda habilitado como analisis opcional segun disponibilidad del entorno."

## 3.5 Nueva seccion: Cumplimiento de Rubrica y Reproducibilidad

Texto sugerido:
"El cumplimiento de pauta se verifica en `docs/verificacion_rubrica_pdf.md`, incluyendo estructura de notebooks, modulos `src`, artefactos de modelos y resultados. La ejecucion reproducible se realiza con:
`python setup_and_run.py --mode run --venv-name .venv --skip-install`"

Agregar nota operativa:
"Se detecto inconsistencia de permisos en `venv/`; el entorno funcional validado para ejecucion integral es `.venv/`."

## 4. Actualizacion sugerida para la PPT

Agregar 5 diapositivas nuevas despues de la actual seccion de EDA/preprocesamiento:

1. "Fase 2: Modelado Supervisado"
- Modelos evaluados: Logistic Regression, SVC, Random Forest.
- Esquema: Pipeline + Stratified CV.
- Criterio: priorizar recall de positivos.

2. "Tuning y Seleccion"
- GridSearchCV y RandomizedSearchCV (rubrica).
- Optuna como extension comparativa.
- Mejor configuracion: logistic_regression balanceada (`models/best_params.json`).

3. "Resultados Holdout"
- Tabla comparativa con los 3 modelos y metricas de `tuned_holdout_metrics.csv`.
- Mensaje clave: Logistic lidera en recall y AUC.

4. "Evaluacion Final del Modelo"
- Recall 0.82, AUC 0.8369, matriz confusion (641, 331, 9, 41).
- Interpretacion clinica: menos FN, mas FP.

5. "Cierre de Rubrica y Deployment Academico"
- Checklist cumplido: notebooks 01-05, src oficial, modelos, results, reproducibilidad.
- Recomendacion: uso como screening + recalibracion de umbral en futuros ciclos.

## 5. Inconsistencias a corregir explicitamente

1. Cambiar redaccion de "trabajos futuros" cuando la tarea ya esta implementada (tuning e interpretabilidad basica).
2. Homologar terminologia de metrica prioritaria: `Recall` y `F1` para clase positiva.
3. Homologar ruta de ejecucion oficial y entorno (`.venv` como estable si `venv` sigue con permisos incorrectos).
4. Evitar afirmar solo precision/accuracy como criterio principal por desbalance.

## 6. Checklist final antes de entregar

- [ ] Informe DOCX incluye secciones nuevas de Fase 2.
- [ ] PPT incluye al menos 5 diapositivas de Fase 2 con metricas reales.
- [ ] Todas las metricas coinciden con archivos en `results/metrics/` y `reports/`.
- [ ] Se menciona evidencia de rubrica en `docs/verificacion_rubrica_pdf.md`.
- [ ] Se incluye comando de reproduccion ejecutable.
