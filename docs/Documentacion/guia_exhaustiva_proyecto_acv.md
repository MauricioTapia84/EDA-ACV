# Guia Exhaustiva del Proyecto ACV (Fase 2)

## 1. Que hace este proyecto

Este proyecto construye un flujo completo de ciencia de datos para predecir riesgo de Accidente Cerebrovascular (ACV) a partir de variables clinicas y demograficas.

Objetivo practico:
- Detectar la mayor cantidad posible de casos positivos (personas con ACV) en un escenario de fuerte desbalance de clases.

Objetivo academico:
- Cumplir la rubrica de la evaluacion parcial 2, dejando evidencia reproducible de:
  - preprocesamiento
  - analisis no supervisado
  - modelado supervisado
  - tuning de hiperparametros
  - evaluacion final
  - interpretabilidad

## 2. Contexto del problema y por que importa

La variable objetivo `stroke` esta muy desbalanceada (aprox. 4.9% positivos). Esto significa que un modelo trivial que prediga "No ACV" casi siempre puede obtener buena accuracy, pero ser inutil para deteccion temprana.

Por eso el proyecto prioriza:
1. Recall positivo: porcentaje de positivos reales detectados.
2. F1 positivo: equilibrio entre precision y recall en clase positiva.
3. ROC-AUC: capacidad global de separacion del modelo.

En terminos clinicos:
- Un falso negativo (no detectar un caso real) suele ser mas costoso que un falso positivo.
- El modelo se usa como apoyo de screening, no como diagnostico unico.

## 3. Estructura del repositorio y rol de cada carpeta

### 3.1 Capa narrativa
- `notebooks/`: cuenta la historia analitica en 5 etapas.
  - `01_exploratory_analysis.ipynb`
  - `02_supervised_modeling.ipynb`
  - `03_model_evaluation.ipynb`
  - `04_hyperparameter_optimization.ipynb`
  - `05_final_analysis.ipynb`

### 3.2 Capa modular reutilizable
- `src/`: implementacion tecnica reusable.
- Modulos oficiales solicitados por pauta:
  - `src/data_preprocessing.py`
  - `src/model_training.py`
  - `src/model_evaluation.py`
  - `src/hyperparameter_tuning.py`

### 3.3 Capa operativa por fases
- Scripts base:
  - `src/preprocess.py`
  - `src/unsupervised.py`
  - `src/tune.py`
  - `src/train.py`
  - `src/evaluate.py`
- Wrappers por fase:
  - `src/0_audit/audit.py`
  - `src/1_prep/preprocess.py`
  - `src/2_unsupervised/unsupervised.py`
  - `src/3_optuna/tune.py`
  - `src/4_train/train.py`
  - `src/5_report/evaluate.py`

### 3.4 Datos y artefactos
- `data/raw/`: datos crudos (inmutable).
- `data/processed/`: datasets procesados y splits.
- `results/metrics/`: tablas de resultados.
- `results/plots/`: visualizaciones.
- `results/reports/`: reportes finales de la capa narrativa.
- `reports/`: reportes consolidados de la capa operativa.
- `models/`: modelos finales y parametros.
- `models/trained_models/`: artefactos de validaciones modulares.

## 4. Flujo metodologico completo (de inicio a fin)

## 4.1 Fase 0-1: auditoria y preprocesamiento

Se realizan tareas de calidad de datos:
1. Deteccion y tratamiento de nulos.
2. Normalizacion de categorias "Unknown" a nulos utiles para imputacion.
3. Tratamiento de outliers (capping robusto).
4. Estandarizacion y codificacion para modelos.
5. Split de entrenamiento y prueba.

Resultado esperado:
- datasets limpios y reproducibles en `data/processed/`.

## 4.2 Fase 2: analisis no supervisado

Objetivo:
- entender estructura latente de los datos antes del modelado final.

Tecnicas usadas:
1. PCA (reduccion de dimensionalidad para visualizacion).
2. KMeans y otras estrategias de agrupamiento para identificar patrones.

Resultado esperado:
- graficos en `results/plots/` y `reports/figures/`.

## 4.3 Fase 3: tuning de hiperparametros

Se comparan estrategias de busqueda:
1. GridSearchCV: explora una grilla predefinida de combinaciones.
2. RandomizedSearchCV: explora muestras aleatorias de una distribucion de parametros.
3. Optuna (extension): busqueda mas eficiente guiada por optimizacion bayesiana.

Resultado esperado:
- comparativas en `results/metrics/`.
- mejor configuracion registrada en `models/best_params.json`.

## 4.4 Fase 4: entrenamiento final

Se entrena el mejor pipeline con los parametros seleccionados.

Resultado esperado:
- modelo serializado en `models/final_model.pkl` y/o `models/best_model.pkl`.

## 4.5 Fase 5: evaluacion final e interpretabilidad

Se mide desempeno en test holdout y se documenta:
1. metricas principales
2. matriz de confusion
3. reporte de clasificacion
4. importancia de variables

Resultado esperado:
- `reports/evaluation_results.md`
- `reports/classification_report.txt`
- `reports/feature_importance.csv`

## 5. Definiciones clave (glosario practico)

## 5.1 Variables y dataset
- Variable objetivo: `stroke` (0: no ACV, 1: ACV).
- Variables predictoras: edad, glucosa, IMC, hipertension, cardiopatia, etc.
- Desbalance de clases: distribucion muy desigual entre clase 0 y clase 1.

## 5.2 Conceptos de ML usados
- Pipeline: cadena ordenada de transformaciones + modelo, ejecutada siempre igual.
- Validacion cruzada estratificada: divide datos en folds conservando proporcion de clases.
- Hiperparametros: configuraciones del modelo que no se aprenden de los datos (ejemplo: C en Logistic Regression, profundidad en Random Forest).
- Holdout test: subconjunto separado que solo se usa al final para medir desempeno real.

## 5.3 Modelos entrenados y que significan

### Logistic Regression
- Modelo lineal para clasificacion binaria.
- Ventajas:
  - estable
  - interpretable
  - buen baseline en datos tabulares
- En este proyecto usa `class_weight=balanced` para compensar desbalance.

### SVC (Support Vector Classifier)
- Busca una frontera de separacion con maximo margen entre clases.
- Puede capturar fronteras no lineales (segun kernel).
- Suele requerir buen escalado y ajuste fino de hiperparametros.

### Random Forest
- Ensamble de multiples arboles de decision.
- Robusto y flexible para relaciones no lineales.
- Puede degradarse en recall positivo si no se ajusta bien en escenarios fuertemente desbalanceados.

## 5.4 Metricas y como leerlas

- Precision (clase positiva): de todo lo predicho como positivo, cuantos eran positivos reales.
- Recall (clase positiva): de todos los positivos reales, cuantos fueron detectados.
- F1 (clase positiva): promedio armonico entre precision y recall.
- ROC-AUC: capacidad del modelo para rankear positivos por sobre negativos en distintos umbrales.
- Accuracy: proporcion total de aciertos; en este caso no es metrica principal por desbalance.

## 5.5 Matriz de confusion

Se interpreta asi:
- TN: negativos reales bien clasificados.
- FP: negativos reales clasificados como positivos.
- FN: positivos reales clasificados como negativos.
- TP: positivos reales bien clasificados.

En screening de ACV:
- FN bajo es prioritario.
- Se acepta mayor FP si eso mejora deteccion de positivos reales.

## 6. Resultados del proyecto y su significado

## 6.1 Resultados comparativos (holdout tuned)

Fuente: `results/metrics/tuned_holdout_metrics.csv`

- logistic_regression: precision=0.13399, recall=0.82, f1=0.23034, roc_auc=0.84154
- svc: precision=0.12342, recall=0.78, f1=0.21311, roc_auc=0.82533
- random_forest: precision=0.12037, recall=0.78, f1=0.20856, roc_auc=0.81907

Lectura:
1. Logistic Regression logra la mejor combinacion de recall y AUC.
2. SVC y Random Forest quedan por debajo en este set de evaluacion.
3. Se justifica seleccionar Logistic Regression como modelo principal.

## 6.2 Evaluacion final consolidada

Fuente: `reports/evaluation_results.md`

- precision (positiva): 0.1102
- recall (positiva): 0.8200
- f1 (positiva): 0.1943
- roc_auc: 0.8369

Matriz:
- TN=641
- FP=331
- FN=9
- TP=41

Lectura en contexto:
1. El modelo detecta 82% de los casos positivos reales (recall alto).
2. Solo 9 positivos reales quedan sin detectar (FN=9), lo cual es favorable para screening.
3. Hay muchos falsos positivos (FP=331), por lo que no debe usarse como diagnostico final.

## 6.3 Impacto practico del trade-off

- Beneficio:
  - mayor sensibilidad para detectar pacientes de riesgo.
- Costo:
  - mas alertas falsas, que requieren evaluacion clinica posterior.

Decision recomendada:
- usar el modelo como filtro de priorizacion (triage), no como decision clinica definitiva.

## 7. Interpretabilidad del modelo

Evidencia:
- `reports/feature_importance.csv`
- `reports/evaluation_results.md`

Que aporta:
1. Permite explicar que variables empujan mas la prediccion.
2. Ayuda a defender el modelo ante publico tecnico y no tecnico.
3. Mejora trazabilidad de decisiones en contexto academico y aplicado.

Nota:
- SHAP se considera complementario, pero puede depender del entorno de ejecucion.

## 8. Cumplimiento de rubrica

Referencia principal:
- `docs/verificacion_rubrica_pdf.md`

Elementos cubiertos:
1. Notebooks 01-05 presentes y operativos.
2. Modulos oficiales de `src` presentes.
3. Evidencia de GridSearchCV y RandomizedSearchCV.
4. Modelos serializados y resultados persistidos.
5. Reportes de evaluacion final.
6. Flujo reproducible por comando.

## 9. Reproducibilidad: como correr el proyecto

Punto de entrada:
- `setup_and_run.py`

Comandos principales:
1. Estado general:
   - `python setup_and_run.py --mode status --venv-name .venv --skip-install`
2. Compatibilidad:
   - `python setup_and_run.py --mode compat --venv-name .venv --skip-install`
3. Ejecucion completa:
   - `python setup_and_run.py --mode run --venv-name .venv --skip-install`
4. Verificacion rapida:
   - `python setup_and_run.py --mode smoke-test --venv-name .venv --skip-install`

Recomendacion operativa:
- usar `.venv` como entorno estable de ejecucion.

## 10. Limitaciones y mejoras futuras

Limitaciones actuales:
1. Precision positiva baja, esperable por desbalance y enfoque en recall.
2. Dependencia de calidad de variables clinicas disponibles en dataset publico.
3. SHAP no siempre disponible segun entorno.

Mejoras sugeridas:
1. Calibracion de umbral para ajustar trade-off recall/precision.
2. Integrar estrategias de balanceo avanzadas dentro de CV (ejemplo: SMOTE en pipeline validado sin leakage).
3. Monitorear drift de datos en nuevas corridas.
4. Evaluar calibracion probabilistica para uso operativo.

## 11. Resumen ejecutivo final

- El proyecto implementa un pipeline completo, modular y reproducible para prediccion de ACV.
- La seleccion metodologica esta alineada al contexto: priorizar recall en clase positiva.
- Logistic Regression balanceada es el modelo mas defendible con la evidencia actual.
- Los resultados son utiles para screening temprano, pero no reemplazan diagnostico medico.
- La documentacion y artefactos del repositorio permiten auditar y replicar el proceso de extremo a extremo.
