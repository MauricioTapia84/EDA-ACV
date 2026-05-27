# Guion de Diapositivas - Fase 2 ACV

Formato: slide por slide, con bullets listos para copiar en la PPT.

## Slide 1 - Portada Fase 2
Titulo:
- Fase 2: Modelado Predictivo en Accidentes Cerebrovasculares

Subtitulo:
- Programacion para la Ciencia de Datos
- Proyecto EDA-ACV

Bullets:
- Integrantes del equipo
- Profesor y seccion
- Fecha de presentacion

## Slide 2 - Objetivo de la Fase 2
Titulo:
- Objetivo de modelado

Bullets:
- Construir y comparar modelos supervisados para predecir `stroke`.
- Priorizar deteccion de casos positivos por desbalance de clases.
- Entregar evaluacion final reproducible con evidencia cuantitativa.

Mensaje del presentador:
- "El foco no es accuracy global, sino sensibilidad clinica para no perder positivos." 

## Slide 3 - Contexto del problema
Titulo:
- Contexto y criterio de evaluacion

Bullets:
- Clase positiva de ACV cercana a 4.9% del total.
- Accuracy aislada puede ser enganosa en datos desbalanceados.
- Metricas prioritarias: Recall positivo, F1 y ROC-AUC.

Mensaje del presentador:
- "En este dominio, un falso negativo pesa mas que un falso positivo." 

## Slide 4 - Arquitectura de trabajo Fase 2
Titulo:
- Flujo tecnico implementado

Bullets:
- Pipeline con preprocesamiento + clasificador.
- Validacion cruzada estratificada.
- Separacion de fases: entrenamiento, tuning, evaluacion, reporte.
- Ejecucion orquestada y reproducible.

## Slide 5 - Modelos comparados
Titulo:
- Modelos supervisados evaluados

Bullets:
- Logistic Regression
- SVC
- Random Forest
- Mismo esquema de evaluacion para comparabilidad justa.

## Slide 6 - Optimizacion de hiperparametros
Titulo:
- Tuning segun rubrica

Bullets:
- GridSearchCV aplicado para busqueda sistematica.
- RandomizedSearchCV aplicado para exploracion eficiente.
- Optuna integrado como extension comparativa del pipeline.
- Evidencia: `results/metrics/tuned_model_comparison.csv`.

## Slide 7 - Mejor configuracion encontrada
Titulo:
- Seleccion final de modelo

Bullets:
- Mejor modelo: Logistic Regression balanceada.
- Parametros clave: `solver=liblinear`, `penalty=l2`, `C=0.0010907785690006091`, `class_weight=balanced`.
- Evidencia: `models/best_params.json`.

Mensaje del presentador:
- "La configuracion favorece sensibilidad en positivos, alineada al objetivo clinico." 

## Slide 8 - Resultados holdout (comparacion)
Titulo:
- Desempeno de modelos tuned en test holdout

Bullets (tabla sugerida):
- Logistic Regression: precision 0.13399 | recall 0.82 | f1 0.23034 | roc_auc 0.84154
- SVC: precision 0.12342 | recall 0.78 | f1 0.21311 | roc_auc 0.82533
- Random Forest: precision 0.12037 | recall 0.78 | f1 0.20856 | roc_auc 0.81907

Mensaje clave:
- "Logistic Regression lidera en recall y AUC dentro del conjunto comparado." 

## Slide 9 - Evaluacion final del modelo seleccionado
Titulo:
- Resultado final consolidado

Bullets:
- Precision positiva: 0.1102
- Recall positiva: 0.8200
- F1 positiva: 0.1943
- ROC-AUC: 0.8369
- Matriz de confusion: TN=641, FP=331, FN=9, TP=41

## Slide 10 - Lectura clinica del trade-off
Titulo:
- Interpretacion del trade-off

Bullets:
- Ventaja: se minimizan falsos negativos (FN=9).
- Costo: aumentan falsos positivos (FP=331).
- Implicancia: modelo apto para screening, no para diagnostico unico.

Mensaje del presentador:
- "Preferimos alertar de mas antes que omitir casos potencialmente criticos." 

## Slide 11 - Interpretabilidad
Titulo:
- Explicabilidad del modelo

Bullets:
- Feature importance disponible en `reports/feature_importance.csv`.
- Variables principales aportan trazabilidad de decision.
- SHAP considerado como analisis opcional segun entorno.

## Slide 12 - Evidencia de cumplimiento de rubrica
Titulo:
- Cumplimiento formal

Bullets:
- Notebooks 01-05 presentes y alineados al flujo.
- Modulos `src` requeridos implementados.
- Modelos serializados y reportes generados.
- Verificacion documentada en `docs/verificacion_rubrica_pdf.md`.

## Slide 13 - Reproducibilidad
Titulo:
- Ejecucion reproducible

Bullets:
- Comando principal: `python setup_and_run.py --mode run --venv-name .venv --skip-install`
- Validaciones: `status`, `compat`, `smoke-test`.
- Artefactos generados en `data/processed/`, `models/` y `reports/`.

Nota para pie de slide:
- "Se uso `.venv` como entorno estable en la validacion integral." 

## Slide 14 - Conclusiones finales
Titulo:
- Conclusiones Fase 2

Bullets:
- El proyecto cumple los componentes tecnicos exigidos por rubrica.
- Logistic Regression balanceada es la opcion mas defendible para tamizaje.
- El enfoque prioriza sensibilidad en positivos por criticidad del problema.
- Queda como mejora futura la calibracion de umbral para optimizar precision/recall.

## Slide 15 - Proximos pasos
Titulo:
- Continuidad del proyecto

Bullets:
- Ajuste de umbral de clasificacion segun criterio clinico-operativo.
- Calibracion probabilistica para mejorar interpretacion de riesgo.
- Monitoreo de drift y reentrenamiento periodico.
- Evaluar SHAP completo cuando el entorno lo permita.

## Slide 16 - Cierre
Titulo:
- Gracias

Bullets:
- Preguntas del jurado
- Repositorio y evidencia disponible
- Contacto del equipo
