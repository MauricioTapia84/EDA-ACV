# 🚀 Plan de Trabajo Colaborativo - Fase 2 (Modelado)

Para avanzar eficientemente, la Fase 2 se divide en tres frentes de trabajo paralelos siguiendo la estructura de 6 pasos.

## 🟢 Parte 1: Auditoría y Preprocesamiento (Fase 0 y 1)
**Responsable:** Miembro A + **@eda-auditor**
- **Objetivo**: Garantizar la calidad de los datos antes de modelar.
- **Tareas**:
  - Validar colinealidad (VIF) de las variables.
  - Ejecutar limpieza final y guardado en `data/processed/`.
- **Archivos**: 
  - `src/0_audit/audit_data.py`
  - `src/1_prep/preprocess.py`

## 🔵 Parte 2: Modelado No Supervisado y PCA (Fase 2)
**Responsable:** Miembro B + **@unsupervised-modeler**
- **Objetivo**: Descubrir patrones ocultos y reducir dimensionalidad.
- **Tareas**:
  - Ejecutar PCA para visualización de clusters.
  - Modelado con K-Means y DBSCAN (usar Índice de Silueta).
- **Archivo**: 
  - `src/2_unsupervised/clustering_analysis.py`

## 🟡 Parte 3: Optimización de Hiperparámetros (Fase 3)
**Responsable:** Miembro C + **@hyperparameter-optimizer**
- **Objetivo**: Encontrar los mejores parámetros para modelos supervisados.
- **Tareas**:
  - Configurar estudios de **Optuna** para XGBoost, Random Forest y Logística.
  - Generar diccionarios de mejores parámetros.
- **Archivo**: 
  - `src/3_optuna/hyperparameter_tuning.py`

---
**Coordinación Global**: Todos los resultados deben ser documentados en `docs/` y los reportes finales generados por el agente **@report-generator**.
