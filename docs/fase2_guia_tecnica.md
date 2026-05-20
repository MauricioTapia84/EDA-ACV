# 📋 Guía Técnica: Transición a Fase 2 (Modelado)

## 🛠️ Paso a Paso Técnico

### 0. Auditoría (Quality Gate)

- **Acción**: Validar que el dataset en `data/processed/` no tenga fugas de datos.

### 1. Pipeline de Preprocesamiento

- **Acción**: Crear un pipeline de `sklearn` que maneje imputación, escalado robusto y encoding.

### 2. Segmentación No Supervisada

- **Acción**: Aplicar PCA y comparar K-Means vs DBSCAN usando el **Índice de Silueta**.

### 3. Optimización con Optuna

- **Acción**: Búsqueda Bayesian de hiperparámetros para Logística, SVM, RF, XGB y LGBM.

### 4. Evaluación y Reporte

- **Acción**: Generar matriz de confusión y reporte de clasificación.
