# 🤖 Ecosistema de Agentes - Proyecto EDA-ACV (Fase 2)

Este archivo entrena a los agentes para seguir la estructura de 6 pasos y la modularidad del repositorio del profesor.

## 🏗️ @env-architect (Arquitectura y Despliegue)

- **Misión**: Mantener `setup_and_run.py` y la estructura de carpetas.
- **Regla**: Todo script nuevo debe ir en su carpeta correspondiente de `src/`.

## 🔍 @eda-auditor (Fase 0: Auditoría)

- **Misión**: Validar que los datos en `data/processed/` estén listos.
- **Instrucción**: Buscar colinealidad (VIF) y fugas de datos antes de modelar.

## 🧬 @unsupervised-modeler (Fase 2: Clustering)

- **Misión**: Ejecutar PCA, K-Means, KNN y DBSCAN.
- **Técnica**: Usar obligatoriamente Índice de Silueta y Método del Codo para justificar clusters.

## 🧪 @hyperparameter-optimizer (Fase 3: Optuna)

- **Misión**: Encontrar los mejores parámetros para Logística, SVM, RF, XGB y LGBM.
- **Referencia**: Seguir la lógica modular de `codon-classification-pipeline`.

## 📊 @report-generator (Fase 5: Documentación)

- **Misión**: Generar archivos en `docs/` y reportes en `reports/`.

---

## 🛠️ Comandos Globales
- **Inicialización**: `python3 setup_and_run.py` (Crea entorno y carpetas).
- **Ambiente Python**: Usar kernel `env_acv` (basado en `.venv/`).
- **Pipeline Modular**: 
  1. `src/0_audit/`
  2. `src/1_prep/`
  3. `src/2_unsupervised/`
  4. `src/3_optuna/`
  5. `src/4_train/`
  6. `src/5_report/`

## 🧹 Reglas de Oro
1. **Inmutabilidad**: NUNCA tocar `data/raw/`.
2. **Modularidad**: Ningún script puede quedar fuera de las carpetas 0-5.
3. **Documentación**: Cada fase debe tener su reporte correspondiente en `docs/`.

