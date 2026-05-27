# 🚀 Bitácora de Progreso - Fase 2: Modelado

Este documento registra el progreso del proyecto ACV siguiendo la estructura modular del repositorio de referencia.

## 📋 Road Map: Flujo de la Fase 2

### 1. Auditoría y Preprocesamiento (`preprocess.py`)

- **Limpieza**: Manejo de valores nulos (`SmartImputer`) y outliers (`OutlierCapper`).
- **Ingeniería**: Codificación de variables categóricas y escalamiento.
- **Auditoría**: Cálculo de VIF (Variance Inflation Factor) para detectar multicolinealidad.
- **Output**: `data/processed/data_audited.csv`

### 2. Análisis No Supervisado (`unsupervised.py`)

- **Reducción**: Aplicación de PCA o t-SNE para visualización de clusters.
- **Clustering**: K-Means o DBSCAN para identificar patrones ocultos en los datos de ACV.
- **Output**: Visualizaciones en `reports/figures/` y etiquetas de cluster.

### 3. Ajuste de Hiperparámetros (`tune.py`)

- **Optimización**: Uso de Optuna para encontrar los mejores parámetros.
- **Prioridad**: Optimización enfocada en **Recall** (minimizar falsos negativos en detección de ACV).
- **Modelos**: XGBoost, LightGBM, RandomForest.
- **Output**: `models/best_params.json`

### 4. Entrenamiento en Pipeline (`train.py`)

- **Arquitectura**: Construcción del `sklearn.pipeline.Pipeline`.
- **CV**: Validación cruzada estratificada.
- **Exportación**: Guardado del modelo serializado (`.pkl` o `.joblib`).
- **Output**: `models/final_model.pkl`

### 5. Evaluación y Reporte (`evaluate.py`)

- **Métricas**: Matriz de confusión, Curva ROC-AUC, Precision-Recall Curve.
- **Interpretación**: Feature Importance y SHAP values.
- **Output**: `reports/evaluation_results.md`

---

## ✅ Estado Actual

- [X] Reorganización de estructura de carpetas (Fase 2).
- [X] Configuración de Agentes Especializados.
- [X] Creación de scripts base en `src/`.
- [X] Ejecución de Auditoría de Datos (`preprocess.py`).
- [X] Modelado No Supervisado.
- [X] Ajuste de hiperparámetros y selección de modelo final.
- [X] Entrenamiento y evaluación final con artefactos en `models/` y `reports/`.
- [X] Estructura por fases 0-5 poblada con scripts orquestadores.

## ⚠️ Brechas Técnicas Detectadas

- [X] Implementar script dedicado de auditoría (`src/audit.py`) o consolidar auditoría + optimización en `src/preprocess.py` con salida reproducible.
- [X] Implementar split Train/Test 80/20 explícito y persistir artefactos (`train.csv`, `test.csv` o equivalentes).
- [X] Implementar `src/unsupervised.py` con PCA + clustering (KNN/DBSCAN/Jerárquico según pauta).
- [X] Integrar Optuna en `src/tune.py` para búsqueda de hiperparámetros.
- [X] Implementar entrenamiento final en `src/train.py` con el mejor modelo e hiperparámetros.
- [X] Implementar evaluación final en Test con `classification_report` y `confusion_matrix` en `src/evaluate.py`.

## 🔁 Compatibilidad y Sincronización con Compañero

Objetivo: traer cambios remotos sin perder los cambios locales en orquestador y bitácora.

1. Guardar cambios locales críticos (incluye `main.py`, orquestador y bitácora):
   - `git stash push -m "wip-orchestrator-progress" .github/agents/data-orchestrator.agent.md progress_log.md main.py`
2. Actualizar rama local con remoto:
   - `git pull --rebase origin main`
3. Recuperar cambios locales:
   - `git stash pop`
4. Validar estado del pipeline:
   - `python3 main.py --compat`
   - `python3 main.py --status`

Nota: Si aparece conflicto en `main.py`, conservar la versión local del orquestador y re-aplicar únicamente las funciones que vengan del compañero.

## 🧭 Contexto Operativo del Orquestador

- Antes de iniciar una tarea, leer este archivo y `.github/copilot-instructions.md`.
- Delegar por especialidad y registrar cada asignación en la sección de actividad.
- Validar generación de artefactos antes de cambiar de fase.
- Solicitar commits solo mediante `@github-git-agent`, con confirmación explícita Y/n.

## 🤖 Registro de Actividad de Agentes

- [2026-05-27] @data-orchestrator: Revisión de contexto inicial y roadmap Fase 2.
- [2026-05-27] @data-orchestrator: Política de delegación y trazabilidad activada.
- [2026-05-27] @data-orchestrator: Sincronización remota completada con `git pull --rebase` preservando cambios locales mediante stash.
- [2026-05-27] @data-orchestrator: Conflicto en `main.py` resuelto fusionando flujo del compañero + orquestación por agentes.
- [2026-05-27] @data-orchestrator: Entorno compatible validado en `.venv` con pandas/numpy/scikit-learn/optuna.
- [Pendiente] @data-cleaner: Auditoría VIF y control de calidad en `src/preprocess.py`.
- [Pendiente] @stats-modeler: Ajuste de hiperparámetros en `src/tune.py` (objetivo recall).
- [Pendiente] @stats-modeler: Entrenamiento final y exportación de modelo en `src/train.py`.
- [Pendiente] @data-visualizer: Visualizaciones de `src/unsupervised.py` y `src/evaluate.py`.
- [2026-05-27 11:36:48] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 11:36:48] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/preprocess.py
- [2026-05-27 11:37:05] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 11:37:05] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/preprocess.py
- [2026-05-27 11:37:21] @data-orchestrator: Fase 2 sin artefactos esperados: data/processed/train.csv, data/processed/test.csv, data/processed/X_train.csv, data/processed/X_test.csv.
- [2026-05-27 11:37:21] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/unsupervised.py
- [2026-05-27 11:37:39] @data-orchestrator: Fase 3 sin artefactos esperados: reports/figures/pca_clusters.png, reports/figures/clustering_summary.png.
- [2026-05-27 11:37:39] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/tune.py
- [2026-05-27 11:37:43] @data-orchestrator: Fase 4 fallo con codigo 1.
- [2026-05-27 11:37:43] @data-orchestrator: Pipeline detenido por error de fase.
- [2026-05-27 11:43:34] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 11:43:34] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/preprocess.py
- [2026-05-27 11:43:39] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 11:43:39] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/preprocess.py
- [2026-05-27 11:43:44] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 11:43:44] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/unsupervised.py
- [2026-05-27 11:43:50] @data-orchestrator: Fase 3 fallo con codigo 1.
- [2026-05-27 11:43:50] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/tune.py
- [2026-05-27 11:46:30] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 11:46:30] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/preprocess.py
- [2026-05-27 11:46:35] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 11:46:35] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/preprocess.py
- [2026-05-27 11:46:40] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 11:46:40] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/unsupervised.py
- [2026-05-27 11:46:46] @data-orchestrator: Fase 3 fallo con codigo 1.
- [2026-05-27 11:46:46] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/tune.py
- [2026-05-27 11:46:52] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 11:46:52] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/train.py
- [2026-05-27 11:46:56] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 11:46:56] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/evaluate.py
- [2026-05-27 11:47:00] @data-orchestrator: Fase 6 fallo con codigo 1.
- [2026-05-27 11:47:00] @data-orchestrator: Pipeline detenido por error de fase.
- [2026-05-27 11:47:59] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 11:47:59] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/preprocess.py
- [2026-05-27 11:48:04] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 11:48:04] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/preprocess.py
- [2026-05-27 11:48:09] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 11:48:09] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/unsupervised.py
- [2026-05-27 11:48:18] @data-orchestrator: Fase 3 completada correctamente.
- [2026-05-27 11:48:18] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/tune.py
- [2026-05-27 11:48:23] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 11:48:23] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/train.py
- [2026-05-27 11:48:27] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 11:48:27] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/evaluate.py
- [2026-05-27 11:48:31] @data-orchestrator: Fase 6 completada correctamente.
- [2026-05-27 11:48:31] @data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).
- [2026-05-27 12:10:00] @data-orchestrator -> @stats-modeler: Integracion de Optuna en `src/tune.py` y comparacion con baseline completada.
- [2026-05-27 11:56:44] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 11:56:44] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/0_audit/audit.py
- [2026-05-27 11:56:49] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 11:56:49] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/1_prep/preprocess.py
- [2026-05-27 11:56:56] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 11:56:56] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/2_unsupervised/unsupervised.py
- [2026-05-27 11:57:11] @data-orchestrator: Fase 3 completada correctamente.
- [2026-05-27 11:57:11] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/3_optuna/tune.py
- [2026-05-27 11:57:16] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 11:57:16] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/4_train/train.py
- [2026-05-27 11:57:20] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 11:57:20] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/5_report/evaluate.py
- [2026-05-27 11:57:24] @data-orchestrator: Fase 6 completada correctamente.
- [2026-05-27 11:57:24] @data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).
- [2026-05-27 12:09:43] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 12:09:43] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/0_audit/audit.py
- [2026-05-27 12:09:47] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 12:09:47] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/1_prep/preprocess.py
- [2026-05-27 12:09:52] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 12:09:52] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/2_unsupervised/unsupervised.py
- [2026-05-27 12:10:01] @data-orchestrator: Fase 3 completada correctamente.
- [2026-05-27 12:10:01] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/3_optuna/tune.py
- [2026-05-27 12:10:10] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 12:10:10] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/4_train/train.py
- [2026-05-27 12:10:15] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 12:10:15] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/5_report/evaluate.py
- [2026-05-27 12:10:19] @data-orchestrator: Fase 6 completada correctamente.
- [2026-05-27 12:10:19] @data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).
- [2026-05-27 12:18:00] @data-orchestrator: Integracion de commits remotos del companero (2.8.0 y 2.9.0) iniciada con fetch y analisis diferencial.
- [2026-05-27 12:21:00] @data-orchestrator: Unificacion aplicada: flujo modular actual preservado, notebooks/evaluaciones del companero incorporados, Optuna mantenido para comparacion.
- [2026-05-27 12:23:00] @data-orchestrator: Validacion pendiente via setup_and_run.py --mode run para confirmar ejecucion end-to-end de la version unificada.
- [2026-05-27 12:27:00] @data-orchestrator: Validacion completada con `setup_and_run.py --mode run --skip-install` en `.venv` (ejecucion OK).
- [2026-05-27 12:15:34] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 12:15:34] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/0_audit/audit.py
- [2026-05-27 12:15:39] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 12:15:39] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/1_prep/preprocess.py
- [2026-05-27 12:15:44] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 12:15:44] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/2_unsupervised/unsupervised.py
- [2026-05-27 12:15:54] @data-orchestrator: Fase 3 completada correctamente.
- [2026-05-27 12:15:54] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/3_optuna/tune.py
- [2026-05-27 12:16:02] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 12:16:02] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/4_train/train.py
- [2026-05-27 12:16:06] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 12:16:06] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/5_report/evaluate.py
- [2026-05-27 12:16:11] @data-orchestrator: Fase 6 completada correctamente.
- [2026-05-27 12:16:11] @data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).
- [2026-05-27 12:24:42] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 12:24:42] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/0_audit/audit.py
- [2026-05-27 12:24:47] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 12:24:47] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/1_prep/preprocess.py
- [2026-05-27 12:24:52] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 12:24:52] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/2_unsupervised/unsupervised.py
- [2026-05-27 12:25:01] @data-orchestrator: Fase 3 completada correctamente.
- [2026-05-27 12:25:01] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/3_optuna/tune.py
- [2026-05-27 12:25:09] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 12:25:09] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/4_train/train.py
- [2026-05-27 12:25:14] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 12:25:14] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/5_report/evaluate.py
- [2026-05-27 12:25:18] @data-orchestrator: Fase 6 completada correctamente.
- [2026-05-27 12:25:18] @data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).
- [2026-05-27 12:33:00] @data-orchestrator: Auditoria de cumplimiento de rubrica realizada: flujo operativo validado con setup_and_run (status/compat/run), brechas documentadas para cierre academico.
- [2026-05-27 12:45:00] @data-orchestrator: Cierre de brechas aplicado: README coherente, `docs/estructura_proyecto.md` y `docs/verificacion_rubrica_pdf.md` creados, interpretabilidad explicita agregada en evaluacion, narrativa reforzada en notebooks 02/03/05.
- [2026-05-27 12:33:18] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 12:33:18] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/0_audit/audit.py
- [2026-05-27 12:33:22] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 12:33:22] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/1_prep/preprocess.py
- [2026-05-27 12:33:27] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 12:33:27] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/2_unsupervised/unsupervised.py
- [2026-05-27 12:33:36] @data-orchestrator: Fase 3 completada correctamente.
- [2026-05-27 12:33:36] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/3_optuna/tune.py
- [2026-05-27 12:33:45] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 12:33:45] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/4_train/train.py
- [2026-05-27 12:33:49] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 12:33:49] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/5_report/evaluate.py
- [2026-05-27 12:33:53] @data-orchestrator: Fase 6 completada correctamente.
- [2026-05-27 12:33:53] @data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).
- [2026-05-27 12:41:51] @data-orchestrator: Inicio de ejecucion del pipeline de Fase 2.
- [2026-05-27 12:41:51] @data-orchestrator -> @data-cleaner: Fase 1 Auditoria y optimizacion de datos en src/0_audit/audit.py
- [2026-05-27 12:41:56] @data-orchestrator: Fase 1 completada correctamente.
- [2026-05-27 12:41:56] @data-orchestrator -> @data-cleaner: Fase 2 Preprocesamiento y split Train/Test en src/1_prep/preprocess.py
- [2026-05-27 12:42:01] @data-orchestrator: Fase 2 completada correctamente.
- [2026-05-27 12:42:01] @data-orchestrator -> @data-visualizer: Fase 3 Analisis no supervisado (PCA y clustering) en src/2_unsupervised/unsupervised.py
- [2026-05-27 12:42:10] @data-orchestrator: Fase 3 completada correctamente.
- [2026-05-27 12:42:10] @data-orchestrator -> @stats-modeler: Fase 4 Ajuste de hiperparametros en src/3_optuna/tune.py
- [2026-05-27 12:42:20] @data-orchestrator: Fase 4 completada correctamente.
- [2026-05-27 12:42:20] @data-orchestrator -> @stats-modeler: Fase 5 Entrenamiento final con Train en src/4_train/train.py
- [2026-05-27 12:42:24] @data-orchestrator: Fase 5 completada correctamente.
- [2026-05-27 12:42:24] @data-orchestrator -> @data-visualizer: Fase 6 Evaluacion final con Test en src/5_report/evaluate.py
- [2026-05-27 12:42:29] @data-orchestrator: Fase 6 completada correctamente.
- [2026-05-27 12:42:29] @data-orchestrator: Pipeline Fase 2 finalizado. Solicitar commit a @github-git-agent (Y/n).
