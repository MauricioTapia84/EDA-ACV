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
- [ ] Ejecución de Auditoría de Datos (`preprocess.py`).
- [ ] Modelado No Supervisado.

## ⚠️ Brechas Técnicas Detectadas

- [ ] Implementar script dedicado de auditoría (`src/audit.py`) o consolidar auditoría + optimización en `src/preprocess.py` con salida reproducible.
- [ ] Implementar split Train/Test 80/20 explícito y persistir artefactos (`train.csv`, `test.csv` o equivalentes).
- [ ] Implementar `src/unsupervised.py` con PCA + clustering (KNN/DBSCAN/Jerárquico según pauta).
- [ ] Integrar Optuna en `src/tune.py` para búsqueda de hiperparámetros.
- [ ] Implementar entrenamiento final en `src/train.py` con el mejor modelo e hiperparámetros.
- [ ] Implementar evaluación final en Test con `classification_report` y `confusion_matrix` en `src/evaluate.py`.

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
