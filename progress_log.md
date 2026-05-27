# 🧠 Registro de Progreso - Proyecto EDA-ACV (Fase 2)

## ✅ Hitos Completados
- Se reestructuró el repositorio para cumplir la rúbrica: nombres de notebooks, carpeta results/, y scripts modulares en src/.
- Se implementaron los transformers reutilizables en src/preprocess.py: UnknownToNaN, SmartImputer, OutlierCapper.
- Se implementó el registro de modelos base en src/train.py: Regresión Logística, Random Forest, SVC, XGBoost.
- Se implementó la capa de evaluación en src/evaluate.py: StratifiedKFold, Precision, Recall, F1, ROC-AUC.
- Se implementó la capa de tuning en src/tune.py usando GridSearchCV y RandomizedSearchCV.
- Se dejaron operativos los notebooks 01 a 04 con PCA, K-Means, integración de Pipelines y persistencia de parámetros.
- Se limpió el repositorio: eliminado venv/, __pycache__/ y scaffolding antiguo.

## 🚀 Siguiente paso recomendado
- Completar `notebooks/05_final_analysis.ipynb` con el análisis técnico final.
- Serializar el mejor modelo en `models/trained_models/`.
- Verificar justificaciones de Recall vs F1 en el tuning.
- Pasada final manual en Jupyter para salidas visibles.
