# 🤖 Ecosistema de Agentes - Proyecto EDA-ACV (Fase 2)

Este archivo entrena a los agentes para seguir la estructura de 6 pasos del pipeline de clasificación.

## 👥 Especialidades

| Agente | Rol | Responsabilidad Principal | Carpeta de Trabajo |
| :--- | :--- | :--- | :--- |
| **@env-architect** | Arquitecto | Mantenimiento de estructura, dependencias y `setup_and_run.py`. | Raíz / `src/` |
| **@eda-auditor** | Auditor | Auditoría de datos, chequeo de colinealidad (VIF) y calidad. | `src/0_audit/` |
| **@preprocessor** | Procesador | Limpieza, feature engineering y escalado. | `src/1_prep/` |
| **@unsupervised-modeler** | Modelador No Supervisado | Clustering (K-Means, DBSCAN) y reducción de dimensionalidad (PCA). | `src/2_unsupervised/` |
| **@hyperparameter-optimizer** | Optimizador | Búsqueda de hiperparámetros con Optuna y selección de modelos. | `src/3_optuna/` |
| **@trainer** | Entrenador | Entrenamiento final del pipeline y validación cruzada. | `src/4_train/` |
| **@report-generator** | Generador de Reportes | Visualización, métricas de desempeño y documentación de resultados. | `src/5_report/` / `reports/` |

## 🛠 Comandos Comunes
- **Configuración**: `python3 setup_and_run.py`
- **Carpeta de Datos**: `data/raw/` (Inmutable), `data/processed/` (Generados)
- **Documentación**: `docs/`
