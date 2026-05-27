# Custom Skills for EP2 Data Science Audit

## Skill: audit-architecture
- **Description:** Compara la estructura actual de carpetas de `mi_proyecto/` con las exigencias de `AGENTS.md`.
- **Prompt:** Toma como referencia de éxito la organización en `contexto_profesor/` y genera un checklist en Markdown indicando qué carpetas o archivos faltan crear o reubicar.

## Skill: check-data-leakage
- **Description:** Examina los cuadernos y scripts de modelado para asegurar que no hay contaminación de datos en la validación cruzada o en el remuestreo con SMOTE.
- **Prompt:** "Revisa los archivos de modelado en `mi_proyecto/notebooks/` y `mi_proyecto/src/`. Busca activamente cualquier señal de fuga de datos (data leakage), verificando que el escalado (StandardScaler), la imputación y cualquier eventual balanceo se ejecuten dentro de `Pipeline` y del bucle de validación cruzada (`StratifiedKFold`). Reporta de forma crítica si los datos de prueba contaminan el entrenamiento y deja la explicación en Markdown, con foco en los módulos oficiales `data_preprocessing.py`, `model_training.py`, `model_evaluation.py` y `hyperparameter_tuning.py`."

## Skill: modularize-colab
- **Description:** Ayuda a extraer bloques de código del Jupyter Notebook/Colab existente y los convierte en módulos limpios `.py` listos para producción.
- **Prompt:** "Toma las funciones de preprocesamiento, modelado o evaluación del notebook que se indique en `mi_proyecto/` y conviértelas en código modular listo para producción dentro de los módulos oficiales de `mi_proyecto/src/`: `data_preprocessing.py`, `model_training.py`, `model_evaluation.py` y `hyperparameter_tuning.py`. Mantén docstrings claras, compatibilidad con Scikit-Learn y trazabilidad para que los notebooks importen desde `src/` en vez de duplicar lógica."
