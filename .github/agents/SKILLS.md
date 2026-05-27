# Custom Skills for EP2 Data Science Audit

## Skill: audit-architecture
- **Description:** Compara la estructura actual de carpetas de `mi_proyecto/` con las exigencias de `AGENTS.md`.
- **Prompt:** Toma como referencia de éxito la organización en `contexto_profesor/` y genera un checklist en Markdown indicando qué carpetas o archivos faltan crear o reubicar.

## Skill: check-data-leakage
- **Description:** Examina los cuadernos y scripts de modelado para asegurar que no hay contaminación de datos en la validación cruzada o en el remuestreo con SMOTE.
- **Prompt:** "Revisa los archivos de modelado en `mi_proyecto/notebooks/` y `mi_proyecto/src/`. Busca activamente cualquier señal de fuga de datos (data leakage), específicamente verificando que el escalado de datos (StandardScaler) y el balanceo por SMOTE se realicen usando `Pipeline` e `imbalanced-learn` dentro del bucle de validación cruzada (StratifiedKFold). Reporta de forma crítica si los datos de prueba están contaminando el entrenamiento. Entrega explicación en formato markdown dentro del notebook, donde expliques cada que haces y por q"

## Skill: modularize-colab
- **Description:** Ayuda a extraer bloques de código del Jupyter Notebook/Colab existente y los convierte en módulos limpios `.py` listos para producción.
- **Prompt:** "Toma las funciones de preprocesamiento/limpieza del notebook que te indique en `mi_proyecto/` (identificando transformers personalizados como UnknownToNaNTransformer, SmartImputerTransformer u OutlierCapper) y conviértelas en código modular listo para producción en `mi_proyecto/src/data_preprocessing.py`. Asegúrate de incluir docstrings, manejo de excepciones con bloques try-except y la herencia correcta de BaseEstimator y TransformerMixin de Scikit-Learn."