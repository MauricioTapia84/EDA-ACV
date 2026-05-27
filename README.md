# Proyecto ACV - EP2 SCY1101

Proyecto de analisis y modelado predictivo para deteccion de accidentes
cerebrovasculares (ACV), alineado a la pauta modular por fases del curso.

## Objetivo

Construir un flujo reproducible de preprocesamiento, modelado, optimizacion y
evaluacion que priorice el recall de la clase positiva (casos ACV), minimizando
falsos negativos.

## Estructura

- `notebooks/`: narrativa de analisis por etapas.
- `src/`: implementacion modular reutilizable.
- `src/0_audit` a `src/5_report`: wrappers por fase exigidos por pauta.
- `data/raw/`: dataset de entrada (inmutable).
- `data/processed/`: datos generados por el pipeline.
- `models/`: parametros y modelos serializados.
- `reports/`: resultados finales y figuras del flujo de ejecucion.
- `results/`: metricas y graficos de soporte generados en notebooks.

Documentacion de estructura: [docs/estructura_proyecto.md](docs/estructura_proyecto.md)
\nVerificacion contra rubrica PDF: [docs/verificacion_rubrica_pdf.md](docs/verificacion_rubrica_pdf.md)

## Ejecucion recomendada

Con `.venv` activo:

```bash
python setup_and_run.py --mode run --skip-install
```

Otros modos utiles:

```bash
python setup_and_run.py --mode status --skip-install
python setup_and_run.py --mode compat --skip-install
```

## Flujo de notebooks

1. `notebooks/01_exploratory_analysis.ipynb`
2. `notebooks/02_supervised_modeling.ipynb`
3. `notebooks/03_model_evaluation.ipynb` (alias mantenido desde `3_model_evaluation.ipynb`)
4. `notebooks/04_hyperparameter_optimization.ipynb`
5. `notebooks/05_final_analysis.ipynb`

## Componentes tecnicos principales

- Preprocesamiento: `src/preprocess.py`, `src/data_preprocessing.py`
- No supervisado: `src/unsupervised.py`
- Tuning (incluye Optuna): `src/tune.py`, `src/hyperparameter_tuning.py`
- Entrenamiento: `src/train.py`, `src/model_training.py`
- Evaluacion e interpretabilidad: `src/evaluate.py`, `src/model_evaluation.py`

## Evidencia generada

- Parametros optimos: `models/best_params.json`
- Estudio Optuna: `models/optuna_study.csv`
- Modelo final: `models/final_model.pkl`
- Reporte final: `reports/evaluation_results.md`
- Importancia de variables: `reports/feature_importance.csv`

## Estado

El flujo completo de Fase 2 esta implementado y ejecuta correctamente con el
launcher oficial. El cierre academico depende de la calidad narrativa final en
los notebooks y de validar el checklist exacto de la pauta PDF.
