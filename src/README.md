# Estructura de `src/`

El proyecto convive con dos capas complementarias:

## 1. Modulos oficiales de la pauta

Son los archivos que la rubrica menciona explicitamente y que usan los notebooks narrativos:

- `data_preprocessing.py`
- `model_training.py`
- `model_evaluation.py`
- `hyperparameter_tuning.py`

## 2. Capa operativa por fases

Permite ejecutar el proyecto de punta a punta desde `setup_and_run.py` y `main.py --run`:

- `0_audit/audit.py`
- `1_prep/preprocess.py`
- `2_unsupervised/unsupervised.py`
- `3_optuna/tune.py`
- `4_train/train.py`
- `5_report/evaluate.py`

Estos wrappers delegan en los scripts:

- `preprocess.py`
- `unsupervised.py`
- `tune.py`
- `train.py`
- `evaluate.py`

## Como leer esta estructura

- Si estas revisando la historia academica del proyecto, mira primero los notebooks y los modulos oficiales.
- Si estas verificando la ejecucion end-to-end por consola, usa la capa por fases.
- Ambas capas deben producir una historia consistente sobre el mismo problema: deteccion de ACV con fuerte desbalance de clases.
