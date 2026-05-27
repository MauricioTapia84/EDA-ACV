# Estructura de fases (0-5)

Este proyecto mantiene los scripts funcionales principales en `src/*.py` y agrega wrappers por fase en subcarpetas para alinearse con la pauta.

## Mapeo oficial por fase

- `src/0_audit/audit.py` -> usa `src/preprocess.py` (`run_preprocessing`) para auditoria reproducible.
- `src/1_prep/preprocess.py` -> usa `src/preprocess.py` para preprocesamiento y split.
- `src/2_unsupervised/unsupervised.py` -> usa `src/unsupervised.py`.
- `src/3_optuna/tune.py` -> usa `src/tune.py`.
- `src/4_train/train.py` -> usa `src/train.py`.
- `src/5_report/evaluate.py` -> usa `src/evaluate.py`.

El orquestador `main.py` prioriza estos wrappers para ejecutar el pipeline por fase.
