# Estructura del Proyecto ACV

Este documento resume la estructura oficial del repositorio para la Fase 2,
alineada con la pauta modular 0-5.

## Directorios principales

- `data/raw/`: datos de entrada originales (no se escriben resultados aqui).
- `data/processed/`: artefactos tabulares generados por preprocesamiento y split.
- `src/`: logica de negocio y scripts por fase.
- `notebooks/`: narrativa y evidencia analitica.
- `models/`: modelos serializados y parametros optimizados.
- `reports/`: reportes finales de evaluacion e interpretabilidad.
- `results/`: metricas y graficos de apoyo generados por notebooks.

## Fases en src

- `src/0_audit/audit.py`: auditoria y limpieza inicial.
- `src/1_prep/preprocess.py`: preprocesamiento y division train/test.
- `src/2_unsupervised/unsupervised.py`: analisis no supervisado (PCA + clustering).
- `src/3_optuna/tune.py`: ajuste de hiperparametros con Optuna y comparacion.
- `src/4_train/train.py`: entrenamiento final con parametros seleccionados.
- `src/5_report/evaluate.py`: evaluacion final y reporte de resultados.

## Artefactos esperados por fase

- Fase 1-2: `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `train.csv`, `test.csv`, `data_audited.csv`.
- Fase 3: `reports/figures/pca_clusters.png`, `reports/figures/clustering_summary.png`.
- Fase 4: `models/best_params.json`, `models/optuna_study.csv`, `models/tuning_comparison.csv`.
- Fase 5: `models/final_model.pkl`, `models/best_model.pkl`.
- Fase 6: `reports/evaluation_results.md`, `reports/classification_report.txt`, `reports/feature_importance.csv`.

## Comando de ejecucion oficial

```bash
python setup_and_run.py --mode run --skip-install
```

## Notebooks recomendados en orden

1. `notebooks/01_exploratory_analysis.ipynb`
2. `notebooks/02_supervised_modeling.ipynb`
3. `notebooks/3_model_evaluation.ipynb`
4. `notebooks/04_hyperparameter_optimization.ipynb`
5. `notebooks/05_final_analysis.ipynb`
