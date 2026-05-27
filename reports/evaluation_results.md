# Evaluation Results

## Metrics
- precision: 0.1102
- recall: 0.8200
- f1: 0.1943
- roc_auc: 0.8369

## Interpretability

Top 10 feature importances:

| feature | importance |
|---|---:|
| 0 | 0.571664 |
| 3 | 0.160281 |
| 8 | 0.104177 |
| 15 | 0.068778 |
| 18 | 0.063949 |
| 19 | 0.057184 |
| 5 | 0.055728 |
| 14 | 0.053992 |
| 12 | 0.050606 |
| 6 | 0.035892 |

SHAP no disponible en el entorno: se omite resumen SHAP.

## Confusion Matrix

|       | pred_0 | pred_1 |
|-------|--------|--------|
| actual_0 | 641 | 331 |
| actual_1 | 9 | 41 |

## Classification Report

```text
              precision    recall  f1-score   support

           0     0.9862    0.6595    0.7904       972
           1     0.1102    0.8200    0.1943        50

    accuracy                         0.6673      1022
   macro avg     0.5482    0.7397    0.4923      1022
weighted avg     0.9433    0.6673    0.7612      1022

```