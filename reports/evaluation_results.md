# Evaluation Results

## Metrics
- precision: 0.1340
- recall: 0.8200
- f1: 0.2303
- roc_auc: 0.8415

## Interpretability

Top 10 feature importances:

| feature | importance |
|---|---:|
| 0 | 1.361755 |
| 3 | 0.221899 |
| 19 | 0.198153 |
| 15 | 0.186428 |
| 9 | 0.147054 |
| 6 | 0.142864 |
| 18 | 0.130719 |
| 5 | 0.117655 |
| 8 | 0.114486 |
| 12 | 0.087915 |

SHAP no disponible en el entorno: se omite resumen SHAP.

## Confusion Matrix

|       | pred_0 | pred_1 |
|-------|--------|--------|
| actual_0 | 707 | 265 |
| actual_1 | 9 | 41 |

## Classification Report

```text
              precision    recall  f1-score   support

           0     0.9874    0.7274    0.8377       972
           1     0.1340    0.8200    0.2303        50

    accuracy                         0.7319      1022
   macro avg     0.5607    0.7737    0.5340      1022
weighted avg     0.9457    0.7319    0.8080      1022

```
