# Instrucciones Globales de Ciencia de Datos

## Stack Tecnológico
- Lenguaje: Python 3.x
- Librerías: pandas, numpy, scikit-learn, optuna, xgboost, lightgbm, matplotlib, seaborn, yellowbrick.

## Políticas de Manejo de Datos
- **Inmutabilidad**: NUNCA modificar archivos en \`data/raw/\`.
- **Estructura**: Siempre crear versiones procesadas en \`data/processed/\`.

## Reglas Globales (Fase 2)
- Seguir la estructura modular de: https://github.com/trigoduoc/codon-classification-pipeline
- Consultar siempre \`AGENTS.md\` para saber qué agente debe realizar cada tarea.
- **Flujo de Trabajo**: Todo script nuevo debe ir dentro de su subcarpeta correspondiente en \`src/\`.

# 🎯 Reglas de Operación - Proyecto EDA-ACV

## 🧬 Contexto de Continuidad
- Estamos en la **Fase 2 (Modelado)**.
- Referencia Técnica: https://github.com/trigoduoc/codon-classification-pipeline

## 🤖 Uso de Agentes
Para cada tarea, utiliza el agente definido en \`AGENTS.md\`:
- Setup/Estructura -> **@env-architect**
- Auditoría Datos -> **@eda-auditor**
- Clustering/PCA -> **@unsupervised-modeler**
- Optuna/XGBoost -> **@hyperparameter-optimizer**
- Docs/Reportes -> **@report-generator**
