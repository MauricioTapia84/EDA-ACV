---
name: Data Analysis Pipeline
description: Flujo secuencial para el proceso de análisis de datos.
---

# Pipeline de Análisis de Datos

Este flujo define la secuencia de trabajo recomendada para los agentes especializados.

## Fases del Pipeline

### Fase 1: Entendimiento
- **Agent**: @data-orchestrator
- **Output**: plan de análisis documentado.

### Fase 2: Extracción  
- **Agent**: @data-extractor
- **Output**: datos crudos localizados + metadata inicial.

### Fase 3: Limpieza
- **Agent**: @data-cleaner  
- **Output**: datos limpios en `/data/processed/` + `quality_report.md`.

### Fase 4: Análisis Exploratorio
- **Agent**: @data-analyzer
- **Output**: estadísticas descriptivas + `hallazgos_iniciales.md`.

### Fase 5: Modelado (Condicional)
- **Agent**: @stats-modeler (si se requiere predicción o inferencia).
- **Output**: entrenamiento de modelos + `metricas.md`.

### Fase 6: Visualización
- **Agent**: @data-visualizer
- **Output**: gráficos (PNG/HTML) + reporte visual.

### Fase 7: Reporte Final
- **Agent**: @data-reporter
- **Output**: `executive_summary.md` con insights de negocio.

### Fase 8: Validación
- **Agent**: @data-orchestrator
- **Output**: consolidación final y verificación de cumplimiento de objetivos.
