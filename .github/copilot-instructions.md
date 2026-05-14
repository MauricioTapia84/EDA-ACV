# Instrucciones Globales de Ciencia de Datos

## Stack Tecnológico
- Lenguaje: Python 3.x
- Librerías principales:
  - Manipulación de datos: `pandas`, `numpy`
  - Visualización: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`, `statsmodels`

## Convenciones de Codificación
- Seguir **PEP 8**.
- Incluir **docstrings** en todas las funciones y clases.
- Usar **type hints** para mejorar la legibilidad y mantenimiento.

## Políticas de Manejo de Datos
- **Inmutabilidad de datos originales**: NUNCA modificar archivos en `/data/raw/` o archivos originales proporcionados.
- **Flujo de trabajo**: Siempre crear versiones procesadas en `/data/processed/` o directorios equivalentes.
- **Documentación**: Cada paso de transformación debe estar documentado en el código o en un archivo README/metadata adjunto.
