\=== INICIO DEL DOCUMENTO \===

# **GUÍA COMPLETA PARA EVALUACIÓN PARCIAL N°1**

## **Programación para la Ciencia de Datos (SCY1101)**

### **Proyecto de EDA con Pipeline de Preprocessing**

Equipo: \[Nombres de los integrantes\]  
Fecha de entrega: \[Fecha\]  
Ponderación: 30% (10% Encargo grupal \+ 20% Presentación individual)

---

## **Tabla de Contenido**

1. Descripción General del Proyecto  
2. Estructura de Carpetas Obligatoria  
3. Distribución de Roles y Responsabilidades  
4. Tareas por Rol (Detallado)  
   * Data Analyst (EDA)  
   * Data Engineer (Transformers)  
   * ML Engineer (Pipeline y Auditoría)  
5. Cronograma de Trabajo (2 semanas)  
6. Reuniones de Sincronización Obligatorias  
7. Entregables por Rol  
8. Checklist Final de Evaluación  
9. Rúbrica de Autoevaluación  
10. Consejos para la Presentación

---

## **1\. Descripción General del Proyecto**

Este proyecto simula un entorno profesional de ciencia de datos. El equipo debe:

* Seleccionar un dataset con complejidad suficiente (nulos, outliers, categóricas, fechas).  
* Realizar un Análisis Exploratorio de Datos (EDA) completo.  
* Implementar un pipeline de preprocessing usando Scikit-Learn con transformers personalizados.  
* Auditar la integridad de los datos en cada etapa (checksums, validación de esquema).  
* Generar un informe técnico de 8-12 páginas.  
* Presentar los resultados en 15 minutos \+ 5 minutos de preguntas.

Requisitos técnicos obligatorios:

* Python, Pandas, NumPy, Scikit-Learn, Matplotlib/Seaborn.  
* Entorno virtual (venv o conda) con requirements.txt.  
* Código modular en src/ con docstrings y manejo de excepciones.  
* Script orquestador main.py que ejecuta todo el flujo.  
* Pipeline construido con sklearn.pipeline.Pipeline y ColumnTransformer.

---

## **2\. Estructura de Carpetas Obligatoria**

`text`

`tu_proyecto_eda/`  
`│`  
`├── data/`  
`│   ├── raw/                  # Datos originales (nunca modificar)`  
`│   │   └── tu_dataset.csv`  
`│   └── processed/            # Datos limpios (output del pipeline)`  
`│       └── tu_dataset_clean.csv`  
`│`  
`├── notebooks/`  
`│   └── 01_eda_completo.ipynb # Análisis exploratorio (Data Analyst)`  
`│   └── 02_test_transformers.ipynb # Pruebas de transformers (Engineer)`  
`│`  
`├── src/                      # Código reutilizable`  
`│   ├── __init__.py`  
`│   ├── transformers.py       # Clases personalizadas (Engineer)`  
`│   ├── pipeline_builder.py   # Función build_preprocessing_pipeline (ML Engineer)`  
`│   ├── audit.py              # Checksums, validación de esquema (ML Engineer)`  
`│   └── utils.py              # Funciones auxiliares (opcional)`  
`│`  
`├── outputs/`  
`│   ├── graficos/             # Visualizaciones para el informe`  
`│   └── audit_log.json        # Log de auditoría (generado por main.py)`  
`│`  
`├── docs/`  
`│   └── informe_tecnico.pdf   # 8-12 páginas (todo el equipo)`  
`│`  
`├── main.py                   # Orquestador principal (ML Engineer)`  
`├── requirements.txt          # Dependencias (ML Engineer)`

`└── README.md                 # Instrucciones de setup y ejecución (ML Engineer)`

---

## **3\. Distribución de Roles y Responsabilidades**

| Rol | Responsabilidad principal | % trabajo | Dependencias |
| :---- | :---- | :---- | :---- |
| Data Analyst (EDA) | Análisis exploratorio, visualizaciones, estadísticas, hallazgos | 35% | Ninguna (empieza primero) |
| Data Engineer (Funciones) | Implementar transformers personalizados en src/transformers.py | 35% | Recibe hallazgos del Analyst |
| ML Engineer (Pipeline) | Construir pipeline, main.py, auditoría, integración | 30% | Depende de transformers del Engineer |

Nota: Todos colaboran en informe técnico, presentación y README.

---

## **4\. Tareas por Rol (Detallado)**

### **4.1 Data Analyst (EDA) \- Días 1 a 6**

Objetivo: Entender el dataset a fondo y generar recomendaciones para el pipeline.

#### **Día 1-2: Carga y exploración inicial**

`python`

*`# En Jupyter Notebook`*  
`import pandas as pd`  
`import numpy as np`  
`import matplotlib.pyplot as plt`  
`import seaborn as sns`

`df = pd.read_csv('data/raw/tu_dataset.csv')`

*`# 1. Metadata básica`*  
`print("Shape:", df.shape)`  
`print("\nTipos de dato:\n", df.dtypes)`  
`print("\nNulos por columna:\n", df.isnull().sum())`  
`print("\nEstadísticas descriptivas:\n", df.describe(include='all'))`

*`# 2. Duplicados`*

`print(f"Filas duplicadas: {df.duplicated().sum()}")`

#### **Día 3-4: Análisis profundo**

`python`

*`# Porcentaje de nulos por columna`*  
`null_pct = df.isnull().sum() / len(df) * 100`  
`cols_high_null = null_pct[null_pct > 80].index.tolist()`  
`print(f"Columnas con >80% nulos (recomendar dropear): {cols_high_null}")`

*`# Detección de outliers con IQR`*  
`for col in df.select_dtypes(include='number').columns:`  
    `Q1 = df[col].quantile(0.25)`  
    `Q3 = df[col].quantile(0.75)`  
    `IQR = Q3 - Q1`  
    `outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]`  
    `print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")`

*`# Matriz de correlación (si hay variable objetivo)`*  
`if 'target' in df.columns:`  
    `corr_matrix = df.corr(numeric_only=True)`  
    `sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')`

    `plt.savefig('../outputs/graficos/correlaciones.png')`

#### **Día 5-6: Visualizaciones clave**

Generar y guardar al menos 5-8 gráficos:

`python`

*`# Histogramas de variables numéricas`*  
`df.hist(figsize=(12, 10), bins=30)`  
`plt.savefig('../outputs/graficos/histogramas.png', dpi=300)`

*`# Boxplots para detectar outliers`*  
`df.boxplot(figsize=(12, 6), rot=90)`  
`plt.savefig('../outputs/graficos/boxplots.png', dpi=300)`

*`# Mapa de calor de nulos`*  
`plt.figure(figsize=(10, 6))`  
`sns.heatmap(df.isnull(), cbar=False, yticklabels=False)`  
`plt.savefig('../outputs/graficos/null_heatmap.png', dpi=300)`

*`# Countplots para categóricas principales`*  
`categoricals = df.select_dtypes(include='object').columns[:4]`  
`for col in categoricals:`  
    `plt.figure()`  
    `df[col].value_counts().head(10).plot(kind='bar')`  
    `plt.title(f'Distribución de {col}')`

    `plt.savefig(f'../outputs/graficos/countplot_{col}.png', dpi=300)`

#### **Entregable del Analyst al equipo (documento markdown o texto):**

`text`

`## HALLAZGOS EDA PARA EL PIPELINE`

`### 1. Columnas a dropear completamente:`  
`- col1: 85% nulos, sin información útil`  
`- col2: ID de fila, sin valor predictivo`

`### 2. Columnas con alto % de nulos (40-80%):`  
`- col3: 65% nulos → imputar con 'missing' (nueva categoría)`  
`- col4: 55% nulos → imputar con mediana`

`### 3. Outliers severos detectados:`  
`- col5: 5% outliers extremos → aplicar capping en percentil 99`  
`- col6: 8% outliers → capping en percentil 95`

`### 4. Variables con varianza cero:`  
`- col7: todos los valores iguales → dropear`

`### 5. Valores 'unknown' en categóricas:`  
`- job: 15% 'unknown' → convertir a NaN primero`  
`- education: 10% 'unknown' → convertir a NaN`

`### 6. Recomendaciones de imputación:`  
`- Numéricas: usar mediana (distribuciones asimétricas)`  
`- Categóricas: usar moda o crear categoría 'missing'`

`### 7. Variables candidatas para encoding:`

`- job, marital, education, default, housing, loan`

---

### **4.2 Data Engineer (Funciones) \- Días 1 a 10**

Objetivo: Implementar todos los transformers personalizados que usará el pipeline.

#### **Implementación completa de `src/transformers.py`:**

`python`

`import pandas as pd`  
`import numpy as np`  
`from sklearn.base import BaseEstimator, TransformerMixin`

`class DropColumnsTransformer(BaseEstimator, TransformerMixin):`  
    `"""Elimina columnas especificadas."""`  
    `def __init__(self, columns_to_drop=None):`  
        `self.columns_to_drop = columns_to_drop if columns_to_drop else []`  
      
    `def fit(self, X, y=None):`  
        `return self`  
      
    `def transform(self, X):`  
        `X_copy = X.copy()`  
        `cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]`  
        `if cols_to_drop:`  
            `X_copy.drop(columns=cols_to_drop, inplace=True)`  
        `return X_copy`

`class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):`  
    `"""Convierte strings como 'unknown', 'Unknown', 'na' a NaN."""`  
    `def __init__(self, unknown_strings=None):`  
        `if unknown_strings is None:`  
            `self.unknown_strings = ['unknown', 'Unknown', 'UNKNOWN', 'na', 'NA', 'N/A', 'n/a', '']`  
        `else:`  
            `self.unknown_strings = unknown_strings`  
      
    `def fit(self, X, y=None):`  
        `return self`  
      
    `def transform(self, X):`  
        `X_copy = X.copy()`  
        `for col in X_copy.select_dtypes(include='object').columns:`  
            `X_copy[col] = X_copy[col].replace(self.unknown_strings, np.nan)`  
        `return X_copy`

`class DropHighMissingTransformer(BaseEstimator, TransformerMixin):`  
    ``"""Elimina columnas con más de `threshold` (0-1) valores nulos."""``  
    `def __init__(self, threshold=0.8):`  
        `self.threshold = threshold`  
        `self.cols_to_drop_ = []`  
      
    `def fit(self, X, y=None):`  
        `missing_pct = X.isnull().sum() / len(X)`  
        `self.cols_to_drop_ = missing_pct[missing_pct > self.threshold].index.tolist()`  
        `return self`  
      
    `def transform(self, X):`  
        `if self.cols_to_drop_:`  
            `return X.drop(columns=self.cols_to_drop_)`  
        `return X`

`class SmartImputerTransformer(BaseEstimator, TransformerMixin):`  
    `"""`  
    `Imputa basado en umbral: si bajo % nulos usa mediana/moda,`  
    `si alto % nulos (pero < threshold) crea categoría 'missing'.`  
    `"""`  
    `def __init__(self, low_threshold=0.10, high_threshold=0.50):`  
        `self.low_threshold = low_threshold`  
        `self.high_threshold = high_threshold`  
        `self.impute_dict_ = {}`  
      
    `def fit(self, X, y=None):`  
        `missing_pct = X.isnull().sum() / len(X)`  
          
        `for col in X.columns:`  
            `pct = missing_pct[col]`  
            `if pct == 0:`  
                `continue`  
            `elif pct <= self.low_threshold:`  
                `if pd.api.types.is_numeric_dtype(X[col]):`  
                    `self.impute_dict_[col] = X[col].median()`  
                `else:`  
                    `self.impute_dict_[col] = X[col].mode()[0] if not X[col].mode().empty else 'missing'`  
            `elif pct <= self.high_threshold:`  
                `# Crear columna flag (esto requiere acceso a X en transform)`  
                `# Para simplificar, imputamos con valor especial`  
                `if pd.api.types.is_numeric_dtype(X[col]):`  
                    `self.impute_dict_[col] = X[col].median()`  
                `else:`  
                    `self.impute_dict_[col] = 'missing'`  
        `return self`  
      
    `def transform(self, X):`  
        `X_copy = X.copy()`  
        `for col, value in self.impute_dict_.items():`  
            `X_copy[col].fillna(value, inplace=True)`  
        `return X_copy`

`class OutlierCapper(BaseEstimator, TransformerMixin):`  
    `"""Aplica capping a outliers usando percentiles."""`  
    `def __init__(self, apply_capping=True, lower_percentile=0.01, upper_percentile=0.99):`  
        `self.apply_capping = apply_capping`  
        `self.lower_percentile = lower_percentile`  
        `self.upper_percentile = upper_percentile`  
        `self.caps_ = {}`  
      
    `def fit(self, X, y=None):`  
        `if not self.apply_capping:`  
            `return self`  
          
        `for col in X.select_dtypes(include='number').columns:`  
            `lower = X[col].quantile(self.lower_percentile)`  
            `upper = X[col].quantile(self.upper_percentile)`  
            `self.caps_[col] = (lower, upper)`  
        `return self`  
      
    `def transform(self, X):`  
        `if not self.apply_capping:`  
            `return X`  
          
        `X_copy = X.copy()`  
        `for col, (lower, upper) in self.caps_.items():`  
            `X_copy[col] = X_copy[col].clip(lower, upper)`  
        `return X_copy`

`class DropZeroVarianceTransformer(BaseEstimator, TransformerMixin):`  
    `"""Elimina columnas numéricas con varianza cero."""`  
    `def __init__(self):`  
        `self.cols_to_drop_ = []`  
      
    `def fit(self, X, y=None):`  
        `numeric_cols = X.select_dtypes(include='number').columns`  
        `for col in numeric_cols:`  
            `if X[col].var() == 0:`  
                `self.cols_to_drop_.append(col)`  
        `return self`  
      
    `def transform(self, X):`  
        `if self.cols_to_drop_:`  
            `return X.drop(columns=self.cols_to_drop_)`

        `return X`

#### **Prueba de transformers (notebook separado):**

Crear `notebooks/02_test_transformers.ipynb` con:

`python`

`import sys`  
`sys.path.append('..')`  
`import pandas as pd`  
`from src.transformers import *`

*`# Datos de prueba`*  
`test_df = pd.DataFrame({`  
    `'age': [25, 30, 35, 1000, 28],`  
    `'job': ['engineer', 'unknown', 'doctor', 'engineer', 'unknown'],`  
    `'salary': [50000, 60000, 55000, np.nan, 52000]`  
`})`

*`# Probar UnknownToNaNTransformer`*  
`cleaner = UnknownToNaNTransformer()`  
`print(cleaner.fit_transform(test_df))`

*`# Probar OutlierCapper`*  
`capper = OutlierCapper(apply_capping=True, lower_percentile=0.05, upper_percentile=0.95)`

`print(capper.fit_transform(test_df))`

---

### **4.3 ML Engineer (Pipeline) \- Días 5 a 12**

Objetivo: Construir el pipeline maestro y orquestar todo el flujo.

#### **Implementación de `src/pipeline_builder.py`:**

`python`

`import pandas as pd`  
`from sklearn.pipeline import Pipeline`  
`from sklearn.compose import ColumnTransformer, make_column_selector`  
`from sklearn.preprocessing import StandardScaler, OneHotEncoder`  
`from src.transformers import (`  
    `DropColumnsTransformer,`  
    `UnknownToNaNTransformer,`  
    `DropHighMissingTransformer,`  
    `SmartImputerTransformer,`  
    `OutlierCapper,`  
    `DropZeroVarianceTransformer`  
`)`

`def build_preprocessing_pipeline(df_sample=None, columns_to_drop=None):`  
    `"""`  
    `Construye pipeline de preprocessing dinámico.`  
      
    `Parameters`  
    `----------`  
    `df_sample : pd.DataFrame, optional`  
        `Muestra para detectar columnas (no usado actualmente)`  
    `columns_to_drop : list, optional`  
        `Columnas a eliminar manualmente (leaks, IDs, etc.)`  
      
    `Returns`  
    `-------`  
    `sklearn.pipeline.Pipeline`  
    `"""`  
    `if columns_to_drop is None:`  
        `columns_to_drop = []`  
      
    `# Pipeline numérico`  
    `num_pipe = Pipeline([`  
        `('capper', OutlierCapper(apply_capping=True)),`  
        `('zero_variance', DropZeroVarianceTransformer()),`  
        `('scaler', StandardScaler())`  
    `])`  
      
    `# Pipeline categórico`  
    `cat_pipe = Pipeline([`  
        `('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))`  
    `])`  
      
    `# ColumnTransformer dinámico`  
    `preprocessor = ColumnTransformer(`  
        `transformers=[`  
            `('num', num_pipe, make_column_selector(dtype_include='number')),`  
            `('cat', cat_pipe, make_column_selector(dtype_exclude='number'))`  
        `],`  
        `remainder='drop'`  
    `)`  
      
    `# Pipeline completo`  
    `full_pipeline = Pipeline([`  
        `('drop_leaks', DropColumnsTransformer(columns_to_drop=columns_to_drop)),`  
        `('clean_unknowns', UnknownToNaNTransformer()),`  
        `('drop_high_nan', DropHighMissingTransformer(threshold=0.8)),`  
        `('smart_imputer', SmartImputerTransformer(low_threshold=0.10, high_threshold=0.50)),`  
        `('preprocessing', preprocessor)`  
    `])`  
    

    `return full_pipeline`

#### **Implementación de `src/audit.py`:**

`python`

`import hashlib`  
`import json`  
`import pandas as pd`  
`from datetime import datetime`

`def compute_checksum(df):`  
    `"""Calcula checksum MD5 de un DataFrame."""`  
    `return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()`

`def audit_dataframe(df, stage_name, log_file='outputs/audit_log.json'):`  
    `"""Registra métricas de auditoría en archivo JSON."""`  
    `audit_entry = {`  
        `'stage': stage_name,`  
        `'timestamp': datetime.now().isoformat(),`  
        `'shape': list(df.shape),`  
        `'nulls_total': int(df.isnull().sum().sum()),`  
        `'nulls_by_column': df.isnull().sum().to_dict(),`  
        `'checksum': compute_checksum(df),`  
        `'columns': list(df.columns)`  
    `}`  
      
    `with open(log_file, 'a') as f:`  
        `f.write(json.dumps(audit_entry) + '\n')`  
      
    `return audit_entry`

`def validate_schema(df, expected_columns):`  
    `"""Valida que todas las columnas esperadas existan."""`  
    `missing = set(expected_columns) - set(df.columns)`  
    `extra = set(df.columns) - set(expected_columns)`  
      
    `return {`  
        `'valid': len(missing) == 0,`  
        `'missing_columns': list(missing),`  
        `'extra_columns': list(extra)`

    `}`

#### **Orquestador principal `main.py`:**

`python`

`import pandas as pd`  
`import sys`  
`import os`  
`sys.path.append('.')`

`from src.pipeline_builder import build_preprocessing_pipeline`  
`from src.audit import audit_dataframe, validate_schema`

`def main():`  
    `print("=" * 50)`  
    `print("PROYECTO EDA - PIPELINE DE PREPROCESSING")`  
    `print("=" * 50)`  
      
    `# 1. Crear carpetas si no existen`  
    `os.makedirs('data/processed', exist_ok=True)`  
    `os.makedirs('outputs', exist_ok=True)`  
      
    `# 2. Cargar datos raw`  
    `print("\n[1/5] Cargando datos raw...")`  
    `df_raw = pd.read_csv('data/raw/tu_dataset.csv')`  
    `audit_dataframe(df_raw, 'raw_input')`  
    `print(f"      Shape: {df_raw.shape}")`  
    `print(f"      Nulos totales: {df_raw.isnull().sum().sum()}")`  
      
    `# 3. Definir columnas a dropear (basado en hallazgos del EDA)`  
    `columns_to_drop = ['id', 'duration']  # Ajustar según dataset`  
      
    `# 4. Construir y ejecutar pipeline`  
    `print("\n[2/5] Construyendo pipeline...")`  
    `pipeline = build_preprocessing_pipeline(df_raw, columns_to_drop=columns_to_drop)`  
      
    `print("\n[3/5] Ejecutando pipeline...")`  
    `try:`  
        `df_processed_array = pipeline.fit_transform(df_raw)`  
        `print("      Pipeline ejecutado sin errores")`  
    `except Exception as e:`  
        `print(f"      ERROR: {e}")`  
        `return`  
      
    `# Convertir array a DataFrame (los nombres de columnas se pierden con OneHot)`  
    `# Para un proyecto EDA, podemos mantener el pipeline hasta el preprocessor`  
    `print("\n[4/5] Auditoría post-procesamiento...")`  
    `# Nota: Como el pipeline devuelve array, creamos un DataFrame dummy`  
    `df_final = pd.DataFrame(df_processed_array)`  
    `audit_dataframe(df_final, 'processed_output')`  
      
    `# 5. Guardar resultado`  
    `print("\n[5/5] Guardando datos procesados...")`  
    `df_final.to_csv('data/processed/tu_dataset_clean.csv', index=False)`  
    `print("      Archivo guardado en data/processed/tu_dataset_clean.csv")`  
      
    `# Resumen final`  
    `print("\n" + "=" * 50)`  
    `print("✅ PIPELINE COMPLETADO CON ÉXITO")`  
    `print("=" * 50)`  
    `print("\nArchivos generados:")`  
    `print("   - data/processed/tu_dataset_clean.csv")`  
    `print("   - outputs/audit_log.json")`  
    `print("\nRevisar informe técnico en docs/informe_tecnico.pdf")`

`if __name__ == "__main__":`

    `main()`

#### **requirements.txt:**

`text`

`pandas>=1.5.0`  
`numpy>=1.23.0`  
`matplotlib>=3.6.0`  
`seaborn>=0.12.0`  
`scikit-learn>=1.2.0`  
`jupyter>=1.0.0`

`nbconvert>=7.0.0`

---

## **5\. Cronograma de Trabajo (2 semanas)**

| Día | Data Analyst (EDA) | Data Engineer (Funciones) | ML Engineer (Pipeline) |
| :---- | :---- | :---- | :---- |
| 1-2 | Elegir dataset, cargar, describir | Configurar entorno, estructura carpetas | Configurar entorno, estructura carpetas |
| 3-4 | EDA completo: nulos, outliers, correlaciones | Esqueleto de transformers.py (clases vacías) | Revisar EDA, definir columnas a dropear |
| 5-6 | Visualizaciones clave, insights | Implementar cada transformer (una por una) | Implementar audit.py, utils.py base |
| 7-8 | Preparar borrador del informe | Probar transformers con datos simulados | Construir build\_preprocessing\_pipeline() |
| 9-10 | Revisar pipeline, sugerir ajustes | Debuggear transformers | Integrar todo en main.py, correr pipeline |
| 11-12 | Finalizar informe, gráficos finales | Documentar funciones (docstrings) | Escribir README, requirements.txt |
| 13-14 | Ensayar presentación | Ensayar presentación | Ensayar presentación |

---

## **6\. Reuniones de Sincronización Obligatorias**

| Día | Agenda | Responsable de liderar | Duración |
| :---- | :---- | :---- | :---- |
| Día 3 | Revisar hallazgos EDA, definir columnas a dropear | Data Analyst | 30 min |
| Día 7 | Revisar transformers implementados, ajustar pipeline | Data Engineer | 30 min |
| Día 10 | Correr pipeline completo por primera vez | ML Engineer | 45 min |
| Día 12 | Revisar resultados, ajustar informe | Todos | 30 min |
| Día 14 | Ensayo de presentación | Todos (rotar) | 20 min |

---

## **7\. Entregables por Rol**

### **Data Analyst entrega:**

* `notebooks/01_eda_completo.ipynb` (limpio, comentado)  
* Carpeta `outputs/graficos/` con 5-8 visualizaciones clave  
* Documento de hallazgos (lista de columnas a dropear, imputaciones, outliers)  
* Contribución al informe técnico (secciones 1 y 2\)

### **Data Engineer entrega:**

* `src/transformers.py` completo y probado  
* `notebooks/02_test_transformers.ipynb` con pruebas unitarias  
* Docstrings en todas las clases y métodos  
* Contribución al informe técnico (sección 3 \- metodología)

### **ML Engineer entrega:**

* `src/pipeline_builder.py` funcionando  
* `src/audit.py` con checksums y validación  
* `main.py` que corre de principio a fin sin errores  
* `requirements.txt` y `README.md`  
* Logs de auditoría en `outputs/audit_log.json`  
* Data procesada en `data/processed/`  
* Contribución al informe técnico (secciones 4 y 5\)

### **Todos colaboran en:**

* `docs/informe_tecnico.pdf` (8-12 páginas)  
* Presentación (15 minutos, todos participan)  
* Revisión final del README

---

## **8\. Checklist Final de Evaluación (Basado en Rúbrica)**

### **Dimensión Encargo (10% grupal)**

#### **Indicador 1: Manipulación en Pandas (3%)**

* Uso de filtros avanzados (loc, iloc, query)  
* Agrupaciones múltiples (groupby con agg)  
* Joins complejos (merge con múltiples condiciones)  
* Código claro y eficiente (sin loops innecesarios)

#### **Indicador 2: Reportes y visualizaciones (7%)**

* Reporte estructurado (8-12 páginas)  
* Visualizaciones claras y pertinentes  
* Análisis de resultados (insights)  
* Gráficos guardados en outputs/graficos/

#### **Indicador 3: Transformaciones avanzadas (5%)**

* Broadcasting (operaciones vectorizadas)  
* Pivot / Melt / Reshape  
* Chunking para datos grandes (si aplica)  
* Optimización de memoria (dtypes, downcasting)

#### **Indicador 4: Flujo de limpieza con justificación (5%)**

* Flujo completo y documentado  
* Múltiples técnicas de imputación  
* Justificación clara en el informe  
* Resultados validados (antes/después)

#### **Indicador 5: Entorno virtual (5%)**

* Entorno virtual creado (.venv o conda)  
* requirements.txt funcional  
* Instrucciones claras en README  
* Reproducible en otra máquina

#### **Indicador 6: Verificación de integridad (5%)**

* Checksums antes/después  
* Validación de esquema  
* Auditoría documentada (audit\_log.json)  
* Justificación de la importancia

### **Dimensión Presentación (20% individual)**

* Explicación clara de operaciones Pandas (10%): argumentos técnicos, ejemplos  
* Comprensión de transformaciones (15%): impacto en rendimiento, memoria  
* Justificación de limpieza (15%): cuantificar impacto, proponer mejoras  
* Justificación del entorno (15%): por qué venv/conda, cómo replicar  
* Validación de datos (15%): importancia, procedimientos, alternativas

---

## **9\. Rúbrica de Autoevaluación (para el equipo)**

| Criterio | Peso | Autoevaluación (1-4) | Evidencia |
| :---- | :---- | :---- | :---- |
| Estructura de carpetas exacta | 5% |  |  |
| Código con docstrings y comentarios | 5% |  |  |
| Funciones reutilizables en src/ | 5% |  |  |
| Manejo de excepciones | 3% |  |  |
| Pipeline con sklearn | 10% |  |  |
| Checksums y auditoría | 5% |  |  |
| Informe de 8-12 páginas | 10% |  |  |
| Visualizaciones de calidad | 7% |  |  |
| README con instrucciones | 5% |  |  |
| Todos participan en presentación | 20% |  |  |
| Justificaciones técnicas claras | 25% |  |  |

Escala: 1 \= Incipiente, 2 \= Aceptable, 3 \= Bueno, 4 \= Muy bueno

---

## **10\. Consejos para la Presentación (15 min \+ 5 preguntas)**

### **Estructura sugerida (diapositivas):**

1. Portada y contexto (1 min)  
   * Nombre del proyecto, integrantes, dataset elegido  
2. Problema y enfoque (2 min)  
   * ¿Qué pregunta responde el EDA?  
   * ¿Por qué este dataset es relevante?  
3. Demo de código clave (4 min)  
   * Mostrar 2-3 fragmentos: un join complejo, una imputación justificada, el pipeline  
   * No mostrar todo el notebook  
4. Justificación de decisiones (3 min)  
   * ¿Por qué mediana y no media?  
   * ¿Por qué ese umbral para outliers?  
   * ¿Por qué venv en lugar de conda?  
5. Resultados e insights (3 min)  
   * Mostrar 2 gráficos impactantes  
   * ¿Qué aprendimos del dataset?  
6. Lecciones aprendidas (2 min)  
   * Dificultades encontradas  
   * Qué harían diferente  
   * Mejoras futuras (ej: Dask para escalar)

### **Posibles preguntas del profesor:**

* "¿Por qué eligieron ese threshold para DropHighMissingTransformer?"  
* "¿Cómo validan que el pipeline es correcto?"  
* "¿Qué pasa si alguien corre main.py sin activar el entorno virtual?"  
* "¿Cómo manejarían un dataset de 10GB?"  
* "¿Por qué usaron OneHotEncoder y no LabelEncoder?"

### **Tips para el día de la presentación:**

* Ensayen 3 veces con cronómetro.  
* Todos deben hablar al menos 2 minutos.  
* No lean diapositivas – usen palabras clave.  
* Preparen una diapositiva de respaldo con código extra por si preguntan.  
* Suban todo a GitHub y tengan el enlace a mano.

---

## **Apéndice: Enlaces Útiles**

* Repositorio de ejemplo del profesor: `https://github.com/trigoduoc/bank_marketing_project/tree/example_evaluation_1`  
* Documentación de Scikit-Learn Pipeline: [https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)  
* Documentación de ColumnTransformer: [https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)

---

¡Éxito en el proyecto\!

\=== FIN DEL DOCUMENTO \===

---

## **📝 Instrucciones para copiar a Google Docs:**

1. Selecciona todo el texto desde `=== INICIO DEL DOCUMENTO ===` hasta `=== FIN DEL DOCUMENTO ===`  
2. Copia (Ctrl+C o Cmd+C)  
3. Abre un documento nuevo en Google Docs  
4. Pega (Ctrl+V o Cmd+V)  
5. Aplica estilos de título:  
   * Selecciona los títulos principales (los que empiezan con `#`) y asígnales "Heading 1"  
   * Los subtítulos con `##` → "Heading 2"  
   * Los subtítulos con `###` → "Heading 3"  
6. Los bloques de código aparecerán como texto normal. Puedes usar el complemento "Code Blocks" en Google Docs o dejarlos como están.

