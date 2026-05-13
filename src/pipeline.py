"""
Pipeline construction module for the ETL process.
Combines custom transformers with scikit-learn preprocessing objects.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

# Imports locales del proyecto
from src.transformers import (
    DropColumnsTransformer, UnknownToNaNTransformer, 
    DropHighMissingTransformer, SmartImputerTransformer, OutlierCapper
)

def build_preprocessing_pipeline(df, target_col='stroke', extra_drop_cols=None):
    """
    Builds and returns the complete scikit-learn preprocessing pipeline.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe used to dynamically identify columns.
    target_col : str, default='stroke'
        The target variable to exclude from transformations.
    extra_drop_cols : list of str, optional
        Additional columns to drop (e.g., identifiers like 'id').
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        The assembled preprocessing pipeline.
    """
    
    # Definir columnas a eliminar por defecto si no se proporcionan
    if extra_drop_cols is None:
        extra_drop_cols = ['id'] 
        
    # Excluimos la variable objetivo y los IDs de las transformaciones para evitar Data Leakage
    cols_to_exclude = [target_col] + extra_drop_cols
    feature_df = df.drop(columns=[col for col in cols_to_exclude if col in df.columns], errors='ignore')

    # Ruta de procesamiento para variables numéricas (Edad, IMC, Niveles de Glucosa, etc.)
    num_pipe = Pipeline([
        ('capper', OutlierCapper(apply_capping=True)),
        ('zero_variance', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler())
    ])
    
    # Ruta de procesamiento para variables categóricas (Género, Estado Civil, etc.)
    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Enrutador que detecta automáticamente los tipos de datos en el DataFrame
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, make_column_selector(dtype_include='number')),
            ('cat', cat_pipe, make_column_selector(dtype_exclude='number'))
        ], 
        remainder='passthrough'
    )

    # Construcción del pipeline maestro combinando limpieza estructural y preprocesamiento
    full_pipeline = Pipeline([
        ('drop_leaks', DropColumnsTransformer(columns_to_drop=extra_drop_cols)),
        ('clean_unknowns', UnknownToNaNTransformer()),
        ('drop_high_nan', DropHighMissingTransformer(threshold=0.8)),
        ('smart_imputer', SmartImputerTransformer()),
        ('preprocessing', preprocessor)
    ])

    return full_pipeline