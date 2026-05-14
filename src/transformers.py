"""
Custom Scikit-Learn transformers for data cleaning and preprocessing.
Contains transformers for dropping columns, handling unknown values, and capping outliers.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to drop specified columns from a pandas DataFrame.
    
    Parameters
    ----------
    columns_to_drop : list of str, optional
        List of column names to drop.
    """
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []
        
    def fit(self, X, y=None):
        """Fits the transformer. No-op for this transformer."""
        return self
        
    def transform(self, X):
        """Drops the specified columns from the DataFrame."""
        X_copy = X.copy()
        
        # Elimina las columnas indicadas ignorando errores si ya no existen
        if self.columns_to_drop:
            X_copy.drop(columns=self.columns_to_drop, errors='ignore', inplace=True)
        return X_copy

class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):
    """
    Converts specific string representations of unknown values to np.nan.
    
    Parameters
    ----------
    unknown_strings : list of str, optional
        List of strings that should be treated as missing values.
    """
    def __init__(self, unknown_strings=None):
        self.unknown_strings = unknown_strings if unknown_strings else [
            'unknown', 'Unknown', 'UNKNOWN', 'na', 'NA', 'N/A', 'n/a', '', 'null', 'NULL', 'None', 'none'
        ]
        
    def fit(self, X, y=None):
        """Fits the transformer. No-op for this transformer."""
        return self
        
    def transform(self, X):
        """Replaces unknown strings with numpy NaN in categorical columns."""
        X_copy = X.copy()
        
        # Filtra solo las variables categóricas (texto)
        categorical_cols = X_copy.select_dtypes(include='object').columns
        
        # Reemplaza los valores basura por nulos reales de Numpy
        for col in categorical_cols:
            X_copy[col] = X_copy[col].replace(self.unknown_strings, np.nan)
        return X_copy

class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """
    Drops columns with a missing value percentage exceeding a specified threshold.
    
    Parameters
    ----------
    threshold : float, default=0.8
        The maximum allowed proportion of missing values (0.0 to 1.0).
    """
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.cols_to_drop_ = None
        
    def fit(self, X, y=None):
        """Identifies columns that exceed the missing value threshold."""
        # Calcula la proporción de nulos por columna en el set de entrenamiento
        missing_ratios = X.isnull().mean()
        
        # Identifica qué columnas superan el límite establecido
        self.cols_to_drop_ = missing_ratios[missing_ratios > self.threshold].index.tolist()
        return self
        
    def transform(self, X):
        """Drops the columns identified during fit."""
        X_copy = X.copy()
        
        # Descarta variables si tienen demasiados datos faltantes (para no inventar datos)
        if self.cols_to_drop_:
            X_copy.drop(columns=self.cols_to_drop_, errors='ignore', inplace=True)
        return X_copy

class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using median for numeric and mode for categorical columns.
    Learns imputation values during fit to prevent data leakage.
    """
    def __init__(self):
        self.impute_dict_ = {}
        
    def fit(self, X, y=None):
        """Calculates and stores median/mode for each column containing missing values."""
        pct_missing = X.isnull().mean()
        
        # Aprende las medianas y modas SOLO en el conjunto de entrenamiento (Evita Data Leakage)
        for col in X.columns:
            pct = pct_missing[col]
            if pct > 0:
                # Estrategia segura: Mediana para numéricos, Moda para categóricos
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.impute_dict_[col] = X[col].median()
                else:
                    mode_val = X[col].mode()
                    self.impute_dict_[col] = mode_val[0] if len(mode_val) > 0 else 'missing'
        return self
        
    def transform(self, X):
        """Applies the learned imputation values to missing entries."""
        X_copy = X.copy()
        
        # Aplica los valores guardados de forma segura sobre los datos nuevos
        for col, value in self.impute_dict_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(value)
        return X_copy

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps numeric outliers using the Interquartile Range (IQR) method.
    
    Parameters
    ----------
    apply_capping : bool, default=True
        Whether to apply the capping logic or bypass it.
    iqr_factor : float, default=1.5
        Multiplier for the IQR to determine outlier bounds.
    """
    def __init__(self, apply_capping=True, iqr_factor=1.5):
        self.apply_capping = apply_capping
        self.iqr_factor = iqr_factor
        self.caps_ = {}
        
    def fit(self, X, y=None):
        """Calculates IQR bounds for all numeric columns."""
        if not self.apply_capping: 
            return self
            
        numeric_cols = X.select_dtypes(include='number').columns
        
        # Calcula límites inferiores y superiores para todas las variables numéricas
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.caps_[col] = (Q1 - self.iqr_factor * IQR, Q3 + self.iqr_factor * IQR)
        return self
        
    def transform(self, X):
        """Clips numeric values based on the bounds calculated during fit."""
        X_copy = X.copy()
        if not self.apply_capping: 
            return X_copy
            
        # Recorta valores atípicos (Winsorización) para no perder datos reales de pacientes
        for col, (lower, upper) in self.caps_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], lower, upper)
        return X_copy