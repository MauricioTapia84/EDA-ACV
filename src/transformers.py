"""
Módulo de transformers personalizados para pipeline de preprocessing.
Cada transformer hereda de BaseEstimator y TransformerMixin de scikit-learn.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas especificadas del DataFrame.
    
    Parámetros
    ----------
    columns_to_drop : list, default=None
        Lista de nombres de columnas a eliminar.
    
    Ejemplo
    -------
    >>> drop_cols = DropColumnsTransformer(columns_to_drop=['id', 'date'])
    >>> df_limpio = drop_cols.fit_transform(df)
    """
    
    def __init__(self, columns_to_drop=None):
        # Si no se especifican columnas, usar lista vacía
        self.columns_to_drop = columns_to_drop if columns_to_drop else []
    
    def fit(self, X, y=None):
        """
        Método fit: no hace nada porque no necesita aprender nada.
        Solo existe por compatibilidad con la API de scikit-learn.
        """
        return self
    
    def transform(self, X):
        """
        Aplica la transformación: elimina las columnas indicadas.
        
        Parámetros
        ----------
        X : pd.DataFrame
            Datos de entrada.
        
        Returns
        -------
        pd.DataFrame
            DataFrame sin las columnas especificadas.
        """
        # Crear copia para no modificar el original
        X_copy = X.copy()
        
        # Filtrar solo columnas que existen en el DataFrame
        cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        
        # Eliminar columnas si hay alguna
        if cols_to_drop:
            X_copy.drop(columns=cols_to_drop, inplace=True)
            print(f"[DropColumnsTransformer] Columnas eliminadas: {cols_to_drop}")
        else:
            print("[DropColumnsTransformer] No se eliminó ninguna columna")
        
        return X_copy


class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):
    """
    Convierte strings como 'unknown', 'na', '' a NaN.
    
    Parámetros
    ----------
    unknown_strings : list, default=None
        Lista de strings a considerar como valores desconocidos.
        Si es None, usa valores por defecto.
    
    Ejemplo
    -------
    >>> cleaner = UnknownToNaNTransformer()
    >>> df_limpio = cleaner.fit_transform(df)
    """
    
    def __init__(self, unknown_strings=None):
        # Lista por defecto de strings que representan valores desconocidos
        if unknown_strings is None:
            self.unknown_strings = [
                'unknown', 'Unknown', 'UNKNOWN',
                'na', 'NA', 'N/A', 'n/a',
                '', 'null', 'NULL', 'None', 'none'
            ]
        else:
            self.unknown_strings = unknown_strings
    
    def fit(self, X, y=None):
        """No necesita aprender nada, solo existe por API."""
        return self
    
    def transform(self, X):
        """
        Reemplaza strings desconocidos por NaN.
        
        Parámetros
        ----------
        X : pd.DataFrame
            Datos de entrada.
        
        Returns
        -------
        pd.DataFrame
            DataFrame con valores desconocidos convertidos a NaN.
        """
        X_copy = X.copy()
        
        # Identificar columnas categóricas (tipo object)
        categorical_cols = X_copy.select_dtypes(include='object').columns
        
        if len(categorical_cols) == 0:
            print("[UnknownToNaNTransformer] No hay columnas categóricas para procesar")
            return X_copy
        
        # Aplicar reemplazo en cada columna categórica
        for col in categorical_cols:
            # Contar cuántos valores desconocidos había
            unknown_mask = X_copy[col].isin(self.unknown_strings)
            unknown_count = unknown_mask.sum()
            
            # Reemplazar por NaN
            X_copy[col] = X_copy[col].replace(self.unknown_strings, np.nan)
            
            if unknown_count > 0:
                print(f"[UnknownToNaNTransformer] Columna '{col}': {unknown_count} valores 'unknown' → NaN")
        
        return X_copy
    


class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas con más de `threshold` (0-1) valores nulos.
    
    Parámetros
    ----------
    threshold : float, default=0.8
        Porcentaje máximo de nulos permitido (0.8 = 80%).
        Columnas con nulos > threshold serán eliminadas.
    
    Ejemplo
    -------
    >>> drop_high = DropHighMissingTransformer(threshold=0.7)
    >>> df_limpio = drop_high.fit_transform(df)
    """
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.cols_to_drop_ = None  # Guardará las columnas a eliminar
    
    def fit(self, X, y=None):
        """
        Identifica qué columnas superan el umbral de nulos.
        """
        # Calcular porcentaje de nulos por columna
        missing_pct = X.isnull().sum() / len(X)
        
        # Identificar columnas que superan el umbral
        self.cols_to_drop_ = missing_pct[missing_pct > self.threshold].index.tolist()
        
        # Mostrar información
        if self.cols_to_drop_:
            print(f"[DropHighMissingTransformer] Se eliminarán {len(self.cols_to_drop_)} columnas con >{self.threshold*100}% nulos:")
            for col in self.cols_to_drop_:
                pct = missing_pct[col] * 100
                print(f"   - {col}: {pct:.1f}% nulos")
        else:
            print(f"[DropHighMissingTransformer] Ninguna columna supera el {self.threshold*100}% de nulos")
        
        return self
    
    def transform(self, X):
        """
        Elimina las columnas identificadas en fit().
        """
        X_copy = X.copy()
        
        if self.cols_to_drop_:
            X_copy.drop(columns=self.cols_to_drop_, inplace=True)
        
        return X_copy

class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Imputa valores nulos de forma inteligente según el porcentaje de nulos.
    
    Estrategias:
    - Bajo % nulos (< low_threshold): usa mediana (numéricas) o moda (categóricas)
    - Nulos moderados (low_threshold a high_threshold): imputa con 'missing' (categóricas) o mediana (numéricas)
    - Alto % nulos (> high_threshold): advierte (deberían haberse dropeado antes)
    
    Parámetros
    ----------
    low_threshold : float, default=0.10
        Umbral bajo (10%). Nulos < 10% se imputan con mediana/moda.
    high_threshold : float, default=0.50
        Umbral alto (50%). Nulos entre 10%-50% usan estrategia especial.
    
    Ejemplo
    -------
    >>> imputer = SmartImputerTransformer(low_threshold=0.10, high_threshold=0.50)
    >>> df_imputado = imputer.fit_transform(df)
    """
    
    def __init__(self, low_threshold=0.10, high_threshold=0.50):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.impute_dict_ = {}
    
    def fit(self, X, y=None):
        """
        Calcula los valores de imputación para cada columna con nulos.
        """
        missing_pct = X.isnull().sum() / len(X)
        
        for col in X.columns:
            pct = missing_pct[col]
            
            # Caso 1: Sin nulos - no hacer nada
            if pct == 0:
                continue
            
            # Caso 2: Bajo % de nulos (< low_threshold)
            elif pct <= self.low_threshold:
                if pd.api.types.is_numeric_dtype(X[col]):
                    value = X[col].median()
                    print(f"[SmartImputer] Columna '{col}': {pct*100:.1f}% nulos → imputar con mediana={value:.2f}")
                else:
                    mode_val = X[col].mode()
                    if len(mode_val) > 0:
                        value = mode_val[0]
                    else:
                        value = 'missing'
                    print(f"[SmartImputer] Columna '{col}': {pct*100:.1f}% nulos → imputar con moda='{value}'")
                
                self.impute_dict_[col] = value
            
            # Caso 3: Nulos moderados (low_threshold a high_threshold)
            elif pct <= self.high_threshold:
                if pd.api.types.is_numeric_dtype(X[col]):
                    value = X[col].median()
                    print(f"[SmartImputer] Columna '{col}': {pct*100:.1f}% nulos → imputar con mediana={value:.2f}")
                else:
                    value = 'missing'
                    print(f"[SmartImputer] Columna '{col}': {pct*100:.1f}% nulos → crear categoría 'missing'")
                
                self.impute_dict_[col] = value
            
            # Caso 4: Alto % de nulos (> high_threshold)
            else:
                print(f"[SmartImputer] ADVERTENCIA: Columna '{col}' tiene {pct*100:.1f}% nulos.")
                print(f"   → Considere usar DropHighMissingTransformer antes de imputar.")
                if pd.api.types.is_numeric_dtype(X[col]):
                    value = X[col].median()
                else:
                    value = 'missing'
                self.impute_dict_[col] = value
        
        return self
    
    def transform(self, X):
        """
        Aplica la imputación a los datos.
        """
        X_copy = X.copy()
        
        for col, value in self.impute_dict_.items():
            if col in X_copy.columns and X_copy[col].isnull().any():
                nulls_before = X_copy[col].isnull().sum()
                X_copy[col] = X_copy[col].fillna(value)  # ← Sin inplace=True
                print(f"[SmartImputer] Columna '{col}': imputados {nulls_before} nulos")
        
        # Método de respaldo para nulos restantes (sin usar 'method')
        if X_copy.isnull().sum().sum() > 0:
            print("[SmartImputer] Advertencia: Aún hay nulos. Aplicando método de respaldo...")
            
            # Rellenar hacia adelante (ffill)
            X_copy = X_copy.ffill()
            
            # Rellenar hacia atrás (bfill)
            X_copy = X_copy.bfill()
            
            # Si aún hay nulos, rellenar con valores por defecto
            for col in X_copy.columns:
                if X_copy[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(X_copy[col]):
                        X_copy[col] = X_copy[col].fillna(0)
                    else:
                        X_copy[col] = X_copy[col].fillna('unknown')
        
        return X_copy


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Aplica capping a outliers usando el método IQR (Rango Intercuartil).
    Esta es la técnica estándar y profesional para detectar outliers.
    
    Parámetros
    ----------
    apply_capping : bool, default=True
        Si es False, no aplica capping.
    iqr_factor : float, default=1.5
        Factor multiplicador del IQR (1.5 = estándar, 3.0 = muy conservador)
    
    La fórmula es:
        Límite inferior = Q1 - iqr_factor * IQR
        Límite superior = Q3 + iqr_factor * IQR
    
    Ejemplo
    -------
    >>> capper = OutlierCapper(iqr_factor=1.5)
    >>> df_sin_outliers = capper.fit_transform(df)
    """
    
    def __init__(self, apply_capping=True, iqr_factor=1.5):
        self.apply_capping = apply_capping
        self.iqr_factor = iqr_factor
        self.caps_ = {}
    
    def fit(self, X, y=None):
        """
        Calcula los límites usando IQR para cada columna numérica.
        """
        if not self.apply_capping:
            print("[OutlierCapper] Capping desactivado")
            return self
        
        numeric_cols = X.select_dtypes(include='number').columns
        
        for col in numeric_cols:
            # Calcular Q1 (25%) y Q3 (75%)
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calcular límites
            lower = Q1 - self.iqr_factor * IQR
            upper = Q3 + self.iqr_factor * IQR
            
            self.caps_[col] = (lower, upper)
            
            # Contar outliers
            outliers_before = ((X[col] < lower) | (X[col] > upper)).sum()
            
            if outliers_before > 0:
                print(f"[OutlierCapper] Columna '{col}':")
                print(f"   - Q1 (25%): {Q1:.2f}")
                print(f"   - Q3 (75%): {Q3:.2f}")
                print(f"   - IQR: {IQR:.2f}")
                print(f"   - Límite inferior: {lower:.2f}")
                print(f"   - Límite superior: {upper:.2f}")
                print(f"   - Outliers detectados: {outliers_before}")
                print(f"   - Rango original: [{X[col].min():.2f}, {X[col].max():.2f}]")
        
        return self
    
    def transform(self, X):
        """
        Aplica el capping.
        """
        if not self.apply_capping:
            return X
        
        X_copy = X.copy()
        
        for col, (lower, upper) in self.caps_.items():
            if col in X_copy.columns:
                original_min = X_copy[col].min()
                original_max = X_copy[col].max()
                
                # Aplicar capping
                X_copy[col] = X_copy[col].clip(lower, upper)
                
                new_min = X_copy[col].min()
                new_max = X_copy[col].max()
                
                if (original_min != new_min) or (original_max != new_max):
                    print(f"[OutlierCapper] Columna '{col}':")
                    print(f"   - Min: {original_min:.2f} → {new_min:.2f}")
                    print(f"   - Max: {original_max:.2f} → {new_max:.2f}")
        
        return X_copy

class DropZeroVarianceTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas numéricas con varianza cero (todos los valores iguales).
    
    Ejemplo
    -------
    >>> drop_zero = DropZeroVarianceTransformer()
    >>> df_sin_constantes = drop_zero.fit_transform(df)
    """
    
    def __init__(self):
        self.cols_to_drop_ = None
    
    def fit(self, X, y=None):
        """
        Identifica columnas numéricas con varianza cero.
        """
        numeric_cols = X.select_dtypes(include='number').columns
        self.cols_to_drop_ = []
        
        for col in numeric_cols:
            if X[col].var() == 0:
                self.cols_to_drop_.append(col)
                print(f"[DropZeroVariance] Columna '{col}' tiene varianza cero (todos los valores = {X[col].iloc[0]}) → será eliminada")
        
        if not self.cols_to_drop_:
            print("[DropZeroVariance] No se encontraron columnas con varianza cero")
        
        return self
    
    def transform(self, X):
        """
        Elimina las columnas identificadas.
        """
        X_copy = X.copy()
        
        if self.cols_to_drop_:
            X_copy.drop(columns=self.cols_to_drop_, inplace=True)
        
        return X_copy