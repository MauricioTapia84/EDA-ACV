import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        if self.columns_to_drop:
            X_copy.drop(columns=self.columns_to_drop, errors='ignore', inplace=True)
        return X_copy

class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_strings=None):
        self.unknown_strings = unknown_strings if unknown_strings else ['unknown', 'Unknown', 'UNKNOWN', 'na', 'NA', 'N/A', 'n/a', '', 'null', 'NULL', 'None', 'none']
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        categorical_cols = X_copy.select_dtypes(include='object').columns
        for col in categorical_cols:
            X_copy[col] = X_copy[col].replace(self.unknown_strings, np.nan)
        return X_copy

class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.cols_to_drop_ = None
    def fit(self, X, y=None):
        missing_pct = X.isnull().sum() / len(X)
        self.cols_to_drop_ = missing_pct[missing_pct > self.threshold].index.tolist()
        return self
    def transform(self, X):
        X_copy = X.copy()
        if self.cols_to_drop_:
            X_copy.drop(columns=self.cols_to_drop_, errors='ignore', inplace=True)
        return X_copy

class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, low_threshold=0.10, high_threshold=0.50):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.impute_dict_ = {}
    def fit(self, X, y=None):
        missing_pct = X.isnull().mean()
        for col in X.columns:
            pct = missing_pct[col]
            if pct > 0:
                # Estrategia segura: Mediana para numéricos, Moda para categóricos
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.impute_dict_[col] = X[col].median()
                else:
                    mode_val = X[col].mode()
                    self.impute_dict_[col] = mode_val[0] if len(mode_val) > 0 else 'missing'
        return self
    def transform(self, X):
        X_copy = X.copy()
        for col, value in self.impute_dict_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(value)
        return X_copy

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, apply_capping=True, iqr_factor=1.5):
        self.apply_capping = apply_capping
        self.iqr_factor = iqr_factor
        self.caps_ = {}
    def fit(self, X, y=None):
        if not self.apply_capping: return self
        numeric_cols = X.select_dtypes(include='number').columns
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.caps_[col] = (Q1 - self.iqr_factor * IQR, Q3 + self.iqr_factor * IQR)
        return self
    def transform(self, X):
        if not self.apply_capping: return X
        X_copy = X.copy()
        for col, (lower, upper) in self.caps_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(lower, upper)
        return X_copy
