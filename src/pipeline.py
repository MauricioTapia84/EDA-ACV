import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from src.transformers import (
    DropColumnsTransformer, UnknownToNaNTransformer, 
    DropHighMissingTransformer, SmartImputerTransformer, OutlierCapper
)

def build_preprocessing_pipeline(df, target_col='stroke', extra_drop_cols=None):
    if extra_drop_cols is None:
        extra_drop_cols = ['id'] 
        
    cols_to_exclude = [target_col] + extra_drop_cols
    feature_df = df.drop(columns=[col for col in cols_to_exclude if col in df.columns], errors='ignore')

    num_pipe = Pipeline([
        ('capper', OutlierCapper(apply_capping=True)),
        ('zero_variance', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, make_column_selector(dtype_include='number')),
            ('cat', cat_pipe, make_column_selector(dtype_exclude='number'))
        ], remainder='passthrough'
    )

    full_pipeline = Pipeline([
        ('drop_leaks', DropColumnsTransformer(columns_to_drop=extra_drop_cols)),
        ('clean_unknowns', UnknownToNaNTransformer()),
        ('drop_high_nan', DropHighMissingTransformer(threshold=0.8)),
        ('smart_imputer', SmartImputerTransformer(low_threshold=0.10)),
        ('preprocessing', preprocessor)
    ])
    return full_pipeline
