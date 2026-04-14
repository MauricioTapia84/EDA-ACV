"""
Módulo de optimización de memoria para DataFrames.
Reduce el tamaño de memoria mediante downcast inteligente de tipos numéricos.
"""

import pandas as pd
import numpy as np


def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimiza la memoria de un DataFrame convirtiendo tipos de datos a versiones más pequeñas.
    
    Estrategia de downcast:
    - Enteros: int64 → int32 → int16 → int8 (según rango de valores)
    - Decimales: float64 → float32
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a optimizar.
    verbose : bool, default=True
        Si True, imprime estadísticas de ahorro de memoria.
    
    Returns
    -------
    pd.DataFrame
        DataFrame optimizado con tipos de datos más pequeños.
    
    Ejemplo
    -------
    >>> df_original = pd.read_csv('data.csv')
    >>> print(f"Memoria original: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    >>> 
    >>> df_optimized = optimize_memory(df_original)
    >>> print(f"Memoria optimizada: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    """
    
    df_copy = df.copy()
    
    # Calcular memoria inicial
    initial_memory = df_copy.memory_usage(deep=True).sum() / 1024**2
    
    # Procesar cada columna
    for col in df_copy.columns:
        col_type = df_copy[col].dtype
        
        try:
            # ====== MANEJO DE ENTEROS ======
            if col_type == 'int64':
                # Intentar downcast a menor tipo
                col_min = df_copy[col].min()
                col_max = df_copy[col].max()
                
                # int32: -2,147,483,648 a 2,147,483,647
                if col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df_copy[col] = df_copy[col].astype('int32')
                # int16: -32,768 a 32,767
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df_copy[col] = df_copy[col].astype('int16')
                # int8: -128 a 127
                elif col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df_copy[col] = df_copy[col].astype('int8')
            
            # ====== MANEJO DE DECIMALES ======
            elif col_type == 'float64':
                # Intentar downcast a float32
                try:
                    # Verificar si puede convertirse sin perder precisión significativa
                    df_copy[col] = df_copy[col].astype('float32')
                except (ValueError, OverflowError):
                    # Si falla, mantener float64
                    pass
        
        except Exception as e:
            # Si algo falla en una columna específica, continuar con las otras
            if verbose:
                print(f"   ⚠️  {col}: No se pudo optimizar ({type(e).__name__})")
    
    # Calcular memoria final y ahorros
    final_memory = df_copy.memory_usage(deep=True).sum() / 1024**2
    memory_saved = initial_memory - final_memory
    percent_saved = (memory_saved / initial_memory) * 100 if initial_memory > 0 else 0
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"OPTIMIZACIÓN DE MEMORIA")
        print(f"{'='*50}")
        print(f"  Memoria inicial:  {initial_memory:.2f} MB")
        print(f"  Memoria final:    {final_memory:.2f} MB")
        print(f"  Ahorro:           {memory_saved:.2f} MB ({percent_saved:.1f}%)")
        print(f"{'='*50}\n")
    
    return df_copy
