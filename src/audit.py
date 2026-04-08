"""
Módulo de auditoría para validación de integridad de datos.
Contiene funciones para checksums, validación de esquema y logging de auditoría.
"""

import hashlib
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union


def compute_checksum(df: pd.DataFrame, algorithm: str = 'md5') -> str:
    """
    Calcula el checksum (huella digital) de un DataFrame.
    
    Útil para verificar que los datos no han cambiado entre etapas.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a auditar.
    algorithm : str, default='md5'
        Algoritmo de hash ('md5', 'sha1', 'sha256').
    
    Returns
    -------
    str
        Hash hexadecimal del DataFrame.
    
    Ejemplo
    -------
    >>> checksum = compute_checksum(df)
    >>> print(f"Huella digital: {checksum}")
    """
    if df is None or len(df) == 0:
        return hashlib.md5(b'').hexdigest()
    
    # Convertir el DataFrame a una representación hashable
    try:
        hash_series = pd.util.hash_pandas_object(df, index=True)
        combined_hash = hashlib.new(algorithm)
        for h in hash_series:
            combined_hash.update(str(h).encode())
        return combined_hash.hexdigest()
    except Exception as e:
        # Fallback: convertir a string y hashear
        print(f"[Audit] Advertencia: Fallback en checksum: {e}")
        return hashlib.md5(df.to_csv().encode()).hexdigest()


def validate_schema(df: pd.DataFrame, 
                    expected_columns: List[str],
                    exact_match: bool = False) -> Dict:
    """
    Valida que el DataFrame tenga las columnas esperadas.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a validar.
    expected_columns : list
        Lista de nombres de columnas esperadas.
    exact_match : bool, default=False
        Si es True, requiere que las columnas sean exactamente las esperadas.
        Si es False, solo verifica que las esperadas estén presentes.
    
    Returns
    -------
    dict
        Diccionario con resultados de validación:
        - valid: bool
        - missing_columns: list
        - extra_columns: list
        - suggestion: str
    """
    if df is None:
        return {
            'valid': False,
            'missing_columns': expected_columns,
            'extra_columns': [],
            'suggestion': 'DataFrame es None'
        }
    
    df_columns = set(df.columns)
    expected_set = set(expected_columns)
    
    missing_columns = list(expected_set - df_columns)
    extra_columns = list(df_columns - expected_set)
    
    valid = len(missing_columns) == 0
    if exact_match:
        valid = valid and len(extra_columns) == 0
    
    suggestion = ""
    if missing_columns:
        suggestion = f"Faltan columnas: {missing_columns}"
    elif extra_columns and exact_match:
        suggestion = f"Columnas extra: {extra_columns}"
    elif extra_columns and not exact_match:
        suggestion = f"Columnas extra detectadas (no hay problema): {extra_columns[:5]}..."
    
    return {
        'valid': valid,
        'missing_columns': missing_columns,
        'extra_columns': extra_columns,
        'suggestion': suggestion
    }


def audit_dataframe(df: pd.DataFrame, 
                    stage_name: str, 
                    log_file: str = 'outputs/audit_log.json',
                    expected_columns: Optional[List[str]] = None) -> Dict:
    """
    Realiza una auditoría completa de un DataFrame y guarda los resultados.
    
    Esta función combina checksums, estadísticas y validación de esquema.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a auditar.
    stage_name : str
        Nombre de la etapa (ej: 'raw_input', 'after_cleaning', 'processed_output').
    log_file : str, default='outputs/audit_log.json'
        Ruta del archivo donde guardar el log de auditoría.
    expected_columns : list, optional
        Lista de columnas esperadas para validación de esquema.
    
    Returns
    -------
    dict
        Diccionario con todos los resultados de auditoría.
    
    Ejemplo
    -------
    >>> audit_dataframe(df_raw, 'raw_input', expected_columns=['age', 'job'])
    >>> # Después de limpiar
    >>> audit_dataframe(df_clean, 'after_cleaning')
    """
    if df is None:
        print(f"[Audit] Error: DataFrame es None en etapa '{stage_name}'")
        return {'error': 'DataFrame es None', 'stage': stage_name}
    
    # 1. Métricas básicas (convertir a tipos nativos de Python para JSON)
    total_rows = int(len(df))
    total_columns = int(len(df.columns))
    total_nulls = int(df.isnull().sum().sum())
    
    # Convertir nulls_by_column a tipos nativos
    nulls_by_column = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items() if v > 0}
    
    # 2. Estadísticas por tipo de dato
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 3. Checksum
    checksum = compute_checksum(df)
    
    # 4. Validación de esquema (si se proporcionaron columnas esperadas)
    schema_validation = None
    if expected_columns:
        schema_validation = validate_schema(df, expected_columns)
        # Convertir a tipos nativos
        if schema_validation:
            schema_validation['missing_columns'] = [str(c) for c in schema_validation['missing_columns']]
            schema_validation['extra_columns'] = [str(c) for c in schema_validation['extra_columns']]
    
    # 5. Detección de problemas comunes
    warnings = []
    
    # Detectar columnas con muchos nulos
    high_null_cols = []
    for col, nulls in df.isnull().sum().to_dict().items():
        if nulls / total_rows > 0.5:
            high_null_cols.append(str(col))
    if high_null_cols:
        warnings.append(f"Columnas con >50% nulos: {high_null_cols}")
    
    # Detectar columnas constantes (varianza cero)
    constant_cols = []
    for col in numeric_cols:
        try:
            if df[col].var() == 0:
                constant_cols.append(str(col))
        except Exception:
            pass
    if constant_cols:
        warnings.append(f"Columnas numéricas constantes: {constant_cols}")
    
    # Detectar valores 'unknown' en categóricas
    unknown_values = []
    unknown_strings = ['unknown', 'Unknown', 'UNKNOWN', 'na', 'NA', '', 'null', 'NULL', 'None']
    for col in categorical_cols:
        try:
            unknown_count = int(df[col].astype(str).isin(unknown_strings).sum())
            if unknown_count > 0:
                unknown_values.append(f"{col}: {unknown_count}")
        except Exception:
            pass
    if unknown_values:
        warnings.append(f"Valores 'unknown' detectados: {unknown_values[:3]}")
    
    # 6. Construir resultado de auditoría (TODO convertido a tipos Python nativos)
    audit_entry = {
        'timestamp': datetime.now().isoformat(),
        'stage': str(stage_name),
        'shape': [total_rows, total_columns],
        'total_nulls': total_nulls,
        'nulls_by_column': nulls_by_column,
        'numeric_columns': int(len(numeric_cols)),
        'categorical_columns': int(len(categorical_cols)),
        'checksum': str(checksum),
        'columns': [str(c) for c in df.columns.tolist()],
        'warnings': [str(w) for w in warnings],
        'schema_validation': schema_validation
    }
    
    # 7. Guardar en archivo log
    try:
        # Crear directorio si no existe
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Leer entradas existentes si el archivo ya existe
        existing_entries = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_entries.append(line)
            except Exception as e:
                print(f"[Audit] Advertencia al leer log existente: {e}")
        
        # Escribir todas las entradas (viejas + nueva)
        with open(log_file, 'w', encoding='utf-8') as f:
            for entry in existing_entries:
                f.write(entry + '\n')
            f.write(json.dumps(audit_entry, ensure_ascii=False) + '\n')
        
        print(f"[Audit] Log guardado en {log_file}")
    except Exception as e:
        print(f"[Audit] Advertencia: No se pudo guardar el log: {e}")
    
    # 8. Imprimir resumen
    print(f"\n{'='*50}")
    print(f"AUDITORÍA: {stage_name.upper()}")
    print(f"{'='*50}")
    print(f"  Shape: {total_rows} filas × {total_columns} columnas")
    print(f"  Nulos totales: {total_nulls}")
    print(f"  Checksum: {checksum[:16]}...")
    if schema_validation:
        print(f"  Esquema válido: {'✅ Sí' if schema_validation['valid'] else '❌ No'}")
        if schema_validation.get('suggestion'):
            print(f"  {schema_validation['suggestion']}")
    if warnings:
        print(f"\n  ⚠️ Advertencias ({len(warnings)}):")
        for w in warnings[:3]:
            print(f"     - {w}")
    print(f"{'='*50}\n")
    
    return audit_entry


def compare_audits(audit_before: Dict, audit_after: Dict) -> Dict:
    """
    Compara dos auditorías (antes/después) y muestra los cambios.
    
    Útil para verificar el impacto de las transformaciones.
    
    Parámetros
    ----------
    audit_before : dict
        Auditoría antes de la transformación.
    audit_after : dict
        Auditoría después de la transformación.
    
    Returns
    -------
    dict
        Diccionario con las diferencias encontradas.
    
    Ejemplo
    -------
    >>> before = audit_dataframe(df_raw, 'raw')
    >>> df_clean = pipeline.fit_transform(df_raw)
    >>> after = audit_dataframe(df_clean, 'clean')
    >>> diff = compare_audits(before, after)
    """
    if not audit_before or not audit_after:
        return {'error': 'Una o ambas auditorías son inválidas'}
    
    if 'error' in audit_before or 'error' in audit_after:
        return {'error': 'Una o ambas auditorías contienen errores'}
    
    rows_before, cols_before = audit_before.get('shape', [0, 0])
    rows_after, cols_after = audit_after.get('shape', [0, 0])
    
    nulls_before = audit_before.get('total_nulls', 0)
    nulls_after = audit_after.get('total_nulls', 0)
    
    nulls_reduction = nulls_before - nulls_after
    nulls_reduction_pct = (nulls_reduction / nulls_before * 100) if nulls_before > 0 else 0
    
    comparison = {
        'rows_change': rows_after - rows_before,
        'cols_change': cols_after - cols_before,
        'nulls_reduction': nulls_reduction,
        'nulls_reduction_pct': nulls_reduction_pct,
        'checksum_changed': audit_before.get('checksum') != audit_after.get('checksum')
    }
    
    print(f"\n{'='*50}")
    print("COMPARACIÓN DE AUDITORÍAS")
    print(f"{'='*50}")
    print(f"  Filas: {rows_before} → {rows_after} ({comparison['rows_change']:+d})")
    print(f"  Columnas: {cols_before} → {cols_after} ({comparison['cols_change']:+d})")
    print(f"  Nulos: {nulls_before} → {nulls_after} (reducción: {comparison['nulls_reduction']} - {comparison['nulls_reduction_pct']:.1f}%)")
    print(f"  Checksum cambió: {'✅ Sí' if comparison['checksum_changed'] else '⚠️ No'}")
    print(f"{'='*50}\n")
    
    return comparison


def validate_data_quality(df: pd.DataFrame, 
                          numeric_bounds: Optional[Dict] = None,
                          categorical_values: Optional[Dict] = None) -> Dict:
    """
    Valida la calidad de los datos según reglas personalizadas.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a validar.
    numeric_bounds : dict, optional
        Diccionario con límites para columnas numéricas.
        Ejemplo: {'age': (0, 120), 'salary': (0, 1_000_000)}
    categorical_values : dict, optional
        Diccionario con valores permitidos para columnas categóricas.
        Ejemplo: {'job': ['engineer', 'doctor', 'teacher']}
    
    Returns
    -------
    dict
        Resultados de validación con problemas encontrados.
    
    Ejemplo
    -------
    >>> bounds = {'age': (0, 120)}
    >>> result = validate_data_quality(df, numeric_bounds=bounds)
    """
    issues = []
    
    if numeric_bounds:
        for col, (min_val, max_val) in numeric_bounds.items():
            if col in df.columns:
                out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_bounds) > 0:
                    issues.append({
                        'column': str(col),
                        'type': 'numeric_out_of_bounds',
                        'count': int(len(out_of_bounds)),
                        'expected_range': (float(min_val), float(max_val)),
                        'actual_values': [float(x) if isinstance(x, (int, float)) else str(x) for x in out_of_bounds[col].tolist()[:5]]
                    })
    
    if categorical_values:
        for col, allowed_values in categorical_values.items():
            if col in df.columns:
                invalid = df[~df[col].isin(allowed_values) & ~df[col].isna()]
                if len(invalid) > 0:
                    unique_invalid = invalid[col].unique().tolist()
                    issues.append({
                        'column': str(col),
                        'type': 'invalid_categorical',
                        'count': int(len(invalid)),
                        'allowed_values': [str(v) for v in allowed_values[:10]],
                        'invalid_values_found': [str(v) for v in unique_invalid[:5]]
                    })
    
    result = {
        'valid': len(issues) == 0,
        'issues_count': len(issues),
        'issues': issues
    }
    
    print(f"\nValidación de calidad de datos:")
    print(f"  ✅ Válido: {result['valid']}")
    if issues:
        print(f"  ⚠️ Problemas encontrados: {len(issues)}")
        for issue in issues[:3]:
            print(f"     - {issue['column']}: {issue['type']} ({issue['count']} casos)")
    
    return result


# ============================================
# Función adicional útil para limpiar el log
# ============================================

def clear_audit_log(log_file: str = 'outputs/audit_log.json') -> bool:
    """
    Limpia el archivo de log de auditoría.
    
    Parámetros
    ----------
    log_file : str, default='outputs/audit_log.json'
        Ruta del archivo de log.
    
    Returns
    -------
    bool
        True si se limpió correctamente, False en caso contrario.
    
    Ejemplo
    -------
    >>> clear_audit_log()
    """
    try:
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"[Audit] Log limpiado: {log_file}")
        return True
    except Exception as e:
        print(f"[Audit] Error al limpiar log: {e}")
        return False