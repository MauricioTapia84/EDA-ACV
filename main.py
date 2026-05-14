"""
Main orchestrator for the ETL (Extract, Transform, Load) pipeline.

This module executes the complete flow:
1. AUDIT: Verifies data integrity.
2. LOAD: Searches for and loads the raw CSV data.
3. OPTIMIZATION: Reduces memory footprint using downcasting techniques.
4. TRANSFORMATION: Applies the preprocessing pipeline.
5. SAVE: Exports the clean dataset to the processed folder.

Usage:
    python main.py
"""

import os
import sys
import pandas as pd
import traceback
from pathlib import Path

# Imports locales del proyecto
from src.audit import audit_dataframe, compare_audits
from src.optimization import optimize_memory
from src.pipeline import build_preprocessing_pipeline


def find_csv_file(raw_data_dir: str) -> str:
    """
    Searches for and returns the path to the first CSV file in the specified directory.
    
    Parameters
    ----------
    raw_data_dir : str
        Path to the directory containing raw data.
    
    Returns
    -------
    str
        Complete path to the located CSV file.
    
    Raises
    ------
    FileNotFoundError
        If no CSV files are found in the directory.
    """
    csv_files = list(Path(raw_data_dir).glob('*.csv'))
    
    # Busca automáticamente el archivo para no depender de nombres rígidos (Hardcoding)
    if not csv_files:
        raise FileNotFoundError(f"❌ No se encontraron archivos CSV en {raw_data_dir}")
    
    csv_path = str(csv_files[0])
    print(f"📁 CSV encontrado: {csv_path}")
    return csv_path


def main():
    """Executes the complete ETL pipeline."""
    
    print("="*60)
    print("🏥 PIPELINE DE DATOS: ACCIDENTES CEREBROVASCULARES (ACV)")
    print("="*60)
    
    try:
        # ============ 1. EXTRACCIÓN (CARGA DE DATOS) ============
        print("\n📥 Fase 1: Extracción de datos")
        raw_dir = "data/raw"
        csv_path = find_csv_file(raw_dir)
        df_raw = pd.read_csv(csv_path)
        
        # ============ 2. AUDITORÍA INICIAL ============
        # Asegura que el dataset no haya sido alterado externamente
        print("\n🔍 Fase 2: Auditoría de integridad")
        audit_dataframe(df_raw, "Carga Inicial")
        
        # ============ 3. OPTIMIZACIÓN DE MEMORIA ============
        # Reduce el peso del DataFrame transformando tipos de datos (ej. float64 -> float32)
        print("\n⚙️  Fase 3: Optimización de memoria")
        df_opt = optimize_memory(df_raw)
        
        # ============ 4. PREPROCESAMIENTO (TRANSFORMACIÓN) ============
        print("\n🏗️  Fase 4: Construcción y aplicación del Pipeline")
        pipeline = build_preprocessing_pipeline(df_opt, target_col='stroke')
        
        # Separamos el target antes de transformar (Para evitar que se modifique o escale)
        y = df_opt['stroke'] if 'stroke' in df_opt.columns else None
        
        # Aplicamos la transformación matemática
        processed_matrix = pipeline.fit_transform(df_opt)
        
        # Recuperamos los nombres de las columnas post-transformación (ej. variables One-Hot)
        try:
            feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
            feature_names = [name.split("__")[-1] for name in feature_names]
        except Exception:
            feature_names = [f"feature_{i}" for i in range(processed_matrix.shape[1])]
            
        df_processed = pd.DataFrame(processed_matrix, columns=feature_names, index=df_opt.index)
        
        # Re-acoplamos la variable objetivo limpia al final del dataset
        if y is not None:
            df_processed['stroke'] = y
            
        # ============ 5. CARGA (GUARDADO FINAL) ============
        print("\n💾 Fase 5: Guardado del dataset procesado")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / "stroke_data_processed.csv"
        
        try:
            df_processed.to_csv(output_path, index=False)
            print(f"✅ Archivo generado exitosamente en: {output_path}")
        except Exception as e:
            raise ValueError(f"❌ Error al guardar CSV: {e}")
        
        # ============ RESUMEN FINAL ============
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"\n📋 Resumen:")
        print(f"   • Entrada:  data/raw/{Path(csv_path).name}")
        print(f"   • Salida:   {output_path}")
        print(f"   • Filas procesadas: {df_processed.shape[0]}")
        print(f"   • Columnas finales: {df_processed.shape[1]}")
        print("\n✨ ¡Listo para el análisis o modelado!\n")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solución:")
        print(f"   1. Verifica que exista la carpeta 'data/raw/'")
        print(f"   2. Coloca tu archivo CSV en esa carpeta")
        print(f"   3. Ejecuta nuevamente: python main.py\n")
        traceback.print_exc()
        return 1
    
    except Exception as e:
        print(f"\n❌ ERROR FATAL DE EJECUCIÓN: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())