"""
Orquestador principal del pipeline ETL (Extract, Transform, Load).

Este módulo ejecuta el flujo completo:
1. AUDITORÍA: Verifica integridad de datos
2. CARGA: Busca y carga CSV de data/raw/
3. OPTIMIZACIÓN: Reduce memoria con downcast de tipos
4. TRANSFORMACIÓN: Aplica pipeline de preprocesamiento
5. GUARDADO: Exporta datos limpios a data/processed/

Uso:
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
    Busca el primer archivo CSV en el directorio de datos crudos.
    
    Parámetros
    ----------
    raw_data_dir : str
        Ruta al directorio con datos crudos.
    
    Returns
    -------
    str
        Ruta completa del archivo CSV encontrado.
    
    Raises
    ------
    FileNotFoundError
        Si no hay archivos CSV en el directorio.
    """
    csv_files = list(Path(raw_data_dir).glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"❌ No se encontraron archivos CSV en {raw_data_dir}")
    
    csv_path = str(csv_files[0])
    print(f"📁 CSV encontrado: {csv_path}")
    return csv_path


def main():
    """
    Ejecuta el pipeline ETL completo.
    
    Flujo:
        1. ✅ Auditoría inicial
        2. 📂 Carga de datos
        3. 🔧 Optimización de memoria
        4. 🔄 Transformación (pipeline)
        5. 💾 Guardado de resultados
    """
    
    print("\n" + "="*60)
    print("🚀 INICIANDO PIPELINE ETL")
    print("="*60 + "\n")
    
    try:
        # ============ PASO 1: AUDITORÍA ============
        print("1️⃣  AUDITORÍA INICIAL")
        print("-" * 60)
        
        raw_data_dir = "data/raw"
        processed_data_dir = "data/processed"
        
        # Asegurar que los directorios existen
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # ============ PASO 2: CARGA DE DATOS ============
        print("\n2️⃣  CARGA DE DATOS")
        print("-" * 60)
        
        csv_path = find_csv_file(raw_data_dir)
        
        try:
            df_raw = pd.read_csv(csv_path)
            print(f"✅ Datos cargados: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")
        except pd.errors.EmptyDataError:
            raise ValueError(f"❌ El archivo CSV está vacío: {csv_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"❌ Error al parsear CSV (formato incorrecto): {e}")
        except Exception as e:
            raise ValueError(f"❌ Error inesperado al leer CSV: {e}")
        
        # Auditar datos crudos
        audit_raw = audit_dataframe(
            df_raw,
            stage_name='raw_input',
            log_file='outputs/audit_log.json'
        )
        
        # ============ PASO 3: OPTIMIZACIÓN ============
        print("\n3️⃣  OPTIMIZACIÓN DE MEMORIA")
        print("-" * 60)
        
        df_optimized = optimize_memory(df_raw, verbose=True)
        
        # ============ PASO 4: TRANSFORMACIÓN ============
        print("\n4️⃣  TRANSFORMACIÓN (PIPELINE)")
        print("-" * 60)
        
        # Construcción del pipeline
        print("🔧 Construyendo pipeline de preprocesamiento...")
        pipeline = build_preprocessing_pipeline(df_optimized)
        
        # Aplicación del pipeline
        print("⚙️  Aplicando transformaciones...")
        df_processed = pipeline.fit_transform(df_optimized)
        
        # Convertir a DataFrame si es necesario (ColumnTransformer puede retornar array)
        if not isinstance(df_processed, pd.DataFrame):
            df_processed = pd.DataFrame(df_processed)
        
        print(f"✅ Datos transformados: {df_processed.shape[0]} filas × {df_processed.shape[1]} columnas")
        
        # Auditar datos procesados
        audit_processed = audit_dataframe(
            df_processed,
            stage_name='processed_output',
            log_file='outputs/audit_log.json'
        )
        
        # Comparar antes/después
        print("\n📊 COMPARACIÓN ANTES/DESPUÉS")
        print("-" * 60)
        
        if audit_raw and audit_processed:
            comparison = compare_audits(audit_raw, audit_processed)
            if 'error' not in comparison:
                print(f"  Filas: {audit_raw['shape'][0]} → {audit_processed['shape'][0]}")
                print(f"  Columnas: {audit_raw['shape'][1]} → {audit_processed['shape'][1]}")
                print(f"  Nulos totales: {audit_raw['total_nulls']} → {audit_processed['total_nulls']}")
        
        # ============ PASO 5: GUARDADO ============
        print("\n5️⃣  GUARDADO DE RESULTADOS")
        print("-" * 60)
        
        output_path = os.path.join(processed_data_dir, 'processed_data.csv')
        
        try:
            df_processed.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ Datos guardados en: {output_path}")
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
        
        return 0  # Éxito
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solución:")
        print(f"   1. Verifica que exista la carpeta 'data/raw/'")
        print(f"   2. Coloca tu archivo CSV en esa carpeta")
        print(f"   3. Ejecuta nuevamente: python main.py\n")
        traceback.print_exc()
        return 1
    
    except IndexError as e:
        print(f"\n❌ ERROR: No se encontró el archivo CSV esperado")
        print(f"   Detalles: {e}")
        return 1
    
    except ValueError as e:
        print(f"\n❌ ERROR DE VALIDACIÓN: {e}")
        traceback.print_exc()
        return 1
    
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        print(f"\n🔍 Tipo: {type(e).__name__}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
