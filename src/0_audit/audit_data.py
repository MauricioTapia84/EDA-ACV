import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# Configuración de rutas
DATA_PATH = 'data/processed/stroke_data_processed.csv'
REPORT_PATH = 'reports/audit_report.png'
LOG_PATH = 'outputs/audit_log.json'

def audit_data():
    print("Iniciando auditoría de datos...")
    
    # 1. Cargar datos
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el archivo en {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    # 2. Análisis de Colinealidad (VIF)
    # Excluimos la variable objetivo 'stroke'
    X = df.drop(columns=['stroke'])
    
    # Añadir una constante para el intercepto (requerido para VIF)
    X_with_const = X.copy()
    X_with_const['intercept'] = 1
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(len(X_with_const.columns))]
    
    # Filtrar el intercepto
    vif_data = vif_data[vif_data['feature'] != 'intercept'].sort_values(by="VIF", ascending=False)
    
    print("\nResultados VIF:")
    print(vif_data)
    
    # 3. Detección de Fugas de Datos (Data Leakage)
    # Calculamos correlaciones con el target
    correlations = df.corr()['stroke'].sort_values(ascending=False)
    
    print("\nCorrelaciones con 'stroke':")
    print(correlations)
    
    # Alerta si hay correlación perfecta o extremadamente alta (pista de leakage)
    leaks = correlations[correlations.abs() > 0.95].index.tolist()
    leaks = [l for l in leaks if l != 'stroke']
    
    if leaks:
        print(f"\n¡ALERTA! Posibles fugas de datos detectadas en: {leaks}")
    else:
        print("\nNo se detectaron fugas de datos evidentes (correlación > 0.95).")

    # 4. Generar Reporte Visual
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: VIF
    plt.subplot(2, 1, 1)
    sns.barplot(x='VIF', y='feature', data=vif_data, palette='viridis')
    plt.axvline(x=5, color='orange', linestyle='--', label='Umbral Sugerido (5)')
    plt.axvline(x=10, color='red', linestyle='--', label='Umbral Crítico (10)')
    plt.title('Factor de Inflación de la Varianza (VIF)')
    plt.legend()
    
    # Subplot 2: Correlación con Target
    plt.subplot(2, 1, 2)
    correlations_no_target = correlations.drop('stroke')
    sns.barplot(x=correlations_no_target.values, y=correlations_no_target.index, palette='coolwarm')
    plt.title('Correlación de Features con la Variable Objetivo (Stroke)')
    plt.xlabel('Coeficiente de Correlación')
    
    plt.tight_layout()
    plt.savefig(REPORT_PATH)
    print(f"\nReporte visual guardado en: {REPORT_PATH}")

    # 5. Conclusión de preparación
    high_vif = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
    
    ready = True
    reasons = []
    
    if high_vif:
        ready = False
        reasons.append(f"Alta colinealidad detectada en: {high_vif}")
    
    if leaks:
        ready = False
        reasons.append(f"Posibles fugas de datos en: {leaks}")
        
    print("\n--- RESUMEN DE AUDITORÍA ---")
    if ready:
        print("ESTADO: DATOS LISTOS PARA MODELAR")
    else:
        print("ESTADO: SE REQUIEREN AJUSTES")
        for r in reasons:
            print(f"- {r}")

if __name__ == "__main__":
    audit_data()
