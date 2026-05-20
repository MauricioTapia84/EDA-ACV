import os, subprocess, sys, platform

def create_structure():
    """Crea la arquitectura modular basada en el repositorio del profesor."""
    folders = [
        "data/raw", "data/processed", "docs", "models", "reports",
        "src/0_audit", "src/1_prep", "src/2_unsupervised", 
        "src/3_optuna", "src/4_train", "src/5_report"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, ".gitkeep"), "w") as f: pass
    print("✅ @env-architect: Estructura modular creada.")

def install_deps():
    """Instala dependencias optimizadas para entrenamiento pesado."""
    packages = [
        "pandas", "numpy", "scikit-learn", "optuna", "xgboost", 
        "lightgbm", "yellowbrick", "ipykernel", "matplotlib", "seaborn"
    ]
    suffix = ".exe" if platform.system() == "Windows" else ""
    pip = os.path.join("venv", "Scripts" if platform.system() == "Windows" else "bin", f"pip{suffix}")
    
    if os.path.exists(pip):
        print("🏗️ @env-architect: Instalando/Actualizando dependencias de Fase 2...")
        subprocess.check_call([pip, "install", "--upgrade", "pip"])
        subprocess.check_call([pip, "install"] + packages)
    else:
        print("❌ Error: No se encontró venv. Ejecuta de nuevo.")

if __name__ == "__main__":
    create_structure()
    if not os.path.exists("venv"):
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    install_deps()
    print("🚀 PROYECTO LISTO PARA FASE 2.")
