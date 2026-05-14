import os
import subprocess
import sys
import platform

def run_command(command, shell=True):
    try:
        subprocess.check_call(command, shell=shell)
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando: {command}\n{e}")
        sys.exit(1)

def main():
    print("🚀 Iniciando configuración de entorno para EDA-ACV...")
    
    # 1. Crear entorno virtual
    if not os.path.exists("venv"):
        print("📦 Creando entorno virtual...")
        run_command(f"{sys.executable} -m venv venv")
    
    # 2. Determinar rutas según SO
    is_windows = platform.system() == "Windows"
    pip_path = os.path.join("venv", "Scripts", "pip") if is_windows else os.path.join("venv", "bin", "pip")
    python_path = os.path.join("venv", "Scripts", "python") if is_windows else os.path.join("venv", "bin", "python")

    # 3. Instalar dependencias
    if os.path.exists("requirements.txt"):
        print("📥 Instalando dependencias...")
        run_command(f"{pip_path} install -r requirements.txt")
    else:
        print("⚠️ No se encontró requirements.txt. Saltando instalación.")

    # 4. Registrar Kernel de Jupyter
    print("📓 Registrando kernel de Jupyter (env_acv)...")
    run_command(f"{python_path} -m ipykernel install --user --name=env_acv --display-name 'Python (env_acv)'")

    # 5. Ejecutar proyecto
    entry_point = "main.py" 
    if os.path.exists(entry_point):
        print(f"🏃 Ejecutando {entry_point}...")
        run_command(f"{python_path} {entry_point}")
    else:
        print(f"\n✅ Configuración finalizada con éxito.")
        print(f"💡 Activar entorno: " + (".\\venv\\Scripts\\Activate.ps1" if is_windows else "source venv/bin/activate"))

if __name__ == "__main__":
    main()