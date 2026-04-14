# EDA-ACV
Análisis exploratorio de datos de accidentes cerebrovascular y su probabilidad de que un paciente lo padezca

## Configuración del Entorno Virtual

Para aislar las dependencias de este análisis EDA y asegurar que el código funcione correctamente en cualquier máquina sin conflictos de versiones, es indispensable la creación de un entorno virtual.

### Instrucciones para replicar el entorno (Windows - PowerShell)

1. **Abre tu terminal (PowerShell) y navega a la raíz del proyecto `EDA-ACV`:**
   ```powershell
   cd (ruta_a_tu_proyecto)/EDA-ACV
   ```

2. **Crea el entorno virtual usando el módulo incorporado de Python (`venv`):**
   ```powershell
   python -m venv venv
   ```
   *Esto creará una carpeta llamada `venv` que contiene los binarios de python aislados.*

3. **Activar el entorno virtual:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   *(Nota: Si recibes un error sobre permisos al ejecutar scripts, corre este comando primero como Administrador o simplemente en tu sesión: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

4. **Instalar los paquetes requeridos desde `requirements.txt`:**
   Con el entorno activado (verás `(venv)` al inicio de tu prompt en la terminal), ejecuta:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Añadir el entorno a Jupyter (Kernel):**
   Para asegurarte de que tu Notebook use el kernel correcto con las librerías instaladas:
   ```powershell
   python -m ipykernel install --user --name=env_acv --display-name "Python (env_acv)"
   ```
   *(Luego, en Jupyter Notebook, debes asegurarte de seleccionar el kernel de nombre "Python (env_acv)").*

### Si usas Conda (Alternativa Anaconda/Miniconda)

1. Crear el entorno:
   ```bash
   conda create --name env_acv python=3.10
   ```
2. Activar entorno:
   ```bash
   conda activate env_acv
   ```
3. Instalar librerías:
   ```bash
   pip install -r requirements.txt
   ```
