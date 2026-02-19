import os
import pandas as pd

def CargarDatos():
    # Directorio actual del script
    script_dir = os.path.dirname(__file__)
    
    # Buscamos cualquier archivo que empiece con 'Base_de_datos'
    archivos = [f for f in os.listdir(script_dir) if f.startswith('Base_de_datos')]
    
    if not archivos:
        raise FileNotFoundError(f"No se encontró ningún archivo 'Base_de_datos' en {script_dir}")
    
    # Tomamos el primero que encuentre
    file_name = archivos[0]
    file_path = os.path.join(script_dir, file_name)
    
    print(f"✅ Cargando archivo: {file_name}")
    
    # Detectar si es CSV o Excel por la extensión
    if file_name.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)