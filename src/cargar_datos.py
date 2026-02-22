import os
import pandas as pd


def CargarDatos():
    # Directorio actual del script (es decir, tu carpeta 'src')
    script_dir = os.path.dirname(__file__)
    print(f"DEBUG: script_dir -> {script_dir}")
    
    # Subimos un nivel en la jerarquía para llegar a la carpeta principal (raíz)
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    print(f"DEBUG: root_dir -> {root_dir}")
    
    # Buscamos cualquier archivo que empiece con 'Base_de_datos' en la carpeta raíz
    archivos = [f for f in os.listdir(root_dir) if f.startswith('Base_de_datos')]
    print(f"DEBUG: archivos encontrados -> {archivos}")
    
    if not archivos:
        raise FileNotFoundError(f"No se encontró ningún archivo 'Base_de_datos' en {root_dir}")
    
    # Tomamos el primero que encuentre
    file_name = archivos[0]
    file_path = os.path.join(root_dir, file_name)
    print(f"DEBUG: file_name seleccionado -> {file_name}")
    print(f"DEBUG: file_path -> {file_path}")
    
    print(f"✅ Cargando archivo: {file_name}")
    
    # Detectar si es CSV o Excel por la extensión
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    print(f"DEBUG: Carga finalizada. Dimensiones del DataFrame: {df.shape}")
    return df

if __name__ == "__main__":
    df = CargarDatos()
    print(df.head())
