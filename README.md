# 🚀 MLOps Pipeline - Credit Scoring Project

Este repositorio contiene la implementación paso a paso de un pipeline de MLOps automatizado para la predicción de riesgo crediticio (Credit Scoring).

## 📂 Estructura del Proyecto

*   `src/cargar_datos.py`: Script para cargar y preparar el dataset [Base_de_datos.xlsx](cci:7://file:///c:/Users/RuVe7/Desktop/ProyectoM5_MatiasGutierrez/Pi_pt_ds_01/mlops_pipeline/Base_de_datos.xlsx:0:0-0:0).
*   `src/comprension_eda.ipynb`: Análisis Exploratorio de Datos (EDA) para entender las distribuciones, identificar outliers (ej. `tipo_credito` muy poco frecuentes) y plantear relaciones con la variable objetivo `Pago_atiempo`.
*   [requirements.txt](cci:7://file:///c:/Users/RuVe7/Desktop/ProyectoM5_MatiasGutierrez/Pi_pt_ds_01/mlops_pipeline/requirements.txt:0:0-0:0): Dependencias de Python necesarias para correr el pipeline.

## 🛠 Instalación y Uso Local

1.  **Clonar este repositorio:**
    ```bash
    git clone https://github.com/RuloVerde746/mlops_pipeline/tree/developer
    cd Pi_pt_ds_01/mlops_pipeline
    ```
2.  **Crear y activar el entorno virtual:**
    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    ```
3.  **Instalar dependencias necesarias:**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Avances (Versión actual 1.0.1)
- [x] Configuración del entorno de desarrollo (venv, requirements).
- [x] Construcción de la función base de carga de datos sin conexión forzada.
- [x] EDA visual completado (Tratamiento de nulos cruzando variables, filtrado de categorías con poco volumen, análisis cruzado de morosidad).
