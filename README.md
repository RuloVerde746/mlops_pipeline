# 🤖 MLOps Pipeline - Predicción de Créditos

**Versión actual: 1.3.0**

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

## 📊 Avances (Versión actual 1.1.0)
- [x] Configuración del entorno de desarrollo (venv, requirements).
- [x] Construcción de la función base de carga de datos sin conexión forzada.
- [x] EDA visual completado (Tratamiento de nulos cruzando variables, filtrado de categorías con poco volumen, análisis cruzado de morosidad).
- [x] **Feature Engineering robusto** (`ft_engineering.py`) - Pipeline completo con sklearn ColumnTransformer
- [x] **Model Training & Evaluation** (`model_training_evaluation.py`) - Entrenamiento y evaluación de 5 modelos con selección automática

## 🤖 Feature Engineering (`src/ft_engineering.py`)

### 🎯 Propósito
Implementa un pipeline robusto de feature engineering para la predicción de pago a tiempo de créditos.

### 🔧 Funcionalidades Principales
- **Feature Creation**: Ratios financieros, indicadores de riesgo, características de sector
- **Data Preprocessing**: Imputación automática, encoding categórico, escalado numérico
- **Pipeline Robusto**: ColumnTransformer + SimpleImputer para manejo de NaNs
- **Train-Test Split**: División estratificada con random_state=42

### 📈 Características Generadas
- **Ratios**: deuda/ingresos, cuota/ingresos, saldo/capital
- **Indicadores**: tiene_mora, múltiples_préstamos, alta_consulta
- **Sectoriales**: total_créditos_formales, prop_créditos_formales
- **Diferenciales**: diff_puntajes (puntaje - puntaje_datacredito)

### ✅ Resultados
- **Dataset procesado**: 10,760 muestras → 36 características finales
- **Distribución**: 95.5% clase 1 (paga), 4.5% clase 0 (no paga)
- **Sin NaNs**: Pipeline robusto garantiza datos limpios

## 🏆 Model Training & Evaluation (`src/model_training_evaluation.py`)

### 🎯 Propósito
Entrena, evalúa y selecciona automáticamente el mejor modelo de clasificación para predicción de pago a tiempo.

### 🤖 Modelos Evaluados
| Modelo | Tipo | Características |
|--------|------|----------------|
| **Logistic Regression** | Lineal | Rápido, interpretable |
| **SVC** | Kernel RBF | Bueno para datos complejos |
| **Decision Tree** | Árbol | Muy interpretable |
| **Random Forest** | Ensemble | Reduce overfitting |
| **XGBoost** | Gradient Boosting | Alto rendimiento |

### 📊 Métricas de Evaluación
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Minimiza falsos positivos
- **Recall**: Minimiza falsos negativos  
- **F1-Score**: Balance precision-recall (criterio de selección)

### 🏅 Resultados Obtenidos
| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|---------|----------|
| **Decision Tree**  | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9991 | 1.0000 | 0.9990 | 0.9995 |
| SVC | 0.9986 | 0.9995 | 0.9990 | 0.9993 |

### 💾 Artefactos Generados
- **Modelo**: `mejor_modelo_decision_tree.pkl` - Modelo ganador
- **Preprocesador**: `preprocesador.pkl` - Pipeline de transformación
- **Referencia**: `data_referencia.csv` - Datos base para monitoreo PSI
- **Metadatos**: `model_metadata.pkl` - Información completa del modelo

### 🚀 Cómo Usar
```bash
# Entrenar y evaluar todos los modelos
python src/model_training_evaluation.py
```

El código seleccionará automáticamente el mejor modelo (Decision Tree) y guardará todos los artefactos necesarios para producción.

---

## 📊 Avances (Versión actual 1.2.0)

### 🔍 AVANCE 3: Model Monitoring y Data Drift Detection
- [x] **Sistema de monitoreo completo** (`src/model_monitoring.py`)
- [x] **Detección de data drift** con múltiples métricas estadísticas:
  - **PSI** (Population Stability Index) - Detecta cambios en distribución poblacional
  - **KS** (Kolmogorov-Smirnov) - Compara distribuciones acumuladas  
  - **Jensen-Shannon** - Mide divergencia entre distribuciones
  - **Chi-cuadrado** - Para variables categóricas
- [x] **Sistema de alertas automático** con 3 niveles:
  - 🔴 **CRITICAL**: 2+ métricas críticas o 1 crítica + 2 advertencias
  - 🟡 **WARNING**: 1 crítica o 2+ advertencias
  - 🟢 **NORMAL**: Métricas dentro de umbrales normales
- [x] **Reportes HTML interactivos** con visualizaciones y tablas de métricas
- [x] **Gráficos comparativos** para variables con alertas (histogramas, box plots, Q-Q plots)
- [x] **Datos para dashboard Streamlit** (`assets/streamlit_dashboard_data.pkl`)
- [x] **Estructura de archivos organizada**:
  - `assets/` - Reportes y datos generados
  - `assets/images/` - Gráficos de monitoreo
  - `assets/drift_report.html` - Reporte principal
  - `assets/drift_report.json` - Datos en formato JSON

### 🏆 Resultados del Monitoreo
| Métrica | Resultado |
|----------|-----------|
| **Variables analizadas** | 37 |
| **Alertas críticas** | 4 (variables: 1, 4, 24, prediction) |
| **Alertas de advertencia** | 17 variables |
| **Variables normales** | 16 variables |

### 🚀 Cómo Usar el Monitoreo
```bash
# Ejecutar monitoreo completo
python src/model_monitoring.py
```

### 📁 Archivos Generados por el Monitoreo
- **`assets/drift_report.html`** - Reporte HTML completo
- **`assets/drift_report.json`** - Datos en formato JSON
- **`assets/images/drift_plot_*.png`** - Gráficos de variables con alertas
- **`assets/streamlit_dashboard_data.pkl`** - Datos para dashboard Streamlit

---

## 🚀 Avance 4: Model Deployment (API)

### 🎯 Objetivos Logrados
- [x] **Disponibilización del modelo mediante una API**: Implementación de un servicio REST para predicciones en tiempo real y por lotes.
- [x] **Creación de imagen Docker**: Preparación del entorno contenedorizado con todas las librerías y el código necesario para la aplicación.

### 🛠 Despliegue del Modelo (`src/model_deploy.py`)

Este script representa el núcleo del despliegue productivo, utilizando **FastAPI** para exponer el modelo como un servicio robusto y escalable.

#### 🔧 Funcionalidades y Responsabilidades
- **Carga de Modelos**: Carga automática del mejor modelo (`mejor_modelo_decision_tree.pkl`) y su preprocesador (`preprocesador.pkl`).
- **Lógica de Predicción**: Implementa la transformación de datos de entrada asegurando consistencia con el entrenamiento.
- **Endpoints REST**:
    - `POST /predict`: Permite enviar múltiples registros para predicción por lotes (batch).
    - `POST /predict_single`: Optimizado para predicciones individuales rápidas.
    - `GET /model_info`: Proporciona metadatos sobre la versión y tipo de modelo cargado.
    - `GET /health`: Verifica el estado de salud del servicio y la carga de artefactos.
- **Soporte Pydantic**: Validación estricta de datos de entrada mediante esquemas definidos.

#### 🚀 Cómo Ejecutar la API
```bash
# Iniciar el servidor Uvicorn
python src/model_deploy.py
```
La documentación interactiva estará disponible automáticamente en `http://localhost:8000/docs`.

---

## 📈 Avance 5: Visualización y Dashboard Interactivo

### 🎯 Objetivos Logrados
- [x] **Dashboard de Monitoreo con Streamlit**: Interfaz gráfica para visualizar la salud del modelo en tiempo real.
- [x] **Integración de Logs Persistentes**: Sistema de auditoría que permite ver el estado del pipeline desde la terminal de Docker o PowerShell.
- [x] **Análisis de Drift Visual**: Pestañas dedicadas para alertas críticas, incluyendo histogramas y gráficos de estabilidad.

---

## 🏁 Guía Paso a Paso: Ejecución Completa del Proyecto

Sigue este flujo para ejecutar el sistema desde cero hasta la visualización en el dashboard.

### 1. Preparación del Entorno
Antes de empezar, asegúrate de tener instalado **Docker Desktop** y Python 3.9+.
- Crea tu entorno virtual: `python -m venv venv`
- Actívalo: `.\venv\Scripts\activate` (Windows)
- Instala dependencias: `pip install -r requirements.txt`

### 2. Procesamiento de Datos y Entrenamiento
Ejecuta los scripts en este orden para generar los artefactos del modelo:
1. **Carga de Datos**: `python src/cargar_datos.py` (Procesa el Excel inicial).
2. **Entrenamiento**: `python src/model_training_evaluation.py` (Entrena 5 modelos, selecciona el mejor y guarda `mejor_modelo_decision_tree.pkl`).

### 3. Despliegue con Docker Desktop
**Docker Desktop** es fundamental aquí porque permite "empaquetar" nuestra API (`FastAPI`) junto con todas sus dependencias en un contenedor. Esto garantiza que el modelo funcione exactamente igual en tu máquina que en un servidor de producción.
- Ejecuta: `docker-compose up --build`
- Esto levantará la API en el puerto `8000`. Puedes verificarlo en `http://localhost:8000/docs`.

### 4. Generación de Monitoreo (Data Drift)
Para simular el paso del tiempo y verificar si el modelo sigue siendo preciso, ejecutamos el sistema de monitoreo:
- Ejecuta: `python src/model_monitoring.py`
- Este script comparará los datos originales contra los nuevos, generará alertas y creará los archivos en la carpeta `assets/`.

### 5. Visualización en Streamlit
**Streamlit** es la herramienta que convierte nuestros scripts de datos en una aplicación web interactiva. No necesitas saber HTML/CSS; Streamlit interpreta el código Python para crear el dashboard.
- Ejecuta: `streamlit run src/streamlit_app.py`
- Se abrirá una ventana en tu navegador (`http://localhost:8501`) donde verás:
    - La salud general del modelo.
    - Las variables que han sufrido desviaciones (Drift).
    - Recomendaciones automáticas sobre si debes reentrenar el modelo.

---
> **Tip de Depuración**: Si activas el **"Modo Depuración"** en el sidebar de Streamlit, podrás ver logs técnicos adicionales y la estructura cruda de los datos procesados.

