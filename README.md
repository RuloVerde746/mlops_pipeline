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
| **Decision Tree** 🥇 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
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
