# -*- coding: utf-8 -*-
"""
ft_engineering.py - Feature Engineering Mejorado

Basado en las mejores prácticas de sklearn Pipeline y ColumnTransformer
Combina la modularidad del código original con la robustez del código de referencia
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuración de columnas (basado en el código de referencia)
TARGET = "Pago_atiempo"

# Columnas numéricas principales
NUMERIC_COLUMNS = [
    "capital_prestado",
    "edad_cliente", 
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "puntaje",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito"
]

# Columnas categóricas
CATEGORICAL_COLUMNS = ["tipo_laboral", "tendencia_ingresos"]
ORDINAL_COLUMNS = ["tipo_credito"]

# Columnas a excluir (verificación segura con logs)
import os
script_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
file_path = os.path.join(root_dir, 'data', 'processed', 'data_cleaned.csv')

print(f"DEBUG: Script directory: {script_dir}")
print(f"DEBUG: Root directory: {root_dir}")
print(f"DEBUG: Intentando leer: {file_path}")
print(f"DEBUG: ¿Existe el archivo?: {os.path.exists(file_path)}")

if os.path.exists(file_path):
    print(f"DEBUG: Columnas en el archivo: {list(pd.read_csv(file_path).columns)}")

try:
    EXCLUDE_COLUMNS = ["fecha_prestamo"] if "fecha_prestamo" in pd.read_csv(file_path).columns else []
    print(f"DEBUG: EXCLUDE_COLUMNS establecido como: {EXCLUDE_COLUMNS}")
except FileNotFoundError:
    EXCLUDE_COLUMNS = []
    print(f"❌ Error: No se encuentra '{file_path}'")
    print("⚠️  Archivo data_cleaned.csv no encontrado. Se ejecutará sin excluir columnas.")

def create_features(df):
    """
    Crea características adicionales (manteniendo tu lógica original)
    """
    df_features = df.copy()
    
    # 1. Ratios financieros
    df_features['ratio_deuda_ingresos'] = (
        df_features['total_otros_prestamos'] / df_features['salario_cliente']
    ).replace([np.inf, -np.inf], 0)
    
    df_features['ratio_cuota_ingresos'] = (
        df_features['cuota_pactada'] / df_features['salario_cliente']
    ).replace([np.inf, -np.inf], 0)
    
    df_features['ratio_saldo_capital'] = (
        df_features['saldo_total'] / df_features['capital_prestado']
    ).replace([np.inf, -np.inf], 0)
    
    # 2. Indicadores de riesgo
    df_features['tiene_mora'] = (df_features['saldo_mora'] > 0).astype(int)
    df_features['multiples_prestamos'] = (df_features['cant_creditosvigentes'] > 3).astype(int)
    df_features['alta_consulta'] = (df_features['huella_consulta'] > 5).astype(int)
    
    # 3. Características de sector
    df_features['total_creditos_formales'] = (
        df_features['creditos_sectorFinanciero'] + 
        df_features['creditos_sectorCooperativo']
    )
    
    total_creditos = df_features['cant_creditosvigentes']
    df_features['prop_creditos_formales'] = (
        df_features['total_creditos_formales'] / total_creditos
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 4. Diferencia de puntajes
    df_features['diff_puntajes'] = (
        df_features['puntaje'] - df_features['puntaje_datacredito']
    ).fillna(0)
    
    # Agregar nuevas columnas numéricas a la lista
    new_numeric_cols = [
        'ratio_deuda_ingresos', 'ratio_cuota_ingresos', 'ratio_saldo_capital',
        'tiene_mora', 'multiples_prestamos', 'alta_consulta',
        'total_creditos_formales', 'prop_creditos_formales', 'diff_puntajes'
    ]
    
    return df_features, new_numeric_cols

def prepare_data_robust(df):
    """
    Prepara datos usando Pipeline robusto de sklearn
    (Basado en el código de referencia pero mejorado)
    """
    print("🔄 Preparando datos con Pipeline robusto...")
    
    # 1. Crear características adicionales
    df_features, new_numeric_cols = create_features(df)
    
    # 2. Convertir categóricas a string
    for col in CATEGORICAL_COLUMNS + ORDINAL_COLUMNS:
        if col in df_features.columns:
            df_features[col] = df_features[col].astype(str)
    
    # 3. Separar X e y
    X = df_features.drop(columns=[TARGET] + EXCLUDE_COLUMNS)
    y = df_features[TARGET]
    
    # 4. Actualizar lista de columnas numéricas
    all_numeric = NUMERIC_COLUMNS + new_numeric_cols
    # Filtrar solo las que existen en el DataFrame
    numeric_cols = [col for col in all_numeric if col in X.columns]
    
    # 5. Crear pipelines
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # 6. Crear preprocessor
    all_categorical = CATEGORICAL_COLUMNS + ORDINAL_COLUMNS
    # Filtrar solo las que existen
    categorical_cols = [col for col in all_categorical if col in X.columns]
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder='drop')  # Descartar columnas no especificadas
    
    # 7. Transformar datos
    X_transformed = preprocessor.fit_transform(X)
    
    print(f"✅ Datos preparados: {X_transformed.shape}")
    print(f"• Columnas numéricas: {len(numeric_cols)}")
    print(f"• Columnas categóricas: {len(categorical_cols)}")
    print(f"• Features finales: {X_transformed.shape[1]}")
    
    return X_transformed, y, preprocessor

def split_data_robust(df, test_size=0.2, random_state=42):
    """
    División de datos robusta usando el pipeline mejorado
    """
    print(f"\n📊 Dividiendo datos (test_size={test_size})...")
    
    # Preparar datos
    X, y, preprocessor = prepare_data_robust(df)
    
    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"• Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"• Test:  {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Distribución de clases
    print(f"\n🎯 Distribución de {TARGET}:")
    print(f"• Train: Clase 0={y_train.value_counts().get(0, 0)}, Clase 1={y_train.value_counts().get(1, 0)}")
    print(f"• Test:  Clase 0={y_test.value_counts().get(0, 0)}, Clase 1={y_test.value_counts().get(1, 0)}")
    
    return X_train, X_test, y_train, y_test, preprocessor

def load_and_prepare_data(file_path=None):
    """
    Función principal que carga y prepara todo el pipeline
    """
    print("🚀 Iniciando pipeline mejorado de Feature Engineering")
    print("="*60)
    
    # Usar la ruta absoluta calculada si no se especifica archivo
    if file_path is None:
        file_path = os.path.join(root_dir, 'data', 'processed', 'data_cleaned.csv')
        print(f"DEBUG: Usando ruta por defecto: {file_path}")
    
    # Cargar datos
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Dataset cargado: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: No se encuentra '{file_path}'")
        return None
    
    # División y preparación
    result = split_data_robust(df)
    
    if result:
        X_train, X_test, y_train, y_test, preprocessor = result
        
        print(f"\n✅ Pipeline completado exitosamente!")
        print(f"📈 Características finales: {X_train.shape[1]}")
        
        return {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'df_original': df
        }
    
    return None

# Función de evaluación compatible
def summarize_classification_robust(model_name, y_true, y_pred, y_pred_proba=None):
    """
    Versión robusta de evaluación (maneja casos donde no hay probabilidades)
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"\n{'='*50}")
    print(f"📊 EVALUACIÓN: {model_name}")
    print('='*50)
    
    # Métricas básicas
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1-Score': f1_score(y_true, y_pred, average='binary'),
    }
    
    # ROC-AUC si hay probabilidades
    if y_pred_proba is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['ROC-AUC'] = 0.0
    
    # Mostrar métricas
    print("\n📈 Métricas:")
    for metric, value in metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    # Reporte de clasificación
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=['No Paga', 'Paga']))
    
    return metrics

if __name__ == "__main__":
    # Ejecutar pipeline completo
    pipeline_result = load_and_prepare_data()
    
    if pipeline_result:
        print(f"\n🎉 Pipeline listo para entrenar modelos!")
        print(f"✅ X_train shape: {pipeline_result['X_train'].shape}")
        print(f"✅ X_test shape: {pipeline_result['X_test'].shape}")
        
        # Guardar datos procesados y preprocesador
        import pickle
        
        # Crear directorio para guardar artefactos si no existe
        artifacts_dir = os.path.join(root_dir, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Guardar preprocesador
        preprocessor_path = os.path.join(artifacts_dir, 'preprocesador.pkl')
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(pipeline_result['preprocessor'], f)
        print(f"✅ Preprocesador guardado en: {preprocessor_path}")
        
        # Guardar datos procesados
        data_path = os.path.join(artifacts_dir, 'datos_procesados.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({
                'X_train': pipeline_result['X_train'],
                'X_test': pipeline_result['X_test'], 
                'y_train': pipeline_result['y_train'],
                'y_test': pipeline_result['y_test'],
                'feature_names': pipeline_result['preprocessor'].get_feature_names_out()
            }, f)
        print(f"✅ Datos procesados guardados en: {data_path}")
        
        print(f"\n📦 Artefactos guardados exitosamente en '{artifacts_dir}'")
    else:
        print("❌ Error en el pipeline")
