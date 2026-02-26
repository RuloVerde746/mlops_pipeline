# -*- coding: utf-8 -*-
"""
model_training_evaluation.py - Entrenamiento y Evaluación de Modelos

Basado en el notebook de referencia pero adaptado para usar ft_engineering.py mejorado
Implementa el flujo completo: carga de datos → entrenamiento → evaluación → guardado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Importar nuestro módulo de feature engineering mejorado
from ft_engineering import load_and_prepare_data, summarize_classification_robust

# Importar modelos de sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

# Configuración de visualizaciones
plt.style.use('default')
sns.set_style("whitegrid")

def define_models():
    """
    Define los modelos a entrenar (basado en el notebook de referencia)
    
    Esta función crea un diccionario con 5 modelos de clasificación:
    - Logistic Regression: Modelo lineal básico para clasificación binaria
    - SVC: Support Vector Classifier con kernel RBF por defecto
    - Decision Tree: Árbol de decisión simple, interpretable
    - Random Forest: Ensemble de árboles, reduce overfitting
    - XGBoost: Gradient boosting, alto rendimiento en datos tabulares
    
    Todos los modelos tienen:
    - random_state=42: Para reproducibilidad
    - class_weight='balanced': Para manejar desbalance de clases
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "SVC": SVC(probability=True, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=1)
    }
    
    print("🤖 Modelos configurados:")
    for name in models.keys():
        print(f"  • {name}")
    
    return models

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa todos los modelos (basado en el notebook de referencia)
    
    Proceso para cada modelo:
    1. Entrenamiento con .fit() usando datos de entrenamiento
    2. Predicción con .predict() sobre datos de prueba
    3. Cálculo de métricas: Accuracy, Precision, Recall, F1-score
    4. Generación de matriz de confusión para análisis de errores
    5. Visualización inmediata de resultados
    
    Maneja excepciones para continuar con otros modelos si alguno falla
    """
    print("\n🚀 Iniciando entrenamiento y evaluación de modelos")
    print("="*60)
    
    resultados = []
    
    for nombre, modelo in models.items():
        print(f"\n📊 Entrenando modelo: {nombre}...")
        
        try:
            # Entrenar modelo con los datos preprocesados
            # X_train: características escaladas y codificadas
            # y_train: variable objetivo (0/1 para Pago_atiempo)
            modelo.fit(X_train, y_train)
            
            # Realizar predicciones sobre el conjunto de prueba
            # X_test: datos nunca vistos por el modelo
            y_pred = modelo.predict(X_test)
            
            # Calcular métricas de evaluación
            # accuracy: proporción de predicciones correctas
            # precision: proporción de positivos predichos que son correctos
            # recall: proporción de positivos reales que fueron detectados
            # f1: media armónica de precision y recall
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Guardar resultados en lista para comparación posterior
            resultados.append({
                "Modelo": nombre,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1
            })
            
            print(f"✅ {nombre} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            # Crear matriz de confusión para análisis detallado
            # Muestra: TP, FP, FN, TN para entender errores del modelo
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"Matriz de confusión - {nombre}")
            plt.xlabel("Predicted")  # Predicciones del modelo
            plt.ylabel("Actual")      # Valores reales
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error entrenando {nombre}: {e}")
            continue
    
    return resultados, models

def compare_models(resultados):
    """
    Compara los resultados de los modelos (basado en el notebook de referencia)
    
    Esta función:
    1. Convierte la lista de resultados a DataFrame de pandas
    2. Muestra tabla comparativa con todas las métricas
    3. Transforma datos a formato largo para visualización
    4. Crea gráfico de barras comparativo por modelo y métrica
    
    El gráfico permite identificar visualmente:
    - Qué modelo tiene mejor accuracy
    - Cuál tiene mejor precision (menos falsos positivos)
    - Cuál tiene mejor recall (menos falsos negativos)
    - Cuál tiene mejor F1-score (balance precision-recall)
    """
    print(f"\n📈 TABLA COMPARATIVA DE MODELOS")
    print("="*60)
    
    # Convertir lista de diccionarios a DataFrame para mejor visualización
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.round(4))
    
    # Preparar datos para gráfico comparativo
    # melt() transforma de formato ancho a largo para seaborn
    # Ej: [Modelo, Accuracy, Precision, Recall, F1] → [Modelo, Métrica, Valor]
    print(f"\n📊 Generando gráfico comparativo...")
    
    df_resultados_melted = df_resultados.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
    
    # Crear gráfico de barras comparativo
    plt.figure(figsize=(12,6))
    sns.barplot(x="Modelo", y="Valor", hue="Métrica", data=df_resultados_melted)
    plt.title("Comparación de métricas por modelo", fontsize=14, fontweight='bold')
    plt.ylim(0,1)  # Las métricas están entre 0 y 1
    plt.xticks(rotation=45)  # Rotar etiquetas para mejor legibilidad
    plt.legend(loc="lower right")  # Ubicar leyenda donde no tape datos
    plt.grid(True, alpha=0.3)  # Cuadrícula sutil para facilitar lectura
    plt.tight_layout()  # Ajustar para que no se corten etiquetas
    plt.show()
    
    return df_resultados

def select_best_model(df_resultados, models):
    """
    Selecciona el mejor modelo basado en F1-score
    
    El F1-score es elegido porque:
    - Es la media armónica de precision y recall
    - Penaliza modelos con performance desbalanceada
    - Es ideal para problemas con clases desbalanceadas como el nuestro
    
    Proceso:
    1. Encontrar índice del máximo F1-score
    2. Obtener nombre y métricas del mejor modelo
    3. Recuperar objeto del modelo entrenado
    4. Mostrar resumen detallado del ganador
    """
    print(f"\n🏆 SELECCIÓN DEL MEJOR MODELO")
    print("="*40)
    
    # Encontrar el mejor modelo por F1-score
    # idxmax() devuelve el índice del valor máximo en la columna
    best_idx = df_resultados['F1-score'].idxmax()
    best_model_name = df_resultados.loc[best_idx, 'Modelo']
    best_f1 = df_resultados.loc[best_idx, 'F1-score']
    
    print(f"🥇 Mejor modelo: {best_model_name}")
    print(f"📈 F1-score: {best_f1:.4f}")
    
    # Mostrar todas las métricas del mejor modelo
    best_metrics = df_resultados.loc[best_idx]
    print(f"\n📊 Métricas completas:")
    for metric, value in best_metrics.items():
        if metric != 'Modelo':
            print(f"  • {metric}: {value:.4f}")
    
    # Obtener el objeto del mejor modelo del diccionario original
    best_model = models.get(best_model_name)
    
    return best_model, best_model_name, best_metrics

def save_model_and_artifacts(best_model, best_model_name, preprocessor, X_train):
    """
    Guarda el mejor modelo y artefactos necesarios (basado en el notebook de referencia)
    
    Archivos generados:
    1. Modelo entrenado (.pkl) - Para predicciones en producción
    2. Preprocesador (.pkl) - Para transformar nuevos datos
    3. Datos referencia (.csv) - Base para monitoreo PSI
    4. Metadatos (.pkl) - Información del modelo para trazabilidad
    
    joblib se usa porque:
    - Es más eficiente que pickle para objetos de sklearn
    - Mantiene compatibilidad entre versiones
    - Permite guardar objetos complejos como pipelines
    """
    import os
    
    print(f"\n💾 GUARDANDO MODELO Y ARTEFACTOS")
    print("="*45)
    
    # Obtener directorio raíz del proyecto
    script_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Crear directorio data/processed si no existe
    processed_dir = os.path.join(root_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Guardar el mejor modelo con nombre dinámico
    # El nombre incluye el tipo de modelo para identificación clara
    model_filename = f'mejor_modelo_{best_model_name.lower().replace(" ", "_")}.pkl'
    model_path = os.path.join(root_dir, model_filename)
    joblib.dump(best_model, model_path)
    print(f"✅ Modelo guardado como: {model_filename}")
    
    # Guardar el preprocesador (pipeline de transformación)
    # Esencial para aplicar las mismas transformaciones a datos nuevos
    preprocessor_path = os.path.join(root_dir, 'preprocesador.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print("✅ Preprocesador guardado como: preprocesador.pkl")
    
    # Guardar X_train para referencia PSI (Population Stability Index)
    # PSI mide si la distribución de datos cambia en producción
    # Los datos de entrenamiento son la línea base
    X_train_df = pd.DataFrame(X_train)
    reference_data_path = os.path.join(processed_dir, 'data_referencia.csv')
    X_train_df.to_csv(reference_data_path, index=False)
    print(f"✅ Datos de referencia guardados como: data/processed/data_referencia.csv")
    
    # Crear archivo de metadatos para trazabilidad completa
    # Incluye información importante para MLOps y auditoría
    metadata = {
        'best_model': best_model_name,
        'model_file': model_filename,
        'preprocessor_file': 'preprocesador.pkl',
        'reference_data': 'data/processed/data_referencia.csv',
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'Predicción de pago a tiempo de créditos',
        'target_variable': 'Pago_atiempo'
    }
    
    metadata_path = os.path.join(root_dir, 'model_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print("✅ Metadatos guardados como: model_metadata.pkl")

def main():
    """
    Función principal que ejecuta el flujo completo de MLOps
    
    Este flujo implementa las mejores prácticas para desarrollo de modelos:
    1. Carga y preparación de datos con feature engineering robusto
    2. Definición de múltiples modelos para comparación
    3. Entrenamiento y evaluación sistemática
    4. Análisis comparativo visual y numérico
    5. Selección automática del mejor modelo
    6. Guardado de artefactos para producción
    
    Cada fase está diseñada para ser:
    - Reproducible (mismos random_state)
    - Escalable (funciona con diferentes datasets)
    - Auditable (logs detallados y metadatos)
    - Robusta (manejo de excepciones)
    """
    print("🎯 INICIANDO ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
    print("="*65)
    
    # 1. Cargar y preparar datos usando ft_engineering.py mejorado
    # Esta función aplica: feature engineering, encoding, scaling, train-test split
    print("\n📊 FASE 1: CARGA Y PREPARACIÓN DE DATOS")
    print("-"*50)
    
    pipeline_result = load_and_prepare_data()
    
    if pipeline_result is None:
        print("❌ Error: No se pudieron cargar los datos")
        return
    
    # Extraer componentes del pipeline
    X_train = pipeline_result['X_train']      # Características para entrenamiento
    X_test = pipeline_result['X_test']        # Características para prueba
    y_train = pipeline_result['y_train']      # Variable objetivo entrenamiento
    y_test = pipeline_result['y_test']        # Variable objetivo prueba
    preprocessor = pipeline_result['preprocessor']  # Pipeline de transformación
    
    print(f"✅ Datos cargados: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # 2. Definir modelos para competición
    # Cada modelo tiene diferentes fortalezas para encontrar el mejor
    print("\n🤖 FASE 2: DEFINICIÓN DE MODELOS")
    print("-"*50)
    
    models = define_models()
    
    # 3. Entrenar y evaluar todos los modelos
    # Esta es la fase principal de machine learning
    print("\n🚀 FASE 3: ENTRENAMIENTO Y EVALUACIÓN")
    print("-"*50)
    
    resultados, trained_models = train_and_evaluate_models(
        models, X_train, X_test, y_train, y_test
    )
    
    # 4. Comparar resultados visualmente
    # Permite identificar patrones y seleccionar el mejor modelo
    print("\n📈 FASE 4: COMPARACIÓN DE MODELOS")
    print("-"*50)
    
    df_resultados = compare_models(resultados)
    
    # 5. Seleccionar automáticamente el mejor modelo
    # Basado en F1-score para balancear precision y recall
    print("\n🏆 FASE 5: SELECCIÓN DEL MEJOR MODELO")
    print("-"*50)
    
    best_model, best_model_name, best_metrics = select_best_model(
        df_resultados, trained_models
    )
    
    # 6. Guardar modelo y artefactos para producción
    # Essential para MLOps y deployment
    print("\n💾 FASE 6: GUARDADO DE ARTEFACTOS")
    print("-"*50)
    
    save_model_and_artifacts(best_model, best_model_name, preprocessor, X_train)
    
    # 7. Resumen final del proceso completo
    # Proporciona visibilidad del éxito y archivos generados
    print(f"\n🎉 PROCESO COMPLETADO EXITOSAMENTE!")
    print("="*50)
    print(f"📊 Resumen final:")
    print(f"  • Modelos evaluados: {len(resultados)}")
    print(f"  • Mejor modelo: {best_model_name}")
    print(f"  • F1-score: {best_metrics['F1-score']:.4f}")
    print(f"  • Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"  • Archivos generados:")
    print(f"    - Modelo: mejor_modelo_{best_model_name.lower().replace(' ', '_')}.pkl")
    print(f"    - Preprocesador: preprocesador.pkl")
    print(f"    - Referencia: data_referencia.csv")
    print(f"    - Metadatos: model_metadata.pkl")
    
    # Retornar diccionario con todos los resultados para uso posterior
    return {
        'results': df_resultados,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_metrics': best_metrics,
        'pipeline': pipeline_result
    }

if __name__ == "__main__":
    results = main()
