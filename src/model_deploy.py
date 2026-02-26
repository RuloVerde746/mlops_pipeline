# -*- coding: utf-8 -*-
"""
model_deploy.py - API de Predicción de Créditos

Basado en el mejor modelo entrenado (Decision Tree) y el pipeline de feature engineering
Implementa FastAPI para disponibilizar el modelo como servicio REST
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np
from typing import List, Optional

# 1. Inicializar la App
app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description="Endpoint para predicciones de pago a tiempo de créditos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 2. Cargar el modelo y el preprocesador
# Usamos rutas relativas seguras
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mejor_modelo_decision_tree.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "..", "models", "preprocesador.pkl")

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✅ Modelo y preprocesador cargados exitosamente")
except Exception as e:
    print(f"❌ Error crítico al cargar los modelos: {e}")
    model = None
    preprocessor = None

# 3. Esquema de entrada basado en las características del dataset
class CreditData(BaseModel):
    """Esquema de datos para predicción individual"""
    capital_prestado: float
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    puntaje: float
    puntaje_datacredito: float
    Pago_atiempo: Optional[int] = None  # No necesario para predicción
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: float
    creditos_sectorCooperativo: float
    creditos_sectorReal: float
    promedio_ingresos_datacredito: Optional[float] = 0.0
    tipo_credito: str
    tendencia_ingresos: Optional[str] = "estable"
    cuota_pactada: float
    cant_creditosvigentes: int
    huella_consulta: int
    total_otros_prestamos: float

class PredictionInput(BaseModel):
    """Esquema para predicción por lotes"""
    data: List[CreditData]

class PredictionResponse(BaseModel):
    """Esquema de respuesta"""
    status: str
    predictions: List[int]
    probabilities: Optional[List[float]] = None
    total_processed: int
    successful_predictions: int

# 4. Función de preprocesamiento interno
def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Aplica el mismo feature engineering que se usó en entrenamiento
    """
    try:
        # Crear características adicionales (igual que en ft_engineering.py)
        df['ratio_deuda_ingresos'] = (
            df['total_otros_prestamos'] / df['salario_cliente']
        ).replace([np.inf, -np.inf], 0)
        
        df['ratio_cuota_ingresos'] = (
            df['cuota_pactada'] / df['salario_cliente']
        ).replace([np.inf, -np.inf], 0)
        
        df['ratio_saldo_capital'] = (
            df['saldo_total'] / df['capital_prestado']
        ).replace([np.inf, -np.inf], 0)
        
        df['tiene_mora'] = (df['saldo_mora'] > 0).astype(int)
        df['multiples_prestamos'] = (df['cant_creditosvigentes'] > 3).astype(int)
        df['alta_consulta'] = (df['huella_consulta'] > 5).astype(int)
        
        df['total_creditos_formales'] = (
            df['creditos_sectorFinanciero'] + df['creditos_sectorCooperativo']
        )
        
        total_creditos = df['cant_creditosvigentes']
        df['prop_creditos_formales'] = (
            df['total_creditos_formales'] / total_creditos
        ).replace([np.inf, -np.inf], 0).fillna(0)
        
        df['diff_puntajes'] = (
            df['puntaje'] - df['puntaje_datacredito']
        ).fillna(0)
        
        # Aplicar el preprocesador
        data_processed = preprocessor.transform(df)
        return data_processed
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        raise HTTPException(status_code=400, detail=f"Error en preprocesamiento: {str(e)}")

# 5. Endpoint principal de predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    """
    Endpoint principal para predicciones
    Acepta datos individuales o por lotes
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([item.dict() for item in input_data.data])
        
        # Aplicar preprocesamiento
        data_processed = preprocess_data(df)
        
        # Realizar predicciones
        predictions = model.predict(data_processed)
        
        # Obtener probabilidades (Decision Tree las tiene)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data_processed)[:, 1].tolist()
        
        # Contar predicciones exitosas
        successful_count = len(predictions)
        
        return PredictionResponse(
            status="success",
            predictions=predictions.tolist(),
            probabilities=probabilities,
            total_processed=len(input_data.data),
            successful_predictions=successful_count
        )
        
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# 6. Endpoint para predicción individual (más simple)
@app.post("/predict_single")
def predict_single(data: CreditData):
    """
    Endpoint simplificado para predicción individual
    """
    batch_input = PredictionInput(data=[data])
    return predict(batch_input)

# 7. Endpoint de información del modelo
@app.get("/model_info")
def model_info():
    """
    Retorna información sobre el modelo cargado
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    return {
        "model_type": "Decision Tree Classifier",
        "model_loaded": True,
        "preprocessor_loaded": preprocessor is not None,
        "features_expected": len(model.feature_importances_) if hasattr(model, 'feature_importances_') else None,
        "model_version": "1.0.0",
        "target": "Pago_atiempo (0=No paga, 1=Paga)",
        "description": "Predice si un cliente pagará su crédito a tiempo"
    }

# 8. Endpoint de salud del servicio
@app.get("/health")
def health_check():
    """
    Endpoint para verificar que el servicio está funcionando
    """
    return {
        "status": "healthy",
        "service": "Model Prediction API",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "version": "1.0.0"
    }

# 9. Endpoint principal
@app.get("/")
def home():
    """
    Endpoint de bienvenida
    """
    return {
        "message": "API de Predicción de Riesgo Crediticio",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - Predicción por lotes",
            "predict_single": "/predict_single - Predicción individual",
            "model_info": "/model_info - Información del modelo",
            "health": "/health - Estado del servicio",
            "docs": "/docs - Documentación interactiva"
        }
    }

# 10. Para ejecutar localmente (opcional)
if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor de API...")
    print("📋 Documentación disponible en: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)