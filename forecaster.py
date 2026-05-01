import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def predict_future_consumption(df):
    """
    IA SMART HYBRID: 
    1. Filtra datos erróneos (Outliers).
    2. Elige el mejor modelo según la cantidad de datos.
    """
    if df.empty: return 0.0

    # --- 1. FILTRADO DE DATOS ERRÓNEOS ---
    # Si un dato es 10 veces mayor al promedio, lo ignoramos (limpieza automática)
    median_val = df['consumption'].median()
    if median_val > 0:
        df = df[df['consumption'] < (median_val * 10)]

    if df.empty: return 0.0

    # --- 2. LÓGICA DE PROYECCIÓN ---
    readings_in_30_days = (30 * 24 * 60 * 60) / 10
    
    # Si tenemos menos de 20 datos o el tiempo es muy corto, usamos promedio estable
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    duration_secs = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()

    if len(df) < 20 or duration_secs < 300:
        # Modo Estabilidad
        return float(df['consumption'].mean() * readings_in_30_days)
    
    # Modo Tendencia (Regresión Lineal)
    try:
        df['secs'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        X = df[['secs']].values
        y = df['consumption'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        
        # Proyectamos al centro del mes futuro
        future_sec = duration_secs + (15 * 24 * 60 * 60)
        predicted_avg = model.predict([[future_sec]])[0]
        
        return max(0, float(predicted_avg * readings_in_30_days))
    except:
        return float(df['consumption'].mean() * readings_in_30_days)
