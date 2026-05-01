import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def predict_future_consumption(df):
    """
    Predice el consumo de los próximos 30 días basándose en la tendencia actual.
    Usa una regresión lineal simple sobre los últimos registros.
    """
    if len(df) < 3:
        # No hay suficientes datos, usamos un promedio simple proyectado
        return df['consumption'].mean() * 30 if not df.empty else 0.0

    # Convertir timestamps a segundos para mayor precisión (útil para datos nuevos)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['seconds_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    
    X = df[['seconds_since_start']].values
    y = df['consumption'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Proyectar el consumo para un periodo de 30 días (2,592,000 segundos)
    # Calculamos el consumo promedio por segundo actual y lo proyectamos
    last_sec = df['seconds_since_start'].max()
    
    # Predicción: Sumar el consumo esperado para 30 días basándose en la tendencia por segundo
    # Para simplificar, calculamos el valor en la mitad del periodo futuro (15 días) y multiplicamos
    mid_future_sec = last_sec + (15 * 24 * 60 * 60)
    predicted_val_per_reading = model.predict(np.array([[mid_future_sec]]))[0]
    
    # Estimamos cuántas "lecturas" habría en 30 días (asumiendo lectura cada 10 seg)
    readings_in_30_days = (30 * 24 * 60 * 60) / 10
    total_predicted = predicted_val_per_reading * readings_in_30_days
    
    return max(0, total_predicted) # El valor ya está en kWh
