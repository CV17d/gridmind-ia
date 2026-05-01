import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def predict_future_consumption(df):
    """
    Versión Robusta: Calcula el promedio actual y lo proyecta a 30 días.
    Ideal para cuando hay pocos datos o el consumo es constante.
    """
    if df.empty:
        return 0.0

    # Calculamos cuántas lecturas de 10 segundos hay en 30 días
    readings_in_30_days = (30 * 24 * 60 * 60) / 10
    
    # Promedio de consumo por lectura (en kWh)
    avg_consumption_per_reading = df['consumption'].mean()
    
    # Proyección simple
    prediction = avg_consumption_per_reading * readings_in_30_days
    
    return max(0, float(prediction))
