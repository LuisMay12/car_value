from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model_optimized import main_data

app = FastAPI()

# Model input
class CarInput(BaseModel):
    kilometers: float
    year: int

# Dummy funtion to estimate the value of a car
def estimate_value(kilometers: float, year: int) -> float:
    coef_km = -0.05
    coef_year = -1000
    valor_base = 30000
    noise = np.random.normal(0, 2000)
    valor = valor_base + coef_km * kilometers + coef_year * (2025 - year) + noise
    return max(valor, 0) 

# Based in the model_optimized.py we have
w_final, b_final,  mean_X, std_X = main_data()

# Real function to estimate the value of a car
def predict_model_value(kilometers: float, year: int) -> float:
    X_input = np.array([[kilometers, 2025 - year]])
    X_input_normalized = (X_input - mean_X) / std_X
    predicted_value = np.dot(X_input_normalized, w_final) + b_final
    return max(predicted_value[0], 0)

@app.post("/predict")
async def predict_value(car: CarInput):
    estimated_value = predict_model_value(car.kilometers, car.year)
    return {"Estimated value": round(estimated_value, 2)}
