from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Model input
class CarInput(BaseModel):
    kilometers: float
    model: str
    year: int

# Dummy funtion to estimate the value of a car
def estimate_value(kilometers: float, model: str, year: int) -> float:
    coef_km = -0.05
    coef_year = -1000
    valor_base = 30000
    noise = np.random.normal(0, 2000)
    valor = valor_base + coef_km * kilometers + coef_year * (2025 - year) + noise
    return max(valor, 0) 

@app.post("/predict")
async def predict_value(car: CarInput):
    print("Recibiendo datos:", car)
    estimated_value = estimate_value(car.kilometers, car.model, car.year)
    return {"valor_estimado": round(estimated_value, 2)}
