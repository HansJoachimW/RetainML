from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("../models/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to RetainML Churn API"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"churn_prediction": int(pred[0])}
