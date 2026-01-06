from fastapi import APIRouter, Form
import joblib
import numpy as np
from pathlib import Path
import sys

router = APIRouter()

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"


from schemas.predict_schema import PredictRequest


models = joblib.load(MODEL_DIR / "all_sklearn_models.pkl")


@router.post("/predict")
def predict(data: PredictRequest):
    X = np.array([[
        data.time,
        data.amount,
        data.v1, data.v2, data.v3, data.v4, data.v5, data.v6, data.v7, data.v8, data.v9,
        data.v10, data.v11, data.v12, data.v13, data.v14, data.v15, data.v16, data.v17, data.v18, data.v19,
        data.v20, data.v21, data.v22, data.v23, data.v24, data.v25, data.v26, data.v27, data.v28
    ]])


    model = models["random_forest"]  # 👈 choose best
    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba > 0.5)



    return {
        "results": {
            "prediction": pred,
            "probability": proba,
            "factors": []
        }
    }
