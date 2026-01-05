from fastapi import APIRouter, WebSocket, Depends, HTTPException
from typing import List, Dict
import joblib
from pathlib import Path
import asyncio
import websockets
import json
import numpy as np
import sys

from sqlalchemy.orm import Session
from sqlalchemy import func, extract, desc
from datetime import datetime

from shared.db.models import FraudTransaction, ModelMetrics, Base
from shared.config.database import get_db
from dateutil.relativedelta import relativedelta

router = APIRouter()

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"


from shared.schemas.model_schema import ModelMetricsSchema

connected_clients: List[WebSocket] = []

models = joblib.load(MODEL_DIR / "all_sklearn_models.pkl")
columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

def run_prediction(tx):
    X = np.array([tx[col] for col in columns]).reshape(1, -1)
    results = {}
    for name, model in models.items():
        proba = model.predict_proba(X)[0][1]
        pred = int(proba > 0.5)
        results[name] = {"prediction": pred, "probability": float(proba)}
    return results

# WebSocket endpoint
@router.websocket("/dashboard")
async def dashboard_ws(client_ws: WebSocket):
    await client_ws.accept()
    connected_clients.append(client_ws)
    print("Client ++++++++++++++++",client_ws)
    try:
        while True:
            await client_ws.receive_text()
    except Exception:
        print("Client disconnected")
        if client_ws in connected_clients:
            connected_clients.remove(client_ws)

async def broadcast_to_dashboards(data: dict):
    disconnected = []
    for client in connected_clients:
        try:
            print("BRAODCASET ADATA+++++++++==========", data)
            await client.send_json(data)
        except:
            disconnected.append(client)
    for c in disconnected:
        connected_clients.remove(c)

@router.get("/start-stream")
async def start_stream(db: Session = Depends(get_db)):
    url = "ws://localhost:8001/stream"

    async def listen():
        while True:
            try:
                async with websockets.connect(url) as ws:
                    async for message in ws:
                        tx = json.loads(message)
                        results = run_prediction(tx)
                        processed = {"transaction": tx, "predictions": results}
                        print("PROBA: ", results)
                        new_fraud_transaction = FraudTransaction(txn_id=tx["txn_id"], 
                        	country=tx["country"], 
                        	city=tx["city"], 
                        	merchant=tx["merchant"], 
                        	card_last4=tx["card_last4"],
                            amount=f"{str(tx['amount'])}$",
                        	is_fraud=results["random_forest"]["prediction"],
                        	score=results.get("random_forest", {}).get("probability", {})
                        )

                        db.add(new_fraud_transaction)
                        db.commit()
                        db.refresh(new_fraud_transaction)

                        await broadcast_to_dashboards(processed)
            except Exception as e:
                print("Error: ", e)
                await asyncio.sleep(5)  # retry after delay

    asyncio.create_task(listen())
    return {"status": "streaming started"}



@router.get("/fraud/dashboard")
def dashboard_data(month: int = None, db: Session = Depends(get_db)):
    query = db.query(FraudTransaction)

    if month:
        query = query.filter(extract('month', FraudTransaction.created_at) == month)

    total_transaction = query.count()
    fraud_detected = query.filter(FraudTransaction.is_fraud == 1).count()
    legitimate = total_transaction - fraud_detected

    last_row_object = db.query(ModelMetrics).order_by(desc(ModelMetrics.id)).first()
    print("last_row_object", last_row_object)
    detectionAccuracyData = last_row_object.confusion_matrix if last_row_object else {}
    print("detectionAccuracyData", detectionAccuracyData)

    # Model performance
    model_perf_query = db.query(ModelMetrics).all()
    model_performance = [{"model": m.model_name, "accuracy": m.accuracy} for m in model_perf_query]
    accuracy_rate = db.query(func.avg(ModelMetrics.accuracy)).scalar() or 0

    # Recent detections
    recent_detections_query = (
        db.query(FraudTransaction)
        .order_by(desc(FraudTransaction.created_at))
        .limit(5)
        .all()
    )
    recent_detection = [
        {
            "id": t.txn_id,
            "amount": t.amount,  # replace with actual amount if needed
            "risk": "High" if t.score > 0.8 else "Medium" if t.score > 0.5 else "Low",
            "time": t.created_at.strftime("%Y-%m-%d %H:%M:%S")
        } for t in recent_detections_query
    ]

    chart_trends = []
    month_date = datetime.now()
    for i in range(5, -1, -1):
        month_iter = month_date - relativedelta(months=i)
        month_num = month_iter.month
        month_name = month_iter.strftime("%b")

        month_total = db.query(FraudTransaction).filter(
            extract('month', FraudTransaction.created_at) == month_num
        ).count()
        month_fraud = db.query(FraudTransaction).filter(
            extract('month', FraudTransaction.created_at) == month_num,
            FraudTransaction.is_fraud == 1
        ).count()
        month_legit = month_total - month_fraud

        chart_trends.append({"month": month_name, "legitimate": month_legit, "fraud": month_fraud})

    return {
        "total_transaction": total_transaction,
        "fraud_detected": fraud_detected,
        "accuracy_rate": accuracy_rate,
        "fraudulent": fraud_detected,
        "legitimate": legitimate,
        "detectionAccuracyData": detectionAccuracyData,
        "model_performance": model_performance,
        "recent_detection": recent_detection,
        "chart_trends": chart_trends
    }


@router.get('/models', response_model=List[ModelMetricsSchema])
def get_models(db: Session = Depends(get_db)):
    query = db.query(ModelMetrics).all()




    if not query:
        raise HTTPException(status_code=400, detail="Models Not Found")


    models = [
        {
            "model_name": model.model_name,
            "accuracy": model.accuracy,
            "precision": model.precision,
            "recall": model.recall,
            "f1_score": model.f1_score,
            "auc_score": model.auc_score,
            "is_production": model.is_production,
            "sample_count": model.sample_count,
            "version": model.version,
            "training_time": model.training_time,
            "created_at": model.created_at
        } for model in query
    ]

    print(models)

    return models








