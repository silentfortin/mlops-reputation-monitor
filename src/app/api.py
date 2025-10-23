from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from transformers import pipeline
import mlflow
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse
from fastapi import UploadFile, File

from src.test_data import preprocess_text_series
from src.inference_data import classifier

app = FastAPI()

# Prometheus counters
REQUEST_COUNT = Counter("app_requests_total", "Total requests")
PRED_POS = Counter("pred_positive_total", "Positive predictions")
PRED_NEG = Counter("pred_negative_total", "Negative predictions")
PRED_NEU = Counter("pred_neutral_total", "Neutral predictions")

class TextIn(BaseModel):
    text: str

@app.post("/predict")
async def predict(payload: TextIn):
    REQUEST_COUNT.inc()
    
    processed_text = preprocess_text_series(pd.Series([payload.text]))[0]

    results = classifier(processed_text)
    label = results[0]['label']
    score = results[0]['score']

    # Increment prometheus counters
    if label.lower().startswith("positive"):
        PRED_POS.inc()
    elif label.lower().startswith("negative"):
        PRED_NEG.inc()
    else:
        PRED_NEU.inc()

    # MLflow log metrics
    mlflow.log_metric("pred_score", score)

    return {"label": label, "score": score}

@app.post("/predict_csv")
async def predict_batch(col_name, file: UploadFile = File(...)):
    REQUEST_COUNT.inc()

    df = pd.read_csv(file.file)



@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
