from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi import UploadFile, File, Form
import mlflow

import pandas as pd
from pydantic import BaseModel
from transformers import pipeline

from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

from src.test_data import preprocess_text_series
from src.inference_data import classifier

app = FastAPI()

# Prometheus counters
REQUEST_COUNT = Counter("app_requests_total", "Total requests")
PRED_POS = Counter("pred_positive_total", "Positive predictions")
PRED_NEG = Counter("pred_negative_total", "Negative predictions")
PRED_NEU = Counter("pred_neutral_total", "Neutral predictions")

def infer_data(processed_text):
    results = classifier(
        processed_text,
        truncation=True,
        max_length=512
        )
    if not results:
        raise HTTPException(status_code=500, detail="Empty classification result")

    label = results[0]["label"] # pyright: ignore[reportArgumentType, reportIndexIssue]
    score = results[0]["score"] # pyright: ignore[reportArgumentType, reportIndexIssue]

    if label.lower().startswith("positive"): # pyright: ignore[reportAttributeAccessIssue]
        PRED_POS.inc()
    elif label.lower().startswith("negative"): # type: ignore
        PRED_NEG.inc()
    else:
        PRED_NEU.inc()

    mlflow.log_metric("pred_score", score) # pyright: ignore[reportArgumentType]

    return {"label": label, "score": score}

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    score: float    

# predict
@app.post("/predict", response_model=PredictionOut)
async def predict(payload: TextIn):
    REQUEST_COUNT.inc()
    processed_text = preprocess_text_series(pd.Series([payload.text]))[0]

    return infer_data(processed_text)

# predict CSV
@app.post("/predict_csv")
async def predict_batch(
    text_column: str = Form(...), 
    file: UploadFile = File(...)
):
    REQUEST_COUNT.inc()
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    if text_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found in CSV.")

    texts = df[text_column].astype(str)
    processed_texts = preprocess_text_series(texts)
    preds = classifier(
        list(processed_texts),
        truncation=True,
        max_length=512
    )

    results = []
    for original_text, pred in zip(texts, preds):
        results.append({
            "text": original_text,
            "label": pred["label"],
            "score": pred["score"]
        })

    return results

# get metrics
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
