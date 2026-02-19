from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from .feature_extractor import extract_features
import validators

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent.parent / "ModelBuilding" / "best_phishing_model.pkl"
model = joblib.load(MODEL_PATH)


class URLInput(BaseModel):
    url: str


@app.post("/predict")
def predict_phishing(data: URLInput):
    url = data.url.strip()

    if not validators.url(url):
        raise HTTPException(status_code=400, detail="Invalid URL format")

    features = extract_features(url)

    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]

    result = "Yes" if pred == 0 else "No"  # 0 = phishing, 1 = legitimate

    return {"url": url, "phishing": result}
