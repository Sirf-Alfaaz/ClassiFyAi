from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load upgraded XGBoost model and LabelEncoder
model = joblib.load("xgboost_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define product schema
class Product(BaseModel):
    name: str
    description: str = ""
    price: float

@app.post("/predict")
def predict_category(product: Product):
    # Combine name and description
    combined_text = product.name + " " + product.description

    # Prepare input DataFrame
    df = pd.DataFrame([{
        "text": combined_text,
        "price": product.price
    }])

    # Predict category
    prediction_encoded = model.predict(df)[0]
    prediction_label = str(label_encoder.inverse_transform([prediction_encoded])[0])

    # Confidence
    try:
        probabilities = model.predict_proba(df)[0]
        confidence = float(max(probabilities)) * 100
    except:
        confidence = 75.0

    # Suspicious product logic
    is_suspicious = bool(product.price <= 0 or product.price > 100000 or confidence < 60)

    return {
        "category": prediction_label,
        "confidence": round(confidence, 2),
        "isSuspicious": is_suspicious
    }
