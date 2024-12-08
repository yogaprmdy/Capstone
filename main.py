from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras.saving import register_keras_serializable
import joblib
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Folder tempat model dan scaler disimpan
MODEL_DIR = "models"
SCALER_DIR = "scalers"

# Daftar ticker saham
TICKERS = [
    "ACES", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ARTO", "ASII",
    "BBCA", "BBNI", "BBRI", "BBTN", "BMRI", "BRIS", "BRPT", "BUKA",
    "CPIN", "EMTK", "ESSA", "EXCL", "GGRM", "GOTO", "HRUM", "ICBP",
    "INCO", "INDF", "INDY", "INKP", "INTP", "ISAT"
]

# Load model dan scaler untuk emas
try:
    gold_model = load_model(os.path.join(MODEL_DIR, "EMAS_model.h5"))
    with open(os.path.join(SCALER_DIR, "EMAS_scaler.pkl"), "rb") as f:
        gold_scaler = pickle.load(f)
except Exception as e:
    logger.error(f"Failed to load gold model or scaler: {e}")
    gold_model = None
    gold_scaler = None

# Registrasi ulang fungsi mse untuk model saham
@register_keras_serializable()
def mse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

# Fungsi utilitas

def preprocess_gold_data(data):
    return gold_scaler.transform(data)

def predict_gold_price(input_data):
    prediction = gold_model.predict(input_data)
    return gold_scaler.inverse_transform(prediction)

def load_scaler(ticker):
    scaler_path = os.path.join(SCALER_DIR, f"{ticker}_features_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    return joblib.load(scaler_path)

def get_prediction_dates():
    today = datetime.today()
    return {
        "1_day_prediction_date": (today + timedelta(days=1)).strftime('%Y-%m-%d'),
        "1_month_prediction_date": (today + timedelta(days=30)).strftime('%Y-%m-%d'),
        "1_year_prediction_date": (today + timedelta(days=365)).strftime('%Y-%m-%d')
    }

def predict_stock_price(ticker):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    scaler = load_scaler(ticker)
    model = load_model(model_path, custom_objects={"mse": mse})

    input_shape = model.input_shape[1:]
    dummy_features = np.random.rand(1, *input_shape)
    prediction = model.predict(dummy_features)
    return float(prediction[0][0])

# Endpoint untuk prediksi harga emas
class GoldPredictionInput(BaseModel):
    gold_price: float

@app.post("/predict/gold")
async def predict_gold(input: GoldPredictionInput):
    try:
        input_data = np.array([[input.gold_price]])
        scaled_input = preprocess_gold_data(input_data)
        predicted_price = predict_gold_price(scaled_input)
        return {
            "input_price": input.gold_price,
            "predicted_price": float(predicted_price[0][0])
        }
    except Exception as e:
        logger.error(f"Gold prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Endpoint untuk prediksi harga saham
@app.get("/predict/stock/{ticker}")
async def predict_stock(ticker: str):
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker: {ticker}")

    prediction_dates = get_prediction_dates()
    predictions = {}

    for period, date in prediction_dates.items():
        try:
            predicted_price = predict_stock_price(ticker)
            predictions[period] = {
                "predicted_price": predicted_price,
                "prediction_date": date
            }
        except Exception as e:
            logger.error(f"Stock prediction error for {ticker} on {period}: {e}")
            predictions[period] = {"error": f"Prediction failed for {period}"}

    return predictions

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Gold and Stock Price Prediction API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))  
    uvicorn.run(app, host="0.0.0.0", port=port)
