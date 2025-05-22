import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
Instrumentator().instrument(app).expose(app)
# Input data model - adjust fields to your actual dataset columns
class ChurnModelInput(BaseModel):
    CreditScore: int = Field(..., example=650)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Female")
    Age: int = Field(..., example=40)
    Tenure: int = Field(..., example=5)
    Balance: float = Field(..., example=50000.0)
    NumOfProducts: int = Field(..., example=2)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=1)
    EstimatedSalary: float = Field(..., example=60000.0)

# Load model and preprocessor once
try:
    model = joblib.load("./Models/GBOOST.pkl")
    preprocessor = joblib.load("transformer.pkl")
    logger.info("Model and preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or preprocessor: {e}")
    raise

@app.get("/")
def home():
    logger.info("Home endpoint called.")
    return {"message": "Welcome to the Bank Churn Prediction API"}

@app.get("/health")
def health():
    logger.info("Health check endpoint called.")
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: ChurnModelInput):
    logger.info(f"Received prediction request: {data}")
    try:
        input_df = pd.DataFrame([data.dict()])
        logger.debug(f"Input data converted to DataFrame: {input_df}")

        X = preprocessor.transform(input_df)
        logger.debug(f"Preprocessed input data: {X}")

        pred = model.predict(X)
        logger.info(f"Model prediction result: {pred}")

        result = "Churn" if pred[0] == 1 else "No Churn"
        logger.info(f"Returning prediction: {result}")

        return {"prediction": result}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
