Here's a comprehensive `README.md` for your Bank Churn Prediction API, highlighting why GBOOST was selected as the best model from your research branch:

```markdown
# Bank Churn Prediction API

A FastAPI-based service that predicts customer churn probability using machine learning.

## Features

- REST API endpoint for churn predictions
- Input validation using Pydantic models
- Comprehensive logging
- Health check endpoint
- Uses optimized GBOOST model from research

## Model Selection

After extensive testing in our research branch, we selected **Gradient Boosting (GBOOST)** as our production model 
## API Endpoints

### `GET /`
- Basic welcome message
- **Response:** 
  ```json
  {"message": "Welcome to the Bank Churn Prediction API"}
  ```

### `GET /health`
- Service health check
- **Response:**
  ```json
  {"status": "healthy"}
  ```

### `POST /predict`
- Main prediction endpoint
- **Request Body:**
  ```json
  {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 5,
    "Balance": 50000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000.0
  }
  ```
- **Successful Response:**
  ```json
  {"prediction": "Churn"}
  ```
- **Error Response:**
  ```json
  {"detail": "Prediction failed."}
  ```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


2. Run the API:
   ```bash
   uvicorn main:app --reload
   ```

## Logging

The API generates logs in two locations:
1. Console output
2. File: `api.log`

Log format:
```
2023-08-20 14:30:45 [INFO] Model and preprocessor loaded successfully.
```

## Example Usage

```python
import requests

data = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 5,
    "Balance": 50000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000.0
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Future Improvements

- Add model versioning
- Implement feature drift monitoring
- Add prediction confidence scores
- Dockerize the application
```
