from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(
    title="ML Model Deployment API",
    description="Iris classification API",
    version="1.0"
)

# Load trained model
model = joblib.load("model/iris_model.pkl")

# Iris class names
class_names = [
    "setosa",
    "versicolor",
    "virginica"
]


@app.get("/")
def home():
    return {
        "message": "ML Model API is running"
    }


@app.post("/predict")
def predict(features: list[float]):

    # Convert input into NumPy array
    data = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction),
        "class": class_names[prediction]
    }