import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Flask
    SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
    SQLITE_PATH: Path = Path(os.environ.get("SQLITE_PATH", "disease_prediction.sqlite"))

    # Model artifacts
    MODEL_DIR: Path = Path(os.environ.get("MODEL_DIR", "models"))
    RF_MODEL_PATH: Path = MODEL_DIR / "random_forest.pkl"
    LR_MODEL_PATH: Path = MODEL_DIR / "logistic_regression.pkl"
    SVM_MODEL_PATH: Path = MODEL_DIR / "svm.pkl"
    SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"
    METADATA_PATH: Path = MODEL_DIR / "metadata.json"

    # Pima dataset features (must match train_model.py exactly)
    FEATURES = (
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    )

    # Risk thresholds based on probability percentage
    # <30 Low, 30-60 Medium, >60 High
    RISK_THRESHOLDS = (30.0, 60.0)

    # Input validation ranges for demo
    RANGES = {
        "Pregnancies": (0, 20),
        "Glucose": (0, 300),
        "BloodPressure": (0, 200),
        "SkinThickness": (0, 100),
        "Insulin": (0, 2000),
        "BMI": (0, 100),
        "DiabetesPedigreeFunction": (0, 5),
        "Age": (0, 120),
    }

cfg = Config()
