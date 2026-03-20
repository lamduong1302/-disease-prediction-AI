import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Flask
    SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

    # MySQL
    MYSQL_HOST: str = os.environ.get("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.environ.get("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.environ.get("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.environ.get("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.environ.get("MYSQL_DATABASE", "disease_db")

    # Model artifacts
    MODEL_DIR: Path = Path(os.environ.get("MODEL_DIR", "models"))
    MODEL_PATH: Path = MODEL_DIR / "model.pkl"
    SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"

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
    # <30 Low, 30-70 Medium, >70 High
    RISK_THRESHOLDS = (30.0, 70.0)

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

