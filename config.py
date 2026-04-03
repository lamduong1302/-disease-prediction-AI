import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class Config:
    # Flask — khóa phiên (đổi trên production)
    SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

    # Chuỗi kết nối PostgreSQL (ưu tiên: POSTGRES_DSN → DATABASE_URL → RAILWAY_DATABASE_URL)
    POSTGRES_DSN: str = (
        os.environ.get("POSTGRES_DSN")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("RAILWAY_DATABASE_URL")
        or ""
    ).strip()

    # Thư mục chứa mô hình đã huấn luyện
    MODEL_DIR: Path = Path(os.environ.get("MODEL_DIR", "models"))
    RF_MODEL_PATH: Path = MODEL_DIR / "random_forest.pkl"
    LR_MODEL_PATH: Path = MODEL_DIR / "logistic_regression.pkl"
    SVM_MODEL_PATH: Path = MODEL_DIR / "svm.pkl"
    SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"
    METADATA_PATH: Path = MODEL_DIR / "metadata.json"

    # Tên cột đặc trưng (phải khớp train_model.py)
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

    # Ngưỡng rủi ro theo % xác suất: <30 thấp, 30–60 trung bình, >60 cao
    RISK_THRESHOLDS = (30.0, 60.0)

    # Giới hạn nhập liệu cho form demo
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
