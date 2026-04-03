import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import cfg

# Trên Windows, console mặc định có thể không phải UTF-8 — tránh lỗi khi in tiếng Việt.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def load_dataset():
    cols = list(cfg.FEATURES) + ["Outcome"]
    df = pd.read_csv(DATA_URL, header=None, names=cols)

    # Các cột này dùng 0 làm giá trị thiếu trong dataset gốc — chuyển 0 thành NaN rồi điền median.
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

    df = df.fillna(df.median(numeric_only=True))

    X = df[list(cfg.FEATURES)].astype(float)
    y = df["Outcome"].astype(int)
    return X, y


def evaluate_binary(y_true, y_pred, y_proba=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "proba_available": bool(y_proba is not None),
    }


def main():
    models_dir = cfg.MODEL_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Nhánh Logistic Regression (dữ liệu đã chuẩn hóa)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    lr_metrics = evaluate_binary(y_test, lr_pred, lr_proba)

    # Random Forest — mô hình chính khi suy luận (không bắt buộc scale)
    rf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    rf_metrics = evaluate_binary(y_test, rf_pred, rf_proba)

    # SVM — mô hình so sánh (dùng dữ liệu đã scale)
    svm = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_proba = svm.predict_proba(X_test_scaled)[:, 1]

    svm_metrics = evaluate_binary(y_test, svm_pred, svm_proba)

    # Lưu cả 3 mô hình; suy luận trên web dùng Random Forest + scaler (LR/SVM chỉ để so sánh offline).
    joblib.dump(rf, str(cfg.RF_MODEL_PATH))
    joblib.dump(lr, str(cfg.LR_MODEL_PATH))
    joblib.dump(svm, str(cfg.SVM_MODEL_PATH))
    joblib.dump(scaler, str(models_dir / "scaler.pkl"))

    ranking = sorted(
        [
            ("random_forest", rf_metrics["accuracy"]),
            ("logistic_regression", lr_metrics["accuracy"]),
            ("svm", svm_metrics["accuracy"]),
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    meta = {
        "main_model": "random_forest",
        "ranking_by_accuracy": ranking,
        "logistic_regression_metrics": lr_metrics,
        "random_forest_metrics": rf_metrics,
        "svm_metrics": svm_metrics,
    }
    (models_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Huấn luyện xong.")
    print("Mô hình chính: random_forest")
    print("Đã lưu:")
    print("- models/random_forest.pkl")
    print("- models/logistic_regression.pkl")
    print("- models/svm.pkl")
    print("- models/scaler.pkl")
    print("- models/metadata.json")


if __name__ == "__main__":
    main()
