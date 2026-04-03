import json
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


DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def load_dataset():
    cols = list(cfg.FEATURES) + ["Outcome"]
    df = pd.read_csv(DATA_URL, header=None, names=cols)

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

    # Logistic Regression branch
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    lr_metrics = evaluate_binary(y_test, lr_pred, lr_proba)

    # Random Forest branch
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

    # SVM branch (comparison model)
    svm = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_proba = svm.predict_proba(X_test_scaled)[:, 1]

    svm_metrics = evaluate_binary(y_test, svm_pred, svm_proba)

    # Save all 3 models, but Random Forest is the main model for inference.
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

    print("Training done.")
    print("Main model: random_forest")
    print("Saved:")
    print("- models/random_forest.pkl")
    print("- models/logistic_regression.pkl")
    print("- models/svm.pkl")
    print("- models/scaler.pkl")
    print("- models/metadata.json")


if __name__ == "__main__":
    main()

