import json
import os
from functools import wraps
from urllib.parse import urlparse, unquote

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from config import cfg


app = Flask(__name__)
app.secret_key = cfg.SECRET_KEY

_DB_INIT_DONE = False
_DB_AVAILABLE = False


def get_db_connection():
    """
    Kết nối PostgreSQL.
    Yêu cầu thiết lập env `POSTGRES_DSN`.
    """
    try:
        if not cfg.POSTGRES_DSN:
            raise RuntimeError(
                "Thiếu cấu hình PostgreSQL. Vui lòng set env `POSTGRES_DSN` hoặc `DATABASE_URL`."
            )

        dsn = cfg.POSTGRES_DSN
        # Railway/Postgres thường yêu cầu SSL. Nếu DSN chưa có sslmode thì bật mặc định.
        if "sslmode=" not in dsn and (dsn.startswith("postgresql://") or dsn.startswith("postgres://")):
            dsn = dsn + ("&" if "?" in dsn else "?") + "sslmode=require"

        conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
        conn.autocommit = True
        return conn
    except Exception as e:
        raise RuntimeError(
            "Không kết nối được PostgreSQL. Vui lòng kiểm tra POSTGRES_DSN. Chi tiết: "
            + str(e)
        )


def ensure_schema():
    global _DB_AVAILABLE
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                result TEXT NOT NULL,
                probability DOUBLE PRECISION NOT NULL,
                risk TEXT NOT NULL,
                features_json TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_user_id ON prediction_logs(user_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_created_at ON prediction_logs(created_at)"
        )
        cur.close()
        conn.close()
        _DB_AVAILABLE = True
    except Exception as e:
        _DB_AVAILABLE = False
        # In ra dạng ASCII an toan (tránh lỗi unicode trong console).
        safe = repr(e).encode("unicode_escape").decode("ascii", errors="ignore")
        print("[warning] PostgreSQL not available:", safe)
        # Không ném lỗi để tránh server bị 500 ở mọi request.
        return False
    return True


@app.before_request
def _bootstrap():
    # Nếu PostgreSQL chưa sẵn sàng, sẽ lỗi ở các endpoint cần DB.
    global _DB_INIT_DONE
    if _DB_INIT_DONE:
        return
    ok = ensure_schema()
    if ok:
        _DB_INIT_DONE = True
    # Không trả về giá trị (Flask before_request phải trả None).
    return None


def _db_error_response(*, status_code: int = 503):
    # Trả lỗi rõ ràng tiếng Việt thay vì 500 chung
    msg = (
        "Không thể kết nối PostgreSQL. "
        "Vui lòng kiểm tra server PostgreSQL đang chạy và biến môi trường "
        "`POSTGRES_DSN` (hoặc `DATABASE_URL`/`RAILWAY_DATABASE_URL`)."
    )
    dsn_preview = None
    try:
        if cfg.POSTGRES_DSN:
            u = urlparse(cfg.POSTGRES_DSN)
            user = unquote(u.username) if u.username else ""
            host = u.hostname or ""
            port = u.port or ""
            dbname = u.path.lstrip("/")
            # An toan: khong hien password.
            auth_part = f"{user}:***@" if user else ""
            port_part = f":{port}" if port else ""
            dsn_preview = f"{u.scheme}://{auth_part}{host}{port_part}/{dbname}"
    except Exception:
        dsn_preview = None

    if request.is_json or request.content_type == "application/json":
        payload = {"error": msg}
        if dsn_preview:
            payload["dsn_preview"] = dsn_preview
        return jsonify(payload), status_code
    return render_template("db_error.html", message=msg, dsn_preview=dsn_preview), status_code


@app.errorhandler(RuntimeError)
def handle_runtime_error(e):
    if "PostgreSQL" in str(e):
        return _db_error_response()
    # fallback
    return render_template("db_error.html", message="Đã xảy ra lỗi trong hệ thống.", dsn=None), 500


@app.errorhandler(psycopg2.OperationalError)
def handle_psycopg_operational_error(e):
    return _db_error_response()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json or request.content_type == "application/json":
                return jsonify({"error": "Unauthorized. Please login first."}), 401
            flash("Vui lòng đăng nhập để tiếp tục.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper


def current_user():
    return {
        "id": session.get("user_id"),
        "username": session.get("username"),
    }


def save_prediction_log(*, user_id: int, result: str, probability: float, risk: str, features: dict, report: dict):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO prediction_logs(user_id, result, probability, risk, features_json, report_json)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            int(user_id),
            result,
            float(probability),
            risk,
            json.dumps(features, ensure_ascii=False),
            json.dumps(report, ensure_ascii=False),
        ),
    )
    cur.close()
    conn.close()

# -------- ML loading --------
RF_MODEL = None
SCALER = None
METADATA = {}


def _load_ml_artifacts():
    global RF_MODEL, SCALER, METADATA
    try:
        if cfg.RF_MODEL_PATH.exists():
            RF_MODEL = joblib.load(str(cfg.RF_MODEL_PATH))
        if cfg.SCALER_PATH.exists():
            SCALER = joblib.load(str(cfg.SCALER_PATH))
        if cfg.METADATA_PATH.exists():
            METADATA = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print("[warning] Failed to load ML artifacts:", e)


_load_ml_artifacts()


def parse_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def validate_features(input_dict):
    if not isinstance(input_dict, dict):
        return None, ["Invalid input: expected a JSON object or form fields."]

    errors = []
    features = {}

    for key in cfg.FEATURES:
        raw = input_dict.get(key)
        val = parse_float(raw)
        if val is None:
            errors.append(f"{key} is required and must be a number.")
            continue

        lo, hi = cfg.RANGES[key]
        if val < lo or val > hi:
            errors.append(f"{key} must be between {lo} and {hi}.")
            continue

        features[key] = val

    if errors:
        return None, errors

    features["Pregnancies"] = int(round(features["Pregnancies"]))
    features["Age"] = int(round(features["Age"]))
    return features, None


def predict_proba(features: dict):
    if RF_MODEL is None:
        raise RuntimeError("Model not loaded. Run `python train_model.py` first.")

    x = pd.DataFrame(
        [np.array([features[f] for f in cfg.FEATURES], dtype=float)],
        columns=list(cfg.FEATURES),
    )
    proba_all = RF_MODEL.predict_proba(x)[0]
    classes = list(getattr(RF_MODEL, "classes_", [0, 1]))
    if 1 in classes:
        return float(proba_all[classes.index(1)])
    return float(proba_all[-1])


def risk_from_probability(probability_01: float):
    p_percent = probability_01 * 100.0
    low_t, high_t = cfg.RISK_THRESHOLDS
    if p_percent < low_t:
        return "Low", p_percent
    if p_percent < high_t:
        return "Medium", p_percent
    return "High", p_percent


def result_label(probability_01: float):
    return "Positive" if probability_01 >= 0.5 else "Negative"


def _indicator_status(feature: str, value: float):
    if feature == "Pregnancies":
        if value <= 2:
            return "Ổn định", "Số lần mang thai ở mức thấp."
        if value <= 5:
            return "Theo dõi", "Số lần mang thai ở mức trung bình, cần theo dõi thêm các chỉ số chuyển hóa."
        return "Cao", "Số lần mang thai khá cao, nên kiểm soát thêm glucose và BMI."

    if feature == "Glucose":
        if value < 100:
            return "Ổn định", "Đường huyết nằm trong ngưỡng tốt."
        if value < 126:
            return "Theo dõi", "Đường huyết tăng nhẹ, nên điều chỉnh chế độ ăn."
        return "Cao", "Đường huyết cao, cần ưu tiên kiểm tra chuyên sâu."

    if feature == "BloodPressure":
        if value < 120:
            return "Ổn định", "Huyết áp ở ngưỡng bình thường."
        if value < 140:
            return "Theo dõi", "Huyết áp hơi cao, cần giảm muối và theo dõi định kỳ."
        return "Cao", "Huyết áp cao, nên thăm khám sớm."

    if feature == "SkinThickness":
        if value <= 30:
            return "Ổn định", "Độ dày da trong ngưỡng thường gặp."
        if value <= 40:
            return "Theo dõi", "Độ dày da tăng nhẹ, nên theo dõi thêm mỡ nội tạng/BMI."
        return "Cao", "Độ dày da cao, có thể liên quan tích lũy mỡ."

    if feature == "Insulin":
        if value < 140:
            return "Ổn định", "Insulin ở ngưỡng tham khảo ổn định."
        if value <= 199:
            return "Theo dõi", "Insulin tăng nhẹ, nên theo dõi kháng insulin."
        return "Cao", "Insulin cao, cần đánh giá thêm rối loạn chuyển hóa."

    if feature == "BMI":
        if value < 25:
            return "Ổn định", "BMI trong ngưỡng hợp lý."
        if value < 30:
            return "Theo dõi", "Thừa cân nhẹ, nên kiểm soát khẩu phần."
        return "Cao", "BMI cao, nên có kế hoạch giảm cân an toàn."

    if feature == "DiabetesPedigreeFunction":
        if value < 0.5:
            return "Ổn định", "Yếu tố di truyền ở mức thấp."
        if value < 1.0:
            return "Theo dõi", "Có yếu tố gia đình mức trung bình."
        return "Cao", "Yếu tố di truyền cao, nên tăng tần suất tầm soát."

    if feature == "Age":
        if value < 35:
            return "Ổn định", "Tuổi còn trẻ, nguy cơ nền thấp hơn."
        if value < 50:
            return "Theo dõi", "Tuổi trung niên, nên tầm soát định kỳ."
        return "Cao", "Tuổi cao hơn, nên theo dõi chuyển hóa sát hơn."

    return "Theo dõi", "Cần theo dõi thêm chỉ số này."


def build_health_report(features: dict, risk: str):
    labels = {
        "Pregnancies": ("Số lần mang thai", "lần"),
        "Glucose": ("Đường huyết", "mg/dL"),
        "BloodPressure": ("Huyết áp", "mmHg"),
        "SkinThickness": ("Độ dày da", "mm"),
        "Insulin": ("Insulin", "μU/mL"),
        "BMI": ("BMI", "kg/m²"),
        "DiabetesPedigreeFunction": ("Yếu tố di truyền", "điểm"),
        "Age": ("Tuổi", "năm"),
    }

    indicator_reviews = []
    recommendations = []
    warning_signals = []

    for feature in cfg.FEATURES:
        value = float(features[feature])
        status, note = _indicator_status(feature, value)
        vn_name, unit = labels[feature]
        indicator_reviews.append(
            {
                "feature": feature,
                "label": vn_name,
                "value": round(value, 2),
                "unit": unit,
                "status": status,
                "note": note,
            }
        )
        recommendations.append(f"{vn_name}: {note}")
        if status == "Cao":
            warning_signals.append(f"{vn_name} đang ở mức cao.")

    priority_actions = []
    if risk == "High":
        priority_actions = [
            "Ưu tiên khám chuyên khoa nội tiết sớm để xác nhận chẩn đoán.",
            "Làm thêm xét nghiệm HbA1c hoặc đường huyết đói theo tư vấn bác sĩ.",
            "Theo dõi huyết áp, cân nặng và đường huyết hàng tuần.",
        ]
    elif risk == "Medium":
        priority_actions = [
            "Điều chỉnh ăn uống và vận động trong 4-8 tuần tới.",
            "Giảm đường tinh luyện, giảm đồ uống ngọt, ngủ đủ giấc.",
            "Tái kiểm tra chỉ số chuyển hóa theo lịch định kỳ.",
        ]
    else:
        priority_actions = [
            "Duy trì thói quen sống lành mạnh hiện tại.",
            "Tập thể dục tối thiểu 150 phút mỗi tuần.",
            "Tầm soát sức khỏe định kỳ để phòng ngừa sớm.",
        ]

    follow_up_plan = [
        "Tuần 1-2: chuẩn hóa chế độ ăn, giảm đường và đồ chiên rán.",
        "Tuần 3-4: duy trì vận động đều, theo dõi cân nặng và huyết áp.",
        "Sau 4-8 tuần: đánh giá lại chỉ số và điều chỉnh kế hoạch.",
    ]

    return {
        "indicator_reviews": indicator_reviews,
        "recommendations": recommendations,
        "warning_signals": warning_signals,
        "priority_actions": priority_actions,
        "follow_up_plan": follow_up_plan,
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash FROM users WHERE username=%s",
            (username,),
        )
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Đăng nhập thành công.", "success")
            return redirect(url_for("index"))

        flash("Sai tài khoản hoặc mật khẩu.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if len(username) < 3:
            flash("Tên đăng nhập phải có ít nhất 3 ký tự.", "danger")
            return render_template("register.html")
        if len(password) < 6:
            flash("Mật khẩu phải có ít nhất 6 ký tự.", "danger")
            return render_template("register.html")
        if password != confirm_password:
            flash("Mật khẩu xác nhận không khớp.", "danger")
            return render_template("register.html")

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users(username, password_hash) VALUES (%s, %s)",
                (username, generate_password_hash(password)),
            )
            flash("Đăng ký thành công. Mời bạn đăng nhập.", "success")
            return redirect(url_for("login"))
        except psycopg2.errors.UniqueViolation:
            flash("Tên đăng nhập đã tồn tại.", "danger")
        finally:
            cur.close()
            conn.close()
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Bạn đã đăng xuất.", "success")
    return redirect(url_for("login"))


@app.route("/history", methods=["GET"])
@login_required
def history():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, result, probability, risk, features_json, report_json, created_at
        FROM prediction_logs
        WHERE user_id=%s
        ORDER BY id DESC
        LIMIT 100
        """,
        (session["user_id"],),
    )
    logs = []
    risk_vi = {"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"}
    risk_badge = {"Low": "text-bg-success", "Medium": "text-bg-warning", "High": "text-bg-danger"}
    result_vi = {"Positive": "Dương tính", "Negative": "Âm tính"}
    for row in cur.fetchall():
        report = json.loads(row["report_json"]) if row["report_json"] else {}
        logs.append(
            {
                "id": row["id"],
                "result": row["result"],
                "result_vi": result_vi.get(row["result"], row["result"]),
                "risk": row["risk"],
                "risk_vi": risk_vi.get(row["risk"], row["risk"]),
                "risk_badge": risk_badge.get(row["risk"], "text-bg-secondary"),
                "probability": float(row["probability"]),
                "features": json.loads(row["features_json"]),
                "report": report,
                "indicator_reviews": report.get("indicator_reviews", []),
                "warning_signals": report.get("warning_signals", []),
                "priority_actions": report.get("priority_actions", []),
                "recommendations": report.get("recommendations", []),
                "follow_up_plan": report.get("follow_up_plan", []),
                "created_at": row["created_at"],
            }
        )
    cur.close()
    conn.close()
    return render_template("history.html", logs=logs)


@app.route("/history/delete/<int:log_id>", methods=["POST"])
@login_required
def delete_history_item(log_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM prediction_logs WHERE id=%s AND user_id=%s",
        (log_id, session["user_id"]),
    )
    deleted = cur.rowcount
    cur.close()
    conn.close()
    if deleted:
        flash("Đã xóa bản ghi thành công.", "success")
    else:
        flash("Không tìm thấy bản ghi hoặc bạn không có quyền xóa.", "warning")
    return redirect(url_for("history"))


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM prediction_logs WHERE user_id=%s",
        (session["user_id"],),
    )
    deleted = cur.rowcount
    cur.close()
    conn.close()
    flash(f"Đã xóa toàn bộ nhật ký ({deleted} bản ghi).", "success")
    return redirect(url_for("history"))


# -------- Pages --------
@app.route("/", methods=["GET"])
@login_required
def index():
    metrics = {
        "logistic": METADATA.get("logistic_regression_metrics"),
        "random_forest": METADATA.get("random_forest_metrics"),
        "svm": METADATA.get("svm_metrics"),
    }
    return render_template("index.html", metrics=metrics, user=current_user())


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    is_json = request.is_json or request.content_type == "application/json"
    if is_json:
        payload = request.get_json(silent=True)
    else:
        payload = request.form.to_dict()

    if is_json and payload is None:
        return (
            jsonify({"error": "Invalid JSON body. Expected JSON object."}),
            400,
        )

    try:
        features, errors = validate_features(payload)
        if errors:
            if is_json:
                return jsonify({"error": "Validation failed", "details": errors}), 400
            return render_template("result.html", error=errors)

        prob = predict_proba(features)
        risk, p_percent = risk_from_probability(prob)
        res = result_label(prob)
        health_report = build_health_report(features, risk)
        save_prediction_log(
            user_id=session["user_id"],
            result=res,
            probability=prob,
            risk=risk,
            features=features,
            report=health_report,
        )
        p_percent_rounded = round(p_percent, 2)
        risk_bar_class = {
            "High": "bg-danger",
            "Medium": "bg-warning text-dark",
            "Low": "bg-success",
        }.get(risk, "bg-primary")
        risk_progress_class = {
            "High": "progress-width-high",
            "Medium": "progress-width-medium",
            "Low": "progress-width-low",
        }.get(risk, "progress-width-medium")

        if is_json:
            return (
                jsonify(
                    {
                        "main_model": "Random Forest",
                        "result": res,
                        "probability": prob,
                        "risk": risk,
                        "probability_percent": p_percent_rounded,
                        "health_report": health_report,
                        "metrics": METADATA,
                    }
                ),
                200,
            )

        return render_template(
            "result.html",
            result=res,
            result_vi="Dương tính" if res == "Positive" else "Âm tính",
            probability=prob,
            risk=risk,
            risk_vi={"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"}.get(risk, risk),
            p_percent=p_percent,
            p_percent_rounded=p_percent_rounded,
            risk_bar_class=risk_bar_class,
            risk_progress_class=risk_progress_class,
            indicator_reviews=health_report["indicator_reviews"],
            recommendations=health_report["recommendations"],
            warning_signals=health_report["warning_signals"],
            priority_actions=health_report["priority_actions"],
            follow_up_plan=health_report["follow_up_plan"],
        )
    except Exception as e:
        if is_json:
            return jsonify({"error": "Server error", "details": str(e)}), 500
        return render_template("result.html", error=[f"Prediction failed: {e}"])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)

