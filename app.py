import csv
import io
import os
from functools import wraps
from pathlib import Path

import joblib
import mysql.connector
import numpy as np
from flask import (
    Flask,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from config import cfg


app = Flask(__name__)
app.secret_key = cfg.SECRET_KEY


_DB_INIT_DONE = False


def get_db_connection():
    host = cfg.MYSQL_HOST
    user = cfg.MYSQL_USER
    password = cfg.MYSQL_PASSWORD
    db_name = cfg.MYSQL_DATABASE

    try:
        return mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=db_name,
            autocommit=False,
        )
    except mysql.connector.Error as e:
        # Unknown database -> create it and reconnect.
        if getattr(e, "errno", None) == 1049:
            admin_conn = mysql.connector.connect(
                host=host, user=user, password=password, autocommit=True
            )
            cur = admin_conn.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            cur.close()
            admin_conn.close()
            return mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=db_name,
                autocommit=False,
            )
        raise


def ensure_schema():
    """
    Auto-create DB tables to make the project copy-run easier.
    You can disable via env AUTO_INIT_DB=0.
    """
    global _DB_INIT_DONE
    if _DB_INIT_DONE:
        return

    if os.environ.get("AUTO_INIT_DB", "1") != "1":
        _DB_INIT_DONE = True
        return

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
          id INT AUTO_INCREMENT PRIMARY KEY,
          username VARCHAR(100) NOT NULL UNIQUE,
          password_hash VARCHAR(255) NOT NULL,
          role VARCHAR(20) NOT NULL DEFAULT 'user',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Nếu DB đã tồn tại từ lần chạy trước (schema cũ), MySQL sẽ không tự thêm cột mới.
    # Ví dụ: users thiếu cột `role` -> đăng ký sẽ lỗi.
    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name='users' AND column_name='role'
        """,
        (cfg.MYSQL_DATABASE,),
    )
    row = cur.fetchone()
    if row and int(row[0]) == 0:
        cur.execute("ALTER TABLE users ADD role VARCHAR(20) NOT NULL DEFAULT 'user'")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
          id INT AUTO_INCREMENT PRIMARY KEY,
          pregnancies INT,
          glucose FLOAT,
          blood_pressure FLOAT,
          bmi FLOAT,
          age INT,
          result VARCHAR(10),
          probability FLOAT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Một số lần bạn đã có table predictions cũ nên có thể thiếu cột.
    # Add cột probability nếu chưa có.
    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name='predictions' AND column_name='probability'
        """,
        (cfg.MYSQL_DATABASE,),
    )
    row = cur.fetchone()
    if row and int(row[0]) == 0:
        cur.execute("ALTER TABLE predictions ADD probability FLOAT DEFAULT 0")

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_result ON predictions(result)"
    )

    conn.commit()
    cur.close()
    conn.close()

    _DB_INIT_DONE = True


@app.before_request
def _bootstrap():
    try:
        ensure_schema()
    except Exception:
        # Don't crash due to schema init if DB isn't ready; real failures happen on query.
        pass


def login_required_web(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper


def login_required_api(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)

    return wrapper


def role_required(role_name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                return jsonify({"error": "Unauthorized"}), 401
            if session.get("role") != role_name:
                return jsonify({"error": "Forbidden"}), 403
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def init_admin_if_needed():
    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if not admin_username or not admin_password:
        return

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id FROM users WHERE username=%s", (admin_username,))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO users(username, password_hash, role) VALUES (%s, %s, %s)",
            (admin_username, generate_password_hash(admin_password), "admin"),
        )
        conn.commit()
        print("[info] Auto-created admin user from env.")
    cur.close()
    conn.close()


@app.before_request
def _init_admin():
    # Cheap check; keeps admin demo user available.
    try:
        init_admin_if_needed()
    except Exception:
        pass


# -------- ML loading --------
MODEL = None
SCALER = None
METADATA = {}


def _load_ml_artifacts():
    global MODEL, SCALER, METADATA
    try:
        if cfg.MODEL_PATH.exists():
            MODEL = joblib.load(str(cfg.MODEL_PATH))
        if cfg.SCALER_PATH.exists():
            SCALER = joblib.load(str(cfg.SCALER_PATH))

        meta_path = cfg.MODEL_DIR / "metadata.json"
        if meta_path.exists():
            import json

            METADATA = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print("[warning] Failed to load ML artifacts:", e)


_load_ml_artifacts()


def parse_float(value):
    # Accept int/float/strings; return None if invalid.
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def validate_features(input_dict):
    """
    Validate non-null, non-negative, and range checks.
    Returns: (features_dict, errors_list)
    """
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

    # Cast integer-like features
    features["Pregnancies"] = int(round(features["Pregnancies"]))
    features["Age"] = int(round(features["Age"]))
    return features, None


def _positive_proba(model, x):
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(x)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return float(proba_all[classes.index(1)])
        # fallback
        return float(proba_all[-1])
    # fallback: map class to probability-ish (not calibrated)
    pred = model.predict(x)[0]
    return 1.0 if int(pred) == 1 else 0.0


def predict_proba(features: dict):
    if MODEL is None:
        raise RuntimeError("Model not loaded. Run `python train_model.py` first.")

    x = np.array([[features[f] for f in cfg.FEATURES]], dtype=float)

    expects_scaled = bool(METADATA.get("expects_scaled", False))
    if expects_scaled:
        if SCALER is None:
            raise RuntimeError("Scaler not loaded but expects_scaled=True.")
        x = SCALER.transform(x)

    # If metadata is missing, try best-effort:
    if METADATA == {} and SCALER is not None:
        # If model is logistic, it should benefit from scaling.
        try:
            x_scaled = SCALER.transform(x)
            return _positive_proba(MODEL, x_scaled)
        except Exception:
            pass

    return _positive_proba(MODEL, x)


def risk_from_probability(probability_01: float):
    """
    probability_01 in [0,1]
    Risk based on probability percentage:
      <30 Low, 30-70 Medium, >70 High
    """
    p_percent = probability_01 * 100.0
    low_t, high_t = cfg.RISK_THRESHOLDS
    if p_percent < low_t:
        return "Low", p_percent
    if p_percent <= high_t:
        return "Medium", p_percent
    return "High", p_percent


def result_label(probability_01: float):
    return "Positive" if probability_01 >= 0.5 else "Negative"


def _feature_bucket(value: float, feature_name: str):
    """
    Bucket hóa cho mục đích giải thích (heuristic), KHÔNG ảnh hưởng dự đoán.
    """
    lo, hi = cfg.RANGES[feature_name]
    if hi <= lo:
        return "trung binh"
    ratio = (value - lo) / (hi - lo)
    if ratio < 0.33:
        return "thap"
    if ratio > 0.66:
        return "cao"
    return "trung binh"


def get_model_feature_scores():
    """
    Score per feature để chọn "top factors" giải thích.
    """
    if MODEL is None:
        return [0.0 for _ in cfg.FEATURES]

    # RandomForestClassifier
    if hasattr(MODEL, "feature_importances_"):
        scores = list(getattr(MODEL, "feature_importances_"))
        if len(scores) != len(cfg.FEATURES):
            return [0.0 for _ in cfg.FEATURES]
        return [float(x) for x in scores]

    # LogisticRegression
    if hasattr(MODEL, "coef_"):
        coef = np.array(getattr(MODEL, "coef_")).reshape(-1)
        if coef.shape[0] != len(cfg.FEATURES):
            return [0.0 for _ in cfg.FEATURES]
        return [float(abs(x)) for x in coef]

    return [0.0 for _ in cfg.FEATURES]


def top_factors_from_inputs(features: dict, top_n: int = 3):
    """
    Chọn top_n feature nổi bật theo score của model (nếu có),
    kèm bucket heuristic dựa trên range cấu hình để mô tả bằng lời.
    """
    scores = get_model_feature_scores()
    ranked = sorted(range(len(cfg.FEATURES)), key=lambda i: scores[i], reverse=True)

    # fallback: nếu score đều 0, dùng lệch khỏi trung tâm range
    if all(s == 0.0 for s in scores):
        ranked = sorted(
            range(len(cfg.FEATURES)),
            key=lambda i: abs(features[cfg.FEATURES[i]] - (sum(cfg.RANGES[cfg.FEATURES[i]]) / 2.0)),
            reverse=True,
        )

    chosen = ranked[:top_n]
    out = []
    bucket_vi = {"thap": "thấp", "trung binh": "trung bình", "cao": "cao"}

    for idx in chosen:
        fname = cfg.FEATURES[idx]
        val = float(features[fname])
        lo, hi = cfg.RANGES[fname]
        bucket_en = _feature_bucket(val, fname)  # thap/trung binh/cao
        out.append(
            {
                "feature": fname,
                "value": val,
                "bucket": bucket_vi.get(bucket_en, bucket_en),
                "range": (lo, hi),
                "score": scores[idx],
            }
        )
    return out


def build_causes(features: dict, risk: str, top_n: int = 3):
    """
    Sinh danh sách nguyên nhân/điểm nổi bật (text) cho UI.
    Heuristic dựa trên input + top factors từ model, KHÔNG thay thế y khoa.
    """
    relation = {
        "Pregnancies": "liên quan đến thay đổi chuyển hóa trong thai kỳ và các yếu tố sức khỏe lâu dài",
        "Glucose": "thường gắn với khả năng kiểm soát đường huyết/kháng insulin",
        "BloodPressure": "có thể đi kèm rối loạn chuyển hóa (hội chứng chuyển hóa)",
        "SkinThickness": "có thể phản ánh mức độ tích tụ mỡ dưới da",
        "Insulin": "có thể liên quan tới kháng insulin và chức năng tiết insulin",
        "BMI": "phản ánh thừa cân/béo phì, thường làm tăng nguy cơ rối loạn chuyển hóa",
        "DiabetesPedigreeFunction": "phản ánh yếu tố di truyền/tiền sử gia đình",
        "Age": "tuổi càng cao thì nguy cơ thường tăng",
    }

    pretty = {
        "Pregnancies": "Pregnancies (lần)",
        "Glucose": "Glucose (mg/dL)",
        "BloodPressure": "BloodPressure (mm Hg)",
        "SkinThickness": "SkinThickness (mm)",
        "Insulin": "Insulin (μU/mL)",
        "BMI": "BMI (kg/m²)",
        "DiabetesPedigreeFunction": "DiabetesPedigreeFunction",
        "Age": "Age (năm)",
    }

    factors = top_factors_from_inputs(features, top_n=top_n)
    causes = []
    for f in factors:
        lo, hi = f["range"]
        causes.append(
            f"{pretty.get(f['feature'], f['feature'])} của bạn đang {f['bucket']} so với khoảng dữ liệu huấn luyện ({lo} - {hi}). "
            f"Điều này thường liên quan: {relation.get(f['feature'], '')}."
        )

    if risk == "High":
        causes.append("Tổng hợp các chỉ số nổi bật tạo ra xác suất rủi ro cao trong mô hình.")
    elif risk == "Medium":
        causes.append("Một số chỉ số nổi bật làm tăng rủi ro ở mức trung bình trong mô hình.")
    else:
        causes.append("Các chỉ số nổi bật không cho thấy rủi ro cao trong mô hình.")
    return causes


def build_recommendations(risk: str):
    if risk == "High":
        return [
            "Nên ưu tiên đi khám/chẩn đoán và trao đổi với bác sĩ về các xét nghiệm phù hợp (ví dụ HbA1c/đường huyết).",
            "Ưu tiên thay đổi lối sống: giảm đường tinh luyện, giảm đồ chế biến sẵn, tăng vận động và duy trì cân nặng hợp lý.",
            "Lên lịch theo dõi định kỳ theo hướng dẫn chuyên môn.",
            "Kết quả dự đoán chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa.",
        ]
    if risk == "Medium":
        return [
            "Điều chỉnh chế độ ăn và tăng vận động để giảm nguy cơ.",
            "Theo dõi đường huyết/xét nghiệm theo lịch, đặc biệt nếu chỉ số thực tế đang cao.",
            "Nếu BMI/BP/đường huyết đang cao, cân nhắc can thiệp sớm hơn.",
            "Kết quả dự đoán chỉ mang tính tham khảo.",
        ]
    return [
        "Duy trì lối sống lành mạnh: ăn cân bằng, hạn chế đường tinh luyện, tăng chất xơ và duy trì vận động.",
        "Theo dõi sức khỏe định kỳ để phát hiện sớm thay đổi.",
        "Nếu có tiền sử gia đình hoặc triệu chứng, nên tư vấn bác sĩ để được cá nhân hóa.",
        "Kết quả dự đoán chỉ mang tính tham khảo.",
    ]


def build_nutrition_tips(features: dict, risk: str):
    """
    Tips dinh dưỡng mức khái quát dựa trên các feature đầu vào (phù hợp dataset Pima).
    """
    tips = []
    glucose = float(features["Glucose"])
    bmi = float(features["BMI"])
    bp = float(features["BloodPressure"])

    if glucose >= 140:
        tips.append("Giảm đường và carb tinh chế: hạn chế nước ngọt/đồ ngọt; ưu tiên carb ít tinh chế và nhiều chất xơ.")
    else:
        tips.append("Duy trì kiểm soát carb: hạn chế đồ ngọt; chọn carb giàu chất xơ (rau, ngũ cốc nguyên hạt).")

    if bmi >= 30:
        tips.append("Nếu BMI cao: ưu tiên giảm cân bền vững (giảm năng lượng nạp + tăng vận động), ưu tiên đạm nạc và rau.")
    elif bmi >= 25:
        tips.append("Nếu BMI hơi cao: điều chỉnh khẩu phần để tránh tăng cân thêm, tăng hoạt động thể lực.")
    else:
        tips.append("BMI ở mức tương đối: tập trung ăn cân bằng và duy trì thói quen lành mạnh.")

    if bp >= 130:
        tips.append("Nếu BP cao: hạn chế muối/đồ chế biến sẵn, tăng thực phẩm tươi (rau quả), ngủ đủ và uống đủ nước.")
    else:
        tips.append("Giữ huyết áp ổn định: giảm đồ mặn, hạn chế rượu/bia và ngủ đủ giấc.")

    if risk == "High":
        tips.append("Nguy cơ cao: nên tham khảo thêm chuyên gia dinh dưỡng để cá nhân hoá kế hoạch ăn uống.")
    elif risk == "Medium":
        tips.append("Nguy cơ trung bình: ưu tiên can thiệp sớm 4-8 tuần và theo dõi lại chỉ số/xét nghiệm theo lịch.")
    else:
        tips.append("Nguy cơ thấp: tiếp tục duy trì chế độ ăn lành mạnh và hạn chế đường tinh chế.")

    return tips


def build_disclaimer():
    return "Lưu ý: nội dung giải thích/khuyến nghị dựa trên dữ liệu và mô hình (Pima Diabetes) + quy tắc tổng quát, chỉ dùng cho mục đích tham khảo demo; không thay thế tư vấn y khoa."


def store_prediction(features: dict, result: str, probability_01: float):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions(pregnancies, glucose, blood_pressure, bmi, age, result, probability)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            int(features["Pregnancies"]),
            float(features["Glucose"]),
            float(features["BloodPressure"]),
            float(features["BMI"]),
            int(features["Age"]),
            result,
            float(probability_01),
        ),
    )
    conn.commit()
    pred_id = cur.lastrowid
    cur.close()
    conn.close()
    return pred_id


def get_dashboard_data():
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT COUNT(*) as total FROM predictions")
    total = int(cur.fetchone()["total"])

    cur.execute("SELECT COUNT(*) as pos FROM predictions WHERE result='Positive'")
    pos = int(cur.fetchone()["pos"])

    cur.execute(
        """
        SELECT id, pregnancies, glucose, blood_pressure, bmi, age, result, probability, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
        """
    )
    recent = cur.fetchall()

    cur.execute("SELECT probability FROM predictions")
    probs = cur.fetchall()

    cur.close()
    conn.close()

    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    for row in probs:
        prob = float(row["probability"])
        risk, _ = risk_from_probability(prob)
        risk_counts[risk] += 1

    positive_rate = (pos / total * 100.0) if total else 0.0

    return {
        "total": total,
        "positive_count": pos,
        "positive_rate": positive_rate,
        "risk_counts": risk_counts,
        "recent": recent,
    }


# -------- Auth --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username/password.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if len(username) < 3 or len(password) < 4:
            flash("Username >= 3 chars, password >= 4 chars.", "danger")
            return render_template("register.html")

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users(username, password_hash, role) VALUES (%s,%s,'user')",
                (username, generate_password_hash(password)),
            )
            conn.commit()
            flash("Register successful. Please login.", "success")
            return redirect(url_for("login"))
        except mysql.connector.IntegrityError:
            flash("Username already exists.", "danger")
        finally:
            cur.close()
            conn.close()
    return render_template("register.html")


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("login"))


# -------- Pages --------
@app.route("/", methods=["GET"])
@login_required_web
def index():
    return render_template("index.html", username=session.get("username"))


@app.route("/predict", methods=["POST"])
@login_required_web
def predict():
    is_json = request.is_json or request.content_type == "application/json"

    try:
        payload = request.get_json(silent=True) if is_json else request.form.to_dict()
        features, errors = validate_features(payload)
        if errors:
            if is_json:
                return jsonify({"error": "Validation failed", "details": errors}), 400
            for e in errors:
                flash(e, "danger")
            return redirect(url_for("index"))

        prob = predict_proba(features)
        risk, _p_percent = risk_from_probability(prob)
        res = result_label(prob)
        pred_id = store_prediction(features, res, prob)
        causes = build_causes(features, risk=risk, top_n=3)
        recommendations = build_recommendations(risk)
        nutrition_tips = build_nutrition_tips(features, risk)
        disclaimer = build_disclaimer()

        if is_json:
            return (
                jsonify(
                    {
                        "result": res,
                        "probability": prob,
                        "risk": risk,
                        "causes": causes,
                        "recommendations": recommendations,
                        "nutrition_tips": nutrition_tips,
                        "disclaimer": disclaimer,
                    }
                ),
                200,
            )

        return render_template(
            "result.html",
            result=res,
            probability=prob,
            risk=risk,
            p_percent=_p_percent,
            prediction_id=pred_id,
            causes=causes,
            recommendations=recommendations,
            nutrition_tips=nutrition_tips,
            disclaimer=disclaimer,
        )

    except Exception as e:
        if is_json:
            return jsonify({"error": "Server error", "details": str(e)}), 500
        flash(f"Prediction failed: {e}", "danger")
        return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET"])
@login_required_web
def dashboard():
    return render_template("dashboard.html", username=session.get("username"))


@app.route("/dashboard-data", methods=["GET"])
@login_required_api
def dashboard_data():
    try:
        data = get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500


@app.route("/admin/export-csv", methods=["GET"])
@login_required_web
def export_csv():
    if session.get("role") != "admin":
        flash("Admin only.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT id, pregnancies, glucose, blood_pressure, bmi, age, result, probability, created_at
        FROM predictions ORDER BY id DESC
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "pregnancies",
            "glucose",
            "blood_pressure",
            "bmi",
            "age",
            "result",
            "probability",
            "created_at",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r["id"],
                r["pregnancies"],
                r["glucose"],
                r["blood_pressure"],
                r["bmi"],
                r["age"],
                r["result"],
                r["probability"],
                r["created_at"],
            ]
        )

    resp = make_response(output.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="predictions_history.csv"'
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)

import os
from functools import wraps
from pathlib import Path

import mysql.connector
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")


def get_db():
    # Defaults giữ nguyên như project bạn đang dùng (root, password rỗng).
    host = os.environ.get("MYSQL_HOST", "localhost")
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "")
    db_name = os.environ.get("MYSQL_DATABASE", "disease_db")

    try:
        return mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=db_name,
        )
    except mysql.connector.Error as e:
        # Nếu DB chưa tồn tại, tạo DB để app chạy ngay.
        # MySQL error code 1049: Unknown database
        if getattr(e, "errno", None) == 1049:
            conn = mysql.connector.connect(host=host, user=user, password=password)
            cur = conn.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            cur.close()
            conn.close()
            return mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=db_name,
            )
        raise


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper


MODEL_PATH = Path(__file__).with_name("model.pkl")
_model = None
if joblib is not None and MODEL_PATH.exists():
    try:
        _model = joblib.load(str(MODEL_PATH))
    except Exception:
        _model = None


def normalize_prediction(pred):
    # Model thực tế có thể trả về:
    # - 0/1 (int/float)
    # - "High Risk"/"Low Risk" (str)
    # - hoặc các nhãn khác
    if isinstance(pred, (int, float)):
        return "High Risk" if float(pred) >= 0.5 else "Low Risk"
    s = str(pred).strip().lower()
    if "high" in s or s in {"1", "true", "high_risk"}:
        return "High Risk"
    if "low" in s or s in {"0", "false", "low_risk"}:
        return "Low Risk"
    return str(pred)


def predict_risk(age, glucose, bp, bmi):
    if _model is not None:
        try:
            pred = _model.predict([[age, glucose, bp, bmi]])[0]
            return normalize_prediction(pred)
        except Exception:
            # Nếu model format input không khớp thì fallback để app vẫn chạy.
            pass
    # Fake logic fallback (như bản gốc).
    return "High Risk" if float(glucose) > 140 else "Low Risk"


def init_db():
    conn = get_db()
    cur = conn.cursor()

    # Patients (giúp chạy ngay kể cả khi bạn chưa chạy database.sql).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            age INT,
            glucose FLOAT,
            blood_pressure FLOAT,
            bmi FLOAT,
            result VARCHAR(50)
        );
        """
    )
    conn.commit()

    # Tạo table users để làm Login.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()

    # Auto-create admin để chạy ngay (demo).
    auto_create = os.environ.get("AUTO_CREATE_ADMIN", "1") == "1"
    if auto_create:
        admin_user = os.environ.get("ADMIN_USERNAME", "admin")
        admin_pass = os.environ.get("ADMIN_PASSWORD", "admin")
        cur.execute("SELECT id FROM users WHERE username=%s", (admin_user,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO users(username, password_hash) VALUES (%s, %s)",
                (admin_user, generate_password_hash(admin_pass)),
            )
            conn.commit()
            print(
                "[warning] Auto-created admin user. Change ADMIN_PASSWORD for production."
            )
    cur.close()
    conn.close()

_db_init_done = False


@app.before_request
def _bootstrap():
    global _db_init_done
    if _db_init_done:
        return
    try:
        init_db()
        _db_init_done = True
    except Exception as e:
        # Không chặn startup nếu DB chưa sẵn sàng (ví dụ bạn chưa chạy database.sql).
        print(f"[warning] init_db failed: {e}")


def parse_patient_form():
    name = (request.form.get("name") or "").strip()
    age = request.form.get("age")
    glucose = request.form.get("glucose")
    bp = request.form.get("bp")
    bmi = request.form.get("bmi")

    if not name:
        raise ValueError("Tên không được để trống.")

    age = int(age)
    glucose = float(glucose)
    bp = float(bp)
    bmi = float(bmi)
    return name, age, glucose, bp, bmi


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
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


@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất.", "success")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template(
        "index.html", patients=data, username=session.get("username")
    )


@app.route("/add", methods=["POST"])
@login_required
def add():
    try:
        name, age, glucose, bp, bmi = parse_patient_form()
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))

    result = predict_risk(age=age, glucose=glucose, bp=bp, bmi=bmi)

    conn = get_db()
    cursor = conn.cursor()
    sql = """
        INSERT INTO patients(name, age, glucose, blood_pressure, bmi, result)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (name, age, glucose, bp, bmi, result))
    conn.commit()
    cursor.close()
    conn.close()
    flash("Đã thêm bệnh nhân.", "success")
    return redirect(url_for("index"))


@app.route("/edit/<int:id>", methods=["GET"])
@login_required
def edit(id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM patients WHERE id=%s", (id,))
    patient = cursor.fetchone()
    cursor.close()
    conn.close()

    if not patient:
        flash("Không tìm thấy bệnh nhân.", "danger")
        return redirect(url_for("index"))
    return render_template(
        "edit.html", patient=patient, username=session.get("username")
    )


@app.route("/update/<int:id>", methods=["POST"])
@login_required
def update(id):
    try:
        name, age, glucose, bp, bmi = parse_patient_form()
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("edit", id=id))

    result = predict_risk(age=age, glucose=glucose, bp=bp, bmi=bmi)

    conn = get_db()
    cursor = conn.cursor()
    sql = """
        UPDATE patients
        SET name=%s, age=%s, glucose=%s, blood_pressure=%s, bmi=%s, result=%s
        WHERE id=%s
    """
    cursor.execute(sql, (name, age, glucose, bp, bmi, result, id))
    conn.commit()
    cursor.close()
    conn.close()

    flash("Đã cập nhật.", "success")
    return redirect(url_for("index"))


@app.route("/delete", methods=["POST"])
@login_required
def delete():
    pid = request.form.get("id")
    if not pid:
        flash("Thiếu id.", "danger")
        return redirect(url_for("index"))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patients WHERE id=%s", (int(pid),))
    conn.commit()
    cursor.close()
    conn.close()
    flash("Đã xóa.", "success")
    return redirect(url_for("index"))


@app.route("/stats", methods=["GET"])
@login_required
def stats():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT result, COUNT(*) as cnt FROM patients GROUP BY result")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    counts = {"High Risk": 0, "Low Risk": 0}
    for r in rows:
        if r["result"] in counts:
            counts[r["result"]] = int(r["cnt"])

    return jsonify(counts=counts)


if __name__ == "__main__":
    app.run(debug=True)
