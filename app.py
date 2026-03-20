import csv
import io
import json
import logging
import os
import sqlite3
from functools import wraps

import joblib
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
from gemini_client import call_gemini


app = Flask(__name__)
app.secret_key = cfg.SECRET_KEY

_DB_INIT_DONE = False


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(cfg.SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema():
    """
    Auto-create SQLite tables so you can run the project immediately.
    """
    global _DB_INIT_DONE
    if _DB_INIT_DONE:
        return

    conn = get_db_connection()
    cur = conn.cursor()

    # users
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # predictions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pregnancies INTEGER,
            glucose REAL,
            blood_pressure REAL,
            bmi REAL,
            age INTEGER,
            result TEXT,
            probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

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
        # If DB isn't ready, real requests will still fail on queries.
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


def init_admin_if_needed():
    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if not admin_username or not admin_password:
        return

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=?", (admin_username,))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO users(username, password_hash, role) VALUES (?,?,?)",
            (admin_username, generate_password_hash(admin_password), "admin"),
        )
        conn.commit()
        print("[info] Auto-created admin user from env.")

    cur.close()
    conn.close()


@app.before_request
def _init_admin():
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
            METADATA = json.loads(meta_path.read_text(encoding="utf-8"))
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
        return float(proba_all[-1])

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

    if METADATA == {} and SCALER is not None:
        try:
            x_scaled = SCALER.transform(x)
            return _positive_proba(MODEL, x_scaled)
        except Exception:
            pass

    return _positive_proba(MODEL, x)


def risk_from_probability(probability_01: float):
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
    if MODEL is None:
        return [0.0 for _ in cfg.FEATURES]

    if hasattr(MODEL, "feature_importances_"):
        scores = list(getattr(MODEL, "feature_importances_"))
        if len(scores) != len(cfg.FEATURES):
            return [0.0 for _ in cfg.FEATURES]
        return [float(x) for x in scores]

    if hasattr(MODEL, "coef_"):
        coef = np.array(getattr(MODEL, "coef_")).reshape(-1)
        if coef.shape[0] != len(cfg.FEATURES):
            return [0.0 for _ in cfg.FEATURES]
        return [float(abs(x)) for x in coef]

    return [0.0 for _ in cfg.FEATURES]


def top_factors_from_inputs(features: dict, top_n: int = 3):
    scores = get_model_feature_scores()
    ranked = sorted(range(len(cfg.FEATURES)), key=lambda i: scores[i], reverse=True)

    if all(s == 0.0 for s in scores):
        ranked = sorted(
            range(len(cfg.FEATURES)),
            key=lambda i: abs(
                features[cfg.FEATURES[i]] - (sum(cfg.RANGES[cfg.FEATURES[i]]) / 2.0)
            ),
            reverse=True,
        )

    chosen = ranked[:top_n]
    out = []
    bucket_vi = {"thap": "thấp", "trung binh": "trung bình", "cao": "cao"}

    for idx in chosen:
        fname = cfg.FEATURES[idx]
        val = float(features[fname])
        lo, hi = cfg.RANGES[fname]
        bucket_en = _feature_bucket(val, fname)
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
    relation = {
        "Pregnancies": "liên quan đến thay đổi chuyển hoá trong thai kỳ và các yếu tố sức khỏe lâu dài",
        "Glucose": "thường gắn với khả năng kiểm soát đường huyết/kháng insulin",
        "BloodPressure": "có thể đi kèm rối loạn chuyển hóa (hội chứng chuyển hóa)",
        "SkinThickness": "có thể phản ánh mức độ tích tụ mỡ dưới da",
        "Insulin": "có thể liên quan tới kháng insulin và chức năng tiết insulin",
        "BMI": "phản ánh thừa cân/béo phì, thường làm tăng nguy cơ rối loạn chuyển hoá",
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
        causes.append("Tổng hợp các chỉ số nổi bật làm mô hình dự đoán nguy cơ cao hơn.")
    elif risk == "Medium":
        causes.append("Một số chỉ số nổi bật làm mô hình dự đoán nguy cơ ở mức trung bình.")
    else:
        causes.append("Các chỉ số nổi bật nhìn chung không cho thấy rủi ro cao trong mô hình.")

    return causes


def build_recommendations(risk: str):
    if risk == "High":
        return [
            "Nên ưu tiên đi khám/chẩn đoán và trao đổi xét nghiệm phù hợp (ví dụ HbA1c/đường huyết).",
            "Ưu tiên thay đổi lối sống: giảm đường tinh luyện, giảm đồ chế biến sẵn, tăng vận động và duy trì cân nặng hợp lý.",
            "Lên lịch theo dõi định kỳ theo hướng dẫn chuyên môn.",
            "Kết quả dự đoán chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa.",
        ]
    if risk == "Medium":
        return [
            "Điều chỉnh chế độ ăn và tăng vận động để giảm nguy cơ.",
            "Theo dõi đường huyết/xét nghiệm theo lịch; cân nhắc can thiệp sớm hơn nếu chỉ số đang cao.",
            "Giữ thói quen sinh hoạt lành mạnh để cải thiện lâu dài.",
            "Kết quả dự đoán chỉ mang tính tham khảo.",
        ]
    return [
        "Duy trì lối sống lành mạnh: ăn cân bằng, hạn chế đường tinh chế, tăng chất xơ và vận động.",
        "Theo dõi sức khỏe định kỳ để phát hiện sớm thay đổi.",
        "Nếu có tiền sử gia đình hoặc triệu chứng, nên tư vấn bác sĩ để được cá nhân hóa.",
        "Kết quả dự đoán chỉ mang tính tham khảo.",
    ]


def build_nutrition_tips(features: dict, risk: str):
    tips = []

    glucose = float(features["Glucose"])
    bmi = float(features["BMI"])
    bp = float(features["BloodPressure"])

    if glucose >= 140:
        tips.append("Giảm đường và carb tinh chế: hạn chế nước ngọt/đồ ngọt; ưu tiên carb ít tinh chế và nhiều chất xơ.")
    else:
        tips.append("Duy trì kiểm soát carb: hạn chế đồ ngọt; chọn carb giàu chất xơ (rau, ngũ cốc nguyên hạt).")

    if bmi >= 30:
        tips.append("Nếu BMI cao: ưu tiên giảm cân bền vững; ưu tiên đạm nạc và rau, giảm năng lượng nạp phù hợp.")
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
    return (
        "Lưu ý: nội dung giải thích/khuyến nghị dựa trên dữ liệu và mô hình (Pima Diabetes) + quy tắc tổng quát, "
        "chỉ dùng cho mục đích tham khảo demo; không thay thế tư vấn y khoa."
    )


def store_prediction(features: dict, result: str, probability_01: float):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions(pregnancies, glucose, blood_pressure, bmi, age, result, probability)
        VALUES (?,?,?,?,?,?,?)
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
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total FROM predictions")
    total = int(cur.fetchone()["total"])

    cur.execute("SELECT COUNT(*) AS pos FROM predictions WHERE result='Positive'")
    pos = int(cur.fetchone()["pos"])

    cur.execute(
        """
        SELECT id, pregnancies, glucose, blood_pressure, bmi, age, result, probability, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
        """
    )
    recent = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT probability FROM predictions")
    probs = cur.fetchall()

    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    for row in probs:
        prob = float(row["probability"])
        risk, _ = risk_from_probability(prob)
        risk_counts[risk] += 1

    positive_rate = (pos / total * 100.0) if total else 0.0

    cur.close()
    conn.close()

    return {
        "total": total,
        "positive_count": pos,
        "positive_rate": positive_rate,
        "risk_counts": risk_counts,
        "recent": recent,
    }


def maybe_call_gemini(
    *,
    features: dict,
    risk: str,
    probability: float,
    result: str,
    causes: list,
    recommendations: list,
    nutrition_tips: list,
):
    """
    Optional: use Gemini to rewrite explanation/recommendations in Vietnamese.
    If Gemini fails or isn't configured, return {}.
    """
    if not cfg.GEMINI_ENABLED or not cfg.GEMINI_API_KEY:
        return {}

    # Format input data for the prompt
    causes_text = "\n".join(f"- {c}" for c in causes)
    recommendations_text = "\n".join(f"- {r}" for r in recommendations)
    nutrition_tips_text = "\n".join(f"- {t}" for t in nutrition_tips)
    disclaimer_text = build_disclaimer()

    # Refined prompt that encourages valid JSON output
    prompt = f"""Bạn là một trợ lý y khoa cung cấp thông tin tham khảo. Hãy tổng hợp và viết lại nội dung giải thích dự đoán bệnh dựa trên dữ liệu sau.

QUAN TRỌNG: Bạn PHẢI trả về kết quả là một JSON object hợp lệ với đúng 3 field: "explanation", "recommendations", "nutrition_tips". Mỗi field phải là string (cho "explanation") hoặc array của strings (cho recommendations/nutrition_tips).

Dữ liệu đầu vào:
- Kết quả dự đoán: {result}
- Mức độ nguy cơ: {risk}
- Xác suất: {probability:.2%}

Các yếu tố chính (Causes):
{causes_text}

Khuyến nghị từ hệ thống:
{recommendations_text}

Lời khuyên dinh dưỡng:
{nutrition_tips_text}

Lưu ý quan trọng: {disclaimer_text}

YÊU CẦU:
1. Viết ngắn gọn, rõ ràng bằng tiếng Việt dễ hiểu cho bệnh nhân/demo.
2. KHÔNG bịa đặt số liệu hoặc khẳng định chắc chắn thay bác sĩ.
3. Dựa trên thông tin đã cho để viết lại/tổng hợp thành lời giải thích mạch lạc.
4. Khuyến nghị có thể tương tự hoặc cải tiến từ danh sách sẵn có.
5. Lời khuyên dinh dưỡng nên cụ thể và áp dụng được.
6. PHẢI trả về kết quả dưới dạng JSON hợp lệ (không markdown, không text khác).

Output format (phải là JSON hợp lệ):
{{
  "explanation": "Giải thích ngắn gọn về kết quả dự đoán...",
  "recommendations": ["Khuyến nghị 1", "Khuyến nghị 2", "Khuyến nghị 3"],
  "nutrition_tips": ["Lời khuyên 1", "Lời khuyên 2", "Lời khuyên 3"]
}}"""

    try:
        resp = call_gemini(
            api_key=cfg.GEMINI_API_KEY,
            model=cfg.GEMINI_MODEL,
            prompt=prompt,
        )
        
        if not isinstance(resp, dict):
            return {}
        
        # Ensure response has expected structure
        if "explanation" not in resp:
            resp["explanation"] = ""
        if "recommendations" not in resp:
            resp["recommendations"] = recommendations
        if "nutrition_tips" not in resp:
            resp["nutrition_tips"] = nutrition_tips
            
        return resp
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in maybe_call_gemini: {e}", exc_info=True)
        return {}


# -------- Auth --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
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
                "INSERT INTO users(username, password_hash, role) VALUES (?,?,?)",
                (username, generate_password_hash(password), "user"),
            )
            conn.commit()
            flash("Register successful. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
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
    # Detect JSON
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

        gemini_explanation = ""
        gemini_payload = maybe_call_gemini(
            features=features,
            risk=risk,
            probability=prob,
            result=res,
            causes=causes,
            recommendations=recommendations,
            nutrition_tips=nutrition_tips,
        )
        
        if gemini_payload:
            # Extract explanation (prioritize Gemini's explanation if provided)
            gemini_explanation = gemini_payload.get("explanation", "") or ""
            
            # If Gemini couldn't match the expected JSON schema, show a helpful fallback
            if (not gemini_explanation) and gemini_payload.get("raw_text"):
                raw = str(gemini_payload.get("raw_text", ""))
                gemini_explanation = (
                    "Gemini trả về nội dung nhưng không parse được theo JSON schema đúng. "
                    f"Nội dung (rút gọn): {raw[:300]}"
                )
            
            # Update recommendations if Gemini provided better ones
            if isinstance(gemini_payload.get("recommendations"), list):
                recommendations = [str(r) for r in gemini_payload["recommendations"]]
            
            # Update nutrition tips if Gemini provided better ones
            if isinstance(gemini_payload.get("nutrition_tips"), list):
                nutrition_tips = [str(t) for t in gemini_payload["nutrition_tips"]]
        else:
            # Gemini enabled but failed -> show a helpful message for demo/debug
            if cfg.GEMINI_ENABLED and cfg.GEMINI_API_KEY:
                gemini_explanation = (
                    "Gemini chưa trả được kết quả phù hợp. Hệ thống vẫn dùng phần "
                    "giải thích/khuyến nghị theo rule-based."
                )

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
                        "gemini_explanation": gemini_explanation,
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
            gemini_explanation=gemini_explanation,
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
        return jsonify(get_dashboard_data())
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500


@app.route("/admin/export-csv", methods=["GET"])
@login_required_web
def export_csv():
    if session.get("role") != "admin":
        flash("Admin only.", "danger")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, pregnancies, glucose, blood_pressure, bmi, age, result, probability, created_at
        FROM predictions
        ORDER BY id DESC
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
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
    resp.headers[
        "Content-Disposition"
    ] = 'attachment; filename="predictions_history.csv"'
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)

