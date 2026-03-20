# Disease Prediction System (Flask + MySQL + AI)

## Features
- Auth: Login/Register (Flask session) + role (`admin`/`user`)
- Prediction API: `POST /predict` (JSON) and form submit (HTML)
- Save history to MySQL (`predictions` table)
- Dashboard: `GET /dashboard-data` for Chart.js + recent predictions
- Model training: `train_model.py` (Logistic Regression vs Random Forest)

## Tech stack
- Backend: Python Flask, MySQL
- ML: scikit-learn + joblib
- Frontend: Bootstrap 5 + Chart.js

## 1) Install dependencies
```bash
pip install -r requirements.txt
```

## 2) Setup MySQL
1. Create database schema:
   - Import `database/schema.sql` (recommended)  
   - or run `database.sql` from project root
2. Make sure MySQL connection works with these env vars (optional):
   - `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`

## 3) Train and save model
```bash
python train_model.py
```
Output:
- `models/model.pkl`
- `models/scaler.pkl`
- `models/metadata.json`

## 4) Run web app
```bash
python app.py
```
Open:
 - `http://127.0.0.1:5000/login`

## 5) Login
- Register a user at `/register`
- (Optional admin export) set:
  - `ADMIN_USERNAME`
  - `ADMIN_PASSWORD`

## REST API
### POST `/predict`
Send JSON:
```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 32.0,
  "DiabetesPedigreeFunction": 0.35,
  "Age": 30
}
```
Response:
```json
{
  "result": "Positive",
  "probability": 0.82,
  "risk": "High",
  "causes": [
    "Glucose của bạn đang cao so với khoảng dữ liệu huấn luyện. Thường liên quan tới khả năng kiểm soát đường huyết/kháng insulin.",
    "BMI của bạn đang cao so với khoảng dữ liệu huấn luyện. Thường làm tăng nguy cơ rối loạn chuyển hoá.",
    "Tuổi của bạn đang cao so với khoảng dữ liệu huấn luyện. Tuổi càng cao thì nguy cơ thường tăng."
  ],
  "recommendations": [
    "Nên cân nhắc khám/chẩn đoán và trao đổi xét nghiệm phù hợp (ví dụ HbA1c/đường huyết).",
    "Ưu tiên thay đổi lối sống: giảm đường tinh luyện, tăng vận động và duy trì cân nặng hợp lý.",
    "Kết quả dự đoán chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa."
  ],
  "nutrition_tips": [
    "Giảm đường và carb tinh chế; ưu tiên carb ít tinh chế và nhiều chất xơ.",
    "Nếu BMI cao: ưu tiên giảm cân bền vững với đạm nạc và rau.",
    "Nếu BP cao: hạn chế muối/đồ chế biến sẵn, tăng thực phẩm tươi."
  ],
  "disclaimer": "Lưu ý: nội dung giải thích/khuyến nghị dựa trên dữ liệu và mô hình (Pima Diabetes) + quy tắc tổng quát, chỉ dùng cho mục đích tham khảo demo."
}
```

### GET `/dashboard-data`
Used by dashboard chart:
```json
{
  "total": 10,
  "positive_count": 3,
  "positive_rate": 30.0,
  "risk_counts": {"Low": 2, "Medium": 5, "High": 3},
  "recent": [ ... ]
}
```

