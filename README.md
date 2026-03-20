# Disease Prediction System (Flask + SQLite + AI + Gemini)

## Features
- Auth: Login/Register (Flask session) + role (`admin`/`user`)
- Prediction API: `POST /predict` (JSON) and form submit (HTML)
- Save history to SQLite (`predictions` table)
- Dashboard: `GET /dashboard-data` for Chart.js + recent predictions
- Model training: `train_model.py` (Logistic Regression vs Random Forest)
- **Gemini AI Integration**: Intelligent explanations and personalized recommendations in Vietnamese

## Tech stack
- Backend: Python Flask, SQLite
- ML: scikit-learn + joblib
- Frontend: Bootstrap 5 + Chart.js
- AI: Google Generative AI (Gemini) for enhanced explanations

## 1) Install dependencies
```bash
pip install -r requirements.txt
```

## 2) Setup SQLite
Project sẽ tự tạo DB khi bạn chạy `app.py` lần đầu (tự tạo file SQLite và các bảng).
- File DB mặc định: `disease_prediction.sqlite` (bạn có thể đổi bằng env `SQLITE_PATH`)

## 3) Train and save model
```bash
python train_model.py
```
Output:
- `models/model.pkl`
- `models/scaler.pkl`
- `models/metadata.json`

## 4) Setup Gemini (Optional but Recommended)
For AI-powered Vietnamese explanations:

1. Get free API key: https://aistudio.google.com/app/apikey
2. Set environment variables:
   ```bash
   set GEMINI_API_KEY=your-api-key-here
   set GEMINI_MODEL=gemini-1.5-flash
   set GEMINI_ENABLED=1
   ```

3. Test your setup:
   ```bash
   python test_gemini.py
   ```

**See [GEMINI_SETUP.md](GEMINI_SETUP.md) for detailed setup instructions and troubleshooting.**

## 5) Run web app
```bash
python app.py
```
Open:
 - `http://127.0.0.1:5000/login`

## 6) Login
- Register a user at `/register`
- (Optional admin export) set:
  - `ADMIN_USERNAME`
  - `ADMIN_PASSWORD`

## Gemini Integration
The app now includes advanced Gemini AI integration that:
- ✅ Generates intelligent Vietnamese explanations of predictions
- ✅ Provides personalized health recommendations
- ✅ Offers specific nutrition tips based on health metrics
- ✅ Falls back gracefully if API is unavailable
- ✅ Includes automatic retry logic and error handling

### Quick Setup
```bash
# 1. Get API key from https://aistudio.google.com/app/apikey
# 2. Set environment variable
set GEMINI_API_KEY=your-key-here

# 3. Test the integration
python test_gemini.py

# 4. Run the app
python app.py
```

### Key Features
- **Fast Responses**: Uses `gemini-1.5-flash` for speed and low cost
- **Vietnamese Content**: All explanations are in Vietnamese
- **Fallback Mode**: Works fine even if Gemini is disabled or API is down
- **Robust Parsing**: Handles various JSON response formats
- **Automatic Retries**: Built-in exponential backoff for rate limits

**For complete setup instructions, see [GEMINI_SETUP.md](GEMINI_SETUP.md)**

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

