# Disease Prediction (Refreshed)

Project da duoc lam moi theo combo:
- Logistic Regression (baseline)
- Random Forest (model chinh)
- SVM (model so sanh)

## Pipeline
1. Data: Pima Indians Diabetes dataset
2. Preprocessing: StandardScaler
3. Train: Logistic Regression, Random Forest, SVM
4. Evaluate: Accuracy, Precision, Recall
5. Compare: chon Random Forest lam model chinh de deploy

## Chay du an
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Mo trinh duyet:
- `http://127.0.0.1:5000`

## Model artifacts
Sau khi train xong se co:
- `models/random_forest.pkl`
- `models/logistic_regression.pkl`
- `models/svm.pkl`
- `models/scaler.pkl`
- `models/metadata.json`

## REST API
### POST `/predict`
Input JSON:
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

## Cau chot di thi
> Trong bai toan Disease Prediction, em su dung Logistic Regression lam baseline, Random Forest lam mo hinh chinh vi do chinh xac cao, va SVM de so sanh. Ngoai ra, em co su dung K-Means de phan tich du lieu va tham khao cac phuong phap khac nhu KNN, Naive Bayes va Neural Network.

