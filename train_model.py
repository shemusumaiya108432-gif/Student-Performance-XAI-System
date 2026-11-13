import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

np.random.seed(42)
n_samples = 800

# 1. Generate realistic student dataset (Quiz out of 15, Midterm out of 25)
data = pd.DataFrame({
    'attendance': np.random.uniform(45, 100, n_samples),
    'prev_cgpa': np.random.uniform(2.00, 4.00, n_samples),
    'study_hours': np.random.uniform(0.5, 8.0, n_samples),
    'social_media_hours': np.random.uniform(0.5, 7.0, n_samples),
    'tuition': np.random.choice([0, 1], size=n_samples),
    'quiz_avg': np.random.uniform(5.0, 15.0, n_samples).round(2),     # Out of 15
    'mid_score': np.random.uniform(8.0, 25.0, n_samples).round(2),    # Out of 25
    'teacher_rating': np.random.randint(1, 6, n_samples)              # 1 to 5
})

# 2. Feature Engineering
data['study_balance'] = data['study_hours'] - (data['social_media_hours'] * 0.4)

# 3. Target Variable Generation (Normalized to 15 & 25 scale)
raw_cgpa = (
    1.10 +
    0.008 * data['attendance'] +
    0.35 * data['prev_cgpa'] +
    0.07 * data['study_hours'] -
    0.04 * data['social_media_hours'] -
    0.08 * data['tuition'] +
    0.027 * data['quiz_avg'] +     # Scaled for max 15
    0.020 * data['mid_score'] +    # Scaled for max 25
    0.06 * data['teacher_rating'] +
    np.random.normal(0, 0.08, n_samples)
)
data['predicted_cgpa'] = raw_cgpa.clip(1.80, 4.00).round(2)

def assign_risk(cgpa):
    if cgpa >= 3.00:
        return 0
    elif cgpa >= 2.50:
        return 1
    else:
        return 2

data['risk_level'] = data['predicted_cgpa'].apply(assign_risk)

features = [
    'attendance', 'prev_cgpa', 'study_hours', 'social_media_hours',
    'tuition', 'quiz_avg', 'mid_score', 'teacher_rating', 'study_balance'
]
X = data[features]
y_cgpa = data['predicted_cgpa']
y_risk = data['risk_level']

X_train, X_test, y_cgpa_train, y_cgpa_test, y_risk_train, y_risk_test = train_test_split(
    X, y_cgpa, y_risk, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Regression Model
reg_model = RandomForestRegressor(n_estimators=150, random_state=42)
reg_model.fit(X_train_scaled, y_cgpa_train)

# 5. Classification Models Comparison
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42)
}

metrics_report = {}
best_clf = None

for name, clf in models.items():
    clf.fit(X_train_scaled, y_risk_train)
    preds = clf.predict(X_test_scaled)
    acc = round(accuracy_score(y_risk_test, preds) * 100, 2)
    f1 = round(f1_score(y_risk_test, preds, average='weighted') * 100, 2)
    cm = confusion_matrix(y_risk_test, preds).tolist()
    metrics_report[name] = {"Accuracy": acc, "F1_Score": f1, "Confusion_Matrix": cm}
    if name == "Random Forest":
        best_clf = clf

# 6. Save Dataset, Models, and Metrics
data.to_csv("student_data.csv", index=False)
joblib.dump(reg_model, "cgpa_model.pkl")
joblib.dump(best_clf, "risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")

with open("model_metrics.json", "w") as f:
    json.dump(metrics_report, f, indent=4)

print("Models re-trained with Quiz (out of 15) and Midterm (out of 25)!")