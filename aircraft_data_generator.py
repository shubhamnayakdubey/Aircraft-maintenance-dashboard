import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Generate synthetic data
np.random.seed(42)
n_aircraft = 120
records_per_aircraft = 80
rows = []
start_date = datetime(2023, 1, 1)

for i in range(1, n_aircraft + 1):
    aid = f"AIR_{i:03d}"
    last_maintenance_days = np.random.randint(10, 180)
    for j in range(records_per_aircraft):
        flight_date = start_date + timedelta(days=np.random.randint(0, 365))
        flight_hours = max(0.5, np.random.normal(2.5, 0.7))
        cycles = np.random.randint(1, 4)
        avg_temp = np.random.normal(15, 10)
        time_since_last_maintenance = last_maintenance_days + np.random.randint(-5, 40)
        sensor_vibration = abs(np.random.normal(0.5, 0.2))
        oil_pressure = np.random.normal(50, 5)
        maintenance_flag = 1 if np.random.rand() < 0.03 else 0
        maintenance_cost = maintenance_flag * np.random.randint(1000, 25000)

        # Calculate risk score proxy
        risk_score = (
            0.4 * (flight_hours / 5.0)
            + 0.25 * (time_since_last_maintenance / 200.0)
            + 0.2 * (sensor_vibration / 1.5)
            + 0.15 * ((60 - oil_pressure) / 20.0)
        )
        fail_prob = 1 / (1 + np.exp(-3 * (risk_score - 0.3)))
        failure_within_30days = 1 if np.random.rand() < fail_prob else 0

        rows.append({
            "aircraft_id": aid,
            "flight_date": flight_date,
            "flight_hours": round(flight_hours, 2),
            "cycles": cycles,
            "avg_temp": round(avg_temp, 1),
            "time_since_last_maintenance": int(time_since_last_maintenance),
            "sensor_vibration": round(sensor_vibration, 3),
            "oil_pressure": round(oil_pressure, 2),
            "maintenance_flag": maintenance_flag,
            "maintenance_cost": maintenance_cost,
            "failure_within_30days": failure_within_30days
        })

df = pd.DataFrame(rows)
df.to_csv("outputs/maintenance_records.csv", index=False)
print("Data generated:", df.shape)

#Train predictive models
feature_cols = [
    "flight_hours", "cycles", "avg_temp",
    "time_since_last_maintenance", "sensor_vibration", "oil_pressure"
]
X = df[feature_cols]
y = df["failure_within_30days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Evaluate
lr_auc = roc_auc_score(y_test, y_proba_lr)
rf_auc = roc_auc_score(y_test, y_proba_rf)
lr_acc = accuracy_score(y_test, y_pred_lr)
rf_acc = accuracy_score(y_test, y_pred_rf)

metrics = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [lr_acc, rf_acc],
    "ROC_AUC": [lr_auc, rf_auc]
})
metrics.to_csv("outputs/model_metrics.csv", index=False)
print("Model metrics saved")

#Add risk score to all records
df["risk_score"] = rf.predict_proba(X)[:, 1]
df.to_csv("outputs/maintenance_records_with_risk.csv", index=False)

# Aggregate data for dashboard
agg = df.groupby("aircraft_id").agg(
    avg_flight_hours=("flight_hours", "mean"),
    avg_sensor_vibration=("sensor_vibration", "mean"),
    avg_oil_pressure=("oil_pressure", "mean"),
    avg_time_since_maintenance=("time_since_last_maintenance", "mean"),
    maintenance_events=("maintenance_flag", "sum"),
    avg_failure_prob=("risk_score", "mean")
).reset_index()
agg.to_csv("outputs/aircraft_risk_summary.csv", index=False)

# Save models
joblib.dump(rf, "outputs/random_forest_model.joblib")
joblib.dump(lr, "outputs/logistic_regression_model.joblib")
print("All files saved in 'outputs/' folder")
