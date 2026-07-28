"""
AI-Based Manufacturing Efficiency Classification
Model Training Script
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import json

print("=" * 60)
print("Loading data...")
df = pd.read_csv('data/Thales_Group_Manufacturing.csv')

# ---------------------------------------------------------
# 1. DATA PREPROCESSING
# ---------------------------------------------------------
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
df = df.sort_values('DateTime').reset_index(drop=True)

df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# Encode categorical variable
op_mode_encoder = LabelEncoder()
df['Operation_Mode_Enc'] = op_mode_encoder.fit_transform(df['Operation_Mode'])

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
# Sensor stability indicator (lower = more stable)
df['Sensor_Stability'] = df['Vibration_Hz'] / (df['Temperature_C'] + 1)

# Energy efficiency ratio (output per unit power)
df['Energy_Efficiency_Ratio'] = df['Production_Speed_units_per_hr'] / (df['Power_Consumption_kW'] + 1)

# Error-to-output ratio
df['Error_to_Output_Ratio'] = df['Error_Rate_%'] / (df['Production_Speed_units_per_hr'] + 1)

# Network reliability score (higher = better network)
df['Network_Reliability_Score'] = 100 - (df['Network_Latency_ms'] * 0.5 + df['Packet_Loss_%'] * 10)

feature_cols = [
    'Machine_ID', 'Operation_Mode_Enc', 'Temperature_C', 'Vibration_Hz',
    'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
    'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
    'Predictive_Maintenance_Score', 'Error_Rate_%', 'Hour', 'DayOfWeek',
    'Sensor_Stability', 'Energy_Efficiency_Ratio', 'Error_to_Output_Ratio',
    'Network_Reliability_Score'
]

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df['Efficiency_Status'])
X = df[feature_cols]

print(f"Features: {len(feature_cols)}, Samples: {len(X)}")
print(f"Classes: {list(target_encoder.classes_)}")

# ---------------------------------------------------------
# 3. TRAIN-TEST SPLIT + SCALING
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. MODEL DEVELOPMENT
# ---------------------------------------------------------
results = {}

print("\nTraining Logistic Regression (baseline)...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr_pred),
    'f1_macro': f1_score(y_test, lr_pred, average='macro')
}

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'f1_macro': f1_score(y_test, rf_pred, average='macro')
}

print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                     random_state=42, eval_metric='mlogloss')
sample_weights = pd.Series(y_train).map(
    pd.Series(y_train).value_counts(normalize=True).apply(lambda x: 1/x)
).values
xgb.fit(X_train, y_train, sample_weight=sample_weights)
xgb_pred = xgb.predict(X_test)
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, xgb_pred),
    'f1_macro': f1_score(y_test, xgb_pred, average='macro')
}

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
for name, metrics in results.items():
    print(f"{name:25s} | Accuracy: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")

# Select best model by F1-macro (better for imbalanced classes)
best_model_name = max(results, key=lambda k: results[k]['f1_macro'])
print(f"\nBest model: {best_model_name}")

best_model = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}[best_model_name]
best_pred = {'Logistic Regression': lr_pred, 'Random Forest': rf_pred, 'XGBoost': xgb_pred}[best_model_name]

print("\nClassification Report (Best Model):")
report = classification_report(y_test, best_pred, target_names=target_encoder.classes_)
print(report)

# ---------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ---------------------------------------------------------
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv('outputs/feature_importance.csv', index=False)
    print("\nTop 5 important features:")
    print(importance_df.head())

# ---------------------------------------------------------
# 6. SAVE ARTIFACTS
# ---------------------------------------------------------
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(op_mode_encoder, 'models/op_mode_encoder.pkl')
joblib.dump(target_encoder, 'models/target_encoder.pkl')

with open('models/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

with open('outputs/model_results.json', 'w') as f:
    json.dump({
        'results': results,
        'best_model': best_model_name,
        'classification_report': report
    }, f, indent=2)

# Save processed dataframe for the Streamlit app (with predictions for historical view)
df['Predicted_Status'] = target_encoder.inverse_transform(
    best_model.predict(X) if best_model_name != 'Logistic Regression' else best_model.predict(scaler.transform(X))
)
df.to_csv('data/processed_data.csv', index=False)

print("\nAll artifacts saved successfully to models/ and outputs/")
