from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib
import os
import json

# Load the dataset
file_path = "./data/Exercise 5.csv"  # Update the path to your CSV file
data = pd.read_csv(file_path)

# Features and Targets
features = ['country', 'office.id', 'tariff.code', 'quantity', 
            'gross.weight', 'fob.value', 'cif.value', 'total.taxes']
target_illicit = 'illicit'
target_revenue = 'revenue'

# Encode categorical features
label_encoders = {}
for col in ['country', 'office.id']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature engineering
data['weight_per_quantity'] = data['gross.weight'] / (data['quantity'] + 1e-9)
data['value_per_weight'] = data['fob.value'] / (data['gross.weight'] + 1e-9)

# Update features list
features += ['weight_per_quantity', 'value_per_weight']

# Separate features and targets
X = data[features]
y_illicit = data[target_illicit]
y_revenue = data[target_revenue]

# Handle skewness in the target variable 'revenue'
y_revenue_log = np.log1p(y_revenue)

# Split the data into training and testing sets
X_train, X_test, y_illicit_train, y_illicit_test, y_revenue_train_log, y_revenue_test_log = train_test_split(
    X, y_illicit, y_revenue_log, test_size=0.2, random_state=42
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier for 'illicit'
clf_illicit = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf_illicit.fit(X_train, y_illicit_train)

# Predict 'illicit' on the test set
y_pred_illicit = clf_illicit.predict(X_test)

# Classification performance evaluation
classification_metrics = classification_report(y_illicit_test, y_pred_illicit, output_dict=True)

# Filter data for illicit trade predictions
X_test_illicit = X_test[y_pred_illicit == 1]
X_train_illicit = X_train[y_illicit_train == 1]
y_revenue_train_log_illicit = y_revenue_train_log[y_illicit_train == 1]

# Hyperparameter tuning for Random Forest Regressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train_illicit, y_revenue_train_log_illicit)

# Use the best model
reg_revenue = grid_search.best_estimator_

# Predict revenue for illicit trade
if X_test_illicit.shape[0] > 0:
    y_pred_revenue_log = reg_revenue.predict(X_test_illicit)
    y_pred_revenue = np.expm1(y_pred_revenue_log)  # Reverse the log transformation
else:
    y_pred_revenue = []

# Evaluate regression performance
if len(y_revenue_test_log[y_pred_illicit == 1]) > 0:
    y_true_revenue_illicit_log = y_revenue_test_log[y_pred_illicit == 1]
    mse = mean_squared_error(y_true_revenue_illicit_log, y_pred_revenue_log)
    r2 = r2_score(y_true_revenue_illicit_log, y_pred_revenue_log)
    regression_metrics = {
        "mean_squared_error": mse,
        "r2_score": r2
    }
else:
    regression_metrics = {
        "mean_squared_error": None,
        "r2_score": None,
        "note": "No illicit trade samples detected in the test set for regression evaluation."
    }

# Save evaluation metrics
evaluation_dir = "./evaluation"
os.makedirs(evaluation_dir, exist_ok=True)

classification_metrics_path = os.path.join(evaluation_dir, "classification_metrics.json")
with open(classification_metrics_path, "w") as f:
    json.dump(classification_metrics, f, indent=4)

regression_metrics_path = os.path.join(evaluation_dir, "regression_metrics.json")
with open(regression_metrics_path, "w") as f:
    json.dump(regression_metrics, f, indent=4)

print(f"Classification metrics saved to {classification_metrics_path}")
print(f"Regression metrics saved to {regression_metrics_path}")

# Save models and encoders
model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(clf_illicit, os.path.join(model_dir, "clf_illicit.joblib"))
joblib.dump(reg_revenue, os.path.join(model_dir, "reg_revenue.joblib"))
joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.joblib"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

print(f"Models, encoders, and scaler saved to {model_dir}.")