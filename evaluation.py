import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load compressed artifacts
model = joblib.load("models/stacking_reg_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load and split dataset
df = pd.read_csv("data/california_housing.csv")
X = df.drop("Price", axis=1)
y = df["Price"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_test_scaled = scaler.transform(X_test)

# Generate Stacking Predictions
stacking_pred = model.predict(X_test_scaled)

# Helper function to compute metrics
def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# Compute Stacking Metrics
stacking_metrics = compute_metrics(y_test, stacking_pred)

# Compare with Individual Base Learners
base_results = {}
for name, estimator in model.named_estimators_.items():
    base_pred = estimator.predict(X_test_scaled)
    base_results[name] = compute_metrics(y_test, base_pred)

# Display Comparison
print("=================== PERFORMANCE COMPARISON ===================")
for name, metrics in base_results.items():
    print(f"Base Learner: {name.upper()}")
    print(f"  MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f}")
print("--------------------------------------------------------------")
print("STACKING MODEL (Meta-Learner: Ridge)")
print(f"  MAE: {stacking_metrics['MAE']:.4f} | RMSE: {stacking_metrics['RMSE']:.4f} | R2: {stacking_metrics['R2']:.4f}")
print("==============================================================")

# Plot actual vs predicted values
plt.scatter(y_test, stacking_pred, alpha=0.2, color="mediumblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price ($100k)")
plt.ylabel("Predicted Price ($100k)")
plt.title("Stacking Regressor: Actual vs Predicted")
plt.show()