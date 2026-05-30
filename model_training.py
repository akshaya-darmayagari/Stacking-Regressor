import os
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Load and save dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target
df.to_csv("data/california_housing.csv", index=False)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define Diverse Base Learners
base_learners = [
    ("dt", DecisionTreeRegressor(max_depth=6, random_state=42)),
    ("rf", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)),
    ("ada", AdaBoostRegressor(n_estimators=50, random_state=42))
]

# 4. Define Meta-Learner and Stacking Regressor
meta_learner = RidgeCV()
stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1,
    passthrough=False # If True, meta-learner also gets raw features
)

print("Training Stacking Regressor...")
stacking_model.fit(X_train_scaled, y_train)

# 5. Export compressed artifacts
joblib.dump(stacking_model, "models/stacking_reg_model.pkl", compress=3)
joblib.dump(scaler, "models/scaler.pkl", compress=3)

print("Stacking Regressor trained and saved successfully.")