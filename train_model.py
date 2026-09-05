import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import shap
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================

df = pd.read_csv("data/training_data.csv")

print("Training data loaded successfully!")

# Features
features = [
    "co",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
    "hour",
    "day",
    "month"
]

# Target: AQI after 1 day
target = "aqi_next_1_day"

X = df[features]
y = df[target]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\n===== TARGET AQI DISTRIBUTION =====")
print(y.value_counts().sort_index())


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split completed!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ==========================================
# 3. RANDOM FOREST MODEL
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== RANDOM FOREST RESULTS =====")
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)
print("R²:", rf_r2)


# ==========================================
# 4. RIDGE REGRESSION MODEL
# ==========================================

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\n===== RIDGE REGRESSION RESULTS =====")
print("RMSE:", ridge_rmse)
print("MAE:", ridge_mae)
print("R²:", ridge_r2)


# ==========================================
# 5. NEURAL NETWORK MODEL (Advanced)
# ==========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=2000,
    random_state=42
)

mlp_model.fit(X_train_scaled, y_train)

mlp_pred = mlp_model.predict(X_test_scaled)

mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
mlp_mae = mean_absolute_error(y_test, mlp_pred)
mlp_r2 = r2_score(y_test, mlp_pred)

print("\n===== NEURAL NETWORK RESULTS =====")
print("RMSE:", mlp_rmse)
print("MAE:", mlp_mae)
print("R²:", mlp_r2)


# ==========================================
# 6. MODEL COMPARISON
# ==========================================

print("\n===== MODEL COMPARISON =====")

print("\nRandom Forest")
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)
print("R²:", rf_r2)

print("\nRidge Regression")
print("RMSE:", ridge_rmse)
print("MAE:", ridge_mae)
print("R²:", ridge_r2)

print("\nNeural Network")
print("RMSE:", mlp_rmse)
print("MAE:", mlp_mae)
print("R²:", mlp_r2)


# ==========================================
# 7. SELECT BEST MODEL
# ==========================================

models_comparison = {
    "Random Forest": (rf_model, rf_rmse, rf_mae, rf_r2),
    "Ridge Regression": (ridge_model, ridge_rmse, ridge_mae, ridge_r2),
    "Neural Network": (mlp_model, mlp_rmse, mlp_mae, mlp_r2)
}

best_model_name = min(
    models_comparison,
    key=lambda name: models_comparison[name][1]
)

best_model, best_rmse, best_mae, best_r2 = models_comparison[best_model_name]

print("\n===== BEST MODEL =====")
print("Model:", best_model_name)
print("RMSE:", best_rmse)
print("MAE:", best_mae)
print("R²:", best_r2)


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_aqi_model.pkl"
)

print("\nBest model saved successfully!")

# Save scaler too (needed if Neural Network is best model)
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler saved successfully!")


# ==========================================
# 9. SAVE MODEL RESULTS
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Ridge Regression",
        "Neural Network"
    ],
    "RMSE": [rf_rmse, ridge_rmse, mlp_rmse],
    "MAE": [rf_mae, ridge_mae, mlp_mae],
    "R2": [rf_r2, ridge_r2, mlp_r2]
})

results.to_csv(
    "data/model_results.csv",
    index=False
)

print("Model results saved successfully!")


# ==========================================
# 10. SHAP EXPLAINABILITY
# ==========================================

try:
    if best_model_name == "Random Forest":
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)

    elif best_model_name == "Ridge Regression":
        explainer = shap.LinearExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)

    else:
        # Neural Network needs scaled data + KernelExplainer
        explainer = shap.KernelExplainer(best_model.predict, X_train_scaled[:50])
        shap_values = explainer.shap_values(X_test_scaled[:50])

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("data/shap_summary.png")
    plt.close()

    print("SHAP explainability plot saved!")

except Exception as e:
    print("SHAP could not be generated:", e)