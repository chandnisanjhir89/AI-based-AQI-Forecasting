import pandas as pd
import joblib
from datetime import timedelta
import os


# ==========================================
# 1. LOAD BEST MODEL
# ==========================================

model = joblib.load("models/best_aqi_model.pkl")

print("Best model loaded successfully!")


# ==========================================
# 2. LOAD SCALER (ONLY IF NEEDED)
# ==========================================

scaler = None

if os.path.exists("models/scaler.pkl"):
    scaler = joblib.load("models/scaler.pkl")
    print("Scaler loaded successfully!")


# ==========================================
# 3. LOAD LATEST HISTORICAL DATA
# ==========================================

df = pd.read_csv("data/historical_pollution.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort data by time
df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 4. GET LATEST RECORD
# ==========================================

latest_data = df.iloc[-1].copy()

latest_timestamp = latest_data["timestamp"]

print("Latest data timestamp:", latest_timestamp)


# ==========================================
# 5. FEATURES USED BY TRAINED MODEL
# ==========================================

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


# ==========================================
# 6. PREDICT NEXT 3 DAYS
# ==========================================

predictions = []

for days_ahead in range(1, 4):

    # Create future date
    future_date = latest_timestamp + timedelta(days=days_ahead)

    # Copy latest pollution data
    future_data = latest_data.copy()

    # Create future time features
    future_data["hour"] = future_date.hour
    future_data["day"] = future_date.day
    future_data["month"] = future_date.month


    # ==========================================
    # PREPARE MODEL INPUT
    # ==========================================

    X_future = pd.DataFrame(
        [[
            future_data["co"],
            future_data["no2"],
            future_data["o3"],
            future_data["so2"],
            future_data["pm2_5"],
            future_data["pm10"],
            future_data["nh3"],
            future_data["hour"],
            future_data["day"],
            future_data["month"]
        ]],
        columns=features
    )


    # ==========================================
    # SCALE DATA ONLY FOR NEURAL NETWORK
    # ==========================================

    if (
        scaler is not None
        and model.__class__.__name__ == "MLPRegressor"
    ):
        X_future_input = scaler.transform(X_future)

    else:
        X_future_input = X_future


    # ==========================================
    # PREDICT AQI
    # ==========================================

    predicted_aqi = model.predict(
        X_future_input
    )[0]


    # Save prediction
    predictions.append({
        "date": future_date.date(),
        "predicted_aqi": round(
            float(predicted_aqi),
            2
        )
    })


# ==========================================
# 7. CREATE PREDICTION DATAFRAME
# ==========================================

prediction_df = pd.DataFrame(
    predictions
)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print(
    "\n===== AQI PREDICTIONS FOR NEXT 3 DAYS ====="
)

print(
    prediction_df
)


# ==========================================
# 9. SAVE RESULTS
# ==========================================

prediction_df.to_csv(
    "data/aqi_predictions.csv",
    index=False
)

print(
    "\nPredictions saved successfully!"
)