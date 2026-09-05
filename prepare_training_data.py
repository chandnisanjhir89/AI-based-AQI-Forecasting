import pandas as pd


# ==========================================
# 1. LOAD HISTORICAL DATA
# ==========================================

df = pd.read_csv("data/historical_pollution.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort data by time
df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. TIME-BASED FEATURES
# ==========================================

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

print("Time-based features created successfully!")


# ==========================================
# 3. DERIVED FEATURE: AQI CHANGE RATE
# ==========================================

df["aqi_change_rate"] = (
    df["openweather_aqi"].diff().fillna(0)
)

print("AQI change rate feature created successfully!")


# ==========================================
# 4. FUTURE AQI TARGETS
# ==========================================

# 24 hourly records ≈ 1 day
df["aqi_next_1_day"] = df["openweather_aqi"].shift(-24)

df["aqi_next_2_day"] = df["openweather_aqi"].shift(-48)

df["aqi_next_3_day"] = df["openweather_aqi"].shift(-72)


print("Future AQI targets created successfully!")


# Remove rows where future targets are missing
df = df.dropna(
    subset=[
        "aqi_next_1_day",
        "aqi_next_2_day",
        "aqi_next_3_day"
    ]
).reset_index(drop=True)


# ==========================================
# 5. FINAL FEATURE LIST
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

    "month",

    "aqi_change_rate"
]


# ==========================================
# 6. TRAINING DATASET COLUMNS
# ==========================================

training_columns = [

    "timestamp",

    "openweather_aqi",

    "aqi_next_1_day",

    "aqi_next_2_day",

    "aqi_next_3_day"

] + features


training_df = df[training_columns]


# ==========================================
# 7. SAVE TRAINING DATASET
# ==========================================

training_df.to_csv(
    "data/training_data.csv",
    index=False
)


print("\nTraining dataset saved successfully!")

print(
    "Total training records:",
    len(training_df)
)

print(
    "Training data shape:",
    training_df.shape
)


# ==========================================
# 8. BASIC EDA / SANITY CHECKS
# ==========================================

print("\n===== DATASET INFO =====")

training_df.info()


print("\n===== STATISTICAL SUMMARY =====")

print(
    training_df.describe()
)


print("\n===== MISSING VALUES =====")

print(
    training_df.isnull().sum()
)


print("\n===== DUPLICATES =====")

print(
    "Duplicate rows:",
    training_df.duplicated().sum()
)


print("\n===== DATASET PREVIEW =====")

print(
    training_df.head()
)