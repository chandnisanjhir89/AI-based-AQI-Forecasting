import pandas as pd


# ==========================================
# 1. LOAD REAL HISTORICAL POLLUTION DATA
# ==========================================

df = pd.read_csv("data/historical_pollution.csv")

print("Historical pollution data loaded successfully!")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort data by time
df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. CREATE TIME-BASED FEATURES
# ==========================================

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

print("Time features created successfully!")


# ==========================================
# 3. CREATE AQI CHANGE RATE FEATURE
# ==========================================

# Difference between current AQI
# and previous hourly AQI value

df["aqi_change_rate"] = (
    df["openweather_aqi"].diff()
)

# First row has no previous value
df["aqi_change_rate"] = (
    df["aqi_change_rate"].fillna(0)
)

print("AQI change rate feature created successfully!")


# ==========================================
# 4. CREATE FUTURE AQI TARGETS
# ==========================================

# Historical data is hourly
# 24 records ≈ 1 day

df["aqi_next_1_day"] = (
    df["openweather_aqi"].shift(-24)
)

df["aqi_next_2_day"] = (
    df["openweather_aqi"].shift(-48)
)

df["aqi_next_3_day"] = (
    df["openweather_aqi"].shift(-72)
)

print("Future AQI targets created successfully!")


# ==========================================
# 5. REMOVE ROWS WITH MISSING FUTURE TARGETS
# ==========================================

df = df.dropna(
    subset=[
        "aqi_next_1_day",
        "aqi_next_2_day",
        "aqi_next_3_day"
    ]
).reset_index(drop=True)

print("Rows with missing future targets removed!")


# ==========================================
# 6. AQI DISTRIBUTION CHECK
# ==========================================

print("\n===== CURRENT AQI VALUE COUNTS =====")

print(
    df["openweather_aqi"]
    .value_counts()
    .sort_index()
)


print("\n===== NEXT DAY AQI VALUE COUNTS =====")

print(
    df["aqi_next_1_day"]
    .value_counts()
    .sort_index()
)


# ==========================================
# 7. FEATURES USED FOR MODEL TRAINING
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
# 8. FINAL TRAINING DATASET
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
# 9. SAVE TRAINING DATASET
# ==========================================

training_df.to_csv(

    "data/training_data.csv",

    index=False

)


print("\n===== TRAINING DATASET SAVED =====")

print(
    "Total training records:",
    len(training_df)
)

print(
    "Training dataset shape:",
    training_df.shape
)


# ==========================================
# 10. DATASET PREVIEW
# ==========================================

print("\n===== DATASET PREVIEW =====")

print(
    training_df.head()
)


# ==========================================
# 11. DATA QUALITY CHECKS
# ==========================================

print("\n===== MISSING VALUES =====")

print(
    training_df.isnull().sum()
)


print("\n===== DUPLICATE ROWS =====")

print(
    "Duplicate rows:",
    training_df.duplicated().sum()
)


print("\n===== DATASET STATISTICAL SUMMARY =====")

print(
    training_df.describe()
)


print("\nFeature engineering completed successfully!")