import os
import requests
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta


# ==========================================
# 1. LOAD API KEY
# ==========================================

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY not found. "
        "Please check your .env file."
    )


# ==========================================
# 2. KARACHI LOCATION
# ==========================================

lat = 24.9056
lon = 67.0822


# ==========================================
# 3. HISTORICAL TIME RANGE
# ==========================================

end_time = datetime.now(timezone.utc)

# Last 30 days
start_time = end_time - timedelta(days=30)

start = int(start_time.timestamp())
end = int(end_time.timestamp())


# ==========================================
# 4. OPENWEATHER API REQUEST
# ==========================================

url = "https://api.openweathermap.org/data/2.5/air_pollution/history"

params = {
    "lat": lat,
    "lon": lon,
    "start": start,
    "end": end,
    "appid": API_KEY
}

response = requests.get(
    url,
    params=params,
    timeout=30
)

print("Status code:", response.status_code)

response.raise_for_status()

pollution_data = response.json()


# ==========================================
# 5. EXTRACT POLLUTION DATA
# ==========================================

records = []

for item in pollution_data["list"]:

    components = item["components"]

    records.append({

        "timestamp": item["dt"],

        "openweather_aqi": item["main"]["aqi"],

        "co": components["co"],

        "no2": components["no2"],

        "o3": components["o3"],

        "so2": components["so2"],

        "pm2_5": components["pm2_5"],

        "pm10": components["pm10"],

        "nh3": components["nh3"]
    })


# ==========================================
# 6. CREATE DATAFRAME
# ==========================================

historical_df = pd.DataFrame(records)


# Convert timestamp to datetime
historical_df["timestamp"] = pd.to_datetime(
    historical_df["timestamp"],
    unit="s",
    utc=True
)


# Convert UTC time to Karachi time
historical_df["timestamp"] = historical_df[
    "timestamp"
].dt.tz_convert(
    "Asia/Karachi"
)


# Remove timezone information for CSV compatibility
historical_df["timestamp"] = historical_df[
    "timestamp"
].dt.tz_localize(None)


# Sort data
historical_df = historical_df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ==========================================
# 7. SAVE HISTORICAL DATA
# ==========================================

os.makedirs("data", exist_ok=True)

historical_df.to_csv(
    "data/historical_pollution.csv",
    index=False
)


# ==========================================
# 8. DATA VALIDATION / QUICK EDA
# ==========================================

print("\nHistorical pollution data saved successfully!")

print("\n===== DATA PREVIEW =====")
print(historical_df.head())


print("\n===== LATEST DATA =====")
print(historical_df.tail(1))


print("\n===== DATASET SHAPE =====")
print(historical_df.shape)


print("\nTotal records:", len(historical_df))


print("\n===== AQI VALUE COUNTS =====")
print(
    historical_df["openweather_aqi"]
    .value_counts()
    .sort_index()
)


print("\n===== PM2.5 RANGE =====")
print("Minimum:", historical_df["pm2_5"].min())
print("Maximum:", historical_df["pm2_5"].max())


print("\n===== PM10 RANGE =====")
print("Minimum:", historical_df["pm10"].min())
print("Maximum:", historical_df["pm10"].max())


print("\n===== MISSING VALUES =====")
print(
    historical_df.isnull().sum()
)


print("\n===== DUPLICATE RECORDS =====")
print(
    historical_df.duplicated().sum()
)


print("\nLatest timestamp:")
print(historical_df["timestamp"].max())