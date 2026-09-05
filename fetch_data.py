import os
import requests
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime


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

print("API key loaded successfully!")


# ==========================================
# 2. FETCH CURRENT KARACHI WEATHER DATA
# ==========================================

weather_url = "https://api.openweathermap.org/data/2.5/weather"

weather_params = {
    "q": "Karachi",
    "appid": API_KEY,
    "units": "metric"
}

weather_response = requests.get(
    weather_url,
    params=weather_params,
    timeout=30
)

print("Weather status code:", weather_response.status_code)

weather_response.raise_for_status()

weather = weather_response.json()


# ==========================================
# 3. EXTRACT WEATHER DATA
# ==========================================

data = {

    "city": weather["name"],

    "temperature": weather["main"]["temp"],

    "humidity": weather["main"]["humidity"],

    "pressure": weather["main"]["pressure"],

    "wind_speed": weather["wind"]["speed"],

    "timestamp": datetime.now()

}

weather_df = pd.DataFrame([data])

os.makedirs("data", exist_ok=True)

weather_df.to_csv(
    "data/weather_data.csv",
    index=False
)

print("\nWeather data saved successfully!")

print(weather_df)


# ==========================================
# 4. GET KARACHI COORDINATES
# ==========================================

lat = weather["coord"]["lat"]
lon = weather["coord"]["lon"]

print("\nKarachi Coordinates:")
print("Latitude:", lat)
print("Longitude:", lon)


# ==========================================
# 5. FETCH CURRENT AIR POLLUTION DATA
# ==========================================

pollution_url = (
    "https://api.openweathermap.org/data/2.5/air_pollution"
)

pollution_params = {

    "lat": lat,

    "lon": lon,

    "appid": API_KEY

}

pollution_response = requests.get(
    pollution_url,
    params=pollution_params,
    timeout=30
)

print(
    "\nPollution status code:",
    pollution_response.status_code
)

pollution_response.raise_for_status()

pollution_data = pollution_response.json()


# ==========================================
# 6. EXTRACT POLLUTION DATA
# ==========================================

pollution_item = pollution_data["list"][0]

components = pollution_item["components"]

pollution_row = {

    "city": weather["name"],

    "co": components["co"],

    "no2": components["no2"],

    "o3": components["o3"],

    "so2": components["so2"],

    "pm2_5": components["pm2_5"],

    "pm10": components["pm10"],

    "nh3": components["nh3"],

    "openweather_aqi":
        pollution_item["main"]["aqi"],

    "timestamp": datetime.now()

}

pollution_df = pd.DataFrame(
    [pollution_row]
)

pollution_df.to_csv(
    "data/pollution_data.csv",
    index=False
)

print("\nPollution data saved successfully!")

print(pollution_df)


# ==========================================
# 7. COMBINE WEATHER + POLLUTION DATA
# ==========================================

combined_df = pd.concat(

    [

        weather_df,

        pollution_df.drop(
            columns=[
                "city",
                "timestamp"
            ]
        )

    ],

    axis=1

)


# ==========================================
# 8. SAVE COMBINED RAW DATA
# ==========================================

combined_df.to_csv(

    "data/aqi_raw_data.csv",

    index=False

)

print(
    "\nCombined dataset saved successfully!"
)

print("\n===== CURRENT KARACHI DATA =====")

print(combined_df)


# ==========================================
# 9. DATA VALIDATION
# ==========================================

print("\n===== DATASET INFO =====")

print(combined_df.info())


print("\n===== MISSING VALUES =====")

print(
    combined_df.isnull().sum()
)