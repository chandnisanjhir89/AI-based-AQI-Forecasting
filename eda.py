import pandas as pd
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv("data/training_data.csv")

# AQI distribution
plt.figure(figsize=(8, 5))

plt.hist(df["openweather_aqi"], bins=5)

plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.title("AQI Distribution")

plt.show()

# AQI trend over time

df["timestamp"] = pd.to_datetime(df["timestamp"])

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestamp"],
    df["openweather_aqi"],
    marker="o"
)

plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("AQI Trend Over Time")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
# Pollutant trends

pollutants = ["pm2_5", "pm10", "co", "no2", "o3", "so2"]

for pollutant in pollutants:
    plt.figure(figsize=(10, 5))

    plt.plot(
        df["timestamp"],
        df[pollutant]
    )

    plt.xlabel("Time")
    plt.ylabel(pollutant)
    plt.title(f"{pollutant} Trend Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()