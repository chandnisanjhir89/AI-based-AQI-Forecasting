# 🌍 AQI Predictor

## 📌 Project Overview

AQI Predictor is a Machine Learning project that predicts the Air Quality Index (AQI) for the next 3 days using historical air pollution and weather data.

The project collects weather and pollution data, performs feature engineering, analyzes AQI trends, trains machine learning models, and displays predictions through an interactive Streamlit dashboard.

---

## 🎯 Project Objectives

- Collect weather data using an API
- Collect air pollution and AQI data
- Create historical air quality data
- Perform feature engineering
- Analyze AQI and pollution trends
- Train and compare multiple machine learning models
- Explain model predictions using Feature Importance and SHAP
- Predict AQI for the next 3 days
- Visualize predictions using Streamlit

---

## 📊 Features Used

The machine learning model uses the following features:

- CO
- NO₂
- O₃
- SO₂
- PM2.5
- PM10
- NH3
- Hour
- Day
- Month
- AQI Change Rate

---

## 🤖 Models Trained

- Random Forest Regressor (best performing model)
- Ridge Regression
- Neural Network

Models are evaluated using **R² Score, RMSE, and MAE**, and results are stored in `model_results.csv`.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Plotly
- Streamlit
- Joblib
- Requests
- OpenWeather API

---

## 📁 Project Structure

```text
AQI-PREDICTOR/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── historical_pollution.csv
│   ├── aqi_predictions.csv
│   ├── model_results.csv
│   └── shap_summary.png
│
├── models/
│   └── best_aqi_model.pkl
│
├── src/
│   ├── fetch_data.py
│   ├── historical_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── .env
├── requirements.txt
└── README.md
```

---

## 🚧 Pending / Not Yet Implemented

- CI/CD automation (e.g., GitHub Actions) to run the feature pipeline hourly and training pipeline daily
- Feature Store / Model Registry (e.g., Hopsworks, Vertex AI)
- Detailed project report