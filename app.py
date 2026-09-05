import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import timedelta


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

HISTORICAL_DATA_PATH = DATA_DIR / "historical_pollution.csv"
PREDICTION_DATA_PATH = DATA_DIR / "aqi_predictions.csv"
MODEL_PATH = MODEL_DIR / "best_aqi_model.pkl"
SHAP_PATH = DATA_DIR / "shap_summary.png"
RESULTS_PATH = DATA_DIR / "model_results.csv"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Karachi Air Quality Forecast",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1624;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}


/* ============================= */
/* METRIC NUMBERS */
/* ============================= */

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 38px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 16px !important;
}


/* ============================= */
/* TITLES */
/* ============================= */

.main-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 0px;
    color: white;
}

.subtitle {
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 25px;
}


/* ============================= */
/* POLLUTION CARDS */
/* ============================= */

.card {
    background-color: #172235;
    border: 1px solid #2b3b55;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.card-title {
    color: #94a3b8;
    font-size: 14px;
}

.card-value {
    font-size: 30px;
    font-weight: bold;
    color: white;
}

.card-unit {
    color: #64748b;
    font-size: 12px;
}


/* ============================= */
/* FORECAST CARDS */
/* ============================= */

.forecast-card {
    background-color: #172235;
    border: 1px solid #2b3b55;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    min-height: 180px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    historical_df = pd.read_csv(
        HISTORICAL_DATA_PATH
    )

    prediction_df = pd.read_csv(
        PREDICTION_DATA_PATH
    )

    historical_df["timestamp"] = pd.to_datetime(
        historical_df["timestamp"]
    )

    prediction_df["date"] = pd.to_datetime(
        prediction_df["date"]
    )

    historical_df = historical_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    prediction_df = prediction_df.sort_values(
        "date"
    ).reset_index(drop=True)

    return historical_df, prediction_df


try:

    historical_df, prediction_df = load_data()

except Exception as e:

    st.error(
        f"Could not load project data: {e}"
    )

    st.stop()


# ==========================================
# LATEST DATA
# ==========================================

latest = historical_df.iloc[-1]

current_aqi = float(
    latest["openweather_aqi"]
)

pm25 = float(
    latest["pm2_5"]
)

pm10 = float(
    latest["pm10"]
)

co = float(
    latest["co"]
)

no2 = float(
    latest["no2"]
)

o3 = float(
    latest["o3"]
)

so2 = float(
    latest["so2"]
)


# ==========================================
# AQI CATEGORY FUNCTION
# ==========================================

def get_aqi_category(aqi):

    if aqi <= 1:
        return "Good 🟢"

    elif aqi <= 2:
        return "Fair 🟡"

    elif aqi <= 3:
        return "Moderate 🟠"

    elif aqi <= 4:
        return "Poor 🔴"

    else:
        return "Very Poor 🟣"


aqi_category = get_aqi_category(
    current_aqi
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌍 AQI Predictor")

    st.divider()

    st.subheader("📍 Location")

    st.write("Karachi, Pakistan")

    st.divider()

    st.subheader("📊 Current AQI")

    st.metric(
        "OpenWeather AQI Scale",
        f"{current_aqi:.1f}"
    )

    st.write(
        aqi_category
    )

    st.divider()

    st.subheader("🔄 Data Source")

    st.success(
        "OpenWeather API"
    )

    st.caption(
        f"Last Updated:\n{latest['timestamp']}"
    )

    st.divider()

    st.subheader("🤖 Machine Learning")

    st.write(
        "Machine learning models trained on "
        "historical Karachi air pollution data."
    )


# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <div class="main-title">
    🌍 Karachi Air Quality Forecast
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-Powered Air Quality Prediction & Forecasting System
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# ALERT
# ==========================================

if current_aqi >= 4:

    st.error(
        "⚠️ Poor Air Quality! Sensitive individuals should "
        "reduce prolonged outdoor activity."
    )

elif current_aqi >= 3:

    st.warning(
        "⚠️ Moderate Air Quality. Sensitive individuals "
        "should take precautions."
    )

else:

    st.success(
        "✅ Current air quality conditions are acceptable."
    )


# ==========================================
# CURRENT AQI + GAUGE
# ==========================================

left, right = st.columns(
    [1, 1.5]
)


with left:

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=current_aqi,

            title={
                "text": "Current AQI"
            },

            gauge={

                "axis": {
                    "range": [1, 5]
                },

                "bar": {
                    "color": "#38bdf8"
                },

                "steps": [

                    {
                        "range": [1, 2],
                        "color": "#14532d"
                    },

                    {
                        "range": [2, 3],
                        "color": "#854d0e"
                    },

                    {
                        "range": [3, 4],
                        "color": "#7c2d12"
                    },

                    {
                        "range": [4, 5],
                        "color": "#7f1d1d"
                    }

                ]

            }

        )

    )

    gauge.update_layout(

        height=350,

        paper_bgcolor="#0e1624",

        font={
            "color": "white"
        }

    )

    st.plotly_chart(

        gauge,

        use_container_width=True

    )

    st.markdown(

        f"### Current Status: {aqi_category}"

    )


# ==========================================
# POLLUTANT CARDS
# ==========================================

with right:

    st.subheader(
        "🌫️ Current Air Pollution"
    )

    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(

            f"""
            <div class="card">

            <div class="card-title">
            PM2.5
            </div>

            <div class="card-value">
            {pm25:.2f}
            </div>

            <div class="card-unit">
            µg/m³
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    with c2:

        st.markdown(

            f"""
            <div class="card">

            <div class="card-title">
            PM10
            </div>

            <div class="card-value">
            {pm10:.2f}
            </div>

            <div class="card-unit">
            µg/m³
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    with c3:

        st.markdown(

            f"""
            <div class="card">

            <div class="card-title">
            CO
            </div>

            <div class="card-value">
            {co:.1f}
            </div>

            <div class="card-unit">
            µg/m³
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    st.write("")


    c4, c5, c6 = st.columns(3)


    with c4:

        st.markdown(

            f"""
            <div class="card">

            <div class="card-title">
            NO₂
            </div>

            <div class="card-value">
            {no2:.2f}
            </div>

            <div class="card-unit">
            µg/m³
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    with c5:

        st.markdown(

            f"""
            <div class="card">

            <div class="card-title">
            O₃
            </div>

            <div class="card-value">
            {o3:.2f}
            </div>

            <div class="card-unit">
            µg/m³
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    with c6:

        st.markdown(

            f"""
            <div class="card">

            <div class="card-title">
            SO₂
            </div>

            <div class="card-value">
            {so2:.2f}
            </div>

            <div class="card-unit">
            µg/m³
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


# ==========================================
# 3 DAY FORECAST
# CURRENT DAY + TOMORROW + DAY AFTER TOMORROW
# ==========================================

st.divider()

st.subheader("🔮 AI Forecast – Next 3 Days")

forecast_columns = st.columns(3)

for index, row in prediction_df.iterrows():

    if index >= 3:
        break

    forecast_aqi = float(row["predicted_aqi"])
    category = get_aqi_category(forecast_aqi)
    date_text = row["date"].strftime("%d %B")

    with forecast_columns[index]:

        st.markdown(
            f"""
            <div class="forecast-card">
            <h4>{date_text}</h4>
            <div class="card-value">
            {forecast_aqi:.2f}
            </div>
            <p>{category}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# HISTORICAL AQI + FORECAST
# ==========================================

st.divider()

st.subheader(
    "📈 Historical AQI Trend & AI Forecast"
)


fig = go.Figure()


fig.add_trace(

    go.Scatter(

        x=historical_df["timestamp"],

        y=historical_df["openweather_aqi"],

        mode="lines",

        name="Historical AQI"

    )

)


fig.add_trace(

    go.Scatter(

        x=prediction_df["date"],

        y=prediction_df["predicted_aqi"],

        mode="lines+markers",

        name="AI Forecast"

    )

)


fig.update_layout(

    height=450,

    paper_bgcolor="#0e1624",

    plot_bgcolor="#111827",

    font={
        "color": "white"
    },

    xaxis_title="Date",

    yaxis_title="AQI (OpenWeather Scale)"

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ==========================================
# POLLUTION TRENDS
# ==========================================

st.divider()

st.subheader(
    "🌫️ Historical Pollution Trends"
)


pollutant = st.selectbox(

    "Select Pollutant",

    [
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "o3",
        "so2"
    ]

)


pollutant_fig = go.Figure()


pollutant_fig.add_trace(

    go.Scatter(

        x=historical_df["timestamp"],

        y=historical_df[pollutant],

        mode="lines",

        name=pollutant.upper()

    )

)


pollutant_fig.update_layout(

    height=400,

    paper_bgcolor="#0e1624",

    plot_bgcolor="#111827",

    font={
        "color": "white"
    },

    xaxis_title="Date",

    yaxis_title=pollutant.upper()

)


st.plotly_chart(

    pollutant_fig,

    use_container_width=True

)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()

st.subheader(
    "🤖 Model Performance Comparison"
)


try:

    results_df = pd.read_csv(
        RESULTS_PATH
    )

    performance_columns = st.columns(
        len(results_df)
    )


    for index, row in results_df.iterrows():

        with performance_columns[index]:

            model_name = row["Model"]


            st.markdown(

                f"### 🤖 {model_name}"

            )


            st.metric(

                "R² Score",

                f"{float(row['R2']):.4f}"

            )


            st.metric(

                "RMSE",

                f"{float(row['RMSE']):.4f}"

            )


            st.metric(

                "MAE",

                f"{float(row['MAE']):.4f}"

            )


            if model_name.lower() == "random forest":

                st.success(

                    "🏆 Model Available for Prediction"

                )


except Exception as e:

    st.warning(

        f"Model performance results could not be loaded: {e}"

    )


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.divider()

st.subheader(
    "🔍 Model Explainability – Feature Importance"
)


st.info(

    "Feature importance shows which air pollutants and "
    "time-based features have the strongest influence "
    "on AQI predictions."

)


try:

    model = joblib.load(
        MODEL_PATH
    )


    # Get actual feature names
    if hasattr(
        model,
        "feature_names_in_"
    ):

        features = list(
            model.feature_names_in_
        )

    else:

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


    # Feature importance for tree models
    if hasattr(
        model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame({

            "Feature": features,

            "Importance":
                model.feature_importances_

        })


        importance_df = importance_df.sort_values(

            "Importance",

            ascending=True

        )


        importance_fig = go.Figure()


        importance_fig.add_trace(

            go.Bar(

                x=importance_df["Importance"],

                y=importance_df["Feature"],

                orientation="h"

            )

        )


        importance_fig.update_layout(

            height=450,

            paper_bgcolor="#0e1624",

            plot_bgcolor="#111827",

            font={
                "color": "white"
            },

            xaxis_title="Feature Importance",

            yaxis_title="Feature"

        )


        st.plotly_chart(

            importance_fig,

            use_container_width=True

        )


    else:

        st.info(

            "Feature importance is available only for "
            "tree-based models such as Random Forest."

        )


except Exception as e:

    st.warning(

        f"Feature importance could not be loaded: {e}"

    )


# ==========================================
# SHAP EXPLAINABILITY
# ==========================================

st.divider()

st.subheader(
    "🧠 SHAP Explainability"
)


st.info(

    "SHAP shows how each feature pushes "
    "the prediction higher or lower."

)


try:

    if SHAP_PATH.exists():

        st.image(

            str(SHAP_PATH),

            use_container_width=True

        )

    else:

        st.warning(

            "SHAP plot not found. "
            "Please run train_model.py again."

        )


except Exception as e:

    st.warning(

        f"SHAP plot could not be displayed: {e}"

    )


# ==========================================
# FOOTER
# ==========================================

st.divider()


st.caption(

    "Karachi AQI Predictor | "
    "Machine Learning & AI Project | "
    "Data Source: OpenWeather API"

)