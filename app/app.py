import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Beijing Air Quality Analysis",
    page_icon="🌫️",
    layout="wide"
)


# ---------------------------------------------------------
# File Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_air_quality.csv"
MODEL_PATH = BASE_DIR / "models" / "pm25_random_forest_model.pkl"
STATION_ENCODER_PATH = BASE_DIR / "models" / "station_encoder.pkl"
WD_ENCODER_PATH = BASE_DIR / "models" / "wd_encoder.pkl"
SEASON_ENCODER_PATH = BASE_DIR / "models" / "season_encoder.pkl"


# ---------------------------------------------------------
# Load Data and Model
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    station_encoder = joblib.load(STATION_ENCODER_PATH)
    wd_encoder = joblib.load(WD_ENCODER_PATH)
    season_encoder = joblib.load(SEASON_ENCODER_PATH)
    return model, station_encoder, wd_encoder, season_encoder


df = load_data()
model, station_encoder, wd_encoder, season_encoder = load_model_files()


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Section",
        "Visualisation Section",
        "Model Output Section"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("**Selected Stations:**")
st.sidebar.write("Urban: Dongsi, Tiantan")
st.sidebar.write("Suburban: Huairou, Dingling")


# ---------------------------------------------------------
# Project Overview
# ---------------------------------------------------------
if page == "Project Overview":
    st.title("🌫️ Beijing Air Quality Analysis and PM2.5 Prediction")

    st.markdown("""
    This application presents an interactive analysis of Beijing air quality data from four monitoring stations.
    The selected stations include two urban stations and two suburban stations.

    The project includes:
    - Dataset exploration
    - Air pollution visualisation
    - Statistical analysis
    - PM2.5 prediction using a machine learning model
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Stations", df["station"].nunique())
    col4.metric("Date Range", f"{df['datetime'].dt.year.min()} - {df['datetime'].dt.year.max()}")

    st.subheader("Selected Monitoring Stations")

    station_type = pd.DataFrame({
        "Station": ["Dongsi", "Tiantan", "Huairou", "Dingling"],
        "Type": ["Urban", "Urban", "Suburban", "Suburban"]
    })

    st.dataframe(station_type, use_container_width=True)

    st.subheader("Dataset Variables")

    st.markdown("""
    **Pollutants:** PM2.5, PM10, SO2, NO2, CO, O3  
    **Meteorological Variables:** TEMP, PRES, DEWP, RAIN, WSPM, wd  
    **Time Variables:** year, month, day, hour, day_of_week, season
    """)


# ---------------------------------------------------------
# Dataset Section
# ---------------------------------------------------------
elif page == "Dataset Section":
    st.title("📊 Dataset Section")

    st.subheader("Dataset Preview")

    selected_station = st.multiselect(
        "Select Station",
        options=sorted(df["station"].unique()),
        default=sorted(df["station"].unique())
    )

    filtered_df = df[df["station"].isin(selected_station)]

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=filtered_df["datetime"].min().date()
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=filtered_df["datetime"].max().date()
        )

    filtered_df = filtered_df[
        (filtered_df["datetime"].dt.date >= start_date) &
        (filtered_df["datetime"].dt.date <= end_date)
    ]

    st.dataframe(filtered_df.head(100), use_container_width=True)

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{filtered_df.shape[0]:,}")
    col2.metric("Columns", filtered_df.shape[1])
    col3.metric("Stations Selected", filtered_df["station"].nunique())

    st.subheader("Missing Values")

    missing_table = pd.DataFrame({
        "Missing Values": filtered_df.isnull().sum(),
        "Missing Percentage": (filtered_df.isnull().sum() / len(filtered_df)) * 100
    }).round(2)

    st.dataframe(missing_table, use_container_width=True)

    st.subheader("Statistical Summary")

    numeric_columns = [
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
    ]

    st.dataframe(filtered_df[numeric_columns].describe().round(2), use_container_width=True)


# ---------------------------------------------------------
# Visualisation Section
# ---------------------------------------------------------
elif page == "Visualisation Section":
    st.title("📈 Visualisation Section")

    numeric_columns = [
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
    ]

    selected_station = st.multiselect(
        "Select Station",
        options=sorted(df["station"].unique()),
        default=sorted(df["station"].unique())
    )

    visual_df = df[df["station"].isin(selected_station)]

    st.subheader("Average PM2.5 by Station")

    fig, ax = plt.subplots(figsize=(10, 5))
    station_pm25 = visual_df.groupby("station")["PM2.5"].mean().sort_values(ascending=False)

    sns.barplot(x=station_pm25.index, y=station_pm25.values, ax=ax)
    ax.set_title("Average PM2.5 Concentration by Station")
    ax.set_xlabel("Station")
    ax.set_ylabel("Average PM2.5")
    st.pyplot(fig)

    st.subheader("Monthly Average PM2.5 Trend")

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_pm25 = visual_df.groupby("month")["PM2.5"].mean()

    sns.lineplot(x=monthly_pm25.index, y=monthly_pm25.values, marker="o", linewidth=2.5, ax=ax)
    ax.set_title("Monthly Average PM2.5 Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average PM2.5")
    ax.set_xticks(range(1, 13))
    st.pyplot(fig)

    st.subheader("Season-wise PM2.5 Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    season_order = ["Winter", "Spring", "Summer", "Autumn"]

    sns.boxplot(data=visual_df, x="season", y="PM2.5", order=season_order, ax=ax)
    ax.set_title("Season-wise PM2.5 Distribution")
    ax.set_xlabel("Season")
    ax.set_ylabel("PM2.5")
    st.pyplot(fig)

    st.subheader("Pollutant Distribution")

    selected_pollutant = st.selectbox(
        "Select Pollutant",
        options=["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(visual_df[selected_pollutant], bins=50, kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_pollutant}")
    ax.set_xlabel(selected_pollutant)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Scatter Plot Analysis")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Select X-axis Variable", options=numeric_columns, index=6)

    with col2:
        y_axis = st.selectbox("Select Y-axis Variable", options=numeric_columns, index=0)

    sample_df = visual_df.sample(min(5000, len(visual_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=sample_df, x=x_axis, y=y_axis, hue="station", alpha=0.6, ax=ax)
    ax.set_title(f"{y_axis} vs {x_axis}")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 8))
    corr_matrix = visual_df[numeric_columns].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


# ---------------------------------------------------------
# Model Output Section
# ---------------------------------------------------------
elif page == "Model Output Section":
    st.title("🤖 Model Output Section")

    st.markdown("""
    This section predicts **PM2.5 concentration** using the trained Random Forest Regression model.
    """)

    st.subheader("Enter Input Values for PM2.5 Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        station = st.selectbox("Station", sorted(df["station"].unique()))
        wd = st.selectbox("Wind Direction", sorted(df["wd"].unique()))
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
        month = st.slider("Month", 1, 12, 1)
        day = st.slider("Day", 1, 31, 1)
        hour = st.slider("Hour", 0, 23, 12)

    with col2:
        pm10 = st.number_input("PM10", min_value=0.0, value=80.0)
        so2 = st.number_input("SO2", min_value=0.0, value=10.0)
        no2 = st.number_input("NO2", min_value=0.0, value=40.0)
        co = st.number_input("CO", min_value=0.0, value=800.0)
        o3 = st.number_input("O3", min_value=0.0, value=50.0)

    with col3:
        temp = st.number_input("Temperature", value=15.0)
        pres = st.number_input("Pressure", value=1010.0)
        dewp = st.number_input("Dew Point", value=2.0)
        rain = st.number_input("Rain", min_value=0.0, value=0.0)
        wspm = st.number_input("Wind Speed", min_value=0.0, value=1.5)

    day_of_week = st.slider("Day of Week", 0, 6, 0)

    station_encoded = station_encoder.transform([station])[0]
    wd_encoded = wd_encoder.transform([wd])[0]
    season_encoded = season_encoder.transform([season])[0]

    input_data = pd.DataFrame({
        "PM10": [pm10],
        "SO2": [so2],
        "NO2": [no2],
        "CO": [co],
        "O3": [o3],
        "TEMP": [temp],
        "PRES": [pres],
        "DEWP": [dewp],
        "RAIN": [rain],
        "WSPM": [wspm],
        "year": [2017],
        "month": [month],
        "day": [day],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "station_encoded": [station_encoded],
        "wd_encoded": [wd_encoded],
        "season_encoded": [season_encoded]
    })

    if st.button("Predict PM2.5"):
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted PM2.5 Concentration: {prediction:.2f}")

        if prediction <= 35:
            category = "Good"
        elif prediction <= 75:
            category = "Moderate"
        elif prediction <= 115:
            category = "Unhealthy"
        elif prediction <= 150:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"

        st.info(f"Predicted Air Quality Category: {category}")

    st.subheader("Model Information")

    st.markdown("""
    **Model Used:** Random Forest Regressor  
    **Target Variable:** PM2.5  
    **Purpose:** Predict PM2.5 concentration using pollutant, meteorological, temporal and station-based features.
    """)

    performance_path = BASE_DIR / "data" / "processed" / "model_performance_results.csv"
    feature_path = BASE_DIR / "data" / "processed" / "feature_importance.csv"

    if performance_path.exists():
        st.subheader("Model Performance Results")
        performance_df = pd.read_csv(performance_path)
        st.dataframe(performance_df.round(4), use_container_width=True)

    if feature_path.exists():
        st.subheader("Feature Importance")

        feature_df = pd.read_csv(feature_path)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_df, x="Importance", y="Feature", ax=ax)
        ax.set_title("Random Forest Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)