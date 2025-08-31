import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Energy Consumption Prediction", layout="wide")

st.title("⚡ Energy Consumption Prediction App")
st.markdown("### AICTE Internship Project – Sustainability & Efficiency")

# Upload dataset
uploaded_file = st.file_uploader("Upload your Household Energy Dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load dataset with proper separator
    try:
        data = pd.read_csv(uploaded_file, sep=";", low_memory=False)
    except:
        data = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset Preview")
    st.write(data.head())

    # Fix datetime
    if "Date" in data.columns and "Time" in data.columns:
        data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
        data.set_index('Datetime', inplace=True)
    
    # Convert numeric columns
    for col in ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()

    # Sidebar options
    st.sidebar.header("Navigation")
    section = st.sidebar.radio("Go to", ["Dataset Info", "EDA", "Model Training", "Prediction"])

    # Dataset Info
    if section == "Dataset Info":
        st.subheader("🔎 Dataset Information")
        st.write(data.describe())
        st.write("Shape:", data.shape)

    # EDA
    elif section == "EDA":
        st.subheader("📈 Exploratory Data Analysis")

        # Daily Consumption
        daily = data['Global_active_power'].resample('D').mean()
        fig, ax = plt.subplots(figsize=(12,6))
        daily.plot(ax=ax)
        ax.set_title("Daily Average Energy Consumption")
        st.pyplot(fig)

        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(data[['Global_active_power','Global_reactive_power','Voltage','Global_intensity']].corr(),
                    annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Model Training
    elif section == "Model Training":
        st.subheader("⚙️ Model Training - Linear Regression")

        X = data[['Global_reactive_power', 'Voltage', 'Global_intensity']]
        y = data['Global_active_power']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.write("**RMSE:**", rmse)
        st.write("**R² Score:**", r2)

        # Prediction Plot
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(y_test.values[:100], label="Actual", marker='o')
        ax.plot(y_pred[:100], label="Predicted", marker='x')
        ax.set_title("Energy Consumption Prediction (Sample 100 points)")
        ax.legend()
        st.pyplot(fig)

    # Prediction
    elif section == "Prediction":
        st.subheader("🔮 Make a Prediction")

        # User input
        col1, col2, col3 = st.columns(3)
        with col1:
            reactive = st.number_input("Global Reactive Power", min_value=0.0, step=0.1)
        with col2:
            voltage = st.number_input("Voltage", min_value=100.0, max_value=300.0, step=0.1)
        with col3:
            intensity = st.number_input("Global Intensity", min_value=0.0, step=0.1)

        # Train simple model
        X = data[['Global_reactive_power', 'Voltage', 'Global_intensity']]
        y = data['Global_active_power']
        model = LinearRegression().fit(X, y)

        if st.button("Predict Energy Consumption"):
            prediction = model.predict([[reactive, voltage, intensity]])
            st.success(f"Predicted Global Active Power: {prediction[0]:.3f} kilowatts")
