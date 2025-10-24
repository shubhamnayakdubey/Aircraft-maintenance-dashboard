import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Aircraft Maintenance Dashboard", layout="wide")

@st.cache_data
def load_data():
    agg = pd.read_csv("outputs/aircraft_risk_summary.csv")
    records = pd.read_csv("outputs/maintenance_records_with_risk.csv")
    return agg, records

agg, records = load_data()

st.title("Aircraft Maintenance Analytics Dashboard")
st.markdown("Predictive maintenance insights using machine learning")

# KPIs
avg_risk = agg["avg_failure_prob"].mean()
high_risk_count = (agg["avg_failure_prob"] > 0.6).sum()
maintenance_events = agg["maintenance_events"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Avg Failure Probability", f"{avg_risk:.2f}")
col2.metric("High-Risk Aircraft", f"{high_risk_count}")
col3.metric("Total Maintenance Events", f"{maintenance_events}")

# Sidebar filter
st.sidebar.header("Filter Options")
risk_threshold = st.sidebar.slider("Failure Risk Threshold", 0.0, 1.0, 0.6, 0.05)
filtered = agg[agg["avg_failure_prob"] > risk_threshold]

# Table
st.subheader(f"Aircraft with Risk > {risk_threshold:.2f}")
st.dataframe(filtered.nlargest(10, "avg_failure_prob")[[
    "aircraft_id", "avg_failure_prob", "maintenance_events", "avg_time_since_maintenance"
]])

# Bar chart
fig, ax = plt.subplots(figsize=(8, 4))
top10 = agg.nlargest(10, "avg_failure_prob")
ax.bar(top10["aircraft_id"], top10["avg_failure_prob"], color="salmon")
ax.set_title("Top 10 Aircraft by Predicted Failure Risk")
ax.set_xlabel("Aircraft ID")
ax.set_ylabel("Failure Probability")
plt.xticks(rotation=45)
st.pyplot(fig)

# Download button
st.download_button(
    "Download Risk Summary CSV",
    data=agg.to_csv(index=False),
    file_name="aircraft_risk_summary.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("Predict Risk for a New Aircraft")

# Load trained Random Forest model
rf_model = joblib.load("outputs/random_forest_model.joblib")

# User inputs
st.markdown("Enter aircraft flight parameters below:")
flight_hours = st.number_input("Flight Hours", min_value=0.0, value=2.5, step=0.1)
cycles = st.number_input("Number of Cycles", min_value=1, max_value=10, value=2)
avg_temp = st.number_input("Average Temperature (°C)", value=15.0)
time_since_last_maintenance = st.number_input("Time Since Last Maintenance (days)", min_value=0, value=50)
sensor_vibration = st.number_input("Sensor Vibration", min_value=0.0, value=0.5, step=0.01)
oil_pressure = st.number_input("Oil Pressure", min_value=0.0, value=50.0, step=0.1)

# Prepare features for prediction
import numpy as np
new_data = np.array([[flight_hours, cycles, avg_temp,
                      time_since_last_maintenance, sensor_vibration, oil_pressure]])

# Predict risk probability
if st.button("Predict Failure Risk"):
    risk_prob = rf_model.predict_proba(new_data)[:,1][0]
    st.success(f"Predicted Probability of Failure within 30 Days: **{risk_prob:.2f}**")
    if risk_prob > 0.6:
        st.warning("High Risk! Consider immediate maintenance.")
    else:
        st.info("Risk is low for this aircraft at the moment.")
