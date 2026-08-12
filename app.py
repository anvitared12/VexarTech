import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
page_title="Vehicle Telemetry Dashboard",
    layout="wide"
)

st.title("Vehicle Telemetry Dashboard")
st.write("Analysis of vehicle trips, sensor behaviour and maintenance indicators")

telemetry = pd.read_excel("./Datasets/Telemetry.xlsx")
vehicles = pd.read_excel("./Datasets/Vehicles.xlsx")
trips = pd.read_excel("./Datasets/Trips.xlsx")

total_trips = telemetry["Trip_ID"].nunique()
total_vehicles = telemetry["Vehicle_ID"].nunique()
total_records = len(telemetry)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Trips",
    total_trips
)

col2.metric(
    "Total Vehicles",
    total_vehicles
)

col3.metric(
    "Telemetry Records",
    total_records
)


st.divider()



st.subheader("Telemetry Records per Trip")

trip_counts = (
    telemetry["Trip_ID"]
    .value_counts()
    .reset_index()
)

trip_counts.columns = ["Trip_ID", "Record_Count"]

st.bar_chart(
    trip_counts.set_index("Trip_ID")["Record_Count"]
)


st.subheader("Vehicle Analysis")

vehicle_list = sorted(
    telemetry["Vehicle_ID"].dropna().unique()
)

selected_vehicle = st.selectbox(
    "Select Vehicle",
    vehicle_list
)


vehicle_data = telemetry[
    telemetry["Vehicle_ID"] == selected_vehicle
]


st.write(
    f"Showing {len(vehicle_data)} telemetry records for vehicle {selected_vehicle}"
)



trip_list = sorted(
    vehicle_data["Trip_ID"].dropna().unique()
)

selected_trip = st.selectbox(
    "Select Trip",
    trip_list
)


trip_data = vehicle_data[
    vehicle_data["Trip_ID"] == selected_trip
]


st.write(
    f"Telemetry records for Trip {selected_trip}: {len(trip_data)}"
)


st.subheader("Acceleration")

accel_columns = [
    "Accel_X_g",
    "Accel_Y_g",
    "Accel_Z_g"
]

existing_accel = [
    col for col in accel_columns
    if col in trip_data.columns
]

if existing_accel:
    st.line_chart(
        trip_data.set_index("Timestamp")[existing_accel]
    )


st.subheader("Gyroscope")

gyro_columns = [
    "Gyro_X_dps",
    "Gyro_Y_dps",
    "Gyro_Z_dps"
]

existing_gyro = [
    col for col in gyro_columns
    if col in trip_data.columns
]

if existing_gyro:
    st.line_chart(
        trip_data.set_index("Timestamp")[existing_gyro]
    )