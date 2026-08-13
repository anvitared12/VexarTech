import os
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Vehicle Health Dashboard",layout="wide",)

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")

TIER_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}

def style_cell_colors(styler, color_map, subset):
    func = lambda v: f"background-color: {color_map.get(v, '')}; color: white; font-weight: bold;" if v in color_map else ""
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)

def load_from_folder(folder):
    df_vehicles = pd.read_excel(os.path.join(folder, "Vehicles.xlsx"))
    df_telemetry = pd.read_excel(os.path.join(folder, "Telemetry.xlsx"))
    trips_path = os.path.join(folder, "Trips.xlsx")
    df_trips = pd.read_excel(trips_path) if os.path.exists(trips_path) else None
    return df_vehicles, df_telemetry, df_trips


def load_from_uploads(files):
    df_vehicles = pd.read_excel(files["Vehicles"])
    df_telemetry = pd.read_excel(files["Telemetry"])
    df_trips = pd.read_excel(files["Trips"]) if files.get("Trips") else None
    return df_vehicles, df_telemetry, df_trips

@st.cache_data(show_spinner=False)
def run_pipeline(df_vehicles, df_telemetry):
    vehicles = df_vehicles.copy()
    telemetry = df_telemetry.copy()

    telemetry["Timestamp"] = pd.to_datetime(telemetry["Timestamp"])
    vehicles["Last_Service_Date"] = pd.to_datetime(vehicles["Last_Service_Date"])

    telemetry["Acceleration_Magnitude"] = np.sqrt(
        telemetry["Accel_X_g"] ** 2 + telemetry["Accel_Y_g"] ** 2 + telemetry["Accel_Z_g"] ** 2
    )
    telemetry["Gyro_Magnitude"] = np.sqrt(
        telemetry["Gyro_X_dps"] ** 2 + telemetry["Gyro_Y_dps"] ** 2 + telemetry["Gyro_Z_dps"] ** 2
    )

    accel_threshold = (
        telemetry["Acceleration_Magnitude"].mean() + 3 * telemetry["Acceleration_Magnitude"].std()
    )
    gyro_threshold = (
        telemetry["Gyro_Magnitude"].mean() + 3 * telemetry["Gyro_Magnitude"].std()
    )

    telemetry["Accel_Abnormal"] = telemetry["Acceleration_Magnitude"] > accel_threshold
    telemetry["Gyro_Abnormal"] = telemetry["Gyro_Magnitude"] > gyro_threshold
    telemetry["Abnormal"] = telemetry["Accel_Abnormal"] | telemetry["Gyro_Abnormal"]

    vehicle_sensor = telemetry.groupby("Vehicle_ID").agg(
        Total_Readings=("Vehicle_ID", "size"),
        Abnormal_Readings=("Abnormal", "sum"),
        Avg_Acceleration=("Acceleration_Magnitude", "mean"),
        Max_Acceleration=("Acceleration_Magnitude", "max"),
        Avg_Gyro=("Gyro_Magnitude", "mean"),
        Max_Gyro=("Gyro_Magnitude", "max"),
    ).reset_index()

    vehicle_sensor["Abnormal_Percentage"] = (
        vehicle_sensor["Abnormal_Readings"] / vehicle_sensor["Total_Readings"] * 100
    )

    result = vehicle_sensor.merge(vehicles, on="Vehicle_ID", how="left")

    current_year = datetime.now().year
    today = pd.Timestamp(datetime.now().date())

    result["Vehicle_Age"] = current_year - result["Manufacture_Year"]
    result["Days_Since_Service"] = (today - result["Last_Service_Date"]).dt.days

    return {
        "telemetry": telemetry,
        "vehicle_sensor": vehicle_sensor,
        "result": result,
        "accel_threshold": accel_threshold,
        "gyro_threshold": gyro_threshold,
    }


def compute_risk_score(row, abn_hi, abn_lo, age_hi, age_lo, odo_hi, odo_lo, svc_hi, svc_lo):
    score = 0

    if row["Abnormal_Percentage"] > abn_hi:
        score += 2
    elif row["Abnormal_Percentage"] > abn_lo:
        score += 1

    if row["Vehicle_Age"] >= age_hi:
        score += 2
    elif row["Vehicle_Age"] >= age_lo:
        score += 1

    odo_col = "Odometer_KM_Start_of_Week" if "Odometer_KM_Start_of_Week" in row.index else None
    if odo_col:
        if row[odo_col] >= odo_hi:
            score += 2
        elif row[odo_col] >= odo_lo:
            score += 1

    if row["Days_Since_Service"] >= svc_hi:
        score += 2
    elif row["Days_Since_Service"] >= svc_lo:
        score += 1

    return score

st.sidebar.title("Vehicle Health Dashboard")
st.sidebar.markdown("Source data (same files used in the notebook).")

data = None
if os.path.isdir(DATASET_DIR) and all(
    os.path.exists(os.path.join(DATASET_DIR, f"{name}.xlsx")) for name in ["Vehicles", "Telemetry"]
):
    st.sidebar.success("Using data found in ./Datasets")
    df_vehicles, df_telemetry, df_trips = load_from_folder(DATASET_DIR)
    data = run_pipeline(df_vehicles, df_telemetry)
else:
    st.sidebar.info("No ./Datasets folder found next to this app. Upload files below.")
    up_vehicles = st.sidebar.file_uploader("Vehicles.xlsx", type="xlsx")
    up_telemetry = st.sidebar.file_uploader("Telemetry.xlsx", type="xlsx")
    up_trips = st.sidebar.file_uploader("Trips.xlsx (optional)", type="xlsx")

    if up_vehicles and up_telemetry:
        df_vehicles, df_telemetry, df_trips = load_from_uploads({
            "Vehicles": up_vehicles, "Telemetry": up_telemetry, "Trips": up_trips,
        })
        data = run_pipeline(df_vehicles, df_telemetry)
    else:
        st.warning(
            "⬅️ Please provide Vehicles.xlsx and Telemetry.xlsx, either via a `Datasets/` "
            "folder next to `vehicle_app.py`, or by uploading them in the sidebar. "
            "Trips.xlsx is optional (used only for a bonus overview tab)."
        )
        st.stop()

result = data["result"]
telemetry = data["telemetry"]
accel_threshold = data["accel_threshold"]
gyro_threshold = data["gyro_threshold"]

ABNORMAL_PCT_HIGH = 5        # >5% abnormal readings -> +2 points
ABNORMAL_PCT_MEDIUM = 2      # >2% abnormal readings -> +1 point

VEHICLE_AGE_HIGH = 7         # age >= 7 years -> +2 points
VEHICLE_AGE_MEDIUM = 4       # age >= 4 years -> +1 point

ODOMETER_HIGH = 100_000      # odometer >= 100,000 km -> +2 points
ODOMETER_MEDIUM = 50_000     # odometer >= 50,000 km -> +1 point

DAYS_SINCE_SERVICE_HIGH = 180    # >= 180 days (~6 months) -> +2 points
DAYS_SINCE_SERVICE_MEDIUM = 90   # >= 90 days (~3 months) -> +1 point

RISK_SCORE_HIGH_CUTOFF = 5    # total score >= 5 -> "High" risk
RISK_SCORE_MEDIUM_CUTOFF = 2  # total score >= 2 -> "Medium" risk

result = result.copy()
result["Risk_Score"] = result.apply(
    lambda r: compute_risk_score(
        r,
        ABNORMAL_PCT_HIGH, ABNORMAL_PCT_MEDIUM,
        VEHICLE_AGE_HIGH, VEHICLE_AGE_MEDIUM,
        ODOMETER_HIGH, ODOMETER_MEDIUM,
        DAYS_SINCE_SERVICE_HIGH, DAYS_SINCE_SERVICE_MEDIUM,
    ),
    axis=1,
)


def assign_tier(score):
    if score >= RISK_SCORE_HIGH_CUTOFF:
        return "High"
    elif score >= RISK_SCORE_MEDIUM_CUTOFF:
        return "Medium"
    else:
        return "Low"


result["Maintenance_Risk"] = result["Risk_Score"].apply(assign_tier)

result = result.sort_values("Abnormal_Percentage", ascending=False).reset_index(drop=True)

st.title("Vehicle Health / Maintenance Risk Dashboard")
st.caption("Based on the analysis and risk-scoring logic from VehicleHealthStatus.ipynb")

vehicle_options = result["Vehicle_ID"].tolist()
selected_vehicle = st.selectbox("Select a vehicle", vehicle_options)

row = result[result["Vehicle_ID"] == selected_vehicle].iloc[0]
tier_color = TIER_COLORS.get(row["Maintenance_Risk"], "#95a5a6")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Maintenance Risk Score", f"{int(row['Risk_Score'])} / 8")
with col2:
    st.markdown(
        f"""<div style="padding:0.6em;border-radius:8px;background-color:{tier_color};
        text-align:center;color:white;font-weight:bold;font-size:1.1em;">
        {row['Maintenance_Risk'].upper()} RISK</div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.metric("Abnormal Readings", f"{row['Abnormal_Percentage']:.1f}%")
with col4:
    st.metric("Vehicle Age", f"{int(row['Vehicle_Age'])} yrs")

st.progress(min(max(row["Risk_Score"] / 8, 0.0), 1.0))

with st.expander("Vehicle details"):
    detail_cols = [c for c in [
        "Vehicle_ID", "Manufacture_Year", "Vehicle_Age", "Odometer_KM_Start_of_Week",
        "Last_Service_Date", "Days_Since_Service", "Total_Readings", "Abnormal_Readings",
        "Abnormal_Percentage", "Avg_Acceleration", "Max_Acceleration", "Avg_Gyro", "Max_Gyro",
        "Risk_Score",
    ] if c in row.index]
    st.dataframe(row[detail_cols].to_frame().T, use_container_width=True)

st.divider()

st.subheader("All Vehicles — Maintenance Risk")
show_cols = [c for c in [
    "Vehicle_ID", "Maintenance_Risk", "Risk_Score", "Abnormal_Percentage",
    "Vehicle_Age", "Odometer_KM_Start_of_Week", "Days_Since_Service",
] if c in result.columns]
st.dataframe(
    style_cell_colors(
        result[show_cols].style,
        TIER_COLORS,
        subset=["Maintenance_Risk"] if "Maintenance_Risk" in show_cols else [],
    ),
    use_container_width=True,
    height=300,
)

st.divider()

st.subheader("Graphs")

tabs = st.tabs(["Sensor Behavior", "Vehicle Profile", "Risk Composition", "Trips Overview"])

with tabs[0]:
    c1, c2 = st.columns(2)
    fig_accel = px.histogram(
        telemetry, x="Acceleration_Magnitude", nbins=40,
        title="Distribution of Acceleration Magnitude",
    )
    fig_accel.add_vline(x=accel_threshold, line_dash="dash", line_color="red",
                         annotation_text=f"Abnormal threshold ({accel_threshold:.2f})")
    c1.plotly_chart(fig_accel, use_container_width=True)

    fig_gyro = px.histogram(
        telemetry, x="Gyro_Magnitude", nbins=40,
        title="Distribution of Gyro Magnitude",
    )
    fig_gyro.add_vline(x=gyro_threshold, line_dash="dash", line_color="red",
                        annotation_text=f"Abnormal threshold ({gyro_threshold:.2f})")
    c2.plotly_chart(fig_gyro, use_container_width=True)

    fig_abn = px.bar(
        result.sort_values("Abnormal_Percentage"),
        x="Abnormal_Percentage", y="Vehicle_ID", orientation="h",
        color="Maintenance_Risk", color_discrete_map=TIER_COLORS,
        title="Abnormal Reading % by Vehicle",
    )
    st.plotly_chart(fig_abn, use_container_width=True)

with tabs[1]:
    c1, c2, c3 = st.columns(3)
    if "Vehicle_Age" in result.columns:
        fig_age = px.histogram(result, x="Vehicle_Age", nbins=15, title="Vehicle Age Distribution")
        c1.plotly_chart(fig_age, use_container_width=True)
    if "Odometer_KM_Start_of_Week" in result.columns:
        fig_odo = px.histogram(result, x="Odometer_KM_Start_of_Week", nbins=20, title="Odometer Distribution")
        c2.plotly_chart(fig_odo, use_container_width=True)
    if "Days_Since_Service" in result.columns:
        fig_svc = px.histogram(result, x="Days_Since_Service", nbins=20, title="Days Since Last Service")
        c3.plotly_chart(fig_svc, use_container_width=True)

with tabs[2]:
    c1, c2 = st.columns(2)
    tier_counts = result["Maintenance_Risk"].value_counts().reset_index()
    tier_counts.columns = ["Maintenance_Risk", "count"]
    fig_pie = px.pie(
        tier_counts, names="Maintenance_Risk", values="count",
        color="Maintenance_Risk", color_discrete_map=TIER_COLORS,
        title="Maintenance Risk Composition",
    )
    c1.plotly_chart(fig_pie, use_container_width=True)

    if "Odometer_KM_Start_of_Week" in result.columns:
        fig_scatter = px.scatter(
            result, x="Vehicle_Age", y="Abnormal_Percentage",
            size="Odometer_KM_Start_of_Week", color="Maintenance_Risk",
            color_discrete_map=TIER_COLORS, hover_data=["Vehicle_ID"],
            title="Vehicle Age vs Abnormal % (bubble size = Odometer)",
        )
    else:
        fig_scatter = px.scatter(
            result, x="Vehicle_Age", y="Abnormal_Percentage",
            color="Maintenance_Risk", color_discrete_map=TIER_COLORS,
            hover_data=["Vehicle_ID"],
            title="Vehicle Age vs Abnormal %",
        )
    c2.plotly_chart(fig_scatter, use_container_width=True)

with tabs[3]:
    if df_trips is not None:
        st.caption(
            "Trips data isn't part of the notebook's maintenance-risk score itself — "
            "it's shown here as extra context on vehicle usage."
        )
        c1, c2 = st.columns(2)
        if "Distance_KM" in df_trips.columns:
            fig_dist = px.histogram(df_trips, x="Distance_KM", nbins=30, title="Trip Distance Distribution")
            c1.plotly_chart(fig_dist, use_container_width=True)
        if "Duration_Min" in df_trips.columns:
            fig_dur = px.histogram(df_trips, x="Duration_Min", nbins=30, title="Trip Duration Distribution")
            c2.plotly_chart(fig_dur, use_container_width=True)
        if "Vehicle_ID" in df_trips.columns:
            trip_counts = df_trips.groupby("Vehicle_ID").size().reset_index(name="Trip_Count")
            fig_tc = px.bar(trip_counts.sort_values("Trip_Count"), x="Trip_Count", y="Vehicle_ID",
                             orientation="h", title="Trip Count by Vehicle")
            st.plotly_chart(fig_tc, use_container_width=True)
    else:
        st.info("Upload Trips.xlsx (optional) to see a usage overview here.")

st.divider()
st.caption(
    f"Risk score built from 4 weighted factors (abnormal %, age, odometer, days since service) · "
    f"Tiers: High ≥ {RISK_SCORE_HIGH_CUTOFF}, Medium ≥ {RISK_SCORE_MEDIUM_CUTOFF}, "
    f"Low < {RISK_SCORE_MEDIUM_CUTOFF} — fixed thresholds matching the original notebook"
)