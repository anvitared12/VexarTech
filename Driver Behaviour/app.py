import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Risky vs Safe Driver Dashboard",layout="wide",)

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datasets")

TIER_COLORS = {"Safe": "#2ecc71", "Risky": "#e74c3c"}

def load_from_folder(folder):
    df_drivers = pd.read_excel(os.path.join(folder, "Drivers.xlsx"))
    df_telemetry = pd.read_excel(os.path.join(folder, "Telemetry.xlsx"))
    df_trips = pd.read_excel(os.path.join(folder, "Trips.xlsx"))
    df_vehicles = pd.read_excel(os.path.join(folder, "Vehicles.xlsx"))
    return df_drivers, df_telemetry, df_trips, df_vehicles


def load_from_uploads(files):
    df_drivers = pd.read_excel(files["Drivers"])
    df_telemetry = pd.read_excel(files["Telemetry"])
    df_trips = pd.read_excel(files["Trips"])
    df_vehicles = pd.read_excel(files["Vehicles"])
    return df_drivers, df_telemetry, df_trips, df_vehicles

@st.cache_data(show_spinner=False)
def run_pipeline(df_drivers, df_telemetry, df_trips, df_vehicles):
    df_drivers = df_drivers.rename(columns={"Primary_Vehicle_ID": "Vehicle_ID"})
    driver_vehicles = df_drivers.merge(df_vehicles, on="Vehicle_ID")

    avg_experience = driver_vehicles["License_Experience_Years"].mean()
    driver_vehicles["Exp_Years_Flag"] = driver_vehicles["License_Experience_Years"].apply(
        lambda x: 0 if x >= avg_experience else 1
    )

    trips = df_trips.copy()
    trips["speed_gap"] = trips["Max_Speed_kmph"] - trips["Avg_Speed_kmph"]
    trips["trip_risky_flag"] = (trips["speed_gap"] > 30).astype(int)

    telemetry = df_telemetry.copy()
    telemetry = telemetry.drop(columns=[c for c in ["Latitude", "Longitude"] if c in telemetry.columns])

    x_flag = ((telemetry["Accel_X_g"] > 0.5) | (telemetry["Accel_X_g"] < -0.5)).astype(int)
    y_flag = ((telemetry["Accel_Y_g"] > 0.4) | (telemetry["Accel_Y_g"] < -0.6)).astype(int)
    z_flag = ((telemetry["Accel_Z_g"] > 1.5) | (telemetry["Accel_Z_g"] < 0.5)).astype(int)
    telemetry["Accel_flag"] = x_flag + y_flag + z_flag

    gx_flag = ((telemetry["Gyro_X_dps"] > 40) | (telemetry["Gyro_X_dps"] < -40)).astype(int)
    gy_flag = ((telemetry["Gyro_Y_dps"] > 25) | (telemetry["Gyro_Y_dps"] < -25)).astype(int)
    gz_flag = ((telemetry["Gyro_Z_dps"] > 35) | (telemetry["Gyro_Z_dps"] < -35)).astype(int)
    telemetry["Gyro_flag"] = gx_flag + gy_flag + gz_flag

    def consolidate_trips(df):
        trip_df = df.groupby("Trip_ID").agg(
            Driver_ID=("Driver_ID", "first"),
            Vehicle_ID=("Vehicle_ID", "first") if "Vehicle_ID" in df.columns else ("Driver_ID", "first"),
            n_readings=("Accel_flag", "size"),
            Accel_flag_events=("Accel_flag", lambda x: (x > 0).sum()),
            Gyro_flag_events=("Gyro_flag", lambda x: (x > 0).sum()),
        ).reset_index()

        trip_df["Accel_flag_rate"] = trip_df["Accel_flag_events"] / trip_df["n_readings"]
        trip_df["Gyro_flag_rate"] = trip_df["Gyro_flag_events"] / trip_df["n_readings"]
        trip_df["trip_risk_rate"] = (trip_df["Accel_flag_rate"] + trip_df["Gyro_flag_rate"]) / 2
        return trip_df

    def consolidate_drivers(trip_df):
        def weighted_rate(sub, rate_col):
            return (sub[rate_col] * sub["n_readings"]).sum() / sub["n_readings"].sum()

        rows = []
        for driver_id, sub in trip_df.groupby("Driver_ID"):
            rows.append({
                "Driver_ID": driver_id,
                "total_trips": sub["Trip_ID"].nunique(),
                "total_readings": sub["n_readings"].sum(),
                "Accel_flag_rate": weighted_rate(sub, "Accel_flag_rate"),
                "Gyro_flag_rate": weighted_rate(sub, "Gyro_flag_rate"),
                "avg_trip_risk_rate": weighted_rate(sub, "trip_risk_rate"),
                "worst_trip_risk_rate": sub["trip_risk_rate"].max(),
            })
        return pd.DataFrame(rows)

    def add_safety_score(driver_df):
        driver_df["safety_score"] = (1 - driver_df["avg_trip_risk_rate"]) * 100
        driver_df["safety_score"] = driver_df["safety_score"].clip(0, 100).round(1)
        return driver_df.sort_values("safety_score", ascending=False).reset_index(drop=True)

    trip_summary = consolidate_trips(telemetry)
    driver_summary = consolidate_drivers(trip_summary)
    driver_summary = add_safety_score(driver_summary)

    driver_info_cols = [c for c in ["Driver_ID", "Driver_Name", "Age", "License_Experience_Years", "Exp_Years_Flag"]
                         if c in driver_vehicles.columns]
    driver_summary = driver_summary.merge(driver_vehicles[driver_info_cols], on="Driver_ID", how="left")

    return {
        "driver_vehicles": driver_vehicles,
        "trips": trips,
        "telemetry": telemetry,
        "trip_summary": trip_summary,
        "driver_summary": driver_summary,
        "avg_experience": avg_experience,
    }


st.sidebar.title("Driver Risk Dashboard")
st.sidebar.markdown("Source data (same 4 files used in the notebook).")

data = None
if os.path.isdir(DATASET_DIR) and all(
    os.path.exists(os.path.join(DATASET_DIR, f"{name}.xlsx"))
    for name in ["Drivers", "Vehicles", "Trips", "Telemetry"]
):
    st.sidebar.success(f"Using data found in ./Datasets")
    df_drivers, df_telemetry, df_trips, df_vehicles = load_from_folder(DATASET_DIR)
    data = run_pipeline(df_drivers, df_telemetry, df_trips, df_vehicles)
else:
    st.sidebar.info("No ./Datasets folder found next to this app. Upload the 4 files below.")
    up_drivers = st.sidebar.file_uploader("Drivers.xlsx", type="xlsx")
    up_vehicles = st.sidebar.file_uploader("Vehicles.xlsx", type="xlsx")
    up_trips = st.sidebar.file_uploader("Trips.xlsx", type="xlsx")
    up_telemetry = st.sidebar.file_uploader("Telemetry.xlsx", type="xlsx")

    if up_drivers and up_vehicles and up_trips and up_telemetry:
        df_drivers, df_telemetry, df_trips, df_vehicles = load_from_uploads({
            "Drivers": up_drivers, "Vehicles": up_vehicles,
            "Trips": up_trips, "Telemetry": up_telemetry,
        })
        data = run_pipeline(df_drivers, df_telemetry, df_trips, df_vehicles)
    else:
        st.warning(
            "Please provide the 4 dataset files (Drivers, Vehicles, Trips, Telemetry) "
            "either via a `Datasets/` folder next to `app.py`, or by uploading them in the sidebar."
        )
        st.stop()

driver_summary = data["driver_summary"]
driver_vehicles = data["driver_vehicles"]
trips = data["trips"]
telemetry = data["telemetry"]

st.sidebar.divider()
st.sidebar.subheader("Risk Segregation")

THRESHOLD = 95.0
st.sidebar.info(f"Fixed Safety Score Threshold: **{THRESHOLD}** (Safe ≥ {THRESHOLD}, Risky < {THRESHOLD})")

def assign_tier(score):
    return "Safe" if score >= THRESHOLD else "Risky"

driver_summary["risk_tier"] = driver_summary["safety_score"].apply(assign_tier)

st.title("Risky vs Safe Driver Dashboard")
st.caption("Based on the analysis and safety-scoring logic from RiskyVsSafeDriver.ipynb")

name_lookup = driver_summary.set_index("Driver_ID")["Driver_Name"] if "Driver_Name" in driver_summary.columns else None

driver_options = driver_summary["Driver_ID"].tolist()
def fmt_driver(did):
    if name_lookup is not None and pd.notna(name_lookup.get(did)):
        return f"{name_lookup[did]} ({did})"
    return str(did)

selected_driver = st.selectbox("Select a driver", driver_options, format_func=fmt_driver)

row = driver_summary[driver_summary["Driver_ID"] == selected_driver].iloc[0]

col1, col2, col3, col4 = st.columns(4)
tier_color = TIER_COLORS.get(row["risk_tier"], "#95a5a6")

with col1:
    st.metric("Final Safety Score", f"{row['safety_score']:.1f} / 100")
with col2:
    st.markdown(
        f"""<div style="padding:0.6em;border-radius:8px;background-color:{tier_color};
        text-align:center;color:white;font-weight:bold;font-size:1.1em;">
        {row['risk_tier'].upper()}</div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.metric("Total Trips", int(row["total_trips"]))
with col4:
    st.metric("Total Telemetry Readings", int(row["total_readings"]))

st.progress(min(max(row["safety_score"] / 100, 0.0), 1.0))

with st.expander("Driver details"):
    detail_cols = [c for c in ["Driver_ID", "Driver_Name", "Age", "License_Experience_Years",
                                "Exp_Years_Flag", "Accel_flag_rate", "Gyro_flag_rate",
                                "avg_trip_risk_rate", "worst_trip_risk_rate"] if c in row.index]
    st.dataframe(row[detail_cols].to_frame().T, use_container_width=True)

st.divider()

st.subheader("All Drivers — Safety Scores")
show_cols = [c for c in ["Driver_ID", "Driver_Name", "safety_score", "risk_tier", "total_trips",
                          "Accel_flag_rate", "Gyro_flag_rate", "avg_trip_risk_rate"] if c in driver_summary.columns]
st.dataframe(
    driver_summary[show_cols].style.applymap(
        lambda v: f"background-color: {TIER_COLORS.get(v, '')}; color: white; font-weight: bold;"
        if v in TIER_COLORS else "",
        subset=["risk_tier"] if "risk_tier" in show_cols else [],
    ),
    use_container_width=True,
    height=300,
)

st.divider()

st.subheader("Graphs from the Notebook")

tab1, tab2, tab3, tab4 = st.tabs(["Driver Demographics", "Safety Scores", "Risk Composition", "Trips & Telemetry"])

with tab1:
    c1, c2 = st.columns(2)
    if "Age" in driver_vehicles.columns:
        fig_age = px.bar(
            driver_vehicles, x="Driver_Name", y="Age",
            title="Driver Name vs Age",
        )
        fig_age.update_xaxes(tickangle=90)
        c1.plotly_chart(fig_age, use_container_width=True)
        c1.caption("Ages range broadly — age alone doesn't appear to determine driving skill.")

    if "License_Experience_Years" in driver_vehicles.columns:
        fig_exp = px.bar(
            driver_vehicles, x="Driver_Name", y="License_Experience_Years",
            title="Driver Name vs License Experience",
        )
        fig_exp.update_xaxes(tickangle=90)
        fig_exp.add_hline(y=data["avg_experience"], line_dash="dash", line_color="red",
                           annotation_text=f"Average = {data['avg_experience']:.2f} yrs")
        c2.plotly_chart(fig_exp, use_container_width=True)
        c2.caption("Drivers below the average experience threshold are flagged (Exp_Years_Flag = 1).")

with tab2:
    scores_sorted = driver_summary.sort_values("safety_score").copy()
    scores_sorted["Driver_Label"] = scores_sorted["Driver_ID"].astype(str)
    if "Driver_Name" in scores_sorted.columns:
        scores_sorted["Driver_Label"] = scores_sorted["Driver_Name"].fillna(scores_sorted["Driver_Label"])

    fig_scores = px.bar(
        scores_sorted,
        x="safety_score", y="Driver_Label",
        color="risk_tier", color_discrete_map=TIER_COLORS,
        orientation="h",
        title="Safety Score by Driver",
        labels={"safety_score": "Safety Score", "Driver_Label": "Driver"},
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    fig_dist = px.histogram(
        driver_summary, x="safety_score", nbins=20,
        color="risk_tier", color_discrete_map=TIER_COLORS,
        title="Distribution of Safety Scores",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    tier_counts = driver_summary["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["risk_tier", "count"]
    fig_pie = px.pie(
        tier_counts, names="risk_tier", values="count",
        color="risk_tier", color_discrete_map=TIER_COLORS,
        title="Driver Risk Tier Composition",
    )
    c1.plotly_chart(fig_pie, use_container_width=True)

    fig_scatter = px.scatter(
        driver_summary, x="Accel_flag_rate", y="Gyro_flag_rate",
        color="risk_tier", color_discrete_map=TIER_COLORS,
        size="total_trips", hover_data=["Driver_ID"] + (["Driver_Name"] if "Driver_Name" in driver_summary else []),
        title="Accelerometer vs Gyroscope Flag Rate",
    )
    c2.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    risky_share = trips["trip_risky_flag"].value_counts().rename({0: "Not Risky", 1: "Risky"}).reset_index()
    risky_share.columns = ["trip_status", "count"]
    fig_trip = px.bar(
        risky_share, x="trip_status", y="count", color="trip_status",
        color_discrete_map={"Risky": TIER_COLORS["Risky"], "Not Risky": TIER_COLORS["Safe"]},
        title="Trips Flagged Risky (speed gap > 30 km/h)",
    )
    c1.plotly_chart(fig_trip, use_container_width=True)

    fig_speedgap = px.histogram(
        trips, x="speed_gap", nbins=30,
        title="Distribution of Trip Speed Gap (Max - Avg Speed)",
    )
    fig_speedgap.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Risk threshold (30)")
    c2.plotly_chart(fig_speedgap, use_container_width=True)

    st.markdown("**Accelerometer / Gyroscope flag counts across all telemetry readings**")
    flag_counts = pd.DataFrame({
        "flag_type": ["Accel_flag > 0", "Gyro_flag > 0"],
        "count": [(telemetry["Accel_flag"] > 0).sum(), (telemetry["Gyro_flag"] > 0).sum()],
    })
    fig_flags = px.bar(flag_counts, x="flag_type", y="count", title="Flagged Telemetry Readings")
    st.plotly_chart(fig_flags, use_container_width=True)

st.divider()
st.caption(
    f"Safety score = (1 − avg_trip_risk_rate) × 100 · "
    f"Tiers: Safe ≥ {THRESHOLD}, Risky < {THRESHOLD} (Fixed Threshold)"
)