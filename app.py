"""
AI-Based Manufacturing Efficiency Classification
Streamlit Dashboard Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Manufacturing Efficiency Classifier",
    page_icon="🏭",
    layout="wide"
)

# BASE always points to the folder this script lives in,
# so it works no matter where/how the app is deployed.
BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "models" / "best_model.pkl")
    scaler = joblib.load(BASE / "models" / "scaler.pkl")
    op_encoder = joblib.load(BASE / "models" / "op_mode_encoder.pkl")
    target_encoder = joblib.load(BASE / "models" / "target_encoder.pkl")
    with open(BASE / "models" / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, op_encoder, target_encoder, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data" / "processed_data.csv", parse_dates=["DateTime"])
    return df

@st.cache_data
def load_feature_importance():
    return pd.read_csv(BASE / "outputs" / "feature_importance.csv")

try:
    model, scaler, op_encoder, target_encoder, feature_cols = load_artifacts()
    df = load_data()
    importance_df = load_feature_importance()
except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e}\n\n"
        "Please make sure the `models/`, `data/`, and `outputs/` folders are uploaded "
        "alongside app.py in your deployment repository."
    )
    st.stop()

STATUS_COLORS = {"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"}

# ---------------------------------------------------------
# SIDEBAR — USER CAPABILITIES (filters/controls)
# ---------------------------------------------------------
st.sidebar.title("🏭 Controls")

machines = sorted(df["Machine_ID"].unique())
selected_machines = st.sidebar.multiselect("Machine selector", machines, default=machines[:10])

date_min, date_max = df["DateTime"].min(), df["DateTime"].max()
date_range = st.sidebar.slider(
    "Time window filter",
    min_value=date_min.to_pydatetime(),
    max_value=date_max.to_pydatetime(),
    value=(date_min.to_pydatetime(), date_max.to_pydatetime())
)

op_modes = df["Operation_Mode"].unique().tolist()
selected_op_modes = st.sidebar.multiselect("Operation mode dropdown", op_modes, default=op_modes)

network_quality = st.sidebar.select_slider(
    "Network quality filter",
    options=["Any", "Good (low latency & loss)", "Poor (high latency/loss)"],
    value="Any"
)

sensitivity = st.sidebar.slider(
    "Metric sensitivity (defect rate % threshold highlight)",
    min_value=0.0, max_value=15.0, value=7.5, step=0.5
)

# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------
mask = (
    df["Machine_ID"].isin(selected_machines if selected_machines else machines) &
    df["Operation_Mode"].isin(selected_op_modes if selected_op_modes else op_modes) &
    (df["DateTime"] >= date_range[0]) & (df["DateTime"] <= date_range[1])
)
if network_quality == "Good (low latency & loss)":
    mask &= (df["Network_Latency_ms"] < df["Network_Latency_ms"].median()) & \
            (df["Packet_Loss_%"] < df["Packet_Loss_%"].median())
elif network_quality == "Poor (high latency/loss)":
    mask &= (df["Network_Latency_ms"] >= df["Network_Latency_ms"].median()) | \
            (df["Packet_Loss_%"] >= df["Packet_Loss_%"].median())

fdf = df[mask].copy()

st.title("🏭 AI-Based Manufacturing Efficiency Classification")
st.caption("Real-time predictive classification using Sensor, Production, and 6G Network Data")

if fdf.empty:
    st.warning("No data matches the current filters. Please adjust the filters in the sidebar.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Efficiency Prediction Dashboard",
    "⚙️ Machine-Level Insights",
    "🔍 Explainability Panel",
    "📡 Operational Monitoring View"
])

# ===========================================================
# TAB 1: EFFICIENCY PREDICTION DASHBOARD
# ===========================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    status_counts = fdf["Efficiency_Status"].value_counts()
    col1.metric("Total Records", f"{len(fdf):,}")
    col2.metric("High Efficiency", f"{status_counts.get('High', 0):,}",
                f"{status_counts.get('High', 0)/len(fdf)*100:.1f}%")
    col3.metric("Medium Efficiency", f"{status_counts.get('Medium', 0):,}",
                f"{status_counts.get('Medium', 0)/len(fdf)*100:.1f}%")
    col4.metric("Low Efficiency", f"{status_counts.get('Low', 0):,}",
                f"{status_counts.get('Low', 0)/len(fdf)*100:.1f}%")

    st.subheader("Real-Time Efficiency Classification")
    left, right = st.columns([1, 1])
    with left:
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                     color=status_counts.index, color_discrete_map=STATUS_COLORS,
                     title="Efficiency Class Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        trend = fdf.set_index("DateTime").resample("H")["Efficiency_Status"].value_counts().unstack().fillna(0)
        fig2 = px.area(trend, title="Efficiency Status Over Time (Hourly)",
                        color_discrete_map=STATUS_COLORS)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🎯 Live Prediction Tool")
    st.write("Enter sensor/network/production readings to get a real-time efficiency prediction:")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_machine = st.selectbox("Machine ID", machines)
            in_op_mode = st.selectbox("Operation Mode", op_modes)
            in_temp = st.number_input("Temperature (°C)", value=float(df["Temperature_C"].mean()))
            in_vibration = st.number_input("Vibration (Hz)", value=float(df["Vibration_Hz"].mean()))
        with c2:
            in_power = st.number_input("Power Consumption (kW)", value=float(df["Power_Consumption_kW"].mean()))
            in_latency = st.number_input("Network Latency (ms)", value=float(df["Network_Latency_ms"].mean()))
            in_packet_loss = st.number_input("Packet Loss (%)", value=float(df["Packet_Loss_%"].mean()))
            in_defect = st.number_input("Quality Control Defect Rate (%)", value=float(df["Quality_Control_Defect_Rate_%"].mean()))
        with c3:
            in_speed = st.number_input("Production Speed (units/hr)", value=float(df["Production_Speed_units_per_hr"].mean()))
            in_maint = st.number_input("Predictive Maintenance Score", value=float(df["Predictive_Maintenance_Score"].mean()))
            in_error = st.number_input("Error Rate (%)", value=float(df["Error_Rate_%"].mean()))
            in_hour = st.slider("Hour of Day", 0, 23, 12)

        submitted = st.form_submit_button("🔮 Predict Efficiency")

    if submitted:
        sensor_stability = in_vibration / (in_temp + 1)
        energy_eff = in_speed / (in_power + 1)
        err_to_output = in_error / (in_speed + 1)
        net_reliability = 100 - (in_latency * 0.5 + in_packet_loss * 10)

        row = pd.DataFrame([{
            "Machine_ID": in_machine,
            "Operation_Mode_Enc": op_encoder.transform([in_op_mode])[0],
            "Temperature_C": in_temp, "Vibration_Hz": in_vibration,
            "Power_Consumption_kW": in_power, "Network_Latency_ms": in_latency,
            "Packet_Loss_%": in_packet_loss, "Quality_Control_Defect_Rate_%": in_defect,
            "Production_Speed_units_per_hr": in_speed, "Predictive_Maintenance_Score": in_maint,
            "Error_Rate_%": in_error, "Hour": in_hour, "DayOfWeek": 0,
            "Sensor_Stability": sensor_stability, "Energy_Efficiency_Ratio": energy_eff,
            "Error_to_Output_Ratio": err_to_output, "Network_Reliability_Score": net_reliability
        }])[feature_cols]

        pred = model.predict(row)[0]
        pred_label = target_encoder.inverse_transform([pred])[0]
        pred_proba = model.predict_proba(row)[0]
        confidence = max(pred_proba) * 100

        color = STATUS_COLORS[pred_label]
        st.markdown(f"### Prediction: <span style='color:{color}'>**{pred_label}**</span>", unsafe_allow_html=True)
        st.progress(confidence / 100)
        st.write(f"**Confidence score:** {confidence:.1f}%")

        proba_df = pd.DataFrame({"Class": target_encoder.classes_, "Probability": pred_proba})
        fig3 = px.bar(proba_df, x="Class", y="Probability", color="Class",
                      color_discrete_map=STATUS_COLORS, title="Prediction Confidence by Class")
        st.plotly_chart(fig3, use_container_width=True)

# ===========================================================
# TAB 2: MACHINE-LEVEL INSIGHTS
# ===========================================================
with tab2:
    st.subheader("Efficiency Trends per Machine")
    machine_eff = fdf.groupby(["Machine_ID", "Efficiency_Status"]).size().unstack(fill_value=0)
    machine_eff_pct = machine_eff.div(machine_eff.sum(axis=1), axis=0) * 100
    fig4 = px.bar(machine_eff_pct, barmode="stack", color_discrete_map=STATUS_COLORS,
                  title="Efficiency Status Share per Machine (%)")
    fig4.update_layout(xaxis_title="Machine ID", yaxis_title="% of records")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Historical Classification Patterns")
    selected_machine_detail = st.selectbox("Select a machine for detailed view", sorted(fdf["Machine_ID"].unique()))
    machine_df = fdf[fdf["Machine_ID"] == selected_machine_detail].sort_values("DateTime")

    fig5 = px.scatter(machine_df, x="DateTime", y="Production_Speed_units_per_hr",
                       color="Efficiency_Status", color_discrete_map=STATUS_COLORS,
                       title=f"Machine {selected_machine_detail}: Production Speed vs Efficiency Over Time")
    st.plotly_chart(fig5, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"Machine {selected_machine_detail} — Avg Defect Rate",
                   f"{machine_df['Quality_Control_Defect_Rate_%'].mean():.2f}%")
    with c2:
        st.metric(f"Machine {selected_machine_detail} — Avg Maintenance Score",
                   f"{machine_df['Predictive_Maintenance_Score'].mean():.2f}")

# ===========================================================
# TAB 3: EXPLAINABILITY PANEL
# ===========================================================
with tab3:
    st.subheader("Feature Importance — Key Drivers of Efficiency Status")
    fig6 = px.bar(importance_df.head(10), x="importance", y="feature", orientation="h",
                  title="Top 10 Most Important Features")
    fig6.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Why Efficiency Dropped or Improved")
    st.write(
        "Comparing average metric values across efficiency classes helps explain "
        "**why** a sample is classified as Low, Medium, or High."
    )
    compare_cols = ["Error_Rate_%", "Production_Speed_units_per_hr", "Quality_Control_Defect_Rate_%",
                     "Network_Latency_ms", "Predictive_Maintenance_Score"]
    comp = fdf.groupby("Efficiency_Status")[compare_cols].mean().reindex(["Low", "Medium", "High"])
    fig7 = px.bar(comp, barmode="group", title="Average Metric Values by Efficiency Class")
    st.plotly_chart(fig7, use_container_width=True)

    st.info(
        "**Interpretation:** Efficiency status is driven mainly by **Error Rate**, "
        "**Error-to-Output Ratio**, and **Production Speed**. High error rates combined "
        "with low production speed strongly indicate Low efficiency, building trust for "
        "engineers and operators reviewing predictions."
    )

# ===========================================================
# TAB 4: OPERATIONAL MONITORING VIEW
# ===========================================================
with tab4:
    st.subheader("Efficiency by Operation Mode")
    op_eff = fdf.groupby(["Operation_Mode", "Efficiency_Status"]).size().unstack(fill_value=0)
    fig8 = px.bar(op_eff, barmode="group", color_discrete_map=STATUS_COLORS,
                  title="Efficiency Status Count by Operation Mode")
    st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Network vs Sensor Impact Comparison")
    c1, c2 = st.columns(2)
    with c1:
        fig9 = px.scatter(fdf.sample(min(3000, len(fdf))), x="Network_Latency_ms", y="Error_Rate_%",
                          color="Efficiency_Status", color_discrete_map=STATUS_COLORS,
                          title="Network Latency vs Error Rate")
        st.plotly_chart(fig9, use_container_width=True)
    with c2:
        fig10 = px.scatter(fdf.sample(min(3000, len(fdf))), x="Vibration_Hz", y="Error_Rate_%",
                           color="Efficiency_Status", color_discrete_map=STATUS_COLORS,
                           title="Sensor Vibration vs Error Rate")
        st.plotly_chart(fig10, use_container_width=True)

    st.subheader(f"⚠️ High Defect-Rate Alerts (threshold: {sensitivity}%)")
    alerts = fdf[fdf["Quality_Control_Defect_Rate_%"] > sensitivity][
        ["DateTime", "Machine_ID", "Operation_Mode", "Quality_Control_Defect_Rate_%", "Efficiency_Status"]
    ].sort_values("DateTime", ascending=False).head(50)
    st.dataframe(alerts, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("AI-Based Manufacturing Efficiency Classification | Powered by Random Forest")
