import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="F1 Analytics Dashboard",
    page_icon="🏎️",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("data/f1_results.csv")

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🏎️ Formula 1 Performance Analytics & Podium Prediction System")

st.info("""
This project collects Formula 1 data using the FastF1 API,
performs data analysis, trains a Machine Learning model,
and provides an interactive dashboard for race analytics
and podium prediction.
""")

# --------------------------------------------------
# SEASON FILTER
# --------------------------------------------------
st.subheader("📅 Season Filter")

selected_year = st.selectbox(
    "Select Season",
    sorted(df["Year"].unique())
)

df_filtered = df[df["Year"] == selected_year]

# --------------------------------------------------
# METRICS
# --------------------------------------------------
st.subheader("📈 Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df_filtered))

with col2:
    st.metric("Drivers", df_filtered["FullName"].nunique())

with col3:
    st.metric("Teams", df_filtered["TeamName"].nunique())

st.divider()

# --------------------------------------------------
# DRIVER RANKINGS
# --------------------------------------------------
st.subheader("🏆 Driver Points Ranking")

driver_points = (
    df_filtered.groupby("FullName")["Points"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(driver_points)

# --------------------------------------------------
# TOP 10 DRIVERS
# --------------------------------------------------
st.subheader("🏅 Top 10 Drivers")

top_drivers = (
    df_filtered.groupby("FullName")["Points"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    .head(10)
)

st.dataframe(top_drivers, use_container_width=True)

st.divider()

# --------------------------------------------------
# TEAM PERFORMANCE
# --------------------------------------------------
st.subheader("🏁 Team Performance")

team_points = (
    df_filtered.groupby("TeamName")["Points"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(team_points)

team_stats = (
    df_filtered.groupby("TeamName")["Points"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

st.dataframe(team_stats, use_container_width=True)

# --------------------------------------------------
# PIE CHART
# --------------------------------------------------
st.subheader("🥧 Team Points Distribution")

fig = px.pie(
    team_stats,
    names="TeamName",
    values="Points",
    title="Team Share of Total Points"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# DRIVER EXPLORER
# --------------------------------------------------
st.subheader("🔍 Driver Explorer")

selected_driver = st.selectbox(
    "Select Driver",
    sorted(df_filtered["FullName"].unique())
)

driver_df = df_filtered[
    df_filtered["FullName"] == selected_driver
]

st.dataframe(
    driver_df[
        [
            "Race",
            "TeamName",
            "GridPosition",
            "Position",
            "Points"
        ]
    ],
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# RACE RESULTS
# --------------------------------------------------
st.subheader("📊 Race Results")

selected_race = st.selectbox(
    "Select Race",
    sorted(df_filtered["Race"].unique())
)

race_df = df_filtered[
    df_filtered["Race"] == selected_race
]

st.dataframe(
    race_df[
        [
            "FullName",
            "TeamName",
            "GridPosition",
            "Position",
            "Points"
        ]
    ],
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# PODIUM FINISHES
# --------------------------------------------------
st.subheader("🥇 Podium Finishes")

podiums = df_filtered[df_filtered["Position"] <= 3]

podium_table = (
    podiums.groupby("FullName")
    .size()
    .reset_index(name="Podiums")
    .sort_values("Podiums", ascending=False)
)

st.dataframe(
    podium_table,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# BEST DRIVER
# --------------------------------------------------
st.subheader("👑 Driver of the Season")

best_driver = (
    df_filtered.groupby("FullName")["Points"]
    .sum()
    .idxmax()
)

best_points = (
    df_filtered.groupby("FullName")["Points"]
    .sum()
    .max()
)

st.success(
    f"Top Driver: {best_driver} ({best_points:.0f} points)"
)

st.divider()

# --------------------------------------------------
# MACHINE LEARNING SECTION
# --------------------------------------------------
st.subheader("🎯 Machine Learning Podium Prediction")

st.write("""
A Random Forest Classifier was trained to predict whether
a driver will finish on the podium (Top 3).

Features:
- Driver
- Team
- Grid Position
- Qualifying Performance

Target:
- Podium Finish (Yes / No)
""")

st.info("Model Used: Random Forest Classifier")

st.success("Model Trained Successfully")

st.divider()

# --------------------------------------------------
# FUTURE WORK
# --------------------------------------------------
st.subheader("🚀 Future Improvements")

st.write("""
- Include multiple seasons of historical data
- Add weather conditions
- Add tyre strategy analysis
- Predict race winners
- Predict championship standings
- Deploy dashboard online
""")

st.divider()

# --------------------------------------------------
# DOWNLOAD DATASET
# --------------------------------------------------
st.subheader("📥 Download Dataset")

st.download_button(
    label="Download F1 Dataset",
    data=df_filtered.to_csv(index=False),
    file_name="f1_results.csv",
    mime="text/csv"
)

st.divider()

# --------------------------------------------------
# PROJECT SUMMARY
# --------------------------------------------------
st.subheader("📌 Project Summary")

st.write("""
✅ Data collected using FastF1 API

✅ Data preprocessing using Pandas

✅ Exploratory Data Analysis

✅ Machine Learning with Random Forest

✅ Interactive Dashboard using Streamlit

✅ Driver Analytics

✅ Team Analytics

✅ Podium Prediction System
""")