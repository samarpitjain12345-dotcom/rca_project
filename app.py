import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3
from sklearn.ensemble import IsolationForest

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


REGIONS = ["North America", "Europe", "Asia"]
DEVICES = ["Mobile", "Desktop"]


def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Generate a 90-day synthetic business dataset with an intentional drop on day 80."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=90, freq="D")

    rows = []
    for date in dates:
        day_index = (date - dates[0]).days + 1
        for region in REGIONS:
            for device in DEVICES:
                base = rng.uniform(800, 1200)
                region_factor = {"North America": 1.2, "Europe": 1.0, "Asia": 0.9}[region]
                device_factor = 1.1 if device == "Mobile" else 1.0
                noise = rng.normal(0, 50)
                revenue = base * region_factor * device_factor + noise

                if day_index == 80 and region == "Asia" and device == "Mobile":
                    revenue *= 0.15

                rows.append(
                    {
                        "Date": date,
                        "Region": region,
                        "Device": device,
                        "Revenue": max(revenue, 0),
                    }
                )

    return pd.DataFrame(rows)


def detect_anomalies(daily_revenue: pd.DataFrame) -> pd.DataFrame:
    """Flag anomalous days using Isolation Forest on total daily revenue."""
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(daily_revenue[["Revenue"]].values)
    daily_revenue = daily_revenue.copy()
    daily_revenue["Is_Anomaly"] = predictions == -1
    return daily_revenue


def find_root_cause(raw_data: pd.DataFrame, anomaly_date: pd.Timestamp) -> dict:
    """Identify the segment with the largest negative variance vs. its 7-day baseline."""
    day_data = raw_data[raw_data["Date"] == anomaly_date]
    baseline_start = anomaly_date - pd.Timedelta(days=7)
    baseline_end = anomaly_date - pd.Timedelta(days=1)
    baseline = raw_data[
        (raw_data["Date"] >= baseline_start) & (raw_data["Date"] <= baseline_end)
    ]

    segment_cols = ["Region", "Device"]
    baseline_avg = (
        baseline.groupby(segment_cols)["Revenue"].mean().reset_index(name="Baseline_Revenue")
    )
    anomaly_segments = (
        day_data.groupby(segment_cols)["Revenue"].sum().reset_index(name="Anomaly_Revenue")
    )

    comparison = anomaly_segments.merge(baseline_avg, on=segment_cols, how="left")
    comparison["Baseline_Revenue"] = comparison["Baseline_Revenue"].fillna(
        comparison["Anomaly_Revenue"]
    )
    comparison["Variance"] = comparison["Anomaly_Revenue"] - comparison["Baseline_Revenue"]
    comparison["Pct_Change"] = (
        (comparison["Variance"] / comparison["Baseline_Revenue"].replace(0, np.nan)) * 100
    ).fillna(0)

    worst = comparison.loc[comparison["Variance"].idxmin()]
    return {
        "segment": worst,
        "comparison": comparison.sort_values("Variance"),
    }


def build_revenue_chart(daily_revenue: pd.DataFrame) -> go.Figure:
    """Plot daily revenue with anomalous days highlighted in red."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_revenue["Date"],
            y=daily_revenue["Revenue"],
            mode="lines+markers",
            name="Daily Revenue",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=6),
        )
    )

    anomalies = daily_revenue[daily_revenue["Is_Anomaly"]]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["Date"],
                y=anomalies["Revenue"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=12, symbol="x"),
            )
        )

    fig.update_layout(
        title="Daily Revenue with Anomaly Detection",
        xaxis_title="Date",
        yaxis_title="Total Revenue ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    return fig


def main():
    st.set_page_config(page_title="Automated RCA", page_icon="📊", layout="wide")
    st.title("📊 Automated Root Cause Analysis")
    st.markdown(
        "Detect revenue anomalies, identify the responsible segment, and generate an AI executive summary."
    )

    api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        help="Your API key is used only in this session and never stored.",
    )

    raw_data = generate_synthetic_data()
    daily_revenue = (
        raw_data.groupby("Date", as_index=False)["Revenue"].sum().sort_values("Date")
    )
    daily_revenue = detect_anomalies(daily_revenue)

    anomaly_days = daily_revenue[daily_revenue["Is_Anomaly"]]

    col1, col2, col3 = st.columns(3)
    col1.metric("Days Analyzed", len(daily_revenue))
    col2.metric("Anomalies Detected", len(anomaly_days))
    col3.metric(
        "Avg Daily Revenue",
        f"${daily_revenue['Revenue'].mean():,.0f}",
    )

    fig = build_revenue_chart(daily_revenue)
    st.plotly_chart(fig, use_container_width=True)

    if anomaly_days.empty:
        st.success("No statistical anomalies detected in daily revenue.")
        return

    primary_anomaly_date = anomaly_days.sort_values("Revenue").iloc[0]["Date"]
    root_cause = find_root_cause(raw_data, primary_anomaly_date)
    segment = root_cause["segment"]

    st.subheader("Root Cause Analysis")
    st.markdown(f"**Primary anomaly date:** {primary_anomaly_date.date()}")
    st.markdown(
        f"**Worst segment:** {segment['Region']} · {segment['Device']} "
        f"(${segment['Anomaly_Revenue']:,.0f} vs baseline ${segment['Baseline_Revenue']:,.0f}, "
        f"{segment['Pct_Change']:.1f}% change)"
    )

    with st.expander("Segment variance breakdown"):
        st.dataframe(
            root_cause["comparison"][
                ["Region", "Device", "Anomaly_Revenue", "Baseline_Revenue", "Variance", "Pct_Change"]
            ].rename(
                columns={
                    "Anomaly_Revenue": "Anomaly Day Revenue",
                    "Baseline_Revenue": "7-Day Baseline",
                    "Pct_Change": "% Change",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Executive Summary")
    if not api_key:
        st.warning("Enter your Google Gemini API key in the sidebar to generate the AI summary.")
        st.markdown(
            f"- Revenue anomaly detected on **{primary_anomaly_date.date()}**.\n"
            f"- Primary driver: **{segment['Region']} / {segment['Device']}** "
            f"with a **{segment['Pct_Change']:.1f}%** drop vs. baseline.\n"
            f"- Immediate investigation of Asia mobile channel recommended."
        )
    else:
        with st.spinner("Generating executive summary with Gemini..."):
            worst_segment_name = f"{segment['Region']} / {segment['Device']}"

            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }
            prompt = f"""
You are a Senior Data Analyst.
We detected a significant revenue anomaly.
The worst performing segment was {worst_segment_name} which dropped severely compared to the baseline.
Write a 3-bullet-point executive summary explaining this root cause and suggesting next steps.
"""
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
            }

            try:
                response = requests.post(url, headers=headers, json=payload, verify=False)
                response_data = response.json()

                if "candidates" in response_data:
                    ai_summary = response_data["candidates"][0]["content"]["parts"][0]["text"]
                    st.markdown(ai_summary)
                else:
                    st.error(f"API Error: {response_data}")
            except Exception as e:
                st.error(f"Connection Error: {e}")


if __name__ == "__main__":
    main()
