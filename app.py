import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3
from sklearn.ensemble import IsolationForest

# 1. Page Config (Must be first)
st.set_page_config(page_title="AI Root Cause Engine", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# 2. Custom CSS for Enterprise Polish
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 0rem;}
        div[data-testid="stMetricValue"] {font-size: 2rem; color: #2962FF;}
    </style>
""", unsafe_allow_html=True)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REGIONS = ["North America", "Europe", "Asia"]
DEVICES = ["Mobile", "Desktop"]

# --- DATA PROCESSING & CACHING ---

@st.cache_data
def load_and_prep_data(file):
    df = pd.read_csv(file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
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
                
                # Intentional Anomaly
                if day_index == 80 and region == "Asia" and device == "Mobile":
                    revenue *= 0.15
                    
                rows.append({"Date": date, "Region": region, "Device": device, "Revenue": max(revenue, 0)})
    return pd.DataFrame(rows)

# --- ANALYSIS ENGINE ---

@st.cache_data
def detect_anomalies(daily_revenue: pd.DataFrame) -> pd.DataFrame:
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(daily_revenue[["Revenue"]].values)
    df_copy = daily_revenue.copy()
    df_copy["Is_Anomaly"] = predictions == -1
    return df_copy

@st.cache_data
def find_root_cause(raw_data: pd.DataFrame, anomaly_date: pd.Timestamp) -> dict:
    day_data = raw_data[raw_data["Date"] == anomaly_date]
    baseline_start = anomaly_date - pd.Timedelta(days=7)
    baseline_end = anomaly_date - pd.Timedelta(days=1)
    
    baseline = raw_data[(raw_data["Date"] >= baseline_start) & (raw_data["Date"] <= baseline_end)]
    segment_cols = ["Region", "Device"]
    
    baseline_avg = baseline.groupby(segment_cols)["Revenue"].mean().reset_index(name="Baseline_Revenue")
    anomaly_segments = day_data.groupby(segment_cols)["Revenue"].sum().reset_index(name="Anomaly_Revenue")
    
    comparison = anomaly_segments.merge(baseline_avg, on=segment_cols, how="left")
    comparison["Baseline_Revenue"] = comparison["Baseline_Revenue"].fillna(comparison["Anomaly_Revenue"])
    comparison["Variance"] = comparison["Anomaly_Revenue"] - comparison["Baseline_Revenue"]
    comparison["Pct_Change"] = (comparison["Variance"] / comparison["Baseline_Revenue"].replace(0, np.nan)) * 100
    comparison["Pct_Change"] = comparison["Pct_Change"].fillna(0)
    
    worst = comparison.loc[comparison["Variance"].idxmin()]
    return {"segment": worst, "comparison": comparison.sort_values("Variance")}

def generate_executive_summary(api_key: str, root_cause: dict, anomaly_date: pd.Timestamp) -> str:
    segment = root_cause["segment"]
    prompt = f"""You are an elite Data Scientist. A revenue anomaly was detected on {anomaly_date.date()}.
    The primary driver was the {segment['Region']} / {segment['Device']} segment, 
    which dropped by {segment['Pct_Change']:.1f}% (Variance: ${segment['Variance']:.2f}) vs its 7-day baseline.
    Write a highly professional, 3-bullet-point executive summary explaining potential technical/business reasons for this drop and highly actionable next steps."""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload, verify=False)
        data = response.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return f"**API Error:** {data}"
    except Exception as e:
        return f"**Connection Error:** {e}"

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- FRONTEND UI ---

def main():
    st.title("⚡ Autonomous Root Cause Engine")
    st.markdown("Enterprise data diagnostics powered by Isolation Forests and Google Gemini.")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password", help="Required for AI generation.")
        st.markdown("---")
        st.subheader("📁 Data Ingestion")
        uploaded_file = st.file_uploader("Upload Raw Data (CSV/TXT)", type=["csv", "txt"])
        
        if uploaded_file:
            df = load_and_prep_data(uploaded_file)
            if 'Date' not in df.columns:
                st.error("Missing 'Date' column.")
                st.stop()
            st.success("Custom data loaded!")
        else:
            df = generate_synthetic_data() 
            st.info("Using synthetic dummy data.")
            
        start_btn = st.button("🚀 Run Analytics Engine", use_container_width=True, type="primary")
        
        if start_btn:
            st.session_state['analysis_started'] = True

    # --- GATEWAY ---
    if 'analysis_started' not in st.session_state:
        st.info("👈 Upload your dataset, input your API key, and initialize the engine from the sidebar.")
        st.stop()
        
    # --- CORE PROCESSING ---
    daily_revenue = df.groupby("Date")["Revenue"].sum().reset_index()
    daily_revenue = detect_anomalies(daily_revenue)
    anomalies = daily_revenue[daily_revenue["Is_Anomaly"]]
    
    # --- TABBED LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["📈 Diagnostic Dashboard", "🧠 AI Insights & RCA", "🗃️ Raw Data Explorer"])
    
    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.markdown("### System KPI Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Days Analyzed", f"{len(df['Date'].unique())}")
        col2.metric("Total Processed Revenue", f"${daily_revenue['Revenue'].sum():,.0f}")
        col3.metric("Critical Anomalies", len(anomalies), "- Action Required", delta_color="inverse")
        
        st.markdown("---")
        daily_revenue['7-Day Trend'] = daily_revenue['Revenue'].rolling(7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_revenue['Date'], y=daily_revenue['Revenue'], mode='lines+markers', name='Revenue', line=dict(color='#2962FF', width=2)))
        fig.add_trace(go.Scatter(x=daily_revenue['Date'], y=daily_revenue['7-Day Trend'], mode='lines', name='7-Day Trend', line=dict(color='#9E9E9E', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Revenue'], mode='markers', name='Anomaly', marker=dict(color='#D50000', size=12, symbol='x', line=dict(width=2))))

        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(showgrid=True, gridcolor='#E0E0E0')
        fig.update_yaxes(showgrid=True, gridcolor='#E0E0E0', tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: AI INSIGHTS ---
    with tab2:
        if not anomalies.empty:
            latest_anomaly = anomalies["Date"].max()
            st.markdown(f"### 🔍 Root Cause Analysis (Anomaly Date: {latest_anomaly.date()})")
            
            rca_data = find_root_cause(df, latest_anomaly)
            worst = rca_data["segment"]
            
            st.error(f"**Primary Variance Driver:** {worst['Region']} · {worst['Device']} | Variance: **${worst['Variance']:,.2f}** ({worst['Pct_Change']:.2f}%)")
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("#### Full Segment Breakdown")
                formatted_df = rca_data["comparison"].copy()
                st.dataframe(formatted_df.style.format({"Variance": "${:,.2f}", "Pct_Change": "{:.2f}%", "Anomaly_Revenue": "${:,.2f}", "Baseline_Revenue": "${:,.2f}"}), use_container_width=True)
            with col_b:
                st.markdown("#### Export Report")
                st.write("Download the mathematical breakdown for your engineering or ops teams.")
                csv = convert_df_to_csv(rca_data["comparison"])
                st.download_button(label="📥 Download CSV Report", data=csv, file_name="rca_variance_report.csv", mime="text/csv", use_container_width=True)
                
            st.markdown("---")
            st.markdown("### 🤖 Executive AI Summary")
            if api_key:
                with st.spinner("Synthesizing multi-dimensional data..."):
                    summary = generate_executive_summary(api_key, rca_data, latest_anomaly)
                    st.info(summary)
            else:
                st.warning("Google Gemini API Key required to generate AI insights. Please add it to the sidebar.")
        else:
            st.success("No anomalies detected in the current dataset.")

    # --- TAB 3: DATA EXPLORER ---
    with tab3:
        st.markdown("### 🗃️ Raw Data Verification")
        st.write("Review the underlying dataset ingested into the analytics engine.")
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()

