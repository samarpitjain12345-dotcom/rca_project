"""
Autonomous Root Cause Engine — Corporate White & Gray Edition
Limits: 50 MB max payload | Capped previews | Clean executive UI
"""
import io, traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False

# Enterprise Guardrails
MAX_ROWS, MAX_MB = 250_000, 50
PALETTE = ["#2563EB", "#0EA5E9", "#10B981", "#F43F5E", "#F59E0B", "#8B5CF6"]
PLOTLY_TEMPLATE = "plotly_white"

# Page Setup
st.set_page_config(page_title="Autonomous Root Cause Engine", page_icon="⚡", layout="wide")

# Custom Corporate White & Slate Gray CSS (Matching Screenshot Sidebar UI)
st.markdown("""
<style>
    /* Global Layout & Padding */
    .block-container { padding-top: 2.2rem !important; padding-bottom: 2rem; max-width: 1400px; }
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    header, footer { visibility: hidden; }
    
    /* Header & Subtitle */
    .main-title { font-size: 2.1rem; font-weight: 800; color: #0F172A; margin-top: -0.5rem; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
    .main-subtitle { font-size: 0.95rem; color: #475569; line-height: 1.5; margin-bottom: 1.4rem; }
    .model-badge { background-color: #EFF6FF; color: #1D4ED8; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 0.82rem; border: 1px solid #BFDBFE; }
    
    /* Sidebar Block Styling (Exact Match to Screenshot) */
    section[data-testid="stSidebar"] { background-color: #F1F5F9; border-right: 1px solid #E2E8F0; }
    
    /* Uploader Container Card */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.8rem 0.8rem 0.2rem 0.8rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] small { display: none !important; }
    
    /* Green Status Pill */
    .pill-green {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        font-size: 0.88rem;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* White Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetricValue"] { color: #2563EB; font-size: 2rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.05em; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 16px; border-bottom: 2px solid #E2E8F0; }
    .stTabs [data-baseweb="tab"] { color: #64748B; font-weight: 600; font-size: 0.95rem; padding-bottom: 8px; }
    .stTabs [aria-selected="true"] { color: #2563EB; border-bottom: 3px solid #2563EB; }
    
    h1, h2, h3 { color: #0F172A; }
</style>
""", unsafe_allow_html=True)

def safe(title):
    class ErrorBoundary:
        def __enter__(self): return self
        def __exit__(self, et, ev, tb):
            if et:
                st.error(f"⚠️ Operation '{title}' failed ({et.__name__}: {ev}). Engine stabilized.")
                with st.expander("Technical Trace"): st.code("".join(traceback.format_exception(et, ev, tb)))
                return True
    return ErrorBoundary()

@st.cache_data(show_spinner=False, max_entries=3)
def load_file(file_bytes: bytes, filename: str):
    warn = []
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_MB:
        return None, warn, f"File size ({size_mb:.1f} MB) exceeds strict {MAX_MB} MB limit."
    
    df = None
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, low_memory=False)
            break
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding=enc, on_bad_lines="skip")
                warn.append("Auto-detected delimiter; malformed lines skipped.")
                break
            except Exception:
                continue

    if df is None or df.empty:
        return None, warn, "Unreadable or empty file payload."

    empty_unnamed = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if empty_unnamed: df = df.drop(columns=empty_unnamed)

    dups = int(df.duplicated().sum())
    if dups: warn.append(f"{dups:,} duplicate row(s) detected.")
    if len(df) > MAX_ROWS:
        warn.append(f"Sampled to {MAX_ROWS:,} rows for real-time responsiveness.")
        df = df.head(MAX_ROWS)

    for c in df.select_dtypes(include=["int64"]).columns: df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float64"]).columns: df[c] = pd.to_numeric(df[c], downcast="float")
    return df, warn, None

@st.cache_data(max_entries=1)
def demo_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=150, freq="D")
    rows = []
    for i, d in enumerate(dates):
        for region in ["North America", "Europe", "Asia", "LATAM"]:
            base = rng.uniform(800, 1400) * {"North America": 1.3, "Europe": 1.0, "Asia": 0.85, "LATAM": 0.6}[region]
            if i == 100 and region == "Asia": base *= 0.12
            if i == 125 and region == "Europe": base *= 2.1
            rows.append({"Date": d, "Region": region, "Channel": rng.choice(["Web", "Mobile", "Partner"]), "Revenue": round(max(base, 0), 2)})
    return pd.DataFrame(rows)

def guess_date_col(df):
    for c in df.columns:
        if "date" in str(c).lower() or "time" in str(c).lower(): return c
    for c in df.columns:
        s = df[c].dropna().head(20)
        if len(s) and pd.to_datetime(s, errors="coerce").notna().mean() > 0.8: return c
    return df.columns[0]

@st.cache_data(show_spinner=False, max_entries=5)
def process_pipeline(file_bytes, filename, date_col, target_col, sensitivity):
    df_raw, _, _ = load_file(file_bytes, filename) if file_bytes is not None else (demo_data(), [], None)
    df = df_raw.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    dropped = int(df[date_col].isna().sum() + df[target_col].isna().sum())
    df = df.dropna(subset=[date_col, target_col])

    if df.empty: return df, pd.DataFrame(), dropped

    df["_d"] = df[date_col].dt.normalize()
    daily = df.groupby("_d", observed=True)[target_col].sum().reset_index().rename(columns={"_d": date_col}).sort_values(date_col).reset_index(drop=True)

    x = pd.to_numeric(daily[target_col], errors="coerce").fillna(0.0)
    n = len(x)

    if n < 8:
        daily["is_anomaly"], daily["severity"] = False, 0.0
    elif SKLEARN_OK and n >= 15:
        model = IsolationForest(contamination=float(np.clip(sensitivity, 0.01, 0.20)), random_state=42, n_jobs=-1)
        pred = model.fit_predict(x.values.reshape(-1, 1))
        scores = -model.score_samples(x.values.reshape(-1, 1))
        span = scores.max() - scores.min()
        daily["is_anomaly"] = pred == -1
        daily["severity"] = (scores - scores.min()) / span if span > 1e-9 else 0.0
    else:
        w = max(5, int(n * 0.15))
        rm = x.rolling(w, min_periods=3, center=True).mean()
        rs = x.rolling(w, min_periods=3, center=True).std().replace(0, np.nan)
        z = ((x - rm) / rs).fillna(0)
        daily["is_anomaly"] = z.abs() > float(np.clip(3.2 - sensitivity * 9, 1.5, 4.0))
        daily["severity"] = (z.abs() / z.abs().max()).clip(0, 1) if z.abs().max() > 1e-9 else 0.0

    return df, daily, dropped

def compute_rca(df, anom_date, date_col, target, dims):
    if not dims: return None
    ts = pd.to_datetime(anom_date)
    today, base = df[df[date_col] == ts], df[(df[date_col] >= ts - pd.Timedelta(days=7)) & (df[date_col] < ts)]
    if today.empty or base.empty: return None

    b = base.groupby(list(dims), observed=True, dropna=False)[target].mean().reset_index(name="baseline")
    a = today.groupby(list(dims), observed=True, dropna=False)[target].sum().reset_index(name="actual")
    comp = a.merge(b, on=list(dims), how="outer").fillna(0)
    if comp.empty: return None

    comp["variance"] = comp["actual"] - comp["baseline"]
    comp["pct_change"] = np.where(comp["baseline"] == 0, 0.0, comp["variance"] / comp["baseline"] * 100)
    return {"worst": comp.loc[comp["variance"].idxmin()], "best": comp.loc[comp["variance"].idxmax()], "table": comp.sort_values("variance")}

@st.cache_data(show_spinner=False, ttl=3600, max_entries=10)
def query_ai_insights(key, prompt):
    if not REQUESTS_OK or not key: return "API key missing or requests library unavailable."
    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=12
        )
        r.raise_for_status()
        cands = r.json().get("candidates", [])
        return cands[0]["content"]["parts"][0]["text"] if cands else "No output generated."
    except Exception as e:
        return f"AI Error: {type(e).__name__}"

def main():
    # --- HEADER & SUBTITLE ---
    st.markdown('<div class="main-title">🔎 Autonomous Root Cause Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">'
        'Upload a CSV or TXT file and it automatically identifies unusual spikes/drops in daily metrics, '
        'explains which segment caused each event using an <span class="model-badge">Isolation Forest ML Model</span>, '
        'and diagnoses root causes without writing formulas.'
        '</div>', 
        unsafe_allow_html=True
    )

    # --- SIDEBAR (Exact Match to Screenshot UI) ---
    with st.sidebar:
        st.markdown("### **1 · Data**")
        up = st.file_uploader("Upload CSV or TXT (max 50 MB)", type=["csv", "txt"])
        
        warnings, err = [], None
        filename = up.name if up is not None else "demo"
        file_bytes = up.getvalue() if up is not None else None

        if up is not None:
            with safe("File Ingestion"):
                df_raw, warnings, err = load_file(file_bytes, filename)
        else:
            df_raw = demo_data()
            st.caption("No file uploaded — showing demo data.")

        if err: st.error(err); st.stop()
        if df_raw is None or df_raw.empty: st.warning("No usable data."); st.stop()

        # Green status pill (matching screenshot)
        st.markdown(f'<div class="pill-green">{len(df_raw):,} rows × {df_raw.shape[1]} cols</div>', unsafe_allow_html=True)
        cols = df_raw.columns.tolist()
        
        date_col = st.selectbox("Date column", cols, index=cols.index(guess_date_col(df_raw)))
        num_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != date_col]
        if not num_cols: st.error("No numeric metrics found."); st.stop()
        
        target_col = st.selectbox("Metric", num_cols, index=len(num_cols) - 1)
        cat_cols = [c for c in cols if c not in (date_col, target_col)]
        dims = tuple(st.multiselect("Breakdown dimensions (max 2)", cat_cols, default=cat_cols[:1], max_selections=2))
        
        sensitivity = st.slider("Anomaly sensitivity", 0.01, 0.20, 0.05, 0.01)
        api_key = st.text_input("Gemini API Key (optional)", type="password") if REQUESTS_OK else ""

    # --- RUN PIPELINE ---
    with safe("Analytics Execution"):
        df, daily, dropped = process_pipeline(file_bytes, filename, date_col, target_col, sensitivity)

    if dropped: st.warning(f"⚠️ Cleaned {dropped:,} row(s) with unparseable date/metric values.")
    if df.empty: st.error("No valid data remaining after sanitization."); return

    anomalies = daily[daily["is_anomaly"]].sort_values("severity", ascending=False) if not daily.empty else pd.DataFrame()

    # --- TABS ---
    t1, t2, t3, t4, t5 = st.tabs(["📈 Trend", "🔬 Breakdown", "🧠 Root Cause", "🗄️ Data Quality", "🤖 AI Insights"])

    # TAB 1: TREND
    with t1, safe("Trend"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Days", len(daily))
        c3.metric(f"Total {target_col}", f"{daily[target_col].sum():,.0f}" if not daily.empty else "0")
        c4.metric("Anomalies", len(anomalies))

        if not daily.empty:
            daily["smooth"] = daily[target_col].rolling(7, min_periods=1).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily[date_col], y=daily[target_col], line=dict(color="#BFDBFE", width=1.2), name="Daily"))
            fig.add_trace(go.Scatter(x=daily[date_col], y=daily["smooth"], line=dict(color="#2563EB", width=2.5), name="7-day trend"))
            
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies[date_col], y=anomalies[target_col], mode="markers", name="Anomaly",
                    marker=dict(color="#DC2626", size=9, symbol="x", line=dict(width=2, color="#FFFFFF"))
                ))

            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=420, hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=15, l=5, r=5, b=5), legend=dict(orientation="h", y=1.12)
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: BREAKDOWN
    with t2, safe("Breakdown"):
        if not dims:
            st.info("Pick a breakdown dimension in the sidebar.")
        else:
            g = df.groupby(list(dims), observed=True, dropna=False)[target_col].sum().reset_index()
            if g.empty:
                st.info("No categorical breakdown data.")
            else:
                label = dims[-1]
                top = g.sort_values(target_col, ascending=False).head(12)
                colA, colB = st.columns([1.3, 1])
                
                with colA:
                    fig_bar = px.bar(
                        top.sort_values(target_col), x=target_col, y=label, orientation="h",
                        color=target_col, color_continuous_scale="Blues", title=f"Top segments by {target_col}",
                        template=PLOTLY_TEMPLATE
                    )
                    fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with colB:
                    pos = g[g[target_col] > 0]
                    if not pos.empty:
                        fig_pie = px.pie(
                            pos.nlargest(8, target_col), names=label, values=target_col, hole=0.55,
                            color_discrete_sequence=PALETTE, title="Share of total", template=PLOTLY_TEMPLATE
                        )
                        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)

    # TAB 3: ROOT CAUSE
    with t3, safe("Root Cause"):
        if anomalies.empty:
            st.success("No anomalies detected at current sensitivity setting.")
        elif not dims:
            st.warning("Pick a breakdown dimension in the sidebar to investigate root causes.")
        else:
            opts = [str(d)[:10] for d in anomalies[date_col]]
            choice = st.selectbox("Investigate Anomaly Date:", opts)
            chosen = anomalies[date_col].iloc[opts.index(choice)]
            
            rca = compute_rca(df, chosen, date_col, target_col, list(dims))
            if rca is None:
                st.info("Not enough baseline history (needs ≥7 prior days for variance).")
            else:
                w, b = rca["worst"], rca["best"]
                sw = " | ".join(str(w[d]) for d in dims)
                sb = " | ".join(str(b[d]) for d in dims)
                
                c1, c2 = st.columns(2)
                c1.error(f"**Biggest drag:** {sw}  \nVariance: {w['variance']:,.2f} ({w['pct_change']:.1f}%)")
                c2.success(f"**Biggest lift:** {sb}  \nVariance: {b['variance']:,.2f} ({b['pct_change']:.1f}%)")
                
                st.markdown("#### Segment Variance Summary")
                st.dataframe(rca["table"], use_container_width=True, hide_index=True)

    # TAB 4: DATA QUALITY (Capped preview strictly under 300 rows)
    with t4, safe("Data Quality"):
        for w in warnings: st.warning(w)
        if not warnings: st.success("No ingestion warnings.")
        
        mem = df_raw.memory_usage(deep=True).sum() / (1024 * 1024)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_raw):,}")
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Memory", f"{mem:,.2f} MB")
        
        st.markdown("#### Data Preview (Top 250 Rows)")
        st.dataframe(df_raw.head(250), use_container_width=True)

    # TAB 5: AI INSIGHTS
    with t5, safe("AI Insights"):
        st.caption("Consolidated single-pass AI briefing for rapid operational analysis.")
        if not REQUESTS_OK:
            st.info("`requests` library is unavailable.")
        elif not api_key:
            st.warning("Add a Gemini API key in the sidebar to generate AI Insights.")
        else:
            if st.button("🚀 Generate AI Insights", type="primary"):
                top_anomaly_str = str(anomalies.iloc[0][date_col])[:10] if not anomalies.empty else "None"
                prompt = (
                    f"You are a senior data analyst reviewing this dataset summary:\n"
                    f"- Metric: {target_col}\n"
                    f"- Days Monitored: {len(daily)}\n"
                    f"- Total Volume: {daily[target_col].sum():,.0f}\n"
                    f"- Anomalies Detected: {len(anomalies)} (Top date: {top_anomaly_str})\n\n"
                    f"Respond with exactly 3 sections:\n"
                    f"1. Executive Summary (2 sentences)\n"
                    f"2. Top 3 Actionable Insights (bullet points)\n"
                    f"3. Operational Risk Factor (1 sentence)."
                )
                with st.spinner("Analyzing dataset..."):
                    st.markdown(query_ai_insights(api_key, prompt))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Unexpected system error. Refresh the page.")
        with st.expander("Details"): st.code(traceback.format_exc())

