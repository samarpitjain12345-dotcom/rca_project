"""
Autonomous Root Cause Explorer — Advanced Visualization Edition
Enhanced Graphs: Telemetry Gradient Fill & Anomaly Spans | Temporal Drift Heatmap | RCA Variance Waterfall
"""
import io, traceback
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st

try: from sklearn.ensemble import IsolationForest; SKLEARN_OK = True
except: SKLEARN_OK = False
try: import requests; REQUESTS_OK = True
except: REQUESTS_OK = False

MAX_ROWS, MAX_MB = 250_000, 50
PALETTE = ["#2563EB", "#0EA5E9", "#10B981", "#F43F5E", "#F59E0B", "#8B5CF6", "#06B6D4", "#6366F1"]
PLOTLY_TEMPLATE = "plotly_white"

st.set_page_config(page_title="Autonomous Root Cause Explorer", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2.2rem !important; padding-bottom: 2rem; max-width: 1400px; }
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    header, footer { visibility: hidden; }
    .main-title { font-size: 2.1rem; font-weight: 800; color: #0F172A; margin-top: -0.5rem; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
    .main-subtitle { font-size: 0.95rem; color: #475569; line-height: 1.5; margin-bottom: 1.4rem; }
    .model-badge { background-color: #EFF6FF; color: #1D4ED8; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 0.82rem; border: 1px solid #BFDBFE; }
    section[data-testid="stSidebar"] { background-color: #F1F5F9; border-right: 1px solid #E2E8F0; }
    div[data-testid="stFileUploader"] { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.8rem 0.8rem 0.2rem 0.8rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04); }
    div[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] small { display: none !important; }
    .pill-green { background-color: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.65rem 1rem; font-weight: 600; font-size: 0.88rem; margin-top: 0.5rem; margin-bottom: 0.8rem; }
    div[data-testid="metric-container"] { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.1rem 1.25rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04); }
    div[data-testid="stMetricValue"] { color: #2563EB; font-size: 1.8rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.05em; }
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
    if (len(file_bytes) / (1024 * 1024)) > MAX_MB: return None, warn, f"File size exceeds strict {MAX_MB} MB limit."
    df = None
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try: df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, low_memory=False); break
        except:
            try: df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding=enc, on_bad_lines="skip"); warn.append("Auto-detected delimiter."); break
            except: continue
    if df is None or df.empty: return None, warn, "Unreadable or empty file payload."
    empty_cols = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if empty_cols: df = df.drop(columns=empty_cols)
    if df.duplicated().sum(): warn.append(f"{df.duplicated().sum():,} duplicate row(s) detected.")
    if len(df) > MAX_ROWS: df = df.head(MAX_ROWS); warn.append(f"Sampled to {MAX_ROWS:,} rows.")
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
        if pd.to_datetime(df[c].dropna().head(20), errors="coerce").notna().mean() > 0.8: return c
    return df.columns[0]

def momentum(daily, target_col, window):
    if len(daily) < window * 2: return None
    rec, pri = daily[target_col].tail(window).sum(), daily[target_col].iloc[-window*2:-window].sum()
    return ((rec - pri) / pri * 100) if pri != 0 else None

def forecast_trend(daily, date_col, target_col, horizon):
    n = len(daily)
    if n < 14: return None
    t, y = np.arange(n), daily[target_col].values
    coeffs = np.polyfit(t, y, 1)
    trend = np.polyval(coeffs, t)
    dow = pd.to_datetime(daily[date_col]).dt.dayofweek
    seasonal_map = pd.Series(y - trend).groupby(dow).mean()
    fitted = trend + dow.map(seasonal_map).fillna(0).values
    std_res = float(np.std(y - fitted)) if n > 1 else 0.0
    fut_dates = pd.date_range(daily[date_col].max() + pd.Timedelta(days=1), periods=horizon)
    fut_t = np.arange(n, n + horizon)
    fut_forecast = np.polyval(coeffs, fut_t) + pd.Series(fut_dates.dayofweek).map(seasonal_map).fillna(0).values
    return {"dates": fut_dates, "forecast": fut_forecast, "upper": fut_forecast + 1.96 * std_res, "lower": fut_forecast - 1.96 * std_res, "slope_pct": (coeffs[0] / (y.mean() + 1e-9)) * 100}

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
    if n < 8: daily["is_anomaly"], daily["severity"] = False, 0.0
    elif SKLEARN_OK and n >= 15:
        model = IsolationForest(contamination=float(np.clip(sensitivity, 0.01, 0.20)), random_state=42, n_jobs=-1)
        pred = model.fit_predict(x.values.reshape(-1, 1))
        scores = -model.score_samples(x.values.reshape(-1, 1))
        span = scores.max() - scores.min()
        daily["is_anomaly"], daily["severity"] = pred == -1, (scores - scores.min()) / span if span > 1e-9 else 0.0
    else:
        w = max(5, int(n * 0.15))
        z = ((x - x.rolling(w, min_periods=3, center=True).mean()) / x.rolling(w, min_periods=3, center=True).std().replace(0, np.nan)).fillna(0)
        daily["is_anomaly"], daily["severity"] = z.abs() > float(np.clip(3.2 - sensitivity * 9, 1.5, 4.0)), (z.abs() / z.abs().max()).clip(0, 1) if z.abs().max() > 1e-9 else 0.0
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
        r = requests.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent", headers={"Content-Type": "application/json", "x-goog-api-key": key}, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=12)
        r.raise_for_status()
        return r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No output generated.")
    except Exception as e: return f"AI Error: {type(e).__name__}"

def main():
    st.markdown('<div class="main-title">🔎 Autonomous Root Cause Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Upload CSV/TXT files to detect spikes/drops via an <span class="model-badge">Isolation Forest ML Model</span> and compute root causes.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### **1 · Data**")
        up = st.file_uploader("Upload CSV or TXT (max 50 MB)", type=["csv", "txt"])
        filename, file_bytes = (up.name, up.getvalue()) if up is not None else ("demo", None)
        df_raw, warnings, err = load_file(file_bytes, filename) if up is not None else (demo_data(), [], None)
        if err: st.error(err); st.stop()
        if df_raw is None or df_raw.empty: st.warning("No usable data."); st.stop()

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

    with safe("Analytics Execution"):
        df, daily, dropped = process_pipeline(file_bytes, filename, date_col, target_col, sensitivity)
    if dropped: st.warning(f"⚠️ Cleaned {dropped:,} row(s) with invalid dates/metrics.")
    if df.empty: st.error("No valid data remaining after sanitization."); return

    anomalies = daily[daily["is_anomaly"]].sort_values("severity", ascending=False) if not daily.empty else pd.DataFrame()
    t1, t2, t3, t4, t5 = st.tabs(["📈 Trend", "🔬 Breakdown", "🧠 Root Cause", "🗄️ Data Quality", "🤖 AI Insights"])

    # TAB 1: ADVANCED TREND TELEMETRY
    with t1, safe("Trend"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}"); c2.metric("Days", len(daily)); c3.metric(f"Total {target_col}", f"{daily[target_col].sum():,.0f}" if not daily.empty else "0"); c4.metric("Anomalies", len(anomalies))
        if not daily.empty:
            daily["smooth"] = daily[target_col].rolling(7, min_periods=1).mean()
            wow, mom = momentum(daily, target_col, 7), momentum(daily, target_col, 30)
            if wow is not None or mom is not None:
                m1, m2 = st.columns(2)
                if wow is not None: m1.metric("Week-over-week", f"{wow:+.1f}%", delta=f"{wow:+.1f}%")
                if mom is not None: m2.metric("Month-over-month", f"{mom:+.1f}%", delta=f"{mom:+.1f}%")
            show_fc = st.checkbox("📊 Show forecast & anomaly guide", value=False)
            horizon = st.slider("Forecast horizon (days)", 7, 60, 14, 7) if show_fc else 0
            
            fig = go.Figure()
            # Smooth area gradient fill for raw daily volume
            fig.add_trace(go.Scatter(x=daily[date_col], y=daily[target_col], fill='tozeroy', fillcolor='rgba(191,219,254,0.3)', line=dict(color="#93C5FD", width=1.2), name="Daily Volume", hovertemplate="%{x|%b %d}: <b>%{y:,.2f}</b><extra></extra>"))
            fig.add_trace(go.Scatter(x=daily[date_col], y=daily["smooth"], line=dict(color="#2563EB", width=2.8), name="7-Day Moving Avg"))
            
            # Draw vertical guide lines for anomalies
            if not anomalies.empty:
                for ad in anomalies[date_col]:
                    fig.add_vline(x=ad, line_width=1, line_dash="dash", line_color="rgba(220,38,38,0.35)")
                fig.add_trace(go.Scatter(x=anomalies[date_col], y=anomalies[target_col], mode="markers", name="Anomaly Flag", marker=dict(color="#DC2626", size=10, symbol="circle-x", line=dict(width=2, color="#FFFFFF")), hovertemplate="Anomaly %{x|%b %d}: <b>%{y:,.2f}</b><extra></extra>"))
            
            if show_fc:
                fc = forecast_trend(daily, date_col, target_col, horizon)
                if fc:
                    fig.add_trace(go.Scatter(x=list(fc["dates"]) + list(fc["dates"][::-1]), y=list(fc["upper"]) + list(fc["lower"][::-1]), fill="toself", fillcolor="rgba(37,99,235,0.12)", line=dict(width=0), name="95% Confidence Band"))
                    fig.add_trace(go.Scatter(x=fc["dates"], y=fc["forecast"], line=dict(color="#7C3AED", width=2.5, dash="dash"), name=f"{horizon}-Day Projection"))
                    st.caption(f"Projection Vector: {'📈 Rising' if fc['slope_pct'] > 0.5 else '📉 Declining' if fc['slope_pct'] < -0.5 else '➡️ Flat'} trajectory at {fc['slope_pct']:+.2f}%/day.")
            
            fig.update_layout(template=PLOTLY_TEMPLATE, height=430, hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=15, l=5, r=5, b=5), legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: ADVANCED BREAKDOWN (4 PERSPECTIVE MODES INC. TEMPORAL DRIFT MATRIX)
    with t2, safe("Breakdown"):
        if not dims: st.info("Pick a breakdown dimension in the sidebar.")
        else:
            g = df.groupby(list(dims), observed=True, dropna=False)[target_col].sum().reset_index()
            if g.empty: st.info("No categorical breakdown data.")
            else:
                label = dims[-1]
                g_sorted = g.sort_values(target_col, ascending=False)
                tot_val = g_sorted[target_col].sum()
                top_r = g_sorted.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Segments", f"{g[label].nunique():,}"); m2.metric("Apex Segment", str(top_r[label])); m3.metric("Apex Volume", f"{top_r[target_col]:,.0f}"); m4.metric("Concentration", f"{(top_r[target_col]/tot_val*100 if tot_val>0 else 0):.1f}% Share")
                st.markdown("<br>", unsafe_allow_html=True)
                ctrl1, ctrl2, ctrl3 = st.columns([2.2, 1, 1])
                with ctrl1: view_m = st.radio("Perspective", ["📊 Bar & Donut", "🌳 Sunburst & Treemap", "📈 Pareto (80/20 Rule)", "🔥 Temporal Drift Heatmap"], horizontal=True, key="bd_v")
                with ctrl2: top_n = st.slider("Display Top N", 3, 25, 10, key="bd_n")
                with ctrl3: grp_oth = st.checkbox("Group tail into 'Other'", value=True, key="bd_o")

                if grp_oth and len(g_sorted) > top_n:
                    top_part = g_sorted.head(top_n).copy()
                    oth_dict = {d: "Other" for d in dims}; oth_dict[target_col] = g_sorted.iloc[top_n:][target_col].sum()
                    plot_df = pd.concat([top_part, pd.DataFrame([oth_dict])], ignore_index=True)
                else: plot_df = g_sorted.head(top_n)

                if view_m == "📊 Bar & Donut":
                    cA, cB = st.columns([1.3, 1])
                    with cA:
                        fb = px.bar(plot_df.sort_values(target_col), x=target_col, y=label, orientation="h", color=target_col, color_continuous_scale="Blues", title=f"Top {top_n} Segments", template=PLOTLY_TEMPLATE, text_auto=".2s")
                        fb.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, height=400)
                        st.plotly_chart(fb, use_container_width=True)
                    with cB:
                        pos = plot_df[plot_df[target_col] > 0]
                        if not pos.empty:
                            fp = px.pie(pos, names=label, values=target_col, hole=0.55, color_discrete_sequence=PALETTE, title="Volume Share Breakdown", template=PLOTLY_TEMPLATE)
                            fp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
                            st.plotly_chart(fp, use_container_width=True)
                elif view_m == "🌳 Sunburst & Treemap":
                    pos_f = g[g[target_col] > 0]; cA, cB = st.columns(2)
                    with cA:
                        ft = px.treemap(pos_f, path=list(dims), values=target_col, color=target_col, color_continuous_scale="Blues", title="Hierarchical Treemap", template=PLOTLY_TEMPLATE)
                        ft.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=420); st.plotly_chart(ft, use_container_width=True)
                    with cB:
                        fs = px.sunburst(pos_f, path=list(dims), values=target_col, color_discrete_sequence=PALETTE, title="Hierarchical Sunburst", template=PLOTLY_TEMPLATE)
                        fs.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=420); st.plotly_chart(fs, use_container_width=True)
                elif view_m == "📈 Pareto (80/20 Rule)":
                    p_df = g_sorted.copy(); p_df["cum_pct"] = (p_df[target_col].cumsum() / tot_val) * 100; p_sub = p_df.head(min(20, len(p_df)))
                    fp = go.Figure()
                    fp.add_trace(go.Bar(x=p_sub[label], y=p_sub[target_col], name="Volume", marker_color="#2563EB"))
                    fp.add_trace(go.Scatter(x=p_sub[label], y=p_sub["cum_pct"], name="Cumulative Share %", yaxis="y2", line=dict(color="#F43F5E", width=2.5)))
                    fp.update_layout(title="Pareto Contribution Map (80/20 Rule)", template=PLOTLY_TEMPLATE, height=420, yaxis=dict(title=target_col), yaxis2=dict(title="Cumulative Share (%)", overlaying="y", side="right", range=[0, 105]), legend=dict(orientation="h", y=1.12), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fp, use_container_width=True)
                elif view_m == "🔥 Temporal Drift Heatmap":
                    top_cats = g_sorted.head(10)[label].tolist()
                    h_df = df[df[label].isin(top_cats)].groupby(["_d", label], observed=True)[target_col].sum().reset_index()
                    pivot_h = h_df.pivot(index=label, columns="_d", values=target_col).fillna(0)
                    fig_h = px.imshow(pivot_h, aspect="auto", color_continuous_scale="Blues", title="Segment Temporal Performance Heatmap (Daily Intensity)", template=PLOTLY_TEMPLATE)
                    fig_h.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_h, use_container_width=True)

                with st.expander("📊 Full Segment Distribution Table"): st.dataframe(g_sorted, use_container_width=True, hide_index=True)

    # TAB 3: ADVANCED ROOT CAUSE (WITH DIVERGING VARIANCE WATERFALL CHART)
    with t3, safe("Root Cause"):
        if anomalies.empty: st.success("No anomalies detected at current sensitivity.")
        elif not dims: st.warning("Pick a breakdown dimension in the sidebar.")
        else:
            opts = [str(d)[:10] for d in anomalies[date_col]]
            choice = st.selectbox("Investigate Anomaly Date:", opts)
            chosen = anomalies[date_col].iloc[opts.index(choice)]
            rca = compute_rca(df, chosen, date_col, target_col, list(dims))
            if rca is None: st.info("Not enough baseline history (needs ≥7 prior days).")
            else:
                w, b = rca["worst"], rca["best"]
                c1, c2 = st.columns(2)
                c1.error(f"**Primary Drag Factor:** {' | '.join(str(w[d]) for d in dims)}  \nVariance: **{w['variance']:,.2f}** ({w['pct_change']:.1f}%)")
                c2.success(f"**Primary Lift Factor:** {' | '.join(str(b[d]) for d in dims)}  \nVariance: **+{b['variance']:,.2f}** (+{b['pct_change']:.1f}%)")
                
                # Diverging Variance Visual Breakdown
                comp_tbl = rca["table"].copy()
                comp_tbl["seg_id"] = comp_tbl[list(dims)].astype(str).agg(" | ".join, axis=1)
                fig_var = px.bar(comp_tbl.head(12), x="variance", y="seg_id", orientation="h", color="variance", color_continuous_scale=["#DC2626", "#CBD5E1", "#16A34A"], title=f"Segment Variance Impact vs 7-Day Baseline ({str(chosen)[:10]})", template=PLOTLY_TEMPLATE, text_auto=".2s")
                fig_var.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_var, use_container_width=True)

                st.dataframe(rca["table"], use_container_width=True, hide_index=True)
                ca, cb = st.columns([1, 2])
                with ca: st.download_button("📥 Download report CSV", rca["table"].to_csv(index=False), f"rca_{str(chosen)[:10]}.csv")
                with cb:
                    if st.button("Generate AI explanation"):
                        st.info(query_ai_insights(api_key, f"Metric '{target_col}' anomaly on {str(chosen)[:10]}. Largest drag: [{' | '.join(str(w[d]) for d in dims)}] ({w['pct_change']:.1f}%). Give 3 bullet points: likely cause, impact, next step."))

    # TAB 4: DATA QUALITY
    with t4, safe("Data Quality"):
        for w in warnings: st.warning(w)
        if not warnings: st.success("No ingestion warnings.")
        mem = df_raw.memory_usage(deep=True).sum() / (1024 * 1024)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_raw):,}"); c2.metric("Columns", df_raw.shape[1]); c3.metric("Memory", f"{mem:,.2f} MB")
        st.dataframe(df_raw.head(250), use_container_width=True)

    # TAB 5: AI INSIGHTS
    with t5, safe("AI Insights"):
        st.caption("Consolidated single-pass AI briefing for rapid operational analysis.")
        if not REQUESTS_OK: st.info("`requests` library unavailable.")
        elif not api_key: st.warning("Add a Gemini API key in the sidebar.")
        else:
            if st.button("🚀 Generate AI Insights", type="primary"):
                top_anom = str(anomalies.iloc[0][date_col])[:10] if not anomalies.empty else "None"
                ctx = f"Metric: {target_col} | Days: {len(daily)} | Total: {daily[target_col].sum():,.0f} | Anomalies: {len(anomalies)} (Top date: {top_anom})"
                with st.spinner("Analyzing dataset..."): st.markdown(query_ai_insights(api_key, f"Data summary:\n{ctx}\nProvide: 1) Executive Summary (2 sentences), 2) Top 3 Actionable Insights (bullets), 3) Key Operational Risk (1 sentence)."))

if __name__ == "__main__":
    try: main()
    except Exception: st.error("Unexpected error."); st.code(traceback.format_exc())

