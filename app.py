import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sqlalchemy import create_engine
import plotly.express as px
import os
from datetime import timedelta
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="CrisisGuard Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #F0F2F6;
        border-left: 5px solid #FF4B4B;
        color: #31333F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .js-plotly-plot .plotly .main-svg { background: rgba(0,0,0,0) !important; }
    button[data-baseweb="tab"] div p { font-size: 18px !important; font-weight: 600 !important; }
    [data-testid="stDataFrame"] div[data-testid="stTable"] tr:has(td:contains("Critical")) {
        background-color: #ffebee !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING ---
try: DB_STRING = st.secrets["DB_CONNECTION_STRING"]
except: DB_STRING = os.getenv('DB_CONNECTION_STRING')

@st.cache_data(ttl=60)
def get_data():
    try:
        engine = create_engine(DB_STRING)
        df = pd.read_sql("SELECT * FROM crisis_events_v4 ORDER BY created_utc DESC LIMIT 2000", engine)
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['short_text'] = df['text'].astype(str).str.slice(0, 60) + "..."
        return df
    except: return pd.DataFrame()

raw_df = get_data()

# --- 3. SCORING LOGIC (3-LEVEL) ---
if not raw_df.empty:
    # 3-Level Score Map
    status_map = {"Critical": 90, "Moderate": 50, "Low": 10}
    raw_df['base_score'] = raw_df['status'].map(status_map).fillna(0)
    
    if 'sentiment' in raw_df.columns:
        raw_df['impact_score'] = raw_df['base_score'] + (raw_df['sentiment'].fillna(0) * -15)
    else:
        raw_df['impact_score'] = raw_df['base_score']
    
    raw_df['impact_score'] = raw_df['impact_score'].clip(0, 100).astype(int)

# --- 4. CONTROLS ---
st.sidebar.header("Command Center")

if not raw_df.empty:
    min_time = raw_df['created_utc'].min()
    max_time = raw_df['created_utc'].max()
    if min_time != max_time:
        time_range = st.sidebar.slider("Timeline", min_value=min_time.to_pydatetime(), max_value=max_time.to_pydatetime(), value=(min_time.to_pydatetime(), max_time.to_pydatetime()), format="MM/DD HH:mm")
        filtered_df = raw_df[(raw_df['created_utc'] >= time_range[0]) & (raw_df['created_utc'] <= time_range[1])]
    else:
        filtered_df = raw_df.copy()
else:
    filtered_df = pd.DataFrame()

if not filtered_df.empty:
    all_statuses = sorted(filtered_df['status'].dropna().unique())
    selected_status = st.sidebar.multiselect("Severity", all_statuses, default=all_statuses)
    if selected_status: filtered_df = filtered_df[filtered_df['status'].isin(selected_status)]

    all_locs = sorted(filtered_df['location_name'].dropna().unique())
    selected_loc = st.sidebar.selectbox("Region", ["All Global Regions"] + all_locs)
    if selected_loc != "All Global Regions": filtered_df = filtered_df[filtered_df['location_name'] == selected_loc]
else:
    selected_loc = "All Global Regions"

# Alert Simulation
st.sidebar.divider()
st.sidebar.subheader("🚨 Emergency Dispatch")
alert_email = st.sidebar.text_input("Officer Email", placeholder="admin@agency.gov")
if st.sidebar.button("Test Alert System"):
    if not filtered_df.empty:
        critical_count = len(filtered_df[filtered_df['status'] == 'Critical'])
        if critical_count > 0:
            st.sidebar.success(f"Alert Sent: {critical_count} critical events flagged.")
        else:
            st.sidebar.info("No critical events.")

# --- 5. DASHBOARD ---
st.title(f"🛡️ CrisisGuard: {selected_loc}")
last_upd = raw_df['created_utc'].max().strftime('%B %d, %H:%M UTC') if not raw_df.empty else "N/A"
st.markdown(f"Monitoring **{len(filtered_df)}** active signals. <span style='color:gray'>Updated: {last_upd}</span>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Incidents", len(filtered_df))
crit = len(filtered_df[filtered_df['status']=="Critical"]) if not filtered_df.empty else 0
c2.metric("Critical Threats", crit, delta_color="inverse")
sent = filtered_df['sentiment'].mean() if not filtered_df.empty else 0
c3.metric("Avg Sentiment", f"{sent:.2f}")
src = filtered_df['subreddit'].mode()[0] if not filtered_df.empty else "N/A"
c4.metric("Top Source", f"r/{src}")

tab_geo, tab_anal, tab_list = st.tabs(["🌍 Live Map", "📊 Analytics", "📋 Alert Feed"])

with tab_geo:
    if not filtered_df.empty:
        map_data = filtered_df.dropna(subset=['lat', 'lon'])
        if not map_data.empty:
            m = folium.Map(location=[map_data['lat'].mean(), map_data['lon'].mean()], zoom_start=2, tiles="CartoDB positron")
            marker_cluster = MarkerCluster().add_to(m)
            
            for idx, row in map_data.iterrows():
                np.random.seed(int(str(ord(row['id'][0])) + str(ord(row['id'][-1]))))
                jit_lat, jit_lon = np.random.uniform(-0.01, 0.01, 2)
                
                # STRICT 3-COLOR MAP
                stat = row['status']
                if stat == "Critical": color = "red"
                elif stat == "Moderate": color = "orange"
                else: color = "green"
                
                pop = f"<b>{row['location_name']}</b><br>{stat}<br>{row.get('risk_factors','')}"
                folium.Marker([row['lat']+jit_lat, row['lon']+jit_lon], popup=folium.Popup(pop, max_width=200), icon=folium.Icon(color=color, icon="warning-sign")).add_to(marker_cluster)
            
            st_folium(m, width=None, height=500, returned_objects=[])
        else: st.warning("No location data.")
    else: st.info("No data.")

with tab_anal:
    if not filtered_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk vs. Sentiment")
            fig_scatter = px.scatter(
                filtered_df,
                x="sentiment", y="status", color="status",
                size="impact_score", hover_data=["short_text"], 
                # Strict 3 Colors
                color_discrete_map={"Critical": "red", "Moderate": "orange", "Low": "green"},
                template="plotly_white"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        with c2:
             st.subheader("Source Volume")
             source_counts = filtered_df['subreddit'].value_counts().reset_index()
             source_counts.columns = ['Subreddit', 'Count']
             fig_bar = px.bar(
                source_counts, x='Subreddit', y='Count', color='Count',
                color_continuous_scale='Reds', template="plotly_white"
             )
             st.plotly_chart(fig_bar, use_container_width=True)

with tab_list:
    if not filtered_df.empty:
        cols = ['created_utc', 'location_name', 'status', 'impact_score', 'risk_factors', 'text', 'url']
        cols = [c for c in cols if c in filtered_df.columns]
        df_sorted = filtered_df.sort_values('impact_score', ascending=False)

        st.dataframe(
            df_sorted[cols],
            column_config={
                "url": st.column_config.LinkColumn("Link"),
                "impact_score": st.column_config.ProgressColumn("Crisis Score", min_value=0, max_value=100, format="%d"),
                "risk_factors": st.column_config.Column("Risk Drivers"),
                "text": st.column_config.TextColumn("Content", width="large"),
                 "sentiment": st.column_config.ProgressColumn("Sentiment", min_value=-1, max_value=1)
            },
            use_container_width=True,
            hide_index=True
        )