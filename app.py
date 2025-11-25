import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sqlalchemy import create_engine
import plotly.express as px
import os
from datetime import timedelta
import numpy as np

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="CrisisGuard Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light Mode
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #F0F2F6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF4B4B;
        color: #31333F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }
    /* Increase font size for tabs */
    button[data-baseweb="tab"] div p {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATABASE CONNECTION ---
try:
    DB_STRING = st.secrets["DB_CONNECTION_STRING"]
except:
    DB_STRING = os.getenv('DB_CONNECTION_STRING')

@st.cache_data(ttl=60)
def get_data():
    try:
        engine = create_engine(DB_STRING)
        df = pd.read_sql("SELECT * FROM crisis_events_v4 ORDER BY created_utc DESC LIMIT 1000", engine)
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        
        # Data Cleaning: Ensure numerical columns are actually numbers
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        
        return df
    except:
        return pd.DataFrame()

raw_df = get_data()

if raw_df.empty:
    st.error("⚠️ Connection established but no data found. Please wait for the pipeline.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Command Center")

# A. Time Slider
min_time = raw_df['created_utc'].min()
max_time = raw_df['created_utc'].max()

if min_time and max_time and min_time != max_time:
    time_range = st.sidebar.slider(
        "Filter Timeline",
        min_value=min_time.to_pydatetime(),
        max_value=max_time.to_pydatetime(),
        value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
        format="MM/DD HH:mm"
    )
    # Filter by Time
    mask_time = (raw_df['created_utc'] >= time_range[0]) & (raw_df['created_utc'] <= time_range[1])
    filtered_df = raw_df[mask_time]
else:
    filtered_df = raw_df.copy()

# B. Risk Filter
if 'status' in filtered_df.columns:
    all_statuses = sorted(filtered_df['status'].dropna().unique())
    selected_status = st.sidebar.multiselect("Severity Level", all_statuses, default=all_statuses)
    if selected_status:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_status)]

# C. Location Filter
if 'location_name' in filtered_df.columns:
    all_locs = sorted(filtered_df['location_name'].dropna().unique())
    selected_loc = st.sidebar.selectbox("Target Region", ["All Global Regions"] + all_locs)

    if selected_loc != "All Global Regions":
        filtered_df = filtered_df[filtered_df['location_name'] == selected_loc]

# D. Alert Simulation
st.sidebar.divider()
st.sidebar.subheader("🚨 Emergency Dispatch")
alert_email = st.sidebar.text_input("Officer Email", placeholder="admin@agency.gov")
if st.sidebar.button("Test Alert System"):
    critical_count = len(filtered_df[filtered_df['status'] == 'Critical'])
    if critical_count > 0:
        st.sidebar.success(f"✅ Alert Sent: {critical_count} critical events flagged in {selected_loc}")
    else:
        st.sidebar.info("No critical events to report at this time.")

# --- 4. MAIN DASHBOARD UI ---
st.title(f"🛡️ CrisisGuard: {selected_loc}")
st.markdown(f"Monitoring **{len(filtered_df)}** active signals.")

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Incidents", len(filtered_df))

if 'status' in filtered_df.columns:
    crit_count = len(filtered_df[filtered_df['status'] == "Critical"])
    c2.metric("Critical Threats", crit_count, delta_color="inverse")
else:
    c2.metric("Critical Threats", 0)

# Safe Average Calculation (Fixes 'nan' error)
if not filtered_df.empty and 'sentiment' in filtered_df.columns:
    avg_sent = filtered_df['sentiment'].mean()
    if pd.notna(avg_sent):
        c3.metric("Avg Sentiment", f"{avg_sent:.2f}", delta="-0.1" if avg_sent < 0 else "0.1")
    else:
        c3.metric("Avg Sentiment", "N/A")

    if 'subreddit' in filtered_df.columns:
        top_source = filtered_df['subreddit'].mode()
        if not top_source.empty:
            c4.metric("Primary Source", f"r/{top_source[0]}")
        else:
            c4.metric("Primary Source", "N/A")

# --- 5. TABS ---
tab_geo, tab_analysis, tab_feed = st.tabs(["🌍 Geospatial Ops", "📊 Risk Analytics", "📋 Data Feed"])

# TAB 1: MAP
with tab_geo:
    # Filter for valid coordinates only
    map_data = filtered_df.dropna(subset=['lat', 'lon'])
    
    if not map_data.empty:
        start_lat = map_data['lat'].mean()
        start_lon = map_data['lon'].mean()
        zoom = 4 if selected_loc != "All Global Regions" else 2
        
        m = folium.Map(location=[start_lat, start_lon], zoom_start=zoom, tiles="CartoDB positron")
        
        # Add points with a jitter so overlapping points (same city) can be seen
        for idx, row in map_data.iterrows():
            color = "#FF0000" if row.get('status') == "Critical" else "#FFA500"
            if row.get('status') == "Moderate": color = "#FFD700"
            
            # Create HTML Popup
            html = f"""
            <div style="font-family: sans-serif; width: 200px; color: #333;">
                <h5 style="margin:0;">{row['location_name']}</h5>
                <span style="color:{color}; font-weight:bold;">{row.get('status', 'Unknown')}</span><br>
                <small>{row['created_utc']}</small>
                <p>{str(row['text'])[:100]}...</p>
                <a href="{row['url']}" target="_blank">View Source</a>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(html, max_width=260),
                tooltip=row['location_name']
            ).add_to(m)
        
        st_folium(m, width=None, height=500)
    else:
        st.warning("No valid geolocation data available for map display.")

# TAB 2: ANALYTICS
with tab_analysis:
    if not filtered_df.empty:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Risk vs. Sentiment Correlation")
            if 'sentiment' in filtered_df.columns and 'status' in filtered_df.columns:
                fig_scatter = px.scatter(
                    filtered_df, 
                    x="sentiment", 
                    y="status", 
                    color="status",
                    size_max=10,
                    hover_data=["text"],
                    color_discrete_map={"Critical": "red", "High": "orange", "Moderate": "#FFD700", "Low": "green"},
                    template="plotly_white"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with c2:
            st.subheader("Incident Volume by Source")
            if 'subreddit' in filtered_df.columns:
                source_counts = filtered_df['subreddit'].value_counts().reset_index()
                source_counts.columns = ['Subreddit', 'Count']
                fig_bar = px.bar(
                    source_counts, 
                    x='Subreddit', 
                    y='Count', 
                    color='Count', 
                    color_continuous_scale='Reds',
                    template="plotly_white"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Not enough data to generate analytics.")

# TAB 3: RAW DATA
with tab_feed:
    st.subheader("Live Intelligence Feed")
    if not filtered_df.empty:
        # Define columns to show
        cols = ['created_utc', 'location_name', 'status', 'sentiment', 'text', 'url']
        # Only show columns that actually exist
        cols = [c for c in cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[cols],
            column_config={
                "url": st.column_config.LinkColumn("Link"),
                "sentiment": st.column_config.ProgressColumn("Sentiment", min_value=-1, max_value=1),
                "status": st.column_config.Column("Risk Level"),
                "text": st.column_config.TextColumn("Content", width="large")
            },
            use_container_width=True,
            hide_index=True  # Hides the 0, 1, 2 index column
        )
    else:
        st.write("No records found.")