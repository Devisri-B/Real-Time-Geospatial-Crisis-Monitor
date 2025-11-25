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
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }
    button[data-baseweb="tab"] div p {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    /* Highlight Critical Rows in Dataframe */
    [data-testid="stDataFrame"] div[data-testid="stTable"] tr:has(td:contains("Critical")) {
        background-color: #ffebee !important;
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
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        return df
    except:
        return pd.DataFrame()

raw_df = get_data()

# --- 3. CALCULATE SCORES (Unified Logic) ---
if not raw_df.empty:
    # 1. Calculate Impact Score (0-100)
    # Base score from AI Risk Class
    status_map = {"Critical": 90, "High": 75, "Moderate": 50, "Low": 20}
    raw_df['base_score'] = raw_df['status'].map(status_map).fillna(0)
    
    # Modify by Sentiment (Negative sentiment increases score)
    # Sentiment is -1 to 1. We invert it so -1 adds +15 points, +1 removes points.
    # Check if sentiment exists to avoid errors
    if 'sentiment' in raw_df.columns:
        raw_df['impact_score'] = raw_df['base_score'] + (raw_df['sentiment'].fillna(0) * -15)
    else:
        raw_df['impact_score'] = raw_df['base_score']

    # Clip to 0-100
    raw_df['impact_score'] = raw_df['impact_score'].clip(0, 100).astype(int)

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.header("Command Center")

# A. Time Slider
if not raw_df.empty:
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
else:
    filtered_df = pd.DataFrame()


# B. Risk Filter
if not filtered_df.empty and 'status' in filtered_df.columns:
    all_statuses = sorted(filtered_df['status'].dropna().unique())
    selected_status = st.sidebar.multiselect("Severity Level", all_statuses, default=all_statuses)
    if selected_status:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_status)]

# C. Location Filter
if not filtered_df.empty and 'location_name' in filtered_df.columns:
    all_locs = sorted(filtered_df['location_name'].dropna().unique())
    selected_loc = st.sidebar.selectbox("Target Region", ["All Global Regions"] + all_locs)

    if selected_loc != "All Global Regions":
        filtered_df = filtered_df[filtered_df['location_name'] == selected_loc]
else:
    selected_loc = "All Global Regions" # Default for title


# D. Alert Simulation
st.sidebar.divider()
st.sidebar.subheader("Emergency Dispatch")
alert_email = st.sidebar.text_input("Officer Email", placeholder="admin@agency.gov")
if st.sidebar.button("Test Alert System"):
    if not filtered_df.empty and 'status' in filtered_df.columns:
        critical_count = len(filtered_df[filtered_df['status'] == 'Critical'])
        if critical_count > 0:
            st.sidebar.success(f"Alert Sent: {critical_count} critical events flagged in {selected_loc}")
        else:
            st.sidebar.info("No critical events to report at this time.")
    else:
        st.sidebar.warning("No data available to alert on.")

# --- 5. MAIN DASHBOARD UI ---
st.title(f"🛡️ CrisisGuard: {selected_loc}")

if not raw_df.empty:
    last_updated_str = raw_df['created_utc'].max().strftime('%B %d, %Y %H:%M UTC')
else:
    last_updated_str = "N/A"

st.markdown(f"""
Monitoring **{len(filtered_df)}** active signals.  
<span style="color: gray; font-size: 0.9em;"> Last Updated: {last_updated_str}</span>
""", unsafe_allow_html=True)

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Incidents", len(filtered_df))

if not filtered_df.empty and 'status' in filtered_df.columns:
    crit_count = len(filtered_df[filtered_df['status'] == "Critical"])
    c2.metric("Critical Threats", crit_count, delta_color="inverse")
else:
    c2.metric("Critical Threats", 0)

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
else:
    c3.metric("Avg Sentiment", "N/A")
    c4.metric("Primary Source", "N/A")


# --- 6. TABS ---
tab_geo, tab_analysis, tab_feed = st.tabs(["Geospatial Ops", "Risk Analytics", "Data Feed"])

# TAB 1: MAP
with tab_geo:
    # Filter for valid coordinates only
    if not filtered_df.empty:
        map_data = filtered_df.dropna(subset=['lat', 'lon'])
        
        if not map_data.empty:
            start_lat = map_data['lat'].mean()
            start_lon = map_data['lon'].mean()
            zoom = 4 if selected_loc != "All Global Regions" else 2
            
            m = folium.Map(location=[start_lat, start_lon], zoom_start=zoom, tiles="CartoDB positron")
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add points with jitter
            for idx, row in map_data.iterrows():
                seed = int(str(ord(row['id'][0])) + str(ord(row['id'][-1]))) 
                np.random.seed(seed)
                jitter_lat = np.random.uniform(-0.01, 0.01) 
                jitter_lon = np.random.uniform(-0.01, 0.01)
                
                color = "red" if row.get('status') == "Critical" else "orange"
                if row.get('status') == "Moderate": color = "gold"
                
                # Check if impact_score exists before using it
                score_display = int(row['impact_score']) if 'impact_score' in row else "N/A"

                html = f"""
                <div style="font-family: sans-serif; width: 200px; color: #333;">
                    <h5 style="margin:0;">{row['location_name']}</h5>
                    <span style="color:{color}; font-weight:bold;">{row.get('status', 'Unknown')}</span><br>
                    <small>Score: {score_display}</small><br>
                    <small>{row['created_utc']}</small>
                    <p>{str(row['text'])[:100]}...</p>
                    <a href="{row['url']}" target="_blank">View Source</a>
                </div>
                """
                
                folium.Marker(
                    location=[row['lat'] + jitter_lat, row['lon'] + jitter_lon], 
                    radius=8,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(html, max_width=260),
                    tooltip=f"{row['location_name']} (Score: {score_display})",
                    icon=folium.Icon(color=color, icon="warning-sign")
                ).add_to(marker_cluster)
            
            st_folium(m, width=None, height=500, returned_objects=[])
        else:
            st.warning("No valid geolocation data available for map display.")
    else:
        st.info("No data to display on map.")

# TAB 2: ANALYTICS
with tab_analysis:
    if not filtered_df.empty:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Risk vs. Sentiment Correlation")
            if 'sentiment' in filtered_df.columns and 'status' in filtered_df.columns and 'impact_score' in filtered_df.columns:
                fig_scatter = px.scatter(
                    filtered_df, 
                    x="sentiment", 
                    y="status", 
                    color="status",
                    size="impact_score", 
                    hover_data=["text"],
                    color_discrete_map={"Critical": "red", "High": "orange", "Moderate": "gold", "Low": "green"},
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
        # Define columns to show - include the new ones if they exist
        cols = ['created_utc', 'location_name', 'status', 'impact_score', 'risk_factors', 'text', 'url']
        cols = [c for c in cols if c in filtered_df.columns]
        
        # Sort by Impact Score so the worst stuff is at the top
        if 'impact_score' in filtered_df.columns:
            df_sorted = filtered_df.sort_values(by='impact_score', ascending=False)
        else:
            df_sorted = filtered_df

        st.dataframe(
            df_sorted[cols],
            column_config={
                "url": st.column_config.LinkColumn("Link"),
                "impact_score": st.column_config.ProgressColumn("Impact Score", min_value=0, max_value=100, format="%d"),
                "status": st.column_config.Column("Risk Level"),
                "risk_factors": st.column_config.Column("Risk Factors"),
                "text": st.column_config.TextColumn("Content", width="large"),
                "sentiment": st.column_config.ProgressColumn("Sentiment", min_value=-1, max_value=1) # Added sentiment column config just in case it's included
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No records found.")