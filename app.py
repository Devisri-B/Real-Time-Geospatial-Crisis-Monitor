import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sqlalchemy import create_engine
import plotly.express as px
import os
from datetime import timedelta

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="CrisisGuard Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    /* Existing Light Mode Metrics Style */
    div[data-testid="stMetric"] {
        background-color: #F0F2F6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF4B4B;
        color: #31333F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Force charts to have white background */
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }

    /* --- INCREASE TAB HEADER SIZE --- */
    button[data-baseweb="tab"] div p {
        font-size: 20px !important;  /* Change this number to make it bigger/smaller */
        font-weight: 600 !important; /* Make it bold */
    }
    
    /* Increase icon size in tabs */
    button[data-baseweb="tab"] div {
        gap: 8px; /* Adds space between emoji and text */
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
        # Fetch latest 1000 entries
        df = pd.read_sql("SELECT * FROM crisis_events_v4 ORDER BY created_utc DESC LIMIT 1000", engine)
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        return df
    except:
        return pd.DataFrame()

raw_df = get_data()

if raw_df.empty:
    st.error("⚠️ Database Connection Established, but table is empty. Wait for the ETL Pipeline (GitHub Action) to finish running.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Command Center")

# A. Time Slider
if not raw_df.empty:
    min_time = raw_df['created_utc'].min()
    max_time = raw_df['created_utc'].max()
    
    if min_time == max_time:
        time_range = (min_time, max_time + timedelta(hours=1))
    else:
        time_range = st.sidebar.slider(
            "Filter Timeline",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(),
            value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
            format="MM/DD HH:mm"
        )

# B. Risk Filter
all_statuses = sorted(raw_df['status'].unique())
selected_status = st.sidebar.multiselect("Severity Level", all_statuses, default=all_statuses)

# C. Location Filter
all_locs = sorted(raw_df['location_name'].unique())
selected_loc = st.sidebar.selectbox("Target Region", ["All Global Regions"] + all_locs)

# --- FILTER LOGIC ---
mask = (raw_df['created_utc'] >= time_range[0]) & (raw_df['created_utc'] <= time_range[1]) & (raw_df['status'].isin(selected_status))
filtered_df = raw_df[mask]

if selected_loc != "All Global Regions":
    filtered_df = filtered_df[filtered_df['location_name'] == selected_loc]

# D. Alert Simulation
st.sidebar.divider()
st.sidebar.subheader("🚨 Emergency Dispatch")
alert_email = st.sidebar.text_input("Officer Email", placeholder="admin@agency.gov")
if st.sidebar.button("Test Alert System"):
    critical_count = len(filtered_df[filtered_df['status'] == 'Critical'])
    if critical_count > 0:
        st.sidebar.success(f" Alert Sent: {critical_count} critical events flagged in {selected_loc}")
    else:
        st.sidebar.info("No critical events to report at this time.")

# --- 4. MAIN DASHBOARD UI ---
st.title(f"🛡️ CrisisGuard: {selected_loc}")
st.markdown(f"Monitoring **{len(filtered_df)}** active signals. Last update: {max_time.strftime('%H:%M UTC')}")

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Incidents", len(filtered_df))
c2.metric("Critical Threats", len(filtered_df[filtered_df['status'] == "Critical"]), delta_color="inverse")

if not filtered_df.empty:
    avg_sent = filtered_df['sentiment'].mean()
    c3.metric("Avg Sentiment", f"{avg_sent:.2f}", delta="-0.1" if avg_sent < 0 else "0.1")
    
    top_source = filtered_df['subreddit'].mode()[0]
    c4.metric("Primary Source", f"r/{top_source}")

# --- 5. TABS ---
tab_geo, tab_analysis, tab_feed = st.tabs(["🌍 Geospatial Ops", "📊 Risk Analytics", "📋 Data Feed"])

# TAB 1: MAP (UPDATED FOR LIGHT MODE)
with tab_geo:
    if not filtered_df.empty:
        start_lat = filtered_df['lat'].mean()
        start_lon = filtered_df['lon'].mean()
        zoom = 4 if selected_loc != "All Global Regions" else 2
        
        m = folium.Map(location=[start_lat, start_lon], zoom_start=zoom, tiles="CartoDB positron")
        
        for idx, row in filtered_df.iterrows():
            color = "#FF0000" if row['status'] == "Critical" else "#FFA500"
            if row['status'] == "Moderate": color = "#FFD700" 
            
            # Rich Popup HTML
            html = f"""
            <div style="font-family: sans-serif; width: 200px; color: #333;">
                <h5 style="margin:0;">{row['location_name']}</h5>
                <span style="color:{color}; font-weight:bold;">{row['status']}</span><br>
                <small>{row['created_utc']}</small>
                <p>{row['text'][:100]}...</p>
                <a href="{row['url']}" target="_blank">View Source</a>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(html, max_width=260)
            ).add_to(m)
        
        st_folium(m, width=None, height=500)
    else:
        st.warning("No geolocation data available for current filters.")

# TAB 2: ANALYTICS 
with tab_analysis:
    if not filtered_df.empty:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Risk vs. Sentiment Correlation")
            fig_scatter = px.scatter(
                filtered_df, 
                x="sentiment", 
                y="status", 
                color="status",
                size_max=10,
                hover_data=["text"],
                color_discrete_map={"Critical": "red", "High": "orange", "Moderate": "#FFD700", "Low": "green"},
                template="plotly_white" # <--- Force white background
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with c2:
            st.subheader("Incident Volume by Source")
            source_counts = filtered_df['subreddit'].value_counts().reset_index()
            source_counts.columns = ['Subreddit', 'Count']
            fig_bar = px.bar(
                source_counts, 
                x='Subreddit', 
                y='Count', 
                color='Count', 
                color_continuous_scale='Reds',
                template="plotly_white" # <--- Force white background
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Not enough data to generate analytics.")

# TAB 3: RAW DATA
with tab_feed:
    st.subheader("Live Intelligence Feed")
    if not filtered_df.empty:
        st.dataframe(
            filtered_df[['created_utc', 'location_name', 'status', 'sentiment', 'text', 'url']],
            column_config={
                "url": st.column_config.LinkColumn("Link"),
                "sentiment": st.column_config.ProgressColumn("Sentiment", min_value=-1, max_value=1),
                "status": st.column_config.Column("Risk Level")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No records found.")