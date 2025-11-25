import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sqlalchemy import create_engine
import os

st.set_page_config(page_title="CrisisGuard", layout="wide")

# Database
try: DB_STRING = st.secrets["DB_CONNECTION_STRING"]
except: DB_STRING = os.getenv('DB_CONNECTION_STRING')

@st.cache_data(ttl=60)
def get_data():
    try:
        engine = create_engine(DB_STRING)
        # Get distinct rows to avoid duplicates if script ran twice
        return pd.read_sql("SELECT DISTINCT * FROM crisis_events_v4 ORDER BY created_utc DESC LIMIT 1000", engine)
    except: return pd.DataFrame()

raw_df = get_data()

if raw_df.empty:
    st.warning("Database empty. Pipeline is gathering initial data...")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header(" Controls")

# 1. Risk Filter
all_statuses = sorted(raw_df['status'].unique())
selected_status = st.sidebar.multiselect("Risk Level", all_statuses, default=all_statuses)

# 2. Location Filter
# We normalize the list so "USA" and "US" are merged in the backend, 
# but here we just show the clean names stored in DB.
all_locs = sorted(raw_df['location_name'].unique())
selected_loc = st.sidebar.selectbox("Filter by Region", ["All"] + all_locs)

# Apply Filters
if selected_loc == "All":
    df = raw_df[raw_df['status'].isin(selected_status)]
else:
    df = raw_df[
        (raw_df['status'].isin(selected_status)) & 
        (raw_df['location_name'] == selected_loc)
    ]

# --- MAIN DASHBOARD ---
st.title(f" CrisisGuard: {selected_loc if selected_loc != 'All' else 'Global View'}")

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Active Events", len(df))
c2.metric("Critical Alerts", len(df[df['status']=="Critical"]))
if not df.empty:
    c3.metric("Avg Sentiment", f"{df['sentiment'].mean():.2f}")

# Map
st.subheader("Live Map")
if not df.empty:
    # Center on data
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    for _, row in df.iterrows():
        color = "red" if row['status'] == "Critical" else "orange"
        folium.Marker(
            [row['lat'], row['lon']],
            popup=f"<b>{row['location_name']}</b><br>{row['status']}",
            icon=folium.Icon(color=color)
        ).add_to(m)
    st_folium(m, width=1200, height=500)
else:
    st.info("No events match your current filters.")

# Data Table
st.subheader("Event Feed")
st.dataframe(
    df[['created_utc', 'location_name', 'status', 'text', 'url']],
    column_config={
        "url": st.column_config.LinkColumn("Link"),
        "text": st.column_config.TextColumn("Post Content", width="large")
    },
    hide_index=True
)