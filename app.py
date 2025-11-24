import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sqlalchemy import create_engine
import os

# --- CONFIG ---
st.set_page_config(page_title="Crisis Geo-Monitor", layout="wide")

try:
    # Try loading from Streamlit Cloud Secrets first
    DB_STRING = st.secrets["DB_CONNECTION_STRING"]
except:
    # Fallback to local environment
    DB_STRING = os.getenv('DB_CONNECTION_STRING')

# --- DATA LOADING ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def load_data():
    if not DB_STRING:
        return pd.DataFrame()
        
    engine = create_engine(DB_STRING)
    try:
        # Only get rows with valid coordinates
        query = """
        SELECT * FROM crisis_events_v2 
        WHERE lat IS NOT NULL 
        ORDER BY created_utc DESC 
        LIMIT 100
        """
        return pd.read_sql(query, engine)
    except:
        return pd.DataFrame()

# --- UI LAYOUT ---
st.title(" Real-Time Crisis Geolocation")
st.markdown("Live NLP extraction of high-risk events from Social Media.")

df = load_data()

if df.empty:
    st.info("Waiting for data pipeline... (No geolocated risks found yet)")
    st.stop()

# METRICS
col1, col2, col3 = st.columns(3)
col1.metric("Active Alerts", len(df))
col1.metric("Avg Risk Score", f"{round(df['risk_score'].mean() * 100)}%")
col3.metric("Top Location", df['location_name'].mode()[0])

# MAP
st.subheader("Live Incident Map")
# Center map on the first available data point
start_lat = df.iloc[0]['lat']
start_lon = df.iloc[0]['lon']
m = folium.Map(location=[start_lat, start_lon], zoom_start=3)

for index, row in df.iterrows():
    # Color logic: Very High Risk = Red, High = Orange
    color = "red" if row['risk_score'] > 0.7 else "orange"
    
    folium.Marker(
        [row['lat'], row['lon']],
        popup=f"<b>{row['location_name']}</b><br>Risk: {round(row['risk_score'], 2)}<br>r/{row['subreddit']}",
        icon=folium.Icon(color=color, icon="info-sign")
    ).add_to(m)

st_folium(m, width=1200, height=500)

# DATA FEED
st.subheader("Incoming Feed")
st.dataframe(
    df[['created_utc', 'location_name', 'subreddit', 'text', 'risk_score', 'url']],
    column_config={
        "url": st.column_config.LinkColumn("Source Link"),
        "risk_score": st.column_config.ProgressColumn("Risk Level", min_value=0, max_value=1)
    },
    hide_index=True
)