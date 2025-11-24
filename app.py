import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Crisis Intelligence", layout="wide")

# 1. Database Connection
try:
    DB_STRING = st.secrets["DB_CONNECTION_STRING"]
except:
    import os
    DB_STRING = os.getenv('DB_CONNECTION_STRING')

@st.cache_data(ttl=600)
def get_data():
    engine = create_engine(DB_STRING)
    # Get data that has location coordinates
    query = """
    SELECT * FROM crisis_events_v2 
    WHERE lat IS NOT NULL 
    ORDER BY created_utc DESC 
    LIMIT 100
    """
    return pd.read_sql(query, engine)

# 2. Dashboard Header
st.title("Real-Time Crisis Geolocation")
st.markdown("NLP-detected risks mapped from Reddit live feeds.")

try:
    df = get_data()
    
    # 3. Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Incidents", len(df))
    col1.metric("Avg Risk Level", f"{round(df['risk_score'].mean() * 100)}%")
    # Top location mentioned
    if not df.empty:
        top_loc = df['location_name'].mode()[0]
        col3.metric("Hotspot", top_loc)

    # 4. The Real Map
    # Center map on the first data point or default to world view
    start_loc = [df.iloc[0]['lat'], df.iloc[0]['lon']] if not df.empty else [20, 0]
    m = folium.Map(location=start_loc, zoom_start=2)

    for index, row in df.iterrows():
        # Color code by risk
        color = 'red' if row['risk_score'] > 0.8 else 'orange'
        
        folium.Marker(
            [row['lat'], row['lon']],
            popup=f"<b>{row['location_name']}</b><br>r/{row['subreddit']}<br>Risk: {row['risk_score']:.2f}",
            icon=folium.Icon(color=color, icon="warning-sign")
        ).add_to(m)

    st_folium(m, width=1000, height=500)

    # 5. Detailed Feed
    st.subheader("Incident Feed")
    st.dataframe(
        df[['created_utc', 'location_name', 'subreddit', 'text', 'risk_score', 'url']],
        column_config={
            "url": st.column_config.LinkColumn("Link"),
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1)
        }
    )

except Exception as e:
    st.info("Waiting for pipeline to run and detect locations...")
    st.error(f"Details: {e}")