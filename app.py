import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sqlalchemy import create_engine
import plotly.express as px
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="CrisisGuard Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATABASE CONNECTION ---
try:
    DB_STRING = st.secrets["DB_CONNECTION_STRING"]
except:
    DB_STRING = os.getenv('DB_CONNECTION_STRING')

@st.cache_data(ttl=60) # Refresh data every minute
def get_data():
    try:
        engine = create_engine(DB_STRING)
        # Fetch latest 500 entries
        return pd.read_sql("SELECT * FROM crisis_events_v4 ORDER BY created_utc DESC LIMIT 500", engine)
    except:
        return pd.DataFrame()

# Load Data
raw_df = get_data()

if raw_df.empty:
    st.error("No data connection. Waiting for pipeline...")
    st.stop()

# --- 3. SIDEBAR: THE CONTROL CENTER ---
st.sidebar.header(" Command Controls")

# A. Risk Level Filter (Your Request)
risk_options = raw_df['status'].unique().tolist()
selected_risks = st.sidebar.multiselect(
    "Filter by Severity",
    options=risk_options,
    default=risk_options # Select all by default
)

# B. Location Filter (Your Request)
# Get unique locations plus an "All" option
loc_options = ["All Global Regions"] + sorted(raw_df['location_name'].unique().tolist())
selected_loc = st.sidebar.selectbox("Focus Region", loc_options)

# C. Apply Filters to create the "Active View" DataFrame
if selected_loc == "All Global Regions":
    filtered_df = raw_df[raw_df['status'].isin(selected_risks)]
else:
    filtered_df = raw_df[
        (raw_df['status'].isin(selected_risks)) & 
        (raw_df['location_name'] == selected_loc)
    ]

# D. "Innovative Feature": Alert Simulator
st.sidebar.divider()
st.sidebar.subheader("System Actions")
if st.sidebar.button("Simulate Dispatch Alert"):
    if not filtered_df.empty:
        top_event = filtered_df.iloc[0]
        st.sidebar.success(f"ALERT SENT: High risk detected in {top_event['location_name']}")
        # In real life, this would trigger Twilio/PagerDuty
    else:
        st.sidebar.warning("No active events to alert.")

# --- 4. MAIN DASHBOARD ---

st.title(f" CrisisGuard: {selected_loc}")
st.markdown(f"Monitoring **{len(filtered_df)}** active signals based on applied filters.")

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Incidents", len(filtered_df))
c2.metric("Critical Threats", len(filtered_df[filtered_df['status'] == "Critical"]), delta_color="inverse")
c3.metric("Avg Sentiment Score", f"{filtered_df['sentiment'].mean():.2f}")
# Dynamic Metric: Shows top keyword if data exists
if not filtered_df.empty:
    top_source = filtered_df['subreddit'].mode()[0]
    c4.metric("Primary Source", f"r/{top_source}")

# TABS LAYOUT
tab_geo, tab_analysis, tab_raw = st.tabs([" Geospatial Ops", " Risk Analytics", " Data Inspector"])

# --- TAB 1: THE MAP (Filtered) ---
with tab_geo:
    col_map, col_details = st.columns([3, 1])
    
    with col_map:
        if not filtered_df.empty:
            # Center map on selection
            start_lat = filtered_df['lat'].mean()
            start_lon = filtered_df['lon'].mean()
            zoom = 4 if selected_loc != "All Global Regions" else 2
            
            m = folium.Map(location=[start_lat, start_lon], zoom_start=zoom, tiles="CartoDB dark_matter")
            
            for _, row in filtered_df.iterrows():
                # Color Logic
                if row['status'] == "Critical": color = "#FF0000"  # Red
                elif "High" in row['status']: color = "#FFA500"    # Orange
                else: color = "#00FF00"                            # Green
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"<b>{row['location_name']}</b><br>{row['status']}"
                ).add_to(m)
            
            st_folium(m, width=None, height=500)
        else:
            st.info("No geolocation data matches your filters.")
            
    with col_details:
        st.subheader("Region Details")
        if selected_loc != "All Global Regions":
            st.write(f"**Target:** {selected_loc}")
            st.write(f"**Reports:** {len(filtered_df)}")
            # Show summary of text in this region
            st.text_area("Content Sample", filtered_df['text'].iloc[0][:200] + "...", height=150)
        else:
            st.write("Select a specific region in the sidebar to see granular details.")

# --- TAB 2: ANALYTICS (Innovative Charts) ---
with tab_analysis:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Risk vs. Sentiment Scatter")
        # This chart is "Innovative" because it visually proves your hypothesis:
        # Lower Sentiment (more negative) should correlate with Critical Status.
        fig_scatter = px.scatter(
            filtered_df, 
            x="sentiment", 
            y="status", 
            color="status",
            hover_data=["text"],
            title="Sentiment Polarity vs Risk Classification"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c2:
        st.subheader("Volume by Source")
        source_counts = filtered_df['subreddit'].value_counts().reset_index()
        source_counts.columns = ['Subreddit', 'Count']
        fig_bar = px.bar(source_counts, x='Subreddit', y='Count', color='Count', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: DATA FEED (Filtered) ---
with tab_raw:
    st.subheader(f"Live Feed: {selected_loc}")
    
    # Show clickable links and formatted status
    st.dataframe(
        filtered_df[['created_utc', 'location_name', 'status', 'text', 'url']],
        column_config={
            "url": st.column_config.LinkColumn("Reddit Link"),
            "text": st.column_config.TextColumn("Post Content", width="large"),
            "status": st.column_config.Column(
                "Risk Status",
                help="Critical = High ML Score + Negative Sentiment",
                width="medium"
            )
        },
        hide_index=True,
        use_container_width=True
    )