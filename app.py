import datetime
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
import smtplib
from email.mime.text import MIMEText

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

@st.cache_data(ttl=600)
def get_data():
    try:
        engine = create_engine(DB_STRING)
        df = pd.read_sql("SELECT * FROM crisis_events_v4 ORDER BY created_utc DESC LIMIT 2000", engine)
        
        # FIX: Ensure columns exist even if empty
        required_cols = ['created_utc', 'sentiment', 'lat', 'lon', 'text', 'status', 'location_name', 'subreddit', 'url', 'impact_score', 'risk_factors']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
                
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        
        # Create SHORT text for hover
        if 'text' in df.columns:
            df['short_text'] = df['text'].astype(str).str.slice(0, 60) + "..."
        return df
    except Exception as e:
        # FIX: Show the actual error to help debugging
        st.error(f"Database Connection Error: {e}")
        return pd.DataFrame()

raw_df = get_data()

# --- 3. SCORING LOGIC ---
if not raw_df.empty:
    # Strict 3-Level Map
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
        # Create a dynamic key based on the latest time
        # This forces the slider to "reset" its position when new data arrives!
        slider_key = f"slider_{max_time.strftime('%Y%m%d%H%M%S')}"
        
        time_range = st.sidebar.slider(
            "Timeline",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(),
            value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
            format="MM/DD HH:mm",
            key=slider_key  
        )
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

# D. REAL Alert System
st.sidebar.divider()
st.sidebar.subheader("Emergency Dispatch")
alert_email = st.sidebar.text_input("Email", placeholder="admin@domain.com", help="Email to send Critical Incident Alerts", key="alert_email")

if st.sidebar.button("Send Critical Alert"):
    # Filter for Critical events in the current view
    critical_events = filtered_df[filtered_df['status'] == 'Critical']
    
    # This keeps the first instance and drops cross-posts
    critical_events = critical_events.drop_duplicates(subset=['text'])

    if not critical_events.empty:
        # --- 1. Prepare Rich Email Content ---
        count = len(critical_events)
        # Get current time for the report header
        report_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        
        subject = f"CrisisGuard ALERT: {count} Critical Incidents in {selected_loc}"
        
        # report format
        body = f"""
        CRISISGUARD INTELLIGENCE REPORT
        ==================================================
        Target Region: {selected_loc}
        Generated At:  {report_time}
        Active Threats: {count}
        ==================================================

        TOP PRIORITY INCIDENTS (High to Low Risk):
        """
        
        # Sort by Impact Score to show worst first, limit to top 5
        top_incidents = critical_events.sort_values('impact_score', ascending=False).head(5)
        
        for i, row in top_incidents.iterrows():
            # Handle missing values gracefully
            event_time = row['created_utc'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['created_utc']) else "N/A"
            score = row.get('impact_score', 'N/A')
            factors = row.get('risk_factors', 'Unknown Factors')
            text_snippet = str(row['text'])[:250].replace('\n', ' ') # Clean up newlines for email
            
            body += f"""
            [{i+1}] SEVERITY SCORE: {score}/100
            --------------------------------------------------
            • Location:   {row['location_name']}
            • Detected:   {event_time}
            • Why Flagged: {factors}
            • Sentiment:  {row.get('sentiment', 0):.2f}
            
            • Intel Snippet: 
              "{text_snippet}..."
              
            • SOURCE LINK: {row['url']}
            
            """
            
        body += f"""
        ==================================================
        End of Report.
        Please verify all intelligence on the CrisisGuard Dashboard https://real-time-geospatial-crisis-monitor.streamlit.app 
        """
            
        # --- 2. Send Email via SMTP ---
        try:
            # Load secrets safely
            sender = st.secrets["EMAIL_SENDER"]
            password = st.secrets["EMAIL_PASSWORD"]
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = alert_email
            
            # Connect to Gmail SMTP
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender, password)
                server.sendmail(sender, alert_email, msg.as_string())
                
            st.sidebar.success(f"Detailed Intelligence Report sent to {alert_email}")
            
        except Exception as e:
            st.sidebar.error(f"Email Failed. Did you set up st.secrets? Error: {e}")
            
    else:
        st.sidebar.info("No critical events found matching current filters.")


# --- 5. DASHBOARD ---
st.title(f"🛡️ CrisisGuard: {selected_loc}")
last_upd = raw_df['created_utc'].max().strftime('%B %d, %H:%M UTC') if not raw_df.empty else "N/A"
st.markdown(f"Monitoring **{len(filtered_df)}** active signals. <span style='color:gray'>Updated: {last_upd}</span>", unsafe_allow_html=True)

# Top Level Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Incidents", len(filtered_df))
crit = len(filtered_df[filtered_df['status']=="Critical"]) if not filtered_df.empty else 0
c2.metric("Critical Threats", crit, delta_color="inverse")
sent = filtered_df['sentiment'].mean() if not filtered_df.empty else 0
c3.metric("Avg Sentiment", f"{sent:.2f}")
src = filtered_df['subreddit'].mode()[0] if not filtered_df.empty else "N/A"
display_src = src if len(src) < 20 else f"{src[:20]}.."
c4.metric("Top Source", f"r/{display_src}", help=f"Full Name: r/{src}")

# SPLIT DATA into Mapped vs Unmapped
mapped_df = filtered_df.dropna(subset=['lat', 'lon'])
unmapped_df = filtered_df[filtered_df['lat'].isna()]

# --- TABS ---
tab_map, tab_mapped_analysis, tab_unmapped_analysis, tab_feed = st.tabs([
    "Live Map", 
    "Mapped Analysis", 
    "Unmapped Intel", 
    "Data Feed"
])

# TAB 1: MAP
with tab_map:
    if not mapped_df.empty:
        m = folium.Map(location=[mapped_df['lat'].mean(), mapped_df['lon'].mean()], zoom_start=2, tiles="CartoDB positron")
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in mapped_df.iterrows():
            np.random.seed(int(str(ord(row['id'][0])) + str(ord(row['id'][-1]))))
            jit_lat, jit_lon = np.random.uniform(-0.01, 0.01, 2)
            
            # Strict Colors
            stat = row['status']
            if stat == "Critical": color = "red"
            elif stat == "Moderate": color = "orange"
            else: color = "green"
             # Rich Popup
            html = f"""
            <div style="font-family: sans-serif; width: 200px;">
                <h5 style="margin:0;">{row['location_name']}</h5>
                <span style="color:{color}; font-weight:bold;">{row['status']}</span><br>
                <small>{row['created_utc']}</small>
                <p>{row['text'][:100]}...</p>
                <a href="{row['url']}" target="_blank">View Source</a>
            </div>
            """
            folium.Marker([row['lat']+jit_lat, row['lon']+jit_lon], popup=folium.Popup(html, max_width=260), icon=folium.Icon(color=color, icon="warning-sign")).add_to(marker_cluster)
        
        st_folium(m, width=None, height=500, returned_objects=[])
    else: st.warning("No location data available.")


# TAB 2: MAPPED ANALYSIS
with tab_mapped_analysis:
    if not mapped_df.empty:
        st.markdown(f"**Insight:** There are **{len(mapped_df)}** geolocated incidents")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hotspot Locations")
            # Bar chart of Top 15 Locations
            loc_counts = mapped_df['location_name'].value_counts().head(15).reset_index()
            loc_counts.columns = ['Location', 'Count']
            fig_loc = px.bar(loc_counts, x='Count', y='Location', orientation='h', 
                             title="Incidents by Region", template="plotly_white", color='Count')
            st.plotly_chart(fig_loc, use_container_width=True)
            
        with col2:
            st.subheader("Average Sentiment by Location")
            sent_loc = mapped_df.groupby('location_name')['sentiment'].mean().reset_index().sort_values('sentiment')
            fig_sent = px.bar(sent_loc, x='sentiment', y='location_name', orientation='h',
                              title="Avg Sentiment Score", template="plotly_white", color='sentiment')
            st.plotly_chart(fig_sent, use_container_width=True)
            
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top Sources of Mapped Risk")
            src_counts = mapped_df['subreddit'].value_counts().head(15).reset_index()
            src_counts.columns = ['Subreddit', 'Count']
            fig_src = px.bar(src_counts, x='Subreddit', y='Count', 
                            title="Volume by Source (Mapped)", template="plotly_white", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig_src, use_container_width=True)

        with c2:
            st.subheader("Risk Distribution (Geolocated)")
            fig_pie = px.pie(mapped_df, names='status', title="Severity of Mapped Events",
                             color='status',
                             color_discrete_map={"Critical": "red", "Moderate": "orange", "Low": "green"},
                             template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)

            
        st.subheader("Timeline of Mapped Events")
        fig_time = px.histogram(mapped_df, x="created_utc", color="status", 
                                color_discrete_map={"Critical": "red", "Moderate": "orange", "Low": "green"},
                                template="plotly_white")
        st.plotly_chart(fig_time, use_container_width=True)

    
    else:
        st.info("No geolocated data to analyze.")

# TAB 3: UNMAPPED ANALYSIS
with tab_unmapped_analysis:
    if not unmapped_df.empty:
        st.markdown(f"**Insight:** There are **{len(unmapped_df)}** incidents that could not be geolocated. These are the posts lacking specific geographic keywords.")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Sources of Hidden Risk")
            # Since we don't have location, Source (Subreddit) is the best proxy
            src_counts = unmapped_df['subreddit'].value_counts().head(15).reset_index()
            src_counts.columns = ['Subreddit', 'Count']
            fig_src = px.bar(src_counts, x='Subreddit', y='Count', 
                             title="Volume by Source (Unmapped)", template="plotly_white", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig_src, use_container_width=True)
            
        with col2:
            st.subheader("Risk Distribution (Hidden)")
            fig_pie_un = px.pie(unmapped_df, names='status', title="Severity of Unmapped Events",
                                color='status',
                                color_discrete_map={"Critical": "red", "Moderate": "orange", "Low": "green"},
                                template="plotly_white")
            st.plotly_chart(fig_pie_un, use_container_width=True)
            
        st.subheader("Timeline of Hidden Events")
        fig_time_un = px.histogram(unmapped_df, x="created_utc", color="status", 
                                   color_discrete_map={"Critical": "red", "Moderate": "orange", "Low": "green"},
                                   template="plotly_white")
        st.plotly_chart(fig_time_un, use_container_width=True)
        
    else:
        st.success("All current data has been successfully geolocated!")

# TAB 4: FULL FEED
with tab_feed:
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
    else:
        st.write("No records found.")