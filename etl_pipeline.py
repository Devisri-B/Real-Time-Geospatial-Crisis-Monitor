import praw
import pandas as pd
import re
import os
import datetime
import time
import spacy
from geopy.geocoders import Nominatim
from sqlalchemy import create_engine
from textblob import TextBlob

# --- CONFIGURATION ---
REDDIT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
DB_STRING = os.getenv('DB_CONNECTION_STRING')

# 1. Load NLP Model (Auto-download if missing)
print("Loading NLP Model...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 2. Initialize Geocoder (The tool that turns "London" into "51.5, -0.1")
geolocator = Nominatim(user_agent="crisis_monitor_production_v2")

# Target Communities
SUBREDDITS = [
    'mentalhealth', 'depression', 'SuicideWatch', 'anxiety',
    'stress', 'legaladvice', 'offmychest', 'worldnews' 
]
# Added 'legaladvice' and 'offmychest' as they often contain location info

def get_reddit_data():
    """Connect to Reddit and fetch posts"""
    if not REDDIT_ID:
        print("Error: No credentials found.")
        return []

    reddit = praw.Reddit(
        client_id=REDDIT_ID,
        client_secret=REDDIT_SECRET,
        user_agent='crisis_monitor_v2'
    )
    
    all_posts = []
    print("Scanning Reddit...")
    
    for sub in SUBREDDITS:
        try:
            # Fetch more posts (50) to increase chance of finding locations
            for post in reddit.subreddit(sub).new(limit=50):
                all_posts.append({
                    'id': post.id,
                    'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
                    'subreddit': sub,
                    'text': f"{post.title} {post.selftext}"[:1000], # Limit text size
                    'url': post.url
                })
        except Exception as e:
            print(f"Skipped r/{sub}: {e}")
            
    return pd.DataFrame(all_posts)

def get_risk_score(text):
    """Simple Sentiment Analysis Risk Score"""
    blob = TextBlob(str(text))
    sentiment = blob.sentiment.polarity # -1 to 1
    # Normalize: -1 (Bad) becomes 1.0 (High Risk)
    risk = (sentiment * -1) + 0.3 
    return max(0.0, min(1.0, risk))

def extract_geolocation(text):
    """
    1. Finds a place name using spaCy.
    2. Converts it to Lat/Lon using Geopy.
    """
    doc = nlp(text)
    
    # Look for Geopolitical Entities (Cities, Countries, States)
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            location_name = ent.text
            try:
                # Delay to be polite to the API
                time.sleep(1) 
                loc = geolocator.geocode(location_name)
                if loc:
                    return location_name, loc.latitude, loc.longitude
            except:
                continue # Try the next entity if this one fails
                
    return None, None, None # No location found

def run_etl():
    # 1. Extract
    df = get_reddit_data()
    if df.empty:
        print("No data fetched.")
        return

    print(f"Processing {len(df)} raw posts...")

    # 2. Transform (Risk)
    df['risk_score'] = df['text'].apply(get_risk_score)
    
    # Filter: Only process locations for risky posts (Score > 0.5) to save time
    risky_df = df[df['risk_score'] > 0.4].copy()
    
    if risky_df.empty:
        print("No high-risk posts found.")
        return

    print(f"Geocoding {len(risky_df)} risky posts (this takes time)...")
    
    # 3. Transform (Location)
    # Apply function and split result into 3 new columns
    location_data = risky_df['text'].apply(extract_geolocation)
    
    risky_df['location_name'] = [x[0] for x in location_data]
    risky_df['lat'] = [x[1] for x in location_data]
    risky_df['lon'] = [x[2] for x in location_data]
    
    # DROP rows that didn't have a location
    final_df = risky_df.dropna(subset=['lat', 'lon'])
    
    print(f"Found {len(final_df)} events with confirmed locations.")

    # 4. Load
    if not final_df.empty:
        engine = create_engine(DB_STRING)
        try:
            final_df.to_sql('crisis_events_v2', engine, if_exists='append', index=False)
            print("Success: Data uploaded to Database.")
        except Exception as e:
            print(f"Database Insert Error: {e}")

if __name__ == "__main__":
    run_etl()