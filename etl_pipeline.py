import praw
import pandas as pd
import re
import os
import datetime
import spacy
from geopy.geocoders import Nominatim
from sqlalchemy import create_engine
from textblob import TextBlob
import time

# 1. CONFIGURATION
REDDIT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
DB_STRING = os.getenv('DB_CONNECTION_STRING')

# Your Subreddits
TARGET_SUBREDDITS = [
    'mentalhealth', 'depression', 'SuicideWatch', 'anxiety',
    'addiction', 'stress', 'traumatoolbox', 'BPD', 'CPTSD'
]

# Your Keywords
CRISIS_KEYWORDS = [
    "suicidal", "overwhelmed", "relapse", "addiction help",
    "self harm", "unwanted life", "end it all", "no way out", 
    "anxiety attack", "hopeless", "need help", "end life",
    "terminate life", "death wish", "want to die"
]

# Load NLP Model
print("Loading NLP Model...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize Geocoder
geolocator = Nominatim(user_agent="crisis_monitor_v1")

def get_reddit_instance():
    return praw.Reddit(
        client_id=REDDIT_ID,
        client_secret=REDDIT_SECRET,
        user_agent='crisis_monitor_v1'
    )

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text) 
    return text

def extract_location(text):
    """
    Uses spaCy to find GPE (Countries, Cities, States).
    Returns (name, lat, lon) or (None, None, None).
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            location_name = ent.text.strip()
            try:
                # We add a tiny delay to be polite to the geocoding API
                time.sleep(1) 
                loc = geolocator.geocode(location_name)
                if loc:
                    return location_name, loc.latitude, loc.longitude
            except:
                pass # If geocoding fails, try the next entity
    return None, None, None

def calculate_risk(text):
    blob = TextBlob(text)
    sentiment_risk = (blob.sentiment.polarity * -1) 
    keyword_hits = sum(1 for word in CRISIS_KEYWORDS if word in text)
    total_risk = sentiment_risk + (keyword_hits * 0.2)
    return max(0.0, min(1.0, total_risk))

def run_pipeline():
    print("--- Starting ETL Pipeline ---")
    if not REDDIT_ID:
        print("Error: Credentials missing.")
        return

    reddit = get_reddit_instance()
    posts_data = []

    # 1. EXTRACT
    for sub in TARGET_SUBREDDITS:
        print(f"Scanning r/{sub}...")
        try:
            # Fetch 10 new posts per subreddit to keep it fast
            for post in reddit.subreddit(sub).new(limit=10):
                full_text = f"{post.title} {post.selftext}"
                posts_data.append({
                    'id': post.id,
                    'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
                    'subreddit': sub,
                    'text': full_text[:500], # Truncate for DB storage
                    'url': post.url
                })
        except Exception as e:
            print(f"Error reading {sub}: {e}")

    if not posts_data:
        print("No data found.")
        return

    df = pd.DataFrame(posts_data)

    # 2. TRANSFORM
    print("Analyzing Risk & Locations...")
    df['clean_text'] = df['text'].apply(clean_text)
    df['risk_score'] = df['clean_text'].apply(calculate_risk)
    
    # Only process location for risky posts to save time
    risky_df = df[df['risk_score'] > 0.3].copy()
    
    # Apply Location Extraction (Returns tuple, we split it into columns)
    location_data = risky_df['text'].apply(extract_location)
    
    # Unpack the tuples into new columns
    risky_df['location_name'] = [x[0] for x in location_data]
    risky_df['lat'] = [x[1] for x in location_data]
    risky_df['lon'] = [x[2] for x in location_data]

    # Drop rows where no location was found (Optional: Keep them if you want a list view)
    # For this map-focused project, let's keep everything but only map valid ones.
    
    print(f"Found {len(risky_df)} high-risk events.")

    # 3. LOAD
    if not risky_df.empty:
        engine = create_engine(DB_STRING)
        # Save only necessary columns
        final_df = risky_df[['id', 'created_utc', 'subreddit', 'text', 'risk_score', 'url', 'location_name', 'lat', 'lon']]
        
        try:
            final_df.to_sql('crisis_events_v2', engine, if_exists='append', index=False)
            print(f"Success! Uploaded {len(final_df)} rows to database.")
        except Exception as e:
            print(f"Database Insert Error (likely duplicates): {e}")

if __name__ == "__main__":
    run_pipeline()