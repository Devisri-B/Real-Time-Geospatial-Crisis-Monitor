import praw
import pandas as pd
import re
import os
import datetime
import time
import pickle
import spacy
from geopy.geocoders import Nominatim
from sqlalchemy import create_engine
from textblob import TextBlob
import contractions
import nltk
from nltk.corpus import stopwords

# --- CONFIG ---
REDDIT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
DB_STRING = os.getenv('DB_CONNECTION_STRING')

# Initialize NLP & Geo
print("Loading AI Models...")
try:
    nltk.download('stopwords', quiet=True)
    try: nlp = spacy.load("en_core_web_sm")
    except: 
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Load ML Models (Ensure these are in your repo!)
    with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f: model = pickle.load(f)
except Exception as e:
    print(f"Model Loading Failed: {e}")
    exit()

# Geocoder setup
geolocator = Nominatim(user_agent="crisis_monitor_v4_smart")
stop = set(stopwords.words('english')) - {"not", "no"}

SUBREDDITS = [ 'mentalhealth', 'depression', 'SuicideWatch', 'anxiety',
    'stress', 'offmychest', 'lonely', 'BPD', 'ptsd',
    'socialanxiety', 'bipolar', 'addiction', 'traumatoolbox', 'CPTSD',
    'selfharm', 'OCD', 'EatingDisorders', 'MentalHealthSupport',
    'schizophrenia', 'insomnia', 'panicattacks', 'ADHD', 'AskReddit',
    'worldnews', 'news']

# --- CORE FUNCTIONS ---

def get_reddit():
    if not REDDIT_ID: return []
    reddit = praw.Reddit(client_id=REDDIT_ID, client_secret=REDDIT_SECRET, user_agent='crisis_v4')
    posts = []
    for sub in SUBREDDITS:
        try:
            # Grab 50 to ensure we find enough valid locations
            for post in reddit.subreddit(sub).new(limit=50):
                posts.append({
                    'id': post.id,
                    'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
                    'subreddit': sub,
                    'text': f"{post.title} {post.selftext}"[:2000], # ORIGINAL TEXT for display
                    'url': post.url
                })
        except: pass
    return pd.DataFrame(posts)

def clean_for_model(text):
    """Aggressive cleaning ONLY for the XGBoost Model. Does not affect display text."""
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop])

def smart_geocode(text):
    """
    Extracts location and VALIDATES it is a real place (City/State/Country).
    Returns: (Official Name, Lat, Lon)
    """
    doc = nlp(text)
    # Only look for 'GPE' (Geo-Political Entity), ignore general 'LOC' (which catches mountains/rivers)
    candidates = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    
    for loc_name in candidates:
        # 1. Pre-Filter: Skip obviously bad short words or common false positives
        if len(loc_name) < 3 or loc_name.lower() in ['help', 'reddit', 'zoom']:
            continue
            
        try:
            time.sleep(1) # Rate limit polite
            # Request address details to check type
            location = geolocator.geocode(loc_name, addressdetails=True, language='en')
            
            if location:
                # 2. VALIDATION: Is this actually a place people live?
                # We check the 'type' of the result.
                # Valid types: city, state, country, administrative, town, village, hamlet
                loc_type = location.raw.get('type', '')
                loc_class = location.raw.get('class', '')
                
                # Reject "Amenity" (Shops), "Highway" (Roads), "Building", etc.
                valid_classes = ['boundary', 'place']
                
                if loc_class in valid_classes:
                    # 3. NORMALIZATION: Use the official name from Geopy
                    # This turns "US", "USA", "United States" all into -> "United States"
                    # It turns "NYC", "New York City" -> "New York"
                    
                    # We try to get a clean name (City or Country)
                    address = location.raw.get('address', {})
                    clean_name = address.get('country') or address.get('city') or address.get('state') or location.address.split(",")[0]
                    
                    return clean_name, location.latitude, location.longitude
        except:
            continue
            
    return None, None, None

def run_pipeline():
    print("1. Extracting Data...")
    df = get_reddit()
    if df.empty: return

    print("2. Risk Analysis...")
    # Clean copy for model
    df['clean_text'] = df['text'].apply(clean_for_model)
    
    # XGBoost Prediction
    features = vectorizer.transform(df['clean_text'])
    df['risk_class'] = model.predict(features) # 2=High, 1=Mod, 0=Low
    
    # TextBlob Verification
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Assign Status
    def get_status(row):
        if row['risk_class'] == 2 and row['sentiment'] < -0.1: return "Critical"
        if row['risk_class'] >= 1 and row['sentiment'] < -0.2: return "High"
        if row['risk_class'] >= 1: return "Moderate"
        return "Low"

    df['status'] = df.apply(get_status, axis=1)
    
    # Filter: Drop Low risk
    active_df = df[df['status'] != "Low"].copy()
    
    print(f"3. Geolocating {len(active_df)} alerts...")
    
    # Run Smart Geocode
    # This returns a tuple (Name, Lat, Lon). We expand it into columns.
    geo_results = active_df['text'].apply(smart_geocode)
    
    active_df['location_name'] = [res[0] for res in geo_results]
    active_df['lat'] = [res[1] for res in geo_results]
    active_df['lon'] = [res[2] for res in geo_results]
    
    # Drop rows where no valid location was found
    final_df = active_df.dropna(subset=['lat'])
    
    print(f"4. Loading {len(final_df)} verified rows...")
    
    if not final_df.empty:
        engine = create_engine(DB_STRING)
        # Save RAW text for display, plus normalized location
        cols = ['id', 'created_utc', 'subreddit', 'text', 'status', 'sentiment', 'url', 'location_name', 'lat', 'lon']
        final_df[cols].to_sql('crisis_events_v4', engine, if_exists='append', index=False)
        print("Success.")

if __name__ == "__main__":
    run_pipeline()