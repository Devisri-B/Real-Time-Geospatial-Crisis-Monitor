import praw
import pandas as pd
import re
import os
import datetime
import time
import pickle
import spacy
from geopy.geocoders import Nominatim
from sqlalchemy import create_engine, text
from textblob import TextBlob
import contractions
import nltk
from nltk.corpus import stopwords

# --- CONFIG ---
REDDIT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
DB_STRING = os.getenv('DB_CONNECTION_STRING')

# How many days of data to keep?
DATA_RETENTION_DAYS = 30 

# --- INIT MODELS ---
print("Loading AI Models...")
try:
    nltk.download('stopwords', quiet=True)
    try: nlp = spacy.load("en_core_web_sm")
    except: 
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f: model = pickle.load(f)
except Exception as e:
    print(f"Model Loading Failed: {e}")
    exit()

geolocator = Nominatim(user_agent="crisis_monitor_v5_cleanup")
stop = set(stopwords.words('english')) - {"not", "no"}

SUBREDDITS = ['mentalhealth', 'depression', 'SuicideWatch', 'anxiety',
    'stress', 'offmychest', 'lonely', 'BPD', 'ptsd',
    'socialanxiety', 'bipolar', 'addiction', 'traumatoolbox', 'CPTSD',
    'selfharm', 'OCD', 'EatingDisorders', 'MentalHealthSupport',
    'schizophrenia', 'insomnia', 'panicattacks', 'ADHD', 'AskReddit',
    'worldnews', 'news']

# --- CORE FUNCTIONS ---

def get_reddit():
    if not REDDIT_ID: return []
    reddit = praw.Reddit(client_id=REDDIT_ID, client_secret=REDDIT_SECRET, user_agent='crisis_v5')
    posts = []
    for sub in SUBREDDITS:
        try:
            for post in reddit.subreddit(sub).new(limit=50):
                posts.append({
                    'id': post.id,
                    'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
                    'subreddit': sub,
                    'text': f"{post.title} {post.selftext}"[:2000], # Truncate to save space
                    'url': post.url
                })
        except: pass
    return pd.DataFrame(posts)

def clean_for_model(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop])

def smart_geocode(text):
    doc = nlp(text)
    candidates = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    
    # Blocklist to prevent false positives
    blocklist = ["AI", "Phobia", "Pandora", "Olivia", "Imagine", "Meth", "Help", "Reddit", "YouTube", "Zoom", "Discord"]

    for loc_name in candidates:
        if len(loc_name) < 3 or loc_name.title() in blocklist: continue
        try:
            time.sleep(1) 
            location = geolocator.geocode(loc_name, addressdetails=True, language='en')
            if location and location.raw.get('class') in ['boundary', 'place']:
                address = location.raw.get('address', {})
                clean_name = address.get('country') or address.get('city') or location.address.split(",")[0]
                return clean_name, location.latitude, location.longitude
        except: continue
    return None, None, None

def cleanup_old_data():
    """Deletes rows older than DATA_RETENTION_DAYS to free up space."""
    print(f"5. Maintenance: Cleaning data older than {DATA_RETENTION_DAYS} days...")
    engine = create_engine(DB_STRING)
    
    # Calculate cutoff date
    cutoff = datetime.datetime.now() - datetime.timedelta(days=DATA_RETENTION_DAYS)
    
    try:
        with engine.connect() as conn:
            # Using SQLAlchemy 'text' for raw SQL
            query = text(f"DELETE FROM crisis_events_v4 WHERE created_utc < '{cutoff}'")
            result = conn.execute(query)
            conn.commit()
            print(f"   - Deleted {result.rowcount} old records.")
    except Exception as e:
        print(f"   - Cleanup warning: {e}")

def run_pipeline():
    print("1. Extracting Data...")
    df = get_reddit()
    if df.empty: return

    print("2. Risk Analysis...")
    df['clean_text'] = df['text'].apply(clean_for_model)
    features = vectorizer.transform(df['clean_text'])
    df['risk_class'] = model.predict(features) 
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    def get_status(row):
        if row['risk_class'] == 2 and row['sentiment'] < -0.1: return "Critical"
        if row['risk_class'] >= 1 and row['sentiment'] < -0.2: return "High"
        if row['risk_class'] >= 1: return "Moderate"
        return "Low"

    df['status'] = df.apply(get_status, axis=1)
    active_df = df[df['status'] != "Low"].copy()
    
    print(f"3. Geolocating {len(active_df)} alerts...")
    geo_results = active_df['text'].apply(smart_geocode)
    active_df['location_name'] = [res[0] for res in geo_results]
    active_df['lat'] = [res[1] for res in geo_results]
    active_df['lon'] = [res[2] for res in geo_results]
    
    final_df = active_df.dropna(subset=['lat'])
    
    print(f"4. Saving {len(final_df)} rows...")
    if not final_df.empty:
        engine = create_engine(DB_STRING)
        cols = ['id', 'created_utc', 'subreddit', 'text', 'status', 'sentiment', 'url', 'location_name', 'lat', 'lon']
        final_df[cols].to_sql('crisis_events_v4', engine, if_exists='append', index=False)
        print("   - Upload Success.")

    # Run Cleanup at the end
    cleanup_old_data()

if __name__ == "__main__":
    run_pipeline()