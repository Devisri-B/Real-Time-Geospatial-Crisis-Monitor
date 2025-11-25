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

geolocator = Nominatim(user_agent="crisis_monitor_v6_dedup")
stop = set(stopwords.words('english')) - {"not", "no"}
SUBREDDITS = ['mentalhealth', 'depression', 'SuicideWatch', 'anxiety',
    'stress', 'offmychest', 'lonely', 'BPD', 'ptsd',
    'socialanxiety', 'bipolar', 'addiction', 'traumatoolbox', 'CPTSD',
    'selfharm', 'OCD', 'EatingDisorders', 'MentalHealthSupport',
    'schizophrenia', 'insomnia', 'panicattacks', 'ADHD', 'AskReddit',
    'worldnews', 'news']

def get_reddit():
    if not REDDIT_ID: return []
    reddit = praw.Reddit(client_id=REDDIT_ID, client_secret=REDDIT_SECRET, user_agent='crisis_v6')
    posts = []
    for sub in SUBREDDITS:
        try:
            for post in reddit.subreddit(sub).new(limit=50):
                posts.append({
                    'id': post.id,
                    'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
                    'subreddit': sub,
                    'text': f"{post.title} {post.selftext}"[:2000],
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
    print(f"5. Maintenance: Cleaning data older than {DATA_RETENTION_DAYS} days...")
    engine = create_engine(DB_STRING)
    cutoff = datetime.datetime.now() - datetime.timedelta(days=DATA_RETENTION_DAYS)
    try:
        with engine.connect() as conn:
            query = text(f"DELETE FROM crisis_events_v4 WHERE created_utc < '{cutoff}'")
            result = conn.execute(query)
            conn.commit()
            print(f"   - Deleted {result.rowcount} old records.")
    except Exception as e:
        print(f"   - Cleanup warning: {e}")

# --- NEW: DEDUPLICATION FUNCTION ---
def filter_existing_posts(df, engine):
    """Checks DB for existing IDs and removes them from the dataframe."""
    if df.empty: return df
    
    new_ids = tuple(df['id'].tolist())
    if not new_ids: return df
    
    # Query DB for these specific IDs
    query = text(f"SELECT id FROM crisis_events_v4 WHERE id IN :ids")
    
    try:
        with engine.connect() as conn:
            existing = pd.read_sql(query, conn, params={"ids": new_ids})
            existing_ids = set(existing['id'].tolist())
            
        unique_df = df[~df['id'].isin(existing_ids)]
        print(f"   - Deduplication: {len(df)} fetched -> {len(unique_df)} new unique posts.")
        return unique_df
    except Exception as e:
        # If table doesn't exist yet, all posts are new
        return df

def run_pipeline():
    print("1. Extracting...")
    df = get_reddit()
    if df.empty: return

    # --- APPLY DEDUPLICATION ---
    engine = create_engine(DB_STRING)
    df = filter_existing_posts(df, engine)

    if df.empty:
        print("   - No new unique posts to process.")
        return

    print("2. Analysis...")
    df['clean_text'] = df['text'].apply(clean_for_model)
    features = vectorizer.transform(df['clean_text'])
    df['risk_class'] = model.predict(features) 
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    def get_status(row):
        r = row['risk_class']
        s = row['sentiment']
        if r == 2:
            if s < -0.05: return "Critical"
            return "High"
        if r == 1:
            if s < -0.3: return "Critical"
            if s < -0.1: return "High"
            return "Moderate"
        if s < -0.6: return "Moderate" 
        return "Low"

    df['status'] = df.apply(get_status, axis=1)
    active_df = df[df['status'] != "Low"].copy()
    
    print(f"3. Geolocating {len(active_df)} new alerts...")
    geo_results = active_df['text'].apply(smart_geocode)
    active_df['location_name'] = [res[0] for res in geo_results]
    active_df['lat'] = [res[1] for res in geo_results]
    active_df['lon'] = [res[2] for res in geo_results]
    
    final_df = active_df.dropna(subset=['lat'])
    
    print(f"4. Saving {len(final_df)} rows...")
    if not final_df.empty:
        cols = ['id', 'created_utc', 'subreddit', 'text', 'status', 'sentiment', 'url', 'location_name', 'lat', 'lon']
        final_df[cols].to_sql('crisis_events_v4', engine, if_exists='append', index=False)
        print("   - Success.")
    
    cleanup_old_data()

if __name__ == "__main__":
    run_pipeline()