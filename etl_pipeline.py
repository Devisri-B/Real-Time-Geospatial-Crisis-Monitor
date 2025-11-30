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
import numpy as np
from xgboost import XGBClassifier

# --- CONFIG ---
REDDIT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
DB_STRING = os.getenv('DB_CONNECTION_STRING')
DATA_RETENTION_DAYS = 15

# --- INIT MODELS ---
print("Loading AI Models...")
try:
    nltk.download('stopwords', quiet=True)
    try: nlp = spacy.load("en_core_web_md")
    except: 
        os.system("python -m spacy download en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    
    with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)
    
    model = XGBClassifier()
    model.load_model("xgb_model.json")
except Exception as e:
    print(f"Model Loading Failed: {e}")
    exit()

geolocator = Nominatim(user_agent="crisis_monitor_v13_robust")
stop = set(stopwords.words('english')) - {"not", "no"}
SUBREDDITS = ['mentalhealth', 'depression', 'SuicideWatch', 'anxiety', 'stress',
    'offmychest', 'lonely', 'BPD', 'ptsd', 'socialanxiety', 'bipolar',
    'addiction', 'traumatoolbox', 'CPTSD', 'selfharm', 'OCD',
    'EatingDisorders', 'MentalHealthSupport', 'schizophrenia',
    'insomnia', 'panicattacks', 'ADHD']

def get_reddit():
    if not REDDIT_ID: return []
    reddit = praw.Reddit(client_id=REDDIT_ID, client_secret=REDDIT_SECRET, user_agent='crisis_v13')
    posts = []
    for sub in SUBREDDITS:
        try:
            for post in reddit.subreddit(sub).new(limit=110):
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

def get_db_engine():
    """Creates a robust engine that checks connection health."""
    return create_engine(DB_STRING, pool_pre_ping=True)

def filter_existing_posts(df):
    if df.empty: return df
    new_ids = tuple(df['id'].tolist())
    if not new_ids: return df
    
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            query = text("SELECT id FROM crisis_events_v4 WHERE id IN :ids")
            existing = pd.read_sql(query, conn, params={"ids": new_ids})
            existing_ids = set(existing['id'].tolist())
        return df[~df['id'].isin(existing_ids)]
    except Exception as e:
        # If table doesn't exist, this is fine
        print(f"   - Deduplication info: {e}")
        return df

def calculate_strict_risk(row, features):
    # 1. Model Assessment
    try:
        probs = model.predict_proba(features)[0]
        pred_class = np.argmax(probs) 
        prob_crit = probs[2]
    except:
        pred_class = 0
        prob_crit = 0.0

    # 2. Sentiment Assessment
    sentiment = row['textblob_score']
    if sentiment < -0.3: sent_class = 2     
    elif sentiment < -0.05: sent_class = 1  
    else: sent_class = 0

    # 3. Max Strategy
    final_class = max(pred_class, sent_class)

    # 4. Strict Mapping
    mapping = {2: "Critical", 1: "Moderate", 0: "Low"}
    status = mapping[final_class]

    # 5. Explainability
    reasons = []
    if pred_class == 2: reasons.append(f"Model Pattern ({int(prob_crit*100)}%)")
    if sent_class == 2: reasons.append(f"Sentiment ({sentiment:.2f})")
    
    if not reasons and final_class > 0: reasons.append("Potential Risk")
    
    risk_factors = " + ".join(reasons) if reasons else "Low Risk"

    return status, risk_factors

def cleanup_old_data():
    engine = get_db_engine()
    cutoff = datetime.datetime.now() - datetime.timedelta(days=DATA_RETENTION_DAYS)
    try:
        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM crisis_events_v4 WHERE created_utc < '{cutoff}'"))
            conn.commit()
    except: pass

def run_pipeline():
    print("1. Extracting...")
    df = get_reddit()
    if df.empty: return

    # --- FIX 1: Deduplicate Cross-Posts in the current batch ---
    # If the same text appears multiple times (different subreddits), keep only the first one
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"   - Removed {initial_count - len(df)} cross-posts from batch.")

    # --- FIX 2: Filter IDs that already exist in DB ---
    df = filter_existing_posts(df)

    if df.empty:
        print("   - No new posts.")
        return

    print("2. Strict 3-Level Analysis...")
    df['clean_text'] = df['text'].apply(clean_for_model)
    tf_idf_matrix = vectorizer.transform(df['clean_text'])
    df['textblob_score'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    statuses = []
    factors = []
    for i in range(len(df)):
        status, factor = calculate_strict_risk(df.iloc[i], tf_idf_matrix[i])
        statuses.append(status)
        factors.append(factor)

    df['status'] = statuses
    df['risk_factors'] = factors
    df['sentiment'] = df['textblob_score']

    print(f"3. Geolocating {len(df)} posts...")
    geo_results = df['text'].apply(smart_geocode)
    df['location_name'] = [res[0] if res else None for res in geo_results]
    df['lat'] = [res[1] if res else None for res in geo_results]
    df['lon'] = [res[2] if res else None for res in geo_results]
    
    print(f"4. Saving {len(df)} rows...")
    if not df.empty:
        cols = ['id', 'created_utc', 'subreddit', 'text', 'status', 'sentiment', 'risk_factors', 'url', 'location_name', 'lat', 'lon']
        engine = get_db_engine()
        try:
            # FIX: Use chunksize to prevent timeouts
            df[cols].to_sql('crisis_events_v4', engine, if_exists='append', index=False, chunksize=500)
            print("   - Success.")
        except Exception as e:
            print(f"   - DB Error: {e}")
    
    cleanup_old_data()

if __name__ == "__main__":
    run_pipeline()