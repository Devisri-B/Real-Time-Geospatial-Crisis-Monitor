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

geolocator = Nominatim(user_agent="crisis_monitor_v10_strict")
stop = set(stopwords.words('english')) - {"not", "no"}
SUBREDDITS = ['mentalhealth', 'depression', 'SuicideWatch', 'anxiety',
    'stress', 'offmychest', 'lonely', 'BPD', 'ptsd',
    'socialanxiety', 'bipolar', 'addiction', 'traumatoolbox', 'CPTSD',
    'selfharm', 'OCD', 'EatingDisorders', 'MentalHealthSupport',
    'schizophrenia', 'insomnia', 'panicattacks', 'ADHD', 'AskReddit']

def get_reddit():
    if not REDDIT_ID: return []
    reddit = praw.Reddit(client_id=REDDIT_ID, client_secret=REDDIT_SECRET, user_agent='crisis_v10')
    posts = []
    for sub in SUBREDDITS:
        try:
            for post in reddit.subreddit(sub).new(limit=10000):
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

def filter_existing_posts(df, engine):
    if df.empty: return df
    new_ids = tuple(df['id'].tolist())
    if not new_ids: return df
    try:
        with engine.connect() as conn:
            query = text("SELECT id FROM crisis_events_v4 WHERE id IN :ids")
            existing = pd.read_sql(query, conn, params={"ids": new_ids})
            existing_ids = set(existing['id'].tolist())
        return df[~df['id'].isin(existing_ids)]
    except: return df

def calculate_strict_risk(row, features):
    """
    STRICT 3-LEVEL LOGIC: Critical, Moderate, Low.
    """
    # 1. Model Assessment (Probabilities)
    try:
        probs = model.predict_proba(features)[0]
        # Get the single class with highest probability
        pred_class = np.argmax(probs) # 0=Low, 1=Mod, 2=Crit
    except:
        pred_class = 0

    # 2. Sentiment Assessment
    sentiment = row['textblob_score']
    if sentiment < -0.3: sent_class = 2     # Critical
    elif sentiment < -0.05: sent_class = 1  # Moderate
    else: sent_class = 0

    # 3. Max Strategy (Take the worst case)
    final_class = max(pred_class, sent_class)

    # 4. Strict Mapping
    mapping = {2: "Critical", 1: "Moderate", 0: "Low"}
    status = mapping[final_class]

    # 5. Explanation
    reasons = []
    if pred_class == 2: reasons.append("AI Model: Critical Pattern")
    if sent_class == 2: reasons.append(f"Sentiment: Critical ({sentiment:.2f})")

    if not reasons and final_class > 0: reasons.append("Monitor")
    
    risk_factors = " + ".join(reasons) if reasons else "Safe"

    return status, risk_factors

def cleanup_old_data():
    engine = create_engine(DB_STRING)
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

    engine = create_engine(DB_STRING)
    df = filter_existing_posts(df, engine)

    if df.empty:
        print("   - No new posts.")
        return

    print("2. Strict Risk Analysis...")
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

    active_df = df[df['status'] != "Low"].copy()
    
    print(f"3. Geolocating {len(active_df)} alerts...")
    geo_results = active_df['text'].apply(smart_geocode)
    active_df['location_name'] = [res[0] for res in geo_results]
    active_df['lat'] = [res[1] for res in geo_results]
    active_df['lon'] = [res[2] for res in geo_results]
    
    final_df = active_df.dropna(subset=['lat'])
    
    print(f"4. Saving {len(final_df)} rows...")
    if not final_df.empty:
        cols = ['id', 'created_utc', 'subreddit', 'text', 'status', 'sentiment', 'risk_factors', 'url', 'location_name', 'lat', 'lon']
        try:
            final_df[cols].to_sql('crisis_events_v4', engine, if_exists='append', index=False)
            print("   - Success.")
        except Exception as e:
            print(f"   - DB Error: {e}")
    
    cleanup_old_data()

if __name__ == "__main__":
    run_pipeline()