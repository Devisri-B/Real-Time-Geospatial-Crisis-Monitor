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
from nltk.corpus import stopwords
import nltk
import contractions

# CONFIG
REDDIT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
DB_STRING = os.getenv('DB_CONNECTION_STRING')

print("Initializing System...")
try:
    nltk.download('stopwords', quiet=True)
    try: nlp = spacy.load("en_core_web_sm")
    except: 
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Load the Brains
    with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f: model = pickle.load(f)
except Exception as e:
    print(f"Init Error: {e}")
    exit()

geolocator = Nominatim(user_agent="crisis_v4_hybrid")
stop = set(stopwords.words('english')) - {"not", "no"}
SUBREDDITS = ['mentalhealth', 'depression', 'SuicideWatch', 'anxiety',
    'stress', 'offmychest', 'lonely', 'BPD', 'ptsd',
    'socialanxiety', 'bipolar', 'addiction', 'traumatoolbox', 'CPTSD',
    'selfharm', 'OCD', 'EatingDisorders', 'MentalHealthSupport',
    'schizophrenia', 'insomnia', 'panicattacks', 'ADHD', 'AskReddit',
    'worldnews', 'news']

def get_reddit():
    if not REDDIT_ID: return []
    reddit = praw.Reddit(client_id=REDDIT_ID, client_secret=REDDIT_SECRET, user_agent='crisis_v4')
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

def clean(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop])

def get_geo(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            try:
                time.sleep(0.5)
                loc = geolocator.geocode(ent.text)
                if loc: return ent.text, loc.latitude, loc.longitude
            except: continue
    return None, None, None

def run_pipeline():
    print("1. Extracting Live Data...")
    df = get_reddit()
    if df.empty: return

    print("2. Two-Stage Analysis...")
    df['clean'] = df['text'].apply(clean)
    
    # STAGE 1: XGBoost Prediction
    features = vectorizer.transform(df['clean'])
    df['xgb_pred'] = model.predict(features) # 0=Low, 1=Mod, 2=High
    
    # STAGE 2: TextBlob Verification
    df['sentiment'] = df['clean'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # LOGIC: Combine them for Final Status
    def verify_risk(row):
        # If Model says High Risk (2)
        if row['xgb_pred'] == 2:
            if row['sentiment'] < -0.1: return "Critical" # Confirmed
            else: return "High (Unverified)" # Model says yes, Sentiment says maybe not
        
        # If Model says Moderate (1)
        elif row['xgb_pred'] == 1:
            if row['sentiment'] < -0.3: return "High (Escalated)" # Sentiment is super bad
            return "Moderate"
            
        return "Low"

    df['status'] = df.apply(verify_risk, axis=1)
    
    # Only keep risks
    risky_df = df[df['status'] != "Low"].copy()
    
    print(f"3. Geocoding {len(risky_df)} alerts...")
    locs = risky_df['text'].apply(get_geo)
    risky_df['location_name'] = [x[0] for x in locs]
    risky_df['lat'] = [x[1] for x in locs]
    risky_df['lon'] = [x[2] for x in locs]
    
    final = risky_df.dropna(subset=['lat'])
    
    print(f"4. Saving {len(final)} verified events...")
    if not final.empty:
        engine = create_engine(DB_STRING)
        cols = ['id', 'created_utc', 'subreddit', 'text', 'status', 'sentiment', 'url', 'location_name', 'lat', 'lon']
        final[cols].to_sql('crisis_events_v4', engine, if_exists='append', index=False)

if __name__ == "__main__":
    run_pipeline()