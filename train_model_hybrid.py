import praw
import pandas as pd
import re
import pickle
import nltk
import os
import time
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import contractions
from nltk.corpus import stopwords
import numpy as np

# --- 1. CONFIGURATION ---
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USER_AGENT = 'crisis_trainer_v2'

# Subreddits for fresh data
SUBREDDITS = [
    'mentalhealth', 'depression', 'SuicideWatch', 'anxiety', 'stress',
    'offmychest', 'lonely', 'BPD', 'ptsd', 'socialanxiety', 'bipolar',
    'addiction', 'traumatoolbox', 'CPTSD', 'selfharm', 'OCD',
    'EatingDisorders', 'MentalHealthSupport', 'schizophrenia',
    'insomnia', 'panicattacks', 'ADHD', 'AskReddit'
]

# Ensure no accidental duplicates (preserve order)
SUBREDDITS = list(dict.fromkeys(SUBREDDITS))


# --- 2. DATA LOADING FUNCTIONS ---

def load_kaggle_data(filename='Combined Data.csv'):
    """Loads the static CSV dataset to boost training volume."""
    print(f"Loading {filename}...")
    if not os.path.exists(filename):
        print(f"   File {filename} not found. Skipping CSV data.")
        return pd.DataFrame()
    
    try:
        # Load CSV. Your snippet shows columns: [Index, statement, status]
        df = pd.read_csv(filename)
        
        # Check if 'statement' column exists (based on your snippet)
        if 'statement' in df.columns:
            # Rename to 'content' to match our pipeline standard
            df = df.rename(columns={'statement': 'content'})
            print(f"   - Successfully loaded {len(df)} rows from CSV.")
            return df[['content']]
        else:
            print("   Column 'statement' not found. Checking columns:", df.columns)
            return pd.DataFrame()
            
    except Exception as e:
        print(f"   Error reading CSV: {e}")
        return pd.DataFrame()

def fetch_reddit_data(target_count=10000000):
    print(f" Connecting to Reddit to fetch posts...")
    
    if not REDDIT_CLIENT_ID:
        raise ValueError(" Error: REDDIT_CLIENT_ID not found in environment variables.")

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=USER_AGENT
    )
    
    all_posts = []
    posts_per_sub = target_count // len(SUBREDDITS)
    
    for sub in SUBREDDITS:
        print(f"   Scanning r/{sub}...")
        try:
            subreddit = reddit.subreddit(sub)
            
            # We combine multiple sorting methods to get more data
            # 1. Get New posts
            for post in subreddit.new(limit=400000):
                all_posts.append(post.title + " " + post.selftext)
            
            # 2. Get Hot posts
            for post in subreddit.hot(limit=200000):
                all_posts.append(post.title + " " + post.selftext)

            # 3. Get Top posts of the Month
            for post in subreddit.top(time_filter="month", limit=300000):
                all_posts.append(post.title + " " + post.selftext)
                
        except Exception as e:
            print(f" Issue with r/{sub}: {e}")
            
    # Deduplicate (Remove identical posts fetched from different lists)
    df = pd.DataFrame(all_posts, columns=['content']).drop_duplicates()
    print(f" Mining Complete. Total unique posts collected: {len(df)}")
    return df

# --- 3. PREPROCESSING ---
print("Initializing...")
nltk.download('stopwords', quiet=True)
stop = set(stopwords.words('english')) - {"not", "no", "nor", "n't"}

def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text) 
    return " ".join([w for w in text.split() if w not in stop and len(w) > 2])

# --- 4. EXECUTION FLOW ---

# A. Merge Data Sources
df_kaggle = load_kaggle_data()
df_api = fetch_reddit_data()

# Combine and remove duplicates
df = pd.concat([df_kaggle, df_api], ignore_index=True)
df = df.drop_duplicates(subset=['content'])
df = df.dropna(subset=['content'])

print(f"Total Training Data: {len(df)} unique items.")

if len(df) < 100:
    print("Not enough data to train. Add 'Combined Data.csv' or check API keys.")
    exit()

# B. Clean
print("Cleaning text data...")
df['clean_text'] = df['content'].apply(clean_text)
df = df[df['clean_text'].str.len() > 10] # Drop empty rows

# C. Unsupervised Clustering (The "Teacher")
print("Running KMeans to Auto-Label Data...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df['clean_text'])

# Cluster into 3 groups (Safe, Moderate, Critical)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_tfidf)

# Determine Labels based on Sentiment
print("   - Mapping clusters to Risk Levels...")
cluster_sentiment = df.groupby('cluster')['clean_text'].apply(
    lambda x: x.apply(lambda t: TextBlob(t).sentiment.polarity).mean()
)

# Sort: Lowest Sentiment = Class 2 (Critical), Middle = 1, Highest = 0
sorted_clusters = cluster_sentiment.sort_values().index 
risk_map = {
    sorted_clusters[0]: 2, # Critical (Most Negative)
    sorted_clusters[1]: 1, # Moderate
    sorted_clusters[2]: 0  # Low (Least Negative)
}
df['label'] = df['cluster'].map(risk_map)

print(f"   - Class Balance: {df['label'].value_counts().to_dict()}")
print(f"     (0=Low, 1=Moderate, 2=Critical)")

# D. Train XGBoost
print("Training XGBoost Model...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='mlogloss'
)
model.fit(X_tfidf, df['label'])

# E. Save Artifacts
print("Saving Artifacts...")
with open('vectorizer.pkl', 'wb') as f: 
    pickle.dump(tfidf, f)

# FIX: Save XGBoost as JSON to remove warnings and improve compatibility
model.save_model("xgb_model.json") 

print("Training Complete! Upload 'vectorizer.pkl' and 'xgb_model.json' to GitHub.")