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

# --- 1. CONFIGURATION ---
# Ensure these are set in your terminal or IDE environment variables!
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USER_AGENT = 'crisis_trainer_v1'

SUBREDDITS = [
    'mentalhealth', 'depression', 'SuicideWatch', 'anxiety',
    'stress', 'offmychest', 'lonely', 'BPD', 'ptsd',
    'socialanxiety', 'bipolar', 'addiction', 'traumatoolbox', 'CPTSD',
    'selfharm', 'OCD', 'EatingDisorders', 'MentalHealthSupport',
    'schizophrenia', 'insomnia', 'panicattacks', 'ADHD', 'AskReddit',
    'worldnews', 'news'
]

# --- 2. BULK DATA MINING ---
def fetch_training_data(target_count=10000000):
    print(f" Connecting to Reddit to fetch ~{target_count} posts...")
    
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
print(" Initializing...")
nltk.download('stopwords', quiet=True)
stop = set(stopwords.words('english')) - {"not", "no", "nor", "n't"}

def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters
    # Simple tokenization and stopword removal
    return " ".join([w for w in text.split() if w not in stop and len(w) > 2])

# --- EXECUTION FLOW ---

# A. GET DATA
df = fetch_training_data(10000000)

if df.empty:
    print(" No data fetched. Check internet or API keys.")
    exit()

# B. CLEAN DATA
print(" Cleaning text data...")
df['clean_text'] = df['content'].apply(clean_text)
# Remove empty rows after cleaning
df = df[df['clean_text'].str.len() > 10] 

# C. UNSUPERVISED LOGIC (The "Teacher")
print(" Running Unsupervised Clustering (KMeans)...")

# Vectorize
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df['clean_text'])

# Cluster into 3 groups (Likely: Safe, Moderate, Crisis)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_tfidf)

# Determine which cluster is "Crisis" using Sentiment
print("   Analyzing Cluster Sentiment to assign labels...")
cluster_sentiment = df.groupby('cluster')['clean_text'].apply(
    lambda x: x.apply(lambda t: TextBlob(t).sentiment.polarity).mean()
)

# Sort clusters by sentiment (Lowest Sentiment = Highest Risk)
sorted_clusters = cluster_sentiment.sort_values().index 
risk_map = {
    sorted_clusters[0]: 2, # Critical (Lowest sentiment)
    sorted_clusters[1]: 1, # Moderate
    sorted_clusters[2]: 0  # Low (Highest sentiment)
}

df['label'] = df['cluster'].map(risk_map)

print(f"   Label Distribution: {df['label'].value_counts().to_dict()}")
print(f"   (0=Low, 1=Moderate, 2=Critical)")

# D. TRAIN XGBOOST (The "Student")
print(" Training XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='mlogloss'
)
model.fit(X_tfidf, df['label'])

# E. SAVE ARTIFACTS
print(" Saving Brains (.pkl files)...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(" DONE! 'vectorizer.pkl' and 'xgb_model.pkl' are ready.")