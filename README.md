# 🛡️ Real-Time Geospatial Crisis Monitor (CrisisGuard)
A real-time intelligence platform that detects, geolocates, and classifies mental health crises using social media signals and AI.
- Live Demo: [Launch App](https://real-time-geospatial-crisis-monitor.streamlit.app)
  
![Overview](assets/Overview.png)

## Project Overview
Mental health crises often manifest on social media long before traditional emergency services are contacted. CrisisGuard is an end-to-end data pipeline and dashboard designed to monitor real-time discussions on Reddit, detect high-risk signals using Natural Language Processing (NLP), and visualize them on a geospatial interface to aid rapid response.

Unlike standard sentiment analysis, this project utilizes a **Hybrid AI approach**, combining unsupervised clustering with supervised XGBoost classification. Data is persistently stored in **Supabase (PostgreSQL)** and updated automatically via **GitHub Actions**.

## System Architecture
The system follows a strict ETL (Extract, Transform, Load) pipeline architecture, automated to run every 6 hours.
![Architecture](assets/Architecture.png)

## Key Features
1. <u>*NLP-Based Geospatial Intelligence</u>* : Reddit posts doesn't contain precise GPS coordinates so, I engineered a custom geolocation extraction pipeline to solve this:
	- **Entity Extraction**: Uses spaCy NLP to identify Geopolitical Entities (GPE) mentioned in user text (e.g., "I'm stuck in London").
	- **Geocoding**: Converts extracted names into coordinates using **Geopy**.
	- **Unmapped Fallback**: Posts with no extractable location are automatically routed to a separate "Unmapped Intel" analysis queue.
	- **Interactive Map**: Features popups with risk levels, sample post, timestamps, and direct source links.

2. <u>*Hybrid AI Risk Engine*</u> :The core of the system is a custom NLP model that classifies text into three risk levels.
	- **Inputs**: TF-IDF vectors + **TextBlob** Sentiment Polarity.
	- **Model**: XGBoost Classifier.
	- **Logic**: A "Max Strategy" that weighs model probability against the raw sentiment score to minimize false negatives in critical scenarios.
3. *Command Center & Filtering* : A centralized control panel allowing analysts to slice and dice data dynamically to find actionable insights:
	- **Timeline Slider**: Filter incidents by specific date and time ranges to analyze trends.
	- **Severity Filters**: Toggle between Critical, Moderate, and Low risk levels to focus purely on immediate threats.
	- **Regional Focus**: Isolate data for specific countries or select "All Global Regions" to generate targeted local reports.
4. <u>*Alert Dispatch System*</u>:  An integrated alerting system allowing administrators to send formatted intelligence reports via SMTP email for "Critical" level events identified in specific regions.
5. <u>*Automated Data Pipeline*</u> : A robust ETL process that ensures real-time relevance:
	- **Orchestration**: A **GitHub Action** workflow triggers the pipeline every 6 hours.
	- **Extraction**: Fetches fresh posts from 20+ subreddits using the Reddit API.
	- **Storage**: Processed data is upserted into Supabase (PostgreSQL) to prevent duplicates.

## Deep Dive: The AI Training Methodology
One of the biggest challenges in mental health AI is the lack of labeled "Critical" vs "Moderate" data. I solved this using a Semi-Supervised workflow combining static datasets with live streaming data.

1. <u>*Data Aggregation*</u>: I constructed a massive training corpus of 75,838 unique items by fusing two sources:
	- Static Data: 53,043 labeled rows from the [Kaggle Mental Health Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health).
	- Live Data: 24,768 fresh posts mined from 20+ live subreddits (r/SuicideWatch, r/depression, r/ptsd) via the PRAW API.

2. <u>*Semi-Supervised Auto-Labeling*</u>: To handle the unlabeled live data, I utilized an unsupervised "Teacher" model to create labels for the "Student" XGBoost model:

	- Clustering: Used **K-Means** to group text into 3 distinct semantic clusters.
	- Sentiment Mapping: Mapped clusters to risk levels based on **TextBlob polarity** averages.
		- Cluster 0 (Low Risk): 39,840 samples
		- Cluster 1 (Moderate Risk): 26,625 samples
		- Cluster 2 (Critical Risk): 7,252 samples

3. *Supervised Training (XGBoost)* : Finally, I trained an **XGBoost Classifier** on this auto-labeled dataset to predict risk levels on future unseen data with high efficiency.

## Technical Architecture

- **Frontend**: Streamlit, Folium, Plotly Express.
- **Backend/Automation**: Python, GitHub Actions (Cron Jobs).
- **Database**: Supabase (PostgreSQL).
- **Machine Learning**: XGBoost, Scikit-Learn (KMeans, TF-IDF), Spacy (NER), TextBlob.
- **Geospatial**: Geopy (Nominatim), Folium.

## Application Tour & Analytics
###  Live Data Feed
A transparent, raw view of the incoming stream, allowing analysts to verify specific row-level data, risk drivers, and sentiment scores.

 ![LiveMap](assets/Livedata.png)
### Mapped Analysis
For incidents where a location was successfully extracted, the dashboard provides deep regional insights:
- *Hotspot Locations & Sentiment*: Compares incident volume by country/city alongside the average sentiment polarity.
  
  ![Mapped_location](assets/Mapped_Loc.png)
- *Sources & Risk Distribution*: Breaks down which subreddits are driving geospatial risk and visualizes the balance of Critical vs. Moderate threats.
  
  ![Mapped_Risk](assets/Mapped_risk.png)
- *Timeline of Events*: A histogram showing the frequency of mapped crisis signals over time.
  
  ![Mapped_Timeline](assets/Mapped_Timeline.png)

### Unmapped Intelligence

A critical component for analyzing the "Silent Majority" of posts that lack geographic keywords.
- *Sources & Risk*: Identifies which communities are most active, even without location data.
  
  ![Unmapped_Risk](assets/Unmapped_Risk.png)
- *Hidden Timeline*: Tracks the volume of non-geolocated posts to spot global trends or outages.
  
  ![Unmapped_Timeline](assets/Unmapped_Timeline.png)

## How to Run Locally

1. Clone the repository

	```
 	git clone [https://github.com/Devisri-B/Real-Time-Geospatial-Crisis-Monitor.git](https://github.com/Devisri-B/Real-Time-Geospatial-Crisis-Monitor.git)
	 cd Real-Time-Geospatial-Crisis-Monitor
	```

2. Install Dependencies
   
  ```
	pip install -r requirements.txt
	python -m spacy download en_core_web_md
  ```

3. Set up Environment Variables
Create a .env file with your credentials:

```
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
DB_CONNECTION_STRING=postgresql://user:pass@supabase-url:5432/postgres
EMAIL_SENDER=your_email
EMAIL_PASSWORD=your_app_password
```

Train the Model (Optional)

`python train_model_hybrid.py`


Run the Pipeline

`python etl_pipeline.py`


Launch the Dashboard

`streamlit run app.py`

## Contributing

Contributions are welcome! Please open issues or submit pull requests to suggest improvements or new features.






