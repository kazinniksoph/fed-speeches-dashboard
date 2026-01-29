"""
BERTopic Analysis for Federal Reserve Speeches
Uses semantic embeddings for more coherent topic modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# BERTopic and dependencies
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

# === Configuration ===
MIN_TOPIC_SIZE = 30  # Minimum documents per topic (lowered for more granular topics)
N_TOP_WORDS = 20
NR_TOPICS = 12  # Target number of topics for reduction

# === Speaker Normalization ===
SPEAKER_NORMALIZATION = {
    'Barkin': 'Thomas Barkin', 'Tom Barkin': 'Thomas Barkin',
    'Bernanke': 'Ben Bernanke', 'Powell': 'Jerome Powell',
    'Yellen': 'Janet Yellen', 'Williams': 'John Williams',
    'Evans': 'Charles Evans', 'Bullard': 'James Bullard',
    'Rosengren': 'Eric Rosengren', 'Dudley': 'William Dudley',
    'Mester': 'Loretta Mester', 'Lacker': 'Jeffrey Lacker',
    'Harker': 'Patrick Harker', 'Lockhart': 'Dennis Lockhart',
    'Kashkari': 'Neel Kashkari', 'Kaplan': 'Robert Kaplan',
    'Poole': 'William Poole', 'Kocherlakota': 'Narayana Kocherlakota',
    'Plosser': 'Charles Plosser', 'Ferguson': 'Roger Ferguson',
    'Brainard': 'Lael Brainard', 'George': 'Esther George',
    'Bowman': 'Michelle Bowman', 'Bostic': 'Raphael Bostic',
    'Hoenig': 'Thomas Hoenig', 'Daly': 'Mary Daly',
    'Pianalto': 'Sandra Pianalto', 'Quarles': 'Randal Quarles',
    'Waller': 'Christopher Waller', 'Moskow': 'Michael Moskow',
    'Clarida': 'Richard Clarida', 'Fischer': 'Stanley Fischer',
    'Duke': 'Elizabeth Duke', 'Kroszner': 'Randy Kroszner',
    'Tarullo': 'Daniel Tarullo', 'Parry': 'Robert Parry',
    'Greenspan': 'Alan Greenspan', 'Stein': 'Jeremy Stein',
    'Fisher': 'Richard Fisher', 'Raskin': 'Sarah Bloom Raskin',
    'Sara Raskin': 'Sarah Bloom Raskin', 'Bies': 'Susan Bies',
    'Mishkin': 'Frederic Mishkin', 'Warsh': 'Kevin Warsh',
    'Gramlich': 'Edward Gramlich', 'Kohn': 'Donald Kohn',
    'Santomero': 'Anthony Santomero', 'Broaddus': 'Alfred Broaddus',
    'Musalem': 'Alberto Musalem',
}

def normalize_speaker(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    return SPEAKER_NORMALIZATION.get(name, name)

# === Load Data ===
print("Loading speech data...")
data_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/year_all'
files = glob.glob(f'{data_dir}/*.csv')
dfs = []
for f in sorted(files):
    temp_df = pd.read_csv(f)
    if len(temp_df) > 0:
        dfs.append(temp_df)
df = pd.concat(dfs, ignore_index=True)
df['speaker'] = df['speaker'].apply(normalize_speaker)

def parse_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        return datetime.strptime(str(date_str), '%d%b%Y')
    except:
        return None

df['parsed_date'] = df['date'].apply(parse_date)
df['year'] = df['parsed_date'].dt.year
df = df.dropna(subset=['text'])
print(f"Loaded {len(df)} speeches")

# === Stopwords for representation (not embeddings) ===
stop_words = list(stopwords.words('english'))

# Add Fed-specific stopwords
fed_stopwords = [
    'federal', 'reserve', 'board', 'governors', 'fomc', 'committee',
    'chairman', 'president', 'vice', 'governor', 'percent', 'year', 'years',
    'would', 'could', 'also', 'well', 'one', 'two', 'three', 'may', 'much',
    'many', 'new', 'like', 'think', 'see', 'time', 'make', 'way', 'good',
    'get', 'take', 'know', 'come', 'go', 'say', 'said', 'will', 'thank',
    'today', 'remarks', 'speech', 'mr', 'ms', 'dr', 'figure', 'chart'
]
stop_words.extend(fed_stopwords)

# === Setup BERTopic Components ===
print("\nSetting up BERTopic...")

# Use a finance/economics-tuned model if available, otherwise general purpose
print("Loading sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# UMAP for dimensionality reduction
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# HDBSCAN for clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=MIN_TOPIC_SIZE,
    min_samples=5,  # Lower for more clusters
    metric='euclidean',
    cluster_selection_method='leaf',  # 'leaf' gives more granular clusters
    prediction_data=True
)

# CountVectorizer for topic representation
vectorizer_model = CountVectorizer(
    stop_words=stop_words,
    min_df=2,  # Lower threshold for small topics
    ngram_range=(1, 2)
)

# === Create BERTopic Model ===
print("Creating BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=N_TOP_WORDS,
    verbose=True,
    calculate_probabilities=True
)

# === Fit Model ===
print("\nFitting BERTopic (this may take a few minutes)...")
docs = df['text'].tolist()
topics, probs = topic_model.fit_transform(docs)

# Reduce topics if we got too many
initial_topics = len(set(topics)) - (1 if -1 in topics else 0)
print(f"\nInitial topics found: {initial_topics}")

if initial_topics > NR_TOPICS:
    print(f"Reducing to {NR_TOPICS} topics...")
    topic_model.reduce_topics(docs, nr_topics=NR_TOPICS)
    topics = topic_model.topics_

# === Get Results ===
topic_info = topic_model.get_topic_info()
print(f"\nFinal topics: {len(topic_info) - 1} (excluding outliers)")

# === Display Topics ===
print(f"\n{'='*60}")
print("BERTOPIC RESULTS")
print(f"{'='*60}\n")

# Skip topic -1 (outliers)
for _, row in topic_info[topic_info['Topic'] != -1].head(20).iterrows():
    topic_id = row['Topic']
    count = row['Count']
    words = topic_model.get_topic(topic_id)
    word_str = ', '.join([w[0] for w in words[:10]])
    print(f"Topic {topic_id} ({count} docs): {word_str}")

# === Add to DataFrame ===
df['bertopic_topic'] = topics
df['bertopic_prob'] = [p.max() if len(p) > 0 else 0 for p in probs]

# Add probability for each topic
n_topics = len(topic_info) - 1  # Exclude outliers
for i in range(min(n_topics, 30)):  # Cap at 30 topics
    df[f'bertopic_{i}_prob'] = [p[i] if i < len(p) else 0 for p in probs]

# === Save Results ===
output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/bertopic_results'
os.makedirs(output_dir, exist_ok=True)

# Save topic info
topic_info.to_csv(f'{output_dir}/topic_info.csv', index=False)

# Save detailed topic definitions
topic_defs = []
for _, row in topic_info[topic_info['Topic'] != -1].iterrows():
    topic_id = row['Topic']
    words = topic_model.get_topic(topic_id)
    topic_defs.append({
        'topic_id': topic_id,
        'n_speeches': row['Count'],
        'top_words': ', '.join([w[0] for w in words[:10]]),
        'all_top_words': ', '.join([w[0] for w in words])
    })
topic_defs_df = pd.DataFrame(topic_defs)
topic_defs_df.to_csv(f'{output_dir}/topic_definitions.csv', index=False)

# Save speech assignments
speech_cols = ['id', 'speaker', 'date', 'year', 'source', 'bertopic_topic', 'bertopic_prob']
speech_topics_df = df[speech_cols].copy()
speech_topics_df.to_csv(f'{output_dir}/speech_topic_assignments.csv', index=False)

# Save the model
topic_model.save(f'{output_dir}/bertopic_model')

# === Generate Visualizations ===
print("\nGenerating visualizations...")

try:
    # Topic visualization
    fig = topic_model.visualize_topics()
    fig.write_html(f'{output_dir}/topic_distance_map.html')

    # Topic hierarchy
    fig = topic_model.visualize_hierarchy()
    fig.write_html(f'{output_dir}/topic_hierarchy.html')

    # Topics over time
    timestamps = df['parsed_date'].tolist()
    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html(f'{output_dir}/topics_over_time.html')

    # Barchart of top topics
    fig = topic_model.visualize_barchart(top_n_topics=15)
    fig.write_html(f'{output_dir}/topic_barchart.html')

    print("Visualizations saved!")
except Exception as e:
    print(f"Warning: Some visualizations failed: {e}")

# === Summary ===
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Total speeches: {len(df)}")
print(f"Topics found: {len(topic_info) - 1}")
print(f"Outlier speeches (topic -1): {(df['bertopic_topic'] == -1).sum()}")
print(f"\nTop 5 topics by document count:")
for _, row in topic_info[topic_info['Topic'] != -1].head(5).iterrows():
    words = topic_model.get_topic(row['Topic'])
    print(f"  Topic {row['Topic']}: {row['Count']} speeches - {', '.join([w[0] for w in words[:5]])}")

print(f"\n{'='*60}")
print("FILES SAVED")
print(f"{'='*60}")
print(f"Results saved to: {output_dir}/")
print(f"  - topic_info.csv")
print(f"  - topic_definitions.csv")
print(f"  - speech_topic_assignments.csv")
print(f"  - bertopic_model/")
print(f"  - topic_distance_map.html")
print(f"  - topic_hierarchy.html")
print(f"  - topics_over_time.html")
print(f"  - topic_barchart.html")
