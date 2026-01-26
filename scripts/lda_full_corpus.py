"""
LDA Topic Modeling Analysis for Federal Reserve Speeches - Full Corpus
Runs LDA on all 7,501 speeches with normalized speaker names.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
import glob

# NLP libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# For text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# === Configuration ===
N_TOPICS = 15  # Match existing analysis
N_TOP_WORDS = 20
MAX_FEATURES = 5000
MIN_DF = 10
MAX_DF = 0.7

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
print("Loading speech data from full corpus...")
data_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/non time stamped speeches'
files = glob.glob(f'{data_dir}/*.csv')
dfs = []
for f in sorted(files):
    temp_df = pd.read_csv(f)
    if len(temp_df) > 0:
        dfs.append(temp_df)
df = pd.concat(dfs, ignore_index=True)

# Normalize speaker names
df['speaker'] = df['speaker'].apply(normalize_speaker)

# Parse dates
def parse_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        return datetime.strptime(str(date_str), '%d%b%Y')
    except:
        return None

df['parsed_date'] = df['date'].apply(parse_date)
df['year'] = df['parsed_date'].dt.year
df['word_count'] = df['text'].str.split().str.len()

# Remove rows with missing text
df = df.dropna(subset=['text'])
print(f"Loaded {len(df)} speeches")
print(f"Unique speakers: {df['speaker'].nunique()}")
print(f"Year range: {int(df['year'].min())} - {int(df['year'].max())}")

# === Text Preprocessing ===
print("\nPreprocessing text...")

# Custom stopwords for Fed speeches
fed_stopwords = {
    'federal', 'reserve', 'board', 'governors', 'fomc', 'committee',
    'percent', 'year', 'years', 'would', 'could', 'also', 'well',
    'one', 'two', 'three', 'may', 'much', 'many', 'new', 'like',
    'think', 'see', 'time', 'make', 'way', 'good', 'get', 'take',
    'know', 'come', 'go', 'say', 'said', 'will', 'thank', 'today',
    'remarks', 'speech', 'chairman', 'president', 'vice', 'governor',
    'mr', 'ms', 'dr', 'figure', 'chart', 'slide', 'footnote', 'note'
}

stop_words = set(stopwords.words('english')).union(fed_stopwords)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 3
    ]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)
print("Text preprocessing complete")

# === Create Document-Term Matrix ===
print("\nCreating document-term matrix...")

vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    ngram_range=(1, 2)
)

doc_term_matrix = vectorizer.fit_transform(df['processed_text'])
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names)}")
print(f"Document-term matrix shape: {doc_term_matrix.shape}")

# === Train LDA Model ===
print(f"\nTraining LDA model with {N_TOPICS} topics...")
print("This may take a few minutes...")

lda_model = LatentDirichletAllocation(
    n_components=N_TOPICS,
    learning_method='online',
    random_state=42,
    max_iter=25,
    n_jobs=-1,
    verbose=1
)

doc_topic_matrix = lda_model.fit_transform(doc_term_matrix)
print("LDA training complete")

# === Extract Results ===
def get_top_words(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics[topic_idx] = top_words
    return topics

topics = get_top_words(lda_model, feature_names, N_TOP_WORDS)

# Display topics
print(f"\n{'='*60}")
print(f"LDA RESULTS - {N_TOPICS} TOPICS")
print(f"{'='*60}\n")

for topic_idx, words in topics.items():
    print(f"Topic {topic_idx}: {', '.join(words[:10])}")

# Add topic assignments to dataframe
df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)
df['dominant_topic_prob'] = doc_topic_matrix.max(axis=1)

# Add all topic probabilities
for i in range(N_TOPICS):
    df[f'topic_{i}_prob'] = doc_topic_matrix[:, i]

# Calculate MP score (monetary policy topics - adjust based on actual topic content)
# This will need to be updated based on which topics are about monetary policy
# For now, we'll compute it after reviewing the topics

# === Save Results ===
output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/lda_full_results'
os.makedirs(output_dir, exist_ok=True)

# 1. Save topic definitions
topic_defs = []
for topic_idx, words in topics.items():
    topic_defs.append({
        'topic_id': topic_idx,
        'n_dominant_speeches': (df['dominant_topic'] == topic_idx).sum(),
        'avg_probability': doc_topic_matrix[:, topic_idx].mean(),
        'top_words': ', '.join(words[:10]),
        'all_top_words': ', '.join(words)
    })
topic_defs_df = pd.DataFrame(topic_defs)
topic_defs_df.to_csv(f'{output_dir}/topic_definitions_k{N_TOPICS}.csv', index=False)

# 2. Save speech topic assignments
speech_cols = ['id', 'speaker', 'date', 'year', 'word_count', 'source',
               'dominant_topic', 'dominant_topic_prob']
topic_prob_cols = [f'topic_{i}_prob' for i in range(N_TOPICS)]
speech_topics_df = df[speech_cols + topic_prob_cols].copy()
speech_topics_df.to_csv(f'{output_dir}/speech_topic_assignments.csv', index=False)

# 3. Save yearly aggregation
yearly_agg = df.groupby('year').agg({
    'id': 'count',
    'dominant_topic': lambda x: (x == x.mode().iloc[0] if len(x.mode()) > 0 else -1).sum(),
}).rename(columns={'id': 'total_speeches'})

for i in range(N_TOPICS):
    yearly_agg[f'topic_{i}_avg'] = df.groupby('year')[f'topic_{i}_prob'].mean()
    yearly_agg[f'topic_{i}_dominant_count'] = df.groupby('year').apply(
        lambda x: (x['dominant_topic'] == i).sum()
    )

yearly_agg.to_csv(f'{output_dir}/yearly_aggregation.csv')

# 4. Save speaker aggregation
speaker_agg = df.groupby('speaker').agg({
    'id': 'count',
    'dominant_topic': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1,
    'dominant_topic_prob': 'mean'
}).rename(columns={'id': 'total_speeches', 'dominant_topic': 'most_common_topic'})

for i in range(N_TOPICS):
    speaker_agg[f'topic_{i}_avg'] = df.groupby('speaker')[f'topic_{i}_prob'].mean()

speaker_agg = speaker_agg.sort_values('total_speeches', ascending=False)
speaker_agg.to_csv(f'{output_dir}/speaker_aggregation.csv')

print(f"\n{'='*60}")
print("FILES SAVED")
print(f"{'='*60}")
print(f"\nResults saved to: {output_dir}/")
print(f"  - topic_definitions_k{N_TOPICS}.csv")
print(f"  - speech_topic_assignments.csv")
print(f"  - yearly_aggregation.csv")
print(f"  - speaker_aggregation.csv")

# Summary statistics
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Total speeches: {len(df)}")
print(f"Topics: {N_TOPICS}")
print(f"Vocabulary size: {len(feature_names)}")
print(f"\nTop 5 topics by speech count:")
topic_counts = df['dominant_topic'].value_counts().head(5)
for topic_idx, count in topic_counts.items():
    print(f"  Topic {topic_idx}: {count} speeches ({100*count/len(df):.1f}%) - {', '.join(topics[topic_idx][:5])}")
