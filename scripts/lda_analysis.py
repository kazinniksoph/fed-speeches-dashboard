"""
LDA Topic Modeling Analysis for Federal Reserve Speeches
Discovers latent topics in the speech corpus using Latent Dirichlet Allocation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import json
from collections import Counter

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
N_TOPICS = 10  # Number of topics to discover
N_TOP_WORDS = 15  # Number of top words to display per topic
MAX_FEATURES = 5000  # Maximum vocabulary size
MIN_DF = 10  # Minimum document frequency for a term
MAX_DF = 0.7  # Maximum document frequency (as proportion)

# === Load Data ===
print("Loading speech data...")
df = pd.read_csv('/Users/sophiakazinnik/Research/central_bank_speeches_communication/speech_data/all_speeches_merged.csv')

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

# Remove rows with missing text
df = df.dropna(subset=['text'])
print(f"Loaded {len(df)} speeches")

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

# Combine with standard English stopwords
stop_words = set(stopwords.words('english')).union(fed_stopwords)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and preprocess text for LDA."""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and short words, then lemmatize
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 3
    ]

    return ' '.join(tokens)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# === Create Document-Term Matrix ===
print("Creating document-term matrix...")

vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    ngram_range=(1, 2)  # Include bigrams
)

doc_term_matrix = vectorizer.fit_transform(df['processed_text'])
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names)}")
print(f"Document-term matrix shape: {doc_term_matrix.shape}")

# === Train LDA Model ===
print(f"\nTraining LDA model with {N_TOPICS} topics...")

lda_model = LatentDirichletAllocation(
    n_components=N_TOPICS,
    learning_method='online',
    random_state=42,
    max_iter=20,
    n_jobs=1,  # Single thread to avoid multiprocessing issues
    verbose=0
)

doc_topic_matrix = lda_model.fit_transform(doc_term_matrix)

# === Extract Results ===
print("\n" + "="*60)
print("LDA TOPIC MODELING RESULTS")
print("="*60)

# Get top words for each topic
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
print(f"TOP {N_TOP_WORDS} WORDS PER TOPIC")
print(f"{'='*60}\n")

topic_labels = {}
for topic_idx, words in topics.items():
    # Create a suggested label from top 3 words
    label = f"Topic {topic_idx + 1}: {', '.join(words[:3])}"
    topic_labels[topic_idx] = label
    print(f"\n{label}")
    print("-" * 50)
    print(", ".join(words))

# === Topic Distribution Over Time ===
print(f"\n{'='*60}")
print("TOPIC PREVALENCE OVER TIME")
print(f"{'='*60}\n")

# Add dominant topic to each document
df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)
df['dominant_topic_prob'] = doc_topic_matrix.max(axis=1)

# Calculate topic prevalence by year
topic_by_year = df.groupby('year')['dominant_topic'].value_counts(normalize=True).unstack(fill_value=0)

print("Dominant topic distribution by year (showing years with most speeches):")
yearly_counts = df.groupby('year').size()
top_years = yearly_counts.nlargest(10).index.tolist()

for year in sorted(top_years):
    if year in topic_by_year.index:
        row = topic_by_year.loc[year]
        top_topic = row.idxmax()
        print(f"  {int(year)}: Topic {top_topic + 1} ({row[top_topic]:.1%}) - {', '.join(topics[top_topic][:3])}")

# === Topic Distribution by Speaker ===
print(f"\n{'='*60}")
print("TOP SPEAKERS BY TOPIC")
print(f"{'='*60}\n")

for topic_idx in range(N_TOPICS):
    topic_speeches = df[df['dominant_topic'] == topic_idx]
    top_speakers = topic_speeches['speaker'].value_counts().head(3)
    print(f"Topic {topic_idx + 1} ({', '.join(topics[topic_idx][:3])}):")
    for speaker, count in top_speakers.items():
        print(f"  - {speaker}: {count} speeches")
    print()

# === Save Results ===
output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output'
import os
os.makedirs(output_dir, exist_ok=True)

# Save topic words
topic_words_df = pd.DataFrame(topics).T
topic_words_df.columns = [f'word_{i+1}' for i in range(N_TOP_WORDS)]
topic_words_df.index = [f'topic_{i+1}' for i in range(N_TOPICS)]
topic_words_df.to_csv(f'{output_dir}/lda_topic_words.csv')

# Save document-topic assignments
doc_topics_df = df[['id', 'speaker', 'date', 'year', 'dominant_topic', 'dominant_topic_prob']].copy()
doc_topics_df['dominant_topic'] = doc_topics_df['dominant_topic'] + 1  # 1-indexed
doc_topics_df.to_csv(f'{output_dir}/lda_document_topics.csv', index=False)

# Save topic prevalence by year
topic_by_year_output = topic_by_year.copy()
topic_by_year_output.columns = [f'topic_{i+1}' for i in topic_by_year_output.columns]
topic_by_year_output.to_csv(f'{output_dir}/lda_topic_by_year.csv')

# Save full topic distribution for each document (for visualization)
topic_dist_df = pd.DataFrame(
    doc_topic_matrix,
    columns=[f'topic_{i+1}' for i in range(N_TOPICS)]
)
topic_dist_df['id'] = df['id'].values
topic_dist_df['year'] = df['year'].values
topic_dist_df['speaker'] = df['speaker'].values
topic_dist_df.to_csv(f'{output_dir}/lda_full_topic_distribution.csv', index=False)

print(f"\n{'='*60}")
print("FILES SAVED")
print(f"{'='*60}")
print(f"\nResults saved to: {output_dir}/")
print("  - lda_topic_words.csv: Top words for each topic")
print("  - lda_document_topics.csv: Dominant topic per speech")
print("  - lda_topic_by_year.csv: Topic prevalence over time")
print("  - lda_full_topic_distribution.csv: Full topic probabilities")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
