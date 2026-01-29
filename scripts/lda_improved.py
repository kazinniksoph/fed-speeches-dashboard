"""
Improved LDA Topic Modeling for Federal Reserve Speeches
- Enhanced stopwords to remove Q&A filler and event metadata
- Coherence-based K optimization
- Better preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Disable multiprocessing for gensim
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# NLP libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# === Configuration ===
K_RANGE = range(8, 25)  # Test different numbers of topics
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

# === Enhanced Stopwords ===
# Base English stopwords
base_stopwords = set(stopwords.words('english'))

# Fed-specific stopwords (institutional terms)
fed_institutional = {
    'federal', 'reserve', 'board', 'governors', 'fomc', 'committee',
    'chairman', 'president', 'vice', 'governor', 'member', 'staff',
    'mr', 'ms', 'dr', 'madam', 'sir'
}

# Presentation artifacts
presentation_artifacts = {
    'figure', 'chart', 'slide', 'table', 'graph', 'exhibit',
    'footnote', 'note', 'appendix', 'page', 'percent', 'percentage'
}

# Conversational filler (Q&A sessions)
conversational_filler = {
    'people', 'thing', 'things', 'going', 'back', 'want', 'question',
    'mean', 'right', 'look', 'great', 'really', 'still', 'something',
    'around', 'keep', 'actually', 'might', 'guess', 'okay', 'yeah',
    'yes', 'no', 'well', 'sort', 'kind', 'lot', 'bit', 'stuff',
    'getting', 'got', 'saying', 'said', 'tell', 'told', 'ask', 'asked',
    'answer', 'point', 'talking', 'talk', 'spoke', 'speak', 'hear',
    'heard', 'listen', 'sure', 'probably', 'maybe', 'perhaps',
    'anyway', 'obviously', 'clearly', 'certainly', 'definitely',
    'absolutely', 'exactly', 'basically', 'essentially', 'generally',
    'typically', 'usually', 'often', 'always', 'never', 'sometimes'
}

# Event/venue metadata
event_metadata = {
    'york', 'washington', 'chicago', 'boston', 'philadelphia',
    'cleveland', 'richmond', 'atlanta', 'louis', 'minneapolis',
    'kansas', 'dallas', 'francisco', 'city', 'conference', 'symposium',
    'meeting', 'speech', 'remarks', 'address', 'lecture', 'event',
    'forum', 'summit', 'workshop', 'seminar', 'panel', 'discussion',
    'today', 'tonight', 'morning', 'afternoon', 'evening', 'yesterday',
    'tomorrow', 'week', 'month', 'year', 'years', 'day', 'days'
}

# Generic verbs and words
generic_words = {
    'would', 'could', 'should', 'may', 'might', 'must', 'will', 'shall',
    'also', 'well', 'one', 'two', 'three', 'four', 'five', 'first',
    'second', 'third', 'new', 'like', 'think', 'see', 'time', 'make',
    'way', 'good', 'get', 'take', 'know', 'come', 'go', 'say', 'thank',
    'use', 'used', 'using', 'work', 'working', 'need', 'needs', 'needed',
    'help', 'important', 'part', 'place', 'case', 'example', 'fact',
    'number', 'amount', 'level', 'term', 'terms', 'area', 'areas'
}

# Combine all stopwords
ALL_STOPWORDS = (base_stopwords | fed_institutional | presentation_artifacts |
                 conversational_filler | event_metadata | generic_words)


def preprocess_text(text, lemmatizer):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in ALL_STOPWORDS and len(token) > 3
    ]
    return ' '.join(tokens)


def compute_coherence_umass(lda_model, doc_term_matrix, feature_names):
    """Compute UMass coherence (doesn't require gensim multiprocessing)."""
    # UMass coherence: sum of log(D(wi, wj) + 1 / D(wj)) for top words
    # Simpler and faster than c_v
    n_topics = lda_model.n_components
    topics = lda_model.components_

    # Get document frequencies
    doc_freq = np.asarray((doc_term_matrix > 0).sum(axis=0)).flatten()
    n_docs = doc_term_matrix.shape[0]

    coherences = []
    for topic in topics:
        top_word_indices = topic.argsort()[:-11:-1]  # Top 10 words
        coherence = 0
        for i, wi in enumerate(top_word_indices[1:], 1):
            for wj in top_word_indices[:i]:
                # Co-occurrence: documents containing both words
                co_doc = ((doc_term_matrix[:, wi] > 0).toarray().flatten() &
                          (doc_term_matrix[:, wj] > 0).toarray().flatten()).sum()
                # UMass formula
                if doc_freq[wj] > 0:
                    coherence += np.log((co_doc + 1) / doc_freq[wj])
        coherences.append(coherence)

    return np.mean(coherences)


def main():
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

    print(f"\nTotal stopwords: {len(ALL_STOPWORDS)}")
    print("Preprocessing text...")
    lemmatizer = WordNetLemmatizer()
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, lemmatizer))
    print("Preprocessing complete")

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

    print(f"\nFinding optimal K (testing {K_RANGE.start}-{K_RANGE.stop-1} topics)...")

    coherence_scores = []
    models = {}

    for k in K_RANGE:
        print(f"  Training K={k}...", end=" ", flush=True)

        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method='online',
            random_state=42,
            max_iter=30,
            n_jobs=1,
            verbose=0
        )
        lda.fit(doc_term_matrix)

        # Use UMass coherence (faster, no multiprocessing)
        coherence = compute_coherence_umass(lda, doc_term_matrix, feature_names)

        # Extract topics
        topics_words = []
        for topic in lda.components_:
            top_word_indices = topic.argsort()[:-N_TOP_WORDS - 1:-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topics_words.append(top_words)

        coherence_scores.append((k, coherence))
        models[k] = (lda, topics_words)
        print(f"coherence={coherence:.4f}")

    # Find optimal K (higher UMass = better, but it's negative so we use max)
    best_k, best_coherence = max(coherence_scores, key=lambda x: x[1])
    print(f"\nOptimal K={best_k} (coherence={best_coherence:.4f})")

    # Use best model
    best_lda, best_topics = models[best_k]
    doc_topic_matrix = best_lda.transform(doc_term_matrix)

    # Display results
    print(f"\n{'='*60}")
    print(f"IMPROVED LDA RESULTS - {best_k} TOPICS")
    print(f"{'='*60}\n")

    for i, words in enumerate(best_topics):
        print(f"Topic {i}: {', '.join(words[:10])}")

    # Add topic assignments
    df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)
    df['dominant_topic_prob'] = doc_topic_matrix.max(axis=1)

    for i in range(best_k):
        df[f'topic_{i}_prob'] = doc_topic_matrix[:, i]

    # Save results
    output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/lda_improved_results'
    os.makedirs(output_dir, exist_ok=True)

    # Save coherence scores
    coherence_df = pd.DataFrame(coherence_scores, columns=['k', 'coherence'])
    coherence_df.to_csv(f'{output_dir}/coherence_scores.csv', index=False)

    # Save topic definitions
    topic_defs = []
    for i, words in enumerate(best_topics):
        topic_defs.append({
            'topic_id': i,
            'n_dominant_speeches': int((df['dominant_topic'] == i).sum()),
            'avg_probability': float(doc_topic_matrix[:, i].mean()),
            'top_words': ', '.join(words[:10]),
            'all_top_words': ', '.join(words)
        })
    topic_defs_df = pd.DataFrame(topic_defs)
    topic_defs_df.to_csv(f'{output_dir}/topic_definitions_k{best_k}.csv', index=False)

    # Save speech assignments
    speech_cols = ['id', 'speaker', 'date', 'year', 'source', 'dominant_topic', 'dominant_topic_prob']
    topic_prob_cols = [f'topic_{i}_prob' for i in range(best_k)]
    speech_topics_df = df[speech_cols + topic_prob_cols].copy()
    speech_topics_df.to_csv(f'{output_dir}/speech_topic_assignments.csv', index=False)

    # Save model info
    with open(f'{output_dir}/model_info.txt', 'w') as f:
        f.write(f"Optimal K: {best_k}\n")
        f.write(f"Best Coherence (UMass): {best_coherence:.4f}\n")
        f.write(f"Total speeches: {len(df)}\n")
        f.write(f"Vocabulary size: {len(feature_names)}\n")
        f.write(f"\nCoherence scores by K:\n")
        for k, c in coherence_scores:
            marker = " <-- BEST" if k == best_k else ""
            f.write(f"  K={k}: {c:.4f}{marker}\n")

    print(f"\n{'='*60}")
    print("FILES SAVED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
