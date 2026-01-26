"""
Automatic Topic Labeling using Word Embeddings
Uses spaCy's pre-trained word vectors to find descriptive labels for LDA topics.
"""

import pandas as pd
import numpy as np

# Try to use spaCy with word vectors
try:
    import spacy
    # Load medium or large model (has word vectors)
    try:
        nlp = spacy.load('en_core_web_md')
    except OSError:
        print("Downloading spaCy model with word vectors...")
        import subprocess
        subprocess.run(['python3', '-m', 'spacy', 'download', 'en_core_web_md'], check=True)
        nlp = spacy.load('en_core_web_md')
    USE_SPACY = True
except ImportError:
    print("spaCy not available, using fallback method")
    USE_SPACY = False

# === Domain-specific candidate labels ===
# These are potential topic names relevant to Federal Reserve speeches
CANDIDATE_LABELS = [
    # Monetary Policy
    "Monetary Policy",
    "Interest Rates",
    "Inflation Targeting",
    "Price Stability",
    "Federal Funds Rate",

    # Labor & Employment
    "Labor Markets",
    "Employment",
    "Unemployment",
    "Workforce Development",
    "Labor Economics",

    # Financial Markets
    "Financial Markets",
    "Asset Prices",
    "Securities Markets",
    "Treasury Markets",
    "Bond Markets",
    "Equity Markets",

    # Banking & Regulation
    "Banking Regulation",
    "Financial Regulation",
    "Bank Supervision",
    "Prudential Regulation",
    "Capital Requirements",
    "Stress Testing",

    # Housing
    "Housing Markets",
    "Mortgage Markets",
    "Real Estate",
    "Housing Finance",
    "Homeownership",

    # Economic Growth
    "Economic Growth",
    "Economic Outlook",
    "GDP Growth",
    "Productivity",
    "Economic Output",

    # Financial Stability
    "Financial Stability",
    "Systemic Risk",
    "Macroprudential Policy",
    "Crisis Management",

    # Consumer & Community
    "Consumer Finance",
    "Consumer Protection",
    "Community Banking",
    "Small Business Lending",
    "Consumer Credit",

    # International
    "International Finance",
    "Global Economy",
    "Exchange Rates",
    "Trade Policy",

    # Fiscal Policy
    "Fiscal Policy",
    "Government Debt",
    "Public Finance",

    # Payments & Technology
    "Payment Systems",
    "Digital Payments",
    "Financial Technology",
    "Central Bank Digital Currency",

    # Education & Research
    "Economic Education",
    "Economic Research",
    "Financial Literacy",

    # Pandemic/Crisis
    "Pandemic Response",
    "Economic Crisis",
    "Emergency Lending",

    # Institutions
    "Financial Institutions",
    "Banking Industry",
    "Insurance Industry",
]

def get_embedding(text, nlp):
    """Get the word vector for a text."""
    doc = nlp(text)
    if doc.vector_norm == 0:
        return None
    return doc.vector

def compute_topic_centroid(words, nlp):
    """Compute the centroid of word embeddings for a list of words."""
    vectors = []
    for word in words:
        vec = get_embedding(word, nlp)
        if vec is not None:
            vectors.append(vec)

    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def find_best_label(topic_words, candidate_labels, nlp):
    """Find the best matching label for a topic based on embedding similarity."""
    # Compute topic centroid from top words
    topic_centroid = compute_topic_centroid(topic_words, nlp)

    if topic_centroid is None:
        return topic_words[0].title()  # Fallback to first word

    # Compute similarity with each candidate label
    best_label = None
    best_similarity = -1

    for label in candidate_labels:
        label_vec = get_embedding(label, nlp)
        if label_vec is None:
            continue

        # Cosine similarity
        similarity = np.dot(topic_centroid, label_vec) / (
            np.linalg.norm(topic_centroid) * np.linalg.norm(label_vec)
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_label = label

    return best_label, best_similarity

def label_topics_with_embeddings(topic_words_df, nlp):
    """Label all topics using word embeddings."""
    labels = {}

    for idx, row in topic_words_df.iterrows():
        topic_num = idx.replace('topic_', '')
        words = [row[f'word_{i}'] for i in range(1, 11) if f'word_{i}' in row.index]

        label, similarity = find_best_label(words, CANDIDATE_LABELS, nlp)
        labels[idx] = {
            'label': label,
            'similarity': similarity,
            'top_words': ', '.join(words[:3])
        }
        print(f"  Topic {topic_num}: {label} (similarity: {similarity:.3f})")
        print(f"    Words: {', '.join(words[:5])}")

    return labels

def label_topics_fallback(topic_words_df):
    """Fallback labeling using simple heuristics when spaCy is not available."""
    # Simple mapping based on keyword matching
    keyword_to_label = {
        'inflation': 'Inflation & Prices',
        'labor': 'Labor Markets',
        'employment': 'Employment',
        'unemployment': 'Unemployment',
        'monetary': 'Monetary Policy',
        'interest': 'Interest Rates',
        'bank': 'Banking',
        'banking': 'Banking',
        'regulation': 'Financial Regulation',
        'regulatory': 'Financial Regulation',
        'capital': 'Capital & Regulation',
        'mortgage': 'Housing & Mortgages',
        'housing': 'Housing Markets',
        'home': 'Housing Markets',
        'credit': 'Credit & Lending',
        'loan': 'Credit & Lending',
        'community': 'Community Banking',
        'consumer': 'Consumer Finance',
        'payment': 'Payment Systems',
        'pandemic': 'Pandemic Response',
        'covid': 'Pandemic Response',
        'asset': 'Asset Markets',
        'security': 'Securities Markets',
        'treasury': 'Treasury Markets',
        'fund': 'Financial Markets',
        'institution': 'Financial Institutions',
        'government': 'Government & Fiscal',
        'fiscal': 'Fiscal Policy',
        'productivity': 'Economic Output',
        'growth': 'Economic Growth',
        'output': 'Economic Output',
        'worker': 'Labor & Workers',
        'education': 'Education & Workforce',
        'research': 'Economic Research',
        'stability': 'Financial Stability',
        'crisis': 'Crisis Response',
        'stress': 'Stress Testing',
        'global': 'Global Economy',
        'country': 'International',
        'central': 'Central Banking',
    }

    labels = {}

    for idx, row in topic_words_df.iterrows():
        topic_num = idx.replace('topic_', '')
        words = [row[f'word_{i}'].lower() for i in range(1, 6) if f'word_{i}' in row.index]

        # Find first matching keyword
        label = None
        for word in words:
            # Check exact match
            if word in keyword_to_label:
                label = keyword_to_label[word]
                break
            # Check partial match
            for keyword, lbl in keyword_to_label.items():
                if keyword in word or word in keyword:
                    label = lbl
                    break
            if label:
                break

        if not label:
            # Fallback: capitalize first word
            label = words[0].replace('_', ' ').title()

        labels[idx] = {
            'label': label,
            'similarity': 0.0,
            'top_words': ', '.join(words[:3])
        }
        print(f"  Topic {topic_num}: {label}")
        print(f"    Words: {', '.join(words[:5])}")

    return labels

# === Main ===
if __name__ == '__main__':
    print("=" * 60)
    print("AUTOMATIC TOPIC LABELING")
    print("=" * 60)

    # Load topic words
    lda_output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output'
    topic_words_df = pd.read_csv(f'{lda_output_dir}/lda_topic_words.csv', index_col=0)

    print(f"\nLoaded {len(topic_words_df)} topics")
    print("\nLabeling topics...\n")

    if USE_SPACY:
        print("Using spaCy word embeddings for semantic matching\n")
        labels = label_topics_with_embeddings(topic_words_df, nlp)
    else:
        print("Using keyword-based fallback method\n")
        labels = label_topics_fallback(topic_words_df)

    # Save labels
    labels_df = pd.DataFrame(labels).T
    labels_df.to_csv(f'{lda_output_dir}/topic_labels.csv')

    print(f"\n{'=' * 60}")
    print("TOPIC LABELS SUMMARY")
    print("=" * 60)
    for topic_id, info in labels.items():
        topic_num = topic_id.replace('topic_', '')
        print(f"  Topic {topic_num}: {info['label']}")

    print(f"\nLabels saved to: {lda_output_dir}/topic_labels.csv")
