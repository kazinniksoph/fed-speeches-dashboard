"""
Score Fed speeches for hawkish/dovish sentiment using FinBERT-FOMC model.
Model: karoldobiczek/finbert-fomc
Labels: hawkish, dovish, neutral
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
import glob
data_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/year_all'
output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output'

print("Loading model...")
classifier = pipeline(
    'text-classification',
    model='karoldobiczek/finbert-fomc',
    device=-1,  # CPU
    truncation=True,
    max_length=512
)

print("Loading speeches...")
files = sorted(glob.glob(f'{data_dir}/*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Parse dates
from datetime import datetime
def parse_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        return datetime.strptime(str(date_str), '%d%b%Y')
    except:
        return None

df['parsed_date'] = df['date'].apply(parse_date)
df['year'] = df['parsed_date'].dt.year

# Filter to valid speeches with text
df = df.dropna(subset=['text', 'year'])
print(f"Processing {len(df)} speeches...")

# Score each speech
# For long speeches, we'll take the first 512 tokens (model limit)
# and also sample from middle and end, then average

def score_speech(text):
    """Score a speech, handling long texts by sampling multiple sections."""
    if pd.isna(text) or len(str(text).strip()) < 50:
        return {'label': 'neutral', 'score': 0.5}

    text = str(text)
    words = text.split()

    # If short enough, score directly
    if len(words) <= 400:
        try:
            result = classifier(text, truncation=True)[0]
            return result
        except:
            return {'label': 'neutral', 'score': 0.5}

    # For long texts, sample beginning, middle, end
    sections = []
    chunk_size = 400  # words

    # Beginning
    sections.append(' '.join(words[:chunk_size]))

    # Middle
    mid_start = len(words) // 2 - chunk_size // 2
    sections.append(' '.join(words[mid_start:mid_start + chunk_size]))

    # End
    sections.append(' '.join(words[-chunk_size:]))

    try:
        results = classifier(sections, truncation=True)

        # Convert to numeric scores: hawkish=+1, dovish=-1, neutral=0
        scores = []
        for r in results:
            if r['label'] == 'hawkish':
                scores.append(r['score'])
            elif r['label'] == 'dovish':
                scores.append(-r['score'])
            else:
                scores.append(0)

        avg_score = np.mean(scores)

        # Determine overall label
        if avg_score > 0.1:
            return {'label': 'hawkish', 'score': abs(avg_score)}
        elif avg_score < -0.1:
            return {'label': 'dovish', 'score': abs(avg_score)}
        else:
            return {'label': 'neutral', 'score': 1 - abs(avg_score)}

    except Exception as e:
        return {'label': 'neutral', 'score': 0.5}

# Process all speeches
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring speeches"):
    result = score_speech(row['text'])
    results.append({
        'idx': idx,
        'year': row['year'],
        'speaker': row['speaker'],
        'label': result['label'],
        'score': result['score']
    })

results_df = pd.DataFrame(results)

# Create numeric sentiment score: hawkish positive, dovish negative
def to_numeric(row):
    if row['label'] == 'hawkish':
        return row['score']
    elif row['label'] == 'dovish':
        return -row['score']
    else:
        return 0

results_df['sentiment_numeric'] = results_df.apply(to_numeric, axis=1)

# Save speech-level results
results_df.to_csv(f'{output_dir}/hawkish_dovish_by_speech.csv', index=False)
print(f"\nSaved speech-level results to {output_dir}/hawkish_dovish_by_speech.csv")

# Aggregate by year
yearly = results_df.groupby('year').agg({
    'sentiment_numeric': 'mean',
    'label': lambda x: (x == 'hawkish').sum() / len(x) * 100  # % hawkish
}).reset_index()
yearly.columns = ['year', 'sentiment_score', 'pct_hawkish']

# Also compute % dovish
yearly_dovish = results_df.groupby('year')['label'].apply(lambda x: (x == 'dovish').sum() / len(x) * 100).reset_index()
yearly_dovish.columns = ['year', 'pct_dovish']
yearly = yearly.merge(yearly_dovish, on='year')

yearly.to_csv(f'{output_dir}/hawkish_dovish_by_year.csv', index=False)
print(f"Saved yearly results to {output_dir}/hawkish_dovish_by_year.csv")

# Aggregate by speaker (min 10 speeches)
speaker_counts = results_df.groupby('speaker').size()
valid_speakers = speaker_counts[speaker_counts >= 10].index

speaker_df = results_df[results_df['speaker'].isin(valid_speakers)]
speaker_agg = speaker_df.groupby('speaker').agg({
    'sentiment_numeric': 'mean',
    'label': lambda x: (x == 'hawkish').sum() / len(x) * 100
}).reset_index()
speaker_agg.columns = ['speaker', 'sentiment_score', 'pct_hawkish']
speaker_agg = speaker_agg.sort_values('sentiment_score')

speaker_agg.to_csv(f'{output_dir}/hawkish_dovish_by_speaker.csv', index=False)
print(f"Saved speaker results to {output_dir}/hawkish_dovish_by_speaker.csv")

# Summary
print(f"\n{'='*60}")
print("HAWKISH/DOVISH SENTIMENT SUMMARY (FinBERT-FOMC)")
print("="*60)

print(f"\nOverall distribution:")
print(results_df['label'].value_counts(normalize=True).mul(100).round(1))

print(f"\nBy year sentiment score (hawkish=+, dovish=-):")
print(f"  Most hawkish year: {yearly.loc[yearly['sentiment_score'].idxmax(), 'year']} ({yearly['sentiment_score'].max():.3f})")
print(f"  Most dovish year:  {yearly.loc[yearly['sentiment_score'].idxmin(), 'year']} ({yearly['sentiment_score'].min():.3f})")

print(f"\nTop 5 most hawkish speakers:")
for _, row in speaker_agg.tail(5).iloc[::-1].iterrows():
    print(f"  {row['speaker']}: {row['sentiment_score']:.3f}")

print(f"\nTop 5 most dovish speakers:")
for _, row in speaker_agg.head(5).iterrows():
    print(f"  {row['speaker']}: {row['sentiment_score']:.3f}")
