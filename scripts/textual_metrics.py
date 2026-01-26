"""
Textual Metrics Analysis for Federal Reserve Speeches
Computes readability, sentiment (hawkish/dovish), and uncertainty metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter

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
df = df.dropna(subset=['text', 'year'])
print(f"Analyzing {len(df)} speeches")

# === 1. Readability Metrics ===
print("\n" + "="*60)
print("COMPUTING READABILITY SCORES")
print("="*60)

def count_syllables(word):
    """Estimate syllable count for a word."""
    word = word.lower()
    if len(word) <= 3:
        return 1

    # Remove trailing e
    word = re.sub(r'e$', '', word)

    # Count vowel groups
    vowels = 'aeiouy'
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    return max(1, count)

def compute_readability(text):
    """Compute Flesch-Kincaid Grade Level and Gunning Fog Index."""
    if pd.isna(text) or len(text) < 100:
        return None, None

    # Clean text
    text = re.sub(r'[^\w\s\.]', ' ', text)

    # Count sentences (approximate)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    num_sentences = max(1, len(sentences))

    # Count words
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    num_words = len(words)

    if num_words < 100:
        return None, None

    # Count syllables and complex words (3+ syllables)
    total_syllables = 0
    complex_words = 0

    for word in words:
        syllables = count_syllables(word)
        total_syllables += syllables
        if syllables >= 3:
            complex_words += 1

    # Flesch-Kincaid Grade Level
    # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59

    # Gunning Fog Index
    # 0.4 * ((words/sentences) + 100 * (complex_words/words))
    fog_index = 0.4 * ((num_words / num_sentences) + 100 * (complex_words / num_words))

    return round(fk_grade, 2), round(fog_index, 2)

print("Computing readability for each speech...")
readability_results = df['text'].apply(lambda x: compute_readability(x))
df['fk_grade'] = readability_results.apply(lambda x: x[0] if x else None)
df['fog_index'] = readability_results.apply(lambda x: x[1] if x else None)

# === 2. Hawkish/Dovish Sentiment ===
print("\n" + "="*60)
print("COMPUTING HAWKISH/DOVISH SENTIMENT")
print("="*60)

# Word lists based on Fed communication research
hawkish_words = {
    # Inflation concerns
    'inflation', 'inflationary', 'overheating', 'overheated', 'price pressure',
    'rising prices', 'price stability', 'above target', 'elevated',
    # Tightening language
    'tighten', 'tightening', 'raise', 'raising', 'hike', 'hiking',
    'increase rates', 'higher rates', 'normalize', 'normalization',
    'reduce accommodation', 'less accommodative', 'restrictive',
    # Strong economy concerns
    'strong growth', 'rapid growth', 'excessive', 'unsustainable',
    'vigilant', 'vigilance', 'concerned', 'upside risk',
    # Balance sheet
    'tapering', 'taper', 'reduce balance sheet', 'quantitative tightening',
}

dovish_words = {
    # Growth concerns
    'slowdown', 'slowing', 'weakness', 'weak', 'recession', 'recessionary',
    'downturn', 'contraction', 'below trend', 'below potential',
    'downside risk', 'headwinds',
    # Easing language
    'ease', 'easing', 'cut', 'cutting', 'lower rates', 'reduce rates',
    'accommodative', 'accommodation', 'stimulus', 'support',
    'patient', 'patience', 'gradual', 'cautious',
    # Employment concerns
    'unemployment', 'jobless', 'job losses', 'labor slack', 'underemployment',
    # Low inflation
    'disinflation', 'deflation', 'below target', 'low inflation', 'subdued inflation',
    # Balance sheet
    'quantitative easing', 'asset purchases', 'expand balance sheet',
}

def compute_sentiment(text):
    """Compute hawkish/dovish sentiment score."""
    if pd.isna(text):
        return None, 0, 0

    text_lower = text.lower()
    word_count = len(text_lower.split())

    if word_count < 100:
        return None, 0, 0

    # Count hawkish and dovish terms
    hawkish_count = 0
    dovish_count = 0

    for term in hawkish_words:
        hawkish_count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))

    for term in dovish_words:
        dovish_count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))

    # Normalize by word count (per 1000 words)
    hawkish_norm = (hawkish_count / word_count) * 1000
    dovish_norm = (dovish_count / word_count) * 1000

    # Sentiment score: positive = hawkish, negative = dovish
    # Range approximately -10 to +10
    if hawkish_norm + dovish_norm > 0:
        sentiment = (hawkish_norm - dovish_norm) / (hawkish_norm + dovish_norm) * 10
    else:
        sentiment = 0

    return round(sentiment, 2), round(hawkish_norm, 2), round(dovish_norm, 2)

print("Computing sentiment for each speech...")
sentiment_results = df['text'].apply(lambda x: compute_sentiment(x))
df['sentiment_score'] = sentiment_results.apply(lambda x: x[0] if x else None)
df['hawkish_density'] = sentiment_results.apply(lambda x: x[1] if x else None)
df['dovish_density'] = sentiment_results.apply(lambda x: x[2] if x else None)

# === 3. Uncertainty Index ===
print("\n" + "="*60)
print("COMPUTING UNCERTAINTY INDEX")
print("="*60)

uncertainty_words = {
    # Modal verbs expressing uncertainty
    'may', 'might', 'could', 'would', 'should',
    # Uncertainty expressions
    'uncertain', 'uncertainty', 'uncertainties',
    'unclear', 'unknown', 'unpredictable',
    'risk', 'risks', 'risky',
    'volatile', 'volatility',
    'possible', 'possibly', 'possibility',
    'potential', 'potentially',
    'likely', 'unlikely', 'likelihood',
    'expect', 'expected', 'expectation', 'expectations',
    'anticipate', 'anticipated',
    'estimate', 'estimated', 'estimates',
    'approximate', 'approximately',
    'perhaps', 'maybe',
    'appear', 'appears', 'seem', 'seems',
    'suggest', 'suggests', 'suggesting',
    'depend', 'depends', 'dependent', 'depending',
    'conditional', 'contingent',
    'tentative', 'preliminary',
    'evolving', 'evolve',
}

def compute_uncertainty(text):
    """Compute uncertainty index based on hedging language."""
    if pd.isna(text):
        return None

    text_lower = text.lower()
    word_count = len(text_lower.split())

    if word_count < 100:
        return None

    # Count uncertainty terms
    uncertainty_count = 0
    for term in uncertainty_words:
        uncertainty_count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))

    # Normalize by word count (per 1000 words)
    uncertainty_index = (uncertainty_count / word_count) * 1000

    return round(uncertainty_index, 2)

print("Computing uncertainty for each speech...")
df['uncertainty_index'] = df['text'].apply(compute_uncertainty)

# === Aggregate by Year ===
print("\n" + "="*60)
print("AGGREGATING METRICS BY YEAR")
print("="*60)

yearly_metrics = df.groupby('year').agg({
    'fk_grade': 'mean',
    'fog_index': 'mean',
    'sentiment_score': 'mean',
    'hawkish_density': 'mean',
    'dovish_density': 'mean',
    'uncertainty_index': 'mean',
}).round(2)

print("\nYearly averages:")
print(yearly_metrics.tail(10))

# === Save Results ===
output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output'

# Save speech-level metrics
speech_metrics = df[['id', 'speaker', 'date', 'year',
                     'fk_grade', 'fog_index',
                     'sentiment_score', 'hawkish_density', 'dovish_density',
                     'uncertainty_index']].copy()
speech_metrics.to_csv(f'{output_dir}/textual_metrics_by_speech.csv', index=False)

# Save yearly aggregates
yearly_metrics.to_csv(f'{output_dir}/textual_metrics_by_year.csv')

print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nReadability (Flesch-Kincaid Grade Level):")
print(f"  Mean: {df['fk_grade'].mean():.1f} (college level)")
print(f"  Range: {df['fk_grade'].min():.1f} - {df['fk_grade'].max():.1f}")

print(f"\nSentiment Score (-10=dovish, +10=hawkish):")
print(f"  Mean: {df['sentiment_score'].mean():.2f}")
print(f"  Most hawkish year: {yearly_metrics['sentiment_score'].idxmax()} ({yearly_metrics['sentiment_score'].max():.2f})")
print(f"  Most dovish year: {yearly_metrics['sentiment_score'].idxmin()} ({yearly_metrics['sentiment_score'].min():.2f})")

print(f"\nUncertainty Index (hedging words per 1000):")
print(f"  Mean: {df['uncertainty_index'].mean():.1f}")
print(f"  Highest uncertainty year: {yearly_metrics['uncertainty_index'].idxmax()} ({yearly_metrics['uncertainty_index'].max():.1f})")

print(f"\n{'='*60}")
print("FILES SAVED")
print("="*60)
print(f"\nResults saved to: {output_dir}/")
print("  - textual_metrics_by_speech.csv: Metrics for each speech")
print("  - textual_metrics_by_year.csv: Yearly aggregates")
