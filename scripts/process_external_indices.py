"""
Process external sentiment/uncertainty indices for the dashboard.
- SF Fed Daily News Sentiment Index
- Economic Policy Uncertainty - Monetary Policy Component
"""

import pandas as pd
import numpy as np

output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output'

# === 1. SF Fed Daily News Sentiment Index ===
print("Processing SF Fed News Sentiment Index...")
news = pd.read_excel(f'{output_dir}/news_sentiment_data.xlsx', sheet_name='Data')
news['date'] = pd.to_datetime(news['date'])
news['year'] = news['date'].dt.year

# Aggregate to yearly
news_yearly = news.groupby('year')['News Sentiment'].mean().reset_index()
news_yearly.columns = ['year', 'news_sentiment']
print(f"  Years: {news_yearly['year'].min()} to {news_yearly['year'].max()}")

# === 2. EPU Monetary Policy Index ===
print("\nProcessing EPU Monetary Policy Index...")
epu = pd.read_csv(f'{output_dir}/epu_monetary.csv')
epu['date'] = pd.to_datetime(epu['observation_date'])
epu['year'] = epu['date'].dt.year

# Aggregate to yearly
epu_yearly = epu.groupby('year')['EPUMONETARY'].mean().reset_index()
epu_yearly.columns = ['year', 'epu_monetary']
print(f"  Years: {epu_yearly['year'].min()} to {epu_yearly['year'].max()}")

# === Merge both ===
merged = pd.merge(news_yearly, epu_yearly, on='year', how='outer')
merged = merged.sort_values('year')

# Filter to our speech data range (1995-2025)
merged = merged[(merged['year'] >= 1995) & (merged['year'] <= 2025)]

# Save
merged.to_csv(f'{output_dir}/external_indices_yearly.csv', index=False)

print(f"\n{'='*60}")
print("EXTERNAL INDICES SUMMARY (1995-2025)")
print("="*60)

print("\nSF Fed News Sentiment (higher = positive economic sentiment):")
print(f"  Mean: {merged['news_sentiment'].mean():.3f}")
print(f"  Min: {merged['news_sentiment'].min():.3f} ({merged.loc[merged['news_sentiment'].idxmin(), 'year']})")
print(f"  Max: {merged['news_sentiment'].max():.3f} ({merged.loc[merged['news_sentiment'].idxmax(), 'year']})")

print("\nEPU Monetary Policy (higher = more policy uncertainty):")
print(f"  Mean: {merged['epu_monetary'].mean():.1f}")
print(f"  Min: {merged['epu_monetary'].min():.1f} ({merged.loc[merged['epu_monetary'].idxmin(), 'year']})")
print(f"  Max: {merged['epu_monetary'].max():.1f} ({merged.loc[merged['epu_monetary'].idxmax(), 'year']})")

print(f"\nSaved to: {output_dir}/external_indices_yearly.csv")
