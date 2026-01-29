#!/usr/bin/env python3
"""
OpenAI Embedding Analysis for Federal Reserve Speeches
Tests text-embedding-3-small and text-embedding-3-large models.
Compares results with existing sentence-transformer embeddings.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import glob
import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
OPENAI_MODELS = [
    "text-embedding-3-small",  # 1536 dims, cheaper
    "text-embedding-3-large",  # 3072 dims, higher quality
]

# Batch size for API calls (OpenAI allows up to 2048 inputs per request)
BATCH_SIZE = 100

# Maximum tokens per text (OpenAI embedding models support 8191 tokens)
MAX_TOKENS = 8000  # Leave some buffer

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


def truncate_text(text, max_words=6000):
    """Truncate text to approximately max_words to stay within token limits."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text


def get_embeddings_batch(client, texts, model):
    """Get embeddings for a batch of texts using OpenAI API."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]


def embed_speeches(client, df, model_name):
    """
    Generate embeddings for all speeches using OpenAI API.
    
    Parameters:
    -----------
    client : OpenAI
        OpenAI client instance
    df : pd.DataFrame
        DataFrame with 'text' column
    model_name : str
        OpenAI embedding model name
    
    Returns:
    --------
    np.ndarray : Embedding matrix of shape (n_speeches, embedding_dim)
    """
    texts = [truncate_text(text) for text in df['text'].tolist()]
    all_embeddings = []
    
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        print(f"    Processing batch {batch_num}/{n_batches} ({len(batch)} texts)...", end=" ", flush=True)
        
        try:
            embeddings = get_embeddings_batch(client, batch, model_name)
            all_embeddings.extend(embeddings)
            print("Done")
            
            # Rate limiting - be nice to the API
            if i + BATCH_SIZE < len(texts):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error: {e}")
            # Retry with smaller batches if needed
            for j, text in enumerate(batch):
                try:
                    emb = get_embeddings_batch(client, [text], model_name)
                    all_embeddings.extend(emb)
                except Exception as e2:
                    print(f"    Failed on text {i+j}: {e2}")
                    # Use zero vector as fallback
                    all_embeddings.append([0.0] * (1536 if "small" in model_name else 3072))
                time.sleep(0.2)
    
    return np.array(all_embeddings)


def get_top_similar_pairs(sim_matrix, df, n_pairs=15, exclude_duplicates=True):
    """Find the most similar speech pairs."""
    n = len(sim_matrix)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if exclude_duplicates and sim_matrix[i, j] > 0.99:
                continue
            pairs.append((
                i, j, sim_matrix[i, j],
                df.iloc[i]['speaker'], df.iloc[j]['speaker'],
                df.iloc[i]['date'], df.iloc[j]['date'],
                df.iloc[i]['year'], df.iloc[j]['year']
            ))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:n_pairs]


def main():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("=" * 80)
    print("OPENAI EMBEDDING ANALYSIS FOR FEDERAL RESERVE SPEECHES")
    print("=" * 80)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    print("\n✓ OpenAI client initialized")
    
    # === Load Data ===
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Try sample_exercise data first (smaller, good for testing)
    sample_dir = Path('/Users/sophiakazinnik/Research/central_bank_speeches_communication/sample_exercise/data')
    
    if sample_dir.exists():
        print("\nLoading from sample_exercise/data/ (3 years: 2005, 2015, 2025)...")
        dfs = []
        for year in [2005, 2015, 2025]:
            fpath = sample_dir / f'speeches_with_time_and_text_{year}.csv'
            if fpath.exists():
                temp_df = pd.read_csv(fpath)
                temp_df['year'] = year
                dfs.append(temp_df)
                print(f"  {year}: {len(temp_df)} speeches")
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            print("No sample data found, trying full corpus...")
            df = None
    else:
        df = None
    
    # Fall back to full corpus if needed
    if df is None or len(df) == 0:
        data_dir = Path('/Users/sophiakazinnik/Research/central_bank_speeches_communication/year_all')
        if data_dir.exists():
            print("\nLoading from year_all/...")
            files = sorted(glob.glob(str(data_dir / '*.csv')))
            dfs = []
            for f in files:
                temp_df = pd.read_csv(f)
                if len(temp_df) > 0:
                    dfs.append(temp_df)
            df = pd.concat(dfs, ignore_index=True)
        else:
            print("ERROR: No speech data found!")
            return
    
    # Normalize speakers and parse dates
    df['speaker'] = df['speaker'].apply(normalize_speaker)
    
    def parse_date(date_str):
        if pd.isna(date_str):
            return None
        try:
            return datetime.strptime(str(date_str), '%d%b%Y')
        except:
            return None
    
    if 'year' not in df.columns:
        df['parsed_date'] = df['date'].apply(parse_date)
        df['year'] = df['parsed_date'].dt.year
    
    df = df.dropna(subset=['text'])
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\nTotal speeches: {len(df)}")
    print(f"Unique speakers: {df['speaker'].nunique()}")
    print(f"Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print(f"Word count: {df['word_count'].min()} - {df['word_count'].max()} (mean: {df['word_count'].mean():.0f})")
    
    # === Generate Embeddings ===
    output_dir = Path('/Users/sophiakazinnik/Research/central_bank_speeches_communication/openai_embedding_results')
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for model_name in OPENAI_MODELS:
        print(f"\n{'=' * 80}")
        print(f"EMBEDDING WITH: {model_name}")
        print("=" * 80)
        
        start_time = time.time()
        embeddings = embed_speeches(client, df, model_name)
        elapsed = time.time() - start_time
        
        print(f"\n  Embedding shape: {embeddings.shape}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        
        # Compute similarity matrix
        print("  Computing cosine similarity matrix...", end=" ", flush=True)
        sim_matrix = cosine_similarity(embeddings)
        print("Done")
        
        # Save embeddings
        model_short = model_name.replace("text-embedding-3-", "")
        np.save(output_dir / f'embeddings_{model_short}.npy', embeddings)
        np.save(output_dir / f'similarity_{model_short}.npy', sim_matrix)
        
        results[model_name] = {
            'embeddings': embeddings,
            'similarity': sim_matrix,
            'dim': embeddings.shape[1],
            'time': elapsed
        }
        
        # Find top similar pairs
        print(f"\n  Top 10 Most Similar Pairs ({model_name}):")
        print("  " + "-" * 70)
        top_pairs = get_top_similar_pairs(sim_matrix, df, n_pairs=10)
        for rank, (i, j, sim, sp1, sp2, d1, d2, y1, y2) in enumerate(top_pairs, 1):
            same = "SAME" if sp1 == sp2 else "DIFF"
            print(f"  {rank:2}. {sim:.4f} [{same}] {sp1} ({d1}, {y1}) ↔ {sp2} ({d2}, {y2})")
    
    # === Model Comparison ===
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Compare the two OpenAI models
    if len(results) == 2:
        models = list(results.keys())
        sim1 = results[models[0]]['similarity']
        sim2 = results[models[1]]['similarity']
        
        upper_tri = np.triu_indices(len(df), k=1)
        flat1 = sim1[upper_tri]
        flat2 = sim2[upper_tri]
        
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        
        print(f"\n  {models[0]} vs {models[1]}:")
        print(f"    Correlation: {correlation:.4f}")
        print(f"    Small model range: [{flat1.min():.3f}, {flat1.max():.3f}] (mean: {flat1.mean():.3f})")
        print(f"    Large model range: [{flat2.min():.3f}, {flat2.max():.3f}] (mean: {flat2.mean():.3f})")
    
    # === Compare with existing embeddings if available ===
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH SENTENCE TRANSFORMERS")
    print("=" * 80)
    
    existing_emb_paths = [
        Path('/Users/sophiakazinnik/Research/central_bank_speeches_communication/sample_exercise/data/similarity_general_full.npy'),
        Path('/Users/sophiakazinnik/Research/central_bank_speeches_communication/sample_exercise/data/similarity_finance_full.npy'),
    ]
    
    for emb_path in existing_emb_paths:
        if emb_path.exists():
            existing_sim = np.load(emb_path)
            model_label = "MPNet (general)" if "general" in emb_path.name else "FinSentenceBERT"
            
            # Check dimensions match
            if existing_sim.shape[0] == len(df):
                upper_tri = np.triu_indices(len(df), k=1)
                existing_flat = existing_sim[upper_tri]
                
                for model_name, data in results.items():
                    openai_flat = data['similarity'][upper_tri]
                    corr = np.corrcoef(existing_flat, openai_flat)[0, 1]
                    print(f"\n  {model_name} vs {model_label}:")
                    print(f"    Correlation: {corr:.4f}")
            else:
                print(f"\n  {model_label}: dimension mismatch ({existing_sim.shape[0]} vs {len(df)})")
    
    # === Within vs Across Speaker Analysis ===
    print(f"\n{'=' * 80}")
    print("WITHIN-SPEAKER VS ACROSS-SPEAKER SIMILARITY")
    print("=" * 80)
    
    for model_name, data in results.items():
        sim_matrix = data['similarity']
        within_speaker = []
        across_speaker = []
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                sim_val = sim_matrix[i, j]
                if sim_val > 0.99:  # Skip near-duplicates
                    continue
                if df.iloc[i]['speaker'] == df.iloc[j]['speaker']:
                    within_speaker.append(sim_val)
                else:
                    across_speaker.append(sim_val)
        
        print(f"\n  {model_name}:")
        print(f"    Within-speaker:  mean={np.mean(within_speaker):.4f}, std={np.std(within_speaker):.4f}")
        print(f"    Across-speaker:  mean={np.mean(across_speaker):.4f}, std={np.std(across_speaker):.4f}")
        print(f"    Difference: {np.mean(within_speaker) - np.mean(across_speaker):.4f}")
    
    # === Save Metadata ===
    meta = df[['speaker', 'date', 'year', 'word_count']].copy()
    meta.to_csv(output_dir / 'speech_metadata.csv', index=False)
    
    # Save summary
    summary = {
        'n_speeches': len(df),
        'n_speakers': df['speaker'].nunique(),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'models': {}
    }
    
    for model_name, data in results.items():
        summary['models'][model_name] = {
            'embedding_dim': data['dim'],
            'processing_time_seconds': round(data['time'], 1),
            'similarity_mean': float(data['similarity'][np.triu_indices(len(df), k=1)].mean()),
            'similarity_std': float(data['similarity'][np.triu_indices(len(df), k=1)].std()),
        }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # === Final Summary ===
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"""
  Speeches analyzed: {len(df)}
  Models tested: {', '.join(OPENAI_MODELS)}
  
  Embedding dimensions:
    - text-embedding-3-small: 1536
    - text-embedding-3-large: 3072
  
  Files saved to: {output_dir}/
    - embeddings_small.npy
    - embeddings_large.npy
    - similarity_small.npy
    - similarity_large.npy
    - speech_metadata.csv
    - analysis_summary.json
""")
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
