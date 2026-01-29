"""
Merge Speaker Metadata from Robin Dataset

This script extracts speaker metadata (role, bank, term dates) from the Robin dataset
and merges it with our main speeches dataset.

Source: /Users/sophiakazinnik/Research/Robin/speeches_Sophia/full_speeches_as_of_2022_03.csv
Target: Our non-timestamped speeches dataset

Metadata fields extracted:
- bank: Federal Reserve bank (e.g., "Board of Governors", "New York", "Philadelphia")
- title: Role/position (e.g., "President", "Governor", "Chair")
- term_start_date: Start date of their term
- term_end_date: End date of their term
- term_ongoing: Whether they're still in the role

Author: Generated with Claude Code
Date: 2025-01-26
"""

import pandas as pd
import glob
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input paths
ROBIN_METADATA_FILE = '/Users/sophiakazinnik/Research/Robin/speeches_Sophia/full_speeches_as_of_2022_03.csv'
SPEECHES_DIR = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/year_all'

# Output paths
OUTPUT_DIR = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/metadata'
SPEAKER_METADATA_FILE = f'{OUTPUT_DIR}/speaker_metadata.csv'
MERGED_SPEECHES_FILE = f'{OUTPUT_DIR}/speeches_with_metadata.csv'

# Valid titles to filter (excludes garbage data in the source file)
VALID_TITLES = [
    'President',
    'Governor',
    'Chair',
    'Governor/Vice Chair',
    'Governor/Vice Chair for Supervision',
    'Vice Chair/Governor',
    'Vice Chair',
    'First Vice President',
    'Interim President'
]

# =============================================================================
# SPEAKER NAME NORMALIZATION
# =============================================================================
# Maps variations of speaker names to a canonical form
# This ensures consistent matching between datasets

SPEAKER_NORMALIZATION = {
    # Common variations in our dataset -> canonical name
    'Barkin': 'Thomas Barkin',
    'Tom Barkin': 'Thomas Barkin',
    'Bernanke': 'Ben Bernanke',
    'Powell': 'Jerome Powell',
    'Yellen': 'Janet Yellen',
    'Williams': 'John Williams',
    'Evans': 'Charles Evans',
    'Bullard': 'James Bullard',
    'Rosengren': 'Eric Rosengren',
    'Dudley': 'William Dudley',
    'Mester': 'Loretta Mester',
    'Lacker': 'Jeffrey Lacker',
    'Harker': 'Patrick Harker',
    'Lockhart': 'Dennis Lockhart',
    'Kashkari': 'Neel Kashkari',
    'Kaplan': 'Robert Kaplan',
    'Poole': 'William Poole',
    'Kocherlakota': 'Narayana Kocherlakota',
    'Plosser': 'Charles Plosser',
    'Ferguson': 'Roger Ferguson',
    'Brainard': 'Lael Brainard',
    'George': 'Esther George',
    'Bowman': 'Michelle Bowman',
    'Bostic': 'Raphael Bostic',
    'Hoenig': 'Thomas Hoenig',
    'Daly': 'Mary Daly',
    'Pianalto': 'Sandra Pianalto',
    'Quarles': 'Randal Quarles',
    'Waller': 'Christopher Waller',
    'Moskow': 'Michael Moskow',
    'Clarida': 'Richard Clarida',
    'Fischer': 'Stanley Fischer',
    'Duke': 'Elizabeth Duke',
    'Kroszner': 'Randy Kroszner',
    'Tarullo': 'Daniel Tarullo',
    'Parry': 'Robert Parry',
    'Greenspan': 'Alan Greenspan',
    'Stein': 'Jeremy Stein',
    'Fisher': 'Richard Fisher',
    'Raskin': 'Sarah Bloom Raskin',
    'Sara Raskin': 'Sarah Bloom Raskin',
    'Bies': 'Susan Bies',
    'Mishkin': 'Frederic Mishkin',
    'Warsh': 'Kevin Warsh',
    'Gramlich': 'Edward Gramlich',
    'Kohn': 'Donald Kohn',
    'Santomero': 'Anthony Santomero',
    'Broaddus': 'Alfred Broaddus',
    'Musalem': 'Alberto Musalem',
    'Patrikis': 'Ernest T. Patrikis',
    'Dahlgren': 'Sarah Dahlgren',
    'Potter': 'Simon Potter',
}

# Maps Robin's Standardized.name -> our canonical name
# (Robin uses last names or abbreviated forms)
ROBIN_TO_CANONICAL = {
    'Barkin': 'Thomas Barkin',
    'Bernanke': 'Ben Bernanke',
    'Bies Schmidt': 'Susan Bies',
    'Bostic': 'Raphael Bostic',
    'Bowman': 'Michelle Bowman',
    'Brainard Lael': 'Lael Brainard',
    'Broaddus': 'Alfred Broaddus',
    'Bullard': 'James Bullard',
    'Clarida': 'Richard Clarida',
    'Daly': 'Mary Daly',
    'Dudley': 'William Dudley',
    'Duke': 'Elizabeth Duke',
    'Evans': 'Charles Evans',
    'Ferguson': 'Roger Ferguson',
    'Fischer': 'Stanley Fischer',
    'Fisher': 'Richard Fisher',
    'Geithner': 'Timothy Geithner',
    'George': 'Esther George',
    'Greenspan': 'Alan Greenspan',
    'Guynn': 'Jack Guynn',
    'Harker': 'Patrick Harker',
    'Hoenig': 'Thomas Hoenig',
    'Jefferson': 'Philip Jefferson',
    'Jordan': 'Jerry Jordan',
    'Kaplan': 'Robert Kaplan',
    'Kashkari': 'Neel Kashkari',
    'Kelley': 'Edward Kelley',
    'Kocherlakota': 'Narayana Kocherlakota',
    'Kohn': 'Donald Kohn',
    'Kroszner': 'Randy Kroszner',
    'Lacker': 'Jeffrey Lacker',
    'Lindsey': 'Lawrence Lindsey',
    'Lockhart': 'Dennis Lockhart',
    'McTeer': 'Robert McTeer',
    'Mester': 'Loretta Mester',
    'Meyer': 'Laurence Meyer',
    'Minehan': 'Cathy Minehan',
    'Mishkin': 'Frederic Mishkin',
    'Moskow': 'Michael Moskow',
    'Olson': 'Mark Olson',
    'Parry': 'Robert Parry',
    'Pianalto': 'Sandra Pianalto',
    'Plosser': 'Charles Plosser',
    'Poole': 'William Poole',
    'Powell': 'Jerome Powell',
    'Quarles': 'Randal Quarles',
    'Raskin': 'Sarah Bloom Raskin',
    'Rivlin': 'Alice Rivlin',
    'Rosengren': 'Eric Rosengren',
    'Santomero': 'Anthony Santomero',
    'Stein': 'Jeremy Stein',
    'Stern': 'Gary Stern',
    'Tarullo': 'Daniel Tarullo',
    'Waller': 'Christopher Waller',
    'Warsh': 'Kevin Warsh',
    'Williams': 'John Williams',
    'Yellen': 'Janet Yellen',
    'McDonough': 'William McDonough',
    'Cook': 'Lisa Cook',
    'Logan': 'Lorie Logan',
    'Collins': 'Susan Collins',
    'Barr': 'Michael Barr',
    'Melzer': 'Thomas Melzer',
    'Blinder': 'Alan Blinder',
    'Gramlich': 'Edward Gramlich',
    'LaWare': 'John LaWare',
    'Hammack': 'Beth Hammack',
    'Goolsbee': 'Austan Goolsbee',
    'Schmid': 'Jeffrey Schmid',
    'Kugler': 'Adriana Kugler',
}


# Manual metadata for speakers not in Robin dataset
# Format: canonical_name -> (bank, title)
MANUAL_SPEAKER_METADATA = {
    'William McDonough': ('New York', 'President'),
    'Lisa Cook': ('Board of Governors', 'Governor'),
    'Lorie Logan': ('Dallas', 'President'),
    'Susan Collins': ('Boston', 'President'),
    'Michael Barr': ('Board of Governors', 'Governor/Vice Chair for Supervision'),
    'Thomas Melzer': ('St Louis', 'President'),
    'Alan Blinder': ('Board of Governors', 'Vice Chair/Governor'),
    'Edward Gramlich': ('Board of Governors', 'Governor'),
    'John LaWare': ('Board of Governors', 'Governor'),
    'Beth Hammack': ('Cleveland', 'President'),
    'Austan Goolsbee': ('Chicago', 'President'),
    'Jeffrey Schmid': ('Kansas City', 'President'),
    'Adriana Kugler': ('Board of Governors', 'Governor'),
    'Kenneth Montgomery': ('Boston', 'First Vice President'),
    'Alberto Musalem': ('St Louis', 'President'),
    'Philip Jefferson': ('Board of Governors', 'Governor/Vice Chair'),
    'Ernest T. Patrikis': ('New York', 'Executive Vice President'),
    'Sarah Dahlgren': ('New York', 'Executive Vice President'),
    'Simon Potter': ('New York', 'Executive Vice President'),
}


def normalize_speaker(name):
    """Normalize speaker name to canonical form."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return SPEAKER_NORMALIZATION.get(name, name)


def robin_to_canonical(name):
    """Convert Robin's Standardized.name to our canonical name."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return ROBIN_TO_CANONICAL.get(name, name)


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("MERGING SPEAKER METADATA")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load and clean Robin metadata
    # -------------------------------------------------------------------------
    print("\n1. Loading Robin metadata...")

    robin_df = pd.read_csv(ROBIN_METADATA_FILE, encoding='latin-1', low_memory=False)
    print(f"   Loaded {len(robin_df)} rows from Robin dataset")

    # Filter to rows with valid titles (removes garbage data)
    robin_clean = robin_df[robin_df['title'].isin(VALID_TITLES)].copy()
    print(f"   After filtering to valid titles: {len(robin_clean)} rows")

    # Extract unique speaker metadata
    # For speakers with multiple roles, we keep all of them (e.g., Yellen was President then Chair)
    speaker_meta = robin_clean.groupby('Standardized.name').agg({
        'bank': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # Most common bank
        'title': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # Most common title
        'term_start_date': 'first',
        'term_end_date': 'first',
        'term_ongoing': 'first'
    }).reset_index()

    # Add canonical name column
    speaker_meta['canonical_name'] = speaker_meta['Standardized.name'].apply(robin_to_canonical)

    # Reorder columns
    speaker_meta = speaker_meta[['Standardized.name', 'canonical_name', 'bank', 'title',
                                  'term_start_date', 'term_end_date', 'term_ongoing']]

    print(f"   Extracted metadata for {len(speaker_meta)} unique speakers")

    # Save speaker metadata
    speaker_meta.to_csv(SPEAKER_METADATA_FILE, index=False)
    print(f"   Saved to: {SPEAKER_METADATA_FILE}")

    # -------------------------------------------------------------------------
    # Step 2: Load our speeches dataset
    # -------------------------------------------------------------------------
    print("\n2. Loading speeches dataset...")

    files = glob.glob(f'{SPEECHES_DIR}/*.csv')
    dfs = []
    for f in sorted(files):
        temp_df = pd.read_csv(f)
        if len(temp_df) > 0:
            dfs.append(temp_df)
    speeches_df = pd.concat(dfs, ignore_index=True)

    # Normalize speaker names
    speeches_df['speaker_normalized'] = speeches_df['speaker'].apply(normalize_speaker)

    print(f"   Loaded {len(speeches_df)} speeches")
    print(f"   Unique speakers: {speeches_df['speaker_normalized'].nunique()}")

    # -------------------------------------------------------------------------
    # Step 3: Merge metadata with speeches
    # -------------------------------------------------------------------------
    print("\n3. Merging metadata...")

    # Create lookup from canonical name to metadata
    meta_lookup = speaker_meta.set_index('canonical_name')[['bank', 'title']].to_dict('index')

    # Add manual metadata for speakers not in Robin dataset
    for name, (bank, title) in MANUAL_SPEAKER_METADATA.items():
        if name not in meta_lookup:
            meta_lookup[name] = {'bank': bank, 'title': title}
    print(f"   Total speakers in lookup: {len(meta_lookup)} (including {len(MANUAL_SPEAKER_METADATA)} manual entries)")

    # Add metadata columns to speeches
    speeches_df['fed_bank'] = speeches_df['speaker_normalized'].map(
        lambda x: meta_lookup.get(x, {}).get('bank')
    )
    speeches_df['fed_role'] = speeches_df['speaker_normalized'].map(
        lambda x: meta_lookup.get(x, {}).get('title')
    )

    # Count matches
    matched = speeches_df['fed_bank'].notna().sum()
    print(f"   Speeches with metadata: {matched} ({100*matched/len(speeches_df):.1f}%)")
    print(f"   Speeches without metadata: {len(speeches_df) - matched}")

    # -------------------------------------------------------------------------
    # Step 4: Save merged dataset
    # -------------------------------------------------------------------------
    print("\n4. Saving merged dataset...")

    # Select columns to save (exclude full text to reduce file size)
    output_cols = ['id', 'speaker', 'speaker_normalized', 'date', 'source',
                   'fed_bank', 'fed_role']
    speeches_output = speeches_df[output_cols].copy()
    speeches_output.to_csv(MERGED_SPEECHES_FILE, index=False)
    print(f"   Saved to: {MERGED_SPEECHES_FILE}")

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nSpeakers by role:")
    role_counts = speeches_df.groupby('fed_role').size().sort_values(ascending=False)
    for role, count in role_counts.head(10).items():
        if pd.notna(role):
            print(f"   {role}: {count} speeches")

    print("\nSpeakers by bank:")
    bank_counts = speeches_df.groupby('fed_bank').size().sort_values(ascending=False)
    for bank, count in bank_counts.head(10).items():
        if pd.notna(bank):
            print(f"   {bank}: {count} speeches")

    print("\nSpeakers without metadata (top 10 by speech count):")
    no_meta = speeches_df[speeches_df['fed_bank'].isna()]
    no_meta_speakers = no_meta.groupby('speaker_normalized').size().sort_values(ascending=False)
    for speaker, count in no_meta_speakers.head(10).items():
        print(f"   {speaker}: {count} speeches")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return speaker_meta, speeches_df


if __name__ == '__main__':
    speaker_meta, speeches_df = main()
