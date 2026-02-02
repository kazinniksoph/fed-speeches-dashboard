#!/usr/bin/env python3
"""
Extract speech titles using hybrid approach:
1. Regex patterns for known formats
2. LLM fallback for unclear cases
"""

import pandas as pd
import glob
import re
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

DATA_DIR = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/year_all'
OUTPUT_DIR = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output'

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available - LLM fallback disabled")


def extract_title_regex(text):
    """
    Try to extract title using regex patterns.
    Returns (title, method) or (None, None) if no match.
    """
    if pd.isna(text) or not text or len(str(text)) < 50:
        return None, None
    text = str(text)

    # Get first 2000 chars for analysis
    header = text[:2000]

    # Pattern 1: "For release on delivery [time] [date] [TITLE] Remarks by [Name]"
    # The title is between the date and "Remarks by"
    match = re.search(
        r'For release on delivery.*?(\d{4})\s+(.+?)\s+(?:Remarks|Statement|Testimony)\s+(?:by|of|By|Of)\s+',
        header,
        re.I
    )
    if match:
        title = match.group(2).strip()
        # Clean up: remove stray punctuation
        title = re.sub(r'^[\s\'"]+|[\s\'"]+$', '', title)
        if 10 < len(title) < 200:
            return title, 'release_header'

    # Pattern 2: "Embargoed for release [time] [date] [TITLE] Remarks by"
    match = re.search(
        r'Embargoed.*?(\d{4})\s+(.+?)\s+(?:Remarks|Statement|Testimony)\s+(?:by|of|By|Of)\s+',
        header,
        re.I
    )
    if match:
        title = match.group(2).strip()
        title = re.sub(r'^[\s\'"]+|[\s\'"]+$', '', title)
        if 10 < len(title) < 200:
            return title, 'embargoed_header'

    # Pattern 3: "[TITLE] Presented by [Name], [Role]"
    match = re.search(
        r'^(.{15,150}?)\s+Presented by\s+[A-Z]',
        header,
        re.I
    )
    if match:
        title = match.group(1).strip()
        if not re.match(r'^(Thank|Today|Good|Let|I\s)', title):
            return title, 'presented_by'

    # Pattern 4: Look for quoted title (but validate it looks like a title)
    quote_match = re.search(r'["\u201c]([^"\u201d]{15,150})["\u201d]', header[:500])
    if quote_match:
        title = quote_match.group(1)
        # Filter out obvious non-titles
        if not re.search(r'^(lunch|dinner|a\s|the\s+\w+\s+is)', title, re.I):
            return title, 'quoted'

    # Pattern 5: Title pattern with common title-ending words followed by speaker info
    # e.g., "The Economic Outlook Remarks by..."
    match = re.search(
        r'\b([A-Z][^.!?]{10,120}(?:Outlook|Policy|Economy|Markets|Crisis|Reform|Regulation|Banking|System|Growth|Inflation|Recovery|Challenges|Future|Perspective|Overview))\s+(?:Remarks|Statement|Speech|Address|Comments)',
        header,
        re.I
    )
    if match:
        title = match.group(1).strip()
        return title, 'title_keyword'

    # Pattern 6: "Remarks at/before/to [Event]" - extract full phrase
    match = re.search(
        r'((?:Remarks|Address|Speech|Comments|Testimony)\s+(?:at|before|to|on|regarding)\s+[^.]{10,150}?)(?:\s+by\s+|\s+[A-Z][a-z]+\s+\d)',
        header,
        re.I
    )
    if match:
        title = match.group(1).strip()
        return title, 'remarks_at'

    # Pattern 7: Modern format - short title followed by name and date
    # "Learning from Our Community Tom Barkin Jan. 4, 2023"
    match = re.search(
        r'^([A-Z][^.!?]{10,80}?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        header
    )
    if match:
        title = match.group(1).strip()
        # Make sure it doesn't look like content or navigation
        if not re.match(r'^(Thank|Today|Good|Let|I\s|The\s+Fed|Home|News|Press|SPEECH)', title):
            return title, 'modern_header'

    # Pattern 8: Title followed by organization/location (no "by")
    # "FOMC Transparency Ozark Chapter of the Society..."
    match = re.search(
        r'^([A-Z][A-Za-z\s\-:]{10,80}?)\s+(?:Ozark|Chapter|Society|Association|Club|Conference|Forum|Summit|Institute|University|College|Bank\s+of|Federal\s+Reserve)',
        header
    )
    if match:
        title = match.group(1).strip()
        if not re.match(r'^(Thank|Today|Good|Let|I\s|Before|At|To)', title):
            return title, 'title_before_org'

    # Pattern 9: Title followed by location pattern "City, State"
    match = re.search(
        r'^([A-Z][A-Za-z\s\-:\'\"]{10,80}?)\s+[A-Z][a-z]+,\s+(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming|D\.C\.|DC)',
        header
    )
    if match:
        title = match.group(1).strip()
        if not re.match(r'^(Thank|Today|Good|Let|I\s)', title):
            return title, 'title_before_location'

    # Pattern 10: "Remarks by [Name]... [Date] [TITLE] [Speech body]"
    # The title is between the date and speech body (Good/Thank/It is/I am)
    match = re.search(
        r'^Remarks by\s+.+?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s+([A-Z][A-Za-z\s,:\-\'\"\(\)&;]+?)\s+(?:Good\s+(?:morning|afternoon|evening)|Thank\s+you|It\s+is\s+(?:a\s+pleasure|my\s+pleasure|an\s+honor)|I\s+am\s+(?:pleased|delighted|honored)|Ladies\s+and\s+gentlemen|It\s+is\s+always)',
        header
    )
    if match:
        title = match.group(1).strip()
        # Clean up trailing punctuation and whitespace
        title = re.sub(r'[\s\*\.,;:]+$', '', title)
        # Make sure it's not just intro phrases
        if 10 < len(title) < 150 and not re.match(r'^(Introduction|Acknowledgments?|Overview)\s*$', title, re.I):
            return title, 'remarks_after_date'

    return None, None


def extract_title_heuristic(text):
    """
    Heuristic fallback for continuous text without clear markers.
    """
    if pd.isna(text) or not text or len(str(text)) < 50:
        return None, None
    text = str(text)

    header = text[:1500]

    # Skip if starts with obvious content patterns
    content_starts = [
        r'^Thank you',
        r'^Good (morning|afternoon|evening)',
        r'^It is (my|a) (pleasure|honor|great)',
        r'^I am (pleased|honored|delighted)',
        r'^Today I',
        r'^Let me begin',
        r'^Highlights:',
    ]
    for pattern in content_starts:
        if re.match(pattern, header, re.I):
            return None, None

    # Try to find a title-like phrase at the start
    # Look for capitalized phrase ending before a speaker indicator
    match = re.search(
        r'^([A-Z][A-Za-z\s,:\-\'"]{15,100}?)(?:\s+(?:by|By|Presented|[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:President|Chairman|Governor)))',
        header
    )
    if match:
        title = match.group(1).strip()
        # Clean trailing punctuation
        title = re.sub(r'[\s,:\-]+$', '', title)
        if 15 < len(title) < 100:
            return title, 'heuristic_start'

    # Look for capitalized phrase followed by organization name
    match = re.search(
        r'^([A-Z][A-Za-z\s,:\-\'"]{15,100}?)(?:\s+(?:Federal Reserve|Fed|Board of Governors|FOMC))',
        header
    )
    if match:
        title = match.group(1).strip()
        title = re.sub(r'[\s,:\-]+$', '', title)
        if 15 < len(title) < 100:
            return title, 'heuristic_org'

    return None, None


def extract_title_llm(text, client):
    """
    Use LLM to extract title from speech text.
    """
    if not text or len(text) < 50:
        return None, None

    # Only send first 800 chars to save tokens
    header = text[:800]

    prompt = f"""Extract the title of this Federal Reserve speech. The title may appear at the beginning, possibly after release time/date info.

If there is a clear title, return ONLY the title text (no quotes, no explanation).
If there is no title (speech starts directly with content), return exactly: NO_TITLE

Text:
{header}

Title:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        result = response.choices[0].message.content.strip()

        if result == "NO_TITLE" or len(result) < 5:
            return None, None

        # Clean up
        result = re.sub(r'^["\']+|["\']+$', '', result)
        result = re.sub(r'^Title:\s*', '', result, flags=re.I)

        if 5 < len(result) < 200:
            return result, 'llm'
        return None, None
    except Exception as e:
        print(f"LLM error: {e}")
        return None, None


def process_single_speech(row, client=None):
    """Process a single speech - for parallel execution."""
    text = row.get('text', '')
    if pd.isna(text):
        text = ''
    text = str(text)

    # Try regex first
    title, method = extract_title_regex(text)

    # Fallback to heuristic
    if not title:
        title, method = extract_title_heuristic(text)

    # Fallback to LLM
    if not title and client:
        title, method = extract_title_llm(text, client)

    return {
        'id': row.get('id'),
        'speaker': row.get('speaker'),
        'date': row.get('date'),
        'extracted_title': title,
        'extraction_method': method,
        'text_preview': text[:200] if text else ''
    }


def process_speeches(sample_size=None, use_llm=False):
    """Load and process all speeches."""

    files = sorted(glob.glob(f'{DATA_DIR}/all_speeches_*.csv'))
    print(f"Found {len(files)} year files", flush=True)

    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total speeches: {len(df)}", flush=True)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Sampling {len(df)} speeches for testing", flush=True)

    # Initialize OpenAI client if using LLM
    client = None
    if use_llm and OPENAI_AVAILABLE:
        client = OpenAI()
        print("LLM fallback enabled (parallel processing)", flush=True)

    # First pass: regex and heuristic only
    print("Pass 1: Regex and heuristic extraction...", flush=True)
    results = []
    needs_llm = []

    for idx, row in df.iterrows():
        text = row.get('text', '')
        if pd.isna(text):
            text = ''
        text = str(text)

        # Try regex first
        title, method = extract_title_regex(text)

        # Fallback to heuristic
        if not title:
            title, method = extract_title_heuristic(text)

        if title:
            results.append({
                'id': row.get('id'),
                'speaker': row.get('speaker'),
                'date': row.get('date'),
                'extracted_title': title,
                'extraction_method': method,
                'text_preview': text[:200] if text else ''
            })
        else:
            needs_llm.append(row.to_dict())

    print(f"  Regex/heuristic extracted: {len(results)}", flush=True)
    print(f"  Needs LLM: {len(needs_llm)}", flush=True)

    # Second pass: LLM for remaining (parallel)
    if client and needs_llm:
        print(f"Pass 2: LLM extraction for {len(needs_llm)} speeches (10 parallel workers)...", flush=True)

        def process_with_llm(row_dict):
            text = str(row_dict.get('text', ''))
            title, method = extract_title_llm(text, client)
            return {
                'id': row_dict.get('id'),
                'speaker': row_dict.get('speaker'),
                'date': row_dict.get('date'),
                'extracted_title': title,
                'extraction_method': method,
                'text_preview': text[:200] if text else ''
            }

        llm_results = []
        completed = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_with_llm, row): row for row in needs_llm}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    llm_results.append(result)
                    completed += 1
                    if completed % 100 == 0:
                        print(f"  LLM processed: {completed}/{len(needs_llm)}", flush=True)
                except Exception as e:
                    row = futures[future]
                    print(f"  Error processing {row.get('id')}: {e}", flush=True)
                    llm_results.append({
                        'id': row.get('id'),
                        'speaker': row.get('speaker'),
                        'date': row.get('date'),
                        'extracted_title': None,
                        'extraction_method': None,
                        'text_preview': str(row.get('text', ''))[:200]
                    })

        results.extend(llm_results)
    elif needs_llm:
        # No LLM - just add empty results
        for row_dict in needs_llm:
            results.append({
                'id': row_dict.get('id'),
                'speaker': row_dict.get('speaker'),
                'date': row_dict.get('date'),
                'extracted_title': None,
                'extraction_method': None,
                'text_preview': str(row_dict.get('text', ''))[:200]
            })

    results_df = pd.DataFrame(results)

    # Count by method
    method_counts = results_df['extraction_method'].value_counts(dropna=False)
    regex_count = sum(method_counts.get(m, 0) for m in ['release_header', 'embargoed_header', 'presented_by', 'quoted', 'modern_header', 'title_keyword', 'remarks_at', 'title_before_org', 'title_before_location'])
    heuristic_count = sum(method_counts.get(m, 0) for m in ['heuristic_start', 'heuristic_org'])
    llm_count = method_counts.get('llm', 0)
    no_title_count = method_counts.get(None, 0) if None in method_counts.index else results_df['extraction_method'].isna().sum()

    print(f"\n{'='*60}", flush=True)
    print("EXTRACTION RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Regex patterns:  {regex_count:,} ({100*regex_count/len(df):.1f}%)", flush=True)
    print(f"Heuristic:       {heuristic_count:,} ({100*heuristic_count/len(df):.1f}%)", flush=True)
    if use_llm:
        print(f"LLM fallback:    {llm_count:,} ({100*llm_count/len(df):.1f}%)", flush=True)
    print(f"No title found:  {no_title_count:,} ({100*no_title_count/len(df):.1f}%)", flush=True)
    print(f"{'='*60}", flush=True)

    # Show method breakdown
    print("\nMethod breakdown:", flush=True)
    for method, count in method_counts.items():
        print(f"  {method or 'None'}: {count}", flush=True)

    return results_df


def show_samples(results_df, n=5):
    """Show sample extractions for each method."""

    print(f"\n{'='*60}")
    print("SAMPLE EXTRACTIONS")
    print(f"{'='*60}")

    methods = results_df['extraction_method'].dropna().unique()

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        samples = results_df[results_df['extraction_method'] == method].head(n)
        for _, row in samples.iterrows():
            print(f"Speaker: {row['speaker']}")
            print(f"Title: {row['extracted_title']}")
            print(f"Preview: {row['text_preview'][:100]}...")
            print()

    # Show some failures
    print(f"\n--- NO TITLE FOUND ---")
    failures = results_df[results_df['extraction_method'].isna()].head(n)
    for _, row in failures.iterrows():
        print(f"Speaker: {row['speaker']}")
        print(f"Preview: {row['text_preview'][:150]}...")
        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract speech titles')
    parser.add_argument('--sample', type=int, default=500, help='Sample size (0 for all)')
    parser.add_argument('--llm', action='store_true', help='Use LLM fallback')
    parser.add_argument('--full', action='store_true', help='Run on all speeches')
    args = parser.parse_args()

    sample_size = None if args.full else args.sample

    if args.full:
        print("Running on ALL speeches...")
    else:
        print(f"Running on sample of {sample_size} speeches...")

    results = process_speeches(sample_size=sample_size, use_llm=args.llm)
    show_samples(results, n=3)

    # Save results
    if args.full:
        output_file = f'{OUTPUT_DIR}/speech_titles.csv'
    else:
        output_file = f'{OUTPUT_DIR}/title_extraction_sample.csv'

    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
