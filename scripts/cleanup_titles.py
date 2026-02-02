#!/usr/bin/env python3
"""
Clean up extracted speech titles - remove bad extractions.
"""

import pandas as pd
import re

INPUT_FILE = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output/speech_titles.csv'
OUTPUT_FILE = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output/speech_titles_cleaned.csv'


def is_bad_title(title, method):
    """Check if a title should be rejected."""
    if pd.isna(title) or not title:
        return True

    title = str(title).strip()

    # Too short (likely fragment) - be more lenient
    if len(title) < 8:
        return True

    # UI/navigation elements
    if re.search(r'Skip to main|Search Submit|Toggle Dropdown|Home\s*/\s*News\s*/\s*Press', title, re.I):
        return True

    # Starts with clear sentence patterns (not titles)
    sentence_starts = [
        r'^(I am pleased|I am happy|I am honored|I am delighted|I would like to)',
        r'^(I have been|I have had|I\'m pleased|I\'m happy)',
        r'^(We are pleased|We are happy)',
        r'^(This conference|This and other transcripts)',
        r'^(It is my pleasure|It is a pleasure|It\'s a pleasure)',
        r'^(Thank you for|Thanks for)',
        r'^(Good morning|Good afternoon|Good evening)\s+(and|everyone|ladies)',
        r'^(Today I would|Today we will)',
        r'^(Let me begin by|Let me start by)',
        r'^(In our (May|June|July|August|September|October|November|December|January|February|March|April) \d{4} meeting)',
        r'^(When I was told)',
        r'^(Looking back over the last)',
        r'^(Sometimes it helps to)',
        r'^Share\s+The\s+',  # UI element followed by text
        r'^(The decline in participation does not)',
        r'^(The three decades preceding)',
    ]
    for pattern in sentence_starts:
        if re.match(pattern, title, re.I):
            return True

    # Just fragments or clearly wrong extractions
    fragments = [
        r'^on the one hand$',
        r'^regulatory-supervisory framework',
        r'^optimal central bank law',
        r'^risk management\.?$',
        r'^economic conditions will evolve',
        r'^annus horribilis\.?$',
        r'^Introduction$',
        r'^Overview$',
        r'^Welcoming Remarks$',
        r'^Federal Reserve$',
        r'^This is your first job',
    ]
    for pattern in fragments:
        if re.match(pattern, title, re.I):
            return True

    # Starts with lowercase (not a title) - check without re.I
    if re.match(r'^[a-z]', title):
        return True

    # Titles that are ONLY dates/times (no actual title content)
    if re.match(r'^(\d{1,2}:\d{2}\s*[ap]\.?m\.?\s*(E[SD]T|C[SD]T|P[SD]T|MST)?|[A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4})$', title, re.I):
        return True

    # Titles that are just metadata (no actual title content)
    metadata_only = [
        r'^(Testimony|Remarks|Statement|Speech|Comments|Address)\s+(by|of)\s+[A-Z][a-z]+\s+[A-Z]',
        r'^(Testimony|Remarks|Statement|Speech)\s+[A-Z][a-z]+\s+[A-Z]',  # "Remarks Alan Greenspan"
    ]
    for pattern in metadata_only:
        if re.match(pattern, title, re.I):
            return True

    return False


def clean_title(title):
    """Clean up a title that passes validation."""
    if pd.isna(title):
        return None

    title = str(title).strip()

    # Remove leading times that got captured
    # e.g., "11:00 a.m. EDT Lessons from..." -> "Lessons from..."
    title = re.sub(r'^\d{1,2}:\d{2}\s*[ap]\.?m\.?\s*(E[SD]T|C[SD]T|P[SD]T|MST)?\s+', '', title, flags=re.I)

    # Remove leading full dates with parenthetical time
    # e.g., "June 2, 2013(9:50 a.m...) Title" -> "Title"
    title = re.sub(r'^[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}\s*\([^)]+\)\s*', '', title)

    # Remove "am/pm EST" patterns at start if followed by actual content
    title = re.sub(r'^[ap]\.?m\.?\s*(E[SD]T|C[SD]T|P[SD]T|MST)?\s+(?=[A-Z])', '', title, flags=re.I)

    # Remove "A.M. LOCAL TIME (...)" patterns
    title = re.sub(r'^[AP]\.?M\.?\s+LOCAL TIME\s*\([^)]+\)\s*', '', title, flags=re.I)

    # Remove trailing metadata patterns
    # "Title by Name Chairman Board of Governors of the" -> "Title"
    title = re.sub(r'\s+(by\s+)?[A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z][a-z]+\s+(Chairman|Vice Chairman|Governor|President|Member)\s+Board of Governors.*$', '', title, flags=re.I)
    title = re.sub(r'\s+Board of Governors of the Federal Reserve System.*$', '', title, flags=re.I)
    title = re.sub(r'\s+Chairman Board of Governors of the.*$', '', title, flags=re.I)
    title = re.sub(r'\s+Board of Governors of the$', '', title, flags=re.I)

    # Remove trailing "Remarks by Name" or "Statement by Name"
    title = re.sub(r'\s+(Remarks|Statement|Testimony|Speech)\s+(by\s+)?[A-Z][a-z]+\s+[A-Z].*$', '', title, flags=re.I)

    # Remove trailing incomplete phrases (repeat until no more matches)
    for _ in range(3):  # Multiple passes to catch "to the" -> "to" -> ""
        old_title = title
        title = re.sub(r'\s+(the|a|an|of|in|on|at|to|for|by|with|and|or|its|our|my|your|this|that)$', '', title, flags=re.I)
        if title == old_title:
            break

    # Remove trailing location patterns like "New York, New York" or "Washington, D.C."
    title = re.sub(r',?\s+[A-Z][a-z]+,\s+[A-Z][a-z]+$', '', title)
    title = re.sub(r',?\s+Washington,?\s+D\.?C\.?$', '', title, flags=re.I)

    # Clean up extra whitespace
    title = ' '.join(title.split())

    # Final length check after cleaning
    if len(title) < 8:
        return None

    return title


def main():
    print("Loading extracted titles...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")

    original_with_title = df['extracted_title'].notna().sum()
    print(f"Original titles: {original_with_title}")

    # Track changes
    rejected_count = 0
    cleaned_count = 0

    cleaned_titles = []
    cleaned_methods = []

    for idx, row in df.iterrows():
        title = row['extracted_title']
        method = row['extraction_method']

        if pd.isna(title):
            cleaned_titles.append(None)
            cleaned_methods.append(None)
            continue

        # Check if should be rejected
        if is_bad_title(title, method):
            cleaned_titles.append(None)
            cleaned_methods.append(None)
            rejected_count += 1
            continue

        # Clean the title
        cleaned = clean_title(title)
        if cleaned and cleaned != title:
            cleaned_count += 1

        if cleaned:
            cleaned_titles.append(cleaned)
            cleaned_methods.append(method)
        else:
            cleaned_titles.append(None)
            cleaned_methods.append(None)
            rejected_count += 1

    df['extracted_title'] = cleaned_titles
    df['extraction_method'] = cleaned_methods

    final_with_title = df['extracted_title'].notna().sum()

    print(f"\n{'='*60}")
    print("CLEANUP RESULTS")
    print(f"{'='*60}")
    print(f"Original titles: {original_with_title}")
    print(f"Rejected:        {rejected_count}")
    print(f"Cleaned:         {cleaned_count}")
    print(f"Final titles:    {final_with_title} ({100*final_with_title/len(df):.1f}%)")
    print(f"{'='*60}")

    # Show method breakdown after cleanup
    print("\nMethod breakdown after cleanup:")
    method_counts = df['extraction_method'].value_counts(dropna=False)
    for method, count in method_counts.items():
        print(f"  {method or 'None'}: {count}")

    # Save cleaned results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nCleaned results saved to: {OUTPUT_FILE}")

    # Show some samples of rejected titles
    print("\n=== SAMPLE REJECTED TITLES ===")
    original_df = pd.read_csv(INPUT_FILE)
    for idx in range(min(20, len(original_df))):
        orig = original_df.iloc[idx]['extracted_title']
        clean = df.iloc[idx]['extracted_title']
        if pd.notna(orig) and pd.isna(clean):
            print(f"  REJECTED: {orig[:70]}...")

    return df


if __name__ == '__main__':
    main()
