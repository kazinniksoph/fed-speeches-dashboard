"""
Generate an interactive HTML dashboard for Fed speeches metadata.
Creates a self-contained HTML file that can be shared and opened in any browser.
"""

import pandas as pd
import json
from datetime import datetime
import os

# Using Plotly CDN for smaller file size (requires internet connection)

# === Load and prepare data ===
import glob

# Speaker name normalization mapping
SPEAKER_NORMALIZATION = {
    # Barkin variations
    'Barkin': 'Thomas Barkin',
    'Tom Barkin': 'Thomas Barkin',
    # Bernanke variations
    'Bernanke': 'Ben Bernanke',
    # Powell variations
    'Powell': 'Jerome Powell',
    # Yellen variations
    'Yellen': 'Janet Yellen',
    # Williams variations
    'Williams': 'John Williams',
    # Evans variations
    'Evans': 'Charles Evans',
    # Bullard variations
    'Bullard': 'James Bullard',
    # Rosengren variations
    'Rosengren': 'Eric Rosengren',
    # Dudley variations
    'Dudley': 'William Dudley',
    # Mester variations
    'Mester': 'Loretta Mester',
    # Lacker variations
    'Lacker': 'Jeffrey Lacker',
    # Harker variations
    'Harker': 'Patrick Harker',
    # Lockhart variations
    'Lockhart': 'Dennis Lockhart',
    # Kashkari variations
    'Kashkari': 'Neel Kashkari',
    # Kaplan variations
    'Kaplan': 'Robert Kaplan',
    # Poole variations
    'Poole': 'William Poole',
    # Kocherlakota variations
    'Kocherlakota': 'Narayana Kocherlakota',
    # Plosser variations
    'Plosser': 'Charles Plosser',
    # Ferguson variations
    'Ferguson': 'Roger Ferguson',
    # Brainard variations
    'Brainard': 'Lael Brainard',
    # George variations
    'George': 'Esther George',
    # Bowman variations
    'Bowman': 'Michelle Bowman',
    # Bostic variations
    'Bostic': 'Raphael Bostic',
    # Hoenig variations
    'Hoenig': 'Thomas Hoenig',
    # Daly variations
    'Daly': 'Mary Daly',
    # Pianalto variations
    'Pianalto': 'Sandra Pianalto',
    # Quarles variations
    'Quarles': 'Randal Quarles',
    # Waller variations
    'Waller': 'Christopher Waller',
    # Moskow variations
    'Moskow': 'Michael Moskow',
    # Clarida variations
    'Clarida': 'Richard Clarida',
    # Fischer variations
    'Fischer': 'Stanley Fischer',
    # Duke variations
    'Duke': 'Elizabeth Duke',
    # Kroszner variations
    'Kroszner': 'Randy Kroszner',
    # Tarullo variations
    'Tarullo': 'Daniel Tarullo',
    # Parry variations
    'Parry': 'Robert Parry',
    # Greenspan variations
    'Greenspan': 'Alan Greenspan',
    # Stein variations
    'Stein': 'Jeremy Stein',
    # Fisher variations
    'Fisher': 'Richard Fisher',
    # Raskin variations
    'Raskin': 'Sarah Bloom Raskin',
    'Sara Raskin': 'Sarah Bloom Raskin',
    # Bies variations
    'Bies': 'Susan Bies',
    # Mishkin variations
    'Mishkin': 'Frederic Mishkin',
    # Warsh variations
    'Warsh': 'Kevin Warsh',
    # Gramlich variations
    'Gramlich': 'Edward Gramlich',
    # Kohn variations
    'Kohn': 'Donald Kohn',
    # Santomero variations
    'Santomero': 'Anthony Santomero',
    # Broaddus variations
    'Broaddus': 'Alfred Broaddus',
    # Musalem variations
    'Musalem': 'Alberto Musalem',
    # Patrikis variations
    'Patrikis': 'Ernest T. Patrikis',
    # Dahlgren variations
    'Dahlgren': 'Sarah Dahlgren',
    # Potter variations
    'Potter': 'Simon Potter',
}

def normalize_speaker(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    return SPEAKER_NORMALIZATION.get(name, name)

# Load all non-timestamped speeches (7,501 speeches - the complete dataset)
nts_files = glob.glob('/Users/sophiakazinnik/Research/central_bank_speeches_communication/non time stamped speeches/*.csv')
nts_dfs = []
for f in sorted(nts_files):
    temp_df = pd.read_csv(f)
    if len(temp_df) > 0:
        nts_dfs.append(temp_df)
df = pd.concat(nts_dfs, ignore_index=True)

# Normalize speaker names
df['speaker'] = df['speaker'].apply(normalize_speaker)
df['match_key'] = df['speaker'].astype(str) + '_' + df['date'].astype(str)

# Load LDA topic assignments and join (using full corpus results)
lda_df = pd.read_csv('/Users/sophiakazinnik/Research/central_bank_speeches_communication/lda_improved_results/speech_topic_assignments.csv')
lda_df['speaker'] = lda_df['speaker'].apply(normalize_speaker)
lda_df['match_key'] = lda_df['speaker'].astype(str) + '_' + lda_df['date'].astype(str)

# Create LDA lookup
lda_cols = ['dominant_topic', 'dominant_topic_prob']
lda_lookup = lda_df.set_index('match_key')[lda_cols].to_dict('index')

# Join LDA data
for col in lda_cols:
    df[col] = df['match_key'].map(lambda x: lda_lookup.get(x, {}).get(col))

# Add flag for speeches with LDA topics
df['has_lda_topic'] = df['dominant_topic'].notna()
df = df.drop(columns=['match_key'])

# Load speaker metadata (bank, role)
speaker_meta_file = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/metadata/speeches_with_metadata.csv'
if os.path.exists(speaker_meta_file):
    meta_df = pd.read_csv(speaker_meta_file)
    meta_df['speaker_normalized'] = meta_df['speaker_normalized'].apply(normalize_speaker)
    meta_lookup = meta_df.groupby('speaker_normalized').first()[['fed_bank', 'fed_role']].to_dict('index')
    df['fed_bank'] = df['speaker'].map(lambda x: meta_lookup.get(x, {}).get('fed_bank'))
    df['fed_role'] = df['speaker'].map(lambda x: meta_lookup.get(x, {}).get('fed_role'))
    print(f"Loaded speaker metadata: {df['fed_bank'].notna().sum()} speeches with bank/role info")
else:
    df['fed_bank'] = None
    df['fed_role'] = None
    print("Speaker metadata file not found - run merge_speaker_metadata.py first")

# Get unique banks for filter dropdown
unique_banks = sorted([b for b in df['fed_bank'].dropna().unique() if b and str(b) != 'nan'])
bank_options_html = '<option value="all">All Banks</option>\n' + '\n'.join([f'                    <option value="{b}">{b}</option>' for b in unique_banks])

print(f"Loaded {len(df)} total speeches")
print(f"Speeches with LDA topics: {df['has_lda_topic'].sum()}")
print(f"Speeches without LDA topics: {(~df['has_lda_topic']).sum()}")
print(f"Unique banks: {len(unique_banks)}")

# Parse dates (format: 09mar1995)
def parse_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        return datetime.strptime(str(date_str), '%d%b%Y')
    except:
        return None

df['parsed_date'] = df['date'].apply(parse_date)
df['year'] = df['parsed_date'].dt.year
df['month'] = df['parsed_date'].dt.month
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

# === Summary Statistics ===
total_speeches = len(df)
unique_speakers = df['speaker'].nunique()
year_range = f"{int(df['year'].min())}-{int(df['year'].max())}"
avg_word_count = int(df['word_count'].mean())
total_words = int(df['word_count'].sum())

# === Key Insights for Dashboard ===
# Most active speaker
top_speaker = df['speaker'].value_counts().head(1)
top_speaker_name = top_speaker.index[0]
top_speaker_count = int(top_speaker.values[0])

# Peak year
peak_year_data = df.groupby('year').size()
peak_year = int(peak_year_data.idxmax())
peak_year_count = int(peak_year_data.max())

# Most common bank (excluding Board of Governors which dominates)
bank_counts = df['fed_bank'].value_counts()
if len(bank_counts) > 0:
    most_common_bank = bank_counts.index[0]
    most_common_bank_count = int(bank_counts.values[0])
else:
    most_common_bank = "N/A"
    most_common_bank_count = 0

# Longest average speech speaker (min 10 speeches)
speaker_avg_words = df.groupby('speaker').agg({'word_count': 'mean', 'speaker': 'count'})
speaker_avg_words.columns = ['avg_words', 'count']
speaker_avg_words = speaker_avg_words[speaker_avg_words['count'] >= 10]
if len(speaker_avg_words) > 0:
    verbose_speaker = speaker_avg_words['avg_words'].idxmax()
    verbose_speaker_avg = int(speaker_avg_words.loc[verbose_speaker, 'avg_words'])
else:
    verbose_speaker = "N/A"
    verbose_speaker_avg = 0

# === Prepare data for charts ===

# Chart 1: Speeches Over Time
yearly_counts = df.groupby('year').size().reset_index(name='count')
timeline_years = [int(x) for x in yearly_counts['year'].tolist()]
timeline_counts = [int(x) for x in yearly_counts['count'].tolist()]

# Chart 2: Top Speakers
top_speakers_df = df['speaker'].value_counts().head(15).reset_index()
top_speakers_df.columns = ['speaker', 'count']
speaker_names = top_speakers_df['speaker'].tolist()
speaker_counts = [int(x) for x in top_speakers_df['count'].tolist()]

# Chart 3: Heatmap by Month/Year
monthly_counts = df.groupby(['year', 'month']).size().reset_index(name='count')
pivot_data = monthly_counts.pivot(index='month', columns='year', values='count').fillna(0)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
heatmap_z = [[float(val) for val in row] for row in pivot_data.values.tolist()]
heatmap_x = [int(x) for x in pivot_data.columns.tolist()]

# Chart 4: Average Speech Length by Speaker (Top 15)
speaker_length = df.groupby('speaker').agg({
    'word_count': 'mean',
    'speaker': 'count'
}).rename(columns={'speaker': 'speech_count', 'word_count': 'avg_words'}).reset_index()
speaker_length = speaker_length[speaker_length['speech_count'] >= 10].nlargest(15, 'avg_words')
avg_length_names = speaker_length['speaker'].tolist()
avg_length_values = [int(x) for x in speaker_length['avg_words'].tolist()]

# Chart 5: Speech Length Distribution
word_counts = df['word_count'].dropna().tolist()

# Chart: Role Distribution
role_counts = df[df['fed_role'].notna()].groupby('fed_role').size().sort_values(ascending=True)
role_names = role_counts.index.tolist()
role_values = [int(x) for x in role_counts.values.tolist()]

# Chart: Role Over Time (stacked area)
role_by_year = df[df['fed_role'].notna()].groupby(['year', 'fed_role']).size().unstack(fill_value=0)
# Convert to percentages
role_by_year_pct = role_by_year.div(role_by_year.sum(axis=1), axis=0) * 100
role_years = [int(y) for y in role_by_year_pct.index.tolist()]
role_traces_data = {role: [float(v) for v in role_by_year_pct[role].values] for role in role_by_year_pct.columns}

# Get unique roles for filter
unique_roles = sorted([r for r in df['fed_role'].dropna().unique() if r and str(r) != 'nan'])

# Chart: Time of Day Distribution
import re

def parse_hour(time_str):
    if pd.isna(time_str):
        return None
    time_str = str(time_str).strip().upper()
    match = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM)?', time_str)
    if match:
        hour = int(match.group(1))
        ampm = match.group(3)
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0
        if 0 <= hour <= 23:  # Filter out bad data
            return hour
    return None

df['hour'] = df['time'].apply(parse_hour)
hour_counts = df['hour'].dropna().value_counts().sort_index()
# Create full 24-hour range with zeros for missing hours
time_hours = list(range(24))
time_counts = [int(hour_counts.get(h, 0)) for h in time_hours]
time_labels = [f"{h:02d}:00" for h in time_hours]
speeches_with_time = int(df['hour'].notna().sum())

# === Speech Explorer Data ===
# Prepare speech data for interactive explorer (with snippets)
def get_snippet(text, length=300):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    if len(text) <= length:
        return text
    return text[:length].rsplit(' ', 1)[0] + "..."

def get_preview(text, length=2000):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    if len(text) <= length:
        return text
    return text[:length].rsplit(' ', 1)[0] + "..."

speech_explorer_data = []
for idx, row in df.iterrows():
    if pd.notna(row['year']) and pd.notna(row['speaker']):
        # Get time if available
        time_str = ''
        hour_val = -1
        if 'time' in row.index and pd.notna(row['time']):
            time_str = str(row['time']).strip()
        if 'hour' in row.index and pd.notna(row['hour']):
            hour_val = int(row['hour'])

        speech_explorer_data.append({
            'id': int(idx),
            'speaker': str(row['speaker']) if pd.notna(row['speaker']) else '',
            'date': row['parsed_date'].strftime('%Y-%m-%d') if pd.notna(row['parsed_date']) else '',
            'time': time_str,
            'hour': hour_val,
            'year': int(row['year']),
            'words': int(row['word_count']) if pd.notna(row['word_count']) else 0,
            'snippet': get_snippet(row['text']),
            'preview': get_preview(row['text']),
            'dominant_topic': int(row['dominant_topic']) if pd.notna(row['dominant_topic']) else -1,
            'has_lda': bool(row['has_lda_topic']),
            'bank': str(row['fed_bank']) if pd.notna(row['fed_bank']) else '',
            'role': str(row['fed_role']) if pd.notna(row['fed_role']) else ''
        })

# === LDA Topic Analysis Data ===
import os
lda_output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/lda_improved_results'

# Load LDA results if available
lda_available = os.path.exists(f'{lda_output_dir}/topic_definitions_k8.csv')

if lda_available:
    # Load topic definitions (8 topics - improved LDA with coherence optimization)
    topic_defs = pd.read_csv(f'{lda_output_dir}/topic_definitions_k8.csv')

    # Create topic labels from top words
    topic_labels = []
    topic_short_labels = []
    topic_top_words = []

    # Define readable labels for the 8 improved topics
    readable_labels = {
        0: 'Banking Regulation',
        1: 'Monetary Policy Theory',
        2: 'Risk Management',
        3: 'Inflation & Labor Market',
        4: 'Payments & Treasury',
        5: 'Community & Education',
        6: 'International & Trade',
        7: 'Credit & Housing'
    }

    for _, row in topic_defs.iterrows():
        topic_id = row['topic_id']
        top_words = row['top_words'].split(', ')[:10]
        label = readable_labels.get(topic_id, f"Topic {topic_id}")
        topic_labels.append(f"{label}: {', '.join(top_words[:3])}")
        topic_short_labels.append(label)
        topic_top_words.append(top_words)

    # Load speech-level topic assignments and aggregate by year
    speech_topics = pd.read_csv(f'{lda_output_dir}/speech_topic_assignments.csv')
    topic_cols = [f'topic_{i}_prob' for i in range(8)]

    # Aggregate by year - average topic probabilities
    yearly_topic_dist = speech_topics.groupby('year')[topic_cols].mean()
    topic_years = [int(x) for x in yearly_topic_dist.index.tolist()]

    # Prepare data for stacked area chart
    topic_traces_data = {}
    for i, label in enumerate(topic_labels):
        col = f'topic_{i}_prob'
        topic_traces_data[label] = [float(x) * 100 for x in yearly_topic_dist[col].tolist()]

    # Calculate speaker aggregation for speaker-topic heatmap
    speaker_agg_df = speech_topics.groupby('speaker').size().reset_index(name='total_speeches')

    # Get top 12 speakers by speech count
    top_speakers_for_topics = speaker_agg_df.nlargest(12, 'total_speeches')['speaker'].tolist()

    # Calculate average topic distribution for each top speaker
    speaker_topic_data = []
    for speaker in top_speakers_for_topics:
        speaker_speeches = speech_topics[speech_topics['speaker'] == speaker]
        avg_topics = speaker_speeches[topic_cols].mean()
        speaker_topic_data.append({
            'speaker': speaker,
            **{f'topic_{i}': avg_topics[f'topic_{i}_prob'] * 100 for i in range(8)}
        })

    speaker_topic_df = pd.DataFrame(speaker_topic_data)
    topic_cols_short = [f'topic_{i}' for i in range(8)]
    speaker_topic_matrix = speaker_topic_df[topic_cols_short].values.tolist()
    speaker_topic_names = speaker_topic_df['speaker'].tolist()

    # Add topic labels to speech explorer data (dominant_topic already set during data load)
    for speech in speech_explorer_data:
        topic_id = speech.get('dominant_topic', -1)
        if topic_id >= 0 and topic_id < len(topic_short_labels):
            speech['topic_id'] = topic_id
            speech['topic'] = topic_short_labels[topic_id]
        else:
            speech['topic_id'] = -1
            speech['topic'] = 'Unknown'

# === Textual Metrics Data ===
analysis_output_dir = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/analysis_output'
textual_metrics_file = f'{analysis_output_dir}/textual_metrics_by_year.csv'
textual_metrics_available = os.path.exists(textual_metrics_file)

if textual_metrics_available:
    textual_metrics = pd.read_csv(textual_metrics_file, index_col=0)
    metrics_years = [int(x) for x in textual_metrics.index.tolist()]
    # Filter out outliers: fog_index > 21 is likely an error (normal range 15-20)
    readability_scores = [float(x) if (not pd.isna(x) and 15 < x < 21) else None for x in textual_metrics['fog_index'].tolist()]
    uncertainty_scores = [float(x) if not pd.isna(x) else None for x in textual_metrics['uncertainty_index'].tolist()]

# === FinBERT-FOMC Hawkish/Dovish Data ===
hawkish_dovish_year_file = f'{analysis_output_dir}/hawkish_dovish_by_year.csv'
hawkish_dovish_available = os.path.exists(hawkish_dovish_year_file)

if hawkish_dovish_available:
    hd_yearly = pd.read_csv(hawkish_dovish_year_file)
    hd_years = [int(x) for x in hd_yearly['year'].tolist()]
    sentiment_scores = [float(x) for x in hd_yearly['sentiment_score'].tolist()]
    pct_hawkish = [float(x) for x in hd_yearly['pct_hawkish'].tolist()]
    pct_dovish = [float(x) for x in hd_yearly['pct_dovish'].tolist()]

    # Load speaker-level hawkish/dovish data
    hawkish_dovish_speaker_file = f'{analysis_output_dir}/hawkish_dovish_by_speaker.csv'
    if os.path.exists(hawkish_dovish_speaker_file):
        hd_speaker = pd.read_csv(hawkish_dovish_speaker_file)
        hd_speaker = hd_speaker.sort_values('sentiment_score')

        # Get top 10 dovish and top 10 hawkish (or less hawkish)
        top_dovish = hd_speaker.head(10)
        top_hawkish = hd_speaker.tail(10).iloc[::-1]

        dovish_speakers = top_dovish['speaker'].tolist()
        dovish_sentiment = [round(x, 3) for x in top_dovish['sentiment_score'].tolist()]
        hawkish_speakers = top_hawkish['speaker'].tolist()
        hawkish_sentiment = [round(x, 3) for x in top_hawkish['sentiment_score'].tolist()]
        speaker_sentiment_available = True
    else:
        speaker_sentiment_available = False
else:
    speaker_sentiment_available = False
    hawkish_dovish_available = False

# === Speaker Table Data ===
speaker_stats = df.dropna(subset=['speaker']).groupby('speaker').agg({
    'parsed_date': ['min', 'max', 'count'],
    'word_count': 'mean'
}).reset_index()
speaker_stats.columns = ['Speaker', 'First Speech', 'Last Speech', 'Total Speeches', 'Avg Words']
speaker_stats['First Speech'] = speaker_stats['First Speech'].dt.strftime('%b %Y')
speaker_stats['Last Speech'] = speaker_stats['Last Speech'].dt.strftime('%b %Y')
speaker_stats['Avg Words'] = speaker_stats['Avg Words'].fillna(0).round(0).astype(int)
speaker_stats = speaker_stats.sort_values('Total Speeches', ascending=False)

# Generate table rows HTML
table_rows = ""
for _, row in speaker_stats.iterrows():
    table_rows += f"""
        <tr>
            <td>{row['Speaker']}</td>
            <td>{row['Total Speeches']}</td>
            <td>{row['First Speech']}</td>
            <td>{row['Last Speech']}</td>
            <td>{row['Avg Words']:,}</td>
        </tr>
    """

# === Generate HTML ===
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federal Reserve Speeches Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #fefefe 0%, #f8fafc 50%, #f0f9ff 100%);
            color: #475569;
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            text-align: center;
            padding: 0.5rem 0;
            background: linear-gradient(135deg, #0c4a6e 0%, #075985 50%, #0369a1 100%);
            color: white;
            margin-bottom: 0.75rem;
            border-radius: 0 0 0.75rem 0.75rem;
            box-shadow: 0 2px 10px rgba(12, 74, 110, 0.2);
        }}

        h1 {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0;
            letter-spacing: -0.3px;
        }}

        .subtitle {{
            font-size: 0.65rem;
            opacity: 0.85;
            font-weight: 400;
        }}

        .stats-grid {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
            padding: 0.85rem 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.1);
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.15rem;
            border: 1px solid rgba(14, 165, 233, 0.15);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.2);
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #0ea5e9, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .stat-label {{
            font-size: 0.65rem;
            color: #8b8ba7;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        .chart-card {{
            background: linear-gradient(135deg, #ffffff 0%, #fafeff 100%);
            border-radius: 1.25rem;
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.08);
            padding: 0.75rem;
            overflow: hidden;
            border: 1px solid rgba(14, 165, 233, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .chart-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(14, 165, 233, 0.12);
        }}

        .chart-card.full-width {{
            grid-column: 1 / -1;
        }}

        .section-title {{
            font-size: 1rem;
            font-weight: 600;
            color: #0284c7;
            margin: 1.5rem 0 0.75rem;
            padding: 0.6rem 1rem;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 1rem;
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid rgba(14, 165, 233, 0.15);
            transition: all 0.2s;
        }}

        .section-title:hover {{
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            border-color: rgba(14, 165, 233, 0.25);
        }}

        .section-title .toggle-icon {{
            font-size: 0.7rem;
            color: #38bdf8;
            transition: transform 0.2s;
        }}

        .section-title.collapsed .toggle-icon {{
            transform: rotate(-90deg);
        }}

        .section-content {{
            transition: max-height 0.3s ease, opacity 0.2s ease;
            overflow: hidden;
        }}

        .section-content.collapsed {{
            max-height: 0 !important;
            opacity: 0;
        }}

        .table-container {{
            background: linear-gradient(135deg, #ffffff 0%, #fafeff 100%);
            border-radius: 1.25rem;
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.08);
            overflow: hidden;
            margin-bottom: 2rem;
            border: 1px solid rgba(14, 165, 233, 0.1);
        }}

        .table-header {{
            padding: 0.85rem 1.25rem;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-bottom: 1px solid rgba(14, 165, 233, 0.12);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .table-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #0284c7;
        }}

        .search-box {{
            padding: 0.5rem 1rem;
            border: 1px solid rgba(14, 165, 233, 0.25);
            border-radius: 0.75rem;
            font-size: 0.9rem;
            width: 250px;
            outline: none;
            transition: all 0.2s;
            background: white;
        }}

        .search-box:focus {{
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .data-table th {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            color: #64748b;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid rgba(14, 165, 233, 0.15);
            cursor: pointer;
            user-select: none;
        }}

        .data-table th:hover {{
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        }}

        .data-table td {{
            padding: 1rem;
            border-bottom: 1px solid rgba(14, 165, 233, 0.08);
            color: #64748b;
        }}

        .data-table tbody tr:hover {{
            background: linear-gradient(135deg, #fafeff 0%, #f0f9ff 100%);
        }}

        .data-table tbody tr:last-child td {{
            border-bottom: none;
        }}

        footer {{
            text-align: center;
            padding: 1rem;
            color: #a0aec0;
            font-size: 0.75rem;
        }}

        .topic-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}

        .topic-card {{
            background: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            padding: 1rem;
            border-left: 4px solid #5a9bd4;
        }}

        .topic-card h3 {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #1a365d;
            margin-bottom: 0.5rem;
        }}

        .topic-words {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
        }}

        .topic-word {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            color: #64748b;
            padding: 0.25rem 0.6rem;
            border-radius: 0.5rem;
            font-size: 0.75rem;
        }}

        .topic-word.primary {{
            background: linear-gradient(135deg, #bae6fd 0%, #7dd3fc 100%);
            color: #0369a1;
            font-weight: 500;
        }}

        /* Speech Explorer Styles */
        .explorer-controls {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}

        .explorer-controls .search-box {{
            flex: 1;
            min-width: 200px;
        }}

        .lda-filter {{
            padding: 0.5rem 1rem;
            border: 1px solid rgba(14, 165, 233, 0.25);
            border-radius: 0.75rem;
            background: white;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .lda-filter:focus {{
            outline: none;
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
        }}

        .clear-filter-btn {{
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, #fb7185 0%, #f43f5e 100%);
            color: white;
            border: none;
            border-radius: 0.75rem;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(244, 63, 94, 0.25);
        }}

        .clear-filter-btn:hover {{
            background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
            transform: translateY(-1px);
        }}

        .filter-badge {{
            background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.7rem;
            margin-left: 0.5rem;
        }}

        .speech-count {{
            color: #94a3b8;
            font-size: 0.85rem;
        }}

        .speech-list {{
            max-height: 600px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .speech-item {{
            background: linear-gradient(135deg, #ffffff 0%, #fafeff 100%);
            border-radius: 1rem;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(14, 165, 233, 0.08);
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid rgba(14, 165, 233, 0.1);
            border-left: 4px solid #7dd3fc;
        }}

        .speech-item:hover {{
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.15);
            transform: translateY(-2px);
            border-color: rgba(14, 165, 233, 0.2);
        }}

        .speech-item-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
        }}

        .speech-item-speaker {{
            font-weight: 600;
            color: #0369a1;
            font-size: 0.95rem;
        }}

        .speech-item-date {{
            color: #94a3b8;
            font-size: 0.8rem;
        }}

        .speech-item-meta {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            flex-wrap: wrap;
        }}

        .speech-item-tag {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            color: #64748b;
            padding: 0.2rem 0.6rem;
            border-radius: 0.5rem;
            font-size: 0.75rem;
        }}

        .speech-item-tag.topic {{
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            color: #065f46;
        }}

        .speech-item-tag.bank {{
            background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
            color: #3730a3;
        }}

        .speech-item-tag.time {{
            background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
            color: #9d174d;
        }}

        .speech-item-tag.no-lda {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #92400e;
            font-style: italic;
        }}

        .speech-item-snippet {{
            color: #64748b;
            font-size: 0.85rem;
            line-height: 1.5;
        }}

        /* Modal Styles */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(15, 23, 42, 0.3);
            backdrop-filter: blur(4px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }}

        .modal-overlay.active {{
            display: flex;
        }}

        .modal-content {{
            background: linear-gradient(135deg, #ffffff 0%, #fafeff 100%);
            border-radius: 1.5rem;
            max-width: 800px;
            width: 100%;
            max-height: 80vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 25px 60px rgba(14, 165, 233, 0.2);
            border: 1px solid rgba(14, 165, 233, 0.15);
        }}

        .modal-close {{
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border: none;
            font-size: 1.25rem;
            cursor: pointer;
            color: #0284c7;
            width: 2.25rem;
            height: 2.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.2s;
        }}

        .modal-close:hover {{
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            color: #0369a1;
            transform: scale(1.05);
        }}

        .modal-header {{
            padding: 1.5rem;
            border-bottom: 1px solid rgba(14, 165, 233, 0.12);
            position: relative;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        }}

        .modal-header h3 {{
            color: #0369a1;
            font-size: 1.25rem;
            margin-bottom: 0.25rem;
            padding-right: 2rem;
        }}

        .modal-header p {{
            color: #94a3b8;
            font-size: 0.9rem;
        }}

        .modal-body {{
            padding: 1.5rem;
            overflow-y: auto;
            flex: 1;
            font-size: 0.95rem;
            line-height: 1.8;
            color: #475569;
            white-space: pre-wrap;
        }}

        /* Clickable chart indicator */
        .chart-card {{
            position: relative;
        }}

        .chart-card::after {{
            content: 'Click to explore';
            position: absolute;
            bottom: 0.5rem;
            right: 0.75rem;
            font-size: 0.7rem;
            color: #7dd3fc;
            opacity: 0;
            transition: opacity 0.2s;
        }}

        .chart-card:hover::after {{
            opacity: 1;
        }}

        /* Quick Navigation */
        .quick-nav {{
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            position: sticky;
            top: 0;
            z-index: 100;
            background: linear-gradient(135deg, rgba(254, 254, 254, 0.95) 0%, rgba(240, 249, 255, 0.95) 100%);
            padding: 0.75rem;
            border-radius: 1rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 10px rgba(14, 165, 233, 0.1);
        }}

        .nav-link {{
            padding: 0.4rem 0.8rem;
            background: white;
            border: 1px solid rgba(14, 165, 233, 0.2);
            border-radius: 0.75rem;
            color: #0284c7;
            text-decoration: none;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .nav-link:hover {{
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            border-color: #38bdf8;
            transform: translateY(-1px);
        }}

        .nav-link .nav-icon {{
            font-size: 0.9rem;
        }}

        /* Key Insights Panel */
        .insights-panel {{
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 1rem;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }}

        .insight-item {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 0.75rem;
            transition: all 0.2s;
        }}

        .insight-item:hover {{
            background: rgba(34, 197, 94, 0.15);
            transform: translateY(-1px);
        }}

        .insight-icon {{
            font-size: 1.25rem;
            width: 2.25rem;
            height: 2.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            border-radius: 0.75rem;
            box-shadow: 0 2px 8px rgba(34, 197, 94, 0.15);
        }}

        .insight-text {{
            flex: 1;
        }}

        .insight-label {{
            font-size: 0.7rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .insight-value {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #166534;
        }}

        /* Global Filter Bar */
        .global-filter-bar {{
            background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
            border: 1px solid rgba(234, 179, 8, 0.25);
            border-radius: 1rem;
            padding: 0.75rem 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .filter-label {{
            font-size: 0.8rem;
            font-weight: 600;
            color: #854d0e;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .filter-group label {{
            font-size: 0.75rem;
            color: #78716c;
        }}

        .global-filter {{
            padding: 0.4rem 0.75rem;
            border: 1px solid rgba(234, 179, 8, 0.35);
            border-radius: 0.5rem;
            background: white;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .global-filter:focus {{
            outline: none;
            border-color: #eab308;
            box-shadow: 0 0 0 3px rgba(234, 179, 8, 0.15);
        }}

        .global-search {{
            flex: 1;
            min-width: 180px;
            padding: 0.4rem 0.75rem;
            border: 1px solid rgba(234, 179, 8, 0.35);
            border-radius: 0.5rem;
            background: white;
            font-size: 0.85rem;
        }}

        .global-search:focus {{
            outline: none;
            border-color: #eab308;
            box-shadow: 0 0 0 3px rgba(234, 179, 8, 0.15);
        }}

        .active-filter-count {{
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 0.5rem;
            font-size: 0.7rem;
            font-weight: 600;
        }}

        /* Section Icons */
        .section-icon {{
            font-size: 1rem;
            margin-right: 0.5rem;
        }}

        /* Section type colors */
        .section-title.temporal {{
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-color: rgba(59, 130, 246, 0.2);
        }}
        .section-title.temporal:hover {{
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        }}

        .section-title.speaker {{
            background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%);
            border-color: rgba(168, 85, 247, 0.2);
        }}
        .section-title.speaker:hover {{
            background: linear-gradient(135deg, #fae8ff 0%, #f5d0fe 100%);
        }}

        .section-title.content {{
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            border-color: rgba(249, 115, 22, 0.2);
        }}
        .section-title.content:hover {{
            background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%);
        }}

        .section-title.topic {{
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-color: rgba(34, 197, 94, 0.2);
        }}
        .section-title.topic:hover {{
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        }}

        .section-title.explorer {{
            background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
            border-color: rgba(234, 179, 8, 0.2);
        }}
        .section-title.explorer:hover {{
            background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%);
        }}

        .section-title.table {{
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-color: rgba(100, 116, 139, 0.2);
        }}
        .section-title.table:hover {{
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        }}

        @media (max-width: 768px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}

            h1 {{
                font-size: 1.5rem;
            }}

            .stats-grid {{
                gap: 0.5rem;
            }}

            .stat-card {{
                padding: 0.5rem 1rem;
            }}

            .topic-grid {{
                grid-template-columns: 1fr;
            }}

            .quick-nav {{
                position: relative;
            }}

            .insights-panel {{
                grid-template-columns: 1fr 1fr;
            }}

            .global-filter-bar {{
                flex-direction: column;
                align-items: stretch;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Federal Reserve Speeches</h1>
            <p class="subtitle">Metadata Dashboard &bull; {year_range}</p>
        </div>
    </header>

    <div class="container">
        <!-- Quick Navigation -->
        <nav class="quick-nav">
            <a class="nav-link" onclick="scrollToSection('section-temporal')"><span class="nav-icon">📅</span> Timeline</a>
            <a class="nav-link" onclick="scrollToSection('section-speaker')"><span class="nav-icon">👤</span> Speakers</a>
            <a class="nav-link" onclick="scrollToSection('section-content')"><span class="nav-icon">📊</span> Content</a>
            <a class="nav-link" onclick="scrollToSection('section-topics')"><span class="nav-icon">🏷️</span> Topics</a>
            <a class="nav-link" onclick="scrollToSection('section-explorer')"><span class="nav-icon">🔍</span> Explorer</a>
            <a class="nav-link" onclick="scrollToSection('section-table')"><span class="nav-icon">📋</span> All Speakers</a>
        </nav>

        <!-- Summary Statistics -->
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-value">{total_speeches:,}</span>
                <span class="stat-label">speeches</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">{unique_speakers}</span>
                <span class="stat-label">speakers</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">{year_range}</span>
                <span class="stat-label">years</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">{total_words/1_000_000:.1f}M</span>
                <span class="stat-label">words</span>
            </div>
        </div>

        <!-- Key Insights Panel -->
        <div class="insights-panel">
            <div class="insight-item" onclick="applySpeechFilter('speaker', '{top_speaker_name}', '{top_speaker_name}')" title="Click to view speeches">
                <div class="insight-icon">🎤</div>
                <div class="insight-text">
                    <div class="insight-label">Most Active Speaker</div>
                    <div class="insight-value">{top_speaker_name} ({top_speaker_count:,} speeches)</div>
                </div>
            </div>
            <div class="insight-item" onclick="applySpeechFilter('year', {peak_year}, 'Year: {peak_year}')" title="Click to view speeches">
                <div class="insight-icon">📈</div>
                <div class="insight-text">
                    <div class="insight-label">Peak Year</div>
                    <div class="insight-value">{peak_year} ({peak_year_count:,} speeches)</div>
                </div>
            </div>
            <div class="insight-item" onclick="applyBankFilter('{most_common_bank}')" title="Click to view speeches">
                <div class="insight-icon">🏦</div>
                <div class="insight-text">
                    <div class="insight-label">Most Active Bank</div>
                    <div class="insight-value">{most_common_bank}</div>
                </div>
            </div>
            <div class="insight-item" onclick="applySpeechFilter('speaker', '{verbose_speaker}', '{verbose_speaker}')" title="Click to view speeches">
                <div class="insight-icon">📝</div>
                <div class="insight-text">
                    <div class="insight-label">Longest Speeches (avg)</div>
                    <div class="insight-value">{verbose_speaker} ({verbose_speaker_avg:,} words)</div>
                </div>
            </div>
        </div>

        <!-- Global Filter Bar -->
        <div class="global-filter-bar">
            <span class="filter-label">🔍 Filter All Charts</span>
            <div class="filter-group">
                <label>Bank:</label>
                <select id="globalBankFilter" class="global-filter" onchange="applyGlobalFilters()">
                    {bank_options_html}
                </select>
            </div>
            <div class="filter-group">
                <label>LDA:</label>
                <select id="globalLdaFilter" class="global-filter" onchange="applyGlobalFilters()">
                    <option value="all">All Speeches</option>
                    <option value="lda">With Topics</option>
                    <option value="no-lda">Without Topics</option>
                </select>
            </div>
            <input type="text" class="global-search" id="globalSearchInput" placeholder="Search speakers, topics..." onkeyup="applyGlobalFilters()">
            <span class="active-filter-count" id="activeFilterCount" style="display:none;">0 active</span>
        </div>

        <!-- Main Charts -->
        <h2 class="section-title temporal" onclick="toggleSection(this)" id="section-temporal">
            <span><span class="section-icon">📅</span> Temporal Analysis</span>
            <span class="toggle-icon">▼</span>
        </h2>
        <div class="section-content" style="max-height: 1000px;">
            <div class="chart-grid">
                <div class="chart-card full-width" id="chart-timeline"></div>
                <div class="chart-card" id="chart-heatmap"></div>
                <div class="chart-card" id="chart-timeofday"></div>
            </div>
        </div>

        <h2 class="section-title speaker collapsed" onclick="toggleSection(this)" id="section-speaker">
            <span><span class="section-icon">👤</span> Speaker Analysis</span>
            <span class="toggle-icon">▼</span>
        </h2>
        <div class="section-content collapsed" style="max-height: 1600px;">
            <div class="chart-grid">
                <div class="chart-card" id="chart-speakers"></div>
                <div class="chart-card" id="chart-avg-length"></div>
                <div class="chart-card" id="chart-roles"></div>
                <div class="chart-card" id="chart-roles-time"></div>
                <div class="chart-card full-width" id="chart-speaker-topics"></div>
            </div>
        </div>

        <h2 class="section-title content collapsed" onclick="toggleSection(this)" id="section-content">
            <span><span class="section-icon">📊</span> Content Analysis</span>
            <span class="toggle-icon">▼</span>
        </h2>
        <div class="section-content collapsed" style="max-height: 1400px;">
            <div class="chart-grid">
                <div class="chart-card full-width" id="chart-length"></div>
                <div class="chart-card full-width" id="chart-readability"></div>
                <div class="chart-card full-width" id="chart-sentiment"></div>
                <div class="chart-card full-width" id="chart-uncertainty"></div>
            </div>
        </div>

        <!-- Topic Analysis (LDA) -->
        <h2 class="section-title topic" onclick="toggleSection(this)" id="section-topics">
            <span><span class="section-icon">🏷️</span> Topic Analysis (LDA)</span>
            <span class="toggle-icon">▼</span>
        </h2>
        <div class="section-content" style="max-height: 1200px;">
            <div class="chart-grid">
                <div class="chart-card full-width" id="chart-topics-time"></div>
            </div>
            <div class="topic-grid" id="topic-cards"></div>
        </div>

        <!-- Speech Explorer -->
        <h2 class="section-title explorer collapsed" onclick="toggleSection(this)" id="section-explorer">
            <span><span class="section-icon">🔍</span> Speech Explorer <span id="speech-filter-badge" class="filter-badge" style="display:none;"></span></span>
            <span class="toggle-icon">▼</span>
        </h2>
        <div class="section-content collapsed" style="max-height: 800px;" id="speech-explorer-section">
            <div class="explorer-controls">
                <input type="text" class="search-box" id="speechSearchInput" placeholder="Search speeches..." onkeyup="filterSpeeches()">
                <select id="ldaFilter" class="lda-filter" onchange="filterSpeeches()">
                    <option value="all">All Speeches</option>
                    <option value="lda">With LDA Topics</option>
                    <option value="no-lda">Without LDA Topics</option>
                </select>
                <select id="bankFilter" class="lda-filter" onchange="filterSpeeches()">
                    {bank_options_html}
                </select>
                <button class="clear-filter-btn" onclick="clearSpeechFilter()" id="clearFilterBtn" style="display:none;">Clear Filter</button>
                <span class="speech-count" id="speechCount"></span>
            </div>
            <div class="speech-list" id="speechList"></div>
        </div>

        <!-- Speech Modal -->
        <div class="modal-overlay" id="speechModal" onclick="closeModal(event)">
            <div class="modal-content" onclick="event.stopPropagation()">
                <button class="modal-close" onclick="closeSpeechModal()">&times;</button>
                <div class="modal-header">
                    <h3 id="modalSpeaker"></h3>
                    <p id="modalMeta"></p>
                </div>
                <div class="modal-body" id="modalBody"></div>
            </div>
        </div>

        <!-- Speaker Table -->
        <h2 class="section-title table collapsed" onclick="toggleSection(this)" id="section-table">
            <span><span class="section-icon">📋</span> All Speakers ({unique_speakers})</span>
            <span class="toggle-icon">▼</span>
        </h2>
        <div class="section-content collapsed" style="max-height: 600px;">
            <div class="table-container">
                <div class="table-header">
                    <span class="table-title">Search and sort by clicking column headers</span>
                    <input type="text" class="search-box" id="searchInput" placeholder="Search speakers..." onkeyup="filterTable()">
                </div>
                <div class="table-content" id="tableContent" style="max-height: 450px; overflow-y: auto;">
                    <table class="data-table" id="speakerTable">
                        <thead>
                            <tr>
                                <th onclick="sortTable(0)">Speaker</th>
                                <th onclick="sortTable(1)">Speeches</th>
                                <th onclick="sortTable(2)">First Speech</th>
                                <th onclick="sortTable(3)">Last Speech</th>
                                <th onclick="sortTable(4)">Avg Words</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <p>Generated on {datetime.now().strftime('%B %d, %Y')} &bull; Data source: Fed Speeches Dataset</p>
    </footer>

    <script>
        // Data passed from Python
        const timelineYears = {json.dumps(timeline_years)};
        const timelineCounts = {json.dumps(timeline_counts)};
        const speakerNames = {json.dumps(speaker_names)};
        const speakerCounts = {json.dumps(speaker_counts)};
        const heatmapZ = {json.dumps(heatmap_z)};
        const heatmapX = {json.dumps(heatmap_x)};
        const heatmapY = {json.dumps(month_names)};
        const avgLengthNames = {json.dumps(avg_length_names)};
        const avgLengthValues = {json.dumps(avg_length_values)};
        const wordCounts = {json.dumps(word_counts)};

        // Time of Day Data
        const timeHours = {json.dumps(time_hours)};
        const timeCounts = {json.dumps(time_counts)};
        const timeLabels = {json.dumps(time_labels)};
        const speechesWithTime = {speeches_with_time};

        // Role Data
        const roleNames = {json.dumps(role_names)};
        const roleValues = {json.dumps(role_values)};
        const roleYears = {json.dumps(role_years)};
        const roleTracesData = {json.dumps(role_traces_data)};

        // LDA Topic Data
        const ldaAvailable = {'true' if lda_available else 'false'};
        {f"const topicYears = {json.dumps(topic_years)};" if lda_available else "const topicYears = [];"}
        {f"const topicLabels = {json.dumps(topic_labels)};" if lda_available else "const topicLabels = [];"}
        {f"const topicShortLabels = {json.dumps(topic_short_labels)};" if lda_available else "const topicShortLabels = [];"}
        {f"const topicTopWords = {json.dumps(topic_top_words)};" if lda_available else "const topicTopWords = [];"}
        {f"const topicTracesData = {json.dumps(topic_traces_data)};" if lda_available else "const topicTracesData = {{}};"}

        // Textual Metrics Data
        const textualMetricsAvailable = {'true' if textual_metrics_available else 'false'};
        {f"const metricsYears = {json.dumps(metrics_years)};" if textual_metrics_available else "const metricsYears = [];"}
        {f"const readabilityScores = {json.dumps(readability_scores)};" if textual_metrics_available else "const readabilityScores = [];"}
        {f"const uncertaintyScores = {json.dumps(uncertainty_scores)};" if textual_metrics_available else "const uncertaintyScores = [];"}

        // FinBERT-FOMC Hawkish/Dovish Data
        const hawkishDovishAvailable = {'true' if hawkish_dovish_available else 'false'};
        {f"const hdYears = {json.dumps(hd_years)};" if hawkish_dovish_available else "const hdYears = [];"}
        {f"const sentimentScores = {json.dumps(sentiment_scores)};" if hawkish_dovish_available else "const sentimentScores = [];"}
        {f"const pctHawkish = {json.dumps(pct_hawkish)};" if hawkish_dovish_available else "const pctHawkish = [];"}
        {f"const pctDovish = {json.dumps(pct_dovish)};" if hawkish_dovish_available else "const pctDovish = [];"}

        // Speaker Sentiment Data
        const speakerSentimentAvailable = {'true' if speaker_sentiment_available else 'false'};
        {f"const dovishSpeakers = {json.dumps(dovish_speakers)};" if speaker_sentiment_available else "const dovishSpeakers = [];"}
        {f"const dovishSentimentValues = {json.dumps(dovish_sentiment)};" if speaker_sentiment_available else "const dovishSentimentValues = [];"}
        {f"const hawkishSpeakers = {json.dumps(hawkish_speakers)};" if speaker_sentiment_available else "const hawkishSpeakers = [];"}
        {f"const hawkishSentimentValues = {json.dumps(hawkish_sentiment)};" if speaker_sentiment_available else "const hawkishSentimentValues = [];"}

        // Speaker Topic Data
        {f"const speakerTopicMatrix = {json.dumps(speaker_topic_matrix)};" if lda_available else "const speakerTopicMatrix = [];"}
        {f"const speakerTopicNames = {json.dumps(speaker_topic_names)};" if lda_available else "const speakerTopicNames = [];"}

        // Speech Explorer Data
        const allSpeeches = {json.dumps(speech_explorer_data)};
        let currentFilter = null;
        let filteredSpeeches = [...allSpeeches];

        // Chart 1: Timeline
        Plotly.newPlot('chart-timeline', [{{
            x: timelineYears,
            y: timelineCounts,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            line: {{ color: '#0284c7', width: 2.5, shape: 'spline' }},
            fillcolor: 'rgba(14, 165, 233, 0.15)',
            hovertemplate: '<b>%{{x}}</b><br>%{{y}} speeches<extra></extra>'
        }}], {{
            title: {{ text: 'Speeches Per Year', font: {{ size: 14, color: '#334155' }} }},
            xaxis: {{ title: '', dtick: 5, tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false, automargin: true }},
            yaxis: {{ title: '', tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
            height: 320,
            margin: {{ l: 45, r: 40, t: 45, b: 35 }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            hovermode: 'x unified'
        }}, {{ responsive: true }});

        // Chart 2: Heatmap
        Plotly.newPlot('chart-heatmap', [{{
            z: heatmapZ,
            x: heatmapX,
            y: heatmapY,
            type: 'heatmap',
            colorscale: [
                [0, '#f8fafc'],
                [0.15, '#e0f2fe'],
                [0.3, '#7dd3fc'],
                [0.5, '#38bdf8'],
                [0.7, '#0284c7'],
                [0.85, '#0369a1'],
                [1, '#0c4a6e']
            ],
            xgap: 2,
            ygap: 2,
            hovertemplate: '<b>%{{y}} %{{x}}</b><br>%{{z}} speeches<extra></extra>',
            colorbar: {{
                title: {{ text: 'Speeches', font: {{ size: 11, color: '#64748b' }} }},
                thickness: 12,
                len: 0.8,
                tickfont: {{ size: 10, color: '#64748b' }},
                outlinewidth: 0
            }}
        }}], {{
            title: {{
                text: 'Speech Frequency by Month and Year',
                font: {{ size: 14, color: '#334155' }}
            }},
            xaxis: {{ title: '', dtick: 5, tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(0,0,0,0)', automargin: true, constrain: 'domain' }},
            yaxis: {{ title: '', tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(0,0,0,0)', automargin: true }},
            height: 320,
            margin: {{ l: 35, r: 60, t: 45, b: 30 }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        }}, {{ responsive: true }});

        // Chart: Time of Day Distribution
        if (speechesWithTime > 0) {{
            Plotly.newPlot('chart-timeofday', [{{
                x: timeLabels,
                y: timeCounts,
                type: 'bar',
                marker: {{
                    color: timeCounts.map((c, i) => {{
                        // Color based on time of day: morning=warm, afternoon=cool, evening=purple
                        if (i >= 6 && i < 12) return 'rgba(251, 191, 36, 0.75)';  // Morning - amber
                        if (i >= 12 && i < 18) return 'rgba(14, 165, 233, 0.75)'; // Afternoon - blue
                        return 'rgba(139, 92, 246, 0.6)';  // Evening/night - purple
                    }}),
                    line: {{ color: 'rgba(100, 116, 139, 0.3)', width: 0.5 }}
                }},
                hovertemplate: '<b>%{{x}}</b><br>%{{y}} speeches<extra></extra>'
            }}], {{
                title: {{ text: 'Time of Day', font: {{ size: 14, color: '#334155' }} }},
                xaxis: {{
                    title: '',
                    tickfont: {{ size: 9, color: '#64748b' }},
                    tickangle: -45,
                    dtick: 2,
                    gridcolor: 'rgba(0,0,0,0)'
                }},
                yaxis: {{
                    title: '',
                    tickfont: {{ size: 10, color: '#64748b' }},
                    gridcolor: 'rgba(148, 163, 184, 0.2)',
                    zeroline: false
                }},
                height: 320,
                margin: {{ l: 45, r: 20, t: 50, b: 60 }},
                bargap: 0.15,
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                annotations: [{{
                    x: 0.5, y: 1.1, xref: 'paper', yref: 'paper',
                    text: speechesWithTime.toLocaleString() + ' speeches · Eastern Time',
                    showarrow: false, font: {{ size: 9, color: '#94a3b8' }}, xanchor: 'center'
                }}]
            }}, {{ responsive: true }});
        }}

        // Chart 3: Top Speakers
        Plotly.newPlot('chart-speakers', [{{
            x: speakerCounts,
            y: speakerNames,
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: speakerCounts.map((_, i) => `rgba(14, 165, 233, ${{0.4 + (i / speakerCounts.length) * 0.5}})`),
                line: {{ color: 'rgba(2, 132, 199, 0.3)', width: 1 }}
            }},
            hovertemplate: '<b>%{{y}}</b><br>%{{x}} speeches<extra></extra>'
        }}], {{
            title: {{ text: 'Most Frequent Speakers', font: {{ size: 14, color: '#334155' }} }},
            xaxis: {{ title: '', tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
            yaxis: {{ automargin: true, tickfont: {{ size: 10, color: '#64748b' }}, ticksuffix: '  —  ' }},
            height: 380,
            margin: {{ l: 130, r: 20, t: 40, b: 25 }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            bargap: 0.15
        }}, {{ responsive: true }});

        // Chart 4: Average Length
        Plotly.newPlot('chart-avg-length', [{{
            x: avgLengthValues,
            y: avgLengthNames,
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: avgLengthValues.map((_, i) => `rgba(16, 185, 129, ${{0.4 + (i / avgLengthValues.length) * 0.5}})`),
                line: {{ color: 'rgba(5, 150, 105, 0.3)', width: 1 }}
            }},
            hovertemplate: '<b>%{{y}}</b><br>%{{x:,}} avg words<extra></extra>'
        }}], {{
            title: {{ text: 'Longest Speeches (avg words)', font: {{ size: 14, color: '#334155' }} }},
            xaxis: {{ title: '', tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
            yaxis: {{ automargin: true, tickfont: {{ size: 10, color: '#64748b' }}, ticksuffix: '  —  ' }},
            height: 380,
            margin: {{ l: 130, r: 20, t: 40, b: 25 }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            bargap: 0.15
        }}, {{ responsive: true }});

        // Chart: Role Distribution
        const roleColorScale = [
            '#0c4a6e', '#0369a1', '#0284c7', '#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd', '#e0f2fe', '#f0f9ff'
        ];
        Plotly.newPlot('chart-roles', [{{
            x: roleValues,
            y: roleNames,
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: roleValues.map((_, i) => roleColorScale[i % roleColorScale.length]),
                line: {{ color: 'rgba(255,255,255,0.5)', width: 1 }}
            }},
            hovertemplate: '<b>%{{y}}</b><br>%{{x:,}} speeches<extra></extra>'
        }}], {{
            title: {{ text: 'Speeches by Role', font: {{ size: 14, color: '#334155' }} }},
            xaxis: {{ title: '', tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
            yaxis: {{ automargin: true, tickfont: {{ size: 11, color: '#475569' }} }},
            height: 380,
            margin: {{ l: 200, r: 30, t: 40, b: 30 }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            bargap: 0.2
        }}, {{ responsive: true }});

        // Chart: Roles Over Time - cleaner color palette
        const roleColors = {{
            'President': '#0c4a6e',
            'Governor': '#0ea5e9',
            'Chair': '#dc2626',
            'Vice Chair/Governor': '#059669',
            'Governor/Vice Chair': '#10b981',
            'Governor/Vice Chair for Supervision': '#f59e0b',
            'Executive Vice President': '#64748b',
            'First Vice President': '#94a3b8',
            'Interim President': '#cbd5e1'
        }};
        const roleLegendNames = {{
            'President': 'President',
            'Governor': 'Governor',
            'Chair': 'Chair',
            'Vice Chair/Governor': 'Vice Chair',
            'Governor/Vice Chair': 'Gov/VC',
            'Governor/Vice Chair for Supervision': 'VC Supervision',
            'Executive Vice President': 'Exec VP',
            'First Vice President': 'First VP',
            'Interim President': 'Interim'
        }};
        const roleTraces = Object.keys(roleTracesData).map(role => ({{
            x: roleYears,
            y: roleTracesData[role],
            name: roleLegendNames[role] || role,
            type: 'scatter',
            mode: 'lines',
            stackgroup: 'one',
            line: {{ width: 0.5, color: 'rgba(255,255,255,0.4)' }},
            fillcolor: roleColors[role] || '#94a3b8',
            hovertemplate: '<b>' + role + '</b><br>%{{y:.1f}}%<extra></extra>'
        }}));

        Plotly.newPlot('chart-roles-time', roleTraces, {{
            title: {{ text: 'Role Mix Over Time', font: {{ size: 14, color: '#334155' }} }},
            xaxis: {{ title: '', dtick: 5, tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(0,0,0,0)', zeroline: false, range: [roleYears[0] - 0.5, roleYears[roleYears.length-1] + 0.5] }},
            yaxis: {{ title: '', ticksuffix: '%', range: [0, 100], tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.15)', zeroline: false }},
            height: 380,
            margin: {{ l: 45, r: 30, t: 40, b: 90 }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            hovermode: 'x unified',
            legend: {{
                orientation: 'h',
                y: -0.25,
                x: 0.5,
                xanchor: 'center',
                font: {{ size: 10, color: '#475569' }},
                bgcolor: 'rgba(255,255,255,0.8)',
                borderwidth: 0
            }}
        }}, {{ responsive: true }});

        // Speaker-Topic Heatmap
        if (ldaAvailable && speakerTopicMatrix.length > 0) {{
            Plotly.newPlot('chart-speaker-topics', [{{
                z: speakerTopicMatrix,
                x: topicShortLabels,
                y: speakerTopicNames,
                type: 'heatmap',
                colorscale: [
                    [0, '#f8fafc'],
                    [0.2, '#e0f2fe'],
                    [0.4, '#7dd3fc'],
                    [0.6, '#38bdf8'],
                    [0.8, '#0284c7'],
                    [1, '#0c4a6e']
                ],
                xgap: 2,
                ygap: 2,
                hovertemplate: '<b>%{{y}}</b><br>%{{x}}: %{{z:.1f}}%<extra></extra>',
                colorbar: {{
                    title: {{ text: '%', font: {{ size: 11, color: '#64748b' }} }},
                    thickness: 12,
                    len: 0.8,
                    tickfont: {{ size: 10, color: '#64748b' }},
                    outlinewidth: 0
                }}
            }}], {{
                title: {{ text: 'Speaker Topic Focus', font: {{ size: 14, color: '#334155' }} }},
                xaxis: {{ title: '', tickangle: -45, tickfont: {{ size: 11, color: '#64748b' }}, automargin: true, gridcolor: 'rgba(0,0,0,0)' }},
                yaxis: {{ title: '', automargin: true, tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(0,0,0,0)' }},
                height: 480,
                margin: {{ l: 120, r: 60, t: 50, b: 180 }},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                annotations: [{{
                    x: 0.5, y: 1.08, xref: 'paper', yref: 'paper',
                    text: 'Avg topic probability per speaker (top 12 by speech count)',
                    showarrow: false, font: {{ size: 10, color: '#94a3b8' }}, xanchor: 'center'
                }}]
            }}, {{ responsive: true }});
        }}

        // Chart 5: Word Count Distribution
        Plotly.newPlot('chart-length', [{{
            x: wordCounts.filter(w => w <= 7000),
            type: 'histogram',
            nbinsx: 30,
            marker: {{
                color: 'rgba(251, 146, 60, 0.75)',
                line: {{ color: 'rgba(234, 88, 12, 0.4)', width: 0.5 }}
            }},
            hovertemplate: '<b>%{{x:,.0f}} words</b><br>%{{y}} speeches<extra></extra>'
        }}], {{
            title: {{ text: 'Distribution of Speech Lengths', font: {{ size: 14, color: '#334155' }} }},
            xaxis: {{
                title: '',
                range: [-200, 7200],
                dtick: 1000,
                tickfont: {{ size: 10, color: '#64748b' }},
                gridcolor: 'rgba(148, 163, 184, 0.2)',
                zeroline: false,
                tickformat: ',d',
                tickvals: [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
            }},
            yaxis: {{
                title: '',
                tickfont: {{ size: 10, color: '#64748b' }},
                gridcolor: 'rgba(148, 163, 184, 0.2)',
                zeroline: false,
                rangemode: 'tozero'
            }},
            height: 280,
            margin: {{ l: 50, r: 40, t: 45, b: 40 }},
            bargap: 0.05,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            annotations: [{{
                x: 0.98, y: 0.95, xref: 'paper', yref: 'paper',
                text: 'excludes ' + wordCounts.filter(w => w > 7000).length + ' speeches >7k',
                showarrow: false, font: {{ size: 9, color: '#94a3b8' }}, xanchor: 'right'
            }}]
        }}, {{ responsive: true }});

        // Textual Metrics Charts
        if (textualMetricsAvailable) {{
            // Chart 6: Readability Over Time
            Plotly.newPlot('chart-readability', [{{
                x: metricsYears,
                y: readabilityScores,
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: '#8b5cf6', width: 2.5, shape: 'spline' }},
                marker: {{ size: 5, color: '#8b5cf6' }},
                connectgaps: true,
                hovertemplate: '<b>%{{x}}</b><br>Fog Index: %{{y:.1f}}<extra></extra>'
            }}], {{
                title: {{ text: 'Readability Over Time (Gunning Fog Index)', font: {{ size: 14, color: '#334155' }} }},
                xaxis: {{ title: '', dtick: 5, range: [1997, 2026], tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
                yaxis: {{ title: '', range: [16.5, 20], tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
                height: 280,
                margin: {{ l: 45, r: 20, t: 55, b: 35 }},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                annotations: [{{
                    x: 0.5, y: 1.15, xref: 'paper', yref: 'paper',
                    text: 'Score ≈ years of education needed to understand',
                    showarrow: false, font: {{ size: 9, color: '#94a3b8' }}, xanchor: 'center'
                }}, {{
                    x: 0.98, y: 0.95, xref: 'paper', yref: 'paper',
                    text: 'Higher = harder',
                    showarrow: false, font: {{ size: 9, color: '#a78bfa' }}, xanchor: 'right'
                }}]
            }}, {{ responsive: true }});

            // Chart 7: Sentiment Over Time (Hawkish/Dovish) - FinBERT-FOMC
            if (hawkishDovishAvailable) {{
                Plotly.newPlot('chart-sentiment', [{{
                    x: hdYears,
                    y: sentimentScores,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{ color: '#475569', width: 2.5, shape: 'spline' }},
                    marker: {{ size: 5, color: '#475569' }},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(71, 85, 105, 0.1)',
                    connectgaps: true,
                    hovertemplate: '<b>%{{x}}</b><br>Score: %{{y:.3f}}<extra></extra>'
                }}], {{
                    title: {{ text: 'Hawkish vs Dovish Sentiment (FinBERT-FOMC)', font: {{ size: 14, color: '#334155' }} }},
                    xaxis: {{ title: '', dtick: 5, range: [1995, 2026], tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
                    yaxis: {{ title: '', zeroline: true, zerolinecolor: '#cbd5e1', zerolinewidth: 1, range: [-0.35, 0.15], tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)' }},
                    height: 280,
                    margin: {{ l: 45, r: 20, t: 55, b: 35 }},
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    annotations: [
                        {{ x: 0.5, y: 1.15, xref: 'paper', yref: 'paper', text: 'Score = avg(+1 hawkish, -1 dovish, 0 neutral)', showarrow: false, font: {{ size: 9, color: '#94a3b8' }}, xanchor: 'center' }},
                        {{ x: 0.98, y: 0.95, xref: 'paper', yref: 'paper', text: '↑ Hawkish', showarrow: false, font: {{ size: 9, color: '#10b981' }}, xanchor: 'right' }},
                        {{ x: 0.98, y: 0.08, xref: 'paper', yref: 'paper', text: '↓ Dovish', showarrow: false, font: {{ size: 9, color: '#f43f5e' }}, xanchor: 'right' }}
                    ]
                }}, {{ responsive: true }});
            }}

            // Chart 8: Uncertainty Index Over Time
            Plotly.newPlot('chart-uncertainty', [{{
                x: metricsYears,
                y: uncertaintyScores,
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: '#f59e0b', width: 2.5, shape: 'spline' }},
                marker: {{ size: 5, color: '#f59e0b' }},
                fill: 'tozeroy',
                fillcolor: 'rgba(245, 158, 11, 0.12)',
                connectgaps: true,
                hovertemplate: '<b>%{{x}}</b><br>%{{y:.1f}} hedging words/1000<extra></extra>'
            }}], {{
                title: {{ text: 'Uncertainty Index Over Time', font: {{ size: 14, color: '#334155' }} }},
                xaxis: {{ title: '', dtick: 5, range: [1997, 2026], tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
                yaxis: {{ title: '', range: [14, 24], tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.2)', zeroline: false }},
                height: 280,
                margin: {{ l: 45, r: 20, t: 55, b: 35 }},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                annotations: [{{
                    x: 0.5, y: 1.15, xref: 'paper', yref: 'paper',
                    text: 'Hedging words per 1000 (may, could, uncertain, risk, likely...)',
                    showarrow: false, font: {{ size: 9, color: '#94a3b8' }}, xanchor: 'center'
                }}]
            }}, {{ responsive: true }});
        }}

        // Chart 9: LDA Topics Over Time (8 improved topics)
        if (ldaAvailable) {{
            // Elegant color palette for 8 topics
            const topicColors = [
                '#0c4a6e', '#0369a1', '#0284c7', '#0ea5e9',  // Blues
                '#059669', '#10b981',  // Greens
                '#d97706', '#f59e0b'  // Amber/Orange
            ];

            // Create traces for stacked area chart
            const topicTraces = topicLabels.map((label, idx) => {{
                const isMonetaryPolicy = topicShortLabels[idx].includes('FOMC') || topicShortLabels[idx].includes('Monetary Policy');
                return {{
                    x: topicYears,
                    y: topicTracesData[label],
                    name: topicShortLabels[idx],
                    type: 'scatter',
                    mode: 'lines',
                    stackgroup: 'one',
                    line: {{ width: 0.5, color: 'rgba(255,255,255,0.3)' }},
                    fillcolor: topicColors[idx % topicColors.length],
                    hovertemplate: '<b>' + topicShortLabels[idx] + '</b><br>%{{y:.1f}}%<extra></extra>'
                }};
            }});

            Plotly.newPlot('chart-topics-time', topicTraces, {{
                title: {{ text: 'Topic Prevalence Over Time', font: {{ size: 14, color: '#334155' }} }},
                xaxis: {{ title: '', dtick: 5, tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(0,0,0,0)', zeroline: false }},
                yaxis: {{ title: '', ticksuffix: '%', range: [0, 100], tickfont: {{ size: 10, color: '#64748b' }}, gridcolor: 'rgba(148, 163, 184, 0.15)', zeroline: false }},
                height: 450,
                margin: {{ l: 45, r: 20, t: 50, b: 110 }},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                hovermode: 'x unified',
                legend: {{
                    orientation: 'h',
                    y: -0.22,
                    x: 0.5,
                    xanchor: 'center',
                    font: {{ size: 9, color: '#64748b' }}
                }},
                annotations: [{{
                    x: 0.5, y: 1.08, xref: 'paper', yref: 'paper',
                    text: 'LDA topic probability averaged by year',
                    showarrow: false, font: {{ size: 9, color: '#94a3b8' }}, xanchor: 'center'
                }}]
            }}, {{ responsive: true }});

            // Create topic cards
            const topicCardsContainer = document.getElementById('topic-cards');
            topicShortLabels.forEach((shortLabel, idx) => {{
                const words = topicTopWords[idx];
                const card = document.createElement('div');
                card.className = 'topic-card';
                card.style.borderLeftColor = topicColors[idx % topicColors.length];
                card.innerHTML = `
                    <h3>${{shortLabel}}</h3>
                    <div class="topic-words">
                        ${{words.slice(0, 3).map(w => `<span class="topic-word primary">${{w}}</span>`).join('')}}
                        ${{words.slice(3).map(w => `<span class="topic-word">${{w}}</span>`).join('')}}
                    </div>
                `;
                topicCardsContainer.appendChild(card);
            }});
        }}

        // Toggle section visibility
        function toggleSection(header) {{
            header.classList.toggle('collapsed');
            const content = header.nextElementSibling;
            content.classList.toggle('collapsed');
        }}

        // Scroll to section (for navigation)
        function scrollToSection(sectionId) {{
            const section = document.getElementById(sectionId);
            if (section) {{
                // Expand the section if collapsed
                if (section.classList.contains('collapsed')) {{
                    section.classList.remove('collapsed');
                    section.nextElementSibling.classList.remove('collapsed');
                }}
                // Scroll to it
                section.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }}

        // Global filter functions
        function applyGlobalFilters() {{
            const bankFilter = document.getElementById('globalBankFilter').value;
            const ldaFilter = document.getElementById('globalLdaFilter').value;
            const searchQuery = document.getElementById('globalSearchInput').value.toLowerCase();

            // Sync with speech explorer filters
            document.getElementById('bankFilter').value = bankFilter;
            document.getElementById('ldaFilter').value = ldaFilter;
            document.getElementById('speechSearchInput').value = searchQuery;

            // Count active filters
            let activeCount = 0;
            if (bankFilter !== 'all') activeCount++;
            if (ldaFilter !== 'all') activeCount++;
            if (searchQuery) activeCount++;

            const countEl = document.getElementById('activeFilterCount');
            if (activeCount > 0) {{
                countEl.style.display = 'inline';
                countEl.textContent = activeCount + ' active';
            }} else {{
                countEl.style.display = 'none';
            }}

            // Apply to speech explorer
            filterSpeeches();
        }}

        // Sync global filters when speech explorer filters change
        function syncGlobalFilters() {{
            document.getElementById('globalBankFilter').value = document.getElementById('bankFilter').value;
            document.getElementById('globalLdaFilter').value = document.getElementById('ldaFilter').value;
            document.getElementById('globalSearchInput').value = document.getElementById('speechSearchInput').value;
            applyGlobalFilters();
        }}

        // Table search functionality
        function filterTable() {{
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const table = document.getElementById('speakerTable');
            const rows = table.getElementsByTagName('tr');

            for (let i = 1; i < rows.length; i++) {{
                const cells = rows[i].getElementsByTagName('td');
                let found = false;
                for (let j = 0; j < cells.length; j++) {{
                    if (cells[j].textContent.toLowerCase().includes(filter)) {{
                        found = true;
                        break;
                    }}
                }}
                rows[i].style.display = found ? '' : 'none';
            }}
        }}

        // Table sort functionality
        let sortDirection = {{}};
        function sortTable(columnIndex) {{
            const table = document.getElementById('speakerTable');
            const rows = Array.from(table.rows).slice(1);
            const isNumeric = columnIndex === 1 || columnIndex === 4;

            sortDirection[columnIndex] = !sortDirection[columnIndex];
            const direction = sortDirection[columnIndex] ? 1 : -1;

            rows.sort((a, b) => {{
                let aVal = a.cells[columnIndex].textContent;
                let bVal = b.cells[columnIndex].textContent;

                if (isNumeric) {{
                    aVal = parseInt(aVal.replace(/,/g, ''));
                    bVal = parseInt(bVal.replace(/,/g, ''));
                    return (aVal - bVal) * direction;
                }}
                return aVal.localeCompare(bVal) * direction;
            }});

            const tbody = table.getElementsByTagName('tbody')[0];
            rows.forEach(row => tbody.appendChild(row));
        }}

        // ============ Speech Explorer Functions ============

        // Render speech list
        function renderSpeeches(speeches, limit = 100) {{
            const container = document.getElementById('speechList');
            const displaySpeeches = speeches.slice(0, limit);

            container.innerHTML = displaySpeeches.map(s => `
                <div class="speech-item" onclick="openSpeechModal(${{s.id}})">
                    <div class="speech-item-header">
                        <span class="speech-item-speaker">${{s.speaker}}${{s.role ? ` <span style="font-weight:400;color:#94a3b8;font-size:0.85em">(${{s.role}})</span>` : ''}}</span>
                        <span class="speech-item-date">${{s.date}}${{s.time ? ` at ${{s.time}}` : ''}}</span>
                    </div>
                    <div class="speech-item-meta">
                        <span class="speech-item-tag">${{s.year}}</span>
                        ${{s.time ? `<span class="speech-item-tag time">🕐 ${{s.time}}</span>` : ''}}
                        ${{s.bank ? `<span class="speech-item-tag bank">${{s.bank}}</span>` : ''}}
                        <span class="speech-item-tag">${{s.words.toLocaleString()}} words</span>
                        ${{s.has_lda ? `<span class="speech-item-tag topic">${{s.topic}}</span>` : '<span class="speech-item-tag no-lda">No LDA Topic</span>'}}
                    </div>
                    <div class="speech-item-snippet">${{s.snippet}}</div>
                </div>
            `).join('');

            // Update count
            const countEl = document.getElementById('speechCount');
            if (speeches.length > limit) {{
                countEl.textContent = `Showing ${{limit}} of ${{speeches.length}} speeches`;
            }} else {{
                countEl.textContent = `${{speeches.length}} speeches`;
            }}
        }}

        // Filter speeches by search text, LDA filter, and bank filter
        function filterSpeeches() {{
            const query = document.getElementById('speechSearchInput').value.toLowerCase();
            const ldaFilter = document.getElementById('ldaFilter').value;
            const bankFilter = document.getElementById('bankFilter').value;
            let results = currentFilter ? filteredSpeeches : allSpeeches;

            // Apply LDA filter
            if (ldaFilter === 'lda') {{
                results = results.filter(s => s.has_lda === true);
            }} else if (ldaFilter === 'no-lda') {{
                results = results.filter(s => s.has_lda === false);
            }}

            // Apply bank filter
            if (bankFilter !== 'all') {{
                results = results.filter(s => s.bank === bankFilter);
            }}

            // Apply search filter
            if (query) {{
                results = results.filter(s =>
                    s.speaker.toLowerCase().includes(query) ||
                    s.snippet.toLowerCase().includes(query) ||
                    (s.topic && s.topic.toLowerCase().includes(query)) ||
                    s.date.includes(query)
                );
            }}

            renderSpeeches(results);
        }}

        // Apply filter from chart click
        function applySpeechFilter(filterType, filterValue, filterLabel, filterValue2 = null) {{
            currentFilter = {{ type: filterType, value: filterValue, label: filterLabel }};

            if (filterType === 'year') {{
                filteredSpeeches = allSpeeches.filter(s => s.year === filterValue);
            }} else if (filterType === 'speaker') {{
                filteredSpeeches = allSpeeches.filter(s => s.speaker === filterValue);
            }} else if (filterType === 'topic') {{
                filteredSpeeches = allSpeeches.filter(s => s.topic_id === filterValue);
            }} else if (filterType === 'wordcount') {{
                // filterValue = min, filterValue2 = max
                filteredSpeeches = allSpeeches.filter(s => s.words >= filterValue && s.words < filterValue2);
            }} else if (filterType === 'hour') {{
                filteredSpeeches = allSpeeches.filter(s => s.hour === filterValue);
            }} else if (filterType === 'role') {{
                filteredSpeeches = allSpeeches.filter(s => s.role === filterValue);
            }}

            // Update UI
            document.getElementById('clearFilterBtn').style.display = 'inline-block';
            document.getElementById('speech-filter-badge').style.display = 'inline';
            document.getElementById('speech-filter-badge').textContent = filterLabel;

            // Scroll to and open speech explorer
            const header = document.getElementById('section-explorer');
            const section = document.getElementById('speech-explorer-section');
            if (header.classList.contains('collapsed')) {{
                header.classList.remove('collapsed');
                section.classList.remove('collapsed');
            }}
            header.scrollIntoView({{ behavior: 'smooth', block: 'start' }});

            // Clear search and render
            document.getElementById('speechSearchInput').value = '';
            renderSpeeches(filteredSpeeches);
        }}

        // Clear filter
        function clearSpeechFilter() {{
            currentFilter = null;
            filteredSpeeches = [...allSpeeches];
            document.getElementById('clearFilterBtn').style.display = 'none';
            document.getElementById('speech-filter-badge').style.display = 'none';
            document.getElementById('speechSearchInput').value = '';
            document.getElementById('ldaFilter').value = 'all';
            document.getElementById('bankFilter').value = 'all';
            document.getElementById('globalBankFilter').value = 'all';
            document.getElementById('globalLdaFilter').value = 'all';
            document.getElementById('globalSearchInput').value = '';
            document.getElementById('activeFilterCount').style.display = 'none';
            renderSpeeches(allSpeeches);
        }}

        // Apply bank filter (for insights panel click)
        function applyBankFilter(bank) {{
            // Set the bank filter
            document.getElementById('bankFilter').value = bank;
            document.getElementById('globalBankFilter').value = bank;

            // Update UI
            document.getElementById('clearFilterBtn').style.display = 'inline-block';
            document.getElementById('speech-filter-badge').style.display = 'inline';
            document.getElementById('speech-filter-badge').textContent = bank;
            document.getElementById('activeFilterCount').style.display = 'inline';
            document.getElementById('activeFilterCount').textContent = '1 active';

            // Scroll to and open speech explorer
            const header = document.getElementById('section-explorer');
            const section = document.getElementById('speech-explorer-section');
            if (header.classList.contains('collapsed')) {{
                header.classList.remove('collapsed');
                section.classList.remove('collapsed');
            }}
            header.scrollIntoView({{ behavior: 'smooth', block: 'start' }});

            // Filter speeches
            filterSpeeches();
        }}

        // Open speech modal
        function openSpeechModal(speechId) {{
            const speech = allSpeeches.find(s => s.id === speechId);
            if (!speech) return;

            document.getElementById('modalSpeaker').textContent = speech.speaker;
            document.getElementById('modalMeta').textContent = `${{speech.date}} · ${{speech.words.toLocaleString()}} words${{speech.topic ? ' · ' + speech.topic : ''}}`;
            document.getElementById('modalBody').textContent = speech.preview;
            document.getElementById('speechModal').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}

        // Close modal
        function closeSpeechModal() {{
            document.getElementById('speechModal').classList.remove('active');
            document.body.style.overflow = '';
        }}

        function closeModal(event) {{
            if (event.target === document.getElementById('speechModal')) {{
                closeSpeechModal();
            }}
        }}

        // Keyboard escape to close modal
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeSpeechModal();
        }});

        // ============ Chart Click Handlers ============

        // Timeline click - filter by year
        document.getElementById('chart-timeline').on('plotly_click', function(data) {{
            const year = data.points[0].x;
            applySpeechFilter('year', year, `Year: ${{year}}`);
        }});

        // Top speakers click - filter by speaker
        document.getElementById('chart-speakers').on('plotly_click', function(data) {{
            const speaker = data.points[0].y;
            applySpeechFilter('speaker', speaker, speaker);
        }});

        // Avg length speakers click - filter by speaker
        document.getElementById('chart-avg-length').on('plotly_click', function(data) {{
            const speaker = data.points[0].y;
            applySpeechFilter('speaker', speaker, speaker);
        }});

        // Heatmap click - filter by year and month
        document.getElementById('chart-heatmap').on('plotly_click', function(data) {{
            const year = data.points[0].x;
            applySpeechFilter('year', year, `Year: ${{year}}`);
        }});

        // Time of day click - filter by hour
        if (speechesWithTime > 0) {{
            document.getElementById('chart-timeofday').on('plotly_click', function(data) {{
                const hourLabel = data.points[0].x;
                const hour = parseInt(hourLabel.split(':')[0]);
                const ampm = hour < 12 ? 'AM' : 'PM';
                const displayHour = hour === 0 ? 12 : (hour > 12 ? hour - 12 : hour);
                applySpeechFilter('hour', hour, `${{displayHour}}:00 ${{ampm}}`);
            }});
        }}

        // Role chart click - filter by role
        document.getElementById('chart-roles').on('plotly_click', function(data) {{
            const role = data.points[0].y.replace('  —  ', '');
            applySpeechFilter('role', role, role);
        }});

        // Topic cards click - make them clickable
        if (ldaAvailable) {{
            document.querySelectorAll('.topic-card').forEach((card, idx) => {{
                card.style.cursor = 'pointer';
                card.addEventListener('click', () => {{
                    applySpeechFilter('topic', idx, topicShortLabels[idx]);
                }});
            }});
        }}

        // Word count histogram click - filter by word count range
        document.getElementById('chart-length').on('plotly_click', function(data) {{
            const point = data.points[0];
            // Get the bin edges from the histogram
            const binStart = Math.floor(point.x / 200) * 200;
            const binEnd = binStart + 200;
            applySpeechFilter('wordcount', binStart, `${{binStart.toLocaleString()}}-${{binEnd.toLocaleString()}} words`, binEnd);
        }});

        // LDA topics over time click - filter by topic
        if (ldaAvailable) {{
            document.getElementById('chart-topics-time').on('plotly_click', function(data) {{
                const topicIdx = data.points[0].curveNumber;
                const topicName = topicShortLabels[topicIdx];
                applySpeechFilter('topic', topicIdx, topicName);
            }});
        }}

        // Sentiment chart click - filter by year
        if (hawkishDovishAvailable) {{
            document.getElementById('chart-sentiment').on('plotly_click', function(data) {{
                const year = data.points[0].x;
                applySpeechFilter('year', year, `Year: ${{year}}`);
            }});
        }}

        // Readability chart click - filter by year
        if (textualMetricsAvailable) {{
            document.getElementById('chart-readability').on('plotly_click', function(data) {{
                const year = data.points[0].x;
                applySpeechFilter('year', year, `Year: ${{year}}`);
            }});
            document.getElementById('chart-uncertainty').on('plotly_click', function(data) {{
                const year = data.points[0].x;
                applySpeechFilter('year', year, `Year: ${{year}}`);
            }});
        }}

        // Initialize speech explorer
        renderSpeeches(allSpeeches);
    </script>
</body>
</html>
"""

# Save the dashboard
output_path = '/Users/sophiakazinnik/Research/central_bank_speeches_communication/fed_speeches_dashboard.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Dashboard saved to: {output_path}")
print(f"File size: {len(html_content) / 1024:.0f} KB (uses Plotly CDN)")
