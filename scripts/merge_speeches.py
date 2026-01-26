"""
Script to merge Fed speeches CSV files into a single file.
All input files are expected to have the same columns:
id, speaker, date, time, text, source
"""

import pandas as pd
from pathlib import Path

def merge_speech_files(input_dir: Path, output_file: Path) -> None:
    """
    Merge all CSV files in input_dir into a single output file.
    Files are processed in chronological order (by year in filename).
    """
    # Find all CSV files matching the expected pattern
    csv_files = sorted(input_dir.glob("speeches_with_time_and_text_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    print(f"Found {len(csv_files)} CSV files to merge:")
    for f in csv_files:
        print(f"  - {f.name}")

    # Read and validate all files
    dataframes = []
    expected_columns = None

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Validate columns match
        if expected_columns is None:
            expected_columns = list(df.columns)
            print(f"\nExpected columns: {expected_columns}")
        elif list(df.columns) != expected_columns:
            raise ValueError(
                f"Column mismatch in {csv_file.name}.\n"
                f"Expected: {expected_columns}\n"
                f"Got: {list(df.columns)}"
            )

        print(f"  {csv_file.name}: {len(df)} rows")
        dataframes.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    print(f"\nTotal rows after merge: {len(merged_df)}")

    # Handle potential missing values in date column
    valid_dates = merged_df['date'].dropna()
    if len(valid_dates) > 0:
        print(f"Date range: {valid_dates.iloc[0]} to {valid_dates.iloc[-1]}")
    print(f"Unique speakers: {merged_df['speaker'].nunique()}")

    # Save merged file
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged file saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    input_dir = Path("/Users/sophiakazinnik/Research/central_bank_speeches_communication/speech_data")
    output_file = input_dir / "all_speeches_merged.csv"

    merge_speech_files(input_dir, output_file)
