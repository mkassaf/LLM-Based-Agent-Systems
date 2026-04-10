import pandas as pd

# CONFIG
MAIN_CSV = "notebooks/csv/github_agent_repos_python_final.csv"
SAMPLE_CSV = "notebooks/csv/clean_sample_for_llm_shortReadme.csv"
OUTPUT_CSV = "notebooks/csv/clean_sample_for_llm_completeReadme.csv"

# Load both CSVs
main_df = pd.read_csv(MAIN_CSV)
sample_df = pd.read_csv(SAMPLE_CSV)

# Get all full_name values from sample
sample_fullnames = sample_df["full_name"].tolist()

# Filter main_df to only rows where full_name is in sample
filtered_df = main_df[main_df["full_name"].isin(sample_fullnames)]

# Find unmatched — full_names in sample but NOT found in main
matched_fullnames = filtered_df["full_name"].tolist()
unmatched = [name for name in sample_fullnames if name not in matched_fullnames]

# Save result
filtered_df.to_csv(OUTPUT_CSV, index=False)

print(f"Sample rows:      {len(sample_df)}")
print(f"Matched rows:     {len(filtered_df)}")
print(f"Unmatched rows:   {len(unmatched)}")

if unmatched:
    print("\nUnmatched full_names (in sample but not found in main):")
    for name in unmatched:
        print(f"  - {name}")
else:
    print("\nAll rows matched successfully!")