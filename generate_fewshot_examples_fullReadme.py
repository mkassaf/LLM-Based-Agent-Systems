import json
import pandas as pd

# CONFIG
FEW_SHOT_JSON_INPUT = "notebooks/data/few_shot_examples_shortReadme_binaryClass.json"
MAIN_CSV = "notebooks/csv/github_agent_repos_python_final.csv"
FEW_SHOT_JSON_OUTPUT = "notebooks/data/few_shot_examples_fullReadme_binaryClass.json"

# Load few-shot JSON
with open(FEW_SHOT_JSON_INPUT, "r") as f:
    few_shot_data = json.load(f)

# Load main CSV
main_df = pd.read_csv(MAIN_CSV)

print(f"Few-shot examples:  {len(few_shot_data)}")
print(f"Main CSV rows:      {len(main_df)}")

# Process each example
output = []
unmatched = []

for ex in few_shot_data:
    full_name = ex.get("full_name", "")
    category = ex.get("category", "")

    # Find matching row in main CSV
    match = main_df[main_df["full_name"] == full_name]

    if match.empty:
        unmatched.append(full_name)
        continue

    # Take first matched row as dict
    row = match.iloc[0].to_dict()

    # Add category from JSON
    row["category"] = category

    output.append(row)

# Save output JSON
with open(FEW_SHOT_JSON_OUTPUT, "w") as f:
    json.dump(output, f, indent=2)

# Summary
print(f"Matched:            {len(output)}")
print(f"Unmatched:          {len(unmatched)}")

if unmatched:
    print("\nUnmatched full_names (in JSON but not found in CSV):")
    for name in unmatched:
        print(f"  - {name}")
else:
    print("\nAll examples matched successfully!")

print(f"\nSaved to: {FEW_SHOT_JSON_OUTPUT}")