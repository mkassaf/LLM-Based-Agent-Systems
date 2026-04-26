"""
evaluate.py

Compares LLM-predicted categories against manually annotated ground truth labels.

Usage:
    python evaluate.py
    python evaluate.py --pred <predicted_csv> --truth <ground_truth_csv> --output <output_csv>
"""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --- DEFAULT PATHS ---
DEFAULT_PRED_CSV   = "notebooks/data/sample_agent_repos_llm_filtered_withFullReadme_DeepSeekR1_mlx_v16.csv"
DEFAULT_TRUTH_CSV  = "notebooks/data/clean_sample_agent_repos_withShortReadme_manual.csv"
DEFAULT_OUTPUT_CSV = "notebooks/data/evaluation_results.csv"

# Column names
PRED_COL   = "category"          # predicted label column in pred CSV
TRUTH_COL  = "manual_normalized" # ground truth column in truth CSV
JOIN_COL   = "full_name"         # key to join the two CSVs

CATEGORIES = ["llm-based agentic system", "other"]


def load_and_merge(pred_path, truth_path):
    pred_df  = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)

    # Normalise labels to lowercase and strip whitespace
    pred_df[PRED_COL]   = pred_df[PRED_COL].str.strip().str.lower()
    truth_df[TRUTH_COL] = truth_df[TRUTH_COL].str.strip().str.lower()

    merged = pd.merge(
        pred_df[[JOIN_COL, PRED_COL] + [c for c in ["confidence", "key_evidence", "reasoning_summary"] if c in pred_df.columns]],
        truth_df[[JOIN_COL, TRUTH_COL] + [c for c in ["Comment"] if c in truth_df.columns]],
        on=JOIN_COL,
        how="inner"
    )

    print(f"Predicted CSV   : {len(pred_df)} rows")
    print(f"Ground truth CSV: {len(truth_df)} rows")
    print(f"Matched on '{JOIN_COL}': {len(merged)} rows\n")

    if len(merged) == 0:
        raise ValueError(f"No rows matched on '{JOIN_COL}'. Check that both files share the same repo names.")

    return merged


def compute_metrics(df):
    y_true = df[TRUTH_COL].tolist()
    y_pred = df[PRED_COL].tolist()

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="llm-based agentic system", zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label="llm-based agentic system", zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label="llm-based agentic system", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_report(metrics, df):
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    n      = len(df)

    print("=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"Total evaluated : {n}")
    print(f"Correct         : {sum(t == p for t, p in zip(y_true, y_pred))}")
    print(f"Incorrect       : {sum(t != p for t, p in zip(y_true, y_pred))}")
    print()
    print(f"Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision : {metrics['precision']:.4f}  (for 'llm-based agentic system')")
    print(f"Recall    : {metrics['recall']:.4f}  (for 'llm-based agentic system')")
    print(f"F1 Score  : {metrics['f1']:.4f}  (for 'llm-based agentic system')")
    print()

    print("--- Per-class Report ---")
    print(classification_report(y_true, y_pred, labels=CATEGORIES, zero_division=0))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    col_w = max(len(c) for c in CATEGORIES) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in CATEGORIES) + "  (predicted)"
    print(header)
    for i, row_label in enumerate(CATEGORIES):
        row = f"{row_label:<{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(CATEGORIES)))
        print(row)
    print("(actual)")
    print()


def save_results(df, output_path):
    df = df.copy()
    df["correct"] = df[TRUTH_COL] == df[PRED_COL]
    df["true_category"] = df[TRUTH_COL]
    df["predicted_category"] = df[PRED_COL]

    # Reorder columns for readability
    front_cols = [JOIN_COL, "true_category", "predicted_category", "correct"]
    optional   = ["confidence", "reasoning_summary", "key_evidence", "Comment"]
    rest       = [c for c in df.columns if c not in front_cols + optional + [TRUTH_COL, PRED_COL]]
    ordered    = front_cols + [c for c in optional if c in df.columns] + rest

    df[ordered].to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")

    # Print incorrect predictions
    wrong = df[~df["correct"]]
    if len(wrong) > 0:
        print(f"\n--- Incorrect Predictions ({len(wrong)}) ---")
        for _, r in wrong.iterrows():
            confidence = f"  confidence={r['confidence']:.2f}" if "confidence" in r else ""
            reasoning  = f"\n    reasoning: {r['reasoning_summary']}" if "reasoning_summary" in r else ""
            print(f"  {r[JOIN_COL]}")
            print(f"    true={r['true_category']}  predicted={r['predicted_category']}{confidence}{reasoning}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM classification results against ground truth.")
    parser.add_argument("--pred",   default=DEFAULT_PRED_CSV,   help="Path to predicted CSV")
    parser.add_argument("--truth",  default=DEFAULT_TRUTH_CSV,  help="Path to ground truth CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Path to save evaluation results CSV")
    args = parser.parse_args()

    df      = load_and_merge(args.pred, args.truth)
    metrics = compute_metrics(df)
    print_report(metrics, df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(df, args.output)


if __name__ == "__main__":
    main()
