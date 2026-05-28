from __future__ import annotations

import argparse
import os

import pandas as pd

from common import ensure_dir, load_data, merge_project_frames, split_by_files, write_metric_table
from modeling import run_model_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    frames = load_data(args.data)
    full = merge_project_frames(frames)
    train, val, test = split_by_files(full, frames)

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "predictions"))
    ensure_dir(os.path.join(args.out, "metrics"))

    methods = [
        "Keyword matching",
        "TF-IDF + Logistic Regression",
        "DistilBERT classifier",
        "FT-Transformer only",
        "Rule-only prover",
        "ML-only fusion model",
    ]

    rows = []
    all_pred = []
    for method in methods:
        pred, _, metrics = run_model_variant(method, train, val, test)
        rows.append(metrics)
        keep_cols = [
            "case_id",
            "rule_family",
            "candidate_rule_id",
            "expected_rule_id",
            "expected_proof_state",
            "ml_prediction",
            "validated_prediction",
            "confidence",
            "human_review_flag",
            "legal_validation_passed",
            "supporting_evidence",
            "missing_evidence",
        ]
        keep_cols = [c for c in keep_cols if c in pred.columns]
        part = pred[keep_cols].copy()
        part["method"] = method
        all_pred.append(part)

    write_metric_table(rows, os.path.join(args.out, "metrics", "baseline_metrics.csv"))
    pd.concat(all_pred, ignore_index=True).to_csv(os.path.join(args.out, "predictions", "baseline_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
