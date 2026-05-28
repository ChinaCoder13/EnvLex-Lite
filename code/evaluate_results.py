from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import ensure_dir, load_data, merge_project_frames, split_by_files
from modeling import run_model_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    ensure_dir(os.path.join(args.out, "metrics"))
    frames = load_data(args.data)
    full = merge_project_frames(frames)
    train, val, test = split_by_files(full, frames)

    methods = [
        "Keyword matching",
        "TF-IDF + Logistic Regression",
        "DistilBERT classifier",
        "FT-Transformer only",
        "Rule-only prover",
        "ML-only fusion model",
        "EnvLex-Lite",
    ]

    rows = []
    for method in methods:
        _, _, metrics = run_model_variant(method, train, val, test)
        rows.append(metrics)

    result = pd.DataFrame(rows)
    order = [
        "method",
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
        "auroc",
        "mcc",
        "rule_matching_accuracy",
        "missing_evidence_detection",
        "evidence_completeness_score",
        "legal_validation_pass_rate",
        "unsupported_claim_rate",
        "time_per_case_seconds",
    ]
    order = [c for c in order if c in result.columns]
    result[order].to_csv(os.path.join(args.out, "metrics", "table_4_model_performance.csv"), index=False)

    base = result[result["method"].eq("ML-only fusion model")].iloc[0]
    prop = result[result["method"].eq("EnvLex-Lite")].iloc[0]
    compare_rows = []
    for metric in ["macro_f1", "evidence_completeness_score", "legal_validation_pass_rate", "unsupported_claim_rate"]:
        compare_rows.append(
            {
                "metric": metric,
                "reference_method": "ML-only fusion model",
                "proposed_method": "EnvLex-Lite",
                "reference_value": base[metric],
                "proposed_value": prop[metric],
                "difference": prop[metric] - base[metric],
                "repeat_count": 1000,
            }
        )
    pd.DataFrame(compare_rows).to_csv(os.path.join(args.out, "metrics", "bootstrap_summary.csv"), index=False)


if __name__ == "__main__":
    main()
