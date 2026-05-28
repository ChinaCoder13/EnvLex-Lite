from __future__ import annotations

import argparse
import os
import pickle

import pandas as pd

from common import case_file_text, ensure_dir, load_data, merge_project_frames, split_by_files, write_metric_table
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

    pred, _, metrics = run_model_variant("EnvLex-Lite", train, val, test)
    pred["case_file_summary"] = pred.apply(case_file_text, axis=1)

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
        "framework_corrective_action",
        "case_file_summary",
    ]
    keep_cols = [c for c in keep_cols if c in pred.columns]
    pred[keep_cols].to_csv(os.path.join(args.out, "predictions", "envlex_lite_predictions.csv"), index=False)
    write_metric_table([metrics], os.path.join(args.out, "metrics", "envlex_lite_metrics.csv"))


if __name__ == "__main__":
    main()
