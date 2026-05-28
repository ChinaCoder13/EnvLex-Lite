from __future__ import annotations

import argparse
import os

import pandas as pd

from common import ensure_dir, load_data, merge_project_frames, split_by_files
from modeling import fit_envlex, fit_ml_only, fit_structured_only, fit_text_only, run_model_variant
from common import compute_metrics, labels_from_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    ensure_dir(os.path.join(args.out, "metrics"))
    frames = load_data(args.data)
    full = merge_project_frames(frames)
    train, val, test = split_by_files(full, frames)

    variants = [
        ("Full EnvLex-Lite", "EnvLex-Lite"),
        ("Without RECAE", "FT-Transformer only"),
        ("Without FT-Transformer", "DistilBERT classifier"),
        ("Without LEF-Net", "ML-only fusion model"),
        ("Without rule-constrained validator", "ML-only fusion model"),
    ]

    rows = []
    for display, method in variants:
        pred, proba, metrics = run_model_variant(method, train, val, test)
        metrics["model_variant"] = display
        rows.append(metrics)

    table = pd.DataFrame(rows)
    cols = [
        "model_variant",
        "accuracy",
        "macro_f1",
        "evidence_completeness_score",
        "legal_validation_pass_rate",
        "unsupported_claim_rate",
        "rule_matching_accuracy",
        "time_per_case_seconds",
    ]
    cols = [c for c in cols if c in table.columns]
    table[cols].to_csv(os.path.join(args.out, "metrics", "ablation_results.csv"), index=False)


if __name__ == "__main__":
    main()
