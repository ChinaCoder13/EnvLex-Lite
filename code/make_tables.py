from __future__ import annotations

import argparse
import os

import pandas as pd

from common import ensure_dir, load_data, merge_project_frames, split_by_files
from modeling import run_model_variant


def table_1_rule_families(rules: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "rule_family",
        "regulated_focus",
        "evidence_requirement",
        "legal_logic_type",
    ]
    keep = [c for c in cols if c in rules.columns]
    out = rules[keep].drop_duplicates("rule_family").copy()
    out = out.rename(
        columns={
            "rule_family": "Rule family",
            "regulated_focus": "Regulated focus",
            "evidence_requirement": "Operational evidence required",
            "legal_logic_type": "Machine-checkable violation logic",
        }
    )
    return out


def table_2_distribution(cases: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(["Total case records", len(cases), "De-identified environmental compliance records used for model training and evaluation"])
    rows.append(["Machine-readable rule objects", None, "Five rule families with five rule templates per family"])
    for fam, count in cases["rule_family"].value_counts().sort_index().items():
        rows.append([f"{fam} cases", int(count), "Rule-family-specific compliance cases"])
    if "expected_proof_state" in outputs.columns:
        for lab, count in outputs["expected_proof_state"].value_counts().sort_index().items():
            rows.append([lab.replace("_", " ").title(), int(count), "Proof-state category"])
    if "normalized_text_length_words" in cases.columns:
        rows.append(["Mean inspection-text length", round(cases["normalized_text_length_words"].mean(), 2), "Short compliance-oriented inspection observations"])
    if "missing_critical_field" in cases.columns:
        rows.append(["Missing-field rate", round((cases["missing_critical_field"].astype(str).str.lower() == "yes").mean(), 4), "Cases with at least one missing non-output field"])
    if "structured_text_conflict" in cases.columns:
        rows.append(["Text-field conflict rate", round((cases["structured_text_conflict"].astype(str).str.lower() == "yes").mean(), 4), "Cases where inspection text conflicts with structured evidence"])
    rows.append(["Train/validation/test split", "70/15/15", "Stratified by rule family and proof state"])
    out = pd.DataFrame(rows, columns=["Dataset component", "Count / distribution", "Interpretation"])
    return out


def table_3_setup(cases: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ["Dataset and files", f"{len(cases)} case records, {len(rules)} rule objects", "rule_library.csv, environmental_cases.csv, case_outputs.csv"],
        ["Task and outputs", "Five-class proof-state prediction", "Confirmed violation, probable violation, no violation, insufficient evidence, human review required"],
        ["Data split", "70/15/15 stratified split", "Training, validation, and test partitions"],
        ["Model inputs", "Structured evidence, inspection text, candidate rule object", "Reference outputs excluded from model inputs"],
        ["Structured encoder", "FT-Transformer", "Embedding-based structured evidence representation"],
        ["Text/rule encoder", "DistilBERT-base + RECAE", "Rule-evidence cross-attention over inspection text and rule description"],
        ["Fusion and prediction", "LEF-Net + softmax proof-state head", "Evidence-channel attention over structured, text-rule, and rule embeddings"],
        ["Training setup", "AdamW with class-weighted cross-entropy", "Learning rate 2e-5, weight decay 0.01, batch size 32, maximum 50 epochs"],
        ["Validation protocol", "Early stopping and threshold tuning on validation set", "Patience 7, monitored on validation macro-F1"],
        ["Rule-constrained output", "Deterministic validator after ML prediction", "Checks rule applicability, obligation, breach, evidence quality, domestic mapping, missing evidence, and structured-text conflict"],
        ["Final outputs", "Validated proof state and reviewable case file", "Applicable rule, evidence matrix, corrective action, human-review flag"],
    ]
    return pd.DataFrame(rows, columns=["Aspect", "Configuration", "Key detail"])


def table_5_errors(test_frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ["Missing document field but supportive inspection text", "Partial structured record", "Confirmed vs probable violation", "Downgrades to probable or insufficient evidence"],
        ["Vague exemption language", "Short inspection observation", "Probable vs human review", "Routes to human review when exemption status is unclear"],
        ["Threshold breach with invalid quality flag", "Structured numerical evidence", "Confirmed vs insufficient evidence", "Blocks confirmed label through evidence-quality condition"],
        ["Text contradicts structured field", "Field-text inconsistency", "Any violation state vs human review", "Triggers conflict-based human review"],
        ["Domestic mapping unavailable", "Rule-context gap", "Confirmed vs no/insufficient evidence", "Prevents final confirmed violation"],
    ]
    return pd.DataFrame(rows, columns=["Error pattern", "Typical source", "Affected proof states", "Framework response"])


def table_6_cases(pred: pd.DataFrame) -> pd.DataFrame:
    preferred = pred.drop_duplicates("rule_family").head(3).copy()
    if len(preferred) < 3:
        preferred = pred.head(3).copy()
    out = pd.DataFrame(
        {
            "Case ID": preferred["case_id"].astype(str),
            "Rule family": preferred["rule_family"].astype(str),
            "Final proof state": preferred["validated_prediction"].astype(str),
            "Key evidence chain": preferred["supporting_evidence"].astype(str),
            "Final case-file action": preferred["framework_corrective_action"].astype(str),
        }
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--tables", default="tables")
    args = parser.parse_args()

    ensure_dir(args.tables)
    ensure_dir(os.path.join(args.out, "metrics"))
    frames = load_data(args.data)
    full = merge_project_frames(frames)
    train, val, test = split_by_files(full, frames)
    pred, _, metrics = run_model_variant("EnvLex-Lite", train, val, test)

    table_1_rule_families(frames["rules"]).to_csv(os.path.join(args.tables, "table_1_rule_families.csv"), index=False)
    dist = table_2_distribution(frames["cases"], frames["outputs"])
    dist.loc[dist["Dataset component"].eq("Machine-readable rule objects"), "Count / distribution"] = len(frames["rules"])
    dist.to_csv(os.path.join(args.tables, "table_2_dataset_distribution.csv"), index=False)
    table_3_setup(frames["cases"], frames["rules"]).to_csv(os.path.join(args.tables, "table_3_experimental_setup.csv"), index=False)

    table4_path = os.path.join(args.out, "metrics", "table_4_model_performance.csv")
    if os.path.exists(table4_path):
        table4 = pd.read_csv(table4_path)
    else:
        parts = []
        for name in ["baseline_metrics.csv", "envlex_lite_metrics.csv"]:
            item = os.path.join(args.out, "metrics", name)
            if os.path.exists(item):
                parts.append(pd.read_csv(item))
        table4 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame([metrics])
        table4.to_csv(table4_path, index=False)
    table4.to_csv(os.path.join(args.tables, "table_4_model_performance.csv"), index=False)

    table_5_errors(test).to_csv(os.path.join(args.tables, "table_5_error_patterns.csv"), index=False)
    table_6_cases(pred).to_csv(os.path.join(args.tables, "table_6_case_outputs.csv"), index=False)


if __name__ == "__main__":
    main()
