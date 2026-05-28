from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


TEXT_COLUMN = "inspection_text"
CASE_COLUMN = "case_id"
TARGET_COLUMN = "expected_proof_state"
RULE_ID_COLUMN = "candidate_rule_id"
EXPECTED_RULE_COLUMN = "expected_rule_id"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    frames = {
        "cases": read_csv(os.path.join(data_dir, "environmental_cases.csv")),
        "rules": read_csv(os.path.join(data_dir, "rule_library.csv")),
        "outputs": read_csv(os.path.join(data_dir, "case_outputs.csv")),
        "train": read_csv(os.path.join(data_dir, "train_ids.csv")),
        "val": read_csv(os.path.join(data_dir, "val_ids.csv")),
        "test": read_csv(os.path.join(data_dir, "test_ids.csv")),
    }
    return frames


def normalize_yes_no(value) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "y"}:
        return "yes"
    if text in {"no", "false", "0", "n"}:
        return "no"
    return text


def safe_ratio(a, b) -> float:
    try:
        a = float(a)
        b = float(b)
        if abs(b) < 1e-12:
            return 0.0
        return round(a / b, 6)
    except Exception:
        return 0.0


def merge_project_frames(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cases = frames["cases"].copy()
    rules = frames["rules"].copy()
    outputs = frames["outputs"].copy()

    rule_cols = [
        "rule_id",
        "rule_description",
        "corrective_action_template",
        "legal_logic_type",
        "risk_weight",
        "regulated_focus",
        "regulated_item",
        "controlled_action",
        "required_document",
        "evidence_requirement",
        "human_review_trigger",
    ]
    rule_cols = [c for c in rule_cols if c in rules.columns]
    rules_small = rules[rule_cols].drop_duplicates("rule_id")
    merged = cases.merge(
        rules_small,
        how="left",
        left_on=RULE_ID_COLUMN,
        right_on="rule_id",
        suffixes=("", "_rule"),
    )

    out_cols = [
        CASE_COLUMN,
        TARGET_COLUMN,
        EXPECTED_RULE_COLUMN,
        "expected_corrective_action",
        "expected_human_review_flag",
    ]
    out_cols = [c for c in out_cols if c in outputs.columns]
    merged = merged.merge(outputs[out_cols], how="left", on=CASE_COLUMN)
    return merged


def split_by_files(frame: pd.DataFrame, frames: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = []
    for key in ["train", "val", "test"]:
        ids = frames[key][CASE_COLUMN].astype(str).tolist()
        part = frame[frame[CASE_COLUMN].astype(str).isin(ids)].copy()
        out.append(part)
    return tuple(out)


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    blocked = {
        CASE_COLUMN,
        TARGET_COLUMN,
        EXPECTED_RULE_COLUMN,
        "expected_corrective_action",
        "expected_human_review_flag",
        "rule_id",
        "corrective_action_template",
    }
    text_cols = [c for c in [TEXT_COLUMN, "rule_description"] if c in df.columns]
    numeric_cols = []
    categorical_cols = []
    for c in df.columns:
        if c in blocked or c in text_cols:
            continue
        if c.startswith("expected_"):
            continue
        if c.startswith("validated_"):
            continue
        if c.startswith("predicted_"):
            continue
        if c.endswith("_flag") and c not in {"active_flag"}:
            categorical_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols, text_cols


def combine_text(df: pd.DataFrame, text_cols: Sequence[str]) -> pd.Series:
    if not text_cols:
        return pd.Series([""] * len(df), index=df.index)
    values = []
    for _, row in df.iterrows():
        parts = []
        for c in text_cols:
            val = row.get(c, "")
            if pd.notna(val):
                parts.append(str(val))
        values.append(" [RULE] ".join(parts))
    return pd.Series(values, index=df.index)


def labels_from_training(y: pd.Series) -> List[str]:
    return sorted(pd.Series(y).dropna().astype(str).unique().tolist())


def safe_probability_array(model, x, labels: List[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        if hasattr(model, "classes_"):
            class_list = [str(c) for c in model.classes_]
            aligned = np.zeros((x.shape[0], len(labels)), dtype=float)
            for idx, lab in enumerate(labels):
                if lab in class_list:
                    aligned[:, idx] = probs[:, class_list.index(lab)]
            row_sum = aligned.sum(axis=1)
            for i, total in enumerate(row_sum):
                if total <= 0:
                    aligned[i, :] = 1.0 / len(labels)
                else:
                    aligned[i, :] /= total
            return aligned
        return probs
    pred = pd.Series(model.predict(x)).astype(str).tolist()
    arr = np.zeros((len(pred), len(labels)), dtype=float)
    for i, p in enumerate(pred):
        if p in labels:
            arr[i, labels.index(p)] = 1.0
        else:
            arr[i, :] = 1.0 / len(labels)
    return arr


def multiclass_auroc(y_true: Sequence[str], proba: np.ndarray, labels: List[str]) -> float:
    try:
        y = pd.Series(y_true).astype(str)
        observed = sorted(y.unique().tolist())
        if len(observed) < 2:
            return float("nan")
        return float(roc_auc_score(y, proba, labels=labels, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def expected_calibration_error(y_true: Sequence[str], proba: np.ndarray, labels: List[str], bins: int = 10) -> float:
    y = pd.Series(y_true).astype(str).tolist()
    conf = proba.max(axis=1)
    pred = [labels[int(np.argmax(row))] for row in proba]
    correct = np.array([int(p == t) for p, t in zip(pred, y)], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (conf > left) & (conf <= right) if i > 0 else (conf >= left) & (conf <= right)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evidence_complete_score(frame: pd.DataFrame) -> float:
    needed = [
        "candidate_rule_id",
        "rule_family",
        "evidence_quality",
        "domestic_mapping_available",
        "missing_critical_field",
        "structured_text_conflict",
    ]
    available = [c for c in needed if c in frame.columns]
    if not available or len(frame) == 0:
        return float("nan")
    vals = []
    for _, row in frame.iterrows():
        score = 0
        for c in available:
            val = row.get(c)
            if pd.notna(val) and str(val).strip() != "":
                score += 1
        vals.append(score / len(available))
    return float(np.mean(vals))


def rule_match_accuracy(frame: pd.DataFrame, pred_rule_col: str = "rule_prediction") -> float:
    if EXPECTED_RULE_COLUMN not in frame.columns or pred_rule_col not in frame.columns:
        return float("nan")
    return float((frame[EXPECTED_RULE_COLUMN].astype(str) == frame[pred_rule_col].astype(str)).mean())


def missing_evidence_detection(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    true_miss = pd.Series(y_true).astype(str).isin(["insufficient_evidence", "human_review_required"])
    pred_miss = pd.Series(y_pred).astype(str).isin(["insufficient_evidence", "human_review_required"])
    return float((true_miss.values == pred_miss.values).mean())


def unsupported_claim_rate(frame: pd.DataFrame, pred_col: str = "validated_prediction") -> float:
    if pred_col not in frame.columns or len(frame) == 0:
        return float("nan")
    confirmed = frame[pred_col].astype(str).eq("confirmed_violation")
    if confirmed.sum() == 0:
        return 0.0
    valid = (
        frame.get("rule_applicability", "yes").map(normalize_yes_no).eq("yes")
        & frame.get("obligation_exists", "yes").map(normalize_yes_no).eq("yes")
        & frame.get("factual_breach_indicator", "yes").map(normalize_yes_no).eq("yes")
        & frame.get("evidence_quality", "verified").astype(str).str.lower().isin(["verified", "valid", "complete", "acceptable"])
        & frame.get("domestic_mapping_available", "yes").map(normalize_yes_no).eq("yes")
    )
    unsupported = confirmed & (~valid)
    return float(unsupported.sum() / confirmed.sum())


def legal_validation_pass_rate(frame: pd.DataFrame, pred_col: str = "validated_prediction") -> float:
    if pred_col not in frame.columns or len(frame) == 0:
        return float("nan")
    ok = []
    for _, row in frame.iterrows():
        pred = str(row.get(pred_col, ""))
        if pred == "confirmed_violation":
            ok.append(
                normalize_yes_no(row.get("rule_applicability")) == "yes"
                and normalize_yes_no(row.get("obligation_exists")) == "yes"
                and normalize_yes_no(row.get("factual_breach_indicator")) == "yes"
                and str(row.get("evidence_quality", "")).lower() in {"verified", "valid", "complete", "acceptable"}
                and normalize_yes_no(row.get("domestic_mapping_available")) == "yes"
            )
        elif pred == "human_review_required":
            ok.append(
                normalize_yes_no(row.get("structured_text_conflict")) == "yes"
                or str(row.get("domestic_mapping_available", "")).lower() in {"unclear", "ambiguous", "no"}
                or str(row.get("exemption_status", "")).lower() in {"unclear", "ambiguous"}
            )
        elif pred == "insufficient_evidence":
            ok.append(
                normalize_yes_no(row.get("missing_critical_field")) == "yes"
                or str(row.get("evidence_quality", "")).lower() in {"invalid", "missing", "weak", "unclear"}
            )
        else:
            ok.append(True)
    return float(np.mean(ok))


def compute_metrics(frame: pd.DataFrame, labels: List[str], pred_col: str, proba: Optional[np.ndarray] = None, time_per_case: float = np.nan) -> Dict[str, float]:
    y_true = frame[TARGET_COLUMN].astype(str).tolist()
    y_pred = frame[pred_col].astype(str).tolist()
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "auroc": multiclass_auroc(y_true, proba, labels) if proba is not None else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "rule_matching_accuracy": rule_match_accuracy(frame),
        "missing_evidence_detection": missing_evidence_detection(y_true, y_pred),
        "evidence_completeness_score": float(frame.get("case_evidence_score", pd.Series([evidence_complete_score(frame)] * len(frame))).mean()),
        "legal_validation_pass_rate": legal_validation_pass_rate(frame, pred_col=pred_col),
        "unsupported_claim_rate": unsupported_claim_rate(frame, pred_col=pred_col),
        "time_per_case_seconds": time_per_case,
    }
    return row


@dataclass
class RuleDecision:
    final_state: str
    review_flag: str
    evidence_text: str
    missing_text: str
    action_text: str
    validation_passed: str


class RuleValidator:
    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def validate_row(self, row: pd.Series, proposed_state: str, confidence: float) -> RuleDecision:
        regulated = normalize_yes_no(row.get("is_regulated_item"))
        controlled = normalize_yes_no(row.get("is_controlled_action"))
        applies = normalize_yes_no(row.get("rule_applicability"))
        obligation = normalize_yes_no(row.get("obligation_exists"))
        breach = normalize_yes_no(row.get("factual_breach_indicator"))
        domestic = normalize_yes_no(row.get("domestic_mapping_available"))
        missing = normalize_yes_no(row.get("missing_critical_field"))
        conflict = normalize_yes_no(row.get("structured_text_conflict"))
        quality = str(row.get("evidence_quality", "")).strip().lower()
        exemption = str(row.get("exemption_status", "")).strip().lower()

        final_state = str(proposed_state)
        review = "no"
        validation = "yes"

        if conflict == "yes" or exemption in {"unclear", "ambiguous"}:
            final_state = "human_review_required"
            review = "yes"
        elif regulated == "no" or controlled == "no" or applies == "no" or obligation == "no" or breach == "no":
            final_state = "no_violation"
        elif domestic in {"no", "unclear", "ambiguous"}:
            final_state = "human_review_required"
            review = "yes"
        elif missing == "yes" or quality in {"invalid", "missing", "weak", "unclear"}:
            final_state = "insufficient_evidence"
        elif confidence < self.threshold and final_state in {"confirmed_violation", "probable_violation"}:
            final_state = "human_review_required"
            review = "yes"

        if final_state == "confirmed_violation":
            action = str(row.get("corrective_action_template", "")).strip()
            if not action:
                action = "Apply corrective action required by the matched rule template"
        elif final_state == "probable_violation":
            action = "Verify supporting evidence and apply the matched corrective action when confirmed"
        elif final_state == "insufficient_evidence":
            action = "Request missing evidence before a final compliance decision"
        elif final_state == "human_review_required":
            action = "Route the case for officer or legal review"
        else:
            action = "No corrective action required; retain the record for audit trail"

        evidence_parts = []
        if regulated == "yes":
            evidence_parts.append("regulated item")
        if controlled == "yes":
            evidence_parts.append("controlled action")
        if breach == "yes":
            evidence_parts.append("breach indicator")
        if quality in {"verified", "valid", "complete", "acceptable"}:
            evidence_parts.append("valid evidence quality")
        if domestic == "yes":
            evidence_parts.append("domestic or permit mapping available")
        if not evidence_parts:
            evidence_parts.append("no complete evidence chain")

        missing_parts = []
        if missing == "yes":
            missing_parts.append("critical field")
        if domestic != "yes":
            missing_parts.append("domestic or permit mapping")
        if quality not in {"verified", "valid", "complete", "acceptable"}:
            missing_parts.append("verified evidence quality")
        if conflict == "yes":
            missing_parts.append("consistent structured and text evidence")
        if not missing_parts:
            missing_parts.append("none")

        if final_state == "confirmed_violation" and validation != "yes":
            validation = "no"

        return RuleDecision(
            final_state=final_state,
            review_flag=review,
            evidence_text="; ".join(evidence_parts),
            missing_text="; ".join(missing_parts),
            action_text=action,
            validation_passed=validation,
        )


def apply_validator(frame: pd.DataFrame, state_col: str, proba_col: str, threshold: float) -> pd.DataFrame:
    validator = RuleValidator(threshold=threshold)
    out = frame.copy()
    states = []
    reviews = []
    evidences = []
    misses = []
    actions = []
    passes = []
    for _, row in out.iterrows():
        dec = validator.validate_row(row, row[state_col], float(row.get(proba_col, 0.0)))
        states.append(dec.final_state)
        reviews.append(dec.review_flag)
        evidences.append(dec.evidence_text)
        misses.append(dec.missing_text)
        actions.append(dec.action_text)
        passes.append(dec.validation_passed)
    out["validated_prediction"] = states
    out["human_review_flag"] = reviews
    out["supporting_evidence"] = evidences
    out["missing_evidence"] = misses
    out["framework_corrective_action"] = actions
    out["legal_validation_passed"] = passes
    return out


def case_file_text(row: pd.Series) -> str:
    return (
        f"{row.get(CASE_COLUMN)}: {row.get('rule_family')} case assessed as "
        f"{row.get('validated_prediction')} using {row.get('supporting_evidence')}."
    )


def make_rule_prediction(frame: pd.DataFrame) -> pd.Series:
    if EXPECTED_RULE_COLUMN in frame.columns and RULE_ID_COLUMN in frame.columns:
        return frame[RULE_ID_COLUMN].astype(str)
    if RULE_ID_COLUMN in frame.columns:
        return frame[RULE_ID_COLUMN].astype(str)
    return pd.Series([""] * len(frame), index=frame.index)


def confidence_from_proba(proba: np.ndarray) -> np.ndarray:
    if proba is None or len(proba) == 0:
        return np.array([])
    return proba.max(axis=1)


def choose_threshold(y_true: Sequence[str], pred: Sequence[str], conf: Sequence[float]) -> float:
    values = np.linspace(0.30, 0.80, 21)
    y_true = pd.Series(y_true).astype(str).tolist()
    pred = pd.Series(pred).astype(str).tolist()
    conf = np.asarray(conf, dtype=float)
    best_t = 0.50
    best_score = -1.0
    for t in values:
        routed = []
        for p, c in zip(pred, conf):
            if c < t and p in {"confirmed_violation", "probable_violation"}:
                routed.append("human_review_required")
            else:
                routed.append(p)
        score = f1_score(y_true, routed, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def write_metric_table(rows: List[Dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    pd.DataFrame(rows).to_csv(path, index=False)


def clean_name(text: str) -> str:
    return str(text).replace(" ", "_").replace("-", "_").replace("+", "plus").replace("/", "_").lower()
