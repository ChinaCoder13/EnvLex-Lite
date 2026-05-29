from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from common import ensure_dir, load_data, merge_project_frames, split_by_files
from modeling import run_model_variant


def save_heatmap(matrix, xlabels, ylabels, title, path, xlabel="", ylabel="", colorbar_label="Score"):
    fig, ax = plt.subplots(figsize=(max(8, len(xlabels) * 0.8), max(5, len(ylabels) * 0.45)))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def load_prediction_table(data_dir, out_dir):
    pred_path = os.path.join(out_dir, "predictions", "envlex_lite_predictions.csv")
    if os.path.exists(pred_path):
        return pd.read_csv(pred_path)
    frames = load_data(data_dir)
    full = merge_project_frames(frames)
    train, val, test = split_by_files(full, frames)
    pred, _, _ = run_model_variant("EnvLex-Lite", train, val, test)
    keep = [
        "case_id",
        "rule_family",
        "expected_proof_state",
        "validated_prediction",
        "confidence",
        "human_review_flag",
    ]
    return pred[[c for c in keep if c in pred.columns]]


def make_performance_heatmap(metrics_path, fig_dir):
    table = pd.read_csv(metrics_path)
    metric_cols = [c for c in table.columns if c != "method" and pd.api.types.is_numeric_dtype(table[c])]
    matrix = table[metric_cols].astype(float).values
    save_heatmap(
        matrix,
        metric_cols,
        table["method"].tolist(),
        "Overall performance heatmap",
        os.path.join(fig_dir, "fig3_performance_heatmap.png"),
        xlabel="Predictive and legal-usefulness metrics",
        ylabel="Compared methods",
        colorbar_label="Metric value",
    )


def make_rule_family_heatmap(pred, fig_dir):
    rows = []
    for fam, group in pred.groupby("rule_family"):
        labels = sorted(group["expected_proof_state"].astype(str).unique())
        for lab in labels:
            sub = group[group["expected_proof_state"].astype(str).eq(lab)]
            score = (sub["validated_prediction"].astype(str) == sub["expected_proof_state"].astype(str)).mean()
            rows.append((fam, lab, score))
    data = pd.DataFrame(rows, columns=["family", "proof_state", "score"])
    pivot = data.pivot_table(index="family", columns="proof_state", values="score", fill_value=np.nan)
    save_heatmap(
        pivot.values,
        pivot.columns.tolist(),
        pivot.index.tolist(),
        "Rule-family and proof-state performance heatmap",
        os.path.join(fig_dir, "fig4_rule_family_heatmap.png"),
        xlabel="Legal-evidence proof state",
        ylabel="Rule family",
        colorbar_label="Performance score",
    )


def make_calibration(pred, fig_dir):
    true = pred["expected_proof_state"].astype(str).values
    proposed = pred["validated_prediction"].astype(str).values
    conf = pd.to_numeric(pred.get("confidence", pd.Series([0.5] * len(pred))), errors="coerce").fillna(0.5).values
    correct = (proposed == true).astype(int)
    bins = np.linspace(0, 1, 11)
    x = []
    y = []
    for i in range(len(bins) - 1):
        mask = (conf >= bins[i]) & (conf <= bins[i + 1]) if i == 0 else (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        x.append(conf[mask].mean())
        y.append(correct[mask].mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal calibration")
    ax.plot(x, y, marker="o", label="EnvLex-Lite")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed correctness")
    ax.set_title("Confidence calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5a_calibration_curve.png"), dpi=300)
    plt.close(fig)


def make_risk_coverage(pred, fig_dir):
    true = pred["expected_proof_state"].astype(str).values
    proposed = pred["validated_prediction"].astype(str).values
    conf = pd.to_numeric(pred.get("confidence", pd.Series([0.5] * len(pred))), errors="coerce").fillna(0.5).values
    order = np.argsort(-conf)
    coverages = []
    risks = []
    for k in range(1, len(order) + 1):
        idx = order[:k]
        coverages.append(k / len(order))
        risks.append(1 - (proposed[idx] == true[idx]).mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(coverages, risks)
    ax.set_xlabel("Decision coverage")
    ax.set_ylabel("Prediction risk")
    ax.set_title("Risk–coverage curve")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5b_risk_coverage_curve.png"), dpi=300)
    plt.close(fig)


def make_confusion(pred, fig_dir):
    labels = sorted(pd.concat([pred["expected_proof_state"], pred["validated_prediction"]]).astype(str).unique())
    cm = confusion_matrix(pred["expected_proof_state"].astype(str), pred["validated_prediction"].astype(str), labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, values_format=".2f")
    ax.set_xlabel("Predicted proof state")
    ax.set_ylabel("True proof state")
    ax.set_title("Enhanced normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig6_confusion_matrix.png"), dpi=300)
    plt.close(fig)


def make_ablation(fig_dir, ablation_path):
    table = pd.read_csv(ablation_path)
    x = table["unsupported_claim_rate"].astype(float)
    y = table["macro_f1"].astype(float)
    size = (table["evidence_completeness_score"].astype(float) * 500).clip(lower=80)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=size, alpha=0.75)
    for _, row in table.iterrows():
        ax.annotate(str(row["model_variant"]), (row["unsupported_claim_rate"], row["macro_f1"]), fontsize=8)
    ax.set_xlabel("Unsupported claim rate")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Ablation Pareto map")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig7_ablation_pareto.png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    ensure_dir(args.figures)
    pred = load_prediction_table(args.data, args.out)

    metrics_path = os.path.join(args.out, "metrics", "table_4_model_performance.csv")
    if os.path.exists(metrics_path):
        make_performance_heatmap(metrics_path, args.figures)

    make_rule_family_heatmap(pred, args.figures)
    make_calibration(pred, args.figures)
    make_risk_coverage(pred, args.figures)
    make_confusion(pred, args.figures)

    ablation_path = os.path.join(args.out, "metrics", "ablation_results.csv")
    if os.path.exists(ablation_path):
        make_ablation(args.figures, ablation_path)


if __name__ == "__main__":
    main()
