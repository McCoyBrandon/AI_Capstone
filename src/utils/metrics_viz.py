"""
Create visualizations from a saved metrics.json produced by TinyTabTransformer grid search training.

Usage:
    python -m src.utils.metrics_viz
    python -m src.utils.metrics_viz --metrics_path runs/selected_checkpoint/metrics.json
    python -m src.utils.metrics_viz --metrics_path .runs.selected_checkpoint.metrics.json --out_dir .runs.selected_checpoint
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.utils.visualizations import plot_confusion_matrix  


def load_metrics(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"metrics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_class_names(metrics: dict):
    """
    Infer ordered class names from metrics.json.
    Prefer 'support_per_class' keys
    """
    if "support_per_class" in metrics:
        return list(metrics["support_per_class"].keys())
    if "per_class_auc" in metrics:
        return list(metrics["per_class_auc"].keys())
    raise ValueError("Could not infer class names from metrics.json.")


def make_confusion_matrix_fig(metrics: dict, out_dir: str):
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    class_names = infer_class_names(metrics)

    fig = plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title="Confusion Matrix",
        cmap="magma_r",
        log_scale=False,
        robust_clip=(1, 99),
        scale_exclude_cells=[(0, 0)], 
        annotate=True,
    )

    out_path = os.path.join(out_dir, "confusion_matrix_from_metrics.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def make_per_class_f1_fig(metrics: dict, out_dir: str):
    """
    Simple horizontal bar chart of per-class F1 scores using 'report' section.
    """
    report = metrics.get("report", {})
    class_names = []
    f1_scores = []

    # Only include real classes, skip "accuracy", "macro avg", "weighted avg"
    for key, val in report.items():
        if key in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(val, dict):
            continue
        if "f1-score" not in val:
            continue
        class_names.append(key)
        f1_scores.append(val["f1-score"])

    if not class_names:
        print("[warning] No per-class f1-score found in 'report'; skipping F1 plot.")
        return

    y_pos = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(7.0, 0.4 * len(class_names) + 1.5))
    ax.barh(y_pos, f1_scores)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("F1-score")
    ax.set_title("Per-class F1 from metrics.json")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "per_class_f1_from_metrics.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize metrics from a saved metrics.json."
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="runs/selected_checkpoint/metrics.json",
        help="Path to metrics.json (or .runs.selected_checkpoint.metrics).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save figures (default: same folder as metrics file).",
    )
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_path)

    # Default output dir = alongside metrics file
    if args.out_dir is None:
        base_dir = os.path.dirname(os.path.abspath(args.metrics_path))
        out_dir = os.path.join(base_dir, "figs_from_metrics")
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] Saving figures to: {out_dir}")

    # 1) Confusion matrix using your existing helper
    if "confusion_matrix" in metrics:
        make_confusion_matrix_fig(metrics, out_dir)
    else:
        print("[warn] 'confusion_matrix' not found in metrics; skipping.")

    # 2) Per-class F1 bar chart using 'report'
    if "report" in metrics:
        make_per_class_f1_fig(metrics, out_dir)
    else:
        print("[warn] 'report' not found in metrics; skipping per-class F1 plot.")

    # You can add more plots later (e.g., AUROC bars, etc.)


if __name__ == "__main__":
    main()
