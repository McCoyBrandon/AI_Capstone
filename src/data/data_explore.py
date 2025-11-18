"""
Generate data-exploration figures for the AI4I predictive-maintenance dataset.

Outputs:
  - src/data/figures/[save_name]/...

Figures:
  1) Feature histograms (overall)
  2) Feature histograms (per class)
  3) Boxplots by class
  4) Correlation heatmap (numeric features)
  5) Class counts: Original vs SMOTE (if --resample smote)
  6) Class distribution bars (pre/post resample)

Usage:
  python -m src.data.data_explore --csv src/data/ai4i2020.csv
  python -m src.data.data_explore --csv src/data/ai4i2020.csv --resample smote --save_name smote_viz
"""

import os
import argparse
import numpy as np

# Project imports
from src.data.preprocess import read_ai4i_csv, encode_features, build_targets,prepare_datasets, NUM_COLS, CLASS_NAMES

from src.utils.visualizations import (
    plot_feature_histograms, plot_boxplots_by_class, plot_correlation_heatmap,
    plot_pre_post_smote_counts, plot_class_distribution
)

def ensure_dirs(save_name: str):
    figs_dir = os.path.join("src", "data", "figures", save_name)
    os.makedirs(figs_dir, exist_ok=True)
    return figs_dir

def main():
    p = argparse.ArgumentParser(description="Create data exploration figures.")
    p.add_argument("--csv", required=True, help="Path to AI4I-style CSV.")
    p.add_argument("--save_name", default="default", help="Subfolder name under src/data/figures/")
    p.add_argument("--normalize", default="zscore", choices=["zscore","minmax"],
                   help="Normalization for prepare_datasets (SMOTE).")
    p.add_argument("--resample", default=None, choices=[None, "smote"],
                   help="If 'smote', also plot pre vs post counts.")
    p.add_argument("--smote_k_neighbors", type=int, default=5)
    p.add_argument("--smote_random_state", type=int, default=42)
    p.add_argument("--smote_sampling", default="auto")
    p.add_argument("--max_cols", type=int, default=4)
    args = p.parse_args()

    # Output folder
    figs_dir = ensure_dirs(args.save_name)

    # ---------- Load dataset ----------
    df = read_ai4i_csv(args.csv)
    y_bin, y_cls = build_targets(df)
    X_num, cat_idx, type_vocab = encode_features(df)

    # ---------- 1) Feature histograms (overall) ----------
    fig = plot_feature_histograms(
        X_num, NUM_COLS,
        suptitle="Feature Histograms (All Samples)"
    )
    fig.savefig(os.path.join(figs_dir, "01_feature_histograms_all.png"),
                dpi=300, bbox_inches="tight")

    # ---------- 2) Feature histograms (per class) ----------
    fig = plot_feature_histograms(
        X_num, NUM_COLS,
        y=y_cls, class_names=CLASS_NAMES,
        per_class=True, max_cols=args.max_cols,
        suptitle="Feature Histograms per Failure Type"
    )
    fig.savefig(os.path.join(figs_dir, "02_feature_histograms_per_class.png"),
                dpi=300, bbox_inches="tight")

    # ---------- 3) Boxplots by class ----------
    fig = plot_boxplots_by_class(
        X_num, NUM_COLS, y_cls, CLASS_NAMES,
        max_cols=min(args.max_cols, 3),
        suptitle="Feature Distributions by Class (Boxplots)"
    )
    fig.savefig(os.path.join(figs_dir, "03_boxplots_by_class.png"),
                dpi=300, bbox_inches="tight")

    # ---------- 4) Correlation heatmap ----------
    fig = plot_correlation_heatmap(
        X_num, NUM_COLS,
        title="Feature Correlation (Pearson)"
    )
    fig.savefig(os.path.join(figs_dir, "04_correlation_heatmap.png"),
                dpi=300, bbox_inches="tight")

    # ---------- 5) Class counts: Original vs SMOTE ----------
    if args.resample == "smote":
        payload = prepare_datasets(
            csv_path=args.csv,
            normalize=args.normalize,
            resample="smote",
            smote_k_neighbors=args.smote_k_neighbors,
            smote_random_state=args.smote_random_state,
            smote_sampling=args.smote_sampling,
        )

        pre_counts  = payload["pre_counts"]
        post_counts = payload["post_counts"]

        # 5a) Pre vs Post SMOTE
        fig = plot_pre_post_smote_counts(
            pre_counts=pre_counts,
            post_counts=post_counts,
            class_names=CLASS_NAMES,
            title="Class Counts: Original vs SMOTE"
        )
        fig.savefig(os.path.join(figs_dir, "05a_pre_post_smote_counts.png"),
                    dpi=300, bbox_inches="tight")

        # 5b) Class distribution pre/post
        y_tr_pre  = payload["y_tr_labels_pre_hist"]
        y_tr_post = payload["y_tr_labels_for_hist"]

        fig = plot_class_distribution(
            y_tr_pre, class_names=CLASS_NAMES, title="Train Labels (Before Resample)"
        )
        fig.savefig(os.path.join(figs_dir, "05b_train_labels_pre_resample.png"),
                    dpi=300, bbox_inches="tight")

        fig = plot_class_distribution(
            y_tr_post, class_names=CLASS_NAMES, title="Train Labels (After Resample)"
        )
        fig.savefig(os.path.join(figs_dir, "05c_train_labels_post_resample.png"),
                    dpi=300, bbox_inches="tight")

    print(f"[data_explore] Figures saved to: {figs_dir}")

if __name__ == "__main__":
    main()
