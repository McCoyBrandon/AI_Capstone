"""
Reusable visualization utilities for AI4I/TinyTabTransformer experiments.

- Minimal deps: NumPy + Matplotlib; scikit-learn only for ROC/PR (optional).
- Functions return figures for the caller to save/log/close.
- TB helpers let you push figures into TensorBoard succinctly.

Example:
    from src.utils.visualizations import (
        plot_pre_post_smote_counts, plot_class_distribution, plot_confusion_matrix,
        plot_training_curves, plot_roc_pr_curves, plot_feature_histograms,
        plot_boxplots_by_class, plot_correlation_heatmap, plot_feature_importance,
        plot_hyperparam_heatmap, plot_embedding_2d,
        tb_log_figure
    )
"""

from typing import List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.metrics import roc_curve, auc,precision_recall_curve, average_precision_score
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False
from matplotlib.colors import LogNorm
    
# Helper Functions
def _as_np(x: Union[np.ndarray, Sequence]) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.asarray(x)

def _same_len(*arrs: Sequence):
    n = len(arrs[0])
    for a in arrs:
        if len(a) != n:
            raise ValueError("Input arrays must have the same length.")

def _auto_bins(x: np.ndarray, max_bins: int = 40) -> int:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return min(5, max_bins)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return min(10, max_bins)
    bw = 2 * iqr / (x.size ** (1/3))
    if bw <= 0:
        return min(10, max_bins)
    return int(np.clip(np.ceil((x.max() - x.min()) / bw), 5, max_bins))


# Functions for distributions / SMOTE
def plot_class_distribution(y: Sequence[int],
                            class_names: Optional[List[str]] = None,
                            title: str = "Class Distribution"):
    y = _as_np(y).astype(int)

    if class_names is None:
        # Fall back to unique labels present
        classes = np.unique(y)
        counts = np.bincount(y - y.min())  # loose fallback
        xticks = np.arange(len(classes))
        xticklabels = [str(c) for c in classes]
    else:
        C = len(class_names)
        if (y < 0).any() or (y >= C).any():
            raise ValueError(
                f"y contains label outside [0, {C-1}] range required by class_names."
            )
        counts = np.bincount(y, minlength=C)
        xticks = np.arange(C)
        xticklabels = class_names

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xticks, counts)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_pre_post_smote_counts(pre_counts: Sequence[int], post_counts: Sequence[int],
                               class_names: List[str],
                               title: str = "Class Counts: Original vs SMOTE"):
    pre_counts = _as_np(pre_counts)
    post_counts = _as_np(post_counts)
    if pre_counts.shape != post_counts.shape:
        raise ValueError("pre_counts and post_counts must have same shape.")
    n = pre_counts.size
    idx = np.arange(n)
    w = 0.42
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(idx - w/2, pre_counts,  width=w, label="Original")
    ax.bar(idx + w/2, post_counts, width=w, label="SMOTE")
    ax.set_xticks(idx)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

# Functions for data exploration
def plot_feature_histograms(X: np.ndarray, feature_names: List[str],
                            y: Optional[Sequence[int]] = None,
                            class_names: Optional[List[str]] = None,
                            bins: Optional[int] = None, per_class: bool = False,
                            max_cols: int = 4, suptitle: Optional[str] = "Feature Histograms"):
    X = _as_np(X)
    D = X.shape[1]
    cols = min(max_cols, D)
    rows = math.ceil(D / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.1*cols, 3.2*rows))
    axs = np.array(axs).reshape(-1)

    if y is not None:
        y = _as_np(y)
        _same_len(y, X[:, 0])

    for j in range(D):
        ax = axs[j]
        xj = X[:, j]
        b = bins or _auto_bins(xj)
        if per_class and y is not None:
            for c in np.unique(y):
                label = class_names[c] if class_names is not None else str(c)
                ax.hist(xj[y == c], bins=b, alpha=0.5, label=label)
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.hist(xj, bins=b)
        ax.set_title(feature_names[j], fontsize=10)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)

    for k in range(D, rows*cols):
        fig.delaxes(axs[k])

    if suptitle:
        fig.suptitle(suptitle, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def plot_boxplots_by_class(X: np.ndarray, feature_names: List[str],
                           y: Sequence[int], class_names: Optional[List[str]] = None,
                           max_cols: int = 3,
                           suptitle: str = "Feature Distributions by Class (Boxplots)"):
    X = _as_np(X); y = _as_np(y); _same_len(y, X[:, 0])
    D = X.shape[1]; classes = np.unique(y)
    cols = min(max_cols, D); rows = math.ceil(D / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.6*cols, 3.6*rows))
    axs = np.array(axs).reshape(-1)

    for j in range(D):
        ax = axs[j]
        data = [X[y == c, j] for c in classes]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(np.arange(1, len(classes)+1))
        ax.set_xticklabels([(class_names[c] if class_names else str(c)) for c in classes])
        ax.set_title(feature_names[j], fontsize=10)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)

    for k in range(D, rows*cols):
        fig.delaxes(axs[k])

    fig.suptitle(suptitle, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def plot_correlation_heatmap(X: np.ndarray, feature_names: List[str],
                             title: str = "Feature Correlation (Pearson)"):
    X = _as_np(X)
    C = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(C, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_yticklabels(feature_names)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.set_ylabel("Correlation", rotation=90)
    fig.tight_layout()
    return fig


# Functions for training & evaluation
def plot_training_curves(epochs: Sequence[int],
                         train_loss: Sequence[float], val_loss: Sequence[float],
                         train_metric: Optional[Sequence[float]] = None,
                         val_metric: Optional[Sequence[float]] = None,
                         metric_name: str = "Macro-F1",
                         suptitle: str = "Training Dynamics"):
    epochs = _as_np(epochs); train_loss = _as_np(train_loss); val_loss = _as_np(val_loss)
    if (train_metric is None) ^ (val_metric is None):
        raise ValueError("Provide both train_metric and val_metric, or neither.")
    ncols = 2 if train_metric is not None else 1
    fig, axs = plt.subplots(1, ncols, figsize=(6.2*ncols, 4.2))
    if ncols == 1: axs = np.array([axs])

    ax = axs[0]
    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss,   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss vs Epoch")
    ax.grid(True, linestyle=":", linewidth=0.5); ax.legend(frameon=False)

    if ncols == 2:
        ax = axs[1]
        ax.plot(epochs, _as_np(train_metric), label=f"Train {metric_name}")
        ax.plot(epochs, _as_np(val_metric),   label=f"Val {metric_name}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(metric_name); ax.set_title(f"{metric_name} vs Epoch")
        ax.grid(True, linestyle=":", linewidth=0.5); ax.legend(frameon=False)

    fig.suptitle(suptitle, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names,
    title: str = "Confusion Matrix",
    cmap: str = "magma",               # nice dark-low → bright-high gradient
    log_scale: bool = False,           # emphasize small-but-nonzero cells
    robust_clip: tuple = (1, 99),      # percentile clip using *focused* cells
    scale_exclude_cells: list = None,  # e.g., [(0, 0)] to exclude NoFailure→NoFailure
    annotate: bool = True
):
    """
    Confusion-matrix heatmap with color scaling based on a focused subset of cells.
    - Uses raw counts (no normalization).
    - Color limits (vmin/vmax) are computed from all cells EXCEPT those listed in
      'scale_exclude_cells', so outliers (e.g., NoFailure→NoFailure) don't crush contrast.
    - You still see every cell and its count; only the color scale ignores excluded cells.
    """
    M = np.asarray(cm, dtype=float)
    C = M.shape[0]

    # Build a mask of cells to use for scaling
    mask = np.ones_like(M, dtype=bool)
    if scale_exclude_cells:
        for (r, c) in scale_exclude_cells:
            if 0 <= r < C and 0 <= c < C:
                mask[r, c] = False

    focus_vals = M[mask]
    # if all focus values are equal/empty, fallback to all non-excluded
    valid_focus = focus_vals[np.isfinite(focus_vals)]
    if valid_focus.size == 0:
        valid_focus = M.ravel()

    lo_pct, hi_pct = robust_clip
    vmin = np.percentile(valid_focus, lo_pct) if valid_focus.size else M.min()
    vmax = np.percentile(valid_focus, hi_pct) if valid_focus.size else M.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, vmin + 1e-6)

    # Plot
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    if log_scale:
        eps = 1e-12
        display = np.where(M <= 0, eps, M)
        im = ax.imshow(display, aspect="auto",
                       norm=LogNorm(vmin=max(vmin, eps), vmax=max(vmax, eps)),
                       cmap=cmap)
        cbar_label = "Count (log scale)"
    else:
        im = ax.imshow(M, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        cbar_label = "Count"

    ax.set_xticks(np.arange(C)); ax.set_yticks(np.arange(C))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)

    if annotate:
        mid = (vmin + vmax) / 2.0
        for i in range(C):
            for j in range(C):
                val = int(M[i, j])
                ax.text(
                    j, i, f"{val}",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if M[i, j] > mid else "black"
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.set_ylabel(cbar_label, rotation=90)
    fig.tight_layout()
    return fig

def plot_roc_pr_curves(
    y_true_onehot: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    suptitle: str = "ROC & Precision-Recall (One-vs-Rest)"
):
    """
    Plots per-class ROC and PR curves.
    - Skips classes with zero positives or zero negatives (ROC/PR undefined).

    y_true_onehot: shape (N, C) of {0,1}
    y_prob:        shape (N, C) of predicted probabilities
    """
    # Check sklearn availability
    if not ("_SKLEARN_AVAILABLE" in globals() and _SKLEARN_AVAILABLE):
        raise ImportError("scikit-learn is required for plot_roc_pr_curves.")

    y_true_onehot = _as_np(y_true_onehot)
    y_prob = _as_np(y_prob)
    N, C = y_true_onehot.shape
    if y_prob.shape != (N, C):
        raise ValueError(f"y_prob shape {y_prob.shape} must match y_true_onehot {y_true_onehot.shape}")

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
    ax_roc, ax_pr = axs

    plotted_any = False
    skipped = []

    for c in range(C):
        y_c = y_true_onehot[:, c]
        p_c = y_prob[:, c]
        pos = int(y_c.sum())
        neg = int((1 - y_c).sum())

        # Skip if class is absent or degenerate for ROC/PR
        if pos == 0 or neg == 0:
            skipped.append(class_names[c] if class_names else str(c))
            continue

        # ROC
        fpr, tpr, _ = roc_curve(y_c, p_c)
        ax_roc.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc(fpr, tpr):.3f})")

        # PR
        precision, recall, _ = precision_recall_curve(y_c, p_c)
        ap = average_precision_score(y_c, p_c)
        ax_pr.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})")

        plotted_any = True

    # ROC panel
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curves")
    ax_roc.grid(True, linestyle=":", linewidth=0.5)
    ax_roc.legend(frameon=False, fontsize=8)

    # PR panel
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.grid(True, linestyle=":", linewidth=0.5)
    ax_pr.legend(frameon=False, fontsize=8)

    # Note skipped classes
    if skipped:
        suptitle = f"{suptitle}\n(skipped: {', '.join(skipped)})"

    if not plotted_any:
        plt.close(fig)
        raise ValueError("No valid classes to plot ROC/PR (all classes were absent or degenerate).")

    fig.suptitle(suptitle, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

# Functions for hyperparam & interpretability
def plot_hyperparam_heatmap(scores_matrix: np.ndarray,
                            x_labels: List[str], y_labels: List[str],
                            title: str = "Validation Macro-F1 by Hyperparameters",
                            x_label: str = "X (e.g., LR)", y_label: str = "Y (e.g., D_MODEL)"):
    M = _as_np(scores_matrix)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(M, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(x_labels))); ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right"); ax.set_yticklabels(y_labels)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85); cbar.ax.set_ylabel("Score", rotation=90)
    fig.tight_layout()
    return fig

def plot_feature_importance(importances: Sequence[float], feature_names: List[str],
                            top_k: Optional[int] = None, title: str = "Feature Importance"):
    imp = _as_np(importances)
    names = list(feature_names)
    idx = np.argsort(imp) if top_k is None else np.argsort(imp)[-top_k:]
    imp = imp[idx]; names = [names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(7.0, 0.36*len(names)+1.5))
    ax.barh(np.arange(len(names)), imp)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Importance"); ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    return fig

def plot_embedding_2d(embedding_2d: np.ndarray, y: Optional[Sequence[int]] = None,
                      class_names: Optional[List[str]] = None, title: str = "2D Embedding"):
    Z = _as_np(embedding_2d)
    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    if y is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=12, alpha=0.85)
    else:
        y = _as_np(y)
        for c in np.unique(y):
            lbl = class_names[c] if class_names is not None else str(c)
            ax.scatter(Z[y == c, 0], Z[y == c, 1], s=12, alpha=0.85, label=lbl)
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    return fig


# TensorBoard convenience
def tb_log_figure(writer, tag: str, fig, step: int, close: bool = True):
    # Add a Matplotlib figure to a TensorBoard writer, optionally closing it.
    writer.add_figure(tag, fig, step)
    if close:
        plt.close(fig)
