# src/utils/metrics.py
"""
Shared metric utilities for TinyTabTransformer experiments.

Functions:
    compute_core_metrics(all_y, preds, class_names)
    compute_auroc_metrics(all_y, probs, class_names, nofail_label="NoFailure")
    confusion_matrix_figure(cm, class_names)

Notes:
    - Robust to degenerate label cases (e.g., AUROC when a class has no positives).
    - Returns JSON-serializable numbers (floats or None).
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def compute_core_metrics(all_y: np.ndarray,
                         preds: np.ndarray,
                         class_names: list) -> Dict:
    n_classes = len(class_names)
    metrics = {
        "accuracy":          float(accuracy_score(all_y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_y, preds)),
        "macro_f1":          float(f1_score(all_y, preds, average="macro", zero_division=0)),
        "weighted_f1":       float(f1_score(all_y, preds, average="weighted", zero_division=0)),
        "support_per_class": dict(zip(class_names, [int(x) for x in np.bincount(all_y, minlength=n_classes)])),
        "report":            classification_report(
                                all_y, preds,
                                labels=list(range(n_classes)),
                                target_names=class_names,
                                zero_division=0,
                                output_dict=True
                             ),
        "confusion_matrix":  confusion_matrix(all_y, preds, labels=list(range(n_classes))).tolist(),
    }
    return metrics


def compute_auroc_metrics(all_y: np.ndarray,
                          probs: np.ndarray,
                          class_names: list,
                          nofail_label: str = "NoFailure") -> Dict:
    # Compute per-class AUROCs, macro/micro/weighted AUROC, and binary AUROC.
    n_classes = len(class_names)
    y_ovr = np.zeros((all_y.shape[0], n_classes), dtype=np.int64)
    y_ovr[np.arange(all_y.shape[0]), all_y] = 1

    per_class_auc = {}
    for i, name in enumerate(class_names):
        try:
            per_class_auc[name] = float(roc_auc_score(y_ovr[:, i], probs[:, i]))
        except ValueError:
            per_class_auc[name] = None

    # Multi-class AUROCs 
    try:
        macro_auc    = float(roc_auc_score(y_ovr, probs, multi_class="ovr", average="macro"))
        micro_auc    = float(roc_auc_score(y_ovr, probs, multi_class="ovr", average="micro"))
        weighted_auc = float(roc_auc_score(y_ovr, probs, multi_class="ovr", average="weighted"))
    except ValueError:
        macro_auc = micro_auc = weighted_auc = None

    # Binary AUROC: any failure vs no-failure
    try:
        nofail_idx = class_names.index(nofail_label)
    except ValueError:
        nofail_idx = 0
    y_bin = (all_y != nofail_idx).astype(int)
    p_bin = 1.0 - probs[:, nofail_idx]  # P(any failure) = 1 - P(NoFailure)
    try:
        binary_auc = float(roc_auc_score(y_bin, p_bin))
    except ValueError:
        binary_auc = None

    return {
        "per_class_auc": per_class_auc,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "weighted_auc": weighted_auc,
        "binary_auc": binary_auc,
    }


def confusion_matrix_figure(cm: np.ndarray, class_names: list) -> Figure:
    # Return a Matplotlib Figure with a labeled confusion matrix
    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)), yticks=range(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label', xlabel='Predicted label', title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

