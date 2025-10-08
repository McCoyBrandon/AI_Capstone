"""
Run a pretrained TinyTabTransformer on the test split and report metrics.
Reuses code style from train.py and data_loader.py.
"""

# Required Packages (same style as your files)
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import os, json, argparse

# Additional Reference code
from src.models.transformer_class import TinyTabTransformer                   # model class
from src.data.data_loader import (                                            # prepared data & metadata
    tr_dl, te_dl, class_weights, class_counts, type_vocab,
    mean, std, NUM_COLS, CLASS_NAMES, N_CLASSES, DEVICE, D_MODEL, NHEAD, TARGET
)

### Starting Variables
parser = argparse.ArgumentParser(description="Evaluate a pretrained model on the test set.")
parser.add_argument("--ckpt", type=str, default="runs/ai4i_run_1/model.ckpt",
                    help="Path to checkpoint saved by train.py")
parser.add_argument("--out", type=str, default="runs/ai4i_run_1/eval_pretrained_metrics.json",
                    help="Where to save evaluation metrics JSON")
args = parser.parse_args()

CKPT_PATH = args.ckpt
OUT_PATH  = args.out

### Load Checkpoint (weights + meta)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
meta = ckpt["meta"]

# Sanity: align dimensions with checkpoint meta
type_vocab = int(meta["type_vocab"])
d_model    = int(meta["d_model"])
nhead      = int(meta["nhead"])
num_cols   = meta["num_cols"]
class_names= meta["class_names"]
n_classes  = len(class_names)

assert len(NUM_COLS) == len(num_cols), "NUM_COLS mismatch vs checkpoint."
assert n_classes == N_CLASSES, "N_CLASSES mismatch vs checkpoint."

### Rebuild Model & Load Weights
model = TinyTabTransformer(
    n_num=len(NUM_COLS),
    type_vocab=type_vocab,
    d_model=d_model,
    nhead=nhead
).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

### Evaluation on Test Set (same style as train.py)
all_logits, all_y = [], []
with torch.no_grad():
    for xb_num, xb_type, yb in te_dl:
        xb_num  = xb_num.to(DEVICE)
        xb_type = xb_type.to(DEVICE)
        logits  = model(xb_num, xb_type)           # [B, C]
        all_logits.append(logits.cpu())
        all_y.append(yb.cpu())

all_logits = torch.cat(all_logits, dim=0).numpy()  # [Nte, C]
all_y      = torch.cat(all_y, dim=0).numpy()       # [Nte]
probs      = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
preds      = probs.argmax(axis=1)

# Core metrics (JSON-friendly)
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

# Terminal prints (same look/feel as train.py)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}")
print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
print("\nClassification report:")
print(classification_report(all_y, preds, labels=list(range(n_classes)), target_names=class_names, zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_y, preds, labels=list(range(n_classes))))

# AUROC (warning-safe for degenerate classes)
Y_true_ovr = np.zeros((all_y.shape[0], n_classes), dtype=np.int64)
Y_true_ovr[np.arange(all_y.shape[0]), all_y] = 1

print("\nPer-class AUROC:")
per_class_auc = {}
for i, name in enumerate(class_names):
    try:
        auc_i = roc_auc_score(Y_true_ovr[:, i], probs[:, i])
        per_class_auc[name] = float(auc_i)
        print(f"{name:>12s}: {auc_i:.4f}")
    except ValueError:
        per_class_auc[name] = None
        print(f"{name:>12s}: N/A (absent or degenerate in test)")

# Macro/micro/weighted AUROC
try:
    macro_auc    = float(roc_auc_score(Y_true_ovr, probs, multi_class="ovr", average="macro"))
    micro_auc    = float(roc_auc_score(Y_true_ovr, probs, multi_class="ovr", average="micro"))
    weighted_auc = float(roc_auc_score(Y_true_ovr, probs, multi_class="ovr", average="weighted"))
    print(f"\nMacro AUROC:    {macro_auc:.4f}")
    print(f"Micro AUROC:    {micro_auc:.4f}")
    print(f"Weighted AUROC: {weighted_auc:.4f}")
except ValueError:
    macro_auc = micro_auc = weighted_auc = None
    print("\nMacro/Micro/Weighted AUROC unavailable (degenerate labels).")

# Binary AUROC (any failure vs. no failure)
p_any_fail  = probs[:, 1:].sum(axis=1)               # P(failure)
y_binary_te = (all_y != 0).astype(int)
try:
    bin_auc = float(roc_auc_score(y_binary_te, p_any_fail))
    print(f"\nBinary AUROC (TARGET='{meta['target']}'): {bin_auc:.4f}")
except ValueError:
    bin_auc = None
    print(f"\nBinary AUROC (TARGET='{meta['target']}') unavailable.")

# Save metrics
metrics.update({
    "per_class_auc": per_class_auc,
    "macro_auc": macro_auc,
    "micro_auc": micro_auc,
    "weighted_auc": weighted_auc,
    "binary_auc": bin_auc,
})
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved eval metrics to: {OUT_PATH}")
