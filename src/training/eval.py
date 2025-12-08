"""
Evaluation script for a pretrained TinyTabTransformer on AI4I-style datasets.

This module loads a saved checkpoint, rebuilds the model from its metadata,
and evaluates on the test split produced by `prepare_datasets()` in
`src.data.preprocess`. It reports core metrics (accuracy, balanced accuracy,
macro/weighted F1), AUROC variants (per-class, macro/micro/weighted, and
binary "any failure" vs "no failure"), and writes a JSON with results.

CSV input is dynamically loaded via `prepare_datasets()`.

Main Function:
    (executed when run directly)
    - Loads checkpoint (weights + meta), rebuilds the model
    - Runs evaluation on test dataloader
    - Prints a readable summary and saves metrics to JSON

Core Components:
    Model:
        TinyTabTransformer from src.models.transformer_class
    Data:
        Prepare_datasets() from src.data.preprocess
    Metrics:
        Shared metric and plotting helpers from src.utils.metrics

Outputs:
    - Metrics JSON: runs/<RUN_NAME>/eval_pretrained_metrics.json
    - Console summary: accuracy / F1 / AUROC / confusion matrix

Terminal run:
    python -m src.training.eval --ckpt runs/ai4i_run_1/model.ckpt --out  runs/ai4i_run_1/eval_pretrained_metrics.json
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
from src.data.preprocess import prepare_datasets, CLASS_NAMES, N_CLASSES, DEVICE
from src.utils.metrics import compute_core_metrics, compute_auroc_metrics, confusion_matrix_figure

# Data Processing
CSV_PATH = "src/data/ai4i2020.csv"   # same headers
data = prepare_datasets(csv_path=CSV_PATH, batch=256, test_size=0.20, random_state=42)

te_dl       = data["te_dl"]
type_vocab  = data["type_vocab"]
NUM_COLS    = data["NUM_COLS"]
CLASS_NAMES = data["CLASS_NAMES"]
N_CLASSES   = data["N_CLASSES"]
D_MODEL     = data["D_MODEL"]
NHEAD       = data["NHEAD"]
TARGET      = data["TARGET"]

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

# Align dimensions with checkpoint meta
# Default to main_save parameters for backwards compatibility
type_vocab = int(meta["type_vocab"])
d_model        = int(meta.get("d_model", 128))
nhead          = int(meta.get("nhead", 2))
dim_feedforward = int(meta.get("dim_feedforward", 128))
dropout         = float(meta.get("dropout", 0.0))
num_layers      = int(meta.get("num_layers", 2))   # <- default 2 for old ckpt
num_cols   = meta["num_cols"]
class_names= meta["class_names"]
n_classes  = len(class_names)

assert len(NUM_COLS) == len(num_cols), "NUM_COLS mismatch vs checkpoint."
assert n_classes == N_CLASSES, "N_CLASSES mismatch vs checkpoint."

# Architecture hyperparameters (with safe defaults for older checkpoints)
dim_feedforward = int(meta.get("dim_feedforward", 128))
dropout        = float(meta.get("dropout", 0.0))
num_layers     = int(meta.get("num_layers", 2))

# Rebuild Model & Load Weights
model = TinyTabTransformer(
    n_num=len(NUM_COLS),
    type_vocab=int(meta["type_vocab"]),
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    num_layers=num_layers,
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
core = compute_core_metrics(all_y, preds, class_names)
aucs = compute_auroc_metrics(all_y, probs, class_names)

print(f"Test accuracy: {core['accuracy']:.4f}")
print(f"Macro F1: {core['macro_f1']:.4f} | Weighted F1: {core['weighted_f1']:.4f}")
print(f"Balanced accuracy: {core['balanced_accuracy']:.4f}")
print("\nClassification report:")
from sklearn.metrics import classification_report as _cr
print(_cr(all_y, preds, labels=list(range(n_classes)),
          target_names=class_names, zero_division=0))
import numpy as _np
print("Confusion matrix (rows=true, cols=pred):")
print(_np.array(core["confusion_matrix"]))

print("\nPer-class AUROC:")
for k, v in aucs["per_class_auc"].items():
    print(f"{k:>12s}: {'N/A' if v is None else f'{v:.4f}'}")
for k in ("macro_auc", "micro_auc", "weighted_auc", "binary_auc"):
    val = aucs[k]
    print(f"{k}: {'N/A' if val is None else f'{val:.4f}'}")

# Save JSON
out_payload = {**core, **aucs}
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out_payload, f, indent=2)
print(f"\nSaved eval metrics to: {OUT_PATH}")


os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved eval metrics to: {OUT_PATH}")
