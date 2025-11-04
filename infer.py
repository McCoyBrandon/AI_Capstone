"""
Inference script for a pretrained TinyTabTransformer on AI4I-style datasets.

This module loads a trained model checkpoint and performs inference on the 
provided CSV dataset using the same preprocessing pipeline as training. It 
outputs evaluation metrics, prints model performance summaries, and saves a 
CSV of all test samples predicted to experience a machine failure (with 
associated probabilities and failure types).

CSV input is dynamic and the path is provided at runtime via CLI.

Main Function:
    (executed when run directly)
    - Loads pretrained checkpoint (weights + meta)
    - Rebuilds model using stored architecture and normalization stats
    - Preprocesses the provided CSV file using shared functions from
      `src.data.preprocess`
    - Evaluates predictions and computes metrics via
      `src.utils.metrics`
    - Exports:
        * Metrics JSON
        * Flagged failures CSV listing predicted failed units

Core Components:
    Model:
        TinyTabTransformer (imported from src.models.transformer_class)
    Data:
        read_ai4i_csv(), build_targets(), encode_features() from src.data.preprocess
    Metrics:
        compute_core_metrics(), compute_auroc_metrics() from src.utils.metrics
    Outputs:
        - JSON metrics summary
        - CSV of predicted failures (row index, predicted class, confidence)

Optional Features:
    - `--product-col` argument allows specifying a column (e.g., "Product ID")
      to include in the flagged-failures CSV for traceability.
    - Computes binary calibration metric (Brier Score) for P(any failure).

Outputs:
    - Metrics JSON:     (--out) e.g., runs/<RUN_NAME>/demo_metrics.json
    - Failures CSV:     (--failures-out) e.g., runs/<RUN_NAME>/flagged_failures.csv
    - Console Summary:  Accuracy, F1, Balanced Accuracy, AUROC per class

Terminal usage example:
    python -m src.training.infer \
        --ckpt runs/ai4i_run_1/model.ckpt \
        --csv  src/data/ai4i2020.csv \
        --out  runs/ai4i_run_1/infer_metrics.json \
        --failures-out runs/ai4i_run_1/flagged_failures.csv \
        [--product-col "Product ID"]
"""

# Required Packages (same style as your files)
import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, brier_score_loss
)

# Additional Reference code 
from src.models.transformer_class import TinyTabTransformer   # model class
from src.data.preprocess import read_ai4i_csv, build_targets, encode_features, standardize_train_test
from src.utils.metrics import compute_core_metrics, compute_auroc_metrics

# Setting up Terminal instructions when using --help
parser = argparse.ArgumentParser(description="Demo: evaluate a pretrained model and list predicted failures.")
parser.add_argument("--ckpt", type=str, default="runs/ai4i_run_1/model.ckpt",
                    help="Path to checkpoint saved by train.py (weights + meta)")
parser.add_argument("--csv", type=str, default="src/data/ai4i2020.csv",
                    help="Path to AI4I 2020 CSV (same file used in training)")
parser.add_argument("--product-col", type=str, default=None,
                    help="Column to use as product identifier in failures CSV (e.g., 'Product ID').")
parser.add_argument("--out", type=str, default="runs/ai4i_run_1/demo_metrics.json",
                    help="Where to save evaluation metrics JSON")
parser.add_argument("--failures-out", type=str, default="runs/ai4i_run_1/flagged_failures.json",
                    help="Optional JSON list of test-set machines predicted to fail")
parser.add_argument("--failures-csv", type=str, default="runs/ai4i_run_1/flagged_failures.csv",
                    help="CSV of test rows predicted to fail (product_id, binary, failure_type)")                    
args = parser.parse_args()

CKPT_PATH    = args.ckpt
CSV_PATH     = args.csv
OUT_PATH     = args.out
FAIL_LIST_OUT= args.failures_out
FAILURES_CSV = args.failures_csv

# Setting up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Variables, paths, and hyperparameters 
if not os.path.exists(CKPT_PATH):
    # friendly fallback to tests/ layout if provided path wasn't found
    alt = CKPT_PATH.replace("runs/", "tests/")
    if os.path.exists(alt):
        CKPT_PATH = alt

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
meta = ckpt["meta"]

CLASS_NAMES = meta["class_names"]              # ["NoFailure","TWF",...]
NUM_COLS    = meta["num_cols"]                 # numeric feature names (order matters)
CAT_COL     = meta["cat_col"]                  # "Type"
TARGET      = meta["target"]                   # "Machine failure"
FAILURE_COLS= meta["failure_cols"]             # ["TWF","HDF",...]
TYPE_VOCAB  = int(meta["type_vocab"])          # e.g., 3 for L/M/H
D_MODEL     = int(meta["d_model"])
NHEAD       = int(meta["nhead"])
N_CLASSES   = len(CLASS_NAMES)


# Read + validate schema, then build targets/features using shared helpers
df = read_ai4i_csv(CSV_PATH, validate_schema=True)
y_bin, y_cls      = build_targets(df)
X_num, cat_idx, _ = encode_features(df)

# Keep original row indices so we can map back to product key from source data
row_idx = np.arange(len(df))

# Recreate the same stratified split (random_state=42) used in data_loader.py
_, Xn_te, _, ty_te, _, y_te, _, yb_te, _, idx_te = train_test_split(
    X_num, cat_idx, y_cls, y_bin, row_idx,  # Added the row index which won't be used by the model, but can be used to backtrack to the productID
    test_size=0.20, random_state=42, stratify=y_cls   # Splitting hyperparameters
)


# Standardize numeric features using TRAIN statistics from checkpoint meta
mean = np.array(meta["standardize_mean"], dtype=np.float32)
std  = np.array(meta["standardize_std"],  dtype=np.float32) + 1e-8
Xn_te = (Xn_te - mean) / std

# Convert to tensors for the model
Xn_te_t = torch.tensor(Xn_te, dtype=torch.float32)
ty_te_t = torch.tensor(ty_te, dtype=torch.int64)
y_te_t  = torch.tensor(y_te,  dtype=torch.int64)

# Load model and weights
model = TinyTabTransformer(
    n_num=len(NUM_COLS),
    type_vocab=TYPE_VOCAB,
    d_model=D_MODEL,
    nhead=NHEAD,
).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Batched inference on test set
BATCH = 256
all_logits = []
with torch.no_grad():
    for i in range(0, len(Xn_te_t), BATCH):
        xb_num  = Xn_te_t[i:i+BATCH].to(DEVICE)
        xb_type = ty_te_t[i:i+BATCH].to(DEVICE)
        logits  = model(xb_num, xb_type)          # [B, C]
        all_logits.append(logits.cpu())

all_logits = torch.cat(all_logits, dim=0).numpy()      # [Nte, C]
probs      = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
preds      = probs.argmax(axis=1)

# Compute binary "any failure" probability and label
# Locate which index represents "NoFailure"
try:
    nofail_idx = CLASS_NAMES.index("NoFailure")
except ValueError:
    nofail_idx = 0  # fallback if label set differs

# Probability that a unit experiences *any* failure
p_any_fail  = 1.0 - probs[:, nofail_idx]        # P(any failure) = 1 - P(NoFailure)
# Binary truth labels: 1 = any failure, 0 = no failure
y_binary_te = (y_te != nofail_idx).astype(int)

# Shareable metrics (JSON)
core = compute_core_metrics(y_te, preds, CLASS_NAMES)
aucs = compute_auroc_metrics(y_te, probs, CLASS_NAMES)

print(f"Test accuracy: {core['accuracy']:.4f}")
print(f"Macro F1: {core['macro_f1']:.4f} | Weighted F1: {core['weighted_f1']:.4f}")
print(f"Balanced accuracy: {core['balanced_accuracy']:.4f}")

from sklearn.metrics import classification_report as _cr
print("\nClassification report:")
print(_cr(y_te, preds, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES, zero_division=0))

import numpy as _np
print("Confusion matrix (rows=true, cols=pred):")
print(_np.array(core["confusion_matrix"]))

print("\nPer-class AUROC:")
for k, v in aucs["per_class_auc"].items():
    print(f"{k:>12s}: {'N/A' if v is None else f'{v:.4f}'}")
for k in ("macro_auc", "micro_auc", "weighted_auc", "binary_auc"):
    val = aucs[k]
    print(f"{k}: {'N/A' if val is None else f'{val:.4f}'}")

metrics = {**core, **aucs}  # single payload to save

# Normalized confusion matrix
cm = confusion_matrix(y_te, preds, labels=list(range(N_CLASSES)))
row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
cm_norm = (cm / row_sums).tolist()
metrics["confusion_matrix_normalized"] = cm_norm
print("Normalized confusion matrix (rows=true):")
print(np.array(cm_norm))

# Per-class precision/recall/F1 as a CSV for quick viewing
_report = metrics["report"]

# Create a CSV with product ID, binary flag, and failure type for flagged rows
# Pick a product identifier column if present, used for more reboust ProductID callback for inference outputs.
prod_col = args.product_col or next((c for c in ["Product ID", "ProductID", "product_id", "productId"]
                                     if c in df.columns), None)
csv_rows = []
for i in range(len(preds)):
    c = int(preds[i])
    if c == 0:
        continue  # only predicted failures (classes 1..5)
    te_row = int(idx_te[i])  # original CSV row index
    prod_val = None
    if prod_col is not None:
        prod_val = str(df.iloc[te_row][prod_col])
    csv_rows.append({
        "csv_row": te_row,
        "product_id": prod_val,
        "predicted_binary": 1,
        "predicted_class": CLASS_NAMES[c],          # e.g., TWF/HDF/...
        "predicted_class_prob": float(probs[i, c]), # confidence for that class
        "pred_any_failure_prob": float(p_any_fail[i]),
    })

if csv_rows and FAILURES_CSV:
    out_dir = os.path.dirname(FAILURES_CSV) or "."
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(FAILURES_CSV, index=False)
    print(f"Saved flagged failures CSV to: {FAILURES_CSV}")

# classification_report(output_dict=True) has a dict-of-dicts; flatten into DataFrame
try:
    import pandas as _pd
    rep_df = _pd.DataFrame(_report).T
    rep_csv_path = (OUT_PATH[:-5] if OUT_PATH.lower().endswith('.json') else OUT_PATH) + "_report.csv"
    rep_df.to_csv(rep_csv_path, index=True)
    print(f"Saved per-class report CSV to: {rep_csv_path}")
    metrics["report_csv"] = rep_csv_path
except Exception as _e:
    print(f"[warn] could not write report CSV: {_e}")

# Brier score (probability calibration) for binary failure vs no-failure
try:
    brier = float(brier_score_loss(y_binary_te, p_any_fail))
    metrics["brier_any_failure"] = brier
    print(f"Brier score (any failure vs none): {brier:.6f}")
except Exception as _e:
    metrics["brier_any_failure"] = None
    print(f"[warn] Brier score unavailable: {_e}")

# Save metrics JSON
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved eval metrics to: {OUT_PATH}")