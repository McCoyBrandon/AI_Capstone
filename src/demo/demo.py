"""
Run a pretrained TinyTabTransformer on the test data, print metrics,
and list test-set machines predicted to FAIL (with failure type).

Terminal use instructions:
  python -m src.training.demo \
    --ckpt runs/ai4i_run_1/model.ckpt \
    --csv  src/data/ai4i2020.csv \
    --out  runs/ai4i_run_1/demo_metrics.json \
    --failures-out runs/ai4i_run_1/flagged_failures.json
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

# Setting up Terminal instructions when using --help
parser = argparse.ArgumentParser(description="Demo: evaluate a pretrained model and list predicted failures.")
parser.add_argument("--ckpt", type=str, default="runs/ai4i_run_1/model.ckpt",
                    help="Path to checkpoint saved by train.py (weights + meta)")
parser.add_argument("--csv", type=str, default="src/data/ai4i2020.csv",
                    help="Path to AI4I 2020 CSV (same file used in training)")
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


## Load CSV and rebuild labels exactly like data_loader.py
##!! REVIEW CREATING FUNCTIONS WITHIN DATA_LOADER.PY TO CALL TO REMOVE DUPLICATE !!##
df = pd.read_csv(CSV_PATH)
# column check 
required_cols = set(NUM_COLS + [CAT_COL, TARGET] + FAILURE_COLS)
missing = list(required_cols - set(df.columns))
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# Extract features
X_num   = df[NUM_COLS].values.astype(np.float32)
cat_idx = df[CAT_COL].astype("category").cat.codes.values.astype(np.int64)
# Build multiclass target exactly as in data_loader.py
# 0 = NoFailure; 1..5 = TWF/HDF/PWF/OSF/RNF
fails   = df[FAILURE_COLS].values.astype(np.int64)  # [N,5]
y_bin   = df[TARGET].values.astype(np.int64)
y_cls   = np.zeros(len(df), dtype=np.int64)
mask_fail = (y_bin == 1)
if mask_fail.any():
    priority = np.array([0,1,2,3,4], dtype=np.int64)   # TWF > HDF > PWF > OSF > RNF
    sub = fails[mask_fail]
    chosen = np.argmax(sub[:, priority] == 1, axis=1)
    none_set = (sub.sum(axis=1) == 0)
    chosen[none_set] = 0
    y_cls[mask_fail] = chosen + 1

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


# Metrics (same look/feel as eval.py)
##!! REVIEW CREATING FUNCTIONS WITHIN METRICS.PY TO CALL TO REMOVE DUPLICATE !!##
metrics = {
    "accuracy":          float(accuracy_score(y_te, preds)),
    "balanced_accuracy": float(balanced_accuracy_score(y_te, preds)),
    "macro_f1":          float(f1_score(y_te, preds, average="macro", zero_division=0)),
    "weighted_f1":       float(f1_score(y_te, preds, average="weighted", zero_division=0)),
    "support_per_class": dict(zip(CLASS_NAMES, [int(x) for x in np.bincount(y_te, minlength=N_CLASSES)])),
    "report":            classification_report(
                              y_te, preds,
                              labels=list(range(N_CLASSES)),
                              target_names=CLASS_NAMES,
                              zero_division=0,
                              output_dict=True,
                          ),
    "confusion_matrix":  confusion_matrix(y_te, preds, labels=list(range(N_CLASSES))).tolist(),
}

print(f"Test accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}")
print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
print("\nClassification report:")
print(classification_report(y_te, preds, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES, zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_te, preds, labels=list(range(N_CLASSES))))

# AUROC diagnostics
Y_true_ovr = np.zeros((y_te.shape[0], N_CLASSES), dtype=np.int64)
Y_true_ovr[np.arange(y_te.shape[0]), y_te] = 1

per_class_auc = {}
print("\nPer-class AUROC:")
for i, name in enumerate(CLASS_NAMES):
    try:
        auc_i = roc_auc_score(Y_true_ovr[:, i], probs[:, i])
        per_class_auc[name] = float(auc_i)
        print(f"{name:>12s}: {auc_i:.4f}")
    except ValueError:
        per_class_auc[name] = None
        print(f"{name:>12s}: N/A (absent or degenerate in test)")

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

# Binary AUROC (NoFailure vs any failure)
p_any_fail  = probs[:, 1:].sum(axis=1)
y_binary_te = (y_te != 0).astype(int)
try:
    bin_auc = float(roc_auc_score(y_binary_te, p_any_fail))
    print(f"\nBinary AUROC (TARGET='{TARGET}'): {bin_auc:.4f}")
except ValueError:
    bin_auc = None
    print(f"\nBinary AUROC (TARGET='{TARGET}') unavailable.")

metrics.update({
    "per_class_auc": per_class_auc,
    "macro_auc": macro_auc,
    "micro_auc": micro_auc,
    "weighted_auc": weighted_auc,
    "binary_auc": bin_auc,
})

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