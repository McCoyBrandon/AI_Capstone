"""
TabTransformer-style classifier for the AI4I 2020 dataset.
Reference: 

"""

# Required Packages
import numpy as np                
import pandas as pd               
import torch                     
import torch.nn as nn             
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import os, json

# Additional Reference code
from transformer_class import TinyTabTransformer                     # model class
from data_loader import (                                            # prepared data & metadata
    tr_dl, te_dl, class_weights, class_counts, type_vocab,
    mean, std, NUM_COLS, CLASS_NAMES, N_CLASSES, DEVICE, D_MODEL, NHEAD, TARGET
)

### Starting Variables
## Variables for the data
CSV_PATH = "ai4i2020.csv"         # Expect the UCI CSV to be in the same folder, will need to adjust when we restructure the folders.
NUM_COLS = [                      # 5 numeric features
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CAT_COL = "Type"                  # Single categorical feature (values like L/M/H)
TARGET = "Machine failure"        # Binary target column (0 or 1)
FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"] # Columns for multi-classifcation
CLASS_NAMES  = ["NoFailure", "TWF", "HDF", "PWF", "OSF", "RNF"] # Classifiers for multi-classification
N_CLASSES    = len(CLASS_NAMES)

# Hyperparameters for the model
BATCH = 256                       # Mini-batch size for training/eval
EPOCHS = 10                       # Number of passes over the training data
LR = 1e-3                         # Adam learning rate
D_MODEL = 64                      # Token embedding size (hidden size)
NHEAD = 2                         # Number of attention heads (must divide D_MODEL)
# Processing managagement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if present, else CPU

### Model Instantiation
model = TinyTabTransformer(
    n_num=len(NUM_COLS),
    type_vocab=type_vocab,
    d_model=D_MODEL,
    nhead=NHEAD
).to(DEVICE)

### Optimizer & Loss evaluation
# Adam optimizer over all model parameters
opt = torch.optim.Adam(model.parameters(), lr=LR)

# Adjusted loss to suit for a multi-class, class imbalance balancing with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.00)

### Training Loop
for epoch in range(1, EPOCHS + 1):
    model.train()                 # set the model to training mode 
    total = 0.0                   # Track sum of losses (for epoch-average)
    count = 0                     # Track number of samples

    for xb_num, xb_type, yb in tr_dl:
        # Move batch to the device
        xb_num  = xb_num.to(DEVICE)    # [B, n_num], float32
        xb_type = xb_type.to(DEVICE)   # [B], int64
        yb      = yb.to(DEVICE)        # [B], float32

        opt.zero_grad()                           # reset gradients from previous
        logits = model(xb_num, xb_type)           # forward pass -> [B]
        loss = loss_fn(logits, yb)                # compute loss 
        loss.backward()                           # backprop: compute gradients
        opt.step()                                # update parameters with Adam

        # Track running loss (!!!multiply by batch size to average later!!!)
        total += loss.item() * xb_num.size(0)
        count += xb_num.size(0)

    # Print average training loss for this epoch
    print(f"Epoch {epoch}/{EPOCHS} - train loss: {total / count:.4f}")

### Evaluation on Test Set
# Using the new multi-class evaluation with a Binary check at the end. And intruduce logits adjustment for class imbalance.
model.eval()
all_logits, all_y = [], []
with torch.no_grad():
    for xb_num, xb_type, yb in te_dl:
        xb_num  = xb_num.to(DEVICE)
        xb_type = xb_type.to(DEVICE)
        logits  = model(xb_num, xb_type)   # [B, C]
        all_logits.append(logits.cpu())
        all_y.append(yb.cpu())

all_logits = torch.cat(all_logits, dim=0).numpy()  # [Nte, C]
all_y = torch.cat(all_y, dim=0).numpy()            # [Nte]
probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
preds = probs.argmax(axis=1)

## Core metrics
# For being saved with the weighted model
metrics = {
    "accuracy":           float(accuracy_score(all_y, preds)),
    "balanced_accuracy":  float(balanced_accuracy_score(all_y, preds)),
    "macro_f1":           float(f1_score(all_y, preds, average="macro", zero_division=0)),
    "weighted_f1":        float(f1_score(all_y, preds, average="weighted", zero_division=0)),
    "support_per_class":  dict(zip(CLASS_NAMES, [int(x) for x in np.bincount(all_y, minlength=N_CLASSES)])),
    "report":             classification_report(
                             all_y, preds,
                             labels=list(range(N_CLASSES)),
                             target_names=CLASS_NAMES,
                             zero_division=0,
                             output_dict=True  # <-- makes it JSON-serializable
                         ),
    "confusion_matrix":   confusion_matrix(all_y, preds, labels=list(range(N_CLASSES))).tolist(),
}
# For terminal outputs during test runs
acc  = accuracy_score(all_y, preds)
bacc = balanced_accuracy_score(all_y, preds)
print(f"Test accuracy: {acc:.4f}")
macro_f1 = f1_score(all_y, preds, average="macro", zero_division=0)
weighted_f1 = f1_score(all_y, preds, average="weighted", zero_division=0)
print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
print(f"Balanced accuracy: {bacc:.4f}")
print("\nClassification report:")
# DEBUGGING Reviewing Classifications showing up
print(classification_report(all_y,preds,labels=list(range(N_CLASSES)),target_names=CLASS_NAMES,zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_y, preds, labels=list(range(N_CLASSES))))

# AUROC
Y_true_ovr = np.zeros((all_y.shape[0], N_CLASSES), dtype=np.int64)
Y_true_ovr[np.arange(all_y.shape[0]), all_y] = 1

print("\nPer-class AUROC:")
per_class_auc = {}
for i, name in enumerate(CLASS_NAMES):
    try:
        auc_i = roc_auc_score(Y_true_ovr[:, i], probs[:, i])
        per_class_auc[name] = auc_i
        print(f"  {name:>10s}: {auc_i:.4f}")
    except ValueError:
        per_class_auc[name] = float("nan")
        print(f"  {name:>10s}: N/A (absent in test)")

# macro/micro/weighted AUC
try:
    macro_auc   = roc_auc_score(Y_true_ovr, probs, multi_class="ovr", average="macro")
    micro_auc   = roc_auc_score(Y_true_ovr, probs, multi_class="ovr", average="micro")
    weighted_auc= roc_auc_score(Y_true_ovr, probs, multi_class="ovr", average="weighted")
    print(f"\nMacro AUROC:    {macro_auc:.4f}") 
    print(f"Micro AUROC:    {micro_auc:.4f}")
    print(f"Weighted AUROC: {weighted_auc:.4f}")
# DEBUGGING Evaluation metric output
except ValueError:
    print("\nMacro/Micro/Weighted AUROC unavailable (degenerate labels).")

# Binary AUROC for your original TARGET (keep using y you already had) ---
# Collapse multi-class probs to “any failure” vs “no failure”: class 0 = NoFailure; classes 1..5 = failures)
p_any_fail = probs[:, 1:].sum(axis=1)          # P(failure)
# build ground truth from y_te (re-slice your saved binary y if needed)
y_binary_te = (all_y != 0).astype(int)        # from multiclass truth: 0 vs 1..5
try:
    bin_auc = roc_auc_score(y_binary_te, p_any_fail)
    print(f"\nBinary AUROC (TARGET='{TARGET}'): {bin_auc:.4f}")
# DEBUGGING Evaluation metric output
except ValueError:
    print(f"\nBinary AUROC (TARGET='{TARGET}') unavailable.")

### Save checkpoint (weights + minimal metadata for future runs)
os.makedirs("runs/ai4i_run_1", exist_ok=True)

ckpt = {
    "model_state_dict": model.state_dict(),
    "meta": {
        "class_names": CLASS_NAMES,
        "num_cols": NUM_COLS,
        "cat_col": "Type",
        "target": TARGET,
        "failure_cols": ["TWF", "HDF", "PWF", "OSF", "RNF"],
        "type_vocab": int(type_vocab),
        "standardize_mean": mean.tolist(),
        "standardize_std":  std.tolist(),
        "d_model": int(D_MODEL),
        "nhead": int(NHEAD),
    },
}
# Save the checkpoint
torch.save(ckpt, "runs/ai4i_run_1/model.ckpt")
# Save the evaluation metrics
with open("runs/ai4i_run_1/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved checkpoint to: runs/ai4i_run_1/model.ckpt")