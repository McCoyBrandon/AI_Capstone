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
# Added for TensorBoard and plotting
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import itertools


# Additional Reference code
from src.models.transformer_class import TinyTabTransformer                   # model class
from src.data.data_loader import (                                            # prepared data & metadata
    tr_dl, te_dl, class_weights, class_counts, type_vocab,
    mean, std, NUM_COLS, CLASS_NAMES, N_CLASSES, DEVICE, D_MODEL, NHEAD, TARGET
)

### Starting Variables
## Variables for the data
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

# TensorBoard setup
log_dir = "runs/ai4i_run_1/tb"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Add the model graph to TensorBoard
try:
    xb_num0, xb_type0, _ = next(iter(te_dl))
    model.eval()  # turn off dropout/fastpaths that can change the traced graph
    with torch.inference_mode():
        writer.add_graph(
            model,
            (xb_num0.to(DEVICE), xb_type0.to(DEVICE)),
            use_strict_trace=False,   # <-- key: don’t fail when tiny diffs exist
        )
    model.train()  # back to training mode
except StopIteration:
    pass
except Exception as e:
    print(f"[TB] Skipping add_graph due to: {e}")



### Training Loop
global_step = 0 # for TensorBoard logging
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

        # Log the loss to TensorBoard
        writer.add_scalar("loss/batch", loss.item(), global_step)
        global_step += 1

        # Track running loss (!!!multiply by batch size to average later!!!)
        total += loss.item() * xb_num.size(0)
        count += xb_num.size(0)

    # Log the average loss for this epoch to TensorBoard
    writer.add_scalar("loss/epoch", total / count, epoch)

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
#!! Need to review moving evaluation metrics to its own file !!#
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

# AUROC calculations (robust to classes with no positives/negatives to deal with RNF issue)
probs_np   = np.asarray(probs)         # shape (N, C)
y_true_np  = np.asarray(all_y)         # shape (N,)
n_classes  = len(CLASS_NAMES)

per_class_auc = {}
for i, name in enumerate(CLASS_NAMES):
    y_i = (y_true_np == i).astype(int)             # one-vs-rest labels
    # AUROC needs both classes present
    if y_i.min() == y_i.max():
        per_class_auc[name] = None
    else:
        try:
            per_class_auc[name] = roc_auc_score(y_i, probs_np[:, i])
        except Exception:
            per_class_auc[name] = None

# Binary AUROC: any failure vs NoFailure
try:
    nofail_idx = CLASS_NAMES.index("NoFailure")
except ValueError:
    nofail_idx = 0  # fallback if not found

y_bin = (y_true_np != nofail_idx).astype(int)      # 1 = any failure, 0 = no failure
p_bin = 1.0 - probs_np[:, nofail_idx]              # prob of "any failure"
if y_bin.min() == y_bin.max():
    bin_auc = None
else:
    try:
        bin_auc = roc_auc_score(y_bin, p_bin)
    except Exception:
        bin_auc = None

# AUROC
Y_true_ovr = np.zeros((all_y.shape[0], N_CLASSES), dtype=np.int64)
Y_true_ovr[np.arange(all_y.shape[0]), all_y] = 1

# Scalar metrics
writer.add_scalar("eval/accuracy", acc, epoch)
writer.add_scalar("eval/balanced_accuracy", bacc, epoch)
writer.add_scalar("eval/macro_f1", macro_f1, epoch)
writer.add_scalar("eval/weighted_f1", weighted_f1, epoch)

# Per-class AUROC (for each failure mode and no-failure)
for name, auc_v in per_class_auc.items():
    if not (auc_v is None or (isinstance(auc_v, float) and (auc_v != auc_v))):  # not NaN
        writer.add_scalar(f"eval/auroc/{name}", auc_v, epoch)

# Binary AUROC for "any failure"
try:
    writer.add_scalar("eval/binary_auroc_any_failure", bin_auc, epoch)
except NameError:
    pass


# print("\nPer-class AUROC:")
# per_class_auc = {}
# for i, name in enumerate(CLASS_NAMES):
#     try:
#         auc_i = roc_auc_score(Y_true_ovr[:, i], probs[:, i])
#         per_class_auc[name] = auc_i
#         print(f"  {name:>10s}: {auc_i:.4f}")
#     except ValueError:
#         per_class_auc[name] = float("nan")
#         print(f"  {name:>10s}: N/A (absent in test)")

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

# Confusion matrix plot
def plot_confusion_matrix(cm, class_names):
    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

cm_fig = plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASS_NAMES)
writer.add_figure("eval/confusion_matrix", cm_fig, epoch)

# Binary AUROC for your original TARGET (keep using y you already had) ---
# Collapse multi-class probs to “any failure” vs “no failure”: class 0 = NoFailure; classes 1..5 = failures)
p_any_fail = probs[:, 1:].sum(axis=1)          # P(failure)
# build ground truth from y_te (0 vs 1..5)
y_binary_te = (all_y != 0).astype(int)        # from multiclass truth: 0 vs 1..5
try:
    bin_auc = roc_auc_score(y_binary_te, p_any_fail)
    print(f"\nBinary AUROC (TARGET='{TARGET}'): {bin_auc:.4f}")
# DEBUGGING Evaluation metric output
except ValueError:
    print(f"\nBinary AUROC (TARGET='{TARGET}') unavailable.")

# Log binary AUROC to TensorBoard
hparam_dict = {
    "lr": LR,
    "batch_size": BATCH,
    "d_model": D_MODEL,
    "nhead": NHEAD,
    "epochs": EPOCHS,
}
metric_dict = {
    "hparam/accuracy": acc,
    "hparam/macro_f1": macro_f1,
    "hparam/weighted_f1": weighted_f1,
    "hparam/balanced_accuracy": bacc,
}
writer.add_hparams(hparam_dict, metric_dict)

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

writer.close()