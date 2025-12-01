"""
Training script for the TinyTabTransformer model using AI4I-style datasets.

This module handles model initialization, optimization, and performance logging 
via TensorBoard. It supports multi-class classification with class-weighted loss 
to address class imbalance, and produces checkpointed model weights and metrics 
for evaluation and reproducibility.

CSV input is **not hardcoded** — the data is dynamically loaded and preprocessed 
through the `prepare_datasets()` function from `src.data.preprocess`.

Main Function:
    (executed when run directly)
    Trains the TinyTabTransformer model end-to-end using:
        - Data from prepare_datasets(csv_path)
        - Model definition from src.models.transformer_class
        - Training hyperparameters defined within this script

Core Components:
    Model:
        TinyTabTransformer (imported from src.models.transformer_class)
    Data:
        prepare_datasets() from src.data.preprocess
    Logging:
        TensorBoard SummaryWriter (logs under runs/<RUN_NAME>)
    Checkpointing:
        model.ckpt and metrics.json saved under runs/<RUN_NAME>

Outputs:
    - Trained model checkpoint: runs/<RUN_NAME>/model.ckpt
    - Metrics file:             runs/<RUN_NAME>/metrics.json
    - TensorBoard logs:         runs/<RUN_NAME>/tb/

Key Functions and Sections:
    - prepare_datasets(csv_path): Loads and preprocesses data
    - TinyTabTransformer(...): Builds model architecture
    - Training Loop: Runs for specified epochs, logs losses and metrics
    - Evaluation Block: Computes accuracy, F1, balanced accuracy, AUROC, etc.
    - TensorBoard Logging: Tracks loss, metrics, and confusion matrix

Debugging test:
    Terminal:
        python -m src.training.train
"""

# Required Packages
import numpy as np                
import pandas as pd               
import torch                     
import torch.nn as nn             
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import classification_report as cr
import os, json
# Added for TensorBoard and plotting
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import itertools
import argparse

## Additional Reference code
from src.models.transformer_class import TinyTabTransformer                   # model class
from src.utils.metrics import compute_core_metrics, compute_auroc_metrics, confusion_matrix_figure
from src.utils.visualizations import plot_pre_post_smote_counts,plot_class_distribution,plot_roc_pr_curves,tb_log_figure, plot_confusion_matrix
from src.utils.drift_detection import run_torchdrift_drift_check
## Data Processing
from src.data.preprocess import prepare_datasets, CLASS_NAMES, N_CLASSES, DEVICE, D_MODEL, NHEAD, NUM_COLS, TARGET

# Argument parsing for dynamic run naming
parser = argparse.ArgumentParser(description="Train TinyTabTransformer single parameter set training.")
parser.add_argument(
    "--run_name",
    type=str,
    default="Test_run_1",
    help="Name of the run folder under 'runs/'. Example: --run_name ai4i_run_1"
)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--nhead", type=int, default=2)
parser.add_argument("--dim_feedforward", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--label_smoothing", type=float, default=0.00)
parser.add_argument("--enable_drift",action="store_true",help="If set, run TorchDrift drift detection after training.")
args = parser.parse_args()
RUN_NAME = args.run_name

CSV_PATH = "src/data/ai4i2020.csv"   # or any CSV with the same headers
data = prepare_datasets(
    csv_path=CSV_PATH,
    batch=256,
    test_size=0.20,
    random_state=42,
    normalize="zscore",   # "zscore" or "minmax"
    resample="smote",     # SMOTE resampling method
    smote_k_neighbors=5,
    smote_random_state=42,
    smote_sampling="auto" # Auto or use dict like {1:desired_count, ...} if you want custom per-class
)

# TensorBoard setup
log_dir = f"runs/{RUN_NAME}/tb"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
# Auto-launch TensorBoard (non-blocking) and open in browser
try:
    import subprocess, sys, time, webbrowser
    tb_cmd = [sys.executable, "-m", "tensorboard", "--logdir", log_dir, "--port", "6006"]
    tb_proc = subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(2)  # give server a moment to start
    webbrowser.open("http://localhost:6006")
    print("TensorBoard started at http://localhost:6006")
except Exception as e:
    print(f"TensorBoard could not auto-launch TensorBoard: {e}\n"
          f"      Run manually: python -m tensorboard --logdir {log_dir} --port 6006")
# Static PNGs used for paper/poster
figs_dir = f"runs/{RUN_NAME}/figs"
os.makedirs(figs_dir, exist_ok=True)

# Class distribution logging (pre/post SMOTE) for TensorBoard
pre_counts  = data.get("pre_counts", None)
post_counts = data.get("post_counts", None)

# Scalars panel: one scalar per class
if pre_counts is not None:
    writer.add_scalars(
        "data/class_counts_pre",
        {name: int(pre_counts[i]) for i, name in enumerate(CLASS_NAMES)},
        global_step=0,
    )

if post_counts is not None:
    writer.add_scalars(
        "data/class_counts_post",
        {name: int(post_counts[i]) for i, name in enumerate(CLASS_NAMES)},
        global_step=0,
    )
# Visual for Original vs SMOTE class counts
if pre_counts is not None and post_counts is not None:
    fig = plot_pre_post_smote_counts(
        pre_counts=pre_counts,
        post_counts=post_counts,
        class_names=CLASS_NAMES,
        title="Class Counts: Original vs SMOTE"
    )
    tb_log_figure(writer, "data/pre_vs_post_smote_counts", fig, step=0)   # TB
    fig.savefig(os.path.join(figs_dir, "pre_post_smote_counts.png"), dpi=300, bbox_inches="tight")
    
# Post Adjustment Histograms panel
import torch as _torch
y_tr_for_hist = data.get("y_tr_labels_for_hist", None)
if y_tr_for_hist is not None:
    writer.add_histogram(
        "data/labels_post_hist",
        _torch.tensor(y_tr_for_hist, dtype=_torch.int64),
        global_step=0,
    )

# Pre Adjustment Histograms panel
y_tr_pre_hist = data.get("y_tr_labels_pre_hist", None)
if y_tr_pre_hist is not None:
    writer.add_histogram(
        "data/labels_pre_hist",
        _torch.tensor(y_tr_pre_hist, dtype=_torch.int64),
        global_step=0,
    )

# Histogram using visualization.py
y_tr_for_hist = data.get("y_tr_labels_for_hist", None)
if y_tr_for_hist is not None:
    fig = plot_class_distribution(y_tr_for_hist, class_names=CLASS_NAMES, title="Train Labels (After SMOTE)")
    tb_log_figure(writer, "data/train_labels_post/bar", fig, step=0)
    fig.savefig(os.path.join(figs_dir, "train_labels_post_bar.png"), dpi=300, bbox_inches="tight")

    
tr_dl         = data["tr_dl"]
te_dl         = data["te_dl"]
class_weights = data["class_weights"]
class_counts  = data["class_counts"]
type_vocab    = data["type_vocab"]
mean          = data["mean"]
std           = data["std"]

# Hyperparameters for the model
BATCH  = 256                            # Mini-batch size for training/eval
EPOCHS = 10                             # Number of passes over the training data
LR     = args.lr                        # Adam learning rate
D_MODEL = args.d_model                  # Token embedding size (hidden size)
NHEAD   = args.nhead                    # Number of attention heads (must divide D_MODEL)
DIM_FEEDFORWARD = args.dim_feedforward
DROPOUT = args.dropout
NUM_LAYERS = args.num_layers
WEIGHT_DECAY = args.weight_decay
LABEL_SMOOTH = args.label_smoothing

# Processing managagement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if present, else CPU

### Model Instantiation
model = TinyTabTransformer(
    n_num=len(NUM_COLS),
    type_vocab=type_vocab,
    d_model=D_MODEL,
    nhead=NHEAD,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    num_layers=NUM_LAYERS
).to(DEVICE)

### Optimizer & Loss evaluation
# Adam optimizer over all model parameters
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Adjusted loss to suit for a multi-class, class imbalance balancing with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

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
model.eval()
all_logits, all_y = [], []
with torch.no_grad():
    for xb_num, xb_type, yb in te_dl:
        xb_num, xb_type = xb_num.to(DEVICE), xb_type.to(DEVICE)
        logits = model(xb_num, xb_type)
        all_logits.append(logits.cpu())
        all_y.append(yb.cpu())

all_logits = torch.cat(all_logits, dim=0).numpy()
all_y      = torch.cat(all_y,      dim=0).numpy()
probs      = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
preds      = probs.argmax(axis=1)
core = compute_core_metrics(all_y, preds, CLASS_NAMES)
aucs = compute_auroc_metrics(all_y, probs, CLASS_NAMES)

# Drift Detetection
drift_metrics = None
if args.enable_drift:
    try:
        drift_metrics = run_torchdrift_drift_check(
            tr_dl=tr_dl,
            te_dl=te_dl,
            device=DEVICE,
            writer=writer,
            tb_step=epoch_tag,
            tb_prefix="drift",
        )
        print("\n[Drift detection] TorchDrift metrics:")
        for k, v in drift_metrics.items():
            print(f"  {k}: {v:.6f}")
    except RuntimeError as e:
        print(f"[Drift detection] Skipping TorchDrift: {e}")

# Console prints
print(f"Test accuracy: {core['accuracy']:.4f}")
print(f"Macro F1: {core['macro_f1']:.4f} | Weighted F1: {core['weighted_f1']:.4f}")
print(f"Balanced accuracy: {core['balanced_accuracy']:.4f}")
print("\nClassification report:")
print(cr(all_y, preds, labels=list(range(N_CLASSES)),
          target_names=CLASS_NAMES, zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(np.array(core["confusion_matrix"]))

print("\nPer-class AUROC:")
for k, v in aucs["per_class_auc"].items():
    print(f"{k:>12s}: {'N/A' if v is None else f'{v:.4f}'}")
for k in ("macro_auc", "micro_auc", "weighted_auc", "binary_auc"):
    val = aucs[k]
    print(f"{k}: {'N/A' if val is None else f'{val:.4f}'}")

# Saving ROC and PR Curves figures for paper
y_true_onehot = np.eye(N_CLASSES, dtype=int)[all_y]  # (N, C)
fig = plot_roc_pr_curves(y_true_onehot=y_true_onehot, y_prob=probs, class_names=CLASS_NAMES)
tb_log_figure(writer, "eval/roc_pr_curves", fig, step=epoch)  # log at final epoch index
fig.savefig(os.path.join(figs_dir, f"roc_pr_curves_epoch{epoch}.png"), dpi=300, bbox_inches="tight")

# TensorBoard scalars
epoch_tag = epoch  # tag with last epoch index
writer.add_scalar("eval/accuracy",          core["accuracy"], epoch_tag)
writer.add_scalar("eval/balanced_accuracy", core["balanced_accuracy"], epoch_tag)
writer.add_scalar("eval/macro_f1",          core["macro_f1"], epoch_tag)
writer.add_scalar("eval/weighted_f1",       core["weighted_f1"], epoch_tag)

for cname, auc_v in aucs["per_class_auc"].items():
    if auc_v is not None:
        writer.add_scalar(f"eval/auroc/{cname}", auc_v, epoch_tag)
for k in ("macro_auc", "micro_auc", "weighted_auc", "binary_auc"):
    if aucs[k] is not None:
        writer.add_scalar(f"eval/{k}", aucs[k], epoch_tag)

# Confusion matrix figure
# Suppose class 0 == "NoFailure"
cm = np.array(core["confusion_matrix"])
cm_fig = plot_confusion_matrix(
    cm=cm,
    class_names=CLASS_NAMES,
    title="Confusion Matrix (Counts; scaled w/o NoFailure→NoFailure)",
    cmap="magma_r",
    log_scale=False,                   # True is also worth trying for even more pop
    robust_clip=(1, 99),
    scale_exclude_cells=[(0, 0)],      # <-- key: ignore NF→NF for color scaling
    annotate=True
)
writer.add_figure("eval/confusion_matrix_focused", cm_fig, epoch_tag)
cm_fig.savefig(os.path.join(figs_dir, f"confusion_matrix_focused_epoch{epoch}.png"),
               dpi=300, bbox_inches="tight")

# hparams summary
hparam_dict = {
    "lr": LR, "batch_size": BATCH, "d_model": D_MODEL, "nhead": NHEAD,
    "epochs": EPOCHS, "dim_feedforward": DIM_FEEDFORWARD, "dropout": DROPOUT,
    "num_layers": NUM_LAYERS, "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTH
}
metric_dict = {
    "hparam/accuracy":          core["accuracy"],
    "hparam/macro_f1":          core["macro_f1"],
    "hparam/weighted_f1":       core["weighted_f1"],
    "hparam/balanced_accuracy": core["balanced_accuracy"],
}
writer.add_hparams(hparam_dict, metric_dict)

# Save Metrics in JSON
out_dir = f"runs/{RUN_NAME}"
os.makedirs(out_dir, exist_ok=True)
payload = {**core, **aucs}
if drift_metrics is not None:
    payload["drift_metrics"] = drift_metrics

with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
    import json as _json
    _json.dump(payload, f, indent=2)


# Save checkpoint
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
torch.save(ckpt, os.path.join(out_dir, "model.ckpt"))
print("\nSaved checkpoint to:", os.path.join(out_dir, "model.ckpt"))

writer.close()
