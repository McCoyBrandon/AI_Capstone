"""
train2.py — Hyperparameter tuning with TensorBoard integration

Terminal Runs in Root Directory:
Run the code:
python -m src.training.train2

View the results:
python -m tensorboard.main --logdir runs/<YOUR RUNS_NAME> --port 6006
python -m tensorboard.main --logdir runs/oct15_run_bm --port 6006
"""

# Required Packages
import numpy as np                
import pandas as pd               
import torch                     
import torch.nn as nn             
from sklearn.metrics import  accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import os, json
# Added for TensorBoard and plotting
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from itertools import product as _prod
import time, platform
try:
    import psutil
except Exception:
    psutil = None


# Additional Reference code
from src.models.transformer_class import TinyTabTransformer                   # model class
from src.data.data_loader import (                                            # prepared data & metadata
    tr_dl, te_dl, class_weights, class_counts, type_vocab,
    mean, std, NUM_COLS, CLASS_NAMES, N_CLASSES, DEVICE, D_MODEL, NHEAD, TARGET
)

# Where to save everything 
RUNS_NAME = "oct15_run_bm"                 # e.g., "oct15_run_bm"
RUNS_ROOT = os.path.join("runs", RUNS_NAME)

# Default hyperparameters
LR      = 1e-3
EPOCHS  = 10
D_MODEL = 64
NHEAD   = 2

## Functions to assist with the hypertuning tracking

def _plot_confusion_matrix(cm_arr, class_names):
    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    im = ax.imshow(cm_arr, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)), yticks=range(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label', xlabel='Predicted label', title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(cm_arr[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def _auroc_block(probs, all_y):
    probs_np  = np.asarray(probs)
    y_true_np = np.asarray(all_y)

    per_class_auc = {}
    for i, name in enumerate(CLASS_NAMES):
        y_i = (y_true_np == i).astype(int)
        if y_i.min() == y_i.max():
            per_class_auc[name] = None
        else:
            try:
                per_class_auc[name] = roc_auc_score(y_i, probs_np[:, i])
            except Exception:
                per_class_auc[name] = None

    try:
        nofail_idx = CLASS_NAMES.index("NoFailure")
    except ValueError:
        nofail_idx = 0
    y_bin = (y_true_np != nofail_idx).astype(int)
    p_bin = 1.0 - probs_np[:, nofail_idx]  # prob of any failure
    if y_bin.min() == y_bin.max():
        bin_auc = None
    else:
        try:
            bin_auc = roc_auc_score(y_bin, p_bin)
        except Exception:
            bin_auc = None

    return per_class_auc, bin_auc

def train_one_run(hp, run_name):
    """
    Train/evaluate a single run with hyperparameters:
      hp = {"LR": float, "D_MODEL": int, "NHEAD": int, "EPOCHS": int}
    Logs to TensorBoard under runs/ai4i_tune/<run_name> and saves a checkpoint/metrics.
    Returns (acc, macro_f1, weighted_f1, bacc).
    """
    # Defaulting if hyper parameters not provided
    LR_     = float(hp.get("LR", LR))
    D_MODEL_= int(hp.get("D_MODEL", D_MODEL))
    NHEAD_  = int(hp.get("NHEAD", NHEAD))
    EPOCHS_ = int(hp.get("EPOCHS", EPOCHS))

    # Build model/opt/loss
    model = TinyTabTransformer(
        n_num=5,  # Number of columns
        type_vocab=type_vocab,
        d_model=D_MODEL_,
        nhead=NHEAD_,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.00)

    # TB writer per run TensorBoard
    log_dir = os.path.join(RUNS_ROOT, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # System and model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_scalar("perf/params_millions", num_params / 1e6, 0)
    writer.add_scalar("perf/trainable_params_millions", trainable_params / 1e6, 0)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    writer.add_text("perf/device", f"{DEVICE} | {device_name}")

    if torch.cuda.is_available():
        # Clear peak tracker for each test
        torch.cuda.reset_peak_memory_stats(device=DEVICE)


    # Creating Graph
    try:
        xb_num0, xb_type0, _ = next(iter(te_dl))
        model.eval()
        with torch.inference_mode():
            writer.add_graph(
                model, (xb_num0.to(DEVICE), xb_type0.to(DEVICE)), use_strict_trace=False
            )
        model.train()
    except Exception as e:
        print(f"[TB] Skipping add_graph due to: {e}")

    # Training procedure
    global_step = 0
    for epoch in range(1, EPOCHS_ + 1):
        model.train()
        total, count = 0.0, 0
        epoch_t0 = time.perf_counter() # Epoch timer
        step_times = []
        thr_samples_per_sec = []
        for xb_num, xb_type, yb in tr_dl:
            step_t0 = time.perf_counter()
            xb_num, xb_type, yb = xb_num.to(DEVICE), xb_type.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb_num, xb_type)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            # batch timer end
            dt = time.perf_counter() - step_t0
            step_times.append(dt)

            # throughput = samples / sec
            bs = getattr(tr_dl, "batch_size", None) or xb_num.size(0)
            thr_samples_per_sec.append(bs / dt if dt > 0 else float("nan"))

            writer.add_scalar("loss/batch", loss.item(), global_step)
            writer.add_scalar("perf/step_time_ms", dt * 1000.0, global_step)
            writer.add_scalar("perf/throughput_samples_per_sec", thr_samples_per_sec[-1], global_step)
            global_step += 1

            total += loss.item() * xb_num.size(0)
            count += xb_num.size(0)

        # epoch timing + aggregates
        epoch_time = time.perf_counter() - epoch_t0
        writer.add_scalar("perf/epoch_time_s", epoch_time, epoch)

        if step_times:
            avg_ms = (sum(step_times) / len(step_times)) * 1000.0
            avg_thr = sum(thr_samples_per_sec) / len(thr_samples_per_sec)
            writer.add_scalar("perf/avg_step_time_ms", avg_ms, epoch)
            writer.add_scalar("perf/avg_throughput_samples_per_sec", avg_thr, epoch)

        # GPU peak memory this epoch
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated(device=DEVICE)
            writer.add_scalar("perf/peak_mem_MB", peak_bytes / (1024 ** 2), epoch)
            torch.cuda.reset_peak_memory_stats(device=DEVICE)

        writer.add_scalar("loss/epoch", total / count, epoch)
        print(f"Epoch {epoch}/{EPOCHS_} - train loss: {total / count:.4f}")

    # Evaluattion Procedure
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb_num, xb_type, yb in te_dl:
            xb_num, xb_type = xb_num.to(DEVICE), xb_type.to(DEVICE)
            logits = model(xb_num, xb_type)
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_y = torch.cat(all_y, dim=0).numpy()
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    preds = probs.argmax(axis=1)

    acc  = accuracy_score(all_y, preds)
    bacc = balanced_accuracy_score(all_y, preds)
    macro_f1 = f1_score(all_y, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_y, preds, average="weighted", zero_division=0)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
    print(f"Balanced accuracy: {bacc:.4f}")
    print("\nClassification report:")
    print(classification_report(all_y, preds, labels=list(range(N_CLASSES)),
                                target_names=CLASS_NAMES, zero_division=0))
    cm = confusion_matrix(all_y, preds, labels=list(range(N_CLASSES)))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # AUROC
    per_class_auc, bin_auc = _auroc_block(probs, all_y)

    # Log TB scalars/figures/hparams
    epoch_tag = EPOCHS_  # tag final metrics with the last epoch index
    writer.add_scalar("eval/accuracy", acc, epoch_tag)
    writer.add_scalar("eval/balanced_accuracy", bacc, epoch_tag)
    writer.add_scalar("eval/macro_f1", macro_f1, epoch_tag)
    writer.add_scalar("eval/weighted_f1", weighted_f1, epoch_tag)

    for name, auc_v in per_class_auc.items():
        if auc_v is not None:
            writer.add_scalar(f"eval/auroc/{name}", auc_v, epoch_tag)
    if bin_auc is not None:
        writer.add_scalar("eval/binary_auroc_any_failure", bin_auc, epoch_tag)

    writer.add_figure("eval/confusion_matrix", _plot_confusion_matrix(cm, CLASS_NAMES), epoch_tag)

    hparam_dict = {
        "lr": LR_, "batch_size": getattr(tr_dl, 'batch_size', None) or 256,
        "d_model": D_MODEL_, "nhead": NHEAD_, "epochs": EPOCHS_,
    }
    metric_dict = {
        "hparam/accuracy": float(acc),
        "hparam/macro_f1": float(macro_f1),
        "hparam/weighted_f1": float(weighted_f1),
        "hparam/balanced_accuracy": float(bacc),
    }
    writer.add_hparams(hparam_dict, metric_dict)

    # Save Checkpoints
    ckpt_dir = os.path.join(RUNS_ROOT, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(ckpt_dir, "model.ckpt"))
    with open(os.path.join(ckpt_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "acc": float(acc), "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1), "bacc": float(bacc),
            "per_class_auc": {k: (None if v is None else float(v)) for k, v in per_class_auc.items()},
            "bin_auc": (None if bin_auc is None else float(bin_auc)),
        }, f, indent=2)

    writer.close()
    return acc, macro_f1, weighted_f1, bacc


# Hyperparameter Driver

def _product_dict(grid):
    keys = list(grid.keys())
    for values in _prod(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def main():
    # Defining hyperparameter grid
    grid = {
        "D_MODEL": [64, 128],
        "NHEAD":  [2, 4],
        "LR":     [1e-3, 3e-4],
        "EPOCHS": [10],  # keep fixed initially for speed/fairness
    }

    best = None
    best_key = None

    for hp in _product_dict(grid):
        run_name = f"D{hp['D_MODEL']}-H{hp['NHEAD']}-lr{hp['LR']}"
        print(f"\nTuning run: {run_name} ===")
        acc, macro_f1, weighted_f1, bacc = train_one_run(hp, run_name)

        score = macro_f1  # selection metric (robust for imbalance)
        if best is None or score > best:
            best = score
            best_key = (run_name, hp)

    print("\nBest run Results")
    print(best_key, "score(macro_f1)=", best)


if __name__ == "__main__":
    main()
