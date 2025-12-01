"""
Hyperparameter tuning script for the TinyTabTransformer model using AI4I-style datasets.

This module automates training runs with different combinations of hyperparameters 
(For example, learning rate, model dimension, attention heads) and evaluates each configuration 
to identify the optimal setup based on performance metrics such as Macro F1.

CSV input is dynamically loaded through the `prepare_datasets()` function from `src.data.preprocess`.

Main Function:
    main()
        - Defines a hyperparameter grid for tuning (e.g., D_MODEL, NHEAD, LR)
        - Iterates over all combinations
        - Trains and evaluates TinyTabTransformer models for each configuration
        - Logs metrics and performance data to TensorBoard
        - Saves model checkpoints and metrics for each run
        - Reports the best run based on macro F1 score

Core Components:
    Model:
        TinyTabTransformer (imported from src.models.transformer_class)
    Data:
        prepare_datasets() from src.data.preprocess
    Logging:
        TensorBoard SummaryWriter (one log per hyperparameter run)
    Checkpointing:
        model.ckpt and metrics.json saved under runs/<RUNS_NAME>/<HP_COMBO>/

Key Functions and Sections:
    - train_one_run(hp, run_name): 
          Trains and evaluates the model for a single hyperparameter combination
    - _product_dict(grid): 
          Generates all combinations of hyperparameters from the defined grid
    - main(): 
          Drives the full tuning experiment and reports the best-performing configuration

Performance Logging:
    - Tracks training loss, throughput, and step/epoch times
    - Logs evaluation metrics: accuracy, balanced accuracy, macro/weighted F1, and AUROC
    - Adds confusion matrices and hyperparameter summaries to TensorBoard

Terminal Run:
    python -m src.training.grid_search_train

View the results:
    python -m tensorboard.main --logdir runs/<YOUR RUNS_NAME> --port 6006
    python -m tensorboard.main --logdir runs/nov11_run_bm --port 6006
"""

# Required Packages
import numpy as np                
import pandas as pd               
import torch                     
import torch.nn as nn             
from sklearn.metrics import classification_report as _cr
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
import argparse


# Additional Reference code
from src.models.transformer_class import TinyTabTransformer                   
from src.data.preprocess import prepare_datasets, CLASS_NAMES, N_CLASSES, DEVICE
from src.utils.metrics import compute_core_metrics, compute_auroc_metrics, confusion_matrix_figure
from src.utils.drift_detection import run_torchdrift_drift_check

# Data Processing
CSV_PATH = "src/data/ai4i2020.csv"
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

tr_dl         = data["tr_dl"]
te_dl         = data["te_dl"]
class_weights = data["class_weights"]
class_counts  = data["class_counts"]
type_vocab    = data["type_vocab"]

# Additional subsections
NUM_COLS    = data["NUM_COLS"]
D_MODEL     = data["D_MODEL"]
NHEAD       = data["NHEAD"]
TARGET      = data["TARGET"]
CLASS_NAMES = data["CLASS_NAMES"]
N_CLASSES   = data["N_CLASSES"]

# Where to save everything 
# Where to save everything (CLI-configurable)
parser = argparse.ArgumentParser(description="Grid search trainer for TinyTabTransformer.")
parser.add_argument(
    "--run_name", "--runs_name", dest="runs_name",
    type=str, default="nov3_run_bm",
    help="Folder under 'runs/' to store all tuning outputs (For example, nov11_run_bm)."
)
args = parser.parse_args()
RUNS_NAME = args.runs_name
RUNS_ROOT = os.path.join("runs", RUNS_NAME)

# Default hyperparameters
LR      = 1e-3
EPOCHS  = 10
D_MODEL = 64
NHEAD   = 2
WD            = 0.0   # weight decay for Adam
LABEL_SMOOTH  = 0.0   # label smoothing for CrossEntropyLoss

## Functions to assist with the hypertuning tracking
def train_one_run(hp, run_name):
    """
    Train/evaluate a single run with hyperparameters:
      hp = {
          "LR": float,
          "D_MODEL": int,
          "NHEAD": int,
          "EPOCHS": int,
          "WD": float,             # weight decay for Adam
          "LABEL_SMOOTH": float,   # label smoothing for CE loss
      }
    Logs to TensorBoard under runs/<RUNS_NAME>/<run_name> and saves a checkpoint/metrics.
    Returns (acc, macro_f1, weighted_f1, bacc).
    """
    # Defaulting if hyper parameters not provided
    LR_     = float(hp.get("LR", LR))
    D_MODEL_= int(hp.get("D_MODEL", D_MODEL))
    NHEAD_  = int(hp.get("NHEAD", NHEAD))
    EPOCHS_ = int(hp.get("EPOCHS", EPOCHS))
    WD_           = float(hp.get("WD", WD)) 
    LABEL_SMOOTH_ = float(hp.get("LABEL_SMOOTH", LABEL_SMOOTH))
    ENABLE_DRIFT = bool(hp.get("ENABLE_DRIFT", False))

    # Build model/opt/loss
    model = TinyTabTransformer(
        n_num=5,
        type_vocab=type_vocab,
        d_model=D_MODEL_,
        nhead=NHEAD_,
        dim_feedforward=hp.get("DIM_FEEDFORWARD", 128),
        dropout=hp.get("DROPOUT", 0.0),
        num_layers=hp.get("NUM_LAYERS", 2),
    ).to(DEVICE)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=LR_,
        weight_decay=WD_, 
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTH_,  
    )

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
        print(f"Tensorboard skipping add_graph due to: {e}")

    # Training procedure
    global_step = 0
    for epoch in range(1, EPOCHS_ + 1):
        model.train()
        total, count = 0.0, 0
        epoch_t0 = time.perf_counter()  # Epoch timer
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
        
    # Evaluattion Procedure and saving metrics with checkpoints
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

    # Shared metrics
    core = compute_core_metrics(all_y, preds, CLASS_NAMES)
    aucs = compute_auroc_metrics(all_y, probs, CLASS_NAMES)

    # Console summary
    print(f"Test accuracy: {core['accuracy']:.4f}")
    print(f"Macro F1: {core['macro_f1']:.4f} | Weighted F1: {core['weighted_f1']:.4f}")
    print(f"Balanced accuracy: {core['balanced_accuracy']:.4f}")
    print("\nClassification report:")
    print(_cr(all_y, preds, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(np.array(core["confusion_matrix"]))

    print("\nPer-class AUROC:")
    for k, v in aucs["per_class_auc"].items():
        print(f"{k:>12s}: {'N/A' if v is None else f'{v:.4f}'}")
    for k in ("macro_auc", "micro_auc", "weighted_auc", "binary_auc"):
        val = aucs[k]
        print(f"{k}: {'N/A' if val is None else f'{val:.4f}'}")

    # TensorBoard: scalars/fig/params
    epoch_tag = EPOCHS_
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

    cm_fig = confusion_matrix_figure(np.array(core["confusion_matrix"]), CLASS_NAMES)
    writer.add_figure("eval/confusion_matrix", cm_fig, epoch_tag)

    hparam_dict = {
        "lr": LR_,
        "batch_size": getattr(tr_dl, 'batch_size', None) or 256,
        "d_model": D_MODEL_,
        "nhead": NHEAD_,
        "epochs": EPOCHS_,
        "weight_decay": WD_,
        "label_smoothing": LABEL_SMOOTH_,
    }
    metric_dict = {
        "hparam/accuracy":          float(core["accuracy"]),
        "hparam/macro_f1":          float(core["macro_f1"]),
        "hparam/weighted_f1":       float(core["weighted_f1"]),
        "hparam/balanced_accuracy": float(core["balanced_accuracy"]),
    }
    writer.add_hparams(hparam_dict, metric_dict)

    # Drift Detection
    drift_metrics = None
    if ENABLE_DRIFT:
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

    # Save Model & Metrics
    ckpt_dir = os.path.join(RUNS_ROOT, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(ckpt_dir, "model.ckpt"))

    metrics_payload = {**core, **aucs}
    if drift_metrics is not None:
        metrics_payload["drift_metrics"] = drift_metrics

    with open(os.path.join(ckpt_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    writer.close()
    return core["accuracy"], core["macro_f1"], core["weighted_f1"], core["balanced_accuracy"]


# Hyperparameter Driver
def _product_dict(grid):
    keys = list(grid.keys())
    for values in _prod(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def main():
    # Defining hyperparameter grid (expand as needed)
    grid = {
        "D_MODEL": [64, 128],
        "NHEAD": [2, 4],
        "LR": [1e-3, 3e-4],
        "EPOCHS": [20],
        "DIM_FEEDFORWARD": [128, 256],
        "DROPOUT": [0.0, 0.1],
        "NUM_LAYERS": [2, 3],
        "WD": [0.0, 1e-4],
        "LABEL_SMOOTH": [0.0, 0.05],
        "ENABLE_DRIFT": [True],
    }

    best = None
    best_key = None

    for hp in _product_dict(grid):
        # Debug: validate D_MODEL % NHEAD == 0
        if int(hp["D_MODEL"]) % int(hp["NHEAD"]) != 0:
            print(f"Skipping invalid combo: D_MODEL={hp['D_MODEL']}, NHEAD={hp['NHEAD']}")
            continue

        run_name = (
            f"D{hp['D_MODEL']}-H{hp['NHEAD']}-lr{hp['LR']}"
            f"-wd{hp['WD']}-ls{hp['LABEL_SMOOTH']}"
            f"-drift{int(hp.get('ENABLE_DRIFT', False))}"
        )

        print(f"\nTuning run: {run_name} ===")
        acc, macro_f1, weighted_f1, bacc = train_one_run(hp, run_name)

        score = macro_f1
        if best is None or score > best:
            best = score
            best_key = (run_name, hp)

    print("\nBest run Results")
    print(best_key, "score(macro_f1)=", best)



if __name__ == "__main__":
    main()
