""" 
Reusable preprocessing for AI4I-style CSVs (same column headers). This module does not hardcode a CSV path. Pass the CSV path in the main function. 

Main Function: 
    prepare_datasets(csv_path, batch=256, test_size=0.20, random_state=42, validate_schema=True) 
    
Functions: 
    read_ai4i_csv(csv_path, validate_schema=True) 
    build_targets(df) encode_features(df) 
    split_train_test(X_num, cat_idx, y_cls, y_bin, test_size=0.20, random_state=42) 
    standardize_train_test(Xn_tr, Xn_te, eps=1e-8) 
    compute_class_weights(y_tr, n_classes=N_CLASSES, device=DEVICE) 
    make_dataloaders(Xn_tr, ty_tr, y_tr, Xn_te, ty_te, y_te, batch=256, shuffle_train=True) 
    
Debugging test: 
    Terminal: 
        python -m src.data.preprocess --csv src/data/ai4i2020.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler


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


# Core functions
def read_ai4i_csv(csv_path, validate_schema=True):
    """
    Read a CSV that follows the AI4I headers format.
    If validate_schema=True, error on missing expected columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if validate_schema:
        required = set(NUM_COLS + [CAT_COL, TARGET] + FAILURE_COLS)
        missing = list(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing expected columns in CSV: {missing}")
    return df


def build_targets(df):
    """
    Build target vectors:
        y_bin: binary 0/1 for TARGET ('Machine failure')
        y_cls: 0->5 multi-class with priority mapping:
               0=NoFailure, 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF
    If multiple failure flags are set, resolves by priority in FAILURE_COLS order.
    """
    y_bin = df[TARGET].values.astype(np.int64)

    fails = df[FAILURE_COLS].values.astype(np.int64)  # [N,5]
    y_cls = np.zeros(len(df), dtype=np.int64)         # default to NoFailure (0)

    mask_fail = (y_bin == 1)
    if mask_fail.any():
        # Priority: TWF > HDF > PWF > OSF > RNF (left-to-right in FAILURE_COLS)
        priority = np.arange(len(FAILURE_COLS), dtype=np.int64)
        sub = fails[mask_fail]
        # choose first True in priority order; if none → stays NoFailure (0)
        chosen = np.argmax(sub[:, priority] == 1, axis=1)
        none_set = (sub.sum(axis=1) == 0)
        chosen[none_set] = 0
        y_cls[mask_fail] = chosen + 1  # shift 1..5

    return y_bin, y_cls


def encode_features(df):
    """
    Returns:
        X_num:    float32 [N, len(NUM_COLS)]
        cat_idx:  int64   [N] (codes for 'Type')
        type_vocab: int   (# categories for 'Type')
    """
    X_num = df[NUM_COLS].values.astype(np.float32)
    cat_idx = df[CAT_COL].astype("category").cat.codes.values.astype(np.int64)
    type_vocab = int(cat_idx.max()) + 1
    return X_num, cat_idx, type_vocab


def split_train_test(
    X_num, cat_idx, y_cls, y_bin,
    test_size=0.20, random_state=42
):
    """
    Stratify on y_cls to preserve class balance across splits.
    Returns:
        Xn_tr, Xn_te, ty_tr, ty_te, y_tr, y_te, yb_tr, yb_te
    """
    Xn_tr, Xn_te, ty_tr, ty_te, y_tr, y_te, yb_tr, yb_te = train_test_split(
        X_num, cat_idx, y_cls, y_bin,
        test_size=test_size,
        random_state=random_state,
        stratify=y_cls
    )
    return Xn_tr, Xn_te, ty_tr, ty_te, y_tr, y_te, yb_tr, yb_te


def standardize_train_test(Xn_tr, Xn_te, eps=1e-8):
    """
    Z-score standardization using train stats only.
    Returns:
        Xn_tr_std, Xn_te_std, mean, std
    """
    mean = Xn_tr.mean(axis=0)
    std = Xn_tr.std(axis=0) + eps
    return (Xn_tr - mean) / std, (Xn_te - mean) / std, mean, std


def compute_class_weights(y_tr, n_classes=N_CLASSES, device=DEVICE):
    """
    Compute normalized inverse-frequency weights (balanced).
    Returns:
        class_weights: torch.float32 tensor on DEVICE
        class_counts:  np.ndarray counts per class
    """
    counts = np.bincount(y_tr, minlength=n_classes)
    eps = 1e-6
    w = counts.max() / (counts + eps)
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    return class_weights, counts


def make_dataloaders(
    Xn_tr, ty_tr, y_tr,
    Xn_te, ty_te, y_te,
    batch=BATCH, shuffle_train=True
):
    """
    Build TensorDatasets and DataLoaders.
    Returns:
        tr_dl, te_dl
    """
    tr_ds = TensorDataset(
        torch.tensor(Xn_tr, dtype=torch.float32),
        torch.tensor(ty_tr, dtype=torch.int64),
        torch.tensor(y_tr, dtype=torch.int64),
    )
    te_ds = TensorDataset(
        torch.tensor(Xn_te, dtype=torch.float32),
        torch.tensor(ty_te, dtype=torch.int64),
        torch.tensor(y_te, dtype=torch.int64),
    )
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=shuffle_train)
    te_dl = DataLoader(te_ds, batch_size=batch, shuffle=False)
    return tr_dl, te_dl

def normalize_train_test(Xn_tr, Xn_te, method="zscore", eps=1e-8):
    """
    Normalize numeric columns using train statistics only.
    method: "zscore" (default) or "minmax".
    Returns:
        Xn_tr_norm, Xn_te_norm, params
        where params={"method":..., "mean":..., "std":..., "min":..., "max":...}
    """
    if method == "zscore":
        mean = Xn_tr.mean(axis=0)
        std  = Xn_tr.std(axis=0) + eps
        return (Xn_tr - mean)/std, (Xn_te - mean)/std, {"method":"zscore","mean":mean,"std":std}
    elif method == "minmax":
        scaler = MinMaxScaler()
        Xn_tr_n = scaler.fit_transform(Xn_tr)
        Xn_te_n = scaler.transform(Xn_te)
        return Xn_tr_n, Xn_te_n, {"method":"minmax","min":scaler.data_min_, "max":scaler.data_max_}
    else:
        raise ValueError(f"Unknown method: {method}")

def smote_nc_resample(Xn_tr_norm, ty_tr, y_tr, k_neighbors=5, random_state=42, sampling_strategy="auto"):
    """
    Apply SMOTE-NC to balance classes in the TRAINING data only.
    Inputs:
        Xn_tr_norm: (N, 5) normalized numeric features
        ty_tr:      (N,)   integer codes for 'Type' categorical feature
        y_tr:       (N,)   multiclass labels in [0..5]
    Returns:
        Xn_tr_rs, ty_tr_rs, y_tr_rs
    """
    # build X matrix: [num | cat]
    import numpy as _np
    X_tr = _np.concatenate([Xn_tr_norm, ty_tr.reshape(-1, 1).astype(_np.float32)], axis=1)

    # index of categorical column in X_tr is the last column (5 numeric, 1 categorical)
    cat_idx = [X_tr.shape[1] - 1]

    smote = SMOTENC(
        categorical_features=cat_idx,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    X_rs, y_rs = smote.fit_resample(X_tr, y_tr)

    # split back out
    Xn_tr_rs = X_rs[:, :Xn_tr_norm.shape[1]].astype(_np.float32)
    ty_tr_rs = X_rs[:,  Xn_tr_norm.shape[1]].astype(_np.int64)
    y_tr_rs  = y_rs.astype(_np.int64)

    return Xn_tr_rs, ty_tr_rs, y_tr_rs

# Main Function: One-call pipeline
def prepare_datasets(
    csv_path,
    batch=BATCH,
    test_size=0.20,
    random_state=42,
    validate_schema=True,
    normalize="zscore",          # "zscore" (default) or "minmax"
    resample=None,               # None or "smote"
    smote_k_neighbors=5,         # Number of neighbors for SMOTE
    smote_random_state=42,       # Random seed for reproducibility
    smote_sampling="auto",       # Sampling strategy for SMOTE
):
    """
    End-to-end preprocessing:
      1) read CSV
      2) build binary & multi-class targets
      3) encode numeric + categorical features
      4) stratified split on multi-class target
      5) normalize numeric features (train stats)
      6) optional: apply SMOTE-NC on training split only
      7) compute balanced class weights
      8) create DataLoaders

    Returns a dict with everything the training/eval code needs:
        {
          "tr_dl", "te_dl", "class_weights", "class_counts",
          "type_vocab", "mean", "std",
          "NUM_COLS", "CLASS_NAMES", "N_CLASSES",
          "DEVICE", "D_MODEL", "NHEAD", "TARGET"
        }
    """

    # 1) Read CSV and validate schema
    df = read_ai4i_csv(csv_path, validate_schema=validate_schema)

    # 2) Build binary & multi-class targets
    y_bin, y_cls = build_targets(df)

    # 3) Encode numeric + categorical features
    X_num, cat_idx, type_vocab = encode_features(df)

    # 4) Stratified split on multi-class target
    Xn_tr, Xn_te, ty_tr, ty_te, y_tr, y_te, yb_tr, yb_te = split_train_test(
        X_num, cat_idx, y_cls, y_bin,
        test_size=test_size, random_state=random_state
    )

    # 5) Normalize numeric features using train statistics only
    Xn_tr, Xn_te, norm_params = normalize_train_test(Xn_tr, Xn_te, method=normalize)
    mean = norm_params.get("mean")
    std  = norm_params.get("std")

    # 6) Optional: Apply SMOTE-NC to handle class imbalance (training split only)
    if resample == "smote":
        Xn_tr, ty_tr, y_tr = smote_nc_resample(
            Xn_tr, ty_tr, y_tr,
            k_neighbors=smote_k_neighbors,
            random_state=smote_random_state,
            sampling_strategy=smote_sampling
        )

    # 7) Compute balanced class weights based on #6 training set
    class_weights, class_counts = compute_class_weights(y_tr, n_classes=N_CLASSES, device=DEVICE)

    # 8) Create DataLoaders for training & testing
    tr_dl, te_dl = make_dataloaders(Xn_tr, ty_tr, y_tr, Xn_te, ty_te, y_te, batch=batch)

    # Return all relevant objects for training/eval scripts
    return {
        # Data
        "tr_dl": tr_dl,
        "te_dl": te_dl,
        "class_weights": class_weights,
        "class_counts": class_counts,

        # Metadata for callers
        "type_vocab": int(type_vocab),
        "mean": mean,
        "std": std,

        # keep these accessible
        "NUM_COLS": NUM_COLS,
        "CLASS_NAMES": CLASS_NAMES,
        "N_CLASSES": N_CLASSES,
        "DEVICE": DEVICE,
        "D_MODEL": D_MODEL,
        "NHEAD": NHEAD,
        "TARGET": TARGET,
    }



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test for preprocess pipeline.")
    parser.add_argument("--csv", required=True, help="Path to AI4I-style CSV.")
    args = parser.parse_args()

    payload = prepare_datasets(args.csv)
    tr_dl, te_dl = payload["tr_dl"], payload["te_dl"]
    print("OK: dataloaders ready.")
    for xb_num, xb_type, yb in te_dl:
        print("Batch shapes:", xb_num.shape, xb_type.shape, yb.shape)
        break