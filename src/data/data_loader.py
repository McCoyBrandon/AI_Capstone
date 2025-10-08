"""
TabTransformer-style classifier for the AI4I 2020 dataset.
Reference: 

"""

# Required Packages
import numpy as np                
import pandas as pd               
import torch                     
import torch.nn as nn             
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

### Starting Variables
## Variables for the data
CSV_PATH = "src.data.ai4i2020.csv"         # Expect the UCI CSV to be in the same folder, will need to adjust when we restructure the folders.
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

### Load & Prepare Data
# Read the CSV into a pandas DataFrame
df = pd.read_csv(CSV_PATH)
print(sorted(df.columns)) # DEBUGGGING dataset column check

# DEBUGGING for column name issues
required_cols = set(NUM_COLS + [CAT_COL, TARGET] + FAILURE_COLS)
missing = list(required_cols - set(df.columns))
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# Extract features
X_num = df[NUM_COLS].values.astype(np.float32) # Numeric
cat_idx = df[CAT_COL].astype("category").cat.codes.values.astype(np.int64) # Convert "Type" to categorical codes (L = 0, M = 1, H = 2)
type_vocab = int(cat_idx.max()) + 1

# Extract target as float32: shape [N]; BCEWithLogitsLoss expects float targets
y_bin = df[TARGET].values.astype(np.int64) # 0/1 (used for binary AUROC and for building y_cls)

# Build 6-class target 0..5: 0=NoFailure, 1..5 = TWF/HDF/PWF/OSF/RNF
fails = df[FAILURE_COLS].values.astype(np.int64) # shape [N,5]
y_cls = np.zeros(len(df), dtype=np.int64) # default 0 (NoFailure)
mask_fail = (y_bin == 1)
if mask_fail.any():
    # Resolve multi-flag rows with fixed priority
    # Priority based opn FAILURE_COLS order: TWF > HDF > PWF > OSF > RNF
    priority = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    sub = fails[mask_fail]
    # choose first True in priority order; if none set (shouldn't happen when y_bin==1), default to 0
    chosen = np.argmax(sub[:, priority] == 1, axis=1)
    none_set = (sub.sum(axis=1) == 0)
    chosen[none_set] = 0
    y_cls[mask_fail] = chosen + 1
    
# DEBUGGING Inspect multi-flag rows count
multi_mask = (fails.sum(axis=1) > 1) & mask_fail
multi_count = int(multi_mask.sum())
if multi_count > 0:
    print(f"[DEBUGGING] {multi_count} rows have multiple failure types; resolved by priority {FAILURE_COLS}.")

## Train/Test Split (80/20, random)
N = len(df)                         # total number of rows
idx = np.random.permutation(N)      # random indices
n_train = int(0.8 * N)              # 80% for training
tr, te = idx[:n_train], idx[n_train:]  # train indices, test indices

# Stratify by y_cls
Xn_tr, Xn_te, ty_tr, ty_te, y_tr, y_te, yb_tr, yb_te = train_test_split(
X_num, cat_idx, y_cls, y_bin, test_size=0.20, random_state=42, stratify=y_cls 
)

# Standardize numeric features using TRAIN statistics only
mean, std = Xn_tr.mean(axis=0), Xn_tr.std(axis=0) + 1e-8
Xn_tr = (Xn_tr - mean) / std
Xn_te = (Xn_te - mean) / std

## Class imbalance handling
# Compute class counts on TRAIN and derive weights
class_counts = np.bincount(y_tr, minlength=N_CLASSES)
eps = 1e-6
class_weights_np = class_counts.max() / (class_counts + eps) # Compute class weights
class_weights_np = class_weights_np / class_weights_np.mean() # Normalize class weights
class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)

# Tensors & DataLoaders
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

# DataLoaders handle batching and shuffling
tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)   # weighted sampling for class balance
te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False)  # no need to shuffle test

