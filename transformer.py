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
from sklearn.metrics import  accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score

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
# !!!! Will need to feature review to make sure values are properly distributed for statistical significance
N = len(df)                         # total number of rows
idx = np.random.permutation(N)      # random indices
n_train = int(0.7 * N)              # 70% for training
tr, te = idx[:n_train], idx[n_train:]  # train indices, test indices

""""
###
# Commented out section to develope multi-classification retaining in case of need.
###
# Slice arrays into train/test sets
Xn_tr, Xn_te = X_num[tr], X_num[te]
ty_tr, ty_te = type_idx[tr], type_idx[te]
y_tr,  y_te  = y[tr], y[te]

# Standardization
# Compute mean and std for training data
m, s = Xn_tr.mean(0), Xn_tr.std(0) + 1e-8   # add epsilon to avoid divide-by-zero

# Apply standardization to train and test (test uses train's m and s)
Xn_tr = (Xn_tr - m) / s
Xn_te = (Xn_te - m) / s

## Build Tensors and DataLoaders
# Wrap numpy arrays into torch.Tensors; keep dtypes consistent with the model:
# - numeric features float32, categorical indices int64, targets float32
tr_ds = TensorDataset(
    torch.tensor(Xn_tr),           # shape [N_train, 5], dtype float32
    torch.tensor(ty_tr),           # shape [N_train],    dtype int64
    torch.tensor(y_tr),            # shape [N_train],    dtype float32
)
te_ds = TensorDataset(
    torch.tensor(Xn_te),
    torch.tensor(ty_te),
    torch.tensor(y_te),
)
"""
# Stratify by y_cls
Xn_tr, Xn_te, ty_tr, ty_te, y_tr, y_te, yb_tr, yb_te = train_test_split(
X_num, cat_idx, y_cls, y_bin, test_size=0.20, random_state=42, stratify=y_cls 
)

# Standardize numeric features using TRAIN statistics only
mean, std = Xn_tr.mean(axis=0), Xn_tr.std(axis=0) + 1e-8
Xn_tr = (Xn_tr - mean) / std
Xn_te = (Xn_te - mean) / std

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
tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)   # shuffle test data
te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False)  # no need to shuffle test

### Model Definition
class TinyTabTransformer(nn.Module):
    def __init__(self, n_num, type_vocab, d_model=64, nhead=2):
        super().__init__()

        # Per-feature linear projection implemented as a learned affine transform:
        # For each numeric feature i, we store W[i] (size d_model) and b[i] (size d_model).
        # Given scalar x_i, the token is: x_i * W[i] + b[i]  -> a vector in R^{d_model}.
        self.W = nn.Parameter(torch.randn(n_num, d_model) * 0.02)  # [n_num, d_model]
        self.b = nn.Parameter(torch.zeros(n_num, d_model))         # [n_num, d_model]

        # Embedding for the categorical "Type" feature
        self.type_emb = nn.Embedding(type_vocab, d_model)          

        # Learned classification [CLS] token (1 x 1 x d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # Single Transformer encoder layer (self-attention + MLP) with GELU activation.
        # batch_first=True -> input/output shape is [B, T, d_model].
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,     # small MLP hidden layer inside the encoder
            dropout=0.0,             # May need to add dropout depending on evaluation
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)

        # Linear head that maps the [CLS] hidden state adjusted to the multi-classifaction
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x_num, x_type):
        """
        x_num:  [B, n_num]  float32 standardized numeric features
        x_type: [B]        int64 category for "Type"
        returns: [B]       raw logits (pre-sigmoid)
        """
        B, n_num = x_num.shape  # batch size and number of numeric features

        # Create numeric tokens by applying the per-feature affine transform:
        # x_num.unsqueeze(-1): [B, n_num, 1]
        # self.W.unsqueeze(0): [1, n_num, d_model]  (broadcast to [B, n_num, d_model])
        # self.b.unsqueeze(0): [1, n_num, d_model]
        num_tok = x_num.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)  # [B, n_num, d_model]

        # Embed the categorical feature and make it a length-1 token sequence:
        type_tok = self.type_emb(x_type).unsqueeze(1)  # [B, 1, d_model]

        # Expand the learned [CLS] token across the batch:
        cls = self.cls.expand(B, 1, -1)                # [B, 1, d_model]

        # Concatenate tokens along sequence dimension: [CLS] + numeric tokens + type token
        seq = torch.cat([cls, num_tok, type_tok], dim=1)  # [B, (1 + n_num + 1), d_model]

        # Pass through the Transformer encoder (self-attention over the short sequence)
        enc = self.encoder(seq)                        # [B, T, d_model]

        # Take the [CLS] hidden state (position 0) and map to a single logit
        logit = self.head(enc[:, 0, :]).squeeze(-1)    # [B]
        return logit

# Compute the categorical vocabulary size from the data (max index + 1)
type_vocab = int(cat_idx.max()) + 1

# Instantiate the model and move to device
model = TinyTabTransformer(len(NUM_COLS), type_vocab, D_MODEL, NHEAD).to(DEVICE)

### Optimizer & Loss evaluation
# Adam optimizer over all model parameters
opt = torch.optim.Adam(model.parameters(), lr=LR)

# Adjusted loss to suit for a multi-class
loss_fn = nn.CrossEntropyLoss() 

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
# Using the new multi-class evaluation with a Binary check at the end.
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

# Core metrics
acc  = accuracy_score(all_y, preds)
bacc = balanced_accuracy_score(all_y, preds)
print(f"Test accuracy: {acc:.4f}")
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


"""
###
# Binary classifcation saved for reference, may remove later
###
model.eval()                        # set eval mode (e.g., disables dropout; not used here)
correct = 0                         # Track number of correct predictions
tot = 0                             # Track total number of samples

with torch.no_grad():               # no gradient tracking needed for evaluation
    for xb_num, xb_type, yb in te_dl:
        xb_num  = xb_num.to(DEVICE)
        xb_type = xb_type.to(DEVICE)
        yb      = yb.to(DEVICE)

        # Model returns raw logits; apply sigmoid to get probabilities in [0,1]
        probs = torch.sigmoid(model(xb_num, xb_type))  # [B]
        # Convert probabilities to class predictions using 0.5 threshold
        preds = (probs >= 0.5).float()                 # [B], 0.0 or 1.0

        # Count how many match the ground truth
        correct += (preds == yb).sum().item()
        tot += yb.numel()

# Output final accuracy on the test split
print(f"Test accuracy: {correct / tot:.4f}")
"""