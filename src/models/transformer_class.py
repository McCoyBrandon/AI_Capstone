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
D_MODEL = 64                      # Hidden layer size (number of nodes)
NHEAD = 2                         # Number of attention heads (must divide D_MODEL)
# Processing managagement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if present, else CPU

### Model Definition
class TinyTabTransformer(nn.Module):
    def __init__(
        self,
        n_num,
        type_vocab,
        d_model: int = 64,
        nhead: int = 2,
        dim_feedforward: int = 128,   
        dropout: float = 0.0,         
        num_layers: int = 2,          
        activation: str = "relu",    
    ):
        super().__init__()

        self.n_num          = n_num
        self.type_vocab     = type_vocab
        self.d_model        = d_model
        self.nhead          = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout        = dropout
        self.num_layers     = num_layers
        self.activation     = activation

        # Per-feature linear projection implemented as a learned affine transform:
        # For each numeric feature i, we store W[i] (size d_model) and b[i] (size d_model).
        # Given scalar x_i, the token is: x_i * W[i] + b[i]  -> a vector in R^{d_model}.
        self.W = nn.Parameter(torch.randn(n_num, d_model) * 0.02)  # [n_num, d_model]
        self.b = nn.Parameter(torch.zeros(n_num, d_model))         # [n_num, d_model]

        # Embedding for the categorical "Type" feature
        self.type_emb = nn.Embedding(type_vocab, d_model)          

        # Learned classification [CLS] token (1 x 1 x d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder layer 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,  
            dropout=self.dropout,                  
            activation=self.activation,            
            batch_first=True,                      
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,           
        )

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
        logits = self.head(enc[:, 0, :])               # [B, N_CLASSES]
        return logits


