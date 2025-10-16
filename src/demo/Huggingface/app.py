
"""
Gradio demo for a pretrained TinyTabTransformer on the AI4I 2020 dataset.

Run locally:
  pip install -r requirements.txt
  python app.py

Deploy on Hugging Face Spaces:
  - Set "SDK" = Gradio, "Python" runtime
  - Put your checkpoint at: runs/ai4i_run_1/model.ckpt (or update CKPT_PATH below)
  - Put your CSV at: src/data/ai4i2020.csv (or update CSV_PATH below)
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr

# Import your model exactly as in your repo style 
from transformer_class import TinyTabTransformer  

# ---- Paths (you can change these via HF "Files" tab or env vars) ----
CKPT_PATH = os.getenv("CKPT_PATH", "runs/ai4i_run_1/model.ckpt")
CSV_PATH  = os.getenv("CSV_PATH",  "src/data/ai4i2020.csv")

# Defaults 
NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CAT_COL = "Type"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#Load checkpoint and meta
def load_ckpt_and_meta(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        alt = ckpt_path.replace("runs/", "tests/")
        if os.path.exists(alt):
            ckpt_path = alt
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    meta = ckpt["meta"]
    return ckpt, meta


# Prepare feature ranges and category mapping from
def derive_ui_specs(csv_path: str, meta: dict):
    # Read CSV to pull min/max and categories (same approach as your demo.py)
    df = pd.read_csv(csv_path)

    # Sanity check: required columns present
    required_cols = set(meta["num_cols"] + [meta["cat_col"]])
    missing = list(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # Numeric min/max (for slider bounds)
    num_bounds = {}
    for c in meta["num_cols"]:
        s = df[c].astype(float)
        # pad bounds by a tiny margin so sliders are easy to move
        mn, mx = float(s.min()), float(s.max())
        pad = (mx - mn) * 0.01 if mx > mn else 1.0
        num_bounds[c] = (mn - pad, mx + pad, float(np.clip(s.median(), mn, mx)))

    # Categories for "Type" — we will compute codes using pandas to stay in sync
    types = list(pd.Categorical(df[meta["cat_col"]]).categories)
    if not types:
        # fallback common order for AI4I
        types = ["L", "M", "H"]
    return num_bounds, types


# Build model for inference
def build_model(meta: dict, ckpt: dict):
    model = TinyTabTransformer(
        n_num=len(meta["num_cols"]),
        type_vocab=int(meta["type_vocab"]),
        d_model=int(meta["d_model"]),
        nhead=int(meta["nhead"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# Inference helpers
def standardize_num(x_num: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.array(std, dtype=np.float32) + 1e-8
    mean = np.array(mean, dtype=np.float32)
    return (x_num.astype(np.float32) - mean) / std


def type_to_code(type_value: str, csv_path: str, meta: dict) -> int:
    # Create codes from CSV to mirror training mapping best-effort
    df = pd.read_csv(csv_path, usecols=[meta["cat_col"]])
    cat = pd.Categorical(df[meta["cat_col"]])
    categories = list(cat.categories)
    if type_value in categories:
        return int(pd.Categorical([type_value], categories=categories).codes[0])
    # fallback heuristic
    fallback = {"L":0, "M":1, "H":2}
    return int(fallback.get(type_value, 0))


def predict_once(inputs: dict, model: nn.Module, meta: dict):
    # Assemble numeric vector in the exact column order
    x_num = np.array([[inputs[c] for c in meta["num_cols"]]], dtype=np.float32)
    # Standardize with TRAIN stats from meta
    x_num = standardize_num(x_num, meta["standardize_mean"], meta["standardize_std"])

    # Encode the categorical Type
    t_code = type_to_code(inputs[meta["cat_col"]], CSV_PATH, meta)
    x_type = np.array([t_code], dtype=np.int64)

    # Torch tensors
    xb = torch.tensor(x_num, dtype=torch.float32, device=DEVICE)
    tb = torch.tensor(x_type, dtype=torch.int64, device=DEVICE)

    with torch.no_grad():
        logits = model(xb, tb)         # [1, C]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    class_names = meta["class_names"]     # ["NoFailure","TWF","HDF","PWF","OSF","RNF"]
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    p_any_failure = float(probs[1:].sum())

    # Binary failure decision
    fail_bool = pred_idx != 0 or p_any_failure >= 0.5
    decision = "FAILURE" if fail_bool else "NO FAILURE"

    # Return a tidy dict for rendering
    prob_table = {name: float(p) for name, p in zip(class_names, probs)}
    return decision, p_any_failure, pred_name, prob_table


#  Gradio UI 
def make_interface():
    ckpt, meta = load_ckpt_and_meta(CKPT_PATH)
    model = build_model(meta, ckpt)
    num_bounds, types = derive_ui_specs(CSV_PATH, meta)

    # Build dynamic components
    inputs = []
    for col, (mn, mx, med) in num_bounds.items():
        # Choose a reasonable step based on range magnitude
        rng = mx - mn
        step = max(rng / 500.0, 0.01)
        inputs.append(gr.Slider(mn, mx, value=med, step=step, label=col))

    type_dd = gr.Dropdown(choices=types, value=types[0], label=meta["cat_col"])

    def _predict(*vals):
        # vals => [num1, num2, num3, num4, num5, type_val]
        payload = {col: float(v) for (col, v) in zip(meta["num_cols"], vals[:-1])}
        payload[meta["cat_col"]] = vals[-1]
        decision, p_any, top_class, table = predict_once(payload, model, meta)

        # Format table into HTML for readability
        rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in table.items())
        html = f"""
        <table>
          <thead><tr><th>Class</th><th>Probability</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """
        return decision, f"{p_any:.4f}", top_class, html

    with gr.Blocks(title="AI4I TinyTabTransformer — Failure Inference") as demo:
        gr.Markdown("# AI4I Failure Inference\nProvide machine conditions to predict FAILURE vs NO FAILURE and view class probabilities.")
        with gr.Row():
            with gr.Column():
                for comp in inputs:
                    comp.render()
                type_dd.render()
                btn = gr.Button("Predict")
            with gr.Column():
                decision = gr.Label(label="Binary Decision")
                p_any = gr.Label(label="P(any failure)")
                top_class = gr.Label(label="Top Predicted Class")
                table = gr.HTML(label="Class Probabilities")
        btn.click(_predict, inputs=inputs + [type_dd], outputs=[decision, p_any, top_class, table])
    return demo

if __name__ == "__main__":
    demo = make_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
