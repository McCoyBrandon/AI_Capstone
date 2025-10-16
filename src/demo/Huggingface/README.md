
# AI4I Failure Inference — Gradio Demo

This Space/app reuses your **TinyTabTransformer** checkpoint and meta to run interactive inference from sliders/dropdowns.

## Files expected
- `runs/ai4i_run_1/model.ckpt` — your trained checkpoint (`model_state_dict` + `meta`)
- `src/data/ai4i2020.csv` — source CSV to derive slider ranges and Type categories

You can change their locations using environment variables:
- `CKPT_PATH` (default: `runs/ai4i_run_1/model.ckpt`)
- `CSV_PATH`  (default: `src/data/ai4i2020.csv`)

## Local run
```bash
pip install -r requirements.txt
python app.py
```

## Hugging Face Spaces
1. Create a new Space (Gradio / Python).
2. Upload:
   - `app.py` (this file)
   - `requirements.txt`
   - Your `model.ckpt` at `runs/ai4i_run_1/model.ckpt` (or set env var)
   - Your `ai4i2020.csv` at `src/data/ai4i2020.csv` (or set env var)
3. Click "Restart" or "Deploy".

**Tip:** If you want a *binary-only* output, you already get the "Binary Decision" plus `P(any failure)`. The class table provides extra detail for debugging.
