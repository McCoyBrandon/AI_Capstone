"""
TorchDrift integration helpers for the AI4I TinyTabTransformer pipeline.

This module lets you:
  - fit a TorchDrift detector on the training distribution
  - evaluate drift on a target distribution (e.g., test/val)
  - optionally log drift metrics to TensorBoard
"""
from typing import Dict, Optional
import torch
import torch.nn as nn

try:
    import torchdrift
    from torchdrift.detectors import KernelMMDDriftDetector
except ImportError:
    torchdrift = None
    KernelMMDDriftDetector = None


def _check_torchdrift_available():
    if torchdrift is None or KernelMMDDriftDetector is None:
        raise RuntimeError(
            "TorchDrift is not installed. Install it with:\n"
            "  pip install torchdrift\n"
            "and re-run with drift detection enabled."
        )


class TabularDriftFeatureExtractor(nn.Module):
    """
    Simple feature extractor for batches of the form:
        (xb_num, xb_type, yb)

    It concatenates numeric features with the categorical 'Type' code
    (cast to float) into a single feature vector per row.

    This is a lightweight, model-agnostic representation; later you can
    switch to TinyTabTransformer embeddings if you want.
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        xb_num, xb_type, *_ = batch
        xb_type_f = xb_type.float().unsqueeze(-1)  # [B, 1]
        return torch.cat([xb_num, xb_type_f], dim=1)  # [B, n_num+1]


def run_torchdrift_drift_check(
    tr_dl,
    te_dl,
    device: torch.device,
    writer: Optional["SummaryWriter"] = None,
    tb_step: int = 0,
    tb_prefix: str = "drift",
) -> Dict[str, float]:
    """
    Fit a TorchDrift detector on training data (reference distribution) and
    evaluate drift on test data (target distribution).

    Args
    ----
    tr_dl: DataLoader yielding (xb_num, xb_type, yb) for training data
    te_dl: DataLoader yielding (xb_num, xb_type, yb) for test/production-like data
    device: torch.device
    writer: optional TensorBoard SummaryWriter for logging
    tb_step: scalar step/index used for TensorBoard (e.g., epoch or run index)
    tb_prefix: scalar name prefix for drift scalars

    Returns
    -------
    drift_metrics: dict with summary stats
        {
            "score_mean", "score_min", "score_max",
            "pval_mean", "pval_min", "pval_max"
        }
    """
    _check_torchdrift_available()

    feature_extractor = TabularDriftFeatureExtractor().to(device)
    drift_detector = KernelMMDDriftDetector().to(device)

    # Fit reference distribution on training data
    torchdrift.utils.fit(tr_dl, feature_extractor, drift_detector, device=device)

    scores = []
    p_vals = []

    for batch in te_dl:
        features = feature_extractor(batch).to(device)
        # Detector returns a scalar score per batch
        score = drift_detector(features)
        p_val = drift_detector.compute_p_value(features)
        scores.append(score.detach().cpu())
        p_vals.append(p_val.detach().cpu())

    scores_t = torch.stack(scores)
    pvals_t = torch.stack(p_vals)

    metrics = {
        "score_mean": float(scores_t.mean().item()),
        "score_min": float(scores_t.min().item()),
        "score_max": float(scores_t.max().item()),
        "pval_mean": float(pvals_t.mean().item()),
        "pval_min": float(pvals_t.min().item()),
        "pval_max": float(pvals_t.max().item()),
    }

    if writer is not None:
        writer.add_scalar(f"{tb_prefix}/score_mean", metrics["score_mean"], tb_step)
        writer.add_scalar(f"{tb_prefix}/pval_mean", metrics["pval_mean"], tb_step)
        writer.add_scalar(f"{tb_prefix}/pval_min", metrics["pval_min"], tb_step)
        writer.add_scalar(f"{tb_prefix}/pval_max", metrics["pval_max"], tb_step)

    return metrics
