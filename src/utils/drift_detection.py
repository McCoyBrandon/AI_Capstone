"""
TorchDrift integration helpers for the AI4I TinyTabTransformer pipeline.

This module:
  - fits a TorchDrift drift detector on the training distribution
  - evaluates drift on the test (or production-like) distribution
  - optionally logs drift metrics to TensorBoard
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


class TabularFeatureExtractor(nn.Module):
    """
    Simple feature extractor for batches of the form:
        (xb_num, xb_type, yb)

    It concatenates numeric features with the categorical 'Type' code
    (cast to float) into a single feature vector per row.
    """
    def __init__(self):
        super().__init__()

    def forward(self, xb_num: torch.Tensor, xb_type: torch.Tensor) -> torch.Tensor:
        """
        xb_num:  [B, n_num] float32
        xb_type: [B]       int64
        returns: [B, n_num + 1] float32
        """
        xb_type_f = xb_type.float().unsqueeze(-1)  # [B, 1]
        return torch.cat([xb_num, xb_type_f], dim=1)


def _collect_features(
    dl,
    feature_extractor: TabularFeatureExtractor,
    device: torch.device,
) -> torch.Tensor:
    """
    Iterate over a DataLoader that yields (xb_num, xb_type, yb),
    and collect all extracted features into a single tensor.
    """
    feats = []
    feature_extractor.eval()

    with torch.no_grad():
        for xb_num, xb_type, _ in dl:
            xb_num = xb_num.to(device)
            xb_type = xb_type.to(device)
            z = feature_extractor(xb_num, xb_type)  # [B, d]
            feats.append(z)

    if not feats:
        raise RuntimeError("DataLoader produced no batches when collecting features.")

    return torch.cat(feats, dim=0)  # [N, d]


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
    writer: TensorBoard SummaryWriter for logging
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

    feature_extractor = TabularFeatureExtractor().to(device)
    drift_detector = KernelMMDDriftDetector().to(device)

    # 1) Fit reference distribution on training data
    ref_feats = _collect_features(tr_dl, feature_extractor, device)  # [N_ref, d]
    drift_detector.fit(ref_feats)

    # 2) Evaluate drift on target/test data
    scores = []
    p_vals = []

    with torch.no_grad():
        for xb_num, xb_type, _ in te_dl:
            xb_num = xb_num.to(device)
            xb_type = xb_type.to(device)
            z = feature_extractor(xb_num, xb_type)  # [B, d]

            # For KernelMMDDriftDetector, calling it returns a scalar distance/score
            score = drift_detector(z)
            pval = drift_detector.compute_p_value(z)

            scores.append(score.cpu())
            p_vals.append(pval.cpu())

    if not scores:
        raise RuntimeError("Target DataLoader produced no batches when evaluating drift.")

    scores_t = torch.stack(scores)  # [num_batches]
    pvals_t = torch.stack(p_vals)   # [num_batches]

    metrics = {
        "score_mean": float(scores_t.mean().item()),
        "score_min": float(scores_t.min().item()),
        "score_max": float(scores_t.max().item()),
        "pval_mean": float(pvals_t.mean().item()),
        "pval_min": float(pvals_t.min().item()),
        "pval_max": float(pvals_t.max().item()),
    }

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar(f"{tb_prefix}/score_mean", metrics["score_mean"], tb_step)
        writer.add_scalar(f"{tb_prefix}/score_min",  metrics["score_min"],  tb_step)
        writer.add_scalar(f"{tb_prefix}/score_max",  metrics["score_max"],  tb_step)
        writer.add_scalar(f"{tb_prefix}/pval_mean",  metrics["pval_mean"],  tb_step)
        writer.add_scalar(f"{tb_prefix}/pval_min",   metrics["pval_min"],   tb_step)
        writer.add_scalar(f"{tb_prefix}/pval_max",   metrics["pval_max"],   tb_step)
    return metrics