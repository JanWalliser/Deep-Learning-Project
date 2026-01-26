from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def expected_calibration_error(logits: torch.Tensor, y_true: torch.Tensor, n_bins: int = 15) -> float:
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(y_true).to(torch.float32)

    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            ece += (mask.to(torch.float32).mean()) * (acc_bin - conf_bin).abs()
    return float(ece.item())

@torch.no_grad()
def reliability_curve(logits: torch.Tensor, y_true: torch.Tensor, n_bins: int = 15):
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(y_true).to(torch.float32)

    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_centers, bin_acc, bin_conf, bin_frac = [], [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        frac = float(mask.to(torch.float32).mean().item())
        if frac > 0:
            bin_centers.append(float(((lo + hi) / 2).item()))
            bin_acc.append(float(correct[mask].mean().item()))
            bin_conf.append(float(conf[mask].mean().item()))
            bin_frac.append(frac)

    return bin_centers, bin_acc, bin_conf, bin_frac
