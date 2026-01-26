from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def weight_norms(model: torch.nn.Module, only_weights: bool = True) -> Dict[str, float]:
    out = {}
    for name, p in model.named_parameters():
        if only_weights and not name.endswith("weight"):
            continue
        out[name] = float(torch.norm(p.detach()).item())
    return out

@torch.no_grad()
def plot_weight_histograms(model: torch.nn.Module, out_path: str, max_tensors: int = 6) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    items = [(n, p.detach().cpu().flatten().numpy()) for n, p in model.named_parameters() if n.endswith("weight")]
    items = items[:max_tensors]

    for n, w in items:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(w, bins=50)
        plt.title(f"Weight histogram: {n}")
        plt.tight_layout()
        fig.savefig(str(Path(out_path).with_name(f"{Path(out_path).stem}_{n.replace('.', '_')}.png")), dpi=200)
        plt.close(fig)
