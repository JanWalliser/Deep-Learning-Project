from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import math
import matplotlib.pyplot as plt
import torch

@dataclass
class GradSnapshot:
    step: int
    global_grad_norm: float
    global_gwr: float
    per_param_grad_norm: Dict[str, float]
    per_param_gwr: Dict[str, float]

@torch.no_grad()
def compute_grad_stats(model: torch.nn.Module, eps: float = 1e-12) -> tuple[float, float, Dict[str, float], Dict[str, float]]:
    per_g = {}
    per_r = {}
    g_sq = 0.0
    r_sq = 0.0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        w = p.detach()

        g_norm = float(torch.norm(g).item())
        w_norm = float(torch.norm(w).item())
        ratio = g_norm / (w_norm + eps)

        per_g[name] = g_norm
        per_r[name] = ratio

        g_sq += g_norm * g_norm
        r_sq += ratio * ratio

    return math.sqrt(g_sq), math.sqrt(r_sq), per_g, per_r

def plot_grad_history(history: List[GradSnapshot], out_prefix: str, topk: int = 6) -> None:
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    steps = [h.step for h in history]
    g = [h.global_grad_norm for h in history]
    r = [h.global_gwr for h in history]

    fig = plt.figure(figsize=(10, 4))
    plt.plot(steps, g)
    plt.title("Global Gradient Norm")
    plt.xlabel("Step")
    plt.ylabel("L2 norm")
    plt.tight_layout()
    fig.savefig(out_prefix + "_gradnorm.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(steps, r)
    plt.title("Global Gradient-to-Weight Ratio")
    plt.xlabel("Step")
    plt.ylabel("||g|| / ||w||")
    plt.tight_layout()
    fig.savefig(out_prefix + "_gwr.png", dpi=200)
    plt.close(fig)

    # Top-k params by max gwr
    names = set()
    for h in history:
        names |= set(h.per_param_gwr.keys())
    names = list(names)
    max_by = {n: max(h.per_param_gwr.get(n, 0.0) for h in history) for n in names}
    top = sorted(names, key=lambda n: max_by[n], reverse=True)[:topk]

    fig = plt.figure(figsize=(10, 6))
    for n in top:
        series = [h.per_param_gwr.get(n, 0.0) for h in history]
        plt.plot(steps, series, label=n)
    plt.title(f"Top-{topk} Parameter GWR")
    plt.xlabel("Step")
    plt.ylabel("||g|| / ||w||")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_prefix + "_gwr_params.png", dpi=200)
    plt.close(fig)
