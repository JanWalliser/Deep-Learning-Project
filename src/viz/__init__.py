from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import torch

@torch.no_grad()
def init_stats(model: torch.nn.Module) -> Dict[str, dict]:
    out = {}
    for name, p in model.named_parameters():
        t = p.detach().flatten()
        out[name] = {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "l2": float(torch.norm(p.detach()).item()),
            "numel": int(t.numel()),
        }
    return out

def save_init_stats(stats: Dict[str, dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
