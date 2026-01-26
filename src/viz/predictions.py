from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F

@dataclass
class PredExample:
    image: torch.Tensor          # [1,28,28] on CPU, unnormalized 0..1
    y_true: int
    y_pred: int
    conf: float                  # max softmax prob
    entropy: float               # predictive entropy
    margin: float                # p_top1 - p_top2 (multiclass)

def unnormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    # expects normalized MNIST: mean=0.1307, std=0.3081
    mean, std = 0.1307, 0.3081
    x = x * std + mean
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_items: Optional[int] = None,
) -> List[PredExample]:
    model.eval()
    out: List[PredExample] = []

    seen = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1)

        conf, pred = probs.max(dim=1)
        # entropy
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=1)

        # margin = top1 - top2
        top2 = torch.topk(probs, k=min(2, probs.size(1)), dim=1).values
        if top2.size(1) == 1:
            margin = torch.zeros_like(conf)
        else:
            margin = top2[:, 0] - top2[:, 1]

        x_cpu = unnormalize_mnist(x.detach().cpu())
        y_cpu = y.detach().cpu()
        pred_cpu = pred.detach().cpu()
        conf_cpu = conf.detach().cpu()
        entropy_cpu = entropy.detach().cpu()
        margin_cpu = margin.detach().cpu()

        for i in range(x_cpu.size(0)):
            out.append(PredExample(
                image=x_cpu[i],
                y_true=int(y_cpu[i].item()),
                y_pred=int(pred_cpu[i].item()),
                conf=float(conf_cpu[i].item()),
                entropy=float(entropy_cpu[i].item()),
                margin=float(margin_cpu[i].item()),
            ))
            seen += 1
            if max_items is not None and seen >= max_items:
                return out

    return out

def split_examples(examples: List[PredExample]) -> Dict[str, List[PredExample]]:
    wrong = [e for e in examples if e.y_pred != e.y_true]
    correct = [e for e in examples if e.y_pred == e.y_true]
    return {"wrong": wrong, "correct": correct}

def top_k_by_conf(examples: List[PredExample], k: int, highest: bool = True) -> List[PredExample]:
    return sorted(examples, key=lambda e: e.conf, reverse=highest)[:k]
