from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate(model, loader, device, collect_outputs: bool = False):
    """Evaluate classification model.

    Returns either:
      metrics dict
    or
      (metrics dict, y_true, y_pred, logits)
    """

    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    all_y, all_pred, all_logits = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)

        if collect_outputs:
            all_y.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())
            all_logits.append(logits.detach().cpu())

    metrics = {"loss": total_loss / n, "acc": correct / n}
    if not collect_outputs:
        return metrics

    return metrics, torch.cat(all_y), torch.cat(all_pred), torch.cat(all_logits)
