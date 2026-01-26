from __future__ import annotations
import torch

@torch.no_grad()
def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    y_true = y_true.to(torch.int64).view(-1)
    y_pred = y_pred.to(torch.int64).view(-1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

@torch.no_grad()
def normalize_confusion_matrix(cm: torch.Tensor, mode: str = "true") -> torch.Tensor:
    cm = cm.to(torch.float32)
    if mode == "true":     # row-normalized -> recall per class
        denom = cm.sum(dim=1, keepdim=True).clamp_min(1.0)
        return cm / denom
    if mode == "pred":     # col-normalized -> precision per predicted class
        denom = cm.sum(dim=0, keepdim=True).clamp_min(1.0)
        return cm / denom
    if mode == "all":
        denom = cm.sum().clamp_min(1.0)
        return cm / denom
    raise ValueError(f"Unknown mode: {mode}")

@torch.no_grad()
def per_class_recall_from_cm(cm: torch.Tensor) -> torch.Tensor:
    cm = cm.to(torch.float32)
    denom = cm.sum(dim=1).clamp_min(1.0)
    return cm.diag() / denom

@torch.no_grad()
def per_class_precision_from_cm(cm: torch.Tensor) -> torch.Tensor:
    cm = cm.to(torch.float32)
    denom = cm.sum(dim=0).clamp_min(1.0)
    return cm.diag() / denom
