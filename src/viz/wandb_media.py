# src/viz/wandb_media.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """
    Robust: Tensor -> uint8 for W&B.
    Handles MNIST tensors: [1,28,28] or [28,28] or even [B,1,28,28] (we take first).
    Rescales per-image to [0,255].
    """
    t = img.detach().cpu()

    # If batched, take first
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)

    t_min = float(t.min())
    t_max = float(t.max())
    if t_max > t_min:
        t = (t - t_min) / (t_max - t_min)
    else:
        t = torch.zeros_like(t)

    arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def to_wandb_image(x: Any):
    import wandb

    if isinstance(x, torch.Tensor):
        return wandb.Image(_tensor_to_uint8(x))
    return wandb.Image(x)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log_confusion_matrix_image(
    wandb_run,
    cm: torch.Tensor,
    epoch: int,
    step: int,
    run_id: str,
    normalize_fn,
    plot_fn,
    key: str = "val/confusion_matrix_img",
):
    """
    Saves and logs confusion matrix as PNG image.
    """
    import wandb

    cm_norm = normalize_fn(cm, mode="true")
    out_path = ensure_dir(f"outputs/reports/{run_id}/cm/cm_epoch_{epoch:02d}.png")

    plot_fn(cm_norm, str(out_path), title=f"Val Confusion (epoch {epoch})", normalize=True)
    wandb_run.log({key: wandb.Image(str(out_path)), "epoch": epoch}, step=step)


def log_confusion_matrix_plot(
    wandb_run,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: List[str],
    epoch: int,
    step: int,
    key: str = "val/confusion_matrix_plot",
):
    """
    Logs W&B interactive confusion matrix plot.
    """
    import wandb

    wandb_run.log(
        {
            key: wandb.plot.confusion_matrix(
                y_true=y_true.numpy().tolist(),
                preds=y_pred.numpy().tolist(),
                class_names=class_names,
            ),
            "epoch": epoch,
        },
        step=step,
    )


def log_prediction_tables(
    wandb_run,
    parts: Dict[str, list],
    step: int,
    max_wrong: int = 24,
):
    """
    parts expected like:
      {"correct": List[Example], "wrong": List[Example]}
    Each Example should have: image, y_true, y_pred, conf.
    """
    import wandb
    from src.viz.predictions import top_k_by_conf

    top3 = top_k_by_conf(parts.get("correct", []), 3, highest=True)
    low3 = top_k_by_conf(parts.get("correct", []), 3, highest=False)
    wrong = top_k_by_conf(parts.get("wrong", []), max_wrong, highest=True)

    def _to_table(rows, name: str):
        wandb_run.log({f"{name}_rows": int(len(rows))}, step=step)
        if len(rows) == 0:
            return

        table = wandb.Table(columns=["image", "true", "pred", "conf"])
        for ex in rows:
            table.add_data(
                to_wandb_image(ex.image),
                int(ex.y_true),
                int(ex.y_pred),
                float(ex.conf),
            )
        wandb_run.log({name: table}, step=step)

    _to_table(top3, "val/top3_confident")
    _to_table(low3, "val/low3_confidence")
    _to_table(wrong, "val/wrong_highconf")


def log_highconf_wrong_media(
    wandb_run,
    wrong_examples: list,
    epoch: int,
    step: int,
    run_id: str,
    k: int = 5,
    key_prefix: str = "val",
):
    """Log the top-k highest-confidence *wrong* predictions as W&B media.

    This is often more visible in W&B than a table, and is exactly what you want
    to inspect when diagnosing minority neglect/overconfidence.
    """

    import wandb
    from src.viz.predictions import top_k_by_conf
    from src.viz.plots import save_examples_grid

    wrong_top = top_k_by_conf(wrong_examples, k=k, highest=True)
    if not wrong_top:
        wandb_run.log({f"{key_prefix}/wrong_highconf_count": 0, "epoch": epoch}, step=step)
        return

    # Individual images (carousel-like)
    imgs = []
    for ex in wrong_top:
        cap = f"t={int(ex.y_true)} p={int(ex.y_pred)} conf={float(ex.conf):.3f}"
        imgs.append(wandb.Image(_tensor_to_uint8(ex.image), caption=cap))

    wandb_run.log({f"{key_prefix}/wrong_highconf_images": imgs, "epoch": epoch}, step=step)

    # Grid image (single artifact)
    out_path = ensure_dir(f"outputs/reports/{run_id}/examples/wrong_highconf_epoch_{epoch:02d}.png")
    save_examples_grid(wrong_top, str(out_path), title=f"Top-{k} high-conf wrong (epoch {epoch})", cols=k)
    wandb_run.log({f"{key_prefix}/wrong_highconf_grid": wandb.Image(str(out_path)), "epoch": epoch}, step=step)
