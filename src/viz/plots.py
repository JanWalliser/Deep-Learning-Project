from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
import torch

def plot_confusion_matrix(
    cm: torch.Tensor,
    out_path: str,
    title: str,
    class_names: Optional[list[str]] = None,
    normalize: bool = False,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mat = cm.detach().cpu().numpy()
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(mat)
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    n = mat.shape[0]
    ticks = np.arange(n)
    plt.xticks(ticks, class_names if class_names else [str(i) for i in ticks])
    plt.yticks(ticks, class_names if class_names else [str(i) for i in ticks])
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_histogram(values, out_path: str, title: str, xlabel: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 4))
    v = np.asarray(values, dtype=float)
    plt.hist(v, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
