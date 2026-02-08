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


def save_examples_grid(
    examples,
    out_path: str,
    title: str,
    cols: int = 5,
) -> None:
    """Save a small grid of MNIST examples (for W&B Media).

    `examples` is expected to be a list of objects with:
      - image: Tensor [1,28,28] or [28,28] (0..1)
      - y_true, y_pred (int)
      - conf (float)
    """

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if examples is None or len(examples) == 0:
        return

    n = len(examples)
    cols = max(1, int(cols))
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(cols * 2.2, rows * 2.3))
    plt.suptitle(title)

    for i, ex in enumerate(examples, start=1):
        ax = plt.subplot(rows, cols, i)
        img = ex.image
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            arr = t.numpy()
        else:
            arr = np.asarray(img)

        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        ax.set_title(f"t={ex.y_true}  p={ex.y_pred}\nconf={ex.conf:.3f}", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overlaid_histogram(
    a,
    b,
    out_path: str,
    title: str,
    xlabel: str,
    label_a: str,
    label_b: str,
    bins: int = 40,
) -> None:
    """Overlaid histogram (two distributions) saved to PNG."""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 4))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    plt.hist(a, bins=bins, alpha=0.6, label=label_a)
    plt.hist(b, bins=bins, alpha=0.6, label=label_b)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
