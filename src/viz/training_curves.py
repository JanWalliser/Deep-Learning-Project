from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def plot_scalar_curves(history: Dict[str, List[float]], out_path: str, title: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 4))
    for k, v in history.items():
        plt.plot(np.arange(1, len(v) + 1), v, label=k)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_per_class_curve(per_class: List[List[float]], out_path: str, title: str) -> None:
    """
    per_class[c][epoch] = metric value
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(per_class, dtype=float)  # [C, E]
    fig = plt.figure(figsize=(10, 5))
    for c in range(arr.shape[0]):
        plt.plot(np.arange(1, arr.shape[1] + 1), arr[c], label=f"class {c}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
