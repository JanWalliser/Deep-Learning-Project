from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_reliability(bin_centers, bin_acc, bin_conf, out_path: str, title: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    x = np.asarray(bin_centers, dtype=float)
    a = np.asarray(bin_acc, dtype=float)
    c = np.asarray(bin_conf, dtype=float)

    plt.plot([0, 1], [0, 1])
    plt.scatter(x, a, label="Accuracy per bin")
    plt.scatter(x, c, label="Confidence per bin")
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy / Confidence")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_confidence_hist(conf, correct_mask, out_path: str, title: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 4))
    conf = np.asarray(conf, dtype=float)
    correct_mask = np.asarray(correct_mask, dtype=bool)

    plt.hist(conf[correct_mask], bins=20, alpha=0.6, label="correct")
    plt.hist(conf[~correct_mask], bins=20, alpha=0.6, label="wrong")
    plt.title(title)
    plt.xlabel("max softmax probability")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
