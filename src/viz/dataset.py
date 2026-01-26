from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_class_distribution(
    labels: Sequence[int],
    num_classes: int,
    out_path: str,
    title: str = "Class distribution",
    class_names: Optional[list[str]] = None,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes)

    x = np.arange(num_classes)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(x, counts)
    plt.xticks(x, class_names if class_names else [str(i) for i in x])
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
