# plot_val_metrics.py
# Makes 3 near-square PNG plots (each with 3 curves: balanced / severe / extreme):
#   1) val accuracy (overall)
#   2) val recall class 0
#   3) val recall class 1
#
# X-axis is "Epoch" compressed to 0..20 by linearly mapping Step -> Epoch via:
#   epoch = step / (max_step / 20)
# (no clipping; just rescaling the axis)

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# CSV loading (W&B export friendly)
# ----------------------------
def _find_step_col(df: pd.DataFrame) -> str:
    for c in ["Step", "step", "global_step", "Global Step", "globalStep", "epoch", "Epoch"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _canonicalize_run_col(c: str) -> str:
    """
    W&B CSV often uses: "<run name> - <metric path>".
    We keep only "<run name>" so different CSVs align on columns.
    """
    c = c.strip()
    if " - " in c:
        c = c.split(" - ", 1)[0]
    return c.strip()


def load_wandb_metric_csv(path: str | Path) -> pd.DataFrame:
    """
    Returns df indexed by Step, with one column per run.
    Drops __MIN/__MAX columns (W&B aggregates).
    """
    df = pd.read_csv(path)
    step_col = _find_step_col(df)
    df = df.rename(columns={step_col: "Step"})

    # drop unnamed index cols
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # keep numeric columns besides Step
    keep = ["Step"] + [
        c for c in df.columns
        if c != "Step" and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df[keep]

    # drop W&B __MIN/__MAX columns (keep only the main metric)
    metric_cols = [c for c in df.columns if c != "Step"]
    metric_cols = [c for c in metric_cols if not (c.endswith("__MIN") or c.endswith("__MAX"))]
    df = df[["Step"] + metric_cols]

    df = df.set_index("Step").sort_index()

    # canonicalize columns to run name
    df = df.rename(columns={c: _canonicalize_run_col(c) for c in df.columns})

    # merge columns that became identical after canonicalization
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T

    return df


# ----------------------------
# Plotting helpers
# ----------------------------
def step_index_to_epoch_index(step_index: pd.Index, target_epochs: int = 20) -> np.ndarray:
    """
    Compresses Step -> Epoch in [0, target_epochs] using linear scaling
    based on max(step_index). Does NOT clip.
    """
    steps = step_index.to_numpy(dtype=float)
    max_step = float(np.nanmax(steps))
    if max_step <= 0:
        return steps  # fallback (shouldn't happen)
    step_per_epoch = max_step / float(target_epochs)
    return steps / step_per_epoch


def pretty_label(run_name: str) -> str:
    """
    Converts run names like 'cnn_extreme_seed42' into legend labels:
    'extreme', 'severe', 'balanced' (fallback: run_name).
    """
    s = run_name.lower()
    for key in ["balanced", "severe", "extreme"]:
        if key in s:
            return key
    return run_name


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures legend order: balanced, severe, extreme (if present).
    """
    wanted = ["balanced", "severe", "extreme"]
    cols = list(df.columns)

    def score(c: str) -> int:
        lab = pretty_label(c)
        return wanted.index(lab) if lab in wanted else 999

    return df[sorted(cols, key=score)]


def save_square_plot(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    out_path: str | Path,
    target_epochs: int = 20,
    ylim=None,
    dpi: int = 300,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))  # near-square

    df = order_columns(df)

    for col in df.columns:
        y = df[col]
        y_valid = y.dropna()
        if y_valid.empty:
            continue

        # Per-run scaling: map this run's last available step to target_epochs
        max_step_col = float(y_valid.index.max())
        if max_step_col <= 0:
            continue

        steps = y_valid.index.to_numpy(dtype=float)
        epoch_x = steps * (target_epochs / max_step_col)

        ax.plot(epoch_x, y_valid.to_numpy(), linewidth=2, label=pretty_label(col))

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    ax.set_xlim(0, target_epochs)
    ax.set_xticks([0, 5, 10, 15, 20])

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncols=1, loc="best")

    fig.tight_layout()
    fig.subplots_adjust(left=0.20, bottom=0.12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print("Saved:", out_path)



# ----------------------------
# Main
# ----------------------------
def main():
    # --- EDIT THESE PATHS ---
    ACC_CSV = "CNN/accuracy.csv"
    R0_CSV = "CNN/recall class 0.csv"
    R1_CSV = "CNN/recall class 1.csv"

    OUT_DIR = Path("plots_val")
    PREFIX = "cnn_val"  # change to "mlp_val" etc.

    # load
    acc = load_wandb_metric_csv(ACC_CSV)
    r0 = load_wandb_metric_csv(R0_CSV)
    r1 = load_wandb_metric_csv(R1_CSV)

    # 3 plots (each contains 3 curves)
    save_square_plot(
        acc,
        title="Validation Accuracy (overall)",
        ylabel="Accuracy",
        out_path=OUT_DIR / f"{PREFIX}_acc.png",
        target_epochs=20,
        ylim=(0.0, 1.0),
    )

    save_square_plot(
        r0,
        title="Validation Recall Mayority (class 0)",
        ylabel="Recall (class 0)",
        out_path=OUT_DIR / f"{PREFIX}_recall_c0.png",
        target_epochs=20,
        ylim=(0.0, 1.0),
    )

    save_square_plot(
        r1,
        title="Validation Recall Minority (class 1)",
        ylabel="Recall (class 1)",
        out_path=OUT_DIR / f"{PREFIX}_recall_c1.png",
        target_epochs=20,
        ylim=(0.0, 1.0),
    )


if __name__ == "__main__":
    main()
