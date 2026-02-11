# plot_grad_geometry.py
# Create paper-style plots from W&B-exported CSVs (pandas + matplotlib).
# You can adapt the PATHS dict at the bottom.

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def _find_step_col(df: pd.DataFrame) -> str:
    for c in ["Step", "step", "global_step", "Global Step", "globalStep"]:
        if c in df.columns:
            return c
    # fallback: first column
    return df.columns[0]


def load_wandb_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    step_col = _find_step_col(df)
    df = df.rename(columns={step_col: "Step"})

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    keep = ["Step"] + [
        c for c in df.columns
        if c != "Step" and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df[keep].set_index("Step").sort_index()

    if df.shape[1] == 1:
        df = df.rename(columns={df.columns[0]: Path(path).stem})

    def canonicalize_run_col(c: str) -> str:
        c = c.strip()
        if " - " in c:
            c = c.split(" - ", 1)[0]
        return c.strip()

    df = df.rename(columns={c: canonicalize_run_col(c) for c in df.columns})

    # merge columns that became identical after canonicalization
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T

    return df




def rolling_mean(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    if window <= 1:
        return df
    return df.rolling(window=window, min_periods=1, center=True).mean()


def common_cols(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    return sorted(set(a.columns).intersection(set(b.columns)))


def safe_log10(x: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    return np.log10(np.clip(x.to_numpy(dtype=float), eps, None))


# ----------------------------
# Core plotting
# ----------------------------
def compute_min_fraction(min_count: pd.DataFrame, maj_count: pd.DataFrame) -> pd.DataFrame:
    cols = common_cols(min_count, maj_count)
    if not cols:
        raise ValueError("min_count and maj_count have no common run-columns.")
    mc = min_count[cols]
    jc = maj_count[cols]
    frac = mc / (mc + jc)
    return frac


def compute_log10_ratio(mean_norm_min: pd.DataFrame, mean_norm_maj: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    cols = common_cols(mean_norm_min, mean_norm_maj)
    if not cols:
        raise ValueError("mean_norm_min and mean_norm_maj have no common run-columns.")
    mn = mean_norm_min[cols]
    mj = mean_norm_maj[cols]
    ratio = mn / (mj + eps)
    out = pd.DataFrame(safe_log10(ratio, eps=eps), index=ratio.index, columns=ratio.columns)
    return out


def _plot_multi(ax, df: pd.DataFrame, title: str, ylabel: str, xlim=None, ylim=None):
    for col in df.columns:
        ax.plot(df.index, df[col], linewidth=2, label=col)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)

def _save_single_panel(df, title, ylabel, out_path, xlim=None, ylim=None):
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))  # <- fast quadratisch
    for col in df.columns:
        ax.plot(df.index, df[col], linewidth=2, label=col)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Step")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncols=1, loc="best")  # ncols=1 = untereinander

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



def plot_grad_geometry(
    min_count_csv: str | Path,
    maj_count_csv: str | Path,
    mean_norm_min_csv: str | Path,
    mean_norm_maj_csv: str | Path,
    cos_min_vs_maj_csv: str | Path,
    out_prefix: str | Path = "grad_geometry",
    smooth_window: int = 1,
    max_step: int | None = None,
):
    # Load
    min_count = load_wandb_csv(min_count_csv)
    maj_count = load_wandb_csv(maj_count_csv)
    mean_norm_min = load_wandb_csv(mean_norm_min_csv)
    mean_norm_maj = load_wandb_csv(mean_norm_maj_csv)
    cos = load_wandb_csv(cos_min_vs_maj_csv)

    # Optional common step clipping (useful if balanced has longer run)
    if max_step is not None:
        min_count = min_count[min_count.index <= max_step]
        maj_count = maj_count[maj_count.index <= max_step]
        mean_norm_min = mean_norm_min[mean_norm_min.index <= max_step]
        mean_norm_maj = mean_norm_maj[mean_norm_maj.index <= max_step]
        cos = cos[cos.index <= max_step]

    # Derived metrics
    min_frac = compute_min_fraction(min_count, maj_count)
    log10_ratio = compute_log10_ratio(mean_norm_min, mean_norm_maj)

    # Smooth (same window everywhere)
    min_frac = rolling_mean(min_frac, smooth_window)
    log10_ratio = rolling_mean(log10_ratio, smooth_window)
    cos = rolling_mean(cos, smooth_window)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    


    out_prefix = Path(out_prefix)
    xlim = (min_frac.index.min(), min_frac.index.max())

    _save_single_panel(
        min_frac,
        title="Minority fraction in batch",
        ylabel="min_count / (min_count + maj_count)",
        out_path=out_prefix.with_name(out_prefix.name + "_min_fraction.png"),
        xlim=xlim,
        ylim=(0.0, 1.0),
    )

    _save_single_panel(
        log10_ratio,
        title="Magnitude asymmetry (log scale)",
        ylabel=r"$\log_{10}(\|g_{min}\| / \|g_{maj}\|)$",
        out_path=out_prefix.with_name(out_prefix.name + "_log10_ratio.png"),
        xlim=xlim,
    )

    _save_single_panel(
        cos,
        title="Directional alignment",
        ylabel=r"$\cos(g_{min}, g_{maj})$",
        out_path=out_prefix.with_name(out_prefix.name + "_cos.png"),
        xlim=xlim,
        ylim=(-1.05, 1.05),
    )





    _plot_multi(
        axes[0],
        min_frac,
        title="Minority fraction in batch",
        ylabel="min_count / (min_count + maj_count)",
        ylim=(0.0, 1.0),
    )

    _plot_multi(
        axes[1],
        log10_ratio,
        title="Magnitude asymmetry (log scale)",
        ylabel=r"$\log_{10}(\|g_{min}\| / \|g_{maj}\|)$",
    )

    _plot_multi(
        axes[2],
        cos,
        title="Directional alignment",
        ylabel=r"$\cos(g_{min}, g_{maj})$",
        ylim=(-1.05, 1.05),
    )
    axes[2].set_xlabel("Step")

    # Single legend (top)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_prefix = Path(out_prefix)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=250)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)






    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


# ----------------------------
# Optional: FC vs CNN faceting
# ----------------------------
def plot_grad_geometry_faceted(arch_to_paths: dict, out_prefix="grad_geometry_faceted", **kwargs):
    """
    arch_to_paths example:
    {
      "FC":  {"min_count": "...", "maj_count": "...", "mean_norm_min": "...", "mean_norm_maj": "...", "cos": "..."},
      "CNN": {"min_count": "...", "maj_count": "...", "mean_norm_min": "...", "mean_norm_maj": "...", "cos": "..."},
    }
    Produces a 3xN grid: rows = metrics, cols = architectures.
    """
    arch_names = list(arch_to_paths.keys())
    n = len(arch_names)

    fig, axes = plt.subplots(3, n, figsize=(4.4 * n, 9), sharex="row")

    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    for j, arch in enumerate(arch_names):
        p = arch_to_paths[arch]
        min_count = load_wandb_csv(p["min_count"])
        maj_count = load_wandb_csv(p["maj_count"])
        mean_norm_min = load_wandb_csv(p["mean_norm_min"])
        mean_norm_maj = load_wandb_csv(p["mean_norm_maj"])
        cos = load_wandb_csv(p["cos"])

        max_step = kwargs.get("max_step")
        smooth_window = kwargs.get("smooth_window", 1)

        if max_step is not None:
            min_count = min_count[min_count.index <= max_step]
            maj_count = maj_count[maj_count.index <= max_step]
            mean_norm_min = mean_norm_min[mean_norm_min.index <= max_step]
            mean_norm_maj = mean_norm_maj[mean_norm_maj.index <= max_step]
            cos = cos[cos.index <= max_step]

        min_frac = rolling_mean(compute_min_fraction(min_count, maj_count), smooth_window)
        log10_ratio = rolling_mean(compute_log10_ratio(mean_norm_min, mean_norm_maj), smooth_window)
        cos = rolling_mean(cos, smooth_window)

        _plot_multi(axes[0, j], min_frac, f"{arch}: Minority fraction", "", ylim=(0.0, 1.0))
        _plot_multi(axes[1, j], log10_ratio, f"{arch}: log10 norm ratio", "", ylim=None)
        _plot_multi(axes[2, j], cos, f"{arch}: cosine alignment", "", ylim=(-1.05, 1.05))
        axes[2, j].set_xlabel("Step")

        axes[0, j].set_ylabel("min_fraction")
        axes[1, j].set_ylabel(r"$\log_{10}(\|g_{min}\|/\|g_{maj}\|)$")
        axes[2, j].set_ylabel(r"$\cos(g_{min}, g_{maj})$")

    # Global legend from first panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_prefix = Path(out_prefix)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=250)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)

    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


# ----------------------------
# Example usage (EDIT PATHS)
# ----------------------------
if __name__ == "__main__":
    # Single-architecture (your current FC CSVs)
    PATHS_FC = {
        "min_count": "MLP/min_count.csv",
        "maj_count": "MLP/maj_count.csv",
        "mean_norm_min": "MLP/mean_norm_min.csv",
        "mean_norm_maj": "MLP/mean_norm_maj.csv",
        "cos": "MLP/cos_min_vs_maj_mean.csv",
    }

    plot_grad_geometry(
        min_count_csv=PATHS_FC["min_count"],
        maj_count_csv=PATHS_FC["maj_count"],
        mean_norm_min_csv=PATHS_FC["mean_norm_min"],
        mean_norm_maj_csv=PATHS_FC["mean_norm_maj"],
        cos_min_vs_maj_csv=PATHS_FC["cos"],
        out_prefix="fc_grad_geometry",
        smooth_window=5,
        max_step=400,  # set None if you want full length

    )



    

    # Uncomment for FC vs CNN (add your CNN paths)
    # PATHS_CNN = {
    #     "min_count": "cnn_min_count.csv",
    #     "maj_count": "cnn_maj_count.csv",
    #     "mean_norm_min": "cnn_mean_norm_min.csv",
    #     "mean_norm_maj": "cnn_mean_norm_maj.csv",
    #     "cos": "cnn_cos_min_vs_maj_mean.csv",
    # }
    #
    # plot_grad_geometry_faceted(
    #     {"FC": PATHS_FC, "CNN": PATHS_CNN},
    #     out_prefix="fc_vs_cnn_grad_geometry",
    #     smooth_window=5,
    #     max_step=400,
    # )
