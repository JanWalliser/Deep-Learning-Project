from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.analysis.classwise_gradients import ClassSpec, ClasswiseGradResult
from src.viz.plots import plot_overlaid_histogram, plot_histogram
from src.viz.wandb_media import ensure_dir


def log_classwise_gradient_diagnostics(
    wandb_run,
    result: ClasswiseGradResult,
    class_spec: ClassSpec,
    epoch: int,
    step: int,
    run_id: str,
    key_prefix: str = "classwise_grad",
) -> None:
    """Log class-wise gradient diagnostics to W&B.

    - Scalars: logged directly (prefixed).
    - Distributions: logged as both Histogram (interactive) and a PNG (media).
    """

    import wandb

    # --- scalars
    scalars: Dict[str, float] = dict(result.scalars)
    scalars["epoch"] = float(epoch)
    wandb_run.log({k: float(v) for k, v in scalars.items()}, step=step)

    # --- histograms (interactive)
    if result.norms_majority.numel() > 0:
        wandb_run.log(
            {
                f"{key_prefix}/hist_norms_majority": wandb.Histogram(result.norms_majority.numpy()),
            },
            step=step,
        )
    if result.norms_minority.numel() > 0:
        wandb_run.log(
            {
                f"{key_prefix}/hist_norms_minority": wandb.Histogram(result.norms_minority.numpy()),
            },
            step=step,
        )
    if result.cos_min_samples_vs_maj_mean.numel() > 0:
        wandb_run.log(
            {
                f"{key_prefix}/hist_cos_min_samples_vs_maj_mean": wandb.Histogram(
                    result.cos_min_samples_vs_maj_mean.numpy()
                )
            },
            step=step,
        )

    # --- PNG plots (Media)
    label_maj = f"majority(label={class_spec.majority_label})"
    label_min = f"minority(label={class_spec.minority_label})"

    # Use log-scale for norms (more readable)
    eps = 1e-12
    maj_log = np.log10(result.norms_majority.numpy() + eps) if result.norms_majority.numel() > 0 else np.asarray([])
    min_log = np.log10(result.norms_minority.numpy() + eps) if result.norms_minority.numel() > 0 else np.asarray([])

    norms_path = ensure_dir(f"outputs/reports/{run_id}/grads/norms_epoch_{epoch:02d}.png")
    if maj_log.size and min_log.size:
        plot_overlaid_histogram(
            maj_log,
            min_log,
            str(norms_path),
            title=f"Per-sample grad norm (log10) @ epoch {epoch}",
            xlabel="log10(||g_i||)",
            label_a=label_maj,
            label_b=label_min,
            bins=40,
        )
        wandb_run.log({f"{key_prefix}/norms_log_hist": wandb.Image(str(norms_path))}, step=step)

    cos_path = ensure_dir(f"outputs/reports/{run_id}/grads/cos_epoch_{epoch:02d}.png")
    if result.cos_min_samples_vs_maj_mean.numel() > 0:
        plot_histogram(
            result.cos_min_samples_vs_maj_mean.numpy(),
            str(cos_path),
            title=f"Cosine(minority sample grad, majority-mean grad) @ epoch {epoch}",
            xlabel="cosine",
        )
        wandb_run.log({f"{key_prefix}/cos_min_samples_vs_maj_mean": wandb.Image(str(cos_path))}, step=step)

    if result.principal_angles is not None and result.principal_angles.numel() > 0:
        ang_path = ensure_dir(f"outputs/reports/{run_id}/grads/angles_epoch_{epoch:02d}.png")
        plot_histogram(
            result.principal_angles.numpy(),
            str(ang_path),
            title=f"Principal angles between gradient subspaces @ epoch {epoch}",
            xlabel="radians",
        )
        wandb_run.log({f"{key_prefix}/principal_angles": wandb.Image(str(ang_path))}, step=step)

    # --- per-parameter tables (top-K)
    def _log_param_table(rows: Optional[list[dict]], key: str):
        if not rows:
            return
        table = wandb.Table(
            columns=[
                "param",
                "cos_min_vs_maj",
                "dot_min_vs_maj",
                "mean_sample_norm_maj",
                "mean_sample_norm_min",
                "ratio_mean_sample_norm_min_over_maj",
                "min_conflict_rate",
            ]
        )
        for r in rows:
            table.add_data(
                r["param"],
                r["cos_min_vs_maj"],
                r["dot_min_vs_maj"],
                r["mean_sample_norm_maj"],
                r["mean_sample_norm_min"],
                r["ratio_mean_sample_norm_min_over_maj"],
                r["min_conflict_rate"],
            )
        wandb_run.log({key: table, "epoch": epoch}, step=step)

    _log_param_table(result.top_conflict_params, f"{key_prefix}/top_conflict_params")
    _log_param_table(result.top_smallminor_params, f"{key_prefix}/top_smallminor_params")
