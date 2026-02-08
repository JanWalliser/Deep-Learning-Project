from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from src.logging.wandb_logger import log


@dataclass
class EpochLogConfig:
    log_tables: bool = True
    max_table_items: int = 24

    # high-confidence wrong examples
    wrong_highconf_k: int = 5

    # classwise gradient analysis
    enable_classwise_grads: bool = True
    grads_source: str = "train"  # train | val
    grads_max_samples: int = 1024
    grads_min_per_class: int = 16
    grads_include_params: Optional[list[str]] = None
    grads_compute_subspace_angles: bool = False
    grads_subspace_rank: int = 16


def _epoch_cfg(cfg: dict) -> EpochLogConfig:
    training_cfg = cfg.get("training", {}) or {}
    analysis_cfg = cfg.get("analysis", {}) or {}

    grads_cfg = (analysis_cfg.get("classwise_grads", {}) or {})

    return EpochLogConfig(
        log_tables=bool(training_cfg.get("log_tables", True)),
        max_table_items=int(training_cfg.get("max_table_items", 24)),
        wrong_highconf_k=int(analysis_cfg.get("wrong_highconf_k", 5)),
        enable_classwise_grads=bool(grads_cfg.get("enable", True)),
        grads_source=str(grads_cfg.get("source", "train")).lower(),
        grads_max_samples=int(grads_cfg.get("max_samples", 1024)),
        grads_min_per_class=int(grads_cfg.get("min_per_class", 16)),
        grads_include_params=grads_cfg.get("include_params", None),
        grads_compute_subspace_angles=bool(grads_cfg.get("compute_subspace_angles", False)),
        grads_subspace_rank=int(grads_cfg.get("subspace_rank", 16)),
    )


class WandbCallbacks:
    """Keeps the training loop clean: all W&B logging lives here."""

    def __init__(self, wandb_run, cfg: dict, device: torch.device):
        self.run = wandb_run
        self.cfg = cfg
        self.device = device

        self.training_cfg = cfg.get("training", {}) or {}
        self.num_classes = int(cfg.get("model", {}).get("num_classes", 2))
        self.log_every_steps = int(self.training_cfg.get("log_every_steps", 50))
        self.log_grads_stepwise = bool(self.training_cfg.get("log_grads", True))
        self.epoch_cfg = _epoch_cfg(cfg)

    def on_train_step(self, model: torch.nn.Module, loss: torch.Tensor, global_step: int, epoch: int) -> None:
        if self.run is None:
            return

        if global_step % self.log_every_steps != 0:
            return

        # --- scalar loss
        log(self.run, {"train/loss": float(loss.item()), "epoch": epoch}, step=global_step)

        # --- simple global grad diagnostics (cheap, step-wise)
        if self.log_grads_stepwise:
            try:
                from src.viz.gradients import compute_grad_stats

                g_norm, gwr, _, _ = compute_grad_stats(model)
                log(
                    self.run,
                    {
                        "grad/global_norm": float(g_norm),
                        "grad/global_gwr": float(gwr),
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            except Exception:
                print("[STEP GRAD LOGGING] failed:")
                traceback.print_exc()

    def on_epoch_end(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        epoch: int,
        global_step: int,
    ) -> None:
        if self.run is None:
            return

        # --- evaluate
        from src.training.eval import evaluate

        val_metrics, y_true, y_pred, val_logits = evaluate(model, val_loader, self.device, collect_outputs=True)
        log(self.run, {f"val/{k}": float(v) for k, v in val_metrics.items()} | {"epoch": epoch}, step=global_step)

        # --- confusion matrix + recall per class
        from src.metrics.classification import confusion_matrix, per_class_recall_from_cm, normalize_confusion_matrix

        cm = confusion_matrix(y_true, y_pred, num_classes=self.num_classes)
        recall = per_class_recall_from_cm(cm)
        for c in range(self.num_classes):
            log(self.run, {f"val/recall_class_{c}": float(recall[c].item()), "epoch": epoch}, step=global_step)

        # --- media logging: confusion matrix + examples
        try:
            from src.viz.plots import plot_confusion_matrix
            from src.viz.wandb_media import (
                log_confusion_matrix_image,
                log_confusion_matrix_plot,
                log_prediction_tables,
                log_highconf_wrong_media,
            )
            from src.viz.predictions import collect_predictions, split_examples

            run_id = self.run.id

            log_confusion_matrix_image(
                self.run,
                cm=cm,
                epoch=epoch,
                step=global_step,
                run_id=run_id,
                normalize_fn=normalize_confusion_matrix,
                plot_fn=plot_confusion_matrix,
            )
            log_confusion_matrix_plot(
                self.run,
                y_true=y_true,
                y_pred=y_pred,
                class_names=[str(i) for i in range(self.num_classes)],
                epoch=epoch,
                step=global_step,
            )

            # collect examples once and reuse for both tables + media
            examples = collect_predictions(model, val_loader, self.device, max_items=4000)
            parts = split_examples(examples)

            # High-confidence wrong as *media* (more visible than table)
            log_highconf_wrong_media(
                self.run,
                wrong_examples=parts.get("wrong", []),
                epoch=epoch,
                step=global_step,
                run_id=run_id,
                k=self.epoch_cfg.wrong_highconf_k,
            )

            if self.epoch_cfg.log_tables:
                log_prediction_tables(self.run, parts, step=global_step, max_wrong=self.epoch_cfg.max_table_items)

        except Exception:
            print("[VAL MEDIA LOGGING] failed:")
            traceback.print_exc()

        # --- classwise gradient analysis (per epoch)
        if self.epoch_cfg.enable_classwise_grads:
            try:
                from src.analysis.classwise_gradients import (
                    resolve_class_spec,
                    collect_analysis_batch,
                    per_sample_grads,
                    compute_classwise_gradient_diagnostics,
                )
                from src.viz.classwise_gradients import log_classwise_gradient_diagnostics

                class_spec = resolve_class_spec(self.cfg)
                src = self.epoch_cfg.grads_source
                loader = train_loader if src == "train" else val_loader

                x_a, y_a = collect_analysis_batch(
                    loader,
                    device=self.device,
                    class_spec=class_spec,
                    max_samples=self.epoch_cfg.grads_max_samples,
                    min_per_class=self.epoch_cfg.grads_min_per_class,
                )

                # Per-sample gradients (deterministic eval-mode within function)
                grads_ps, names = per_sample_grads(
                    model=model,
                    x=x_a,
                    y=y_a,
                    loss_fn=F.cross_entropy,
                    include_params=self.epoch_cfg.grads_include_params,
                )

                lr = float(self.training_cfg.get("lr", 0.0))
                res = compute_classwise_gradient_diagnostics(
                    grads=grads_ps,
                    param_names=names,
                    y=y_a.detach().cpu(),
                    class_spec=class_spec,
                    lr_for_harm_score=lr if lr > 0 else None,
                    compute_subspace_angles=self.epoch_cfg.grads_compute_subspace_angles,
                    subspace_rank=self.epoch_cfg.grads_subspace_rank,
                )

                log_classwise_gradient_diagnostics(
                    self.run,
                    result=res,
                    class_spec=class_spec,
                    epoch=epoch,
                    step=global_step,
                    run_id=self.run.id,
                )
            except Exception:
                print("[CLASSWISE GRAD ANALYSIS] failed:")
                traceback.print_exc()
