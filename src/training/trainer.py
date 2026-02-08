# src/training/trainer.py
from __future__ import annotations

import torch
from torch.nn import functional as F

from src.logging.wandb_logger import config_update
from src.training.eval import evaluate
from src.training.wandb_callbacks import WandbCallbacks


def _as_tuple2(x, default=(0.9, 0.999)):
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (float(x[0]), float(x[1]))
    raise ValueError(f"betas must be a list/tuple of length 2, got: {x!r}")


def build_optimizer(model: torch.nn.Module, training_cfg: dict) -> torch.optim.Optimizer:
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    opt_cfg = training_cfg.get("optimizer", {}) or {}
    name = str(opt_cfg.get("name", "sgd")).lower()

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.0))
        nesterov = bool(opt_cfg.get("nesterov", False))
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )

    if name == "adam":
        betas = _as_tuple2(opt_cfg.get("betas", None))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    if name == "adamw":
        betas = _as_tuple2(opt_cfg.get("betas", None))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer name: {name!r}. Use one of: sgd, adam, adamw.")


__all__ = ["build_optimizer", "evaluate", "train"]


def train(model, train_loader, val_loader, device, cfg: dict, wandb_run=None):
    training_cfg = cfg["training"]
    epochs = int(training_cfg["epochs"])
    opt = build_optimizer(model, training_cfg)

    callbacks = WandbCallbacks(wandb_run, cfg=cfg, device=device)

    # Put mapping into config once (binary)
    if wandb_run is not None and cfg.get("task", {}).get("mode") == "binary":
        pos = int(cfg["task"]["positive_digit"])
        digits = [int(d) for d in cfg["task"]["digits"]]
        neg = [d for d in digits if d != pos][0]
        config_update(
            wandb_run,
            {"task/label_mapping": {"label0": neg, "label1": pos}, "task/positive_digit": pos, "task/digits": digits},
        )

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            callbacks.on_train_step(model=model, loss=loss, global_step=global_step, epoch=epoch)

            global_step += 1

        # ---- end epoch callbacks (evaluation, confusion matrix, examples, classwise gradients)
        callbacks.on_epoch_end(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch,
            global_step=global_step,
        )

    return model
