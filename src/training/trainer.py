# src/training/trainer.py
from __future__ import annotations

import traceback
import torch
from torch.nn import functional as F

from src.logging.wandb_logger import log, config_update


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


@torch.no_grad()
def evaluate(model, loader, device, collect_outputs: bool = False):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    all_y, all_pred, all_logits = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)

        if collect_outputs:
            all_y.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())
            all_logits.append(logits.detach().cpu())

    metrics = {"loss": total_loss / n, "acc": correct / n}
    if not collect_outputs:
        return metrics

    return metrics, torch.cat(all_y), torch.cat(all_pred), torch.cat(all_logits)


def train(model, train_loader, val_loader, device, cfg: dict, wandb_run=None):
    training_cfg = cfg["training"]
    epochs = int(training_cfg["epochs"])
    opt = build_optimizer(model, training_cfg)

    log_every_steps = int(training_cfg.get("log_every_steps", 50))
    log_grads = bool(training_cfg.get("log_grads", True))
    log_tables = bool(training_cfg.get("log_tables", True))
    max_table_items = int(training_cfg.get("max_table_items", 24))

    num_classes = int(cfg["model"]["num_classes"])

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

            if wandb_run is not None and log_grads and (global_step % log_every_steps == 0):
                from src.viz.gradients import compute_grad_stats

                g_norm, gwr, _, _ = compute_grad_stats(model)
                log(wandb_run, {"grad/global_norm": float(g_norm), "grad/global_gwr": float(gwr), "epoch": epoch}, step=global_step)

            opt.step()

            if wandb_run is not None and (global_step % log_every_steps == 0):
                log(wandb_run, {"train/loss": float(loss.item()), "epoch": epoch}, step=global_step)

            global_step += 1

        # ---- end epoch validation
        if wandb_run is None:
            _ = evaluate(model, val_loader, device)
            continue

        val_metrics, y_true, y_pred, val_logits = evaluate(model, val_loader, device, collect_outputs=True)
        log(wandb_run, {f"val/{k}": float(v) for k, v in val_metrics.items()} | {"epoch": epoch}, step=global_step)

        # ---- metrics derived from confusion matrix
        from src.metrics.classification import confusion_matrix, per_class_recall_from_cm, normalize_confusion_matrix

        cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
        recall = per_class_recall_from_cm(cm)
        for c in range(num_classes):
            log(wandb_run, {f"val/recall_class_{c}": float(recall[c].item())}, step=global_step)

        # ---- confusion matrix media + tables
        try:
            from src.viz.plots import plot_confusion_matrix
            from src.viz.wandb_media import log_confusion_matrix_image, log_confusion_matrix_plot, log_prediction_tables
            from src.viz.predictions import collect_predictions, split_examples

            run_id = wandb_run.id

            log_confusion_matrix_image(
                wandb_run,
                cm=cm,
                epoch=epoch,
                step=global_step,
                run_id=run_id,
                normalize_fn=normalize_confusion_matrix,
                plot_fn=plot_confusion_matrix,
            )
            log_confusion_matrix_plot(
                wandb_run,
                y_true=y_true,
                y_pred=y_pred,
                class_names=[str(i) for i in range(num_classes)],
                step=global_step,
            )

            if log_tables:
                examples = collect_predictions(model, val_loader, device, max_items=2000)
                parts = split_examples(examples)
                log_prediction_tables(wandb_run, parts, step=global_step, max_wrong=max_table_items)

        except Exception:
            print("[VAL MEDIA LOGGING] failed:")
            traceback.print_exc()

    return model
