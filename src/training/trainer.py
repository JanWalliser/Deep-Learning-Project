import torch
from torch.nn import functional as F


def _as_tuple2(x, default=(0.9, 0.999)):
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (float(x[0]), float(x[1]))
    raise ValueError(f"betas must be a list/tuple of length 2, got: {x!r}")

def build_optimizer(model: torch.nn.Module, training_cfg: dict) -> torch.optim.Optimizer:
    """
    Expects config like:
      training:
        lr: 0.01
        weight_decay: 0.0
        optimizer:
          name: sgd|adam|adamw
          momentum: 0.9        # for sgd
          nesterov: false      # for sgd
          betas: [0.9, 0.999]  # for adam/adamw
          eps: 1.0e-8          # for adam/adamw
    """
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    opt_cfg = training_cfg.get("optimizer", {}) or {}
    name = str(opt_cfg.get("name", "sgd")).lower()

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.0))
        nesterov = bool(opt_cfg.get("nesterov", False))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if name == "adam":
        betas = _as_tuple2(opt_cfg.get("betas", None))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    if name == "adamw":
        betas = _as_tuple2(opt_cfg.get("betas", None))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer name: {name!r}. Use one of: sgd, adam, adamw.")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return {"loss": total_loss / n, "acc": correct / n}


def train(model, train_loader, val_loader, device, cfg: dict, wandb_run=None):
    training_cfg = cfg["training"]
    epochs = int(training_cfg["epochs"])
    opt = build_optimizer(model, training_cfg)


    if wandb_run is not None:
        opt_name = str(training_cfg.get("optimizer", {}).get("name", "sgd"))
        wandb_run.log({"training/optimizer": opt_name, "epoch": 0})
 



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

            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(), "step": global_step, "epoch": epoch})
            global_step += 1

        val_metrics = evaluate(model, val_loader, device)
        if wandb_run is not None:
            wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch})

    return model
