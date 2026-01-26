import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .device import get_device, seed_everything
from .datasets.mnist import get_mnist
from .models.factory import build_model
from .training.trainer import train, evaluate
from .logging.wandb_logger import init_wandb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a YAML config. Can be repeated.",
    )
    return p.parse_args()


def count_labels_from_loader(loader) -> dict[int, int]:
    """
    Counts labels by iterating through a DataLoader (robust, works for Subset/custom datasets).
    Note: this counts the samples that the loader actually yields.
    """
    c = Counter()
    for _, y in loader:
        c.update(y.detach().cpu().view(-1).tolist())
    return dict(c)


def count_labels_from_dataset(ds) -> dict[int, int]:
    """
    Counts labels directly from a dataset when possible.
    Falls back to iterating the dataset (slower but reliable).
    """
    # Common: torchvision MNIST uses .targets (Tensor)
    if hasattr(ds, "targets"):
        y = ds.targets
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().to(torch.long).view(-1)
            counts = torch.bincount(y)
            return {int(i): int(counts[i].item()) for i in range(counts.numel()) if counts[i].item() > 0}

    # Custom datasets sometimes store labels explicitly
    if hasattr(ds, "labels"):
        return dict(Counter([int(t) for t in ds.labels]))

    # Fallback: iterate dataset
    c = Counter()
    for _, y in ds:
        c.update([int(y)])
    return dict(c)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    run = init_wandb(cfg)

    # Build datasets
    train_ds, val_ds, test_ds = get_mnist(cfg)

    # Debug prints (local console) to verify config and dataset composition
    print("[DEBUG] task cfg:", cfg.get("task", None))
    print("[DEBUG] imbalance cfg:", cfg.get("imbalance", None))
    print("[DEBUG] train_ds counts:", count_labels_from_dataset(train_ds))
    print("[DEBUG] val_ds counts:", count_labels_from_dataset(val_ds))
    print("[DEBUG] test_ds counts:", count_labels_from_dataset(test_ds))

    # DataLoaders
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin,
    )

    # Log dataset composition (counts) + mapping to W&B
    if run is not None:
        train_counts = count_labels_from_loader(train_loader)
        val_counts = count_labels_from_loader(val_loader)
        test_counts = count_labels_from_loader(test_loader)

        # total sizes
        run.log(
            {
                "data/train/n": int(sum(train_counts.values())),
                "data/val/n": int(sum(val_counts.values())),
                "data/test/n": int(sum(test_counts.values())),
            },
            step=0,
        )

        # per-class counts
        for k, v in sorted(train_counts.items()):
            run.log({f"data/train/count_class_{k}": int(v)}, step=0)
        for k, v in sorted(val_counts.items()):
            run.log({f"data/val/count_class_{k}": int(v)}, step=0)
        for k, v in sorted(test_counts.items()):
            run.log({f"data/test/count_class_{k}": int(v)}, step=0)

        # Explicit mapping for binary tasks (so you know what 0/1 mean)
        if cfg.get("task", {}).get("mode", None) == "binary":
            pos = int(cfg["task"]["positive_digit"])
            digits = [int(d) for d in cfg["task"]["digits"]]
            neg = [d for d in digits if d != pos][0]
            run.config.update(
                {
                    "task/label0_digit": neg,
                    "task/label1_digit": pos,
                    "task/positive_digit": pos,
                    "task/digits": digits,
                },
                allow_val_change=True,
            )

        # Also log device info for traceability
        run.config.update(
            {
                "device/type": device.type,
                "device/cuda_available": bool(torch.cuda.is_available()),
                "device/name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            },
            allow_val_change=True,
        )

    # Model
    model = build_model(cfg).to(device)

    # Train
    train(model, train_loader, val_loader, device, cfg=cfg, wandb_run=run)

    # Test
    test_metrics = evaluate(model, test_loader, device)
    if run is not None:
        run.log({f"test/{k}": v for k, v in test_metrics.items()})
        run.finish()

    print("Test:", test_metrics)


if __name__ == "__main__":
    main()
