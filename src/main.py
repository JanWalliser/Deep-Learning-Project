import argparse
from torch.utils.data import DataLoader

from .config import load_config
from .device import get_device, seed_everything
from .datasets.mnist import get_mnist
from .models.factory import build_model
from .training.trainer import train, evaluate
from .logging.wandb_logger import init_wandb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", action="append", required=True, help="Path to a YAML config. Can be repeated.")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    run = init_wandb(cfg)

    train_ds, val_ds, test_ds = get_mnist(cfg)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"], pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                             num_workers=cfg["data"]["num_workers"], pin_memory=pin)

    model = build_model(cfg).to(device)

    train(model, train_loader, val_loader, device,
          epochs=cfg["training"]["epochs"],
          lr=cfg["training"]["lr"],
          weight_decay=cfg["training"]["weight_decay"],
          wandb_run=run)

    test_metrics = evaluate(model, test_loader, device)
    if run is not None:
        run.log({f"test/{k}": v for k, v in test_metrics.items()})
        run.finish()

    print("Test:", test_metrics)

if __name__ == "__main__":
    main()
