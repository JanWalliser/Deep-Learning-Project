# src/logging/wandb_logger.py
from __future__ import annotations

from typing import Any, Optional


def init_wandb(cfg: dict):
    if not cfg.get("logging", {}).get("use_wandb", False):
        return None

    import wandb

    run_name = cfg["logging"].get("run_name")
    if run_name is None:
        run_name = f"{cfg['model']['name']}_{cfg.get('imbalance', {}).get('setting', 'none')}_seed{cfg['seed']}"

    return wandb.init(
        project=cfg["logging"]["project"],
        entity=cfg["logging"].get("entity"),
        name=run_name,
        config=cfg,
    )


def log(run, data: dict, step: Optional[int] = None):
    """
    Unified logging for scalars/images/tables/plots.
    """
    if run is None:
        return
    if step is None:
        run.log(data)
    else:
        run.log(data, step=step)


def config_update(run, data: dict, allow_val_change: bool = True):
    """
    Central place for W&B config updates.
    """
    if run is None:
        return
    run.config.update(data, allow_val_change=allow_val_change)


def finish(run):
    if run is None:
        return
    run.finish()
