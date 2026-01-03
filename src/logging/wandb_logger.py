import wandb

def init_wandb(cfg: dict):
    if not cfg["logging"].get("use_wandb", False):
        return None

    run_name = cfg["logging"].get("run_name")
    if run_name is None:
        run_name = f"{cfg['model']['name']}_{cfg.get('imbalance', {}).get('setting', 'none')}_seed{cfg['seed']}"

    return wandb.init(
        project=cfg["logging"]["project"],
        entity=cfg["logging"].get("entity"),
        name=run_name,
        config=cfg,
    )
