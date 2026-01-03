from .fc import FCNet

def build_model(cfg: dict):
    name = cfg["model"]["name"]
    if name == "fc":
        return FCNet(
            hidden_sizes=cfg["model"].get("hidden_sizes", [256, 128]),
            dropout=cfg["model"].get("dropout", 0.1),
            num_classes=cfg["model"].get("num_classes", 10),
        )
    raise ValueError(f"Unknown model name: {name}")
