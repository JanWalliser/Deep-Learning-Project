from .fc import FCNet
from .cnn import CNNNet

def build_model(cfg: dict):
    name = cfg["model"]["name"].lower()
    num_classes = cfg["model"].get("num_classes", 10)

    if name == "fc":
        return FCNet(
            hidden_sizes=cfg["model"].get("hidden_sizes", [256, 128]),
            dropout=cfg["model"].get("dropout", 0.1),
            num_classes=num_classes,
        )

    if name == "cnn":
        channels = cfg["model"].get("channels", [32, 64])

        return CNNNet(
            in_channels=cfg["model"].get("in_channels", 1),
            channels=channels,
            fc_dim=cfg["model"].get("fc_dim", 128),
            dropout=cfg["model"].get("dropout", 0.25),
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model name: {name}")
