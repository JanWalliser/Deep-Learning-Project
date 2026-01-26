from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Subset
import torch

from .imbalance import apply_imbalance_to_train_indices


class BinaryTargetTransform:
    def __init__(self, positive_digit: int):
        self.positive_digit = int(positive_digit)

    def __call__(self, y: int) -> int:
        y = int(y)
        return 1 if y == self.positive_digit else 0


def _filter_indices_by_digits(targets: torch.Tensor, digits: list[int]) -> list[int]:
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for d in digits:
        mask |= (targets == int(d))
    return mask.nonzero(as_tuple=False).squeeze(1).tolist()


def _split_indices(indices: list[int], val_split: float, seed: int) -> tuple[list[int], list[int]]:
    """
    Deterministic split of a list of base-dataset indices into train/val.
    """
    assert 0.0 < val_split < 1.0
    n = len(indices)
    n_val = int(n * val_split)
    n_train = n - n_val

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)

    idx = torch.tensor(indices, dtype=torch.long)
    idx = idx[perm]

    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()
    return train_idx, val_idx


def get_mnist(cfg: dict):
    tfm = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    root = cfg["data"]["root"]
    mode = cfg["task"]["mode"]
    seed = int(cfg.get("seed", 42))
    val_split = float(cfg["data"]["val_split"])

    if mode == "binary":
        digits = [int(d) for d in cfg["task"]["digits"]]
        pos = int(cfg["task"]["positive_digit"])
        tt = BinaryTargetTransform(pos)

        base_train = MNIST(root=root, train=True, download=True, transform=tfm, target_transform=tt)
        base_test = MNIST(root=root, train=False, download=True, transform=tfm, target_transform=tt)

        # Filter to only the selected digits (indices w.r.t. base dataset)
        all_idx = _filter_indices_by_digits(base_train.targets, digits)
        test_idx = _filter_indices_by_digits(base_test.targets, digits)

        # Split indices into train/val (still base-dataset indices)
        train_idx, val_idx = _split_indices(all_idx, val_split=val_split, seed=seed)

        # Apply imbalance ONLY to the training indices (uses ORIGINAL digit targets 0-9)
        train_idx = apply_imbalance_to_train_indices(train_idx, base_train.targets, cfg)

        train = Subset(base_train, train_idx)
        val = Subset(base_train, val_idx)
        test = Subset(base_test, test_idx)
        return train, val, test

    # ---- multiclass (0-9)
    base_train = MNIST(root=root, train=True, download=True, transform=tfm)
    base_test = MNIST(root=root, train=False, download=True, transform=tfm)

    all_idx = list(range(len(base_train)))
    train_idx, val_idx = _split_indices(all_idx, val_split=val_split, seed=seed)

    # Optional: if you also want imbalance in multiclass, it will work here too
    train_idx = apply_imbalance_to_train_indices(train_idx, base_train.targets, cfg)

    train = Subset(base_train, train_idx)
    val = Subset(base_train, val_idx)
    test = base_test
    return train, val, test
