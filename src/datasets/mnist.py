from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Subset, random_split
import torch

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

def get_mnist(cfg: dict):
    tfm = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    root = cfg["data"]["root"]
    mode = cfg["task"]["mode"]
    seed = int(cfg.get("seed", 42))

    if mode == "binary":
        digits = cfg["task"]["digits"]              
        pos = cfg["task"]["positive_digit"]         
        tt = BinaryTargetTransform(pos)

        full_train = MNIST(root=root, train=True, download=True, transform=tfm, target_transform=tt)
        test = MNIST(root=root, train=False, download=True, transform=tfm, target_transform=tt)

        train_idx = _filter_indices_by_digits(full_train.targets, digits)
        test_idx = _filter_indices_by_digits(test.targets, digits)

        full_train = Subset(full_train, train_idx)
        test = Subset(test, test_idx)

    else:
        full_train = MNIST(root=root, train=True, download=True, transform=tfm)
        test = MNIST(root=root, train=False, download=True, transform=tfm)

    val_split = float(cfg["data"]["val_split"])
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val

    gen = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [n_train, n_val], generator=gen)

    return train, val, test