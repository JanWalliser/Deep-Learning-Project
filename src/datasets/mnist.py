from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split

def get_mnist(cfg: dict):
    tfm = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    root = cfg["data"]["root"]
    full_train = MNIST(root=root, train=True, download=True, transform=tfm)
    test = MNIST(root=root, train=False, download=True, transform=tfm)

    val_split = float(cfg["data"]["val_split"])
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train, val = random_split(full_train, [n_train, n_val])
    return train, val, test
