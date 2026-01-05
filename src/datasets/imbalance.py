
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np
import torch

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def sample_indices(pool: Sequence[int], n: int, rng: np.random.Generator, replace: bool) -> List[int]:
    pool = list(pool)
    if n <= 0:
        return []
    if len(pool) == 0:
        raise ValueError("Cannot sample from an empty pool.")
    if (not replace) and n > len(pool):
        raise ValueError(f"Requested n={n} without replacement but only {len(pool)} available.")
    chosen = rng.choice(pool, size=n, replace=replace)
    return [int(i) for i in chosen.tolist()]

def apply_imbalance_to_train_indices(
    train_indices: Sequence[int],
    targets: torch.Tensor,
    cfg: dict,
) -> List[int]:
    """
    targets: original MNIST digit targets (0-9), aligned with the *base dataset* indices.
    train_indices: indices currently in the train split (w.r.t base dataset).
    Returns new train indices according to imbalance config.
    """
    imb = cfg.get("imbalance", {}) or {}
    setting = imb.get("setting", "none")

    if setting == "none":
        return list(train_indices)

    seed = int(cfg.get("seed", 42))
    rng = _rng(seed + 12345)

    major = int(imb["major_digit"])
    minor = int(imb["minor_digit"])
    n_major = int(imb["n_major"])
    n_minor = int(imb["n_minor"])

    replace = bool(imb.get("sample_with_replacement", True))
    policy = imb.get("other_digits_policy", "keep_all")

    train_set = set(int(i) for i in train_indices)
    pools: Dict[int, List[int]] = {d: [] for d in range(10)}
    for idx in train_indices:
        d = int(targets[int(idx)])
        pools[d].append(int(idx))

    out = []
    out += sample_indices(pools[major], n_major, rng, replace=replace)
    out += sample_indices(pools[minor], n_minor, rng, replace=replace)

    other_digits = [d for d in range(10) if d not in (major, minor)]

    if policy == "keep_all":
        for d in other_digits:
            out += pools[d]

    elif policy == "uniform_subsample":
        n_other = imb.get("n_other", None)
        n_total = imb.get("n_total", None)

        if n_other is None and n_total is None:
            raise ValueError("uniform_subsample requires either imbalance.n_other or imbalance.n_total")

        if n_other is None:
            remaining = int(n_total) - n_major - n_minor
            if remaining < 0:
                raise ValueError("imbalance.n_total is smaller than n_major + n_minor")
            n_other = remaining // len(other_digits)

        n_other = int(n_other)
        for d in other_digits:
            out += sample_indices(pools[d], n_other, rng, replace=replace)

    else:
        raise ValueError(f"Unknown other_digits_policy: {policy}")

    rng.shuffle(out)
    return out
