from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ClassSpec:
    """Specification of majority/minority classes in *label space*.

    - In binary mode, labels are typically {0,1}.
    - In multiclass mode, labels are digits {0..9}.
    """

    majority_label: int
    minority_label: int


def resolve_class_spec(cfg: dict) -> ClassSpec:
    """Resolve which labels to treat as majority/minority.

    Defaults are chosen for the project's primary setting:
      - binary: label 0 vs label 1 (positive digit is label 1)
      - multiclass: digit 1 vs digit 7
    You can override with cfg["analysis"]["majority_label"] / ["minority_label"].
    """

    a = (cfg.get("analysis", {}) or {})
    if "majority_label" in a and "minority_label" in a:
        return ClassSpec(int(a["majority_label"]), int(a["minority_label"]))

    task = (cfg.get("task", {}) or {})
    mode = str(task.get("mode", "binary")).lower()

    if mode == "binary":
        # by construction in datasets.mnist.BinaryTargetTransform
        return ClassSpec(majority_label=0, minority_label=1)

    # multiclass default (your main comparison)
    return ClassSpec(majority_label=1, minority_label=7)


def _ensure_torch_func():
    try:
        from torch.func import functional_call, vmap, grad  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Per-sample gradients require torch.func (PyTorch >= 2.0). "
            "Install/upgrade PyTorch or disable per-sample gradient logging."
        ) from e


def _flatten_per_sample_grads(
    grads: Dict[str, torch.Tensor],
    param_names: List[str],
) -> torch.Tensor:
    """Convert dict[name] -> (N, P) matrix."""
    mats = []
    for n in param_names:
        g = grads[n]  # (N, *shape)
        mats.append(g.reshape(g.shape[0], -1))
    if not mats:
        raise ValueError("No parameters selected for gradient flattening.")
    return torch.cat(mats, dim=1)


def _select_param_names(
    params: Dict[str, torch.Tensor],
    include: Optional[List[str]],
) -> List[str]:
    """Select parameter names.

    include:
      - None or empty -> all
      - list[str] -> keep names where any token is a substring of the param name
    """
    names = list(params.keys())
    if not include:
        return names
    keep = []
    for n in names:
        if any(tok in n for tok in include):
            keep.append(n)
    return keep


@torch.no_grad()
def collect_analysis_batch(
    loader,
    device: torch.device,
    class_spec: ClassSpec,
    max_samples: int,
    min_per_class: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect a (x,y) batch for analysis.

    Iterates the loader until we have up to max_samples and at least min_per_class
    samples from each of majority/minority.
    """

    xs, ys = [], []
    maj = class_spec.majority_label
    min_ = class_spec.minority_label
    maj_n = 0
    min_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        xs.append(x)
        ys.append(y)

        maj_n += int((y == maj).sum().item())
        min_n += int((y == min_).sum().item())

        n = sum(t.size(0) for t in xs)
        if n >= max_samples and maj_n >= min_per_class and min_n >= min_per_class:
            break

    if not xs:
        raise RuntimeError("Analysis batch collection failed: loader produced no data.")

    x_all = torch.cat(xs, dim=0)[:max_samples]
    y_all = torch.cat(ys, dim=0)[:max_samples]

    # If we still don't have enough per class, we continue collecting (best effort)
    if int((y_all == maj).sum().item()) < min_per_class or int((y_all == min_).sum().item()) < min_per_class:
        # no exception: experiment configs (extreme imbalance) can cause this
        return x_all, y_all

    return x_all, y_all


def per_sample_grads(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn,
    include_params: Optional[List[str]] = None,
    dtype_for_metrics: torch.dtype = torch.float64,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Compute per-sample gradients (PyTorch 2.x torch.func).

    Returns:
      grads: dict[param_name] -> Tensor (N, *param_shape)
      param_names: list of param names included (order used for flattening)
    """
    _ensure_torch_func()
    from torch.func import functional_call, vmap, grad

    model.eval()  # make deterministic for analysis (Dropout off)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    param_names = _select_param_names(params, include_params)

    # We compute grads w.r.t *all* params for correctness, then select.
    # This avoids fragile partial-parameter functional_call plumbing.
    def _single_loss(p: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], x_i, y_i):
        logits = functional_call(model, (p, b), (x_i.unsqueeze(0),))
        return loss_fn(logits, y_i.unsqueeze(0))

    g_fn = grad(_single_loss)
    grads_all = vmap(g_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)

    grads = {n: grads_all[n].to(dtype_for_metrics) for n in param_names}
    return grads, param_names


@dataclass
class ClasswiseGradResult:
    # scalar diagnostics
    scalars: Dict[str, float]

    # distributions for visualization
    # global norms per sample, separated by class
    norms_majority: torch.Tensor  # (N_maj,)
    norms_minority: torch.Tensor  # (N_min,)

    # cosine of minority sample gradients vs majority class-mean gradient
    cos_min_samples_vs_maj_mean: torch.Tensor  # (N_min,)

    # optional: principal angles (radians)
    principal_angles: Optional[torch.Tensor] = None

    # compact per-parameter summaries (top-K only; for W&B tables)
    top_conflict_params: Optional[list[dict]] = None
    top_smallminor_params: Optional[list[dict]] = None


def compute_classwise_gradient_diagnostics(
    grads: Dict[str, torch.Tensor],
    param_names: List[str],
    y: torch.Tensor,
    class_spec: ClassSpec,
    lr_for_harm_score: Optional[float] = None,
    compute_subspace_angles: bool = False,
    subspace_rank: int = 16,
    per_param_topk: int = 10,
    eps: float = 1e-12,
) -> ClasswiseGradResult:
    """Compute class-wise gradient geometry diagnostics."""

    maj = class_spec.majority_label
    min_ = class_spec.minority_label

    G = _flatten_per_sample_grads(grads, param_names)  # (N, P)
    y = y.view(-1)

    mask_maj = (y == maj)
    mask_min = (y == min_)

    G_maj = G[mask_maj]
    G_min = G[mask_min]

    # per-sample global norms
    n_maj = G_maj.norm(dim=1)
    n_min = G_min.norm(dim=1)

    if G_maj.numel() == 0 or G_min.numel() == 0:
        # extreme imbalance: return best-effort with empty tensors
        return ClasswiseGradResult(
            scalars={
                "classwise_grad/maj_count": float(G_maj.shape[0]),
                "classwise_grad/min_count": float(G_min.shape[0]),
            },
            norms_majority=n_maj,
            norms_minority=n_min,
            cos_min_samples_vs_maj_mean=torch.empty((0,), dtype=G.dtype, device=G.device),
            principal_angles=None,
            top_conflict_params=None,
            top_smallminor_params=None,
        )

    g_maj = G_maj.mean(dim=0)
    g_min = G_min.mean(dim=0)
    g_all = G.mean(dim=0)

    # norms (class means)
    gmaj_norm = g_maj.norm() + eps
    gmin_norm = g_min.norm() + eps
    gall_norm = g_all.norm() + eps

    dot_min_maj = torch.dot(g_min, g_maj)
    cos_min_maj = dot_min_maj / (gmin_norm * gmaj_norm)
    cos_min_all = torch.dot(g_min, g_all) / (gmin_norm * gall_norm)
    proj_share_min_on_all = torch.dot(g_min, g_all) / (gall_norm * gall_norm)

    # minority sample alignment vs majority mean
    # cos(g_i, g_maj_mean)
    denom = (n_min + eps) * gmaj_norm
    cos_min_samples_vs_maj = (G_min @ g_maj) / denom
    conflict_rate = (cos_min_samples_vs_maj < 0).to(torch.float32).mean()

    # approximate change in minority loss under a majority-only SGD step:
    #   ΔL_min ≈ -lr * (g_min · g_maj)
    harm_score = None
    if lr_for_harm_score is not None:
        harm_score = float((-float(lr_for_harm_score)) * float(dot_min_maj))

    scalars = {
        "classwise_grad/maj_count": float(G_maj.shape[0]),
        "classwise_grad/min_count": float(G_min.shape[0]),
        "classwise_grad/mean_norm_maj": float(n_maj.mean().item()),
        "classwise_grad/mean_norm_min": float(n_min.mean().item()),
        "classwise_grad/ratio_mean_norm_min_over_maj": float((n_min.mean() / (n_maj.mean() + eps)).item()),
        "classwise_grad/cos_min_vs_maj_mean": float(cos_min_maj.item()),
        "classwise_grad/cos_min_vs_all": float(cos_min_all.item()),
        "classwise_grad/proj_share_min_on_all": float(proj_share_min_on_all.item()),
        "classwise_grad/dot_min_vs_maj_mean": float(dot_min_maj.item()),
        "classwise_grad/min_conflict_rate_vs_maj_mean": float(conflict_rate.item()),
    }
    if harm_score is not None:
        scalars["classwise_grad/approx_delta_Lmin_after_maj_step"] = float(harm_score)

    principal_angles = None
    if compute_subspace_angles:
        # principal angles between subspaces spanned by gradients (top-k singular vectors)
        # Use centered matrices to reduce bias from mean.
        A = (G_maj - g_maj).to(torch.float64)
        B = (G_min - g_min).to(torch.float64)
        k = int(subspace_rank)
        k = max(1, min(k, min(A.shape[0], B.shape[0], A.shape[1])))

        # compute orthonormal bases via SVD (economic)
        Ua = torch.linalg.svd(A, full_matrices=False).U[:, :k]
        Ub = torch.linalg.svd(B, full_matrices=False).U[:, :k]
        # singular values of Ua^T Ub are cos(principal angles)
        M = Ua.T @ Ub
        s = torch.linalg.svdvals(M).clamp(min=0.0, max=1.0)
        principal_angles = torch.arccos(s)

        scalars.update(
            {
                "classwise_grad/subspace_k": float(k),
                "classwise_grad/principal_angle_min": float(principal_angles.min().item()),
                "classwise_grad/principal_angle_mean": float(principal_angles.mean().item()),
            }
        )

    # --- per-parameter summaries (helpful to see which layer "ignores" minority)
    top_conflict = []
    top_smallminor = []
    try:
        per_param_rows = []
        for n in param_names:
            g = grads[n].reshape(grads[n].shape[0], -1)  # (N, Pn)
            g_maj_n = g[mask_maj]
            g_min_n = g[mask_min]
            if g_maj_n.numel() == 0 or g_min_n.numel() == 0:
                continue

            # class-mean vectors for this param
            gm = g_maj_n.mean(dim=0)
            gn = g_min_n.mean(dim=0)
            gm_norm = gm.norm() + eps
            gn_norm = gn.norm() + eps
            dot = torch.dot(gn, gm)
            cos = dot / (gn_norm * gm_norm)

            # average per-sample grad norms (this param only)
            maj_sample_norm = g_maj_n.norm(dim=1).mean()
            min_sample_norm = g_min_n.norm(dim=1).mean()
            ratio = min_sample_norm / (maj_sample_norm + eps)

            # minority sample conflict rate (this param only)
            cos_min_samples = (g_min_n @ gm) / ((g_min_n.norm(dim=1) + eps) * gm_norm)
            conflict = (cos_min_samples < 0).to(torch.float32).mean()

            per_param_rows.append(
                {
                    "param": n,
                    "cos_min_vs_maj": float(cos.item()),
                    "dot_min_vs_maj": float(dot.item()),
                    "mean_sample_norm_maj": float(maj_sample_norm.item()),
                    "mean_sample_norm_min": float(min_sample_norm.item()),
                    "ratio_mean_sample_norm_min_over_maj": float(ratio.item()),
                    "min_conflict_rate": float(conflict.item()),
                }
            )

        k = int(per_param_topk)
        if k > 0 and per_param_rows:
            top_conflict = sorted(per_param_rows, key=lambda r: r["cos_min_vs_maj"])[:k]
            top_smallminor = sorted(per_param_rows, key=lambda r: r["ratio_mean_sample_norm_min_over_maj"])[:k]
    except Exception:
        # best effort; do not fail training due to analysis
        top_conflict, top_smallminor = None, None

    return ClasswiseGradResult(
        scalars=scalars,
        norms_majority=n_maj.detach().cpu(),
        norms_minority=n_min.detach().cpu(),
        cos_min_samples_vs_maj_mean=cos_min_samples_vs_maj.detach().cpu(),
        principal_angles=principal_angles.detach().cpu() if principal_angles is not None else None,
        top_conflict_params=top_conflict if top_conflict else None,
        top_smallminor_params=top_smallminor if top_smallminor else None,
    )
