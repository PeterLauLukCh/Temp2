"""Dense reference implementation for the FlashWasserstein semi-dual."""

from __future__ import annotations

from .optim import run_semidual_subgradient
from .utils import require_torch, validate_points


def dense_c_transform(x, y, psi, *, cost_scale: float = 0.5):
    """Dense reference c-transform.

    Computes ``min_j cost_scale * ||x_i - y_j||^2 - psi_j`` and materializes the
    full ``n x m`` interaction matrix. This is for correctness checks and small
    baselines, not for scaling.
    """

    th = require_torch()
    validate_points(x, y)
    if psi.shape != (y.shape[0],):
        raise ValueError(f"psi must have shape ({y.shape[0]},), got {tuple(psi.shape)}.")
    cost = cost_scale * th.cdist(x.float(), y.float(), p=2).pow(2)
    values = cost - psi.float().unsqueeze(0)
    c_values, assignment = values.min(dim=1)
    return c_values.float(), assignment.long()


def solve_dense_semidual(
    x,
    y,
    *,
    a=None,
    b=None,
    cost_scale: float = 0.5,
    max_iter: int = 200,
    lr: float = 1.0,
    tol: float = 1e-4,
    gauge: str = "weighted_mean_zero",
    lr_schedule: str = "sqrt_decay",
    psi_init=None,
    record_every: int = 1,
    return_history: bool = True,
    normalize_weights: bool = True,
):
    """Dense semi-dual subgradient solver."""

    def oracle(psi):
        return dense_c_transform(x, y, psi, cost_scale=cost_scale)

    return run_semidual_subgradient(
        x,
        y,
        oracle=oracle,
        a=a,
        b=b,
        psi_init=psi_init,
        cost_scale=cost_scale,
        max_iter=max_iter,
        lr=lr,
        tol=tol,
        gauge=gauge,
        lr_schedule=lr_schedule,
        record_every=record_every,
        return_history=return_history,
        normalize_weights=normalize_weights,
        backend="dense",
    )
