"""Flash semi-dual Wasserstein solver.

This module orchestrates the existing FlashSinkhorn hard c-transform kernel:
``flash_sinkhorn.c_transform_fwd``. No new Triton kernels are introduced here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .optim import run_semidual_subgradient
from .utils import require_torch, validate_points


def _load_flash_c_transform():
    """Import FlashSinkhorn's c-transform, adding the repo-local src if needed."""

    try:
        from flash_sinkhorn import c_transform_fwd

        return c_transform_fwd
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        src = repo_root / "code" / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from flash_sinkhorn import c_transform_fwd

        return c_transform_fwd


def flash_c_transform(
    x,
    y,
    psi,
    *,
    cost_scale: float = 0.5,
    allow_tf32: bool = True,
    autotune: bool = True,
    **kernel_kwargs,
):
    """Streaming Flash c-transform using the existing FlashSinkhorn kernel."""

    validate_points(x, y)
    if not x.is_cuda:
        raise ValueError("flash_c_transform requires CUDA tensors; use dense_c_transform on CPU.")
    c_transform_fwd = _load_flash_c_transform()
    return c_transform_fwd(
        x,
        y,
        psi,
        cost_scale=cost_scale,
        allow_tf32=allow_tf32,
        autotune=autotune,
        **kernel_kwargs,
    )


def solve_flash_wasserstein(
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
    allow_tf32: bool = True,
    autotune: bool = True,
    **kernel_kwargs,
):
    """Semi-dual subgradient solver backed by FlashSinkhorn's hard c-transform."""

    require_torch()
    validate_points(x, y)
    if not x.is_cuda:
        raise ValueError("solve_flash_wasserstein requires CUDA tensors; use solve_dense_semidual on CPU.")

    def oracle(psi):
        return flash_c_transform(
            x,
            y,
            psi,
            cost_scale=cost_scale,
            allow_tf32=allow_tf32,
            autotune=autotune,
            **kernel_kwargs,
        )

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
        backend="flash",
    )
