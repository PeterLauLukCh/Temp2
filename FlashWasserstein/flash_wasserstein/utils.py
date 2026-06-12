"""Shared utilities for FlashWasserstein solvers."""

from __future__ import annotations

from typing import Optional, Tuple

try:  # pragma: no cover - exercised by dependency-free import checks.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def require_torch():
    """Return torch or raise a clear optional dependency error."""

    if torch is None:
        raise ImportError(
            "FlashWasserstein requires PyTorch for this operation. "
            "Install torch, or run only static/documentation tooling."
        )
    return torch


def validate_points(x, y) -> Tuple[int, int, int]:
    """Validate point-cloud tensors and return ``(n, m, d)``."""

    th = require_torch()
    if not th.is_tensor(x) or not th.is_tensor(y):
        raise TypeError("x and y must be torch.Tensor objects.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2D tensors, got {x.shape} and {y.shape}.")
    n, d = x.shape
    m, d_y = y.shape
    if n <= 0 or m <= 0:
        raise ValueError("x and y must contain at least one point.")
    if d != d_y:
        raise ValueError(f"x and y must share feature dimension, got {d} and {d_y}.")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} and {y.device}.")
    return int(n), int(m), int(d)


def prepare_weights(
    x,
    y,
    a=None,
    b=None,
    *,
    normalize: bool = True,
) -> Tuple[object, object]:
    """Create or validate source/target weights as float32 tensors."""

    th = require_torch()
    n, m, _ = validate_points(x, y)
    device = x.device

    if a is None:
        a_t = th.full((n,), 1.0 / n, device=device, dtype=th.float32)
    else:
        a_t = th.as_tensor(a, device=device, dtype=th.float32)
    if b is None:
        b_t = th.full((m,), 1.0 / m, device=device, dtype=th.float32)
    else:
        b_t = th.as_tensor(b, device=device, dtype=th.float32)

    if a_t.shape != (n,):
        raise ValueError(f"a must have shape ({n},), got {tuple(a_t.shape)}.")
    if b_t.shape != (m,):
        raise ValueError(f"b must have shape ({m},), got {tuple(b_t.shape)}.")
    if a_t.requires_grad or b_t.requires_grad:
        raise ValueError("a and b must not require gradients.")
    if (a_t < 0).any().item() or (b_t < 0).any().item():
        raise ValueError("a and b must be nonnegative.")

    if normalize:
        a_sum = a_t.sum()
        b_sum = b_t.sum()
        if (not th.isfinite(a_sum).item()) or a_sum.item() <= 0:
            raise ValueError("a must have positive finite total mass.")
        if (not th.isfinite(b_sum).item()) or b_sum.item() <= 0:
            raise ValueError("b must have positive finite total mass.")
        a_t = a_t / a_sum
        b_t = b_t / b_sum
    else:
        if not th.allclose(a_t.sum(), b_t.sum(), rtol=1e-5, atol=1e-7):
            raise ValueError("a and b must have the same total mass when normalize=False.")

    return a_t.float(), b_t.float()


def prepare_psi(y, psi_init=None):
    """Create or validate the target potential."""

    th = require_torch()
    m = int(y.shape[0])
    if psi_init is None:
        return th.zeros((m,), device=y.device, dtype=th.float32)
    psi = th.as_tensor(psi_init, device=y.device, dtype=th.float32)
    if psi.shape != (m,):
        raise ValueError(f"psi_init must have shape ({m},), got {tuple(psi.shape)}.")
    return psi.clone()


def apply_gauge(psi, b, gauge: str = "weighted_mean_zero"):
    """Fix the additive-potential gauge."""

    if gauge == "weighted_mean_zero":
        return psi - (b * psi).sum()
    if gauge == "mean_zero":
        return psi - psi.mean()
    if gauge in ("none", None):
        return psi
    raise ValueError(
        'gauge must be one of {"weighted_mean_zero", "mean_zero", "none"}.'
    )


def assigned_mass_from_assignment(assignment, a, m: int):
    """Compute target masses induced by a deterministic assignment."""

    th = require_torch()
    idx = assignment.to(device=a.device, dtype=th.long)
    mass = th.zeros((m,), device=a.device, dtype=th.float32)
    mass.scatter_add_(0, idx, a.float())
    return mass


def semidual_value(c_values, psi, a, b) -> float:
    """Return the scalar semi-dual objective as a Python float."""

    return float(((a * c_values.float()).sum() + (b * psi.float()).sum()).item())


def transport_cost_from_assignment(x, y, assignment, a, cost_scale: float) -> float:
    """Cost of the deterministic Monge assignment induced by ``assignment``."""

    y_matched = y.float()[assignment.to(dtype=require_torch().long)]
    diff = x.float() - y_matched
    cost = cost_scale * (diff * diff).sum(dim=1)
    return float((a.float() * cost).sum().item())


def mass_error_l1(assigned_mass, b) -> float:
    """L1 target-marginal residual."""

    return float((assigned_mass.float() - b.float()).abs().sum().item())


def step_size(lr: float, step: int, schedule: str) -> float:
    """Learning-rate schedule for semi-dual ascent."""

    if schedule == "sqrt_decay":
        return float(lr) / float(step + 1) ** 0.5
    if schedule == "constant":
        return float(lr)
    raise ValueError('lr_schedule must be one of {"sqrt_decay", "constant"}.')
