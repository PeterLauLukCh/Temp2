"""Structured-cost Sinkhorn solvers built on FlashSinkhorn kernels.

This module implements balanced entropic OT for costs of the form

    C_ij = u_i + v_j - q_i^T k_j.

The solver reuses the existing FlashSinkhorn symmetric step by storing shifted
potentials f_hat = f - u and g_hat = g - v. Setting cost_scale=0.5 makes the
kernel dot contribution equal q_i^T k_j.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from flash_sinkhorn.kernels._common import log_weights
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    flashsinkhorn_symmetric_step,
)


def _make_eps_list(
    *,
    eps: Optional[float],
    n_iters: Optional[int],
    eps_list: Optional[Sequence[float]],
) -> Tuple[float, ...]:
    if eps_list is not None:
        out = tuple(float(e) for e in eps_list)
        if n_iters is not None:
            if int(n_iters) <= 0:
                raise ValueError("n_iters must be positive.")
            out = out[: int(n_iters)]
        if len(out) == 0:
            raise ValueError("eps_list must contain at least one value.")
    else:
        if eps is None or n_iters is None:
            raise ValueError("Structured Sinkhorn requires either eps_list or eps+n_iters.")
        if int(n_iters) <= 0:
            raise ValueError("n_iters must be positive.")
        out = tuple(float(eps) for _ in range(int(n_iters)))

    if any(e <= 0.0 for e in out):
        raise ValueError("All epsilon values must be positive.")
    return out


def _validate_structured_inputs(
    a: torch.Tensor,
    u: torch.Tensor,
    q: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
) -> None:
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("q and k must be 2D tensors with shapes (n,r) and (m,r).")
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError("u and v must be 1D tensors with shapes (n,) and (m,).")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D tensors with shapes (n,) and (m,).")

    n, r = q.shape
    m, r2 = k.shape
    if r != r2:
        raise ValueError("q and k must have the same feature dimension.")
    if u.shape[0] != n or a.shape[0] != n:
        raise ValueError("a/u shapes must match q.shape[0].")
    if v.shape[0] != m or b.shape[0] != m:
        raise ValueError("b/v shapes must match k.shape[0].")
    if not q.is_cuda:
        raise ValueError("Structured FlashSinkhorn requires CUDA tensors.")

    device = q.device
    for name, tensor in (("k", k), ("u", u), ("v", v), ("a", a), ("b", b)):
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same device as q.")
    for name, tensor in (("q", q), ("k", k), ("u", u), ("v", v), ("a", a), ("b", b)):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor.")


def sinkhorn_flashstyle_structured_symmetric(
    a: torch.Tensor,
    u: torch.Tensor,
    q: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    *,
    eps: Optional[float] = None,
    n_iters: Optional[int] = None,
    eps_list: Optional[Sequence[float]] = None,
    last_extrapolation: bool = True,
    allow_tf32: bool = True,
    use_exp2: bool = True,
    autotune: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    threshold: Optional[float] = None,
    check_every: int = 10,
    return_prelast: bool = False,
    return_n_iters: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run balanced symmetric Sinkhorn for ``u + v - q @ k.T``.

    Returns standard GeomLoss-style potentials ``f`` and ``g``. If
    ``return_prelast=True`` and ``last_extrapolation=True``, also returns the
    pre-final-extrapolation potentials used by the analytic gradient convention.
    """
    _validate_structured_inputs(a, u, q, b, v, k)
    schedule = _make_eps_list(eps=eps, n_iters=n_iters, eps_list=eps_list)
    if check_every <= 0:
        raise ValueError("check_every must be positive.")

    q_f32 = q.float().contiguous()
    k_f32 = k.float().contiguous()
    u_f32 = u.float().contiguous()
    v_f32 = v.float().contiguous()
    log_a = log_weights(a).contiguous()
    log_b = log_weights(b).contiguous()

    # Standard zero potentials imply shifted initialization f-u=-u, g-v=-v.
    f_hat = -u_f32.clone()
    g_hat = -v_f32.clone()

    prev_f_hat = None
    prev_g_hat = None
    n_iters_used = 0

    def step(
        f_current: torch.Tensor,
        g_current: torch.Tensor,
        step_eps: float,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return flashsinkhorn_symmetric_step(
            q_f32,
            k_f32,
            f_current,
            g_current,
            log_a,
            log_b,
            float(step_eps),
            cost_scale=0.5,
            alpha=float(alpha),
            damping_f=1.0,
            damping_g=1.0,
            allow_tf32=allow_tf32,
            use_exp2=use_exp2,
            autotune=autotune,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    eps0 = schedule[0]
    f_hat, g_hat = step(f_hat, g_hat, eps0, alpha=1.0)
    n_iters_used += 1

    for iter_idx, step_eps in enumerate(schedule):
        old_f_hat = f_hat
        old_g_hat = g_hat
        f_hat, g_hat = step(old_f_hat, old_g_hat, step_eps, alpha=0.5)
        n_iters_used += 1

        if threshold is not None and (iter_idx + 1) % check_every == 0:
            if prev_f_hat is None:
                prev_f_hat = f_hat.clone()
                prev_g_hat = g_hat.clone()
            else:
                f_change = (f_hat - prev_f_hat).abs().max().item()
                g_change = (g_hat - prev_g_hat).abs().max().item()
                if max(f_change, g_change) < float(threshold):
                    break
                prev_f_hat.copy_(f_hat)
                prev_g_hat.copy_(g_hat)

    if last_extrapolation:
        f_prelast = f_hat + u_f32
        g_prelast = g_hat + v_f32
        final_eps = schedule[-1]
        f_hat, g_hat = step(f_hat, g_hat, final_eps, alpha=1.0)
        n_iters_used += 1

    f = f_hat + u_f32
    g = g_hat + v_f32

    if return_prelast and last_extrapolation:
        if return_n_iters:
            return f, g, f_prelast, g_prelast, n_iters_used
        return f, g, f_prelast, g_prelast
    if return_n_iters:
        return f, g, n_iters_used
    return f, g
