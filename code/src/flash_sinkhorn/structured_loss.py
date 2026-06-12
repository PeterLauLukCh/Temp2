"""Public API for structured-cost FlashSinkhorn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from flash_sinkhorn.kernels._common import log_weights
from flash_sinkhorn.kernels.apply_flash import apply_plan_mat_flashstyle
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import flashsinkhorn_lse_fused
from flash_sinkhorn.structured_solvers import (
    _make_eps_list,
    sinkhorn_flashstyle_structured_symmetric,
)


@dataclass(frozen=True)
class _StructuredConfig:
    last_extrapolation: bool
    allow_tf32: bool
    use_exp2: bool
    autotune: bool
    block_m: Optional[int]
    block_n: Optional[int]
    block_k: Optional[int]
    num_warps: Optional[int]
    num_stages: int
    threshold: Optional[float]
    inner_iterations: int


@dataclass(frozen=True)
class _ParsedStructuredInputs:
    a: torch.Tensor
    u: torch.Tensor
    q: torch.Tensor
    b: torch.Tensor
    v: torch.Tensor
    k: torch.Tensor
    u_view_shape: Tuple[int, ...]
    v_view_shape: Tuple[int, ...]


def _as_float_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not x.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor.")
    return x


def _normalize_weights(w: torch.Tensor) -> torch.Tensor:
    w = w.float()
    z = w.sum(dim=-1, keepdim=True).clamp(min=1e-40)
    return w / z


def _as_vector(x: torch.Tensor, n: int, name: str) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    x = _as_float_tensor(x, name)
    if x.ndim == 1 and x.shape == (n,):
        return x, (n,)
    if x.ndim == 2 and x.shape == (n, 1):
        return x[:, 0], (n, 1)
    raise ValueError(f"{name} must have shape ({n},) or ({n},1).")


def _process_structured_args(*args, normalize: bool) -> _ParsedStructuredInputs:
    if len(args) == 4:
        u, q, v, k = args
        a = None
        b = None
    elif len(args) == 6:
        a, u, q, b, v, k = args
    else:
        raise TypeError(
            "StructuredSamplesLoss expects either (u, q, v, k) or "
            f"(a, u, q, b, v, k). Got {len(args)} arguments."
        )

    q = _as_float_tensor(q, "q")
    k = _as_float_tensor(k, "k")
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("q and k must be 2D tensors with shapes (n,r) and (m,r).")
    n, r = q.shape
    m, r2 = k.shape
    if r != r2:
        raise ValueError("q and k must have the same feature dimension.")

    u, u_view_shape = _as_vector(u, n, "u")
    v, v_view_shape = _as_vector(v, m, "v")

    if a is None:
        a = torch.full((n,), 1.0 / n, device=q.device, dtype=torch.float32)
    else:
        a, _ = _as_vector(a, n, "a")
    if b is None:
        b = torch.full((m,), 1.0 / m, device=k.device, dtype=torch.float32)
    else:
        b, _ = _as_vector(b, m, "b")

    if normalize:
        a = _normalize_weights(a)
        b = _normalize_weights(b)

    return _ParsedStructuredInputs(
        a=a,
        u=u,
        q=q,
        b=b,
        v=v,
        k=k,
        u_view_shape=u_view_shape,
        v_view_shape=v_view_shape,
    )


def _validate_same_device(parsed: _ParsedStructuredInputs) -> None:
    if not parsed.q.is_cuda:
        raise ValueError("StructuredSamplesLoss requires CUDA tensors.")
    device = parsed.q.device
    for name in ("a", "u", "b", "v", "k"):
        tensor = getattr(parsed, name)
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as q.")


class _StructuredPotentialGradFn(torch.autograd.Function):
    """First-order structured gradients; double backward is intentionally absent."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        f_grad: torch.Tensor,
        g_grad: torch.Tensor,
        eps: float,
        allow_tf32: bool,
        use_exp2: bool,
        autotune: bool,
        block_m: Optional[int],
        block_n: Optional[int],
        block_k: Optional[int],
        num_warps: Optional[int],
        num_stages: int,
        grad_scale: torch.Tensor,
        compute_grad_q: bool,
        compute_grad_k: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        q_f32 = q.float().contiguous()
        k_f32 = k.float().contiguous()
        f_hat = (f_grad.float() - u.float()).contiguous()
        g_hat = (g_grad.float() - v.float()).contiguous()
        log_a = log_weights(a).contiguous()
        log_b = log_weights(b).contiguous()

        scale = grad_scale.to(device=q.device, dtype=torch.float32)
        if scale.numel() != 1:
            raise ValueError("grad_scale must be scalar.")

        grad_q = None
        grad_k = None
        if compute_grad_q:
            f_cond_hat = flashsinkhorn_lse_fused(
                q_f32,
                k_f32,
                g_hat,
                log_b,
                float(eps),
                cost_scale=0.5,
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            grad_q = -apply_plan_mat_flashstyle(
                q_f32,
                k_f32,
                f_cond_hat,
                g_hat,
                log_a,
                log_b,
                k_f32,
                eps=float(eps),
                axis=1,
                cost_scale=0.5,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=4 if num_warps is None else int(num_warps),
                num_stages=num_stages,
                use_exp2=use_exp2,
                allow_tf32=allow_tf32,
                autotune=autotune,
            )
            grad_q = grad_q * scale

        if compute_grad_k:
            g_cond_hat = flashsinkhorn_lse_fused(
                k_f32,
                q_f32,
                f_hat,
                log_a,
                float(eps),
                cost_scale=0.5,
                allow_tf32=allow_tf32,
                use_exp2=use_exp2,
                autotune=autotune,
                block_m=block_n,
                block_n=block_m,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            grad_k = -apply_plan_mat_flashstyle(
                q_f32,
                k_f32,
                f_hat,
                g_cond_hat,
                log_a,
                log_b,
                q_f32,
                eps=float(eps),
                axis=0,
                cost_scale=0.5,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=4 if num_warps is None else int(num_warps),
                num_stages=num_stages,
                use_exp2=use_exp2,
                allow_tf32=allow_tf32,
                autotune=autotune,
            )
            grad_k = grad_k * scale

        return grad_q, grad_k

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_grad_q: Optional[torch.Tensor], grad_grad_k: Optional[torch.Tensor]
    ):
        if grad_grad_q is not None or grad_grad_k is not None:
            raise NotImplementedError(
                "Double backward/HVP is not supported for StructuredSamplesLoss."
            )
        return (None,) * 20


class _StructuredCostFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        a: torch.Tensor,
        u: torch.Tensor,
        q: torch.Tensor,
        b: torch.Tensor,
        v: torch.Tensor,
        k: torch.Tensor,
        eps_list: Tuple[float, ...],
        config: _StructuredConfig,
    ) -> torch.Tensor:
        if config.last_extrapolation:
            f_cost, g_cost, f_grad, g_grad = sinkhorn_flashstyle_structured_symmetric(
                a,
                u,
                q,
                b,
                v,
                k,
                eps_list=eps_list,
                last_extrapolation=True,
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                autotune=config.autotune,
                block_m=config.block_m,
                block_n=config.block_n,
                block_k=config.block_k,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                threshold=config.threshold,
                check_every=config.inner_iterations,
                return_prelast=True,
            )
        else:
            f_cost, g_cost = sinkhorn_flashstyle_structured_symmetric(
                a,
                u,
                q,
                b,
                v,
                k,
                eps_list=eps_list,
                last_extrapolation=False,
                allow_tf32=config.allow_tf32,
                use_exp2=config.use_exp2,
                autotune=config.autotune,
                block_m=config.block_m,
                block_n=config.block_n,
                block_k=config.block_k,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                threshold=config.threshold,
                check_every=config.inner_iterations,
            )
            f_grad, g_grad = f_cost, g_cost

        ctx.save_for_backward(a, u, q, b, v, k, f_cost, g_cost, f_grad, g_grad)
        ctx.eps = float(eps_list[-1])
        ctx.config = config
        return (a.float() * f_cost).sum() + (b.float() * g_cost).sum()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, u, q, b, v, k, f_cost, g_cost, f_grad, g_grad = ctx.saved_tensors
        config = ctx.config

        grad_a = grad_u = grad_q = grad_b = grad_v = grad_k = None

        if q.requires_grad or k.requires_grad:
            grad_q_val, grad_k_val = _StructuredPotentialGradFn.apply(
                q,
                k,
                a,
                b,
                u,
                v,
                f_grad,
                g_grad,
                ctx.eps,
                config.allow_tf32,
                config.use_exp2,
                config.autotune,
                config.block_m,
                config.block_n,
                config.block_k,
                config.num_warps,
                config.num_stages,
                grad_out,
                q.requires_grad,
                k.requires_grad,
            )
            grad_q = grad_q_val if q.requires_grad else None
            grad_k = grad_k_val if k.requires_grad else None

        if a.requires_grad:
            grad_a = grad_out * f_cost
        if b.requires_grad:
            grad_b = grad_out * g_cost
        if u.requires_grad:
            grad_u = grad_out * a.float()
        if v.requires_grad:
            grad_v = grad_out * b.float()

        return (
            grad_a,
            grad_u,
            grad_q,
            grad_b,
            grad_v,
            grad_k,
            None,
            None,
        )


class StructuredSamplesLoss(torch.nn.Module):
    """Sinkhorn OT for finite structured costs ``u_i + v_j - q_i^T k_j``.

    This v1 API supports balanced, non-debiased, single-cloud OT on CUDA. It is
    intended for finite-rank structured costs such as spectral intrinsic costs
    and cosine/DINO costs represented as biased dot products.
    """

    def __init__(
        self,
        *,
        eps: Optional[float] = None,
        n_iters: Optional[int] = None,
        eps_list: Optional[Sequence[float]] = None,
        potentials: bool = False,
        normalize: bool = True,
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
        inner_iterations: int = 10,
    ):
        super().__init__()
        self.eps_list = _make_eps_list(eps=eps, n_iters=n_iters, eps_list=eps_list)
        if inner_iterations <= 0:
            raise ValueError("inner_iterations must be positive.")

        self.potentials = bool(potentials)
        self.normalize = bool(normalize)
        self.config = _StructuredConfig(
            last_extrapolation=bool(last_extrapolation),
            allow_tf32=bool(allow_tf32),
            use_exp2=bool(use_exp2),
            autotune=bool(autotune),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=int(num_stages),
            threshold=None if threshold is None else float(threshold),
            inner_iterations=int(inner_iterations),
        )

    def forward(self, *args):  # type: ignore[override]
        parsed = _process_structured_args(*args, normalize=self.normalize)
        _validate_same_device(parsed)

        if self.potentials:
            f, g = sinkhorn_flashstyle_structured_symmetric(
                parsed.a,
                parsed.u,
                parsed.q,
                parsed.b,
                parsed.v,
                parsed.k,
                eps_list=self.eps_list,
                last_extrapolation=self.config.last_extrapolation,
                allow_tf32=self.config.allow_tf32,
                use_exp2=self.config.use_exp2,
                autotune=self.config.autotune,
                block_m=self.config.block_m,
                block_n=self.config.block_n,
                block_k=self.config.block_k,
                num_warps=self.config.num_warps,
                num_stages=self.config.num_stages,
                threshold=self.config.threshold,
                check_every=self.config.inner_iterations,
            )
            return f.reshape(parsed.u_view_shape), g.reshape(parsed.v_view_shape)

        return _StructuredCostFn.apply(
            parsed.a,
            parsed.u,
            parsed.q,
            parsed.b,
            parsed.v,
            parsed.k,
            self.eps_list,
            self.config,
        )
