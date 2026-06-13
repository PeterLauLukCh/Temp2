"""Shared row-conditional OT couplers for image flow matching experiments.

The coupler in this module is intentionally independent from the CFM loss:
it only chooses paired endpoints.  The training script then applies the
standard linear CFM path ``x_t = (1 - t) x_0 + t x_1`` and target
``u_t = x_1 - x_0``.  This keeps the contract compatible with OT-CFM while
letting us swap local POT, dense Sinkhorn, and FlashSinkhorn couplings.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


COUPLING_MODES = {
    "independent",
    "local_exact_pot",
    "local_entropic",
    "global_pot_exact_small",
    "allgather_dense_entropic",
    "flash_global_entropic",
}

_MODE_ALIASES = {
    "local_pot_exact_row": "local_exact_pot",
    "local_pot_exact_stock": "local_exact_pot",
    "global_dense_sinkhorn": "allgather_dense_entropic",
    "global_flash_sinkhorn": "flash_global_entropic",
}


def normalize_coupling_mode(mode: str) -> str:
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in COUPLING_MODES:
        valid = sorted(COUPLING_MODES | set(_MODE_ALIASES))
        raise ValueError(f"unknown coupling mode {mode!r}; expected one of {valid}")
    return mode


def parse_csv_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def parse_csv_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v]


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distributed_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def all_gather_tensor(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, dim=0)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


@dataclass
class CoupledBatch:
    x0: torch.Tensor
    x1: torch.Tensor
    metrics: dict[str, Any]
    y0: Optional[torch.Tensor] = None
    y1: Optional[torch.Tensor] = None


class FeatureProjector:
    """Flatten features, optionally through a fixed random projection."""

    def __init__(self, feature_dim: int = 0, seed: int = 0):
        self.feature_dim = int(feature_dim)
        self.seed = int(seed)
        self.proj: Optional[torch.Tensor] = None
        self.flat_dim: Optional[int] = None

    @property
    def dim(self) -> Optional[int]:
        if self.feature_dim > 0:
            return self.feature_dim
        return self.flat_dim

    def _maybe_init_projection(self, flat_dim: int, device: torch.device) -> None:
        if self.feature_dim <= 0:
            self.flat_dim = flat_dim
            return
        if self.proj is not None:
            if self.flat_dim != flat_dim:
                raise ValueError(f"projector expected flat dim {self.flat_dim}, got {flat_dim}")
            if self.proj.device != device:
                self.proj = self.proj.to(device)
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        proj = torch.randn(self.feature_dim, flat_dim, generator=generator, dtype=torch.float32)
        proj = proj / math.sqrt(float(flat_dim))
        self.proj = proj.to(device)
        self.flat_dim = flat_dim

    @torch.no_grad()
    def project(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.detach().float().flatten(1)
        self._maybe_init_projection(flat.shape[1], flat.device)
        if self.proj is None:
            return flat.contiguous()
        return (flat @ self.proj.t()).contiguous()


class TensorQueue:
    """Newest-first FIFO queue for target samples, features, and optional labels."""

    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.x: Optional[torch.Tensor] = None
        self.h: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

    def update(
        self,
        x_new: torch.Tensor,
        h_new: torch.Tensor,
        y_new: Optional[torch.Tensor] = None,
    ) -> None:
        x_new = x_new.detach()
        h_new = h_new.detach()
        y_new = y_new.detach() if y_new is not None else None
        if self.x is None:
            self.x = x_new[: self.max_size].contiguous()
            self.h = h_new[: self.max_size].contiguous()
            self.y = y_new[: self.max_size].contiguous() if y_new is not None else None
            return
        self.x = torch.cat([x_new, self.x], dim=0)[: self.max_size].contiguous()
        self.h = torch.cat([h_new, self.h], dim=0)[: self.max_size].contiguous()
        if y_new is None:
            self.y = None
            return
        old_y = self.y
        if old_y is None:
            old_len = max(self.x.shape[0] - min(y_new.shape[0], self.max_size), 0)
            old_y = torch.full((old_len,), -1, dtype=y_new.dtype, device=y_new.device)
        self.y = torch.cat([y_new, old_y], dim=0)[: self.max_size].contiguous()

    def get(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.x is None or self.h is None:
            raise RuntimeError("target queue is empty.")
        return self.x, self.h, self.y


def default_cost_scale(feature_dim: int) -> float:
    return 1.0 / (2.0 * float(feature_dim))


def normalized_sqeuclidean_cost(x: torch.Tensor, y: torch.Tensor, cost_scale: float) -> torch.Tensor:
    return torch.cdist(x.float(), y.float(), p=2).pow(2).mul_(float(cost_scale))


def pair_cost(x: torch.Tensor, y: torch.Tensor, cost_scale: float) -> torch.Tensor:
    return (x.float() - y.float()).pow(2).sum(dim=1).mul(float(cost_scale))


def duplicate_fraction(indices: torch.Tensor, m: int) -> float:
    if indices.numel() == 0:
        return 0.0
    unique = int(torch.unique(indices.detach()).numel())
    return float(1.0 - unique / min(int(indices.numel()), int(m)))


def _np_rng(seed: int, step: int, rank: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) + 1009 * int(step) + 9176 * int(rank))


def _sample_rows_from_numpy_plan(
    plan: np.ndarray,
    row_ids: np.ndarray,
    *,
    seed: int,
    step: int,
    rank: int,
) -> np.ndarray:
    rng = _np_rng(seed, step, rank)
    n, m = plan.shape
    out = np.empty(len(row_ids), dtype=np.int64)
    for out_idx, row in enumerate(row_ids):
        probs = plan[int(row)].astype(np.float64) * float(n)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            probs = np.full(m, 1.0 / m, dtype=np.float64)
        else:
            probs = probs / total
        out[out_idx] = rng.choice(m, p=probs)
    return out


@torch.no_grad()
def pot_row_conditional_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    cost_scale: float,
    seed: int,
    step: int,
    rank: int,
    num_threads: int | str = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    import ot as pot

    x_flat = x.float().flatten(1)
    y_flat = y.float().flatten(1)
    n, m = x_flat.shape[0], y_flat.shape[0]
    a = pot.unif(n)
    b = pot.unif(m)
    sync_if_cuda(x.device)
    t0 = time.perf_counter()
    cost = normalized_sqeuclidean_cost(x_flat, y_flat, cost_scale)
    sync_if_cuda(x.device)
    cost_time = time.perf_counter() - t0
    cost_np = cost.detach().cpu().numpy()
    t1 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = pot.emd(a, b, cost_np, numThreads=num_threads)
    emd_time = time.perf_counter() - t1
    rows_np = row_ids.detach().cpu().numpy()
    j_np = _sample_rows_from_numpy_plan(plan, rows_np, seed=seed, step=step, rank=rank)
    j = torch.from_numpy(j_np).to(device=x.device, dtype=torch.long)
    col = plan.sum(axis=0)
    row = plan.sum(axis=1)
    metrics = {
        "pot_cost_time_s": cost_time,
        "pot_emd_time_s": emd_time,
        "marginal_l1": float(np.abs(row - a).sum() + np.abs(col - b).sum()),
        "plan_cost": float((plan * cost_np).sum()),
        "warnings": "; ".join(str(w.message) for w in caught),
    }
    return j, metrics


@torch.no_grad()
def dense_sinkhorn_row_conditional_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    eps: float,
    n_iters: int,
    cost_scale: float,
    seed: int,
    step: int,
    diagnostics_plan_limit: int = 4_000_000,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x_flat = x.float().flatten(1)
    y_flat = y.float().flatten(1)
    n, m = x_flat.shape[0], y_flat.shape[0]
    cost = normalized_sqeuclidean_cost(x_flat, y_flat, cost_scale)
    log_k = -cost / float(eps)
    log_a = torch.full((n,), -math.log(n), device=x.device, dtype=torch.float32)
    log_b = torch.full((m,), -math.log(m), device=x.device, dtype=torch.float32)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(int(n_iters)):
        log_u = log_a - torch.logsumexp(log_k + log_v.view(1, -1), dim=1)
        log_v = log_b - torch.logsumexp(log_k.t() + log_u.view(1, -1), dim=1)

    logits = log_k[row_ids] + log_v.view(1, -1)
    probs = F.softmax(logits, dim=1)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) + 104729 * int(step))
    j = torch.multinomial(probs, 1, replacement=True, generator=generator).flatten()

    metrics: dict[str, Any] = {"sinkhorn_iters": int(n_iters)}
    if n * m <= diagnostics_plan_limit:
        plan = torch.exp(log_k + log_u.view(-1, 1) + log_v.view(1, -1))
        marginal_l1 = (plan.sum(dim=1) - torch.exp(log_a)).abs().sum()
        marginal_l1 = marginal_l1 + (plan.sum(dim=0) - torch.exp(log_b)).abs().sum()
        metrics["marginal_l1"] = float(marginal_l1.item())
        metrics["plan_cost"] = float((plan * cost).sum().item())
    return j, metrics


def _add_flash_sinkhorn_path() -> None:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "code" / "src",
        here.parents[2] / "code" / "src",
        here.parents[1] / "code" / "src",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _import_flash_sinkhorn_solver():
    try:
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating

        return sinkhorn_flashstyle_alternating
    except ModuleNotFoundError:
        _add_flash_sinkhorn_path()
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating

        return sinkhorn_flashstyle_alternating


@torch.no_grad()
def flash_sinkhorn_row_conditional_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    eps: float,
    n_iters: int,
    cost_scale: float,
    seed: int,
    step: int,
    allow_tf32: bool = True,
    autotune: bool = True,
    threshold: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not x.is_cuda:
        raise ValueError("flash_global_entropic requires CUDA tensors.")
    solver = _import_flash_sinkhorn_solver()
    n, m = x.shape[0], y.shape[0]
    a = torch.full((n,), 1.0 / n, device=x.device, dtype=torch.float32)
    b = torch.full((m,), 1.0 / m, device=y.device, dtype=torch.float32)
    sync_if_cuda(x.device)
    t0 = time.perf_counter()
    _f, g, used = solver(
        x.float().contiguous(),
        y.float().contiguous(),
        a,
        b,
        eps=float(eps),
        n_iters=int(n_iters),
        cost_scale=float(cost_scale),
        allow_tf32=allow_tf32,
        autotune=autotune,
        threshold=threshold,
        return_n_iters=True,
    )
    sync_if_cuda(x.device)
    solve_time = time.perf_counter() - t0
    rows = x[row_ids].float()
    cost_rows = normalized_sqeuclidean_cost(rows, y.float(), cost_scale)
    logits = (g.float().view(1, -1) - cost_rows) / float(eps) - math.log(m)
    probs = F.softmax(logits, dim=1)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) + 104729 * int(step))
    j = torch.multinomial(probs, 1, replacement=True, generator=generator).flatten()
    return j, {
        "sinkhorn_iters": int(used),
        "flash_sinkhorn_solve_time_s": solve_time,
    }


class OTCouplingSampler:
    """One row-conditional coupling interface for pixel and latent OT-CFM."""

    def __init__(
        self,
        *,
        mode: str,
        context_size: int = 8192,
        eps: float = 0.05,
        sinkhorn_iters: int = 80,
        cost_scale: Optional[float] = None,
        cost_feature_dim: int = 0,
        feature_projector: Optional[Any] = None,
        seed: int = 0,
        pot_max_context: int = 2048,
        pot_num_threads: int | str = 1,
        flash_allow_tf32: bool = True,
        flash_autotune: bool = True,
        diagnostics_plan_limit: int = 4_000_000,
        class_aware: bool = False,
    ):
        self.mode = normalize_coupling_mode(mode)
        self.context_size = int(context_size)
        self.eps = float(eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.cost_scale = None if cost_scale is None else float(cost_scale)
        self.seed = int(seed)
        self.pot_max_context = int(pot_max_context)
        self.pot_num_threads = pot_num_threads
        self.flash_allow_tf32 = bool(flash_allow_tf32)
        self.flash_autotune = bool(flash_autotune)
        self.diagnostics_plan_limit = int(diagnostics_plan_limit)
        self.class_aware = bool(class_aware)
        self.feature_projector = feature_projector or FeatureProjector(cost_feature_dim, seed=seed)
        self.target_queue = TensorQueue(max_size=self.context_size)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_projector.project(x)

    def _cost_scale_for(self, h: torch.Tensor) -> float:
        return self.cost_scale if self.cost_scale is not None else default_cost_scale(h.shape[1])

    @torch.no_grad()
    def _sample_unrestricted(
        self,
        source_h: torch.Tensor,
        target_h: torch.Tensor,
        row_ids: torch.Tensor,
        *,
        rank: int,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if target_h.shape[0] == 0:
            raise RuntimeError("cannot sample from an empty target context.")
        cost_scale = self._cost_scale_for(source_h)
        if self.mode in {"local_exact_pot", "global_pot_exact_small"}:
            if self.mode == "global_pot_exact_small" and target_h.shape[0] > self.pot_max_context:
                raise ValueError(
                    f"context_size={target_h.shape[0]} exceeds pot_max_context={self.pot_max_context}"
                )
            return pot_row_conditional_indices(
                source_h,
                target_h,
                row_ids,
                cost_scale=cost_scale,
                seed=self.seed,
                step=step,
                rank=rank,
                num_threads=self.pot_num_threads,
            )
        if self.mode in {"local_entropic", "allgather_dense_entropic"}:
            return dense_sinkhorn_row_conditional_indices(
                source_h,
                target_h,
                row_ids,
                eps=self.eps,
                n_iters=self.sinkhorn_iters,
                cost_scale=cost_scale,
                seed=self.seed,
                step=step,
                diagnostics_plan_limit=self.diagnostics_plan_limit,
            )
        if self.mode == "flash_global_entropic":
            return flash_sinkhorn_row_conditional_indices(
                source_h,
                target_h,
                row_ids,
                eps=self.eps,
                n_iters=self.sinkhorn_iters,
                cost_scale=cost_scale,
                seed=self.seed,
                step=step,
                allow_tf32=self.flash_allow_tf32,
                autotune=self.flash_autotune,
            )
        raise AssertionError(f"unhandled mode {self.mode}")

    @torch.no_grad()
    def _sample_class_aware(
        self,
        source_h: torch.Tensor,
        target_h: torch.Tensor,
        row_ids: torch.Tensor,
        source_y: torch.Tensor,
        target_y: torch.Tensor,
        *,
        rank: int,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        out = torch.empty(row_ids.shape[0], device=row_ids.device, dtype=torch.long)
        row_labels = source_y[row_ids]
        labels = torch.unique(row_labels)
        aggregate: dict[str, Any] = {
            "class_aware": True,
            "class_groups": int(labels.numel()),
            "class_fallbacks": 0,
        }
        for label in labels.tolist():
            local_mask = row_labels == label
            source_idx = torch.nonzero(source_y == label, as_tuple=False).flatten()
            target_idx = torch.nonzero(target_y == label, as_tuple=False).flatten()
            if source_idx.numel() == 0 or target_idx.numel() == 0:
                aggregate["class_fallbacks"] += int(local_mask.sum().item())
                j, _ = self._sample_unrestricted(
                    source_h,
                    target_h,
                    row_ids[local_mask],
                    rank=rank,
                    step=step + int(label),
                )
                out[local_mask] = j
                continue
            sub_rows = torch.searchsorted(source_idx, row_ids[local_mask])
            j_sub, _ = self._sample_unrestricted(
                source_h[source_idx],
                target_h[target_idx],
                sub_rows,
                rank=rank,
                step=step + int(label),
            )
            out[local_mask] = target_idx[j_sub]
        return out, aggregate

    @torch.no_grad()
    def sample_pairs(
        self,
        x0_local: torch.Tensor,
        x1_local: torch.Tensor,
        *,
        y1_local: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> CoupledBatch:
        rank, world_size = distributed_rank_world()
        local_batch = x0_local.shape[0]
        device = x0_local.device
        metrics: dict[str, Any] = {
            "mode": self.mode,
            "rank": rank,
            "world_size": world_size,
            "local_batch": int(local_batch),
            "eps": self.eps,
            "sinkhorn_iters": self.sinkhorn_iters,
        }

        h0_local = self._project(x0_local)
        h1_local = self._project(x1_local)
        cost_scale = self._cost_scale_for(h0_local)
        metrics["feature_dim"] = int(h0_local.shape[1])
        metrics["cost_scale"] = float(cost_scale)

        if self.mode == "independent":
            metrics.update(
                {
                    "context_size": int(local_batch),
                    "source_context_size": int(local_batch),
                    "sample_cost": float(pair_cost(h0_local, h1_local, cost_scale).mean().item()),
                    "duplicate_fraction": 0.0,
                }
            )
            return CoupledBatch(
                x0_local.detach(),
                x1_local.detach(),
                metrics,
                y1=y1_local.detach() if y1_local is not None else None,
            )

        if self.mode in {"local_exact_pot", "local_entropic"}:
            row_ids = torch.arange(local_batch, device=device, dtype=torch.long)
            if self.class_aware and y1_local is not None:
                j, extra = self._sample_class_aware(
                    h0_local,
                    h1_local,
                    row_ids,
                    y1_local,
                    y1_local,
                    rank=rank,
                    step=step,
                )
            else:
                j, extra = self._sample_unrestricted(h0_local, h1_local, row_ids, rank=rank, step=step)
            paired_y1 = y1_local[j] if y1_local is not None else None
            metrics.update(extra)
            metrics.update(
                {
                    "context_size": int(local_batch),
                    "source_context_size": int(local_batch),
                    "sample_cost": float(pair_cost(h0_local, h1_local[j], cost_scale).mean().item()),
                    "duplicate_fraction": duplicate_fraction(j, local_batch),
                }
            )
            return CoupledBatch(x0_local.detach(), x1_local[j].detach(), metrics, y1=paired_y1)

        source_x = all_gather_tensor(x0_local.detach())
        target_x_fresh = all_gather_tensor(x1_local.detach())
        source_h = all_gather_tensor(h0_local.detach())
        target_h_fresh = all_gather_tensor(h1_local.detach())
        labels_global = all_gather_tensor(y1_local.detach()) if y1_local is not None else None
        self.target_queue.update(target_x_fresh, target_h_fresh, labels_global)
        target_x, target_h, target_y = self.target_queue.get()
        target_x = target_x[: self.context_size]
        target_h = target_h[: self.context_size]
        if target_y is not None:
            target_y = target_y[: self.context_size]

        start = rank * local_batch
        row_ids = torch.arange(start, start + local_batch, device=device, dtype=torch.long)
        if row_ids[-1].item() >= source_h.shape[0]:
            raise RuntimeError("local DDP row slice exceeds gathered source context.")

        if self.class_aware:
            if labels_global is None or target_y is None:
                raise ValueError("class_aware=True requires y1_local labels.")
            j, extra = self._sample_class_aware(
                source_h,
                target_h,
                row_ids,
                labels_global,
                target_y,
                rank=rank,
                step=step,
            )
        else:
            j, extra = self._sample_unrestricted(source_h, target_h, row_ids, rank=rank, step=step)

        paired_y1 = target_y[j] if target_y is not None else None
        metrics.update(extra)
        metrics.update(
            {
                "context_size": int(target_h.shape[0]),
                "source_context_size": int(source_h.shape[0]),
                "sample_cost": float(pair_cost(source_h[row_ids], target_h[j], cost_scale).mean().item()),
                "duplicate_fraction": duplicate_fraction(j, target_h.shape[0]),
            }
        )
        return CoupledBatch(source_x[row_ids].detach(), target_x[j].detach(), metrics, y1=paired_y1)
