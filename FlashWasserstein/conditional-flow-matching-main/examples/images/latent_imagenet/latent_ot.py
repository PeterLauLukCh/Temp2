"""Latent OT samplers for ImageNet latent CFM experiments.

The training-compatible contract is intentionally small: given local source
latents ``z0`` and local target latents ``z1``, return paired tensors with the
same shape.  The global methods all-gather the fresh DDP minibatch and maintain
a target FIFO queue, then sample targets from a row conditional OT coupling.
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


_LOCAL_PKG_ROOT = Path(__file__).resolve().parents[3]
if str(_LOCAL_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PKG_ROOT))


def _add_flash_sinkhorn_path() -> None:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[5] / "code" / "src",
        here.parents[4] / "code" / "src",
        here.parents[4].parent / "code" / "src",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


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


def parse_csv_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def parse_csv_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v]


def parse_num_threads(value: int | str) -> int | str:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("num_threads must be positive or 'max'")
        return value
    value = str(value).strip()
    if value == "max":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("num_threads must be positive or 'max'")
    return parsed


@dataclass
class PairSample:
    z0: torch.Tensor
    z1: torch.Tensor
    metrics: dict[str, Any]


class LatentProjector:
    """Fixed random projection plus feature standardization."""

    def __init__(self, proj: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        if proj.dim() != 2:
            raise ValueError("proj must have shape [proj_dim, flat_dim].")
        self.proj = proj.float().contiguous()
        self.mean = mean.float().contiguous().view(1, -1)
        self.std = std.float().clamp_min(1e-6).contiguous().view(1, -1)
        if self.mean.shape[1] != self.proj.shape[0] or self.std.shape[1] != self.proj.shape[0]:
            raise ValueError("projection statistics do not match projection dimension.")

    @property
    def dim(self) -> int:
        return int(self.proj.shape[0])

    @property
    def flat_dim(self) -> int:
        return int(self.proj.shape[1])

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "LatentProjector":
        payload = torch.load(Path(path).expanduser(), map_location="cpu")
        projector = cls(payload["proj"], payload["mean"], payload["std"])
        return projector.to(device)

    @classmethod
    def identity(cls, flat_dim: int, device: torch.device) -> "LatentProjector":
        proj = torch.eye(flat_dim, dtype=torch.float32)
        mean = torch.zeros(flat_dim, dtype=torch.float32)
        std = torch.ones(flat_dim, dtype=torch.float32)
        return cls(proj, mean, std).to(device)

    def to(self, device: torch.device) -> "LatentProjector":
        self.proj = self.proj.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @torch.no_grad()
    def project(self, z: torch.Tensor) -> torch.Tensor:
        flat = z.detach().float().flatten(1)
        if flat.shape[1] != self.flat_dim:
            raise ValueError(f"expected flattened latent dim {self.flat_dim}, got {flat.shape[1]}")
        h = flat @ self.proj.t()
        return ((h - self.mean) / self.std).contiguous()


class TensorQueue:
    """Newest-first FIFO queue for target latents and projected features."""

    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.z: Optional[torch.Tensor] = None
        self.h: Optional[torch.Tensor] = None

    def update(self, z_new: torch.Tensor, h_new: torch.Tensor) -> None:
        if self.max_size <= 0:
            self.z = z_new.detach()
            self.h = h_new.detach()
            return
        z_new = z_new.detach()
        h_new = h_new.detach()
        if self.z is None:
            self.z = z_new[: self.max_size].contiguous()
            self.h = h_new[: self.max_size].contiguous()
        else:
            self.z = torch.cat([z_new, self.z], dim=0)[: self.max_size].contiguous()
            self.h = torch.cat([h_new, self.h], dim=0)[: self.max_size].contiguous()

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.z is None or self.h is None:
            raise RuntimeError("target queue is empty.")
        return self.z, self.h


def normalized_sqeuclidean_cost(x: torch.Tensor, y: torch.Tensor, cost_scale: float) -> torch.Tensor:
    return torch.cdist(x.float(), y.float(), p=2).pow(2).mul_(cost_scale)


def pair_cost(x: torch.Tensor, y: torch.Tensor, cost_scale: float) -> torch.Tensor:
    return (x.float() - y.float()).pow(2).sum(dim=1).mul(cost_scale)


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
        raise ValueError("global_flash_sinkhorn requires CUDA tensors.")
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


class LatentOTPlanSampler:
    """One sampler interface for all latent CFM coupling modes."""

    def __init__(
        self,
        *,
        mode: str,
        projector: LatentProjector,
        context_size: int = 8192,
        eps: float = 0.05,
        sinkhorn_iters: int = 80,
        cost_scale: Optional[float] = None,
        seed: int = 0,
        pot_max_context: int = 2048,
        pot_num_threads: int | str = 1,
        flash_allow_tf32: bool = True,
        flash_autotune: bool = True,
        diagnostics_plan_limit: int = 4_000_000,
    ):
        valid = {
            "independent",
            "local_pot_exact_stock",
            "local_pot_exact_row",
            "global_pot_exact_small",
            "global_dense_sinkhorn",
            "global_flash_sinkhorn",
        }
        if mode not in valid:
            raise ValueError(f"unknown latent OT mode {mode!r}; expected one of {sorted(valid)}")
        self.mode = mode
        self.projector = projector
        self.context_size = int(context_size)
        self.eps = float(eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.cost_scale = float(cost_scale) if cost_scale is not None else 1.0 / (2.0 * projector.dim)
        self.seed = int(seed)
        self.pot_max_context = int(pot_max_context)
        self.pot_num_threads = parse_num_threads(pot_num_threads)
        self.flash_allow_tf32 = bool(flash_allow_tf32)
        self.flash_autotune = bool(flash_autotune)
        self.diagnostics_plan_limit = int(diagnostics_plan_limit)
        self.target_queue = TensorQueue(max_size=self.context_size)

    @torch.no_grad()
    def _make_extra_source(
        self,
        n_extra: int,
        *,
        like: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_extra <= 0:
            empty_z = like[:0]
            empty_h = self.projector.project(empty_z)
            return empty_z, empty_h
        generator = torch.Generator(device=like.device)
        generator.manual_seed(self.seed + 8191 * int(step) + 53)
        z_extra = torch.randn(
            (n_extra, *like.shape[1:]),
            generator=generator,
            device=like.device,
            dtype=like.dtype,
        )
        h_extra = self.projector.project(z_extra)
        return z_extra, h_extra

    @torch.no_grad()
    def sample_pairs(self, z0_local: torch.Tensor, z1_local: torch.Tensor, *, step: int = 0) -> PairSample:
        rank, world_size = distributed_rank_world()
        local_batch = z0_local.shape[0]
        device = z0_local.device
        metrics: dict[str, Any] = {
            "mode": self.mode,
            "rank": rank,
            "world_size": world_size,
            "local_batch": int(local_batch),
        }
        h0_local = self.projector.project(z0_local)
        h1_local = self.projector.project(z1_local)

        if self.mode == "independent":
            metrics.update(
                {
                    "context_size": int(local_batch),
                    "sample_cost": float(pair_cost(h0_local, h1_local, self.cost_scale).mean().item()),
                    "duplicate_fraction": 0.0,
                }
            )
            return PairSample(z0_local.detach(), z1_local.detach(), metrics)

        if self.mode == "local_pot_exact_stock":
            from torchcfm.optimal_transport import OTPlanSampler

            t0 = time.perf_counter()
            sampler = OTPlanSampler(method="exact", num_threads=self.pot_num_threads)
            paired_z0, paired_z1 = sampler.sample_plan(z0_local.detach(), z1_local.detach())
            sync_if_cuda(device)
            h0_pair = self.projector.project(paired_z0)
            h1_pair = self.projector.project(paired_z1)
            metrics.update(
                {
                    "context_size": int(local_batch),
                    "sample_cost": float(pair_cost(h0_pair, h1_pair, self.cost_scale).mean().item()),
                    "duplicate_fraction": float("nan"),
                    "sampler_time_s": time.perf_counter() - t0,
                }
            )
            return PairSample(paired_z0.detach(), paired_z1.detach(), metrics)

        if self.mode == "local_pot_exact_row":
            row_ids = torch.arange(local_batch, device=device, dtype=torch.long)
            j, extra = pot_row_conditional_indices(
                h0_local,
                h1_local,
                row_ids,
                cost_scale=self.cost_scale,
                seed=self.seed,
                step=step,
                rank=rank,
                num_threads=self.pot_num_threads,
            )
            paired_z1 = z1_local[j]
            metrics.update(extra)
            metrics.update(
                {
                    "context_size": int(local_batch),
                    "sample_cost": float(pair_cost(h0_local, h1_local[j], self.cost_scale).mean().item()),
                    "duplicate_fraction": duplicate_fraction(j, local_batch),
                }
            )
            return PairSample(z0_local.detach(), paired_z1.detach(), metrics)

        z0_global = all_gather_tensor(z0_local.detach())
        z1_global = all_gather_tensor(z1_local.detach())
        h0_global = all_gather_tensor(h0_local.detach())
        h1_global = all_gather_tensor(h1_local.detach())
        self.target_queue.update(z1_global, h1_global)
        target_z, target_h = self.target_queue.get()
        target_z = target_z[: self.context_size]
        target_h = target_h[: self.context_size]
        m = target_h.shape[0]
        if m == 0:
            raise RuntimeError("global sampler received an empty target context.")

        n_fresh = h0_global.shape[0]
        if n_fresh >= m:
            source_z = z0_global[:m]
            source_h = h0_global[:m]
        else:
            extra_z, extra_h = self._make_extra_source(m - n_fresh, like=z0_local, step=step)
            source_z = torch.cat([z0_global, extra_z], dim=0)
            source_h = torch.cat([h0_global, extra_h], dim=0)

        start = rank * local_batch
        row_ids = torch.arange(start, start + local_batch, device=device, dtype=torch.long)
        if row_ids[-1].item() >= source_h.shape[0]:
            raise RuntimeError("local DDP row slice exceeds source context.")

        if self.mode == "global_pot_exact_small":
            if m > self.pot_max_context:
                raise ValueError(
                    f"context_size={m} exceeds pot_max_context={self.pot_max_context}; "
                    "use a smaller context for global_pot_exact_small."
                )
            j, extra = pot_row_conditional_indices(
                source_h,
                target_h,
                row_ids,
                cost_scale=self.cost_scale,
                seed=self.seed,
                step=step,
                rank=rank,
                num_threads=self.pot_num_threads,
            )
        elif self.mode == "global_dense_sinkhorn":
            j, extra = dense_sinkhorn_row_conditional_indices(
                source_h,
                target_h,
                row_ids,
                eps=self.eps,
                n_iters=self.sinkhorn_iters,
                cost_scale=self.cost_scale,
                seed=self.seed,
                step=step,
                diagnostics_plan_limit=self.diagnostics_plan_limit,
            )
        elif self.mode == "global_flash_sinkhorn":
            j, extra = flash_sinkhorn_row_conditional_indices(
                source_h,
                target_h,
                row_ids,
                eps=self.eps,
                n_iters=self.sinkhorn_iters,
                cost_scale=self.cost_scale,
                seed=self.seed,
                step=step,
                allow_tf32=self.flash_allow_tf32,
                autotune=self.flash_autotune,
            )
        else:
            raise AssertionError(f"unhandled global sampler mode {self.mode}")

        paired_z0 = source_z[row_ids]
        paired_z1 = target_z[j]
        metrics.update(extra)
        metrics.update(
            {
                "context_size": int(m),
                "sample_cost": float(pair_cost(source_h[row_ids], target_h[j], self.cost_scale).mean().item()),
                "duplicate_fraction": duplicate_fraction(j, m),
            }
        )
        return PairSample(paired_z0.detach(), paired_z1.detach(), metrics)
