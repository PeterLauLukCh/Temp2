"""Auction bid kernels for squared Euclidean assignment.

This module provides the fused per-round primitive needed by the balanced
epsilon-auction solver used for OT Flow Matching minibatches.  For each
currently unassigned source row ``i``, it streams over all targets ``j`` and
computes the two smallest reduced costs

    cost_scale * ||x_i - y_j||^2 + price_j.

The kernel writes only the preferred target and the corresponding auction
offer

    price_j + (second_best - best) + epsilon,

so the Python solver does not materialize sliced source batches or the full
top-2 value/index tuple every auction round.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flash_sinkhorn.kernels._common import _cache_key_bucket, _validate_device
from flash_sinkhorn.kernels.c_transform_sqeuclid import _default_block_sizes


@triton.heuristics({
    "EVEN_N": lambda args: args["n_targets"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _auction_bid_kernel_impl(
    x_ptr,              # Source coordinates: [n, d]
    y_ptr,              # Target coordinates: [n_targets, d]
    y_norm_ptr,         # Precomputed ||y_j||^2: [n_targets]
    prices_ptr,         # Auction prices: [n_targets]
    row_ids_ptr,        # Source row ids to process: [n_rows]
    out_target_ptr,     # Preferred target per processed row: [n_rows]
    out_offer_ptr,      # Offer per processed row: [n_rows]
    n_rows,
    n_targets,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_y_norm,
    stride_prices,
    stride_row_ids,
    stride_out_target,
    stride_out_offer,
    cost_scale,
    coord_scale,
    epsilon,
    CACHE_KEY_N,
    CACHE_KEY_M,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Emit one epsilon-auction bid for each requested source row."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_rows
    row_ids = tl.load(
        row_ids_ptr + offs_m * stride_row_ids,
        mask=mask_m,
        other=0,
        eviction_policy="evict_first",
    )

    best_val = tl.full([BLOCK_M], float("inf"), tl.float32)
    second_val = tl.full([BLOCK_M], float("inf"), tl.float32)
    best_idx = tl.full([BLOCK_M], -1, tl.int32)
    second_idx = tl.full([BLOCK_M], -1, tl.int32)

    for j0 in range(0, n_targets, BLOCK_N):
        j0 = tl.multiple_of(j0, BLOCK_N)
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_targets

        if EVEN_N:
            y_norm = tl.load(
                y_norm_ptr + offs_n * stride_y_norm,
                eviction_policy="evict_first",
            ).to(tl.float32)
            prices = tl.load(
                prices_ptr + offs_n * stride_prices,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            y_norm = tl.load(
                y_norm_ptr + offs_n * stride_y_norm,
                mask=mask_n,
                other=float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            prices = tl.load(
                prices_ptr + offs_n * stride_prices,
                mask=mask_n,
                other=float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)

        bias = cost_scale * y_norm + prices

        dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k0 = tl.multiple_of(k0, BLOCK_K)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < D
            x_block = tl.load(
                x_ptr + row_ids[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            y_block = tl.load(
                y_ptr + offs_n[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            dot += tl.dot(x_block, y_block, allow_tf32=ALLOW_TF32)

        vals = -coord_scale * dot + bias[None, :]
        vals = tl.where(mask_n[None, :], vals, float("inf"))

        tile_best_val = tl.min(vals, axis=1)
        tile_best_idx = tl.argmin(vals, axis=1).to(tl.int32) + j0

        vals_without_best = tl.where(
            offs_n[None, :] == tile_best_idx[:, None],
            float("inf"),
            vals,
        )
        tile_second_val = tl.min(vals_without_best, axis=1)
        tile_second_idx = tl.argmin(vals_without_best, axis=1).to(tl.int32) + j0
        tile_second_idx = tl.where(tile_second_val == float("inf"), -1, tile_second_idx)

        tile_wins_best = tile_best_val < best_val

        candidate_second_val_if_win = tl.minimum(best_val, tile_second_val)
        old_best_is_second = best_val <= tile_second_val
        candidate_second_idx_if_win = tl.where(old_best_is_second, best_idx, tile_second_idx)

        candidate_second_val_if_lose = tl.minimum(second_val, tile_best_val)
        tile_best_is_second = tile_best_val < second_val
        candidate_second_idx_if_lose = tl.where(tile_best_is_second, tile_best_idx, second_idx)

        second_val = tl.where(
            tile_wins_best,
            candidate_second_val_if_win,
            candidate_second_val_if_lose,
        )
        second_idx = tl.where(
            tile_wins_best,
            candidate_second_idx_if_win,
            candidate_second_idx_if_lose,
        )
        best_val = tl.where(tile_wins_best, tile_best_val, best_val)
        best_idx = tl.where(tile_wins_best, tile_best_idx, best_idx)

    best_price = tl.load(
        prices_ptr + best_idx * stride_prices,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    gap = tl.maximum(second_val - best_val, 0.0)
    offer = best_price + gap + epsilon

    tl.store(out_target_ptr + offs_m * stride_out_target, best_idx, mask=mask_m)
    tl.store(out_offer_ptr + offs_m * stride_out_offer, offer, mask=mask_m)


def auction_bid_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    y_norm: torch.Tensor,
    prices: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float = 1.0,
    allow_tf32: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute fused auction bids for a subset of source rows.

    Args:
        x: Source coordinates ``[n, d]``, CUDA.
        y: Target coordinates ``[m, d]``, CUDA.
        y_norm: Precomputed squared target norms ``[m]``.
        prices: Current auction prices ``[m]``.
        row_ids: Source row indices to process ``[r]``.
        epsilon: Positive auction epsilon for this stage.
        cost_scale: Cost scaling.

    Returns:
        target: Preferred target index for each row id, int64 ``[r]``.
        offer: Auction offer for each row id, float32 ``[r]``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if y_norm.ndim != 1 or prices.ndim != 1 or row_ids.ndim != 1:
        raise ValueError("y_norm, prices, and row_ids must be 1D tensors")

    n, d = x.shape
    m, d2 = y.shape
    if d != d2:
        raise ValueError("x and y must have same feature dimension")
    if y_norm.shape[0] != m:
        raise ValueError(f"y_norm must have length {m}, got {y_norm.shape[0]}")
    if prices.shape[0] != m:
        raise ValueError(f"prices must have length {m}, got {prices.shape[0]}")
    if m < 2:
        raise ValueError("auction_bid_kernel requires at least two targets")

    _validate_device(x, [("y", y), ("y_norm", y_norm), ("prices", prices), ("row_ids", row_ids)])

    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    x = x.contiguous().float()
    y = y.contiguous().float()
    y_norm = y_norm.contiguous().float()
    prices = prices.contiguous().float()
    row_ids = row_ids.contiguous().to(dtype=torch.long)

    r = row_ids.shape[0]
    out_target = torch.empty((r,), device=x.device, dtype=torch.long)
    out_offer = torch.empty((r,), device=x.device, dtype=torch.float32)
    if r == 0:
        return out_target, out_offer

    bm, bn, bk, nw = _default_block_sizes(r, m, d)
    bm = block_m if block_m is not None else bm
    bn = block_n if block_n is not None else bn
    bk = block_k if block_k is not None else bk
    nw = num_warps if num_warps is not None else nw
    if bk < 16:
        bk = 16

    grid = (triton.cdiv(r, bm),)
    _auction_bid_kernel_impl[grid](
        x,
        y,
        y_norm,
        prices,
        row_ids,
        out_target,
        out_offer,
        r,
        m,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        y_norm.stride(0),
        prices.stride(0),
        row_ids.stride(0),
        out_target.stride(0),
        out_offer.stride(0),
        float(cost_scale),
        float(2.0 * cost_scale),
        float(epsilon),
        _cache_key_bucket(n),
        _cache_key_bucket(m),
        D=d,
        ALLOW_TF32=allow_tf32,
        DTYPE_ID=dtype_id,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=nw,
        num_stages=num_stages,
    )
    return out_target, out_offer


@triton.jit
def _auction_reset_kernel(
    best_offer_ptr,
    winner_source_ptr,
    n_targets,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_targets
    tl.store(best_offer_ptr + offs, tl.full([BLOCK], -float("inf"), tl.float32), mask=mask)
    tl.store(winner_source_ptr + offs, tl.full([BLOCK], -1, tl.int64), mask=mask)


@triton.jit
def _auction_best_offer_kernel(
    target_ptr,
    offer_ptr,
    best_offer_ptr,
    n_bidders,
    stride_target,
    stride_offer,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_bidders
    targets = tl.load(target_ptr + offs * stride_target, mask=mask, other=0)
    offers = tl.load(offer_ptr + offs * stride_offer, mask=mask, other=-float("inf")).to(tl.float32)
    tl.atomic_max(best_offer_ptr + targets, offers, sem="relaxed", mask=mask)


@triton.jit
def _auction_winner_source_kernel(
    source_ptr,
    target_ptr,
    offer_ptr,
    best_offer_ptr,
    winner_source_ptr,
    n_bidders,
    stride_source,
    stride_target,
    stride_offer,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_bidders
    sources = tl.load(source_ptr + offs * stride_source, mask=mask, other=-1)
    targets = tl.load(target_ptr + offs * stride_target, mask=mask, other=0)
    offers = tl.load(offer_ptr + offs * stride_offer, mask=mask, other=-float("inf")).to(tl.float32)
    best = tl.load(best_offer_ptr + targets, mask=mask, other=float("inf")).to(tl.float32)
    candidates = tl.where(offers == best, sources, -1)
    tl.atomic_max(winner_source_ptr + targets, candidates, sem="relaxed", mask=mask)


@triton.jit
def _auction_evict_kernel(
    owner_ptr,
    mate_ptr,
    winner_source_ptr,
    n_targets,
    BLOCK: tl.constexpr,
):
    targets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = targets < n_targets
    winners = tl.load(winner_source_ptr + targets, mask=mask, other=-1)
    previous = tl.load(owner_ptr + targets, mask=mask, other=-1)
    evict = mask & (winners >= 0) & (previous >= 0) & (previous != winners)
    tl.store(mate_ptr + previous, tl.full([BLOCK], -1, tl.int64), mask=evict)


@triton.jit
def _auction_assign_kernel(
    owner_ptr,
    mate_ptr,
    prices_ptr,
    best_offer_ptr,
    winner_source_ptr,
    n_targets,
    BLOCK: tl.constexpr,
):
    targets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = targets < n_targets
    winners = tl.load(winner_source_ptr + targets, mask=mask, other=-1)
    offers = tl.load(best_offer_ptr + targets, mask=mask, other=0.0).to(tl.float32)
    assign = mask & (winners >= 0)
    tl.store(owner_ptr + targets, winners, mask=assign)
    tl.store(mate_ptr + winners, targets, mask=assign)
    tl.store(prices_ptr + targets, offers, mask=assign)


def auction_accept_kernel(
    source_idx: torch.Tensor,
    target_idx: torch.Tensor,
    offers: torch.Tensor,
    owner: torch.Tensor,
    mate: torch.Tensor,
    prices: torch.Tensor,
    best_offer: torch.Tensor,
    winner_source: torch.Tensor,
    *,
    block: int = 256,
) -> None:
    """Accept one parallel auction bid round in-place.

    The tie-breaking rule matches the PyTorch reference path: highest offer wins
    each target, and exactly equal offers are broken by largest source index.
    """
    if source_idx.ndim != 1 or target_idx.ndim != 1 or offers.ndim != 1:
        raise ValueError("source_idx, target_idx, and offers must be 1D tensors")
    if source_idx.shape != target_idx.shape or source_idx.shape != offers.shape:
        raise ValueError("source_idx, target_idx, and offers must have matching shapes")
    n_bidders = source_idx.shape[0]
    n_targets = owner.shape[0]
    if mate.shape != owner.shape or prices.shape != owner.shape:
        raise ValueError("owner, mate, and prices must have matching target-size shapes")
    if best_offer.shape != owner.shape or winner_source.shape != owner.shape:
        raise ValueError("best_offer and winner_source must have target-size shapes")
    if n_bidders == 0:
        return
    _validate_device(
        owner,
        [
            ("source_idx", source_idx),
            ("target_idx", target_idx),
            ("offers", offers),
            ("mate", mate),
            ("prices", prices),
            ("best_offer", best_offer),
            ("winner_source", winner_source),
        ],
    )

    grid_targets = (triton.cdiv(n_targets, block),)
    grid_bidders = (triton.cdiv(n_bidders, block),)
    _auction_reset_kernel[grid_targets](best_offer, winner_source, n_targets, BLOCK=block)
    _auction_best_offer_kernel[grid_bidders](
        target_idx,
        offers,
        best_offer,
        n_bidders,
        target_idx.stride(0),
        offers.stride(0),
        BLOCK=block,
    )
    _auction_winner_source_kernel[grid_bidders](
        source_idx,
        target_idx,
        offers,
        best_offer,
        winner_source,
        n_bidders,
        source_idx.stride(0),
        target_idx.stride(0),
        offers.stride(0),
        BLOCK=block,
    )
    _auction_evict_kernel[grid_targets](owner, mate, winner_source, n_targets, BLOCK=block)
    _auction_assign_kernel[grid_targets](
        owner,
        mate,
        prices,
        best_offer,
        winner_source,
        n_targets,
        BLOCK=block,
    )
