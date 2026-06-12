"""Fused primitives for epsilon-auction assignment."""

from __future__ import annotations

import torch

from flash_sinkhorn.kernels.auction_sqeuclid import auction_accept_kernel, auction_bid_kernel


def auction_bid_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    y_norm: torch.Tensor,
    prices: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float = 1.0,
    allow_tf32: bool = True,
    **kernel_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one fused epsilon-auction bid per requested source row.

    For each ``row_ids[k] = i``, this returns the best target ``j`` under
    reduced cost ``cost_scale * ||x_i - y_j||^2 + prices_j`` and the auction
    offer ``prices_j + second_best - best + epsilon``.
    """
    return auction_bid_kernel(
        x,
        y,
        y_norm,
        prices,
        row_ids,
        epsilon=epsilon,
        cost_scale=cost_scale,
        allow_tf32=allow_tf32,
        **kernel_kwargs,
    )


def auction_accept_fwd(
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
    """Accept one parallel epsilon-auction bid round in-place."""
    auction_accept_kernel(
        source_idx,
        target_idx,
        offers,
        owner,
        mate,
        prices,
        best_offer,
        winner_source,
        block=block,
    )
