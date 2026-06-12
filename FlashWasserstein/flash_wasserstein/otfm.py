"""OT flow-matching compatible balanced pair extraction.

This module implements the equal-size uniform minibatch case:

    min_{sigma in S_n} (1/n) sum_i c(x_i, y_{sigma(i)})

and returns a permutation coupling that can directly feed OT Flow Matching.
The Flash backend uses a streaming top-2 reduced-cost oracle; the dense backend
is a correctness/reference path.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from .types import OTFMPairingResult
from .utils import require_torch, validate_points


Top2Oracle = Callable[[object], Tuple[object, object, object, object]]


def _validate_otfm_inputs(x, y):
    n, m, _ = validate_points(x, y)
    if n != m:
        raise ValueError(
            "OT-FM balanced pairing currently requires equal-size minibatches; "
            f"got n={n}, m={m}."
        )
    return n


def _transport_cost_from_permutation(x, y, permutation, cost_scale: float) -> float:
    th = require_torch()
    diff = x.float() - y.float()[permutation.to(dtype=th.long)]
    cost = cost_scale * (diff * diff).sum(dim=1)
    return float(cost.sum().item())


def dense_top2_reduced_cost(x, y, prices, *, cost_scale: float = 0.5):
    """Dense reference top-2 reduced costs for c(x_i, y_j) + prices_j."""

    th = require_torch()
    validate_points(x, y)
    if prices.shape != (y.shape[0],):
        raise ValueError(f"prices must have shape ({y.shape[0]},), got {tuple(prices.shape)}.")
    cost = cost_scale * th.cdist(x.float(), y.float(), p=2).pow(2)
    reduced = cost + prices.float().unsqueeze(0)
    k = 2 if y.shape[0] >= 2 else 1
    vals, idx = th.topk(reduced, k=k, dim=1, largest=False, sorted=True)
    if k == 1:
        inf = th.full_like(vals[:, 0], float("inf"))
        neg = th.full_like(idx[:, 0], -1)
        return vals[:, 0], idx[:, 0].long(), inf, neg.long()
    return vals[:, 0], idx[:, 0].long(), vals[:, 1], idx[:, 1].long()


def flash_top2_reduced_cost(
    x,
    y,
    prices,
    *,
    cost_scale: float = 0.5,
    allow_tf32: bool = True,
    autotune: bool = False,
    **kernel_kwargs,
):
    """Flash top-2 reduced costs for c(x_i, y_j) + prices_j."""

    validate_points(x, y)
    if not x.is_cuda:
        raise ValueError("flash_top2_reduced_cost requires CUDA tensors.")
    if prices.shape != (y.shape[0],):
        raise ValueError(f"prices must have shape ({y.shape[0]},), got {tuple(prices.shape)}.")
    try:
        from flash_sinkhorn import c_transform_top2_fwd
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        src = repo_root / "code" / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from flash_sinkhorn import c_transform_top2_fwd

    # c_transform computes c(x_i, y_j) - psi_j. Setting psi=-prices gives
    # reduced costs c(x_i, y_j) + prices_j.
    return c_transform_top2_fwd(
        x,
        y,
        -prices.float(),
        cost_scale=cost_scale,
        allow_tf32=allow_tf32,
        autotune=autotune,
        **kernel_kwargs,
    )


def flash_auction_bid(
    x,
    y,
    y_norm,
    prices,
    row_ids,
    *,
    epsilon: float,
    cost_scale: float = 0.5,
    allow_tf32: bool = True,
    **kernel_kwargs,
):
    """Fused Flash bid oracle for one epsilon-auction round.

    The kernel streams over all targets, computes top-2 reduced costs, and
    directly returns ``(preferred_target, offer)`` for each requested source
    row.  This avoids materializing ``x[row_ids]`` and avoids returning unused
    top-2 tensors in the hot auction loop.
    """

    validate_points(x, y)
    if not x.is_cuda:
        raise ValueError("flash_auction_bid requires CUDA tensors.")
    if prices.shape != (y.shape[0],):
        raise ValueError(f"prices must have shape ({y.shape[0]},), got {tuple(prices.shape)}.")
    if y_norm.shape != (y.shape[0],):
        raise ValueError(f"y_norm must have shape ({y.shape[0]},), got {tuple(y_norm.shape)}.")
    try:
        from flash_sinkhorn import auction_bid_fwd
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        src = repo_root / "code" / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from flash_sinkhorn import auction_bid_fwd

    return auction_bid_fwd(
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


def flash_auction_accept(
    source_idx,
    target_idx,
    offers,
    owner,
    mate,
    prices,
    best_offer,
    winner_source,
    *,
    block: int = 256,
):
    """Fused Flash accept/update step for one parallel auction round."""

    try:
        from flash_sinkhorn import auction_accept_fwd
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        src = repo_root / "code" / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from flash_sinkhorn import auction_accept_fwd

    return auction_accept_fwd(
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


def _select_parallel_auction_winners(source_idx, target_idx, offers, n_targets: int):
    """Select one winning bidder per target on device."""

    th = require_torch()
    best_offer = th.full(
        (n_targets,),
        -float("inf"),
        device=offers.device,
        dtype=offers.dtype,
    )
    best_offer.scatter_reduce_(0, target_idx, offers, reduce="amax", include_self=True)

    is_best_offer = offers == best_offer[target_idx]
    candidates = th.where(
        is_best_offer,
        source_idx,
        th.full_like(source_idx, -1),
    )
    winner_source = th.full(
        (n_targets,),
        -1,
        device=source_idx.device,
        dtype=source_idx.dtype,
    )
    # Deterministic tie-break among exactly equal offers: largest source index.
    winner_source.scatter_reduce_(0, target_idx, candidates, reduce="amax", include_self=True)
    return source_idx == winner_source[target_idx]


def _epsilon_cs_violation(x, y, prices, permutation, oracle: Top2Oracle, cost_scale: float, epsilon: float):
    th = require_torch()
    best_vals, _, _, _ = oracle(prices)
    assigned = cost_scale * (x.float() - y.float()[permutation]).pow(2).sum(dim=1)
    assigned_reduced = assigned + prices.float()[permutation]
    violation = assigned_reduced - best_vals.float() - float(epsilon)
    return float(th.clamp(violation, min=0).max().item())


def _solve_otfm_auction(
    x,
    y,
    *,
    oracle: Top2Oracle,
    bid_oracle: Optional[Callable[[object, object, float], Tuple[object, object]]] = None,
    accept_oracle: Optional[Callable[[object, object, object, object, object, object], None]] = None,
    cost_scale: float,
    epsilon: float,
    epsilon_schedule,
    max_rounds: Optional[int],
    raise_on_nonconvergence: bool,
    verify: bool,
    backend: str,
) -> OTFMPairingResult:
    th = require_torch()
    n = _validate_otfm_inputs(x, y)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if epsilon_schedule is None:
        schedule = [float(epsilon)]
    else:
        schedule = [float(value) for value in epsilon_schedule]
        if not schedule:
            raise ValueError("epsilon_schedule must be nonempty.")
        if any(value <= 0 for value in schedule):
            raise ValueError("all epsilon_schedule values must be positive.")
        if abs(schedule[-1] - float(epsilon)) > 1e-12:
            schedule.append(float(epsilon))
    if max_rounds is None:
        max_rounds = max(1000, 20 * n)
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")

    device = x.device
    if n == 1:
        permutation = th.zeros((1,), device=device, dtype=th.long)
        prices = th.zeros((1,), device=device, dtype=th.float32)
        cost = _transport_cost_from_permutation(x, y, permutation, cost_scale)
        return OTFMPairingResult(
            permutation=permutation,
            pair_i=permutation.clone(),
            pair_j=permutation.clone(),
            paired_x=x,
            paired_y=y,
            prices=prices,
            transport_cost=cost,
            normalized_cost=cost,
            epsilon=float(epsilon),
            n_rounds=0,
            n_bids=0,
            converged=True,
            max_epsilon_cs_violation=0.0,
            backend=backend,
        )

    prices = th.zeros((n,), device=device, dtype=th.float32)
    n_bids = 0
    converged = False
    rounds_done = 0
    owner = th.full((n,), -1, device=device, dtype=th.long)
    mate = th.full((n,), -1, device=device, dtype=th.long)
    has_full_assignment = False

    with th.no_grad():
        for stage_epsilon in schedule:
            stage_converged = False
            if has_full_assignment:
                # Epsilon scaling refinement: a coarse-stage assignment is a
                # valid warm start for the next stage.  Only rows that violate
                # the tighter epsilon-CS certificate must re-enter the auction.
                best_vals, _, _, _ = oracle(prices)
                assigned_targets = mate.clone()
                assigned = cost_scale * (x.float() - y.float()[assigned_targets]).pow(2).sum(dim=1)
                assigned_reduced = assigned + prices.float()[assigned_targets]
                violation = assigned_reduced - best_vals.float() - float(stage_epsilon)
                needs_rebid = violation > 1e-6
                if needs_rebid.any().item():
                    freed_targets = mate[needs_rebid]
                    owner[freed_targets] = -1
                    mate[needs_rebid] = -1
                else:
                    stage_converged = True
                    converged = True
                    continue
            else:
                owner.fill_(-1)
                mate.fill_(-1)

            for round_idx in range(max_rounds):
                unassigned = th.nonzero(mate < 0, as_tuple=False).flatten()
                if unassigned.numel() == 0:
                    stage_converged = True
                    rounds_done += round_idx
                    break

                if bid_oracle is None:
                    val1, target1, val2, _ = oracle(prices, rows=unassigned)
                    if not th.isfinite(val2).all().item():
                        raise RuntimeError("top-2 oracle returned non-finite second-best values.")

                    increments = (val2.float() - val1.float()).clamp_min(0.0) + float(stage_epsilon)
                    offers = prices[target1] + increments
                else:
                    target1, offers = bid_oracle(prices, unassigned, float(stage_epsilon))

                if accept_oracle is None:
                    accepted = _select_parallel_auction_winners(unassigned, target1, offers, n)
                    if not accepted.any().item():
                        rounds_done += round_idx + 1
                        break

                    accepted_sources = unassigned[accepted]
                    accepted_targets = target1[accepted].long()
                    accepted_offers = offers[accepted].float()

                    previous_sources = owner[accepted_targets].clone()
                    evict = previous_sources >= 0
                    if evict.any().item():
                        mate[previous_sources[evict]] = -1

                    owner[accepted_targets] = accepted_sources
                    mate[accepted_sources] = accepted_targets
                    prices[accepted_targets] = accepted_offers
                else:
                    accept_oracle(unassigned, target1.long(), offers.float(), owner, mate, prices)
                n_bids += int(unassigned.numel())

            if not stage_converged:
                stage_converged = bool((mate >= 0).all().item())
                if stage_converged:
                    rounds_done += max_rounds
            converged = stage_converged
            has_full_assignment = stage_converged
            if not stage_converged:
                break

    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "FlashWasserstein OT-FM auction did not converge to a full permutation "
            f"within max_rounds={max_rounds}."
        )

    if converged:
        unique_targets = th.unique(mate)
        if unique_targets.numel() != n or (mate < 0).any().item():
            raise RuntimeError("Auction internal error: converged without a valid permutation.")

    permutation = mate.detach()
    if converged:
        cost = _transport_cost_from_permutation(x, y, permutation, cost_scale)
        paired_y = y[permutation]
        pair_j = permutation.clone()
    else:
        valid = permutation >= 0
        safe_perm = th.where(valid, permutation, th.zeros_like(permutation))
        cost = _transport_cost_from_permutation(x, y, safe_perm, cost_scale)
        paired_y = y[safe_perm]
        pair_j = permutation.clone()

    max_violation = None
    if verify and converged:
        full_oracle = lambda p, rows=None: oracle(p, rows=rows)
        max_violation = _epsilon_cs_violation(
            x,
            y,
            prices,
            permutation,
            full_oracle,
            cost_scale,
            epsilon,
        )

    pair_i = th.arange(n, device=device, dtype=th.long)
    return OTFMPairingResult(
        permutation=permutation,
        pair_i=pair_i,
        pair_j=pair_j,
        paired_x=x,
        paired_y=paired_y,
        prices=prices.detach(),
        transport_cost=cost,
        normalized_cost=cost / float(n),
        epsilon=float(epsilon),
        n_rounds=int(rounds_done),
        n_bids=int(n_bids),
        converged=bool(converged),
        max_epsilon_cs_violation=max_violation,
        backend=backend,
    )


def solve_dense_otfm_pairs(
    x,
    y,
    *,
    cost_scale: float = 0.5,
    epsilon: float = 1e-3,
    epsilon_schedule=None,
    max_rounds: Optional[int] = None,
    raise_on_nonconvergence: bool = True,
    verify: bool = True,
) -> OTFMPairingResult:
    """Dense reference epsilon-auction OT-FM pairing solver."""

    _validate_otfm_inputs(x, y)

    def oracle(prices, rows=None):
        x_rows = x if rows is None else x[rows]
        return dense_top2_reduced_cost(x_rows, y, prices, cost_scale=cost_scale)

    return _solve_otfm_auction(
        x,
        y,
        oracle=oracle,
        accept_oracle=None,
        cost_scale=cost_scale,
        epsilon=epsilon,
        epsilon_schedule=epsilon_schedule,
        max_rounds=max_rounds,
        raise_on_nonconvergence=raise_on_nonconvergence,
        verify=verify,
        backend="dense_auction",
    )


def solve_flash_otfm_pairs(
    x,
    y,
    *,
    cost_scale: float = 0.5,
    epsilon: float = 1e-3,
    epsilon_schedule=None,
    max_rounds: Optional[int] = None,
    raise_on_nonconvergence: bool = True,
    verify: bool = True,
    allow_tf32: bool = True,
    autotune: bool = False,
    fused_bids: bool = True,
    fused_accept: bool = True,
    **kernel_kwargs,
) -> OTFMPairingResult:
    """Flash epsilon-auction OT-FM pairing solver.

    This is the OT-FM-compatible FlashWasserstein API for the equal-size uniform
    minibatch setting. It returns a balanced permutation, not an independent
    argmin map.
    """

    _validate_otfm_inputs(x, y)
    if not x.is_cuda:
        raise ValueError("solve_flash_otfm_pairs requires CUDA tensors.")
    th = require_torch()

    def oracle(prices, rows=None):
        x_rows = x if rows is None else x[rows]
        return flash_top2_reduced_cost(
            x_rows,
            y,
            prices,
            cost_scale=cost_scale,
            allow_tf32=allow_tf32,
            autotune=autotune,
            **kernel_kwargs,
        )

    bid_oracle = None
    accept_oracle = None
    backend = "flash_auction"
    if fused_bids:
        x_bid = x.contiguous().float()
        y_bid = y.contiguous().float()
        y_norm = (y_bid * y_bid).sum(dim=1).contiguous()

        def bid_oracle(prices, rows, stage_epsilon):
            return flash_auction_bid(
                x_bid,
                y_bid,
                y_norm,
                prices,
                rows,
                epsilon=stage_epsilon,
                cost_scale=cost_scale,
                allow_tf32=allow_tf32,
                **kernel_kwargs,
            )

        backend = "flash_fused_bid_auction"

    if fused_accept:
        best_offer = th.empty((y.shape[0],), device=x.device, dtype=th.float32)
        winner_source = th.empty((y.shape[0],), device=x.device, dtype=th.long)

        def accept_oracle(source_idx, target_idx, offers, owner, mate, prices):
            return flash_auction_accept(
                source_idx,
                target_idx,
                offers,
                owner,
                mate,
                prices,
                best_offer,
                winner_source,
            )

        backend = "flash_fused_auction" if fused_bids else "flash_fused_accept_auction"

    return _solve_otfm_auction(
        x,
        y,
        oracle=oracle,
        bid_oracle=bid_oracle,
        accept_oracle=accept_oracle,
        cost_scale=cost_scale,
        epsilon=epsilon,
        epsilon_schedule=epsilon_schedule,
        max_rounds=max_rounds,
        raise_on_nonconvergence=raise_on_nonconvergence,
        verify=verify,
        backend=backend,
    )
