from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT / "code" / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from flash_wasserstein import (
    dense_c_transform,
    dense_top2_reduced_cost,
    flash_auction_accept,
    flash_auction_bid,
    flash_c_transform,
    flash_top2_reduced_cost,
    solve_dense_semidual,
    solve_flash_otfm_pairs,
    solve_flash_wasserstein,
)


def _flash_available():
    try:
        import flash_sinkhorn  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _flash_available(), reason="flash_sinkhorn package unavailable")
def test_flash_ctransform_matches_dense():
    torch.manual_seed(0)
    x = torch.randn(64, 8, device="cuda")
    y = torch.randn(48, 8, device="cuda")
    psi = torch.randn(48, device="cuda")
    c_flash, idx_flash = flash_c_transform(
        x, y, psi, cost_scale=0.5, allow_tf32=False, autotune=False
    )
    c_dense, idx_dense = dense_c_transform(x, y, psi, cost_scale=0.5)
    torch.testing.assert_close(c_flash, c_dense, atol=1e-4, rtol=1e-5)
    assert torch.equal(idx_flash, idx_dense)


@pytest.mark.skipif(not _flash_available(), reason="flash_sinkhorn package unavailable")
def test_flash_solver_matches_dense_small():
    torch.manual_seed(1)
    x = torch.randn(32, 4, device="cuda")
    y = torch.randn(32, 4, device="cuda")
    kwargs = dict(cost_scale=0.5, max_iter=5, lr=0.5, return_history=False)
    dense = solve_dense_semidual(x, y, **kwargs)
    flash = solve_flash_wasserstein(x, y, allow_tf32=False, autotune=False, **kwargs)
    torch.testing.assert_close(flash.psi, dense.psi, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(flash.assigned_mass, dense.assigned_mass, atol=1e-6, rtol=1e-6)
    assert flash.assignment.shape == dense.assignment.shape


@pytest.mark.skipif(not _flash_available(), reason="flash_sinkhorn package unavailable")
def test_flash_top2_matches_dense_top2():
    torch.manual_seed(3)
    x = torch.randn(17, 4, device="cuda")
    y = torch.randn(19, 4, device="cuda")
    prices = torch.randn(19, device="cuda")
    dense = dense_top2_reduced_cost(x, y, prices, cost_scale=0.5)
    flash = flash_top2_reduced_cost(
        x,
        y,
        prices,
        cost_scale=0.5,
        allow_tf32=False,
        autotune=False,
    )
    torch.testing.assert_close(flash[0], dense[0], atol=1e-5, rtol=1e-5)
    assert torch.equal(flash[1], dense[1])
    torch.testing.assert_close(flash[2], dense[2], atol=1e-5, rtol=1e-5)
    assert torch.equal(flash[3], dense[3])


@pytest.mark.skipif(not _flash_available(), reason="flash_sinkhorn package unavailable")
def test_flash_auction_bid_matches_dense_top2_offer():
    torch.manual_seed(5)
    x = torch.randn(23, 5, device="cuda")
    y = torch.randn(23, 5, device="cuda")
    prices = torch.randn(23, device="cuda").abs()
    rows = torch.tensor([0, 3, 4, 8, 12, 22], device="cuda")
    epsilon = 1e-2

    target, offer = flash_auction_bid(
        x,
        y,
        (y.float() * y.float()).sum(dim=1),
        prices,
        rows,
        epsilon=epsilon,
        cost_scale=0.5,
        allow_tf32=False,
    )
    val1, idx1, val2, _ = dense_top2_reduced_cost(x[rows], y, prices, cost_scale=0.5)
    expected_offer = prices[idx1] + (val2 - val1).clamp_min(0.0) + epsilon

    assert torch.equal(target, idx1)
    torch.testing.assert_close(offer, expected_offer, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _flash_available(), reason="flash_sinkhorn package unavailable")
def test_flash_auction_accept_matches_reference_round():
    source = torch.tensor([1, 2, 3], device="cuda")
    target = torch.tensor([0, 0, 2], device="cuda")
    offers = torch.tensor([2.0, 3.0, 1.5], device="cuda")
    owner = torch.tensor([0, -1, -1, -1], device="cuda")
    mate = torch.tensor([0, -1, -1, -1], device="cuda")
    prices = torch.zeros(4, device="cuda")
    best_offer = torch.empty(4, device="cuda")
    winner_source = torch.empty(4, device="cuda", dtype=torch.long)

    flash_auction_accept(source, target, offers, owner, mate, prices, best_offer, winner_source)

    expected_owner = torch.tensor([2, -1, 3, -1], device="cuda")
    expected_mate = torch.tensor([-1, -1, 0, 2], device="cuda")
    expected_prices = torch.tensor([3.0, 0.0, 1.5, 0.0], device="cuda")
    assert torch.equal(owner, expected_owner)
    assert torch.equal(mate, expected_mate)
    torch.testing.assert_close(prices, expected_prices)


@pytest.mark.skipif(not _flash_available(), reason="flash_sinkhorn package unavailable")
def test_flash_otfm_pairs_returns_permutation():
    torch.manual_seed(4)
    n = 16
    x = torch.randn(n, 2, device="cuda")
    y = torch.randn(n, 2, device="cuda") + torch.tensor([1.0, -0.5], device="cuda")
    result = solve_flash_otfm_pairs(
        x,
        y,
        epsilon=1e-2,
        max_rounds=10000,
        verify=True,
        allow_tf32=False,
        autotune=False,
    )
    assert result.converged
    assert torch.unique(result.permutation).numel() == n
    assert result.max_epsilon_cs_violation <= 1e-5
