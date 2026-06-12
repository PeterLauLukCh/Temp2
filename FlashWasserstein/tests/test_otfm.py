from itertools import permutations

import pytest

torch = pytest.importorskip("torch")

from flash_wasserstein import (
    dense_top2_reduced_cost,
    solve_dense_otfm_pairs,
)


def exact_assignment_cost(x, y, cost_scale=0.5):
    n = x.shape[0]
    best = float("inf")
    best_perm = None
    cost = cost_scale * torch.cdist(x.float(), y.float(), p=2).pow(2)
    for perm in permutations(range(n)):
        idx = torch.tensor(perm, dtype=torch.long)
        value = float(cost[torch.arange(n), idx].sum().item())
        if value < best:
            best = value
            best_perm = idx
    return best, best_perm


def test_dense_top2_reduced_cost_matches_torch_topk():
    torch.manual_seed(0)
    x = torch.randn(7, 3)
    y = torch.randn(5, 3)
    prices = torch.randn(5)
    values1, idx1, values2, idx2 = dense_top2_reduced_cost(x, y, prices)

    reduced = 0.5 * torch.cdist(x, y).pow(2) + prices.unsqueeze(0)
    vals, idx = torch.topk(reduced, k=2, dim=1, largest=False, sorted=True)
    assert torch.allclose(values1, vals[:, 0])
    assert torch.equal(idx1, idx[:, 0])
    assert torch.allclose(values2, vals[:, 1])
    assert torch.equal(idx2, idx[:, 1])


def test_dense_otfm_pairs_returns_balanced_permutation_and_cost_bound():
    torch.manual_seed(1)
    n = 6
    x = torch.randn(n, 2)
    y = torch.randn(n, 2) + torch.tensor([1.0, -0.5])
    epsilon = 1e-3
    result = solve_dense_otfm_pairs(
        x,
        y,
        epsilon=epsilon,
        max_rounds=10000,
        verify=True,
    )
    assert result.converged
    assert result.permutation.shape == (n,)
    assert torch.unique(result.permutation).numel() == n
    assert result.max_epsilon_cs_violation <= 1e-5

    exact_cost, _ = exact_assignment_cost(x, y)
    assert result.transport_cost <= exact_cost + n * epsilon + 1e-4
    assert result.normalized_cost <= exact_cost / n + epsilon + 1e-4


def test_dense_otfm_rejects_unequal_minibatches():
    x = torch.randn(4, 2)
    y = torch.randn(5, 2)
    with pytest.raises(ValueError, match="equal-size minibatches"):
        solve_dense_otfm_pairs(x, y)


def test_dense_otfm_epsilon_schedule_path():
    torch.manual_seed(2)
    n = 8
    x = torch.randn(n, 2)
    y = torch.randn(n, 2) + 0.5
    result = solve_dense_otfm_pairs(
        x,
        y,
        epsilon=1e-2,
        epsilon_schedule=[0.5, 0.1, 0.01],
        max_rounds=10000,
        verify=True,
    )
    assert result.converged
    assert torch.unique(result.permutation).numel() == n
    assert result.epsilon == pytest.approx(1e-2)
