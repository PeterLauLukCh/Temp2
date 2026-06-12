from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_wasserstein import dense_c_transform, monge_map, solve_dense_semidual
from flash_wasserstein.utils import assigned_mass_from_assignment, prepare_weights


def test_dense_ctransform_gauge_invariance():
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    y = torch.randn(11, 3)
    psi = torch.randn(11)
    c0, idx0 = dense_c_transform(x, y, psi, cost_scale=0.5)
    c1, idx1 = dense_c_transform(x, y, psi + 7.0, cost_scale=0.5)
    torch.testing.assert_close(c1, c0 - 7.0)
    assert torch.equal(idx0, idx1)


def test_supergradient_is_mass_residual():
    torch.manual_seed(1)
    x = torch.randn(20, 2)
    y = torch.randn(9, 2)
    psi = torch.randn(9)
    a = torch.softmax(torch.randn(20), dim=0)
    b = torch.softmax(torch.randn(9), dim=0)
    _, assignment = dense_c_transform(x, y, psi, cost_scale=1.0)
    assigned = assigned_mass_from_assignment(assignment, a, y.shape[0])
    residual = b - assigned
    result = monge_map(x, y, psi, a=a, b=b, cost_scale=1.0, backend="dense")
    torch.testing.assert_close(result.assigned_mass, assigned)
    torch.testing.assert_close(residual, b - result.assigned_mass)


def test_weighted_marginals_and_result_fields():
    torch.manual_seed(2)
    x = torch.randn(24, 4)
    y = torch.randn(7, 4)
    a = torch.softmax(torch.randn(24), dim=0)
    b = torch.softmax(torch.randn(7), dim=0)
    result = solve_dense_semidual(
        x,
        y,
        a=a,
        b=b,
        cost_scale=0.5,
        max_iter=3,
        lr=0.1,
        return_history=True,
    )
    assert result.psi.shape == (7,)
    assert result.f.shape == (24,)
    assert result.assignment.shape == (24,)
    assert result.assigned_mass.shape == (7,)
    assert isinstance(result.mass_error_l1, float)
    assert result.history


def test_tie_breaking_smallest_index_dense():
    x = torch.zeros(1, 2)
    y = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [10.0, 0.0]])
    psi = torch.zeros(3)
    _, assignment = dense_c_transform(x, y, psi, cost_scale=1.0)
    assert assignment.item() == 0


def test_dense_solver_can_reduce_biased_initial_mass_error():
    x = torch.tensor([[0.0], [10.0]])
    y = torch.tensor([[0.0], [10.0]])
    psi_init = torch.tensor([100.0, 0.0])
    a, b = prepare_weights(x, y)
    _, initial_assignment = dense_c_transform(x, y, psi_init, cost_scale=0.5)
    initial_mass = assigned_mass_from_assignment(initial_assignment, a, y.shape[0])
    initial_error = (initial_mass - b).abs().sum().item()

    result = solve_dense_semidual(
        x,
        y,
        psi_init=psi_init,
        cost_scale=0.5,
        max_iter=50,
        lr=10.0,
        tol=1e-6,
        return_history=False,
    )
    assert result.mass_error_l1 <= initial_error


def test_monge_map_reports_mass_residual():
    x = torch.randn(5, 2)
    y = torch.randn(3, 2)
    psi = torch.zeros(3)
    result = monge_map(x, y, psi, backend="dense")
    assert result.mapped_y.shape == x.shape
    assert result.barycentric_y.shape == x.shape
    assert result.mass_error_l1 >= 0.0
