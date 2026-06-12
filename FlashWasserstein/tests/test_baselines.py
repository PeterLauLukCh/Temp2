from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ot")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_wasserstein import monge_map, pot_exact_ot


def test_pot_exact_ot_small_problem():
    torch.manual_seed(0)
    x = torch.randn(8, 2)
    y = torch.randn(8, 2)
    pot = pot_exact_ot(x, y, cost_scale=0.5, max_size=16)
    monge = monge_map(x, y, torch.zeros(8), cost_scale=0.5, backend="dense")
    assert pot.row_error_l1 < 1e-5
    assert pot.col_error_l1 < 1e-5
    assert pot.cost >= 0.0
    # The deterministic Monge assignment may violate the target marginal, so its
    # raw cost is not necessarily an admissible upper bound on Kantorovich OT.
    if monge.mass_error_l1 < 1e-6:
        assert pot.cost <= monge.transport_cost + 1e-5
