"""Optional small-problem baselines."""

from __future__ import annotations

from .types import POTResult
from .utils import prepare_weights, require_torch, validate_points


def pot_exact_ot(
    x,
    y,
    *,
    a=None,
    b=None,
    cost_scale: float = 0.5,
    normalize_weights: bool = True,
    max_size: int = 2048,
) -> POTResult:
    """Compute exact discrete OT with POT for small problems.

    POT is CPU/NumPy oriented and materializes the dense cost and plan, so this
    function is intentionally a validation baseline rather than a scalable path.
    """

    th = require_torch()
    n, m, _ = validate_points(x, y)
    if n > max_size or m > max_size:
        raise ValueError(
            f"POT baseline is capped at max_size={max_size}; got n={n}, m={m}."
        )
    try:
        import ot
    except ModuleNotFoundError as exc:
        raise ImportError("POT baseline requires `pip install pot`.") from exc

    a_t, b_t = prepare_weights(x, y, a, b, normalize=normalize_weights)
    x_cpu = x.detach().float().cpu()
    y_cpu = y.detach().float().cpu()
    cost = cost_scale * th.cdist(x_cpu, y_cpu, p=2).pow(2)
    a_np = a_t.detach().cpu().numpy()
    b_np = b_t.detach().cpu().numpy()
    cost_np = cost.numpy()
    plan_np = ot.emd(a_np, b_np, cost_np)
    plan = th.as_tensor(plan_np, dtype=th.float32)
    row_error = float((plan.sum(dim=1) - th.as_tensor(a_np)).abs().sum().item())
    col_error = float((plan.sum(dim=0) - th.as_tensor(b_np)).abs().sum().item())
    ot_cost = float((plan * cost).sum().item())
    return POTResult(
        plan=plan,
        cost=ot_cost,
        row_error_l1=row_error,
        col_error_l1=col_error,
        n=n,
        m=m,
        cost_scale=float(cost_scale),
    )
