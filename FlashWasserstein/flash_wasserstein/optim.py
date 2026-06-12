"""Shared semi-dual optimization loop."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from .types import FlashWassersteinResult
from .utils import (
    apply_gauge,
    assigned_mass_from_assignment,
    mass_error_l1,
    prepare_psi,
    prepare_weights,
    require_torch,
    semidual_value,
    step_size,
    transport_cost_from_assignment,
    validate_points,
)


Oracle = Callable[[object], Tuple[object, object]]


def run_semidual_subgradient(
    x,
    y,
    *,
    oracle: Oracle,
    a=None,
    b=None,
    psi_init=None,
    cost_scale: float = 0.5,
    max_iter: int = 200,
    lr: float = 1.0,
    tol: float = 1e-4,
    gauge: str = "weighted_mean_zero",
    lr_schedule: str = "sqrt_decay",
    record_every: int = 1,
    return_history: bool = True,
    normalize_weights: bool = True,
    backend: str = "",
) -> FlashWassersteinResult:
    """Maximize the hard semi-dual using target-mass residual ascent."""

    th = require_torch()
    _, m, _ = validate_points(x, y)
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative.")
    if record_every <= 0:
        raise ValueError("record_every must be positive.")

    a_t, b_t = prepare_weights(x, y, a, b, normalize=normalize_weights)
    psi = apply_gauge(prepare_psi(y, psi_init), b_t, gauge)
    history: List[Dict[str, float]] = []

    def evaluate(current_psi):
        c_values, assignment = oracle(current_psi)
        assignment = assignment.to(device=x.device, dtype=th.long)
        assigned_mass = assigned_mass_from_assignment(assignment, a_t, m)
        residual = b_t - assigned_mass
        err = mass_error_l1(assigned_mass, b_t)
        semi = semidual_value(c_values, current_psi, a_t, b_t)
        primal = transport_cost_from_assignment(x, y, assignment, a_t, cost_scale)
        return c_values, assignment, assigned_mass, residual, err, semi, primal

    last = None
    converged = False
    n_iter = 0
    with th.no_grad():
        for step in range(max_iter):
            last = evaluate(psi)
            _, _, _, residual, err, semi, primal = last
            if return_history and step % record_every == 0:
                history.append(
                    {
                        "step": float(step),
                        "lr": 0.0,
                        "mass_error_l1": err,
                        "semidual_value": semi,
                        "transport_cost": primal,
                    }
                )
            if err <= tol:
                converged = True
                n_iter = step
                break
            lr_t = step_size(lr, step, lr_schedule)
            if return_history and history and history[-1]["step"] == float(step):
                history[-1]["lr"] = lr_t
            psi = apply_gauge(psi + lr_t * residual, b_t, gauge)
            n_iter = step + 1

        if last is None or not converged:
            last = evaluate(psi)
            c_values, assignment, assigned_mass, _, err, semi, primal = last
            converged = err <= tol
            if return_history:
                if not history or history[-1]["step"] != float(n_iter):
                    history.append(
                        {
                            "step": float(n_iter),
                            "lr": 0.0,
                            "mass_error_l1": err,
                            "semidual_value": semi,
                            "transport_cost": primal,
                        }
                    )
        else:
            c_values, assignment, assigned_mass, _, err, semi, primal = last

    return FlashWassersteinResult(
        psi=psi.detach(),
        f=c_values.detach(),
        assignment=assignment.detach(),
        assigned_mass=assigned_mass.detach(),
        mass_error_l1=err,
        semidual_value=semi,
        transport_cost=primal,
        n_iter=int(n_iter),
        converged=bool(converged),
        history=history,
        backend=backend,
        cost_scale=float(cost_scale),
    )
