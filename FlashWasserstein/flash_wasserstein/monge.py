"""Monge-map diagnostics induced by a semi-dual potential."""

from __future__ import annotations

from .dense import dense_c_transform
from .solver import flash_c_transform
from .types import MongeMapResult
from .utils import (
    assigned_mass_from_assignment,
    mass_error_l1,
    prepare_weights,
    require_torch,
    semidual_value,
    transport_cost_from_assignment,
    validate_points,
)


def monge_map(
    x,
    y,
    psi,
    *,
    a=None,
    b=None,
    cost_scale: float = 0.5,
    backend: str = "flash",
    normalize_weights: bool = True,
    allow_tf32: bool = True,
    autotune: bool = True,
    **kernel_kwargs,
) -> MongeMapResult:
    """Evaluate the deterministic map ``x_i -> y_argmin`` induced by ``psi``.

    The returned map is a Monge assignment. For finite empirical measures it may
    not satisfy the target marginal exactly; always inspect ``mass_error_l1``.
    """

    th = require_torch()
    _, m, _ = validate_points(x, y)
    a_t, b_t = prepare_weights(x, y, a, b, normalize=normalize_weights)
    psi_t = th.as_tensor(psi, device=y.device, dtype=th.float32)
    if psi_t.shape != (m,):
        raise ValueError(f"psi must have shape ({m},), got {tuple(psi_t.shape)}.")

    with th.no_grad():
        if backend == "dense":
            f, assignment = dense_c_transform(x, y, psi_t, cost_scale=cost_scale)
        elif backend == "flash":
            f, assignment = flash_c_transform(
                x,
                y,
                psi_t,
                cost_scale=cost_scale,
                allow_tf32=allow_tf32,
                autotune=autotune,
                **kernel_kwargs,
            )
        else:
            raise ValueError('backend must be one of {"flash", "dense"}.')

        assignment = assignment.to(dtype=th.long, device=x.device)
        mapped_y = y[assignment]
        assigned_mass = assigned_mass_from_assignment(assignment, a_t, m)
        semi = semidual_value(f, psi_t, a_t, b_t)
        primal = transport_cost_from_assignment(x, y, assignment, a_t, cost_scale)
        err = mass_error_l1(assigned_mass, b_t)

    return MongeMapResult(
        mapped_y=mapped_y.detach(),
        barycentric_y=mapped_y.detach(),
        psi=psi_t.detach(),
        f=f.detach(),
        assignment=assignment.detach(),
        assigned_mass=assigned_mass.detach(),
        mass_error_l1=err,
        semidual_value=semi,
        transport_cost=primal,
        backend=backend,
        cost_scale=float(cost_scale),
    )
