"""Lightweight result containers for FlashWasserstein.

The annotations intentionally avoid importing torch so the package can be
imported on documentation or CPU-only machines before optional dependencies are
installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FlashWassersteinResult:
    """Result returned by semi-dual Wasserstein solvers."""

    psi: Any
    f: Any
    assignment: Any
    assigned_mass: Any
    mass_error_l1: float
    semidual_value: float
    transport_cost: float
    n_iter: int
    converged: bool
    history: List[Dict[str, float]] = field(default_factory=list)
    backend: str = ""
    cost_scale: float = 0.5


@dataclass
class MongeMapResult:
    """Diagnostics for the deterministic map induced by a dual potential."""

    mapped_y: Any
    barycentric_y: Any
    psi: Any
    f: Any
    assignment: Any
    assigned_mass: Any
    mass_error_l1: float
    semidual_value: float
    transport_cost: float
    backend: str = ""
    cost_scale: float = 0.5


@dataclass
class POTResult:
    """Small-problem exact Kantorovich baseline from Python Optimal Transport."""

    plan: Any
    cost: float
    row_error_l1: float
    col_error_l1: float
    n: int
    m: int
    cost_scale: float = 0.5
    available: bool = True
    message: Optional[str] = None


@dataclass
class OTFMPairingResult:
    """Balanced minibatch OT pairs for OT flow matching."""

    permutation: Any
    pair_i: Any
    pair_j: Any
    paired_x: Any
    paired_y: Any
    prices: Any
    transport_cost: float
    normalized_cost: float
    epsilon: float
    n_rounds: int
    n_bids: int
    converged: bool
    max_epsilon_cs_violation: Optional[float] = None
    backend: str = ""
