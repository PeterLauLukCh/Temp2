"""FlashWasserstein: semi-dual hard-OT research prototype."""

from .baselines import pot_exact_ot
from .dense import dense_c_transform, solve_dense_semidual
from .monge import monge_map
from .otfm import (
    dense_top2_reduced_cost,
    flash_auction_accept,
    flash_auction_bid,
    flash_top2_reduced_cost,
    solve_dense_otfm_pairs,
    solve_flash_otfm_pairs,
)
from .solver import flash_c_transform, solve_flash_wasserstein
from .types import FlashWassersteinResult, MongeMapResult, OTFMPairingResult, POTResult

__all__ = [
    "FlashWassersteinResult",
    "MongeMapResult",
    "OTFMPairingResult",
    "POTResult",
    "dense_c_transform",
    "dense_top2_reduced_cost",
    "flash_auction_accept",
    "flash_auction_bid",
    "flash_c_transform",
    "flash_top2_reduced_cost",
    "monge_map",
    "pot_exact_ot",
    "solve_dense_semidual",
    "solve_dense_otfm_pairs",
    "solve_flash_otfm_pairs",
    "solve_flash_wasserstein",
]

__version__ = "0.1.0"
