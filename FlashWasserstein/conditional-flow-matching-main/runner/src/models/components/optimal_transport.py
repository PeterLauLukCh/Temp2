import math
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import ot as pot
import torch


_FLASH_METHODS = {"flash", "flash_auction", "flashwasserstein", "flash_wasserstein"}


def _is_flash_method(method: str) -> bool:
    return method.lower() in _FLASH_METHODS


def _import_flash_wasserstein():
    try:
        from flash_wasserstein import solve_dense_otfm_pairs, solve_flash_otfm_pairs

        return solve_dense_otfm_pairs, solve_flash_otfm_pairs
    except ModuleNotFoundError:
        flash_wasserstein_root = Path(__file__).resolve().parents[5]
        if flash_wasserstein_root.exists() and str(flash_wasserstein_root) not in sys.path:
            sys.path.insert(0, str(flash_wasserstein_root))
        from flash_wasserstein import solve_dense_otfm_pairs, solve_flash_otfm_pairs

        return solve_dense_otfm_pairs, solve_flash_otfm_pairs


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an squared L2 OT plan with
    different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost=False,
        flash_epsilon: float = 1e-2,
        flash_epsilon_schedule=None,
        flash_max_rounds: Optional[int] = None,
        flash_backend: str = "auto",
        flash_cost_scale: float = 1.0,
        flash_verify: bool = False,
        flash_raise_on_nonconvergence: bool = True,
        flash_fused_bids: bool = True,
        flash_fused_accept: bool = True,
        flash_allow_tf32: bool = True,
        **kwargs,
    ):
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        elif _is_flash_method(method):
            self.ot_fn = None
        else:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.kwargs = kwargs
        self.flash_epsilon = flash_epsilon
        self.flash_epsilon_schedule = (
            [0.5, 0.2, 0.1, 0.05, 0.01]
            if flash_epsilon_schedule is None
            else flash_epsilon_schedule
        )
        self.flash_max_rounds = flash_max_rounds
        self.flash_backend = flash_backend
        self.flash_cost_scale = flash_cost_scale
        self.flash_verify = flash_verify
        self.flash_raise_on_nonconvergence = flash_raise_on_nonconvergence
        self.flash_fused_bids = flash_fused_bids
        self.flash_fused_accept = flash_fused_accept
        self.flash_allow_tf32 = flash_allow_tf32

    @staticmethod
    def _flatten_for_ot(x):
        if x.dim() > 2:
            return x.reshape(x.shape[0], -1)
        return x

    def _solve_flash_pairing(self, x0, x1):
        if x0.shape[0] != x1.shape[0]:
            raise ValueError(
                "FlashWasserstein OT-FM currently requires equal-size minibatches; "
                f"got {x0.shape[0]} and {x1.shape[0]}."
            )
        if self.normalize_cost:
            raise ValueError(
                "normalize_cost=True would require materializing the dense cost matrix; "
                "disable it when using method='flash'."
            )
        if self.flash_backend not in {"auto", "cuda", "dense"}:
            raise ValueError("flash_backend must be one of {'auto', 'cuda', 'dense'}.")

        solve_dense_otfm_pairs, solve_flash_otfm_pairs = _import_flash_wasserstein()
        x0_flat = self._flatten_for_ot(x0.detach())
        x1_flat = self._flatten_for_ot(x1.detach())
        solver_kwargs = dict(
            cost_scale=self.flash_cost_scale,
            epsilon=self.flash_epsilon,
            epsilon_schedule=self.flash_epsilon_schedule,
            max_rounds=self.flash_max_rounds,
            raise_on_nonconvergence=self.flash_raise_on_nonconvergence,
            verify=self.flash_verify,
        )
        if self.flash_backend in {"auto", "cuda"} and x0_flat.is_cuda and x1_flat.is_cuda:
            return solve_flash_otfm_pairs(
                x0_flat,
                x1_flat,
                allow_tf32=self.flash_allow_tf32,
                fused_bids=self.flash_fused_bids,
                fused_accept=self.flash_fused_accept,
                **solver_kwargs,
            )
        if self.flash_backend == "cuda":
            raise ValueError("flash_backend='cuda' requires CUDA minibatches.")
        return solve_dense_otfm_pairs(x0_flat, x1_flat, **solver_kwargs)

    def get_map(self, x0, x1):
        if _is_flash_method(self.method):
            result = self._solve_flash_pairing(x0, x1)
            j = result.permutation.detach().cpu().numpy()
            p = np.zeros((x0.shape[0], x1.shape[0]), dtype=np.float64)
            p[np.arange(x0.shape[0]), j] = 1.0 / x0.shape[0]
            return p

        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        x0 = self._flatten_for_ot(x0)
        x1 = self._flatten_for_ot(x1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        return p

    def sample_map(self, pi, batch_size):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1):
        if _is_flash_method(self.method):
            result = self._solve_flash_pairing(x0, x1)
            permutation = result.permutation.to(device=x0.device, dtype=torch.long)
            i = torch.randint(x0.shape[0], (x0.shape[0],), device=x0.device)
            j = permutation[i]
            return x0[i], x1[j]

        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0])
        return x0[i], x1[j]

    def sample_trajectory(self, X):
        # Assume X is [batch, times, dim]
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret
