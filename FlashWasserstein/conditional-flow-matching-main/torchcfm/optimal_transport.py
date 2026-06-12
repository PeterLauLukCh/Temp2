import math
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Union

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
        flash_wasserstein_root = Path(__file__).resolve().parents[2]
        if flash_wasserstein_root.exists() and str(flash_wasserstein_root) not in sys.path:
            sys.path.insert(0, str(flash_wasserstein_root))
        from flash_wasserstein import solve_dense_otfm_pairs, solve_flash_otfm_pairs

        return solve_dense_otfm_pairs, solve_flash_otfm_pairs


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
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
    ) -> None:
        """Initialize the OTPlanSampler class.

        Parameters
        ----------
        method: str
            choose which optimal transport solver you would like to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced",
            "partial", "flash"] OT solvers.
        reg: float, optional
            regularization parameter to use for Sinkhorn-based iterative solvers.
        reg_m: float, optional
            regularization weight for unbalanced Sinkhorn-knopp solver.
        normalize_cost: bool, optional
            normalizes the cost matrix so that the maximum cost is 1. Helps
            stabilize Sinkhorn-based solvers. Should not be used in the vast
            majority of cases.
        num_threads: int or str, optional
            number of threads to use for the "exact" OT solver. If "max", uses
            the maximum number of threads.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        flash_epsilon: float, optional
            final epsilon for the FlashWasserstein auction backend.
        flash_epsilon_schedule: sequence, optional
            coarse-to-fine epsilon schedule. If omitted, FlashWasserstein uses
            a single epsilon stage.
        flash_max_rounds: int, optional
            maximum auction rounds per epsilon stage.
        flash_backend: str, optional
            "auto", "cuda", or "dense". "auto" uses the fused CUDA backend for
            CUDA minibatches and the dense reference backend otherwise.
        flash_cost_scale: float, optional
            squared-cost scale used by FlashWasserstein. OT-FM/POT uses
            ``||x-y||^2``, so the default is 1.0.
        flash_verify: bool, optional
            if True, compute the Flash epsilon-CS certificate after solving.
        flash_fused_bids: bool, optional
            if True, use the fused Flash bid oracle.
        flash_fused_accept: bool, optional
            if True, use the fused Flash accept/update kernels.
        """
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        self.method = method
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
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
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn
        self.flash_epsilon = flash_epsilon
        self.flash_epsilon_schedule = flash_epsilon_schedule
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

        kwargs = dict(
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
                **kwargs,
            )
        if self.flash_backend == "cuda":
            raise ValueError("flash_backend='cuda' requires CUDA minibatches.")
        return solve_dense_otfm_pairs(x0_flat, x1_flat, **kwargs)

    def _sample_flash_indices(self, x0, x1, replace=True):
        result = self._solve_flash_pairing(x0, x1)
        permutation = result.permutation.to(device=x0.device, dtype=torch.long)
        batch_size = x0.shape[0]
        if replace:
            i = torch.randint(batch_size, (batch_size,), device=x0.device)
            j = permutation[i]
        else:
            i = torch.arange(batch_size, device=x0.device)
            j = permutation
        return i, j, result

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        if _is_flash_method(self.method):
            _, j, _ = self._sample_flash_indices(x0, x1, replace=False)
            j_np = j.detach().cpu().numpy()
            p = np.zeros((x0.shape[0], x1.shape[0]), dtype=np.float64)
            p[np.arange(x0.shape[0]), j_np] = 1.0 / x0.shape[0]
            return p

        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        x0 = self._flatten_for_ot(x0)
        x1 = self._flatten_for_ot(x1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        if _is_flash_method(self.method):
            i, j, _ = self._sample_flash_indices(x0, x1, replace=replace)
            return x0[i], x1[j]

        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_scipy(self, x0, x1):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch using scipy and draw source and target samples from pi $(x,z) \sim \pi$.

        This sampler has two advantages:
        * Reduced variance compared to sampling from the OT plan
        * Preserves the order of x1 by construction
        * Preserves entire batch if x0 and x1 have the same size

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        import scipy

        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0.detach(), x1.detach()) ** 2
        if self.normalize_cost:
            M = M / M.max()
        _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
        pi_x0 = x0
        pi_x1 = x1[j]
        return pi_x0, pi_x1

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target labeled samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs)
            represents the source label minibatch
        y1 : Tensor, shape (bs)
            represents the target label minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the target minibatch drawn from $\pi$
        y0[i] : Tensor, shape (bs, *dim)
            represents the source label minibatch drawn from $\pi$
        y1[j] : Tensor, shape (bs, *dim)
            represents the target label minibatch drawn from $\pi$
        """
        if _is_flash_method(self.method):
            i, j, _ = self._sample_flash_indices(x0, x1, replace=replace)
            return (
                x0[i],
                x1[j],
                y0[i] if y0 is not None else None,
                y1[j] if y1 is not None else None,
            )

        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, X):
        """Compute the OT trajectories between different sample populations moving from the source
        to the target distribution.

        Parameters
        ----------
        X : Tensor, (bs, times, *dim)
            different populations of samples moving from the source to the target distribution.

        Returns
        -------
        to_return : Tensor, (bs, times, *dim)
            represents the OT sampled trajectories over time.
        """
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
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
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
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret
