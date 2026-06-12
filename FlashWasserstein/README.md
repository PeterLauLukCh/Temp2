# FlashWasserstein

FlashWasserstein is a standalone research prototype for unregularized quadratic
optimal transport. It contains two related paths:

- a semi-dual hard-OT solver backed by FlashSinkhorn's streaming `c_transform`;
- an OT Flow Matching compatible equal-size uniform minibatch assignment solver
  backed by fused Flash/Triton epsilon-auction primitives.

The semi-dual objective is

```text
Phi(psi) = sum_i a_i min_j [s ||x_i - y_j||^2 - psi_j] + sum_j b_j psi_j
```

with mass-residual supergradient

```text
grad_psi_j = b_j - sum_{i: j*(i)=j} a_i.
```

The deterministic assignment `j*(i)` is a Monge assignment induced by `psi`.
For finite empirical measures this assignment may not satisfy the target
marginal exactly, so every result reports `mass_error_l1`.

## Layout

```text
flash_wasserstein/
  solver.py      # Flash c-transform + subgradient semi-dual solver
  dense.py       # Dense reference c-transform and solver
  monge.py       # Monge assignment/map diagnostics
  baselines.py   # Optional POT exact-OT baseline for small problems
benchmarks/
  bench_semidual.py
  bench_gaussian_2d.py
  bench_otfm_flash.py
tests/
```

## Usage

From the repository root:

```bash
export PYTHONPATH="$PWD/FlashWasserstein:$PWD/code/src:$PYTHONPATH"
```

Dense CPU/GPU reference:

```python
import torch
from flash_wasserstein import solve_dense_semidual

x = torch.randn(256, 8)
y = torch.randn(256, 8)
result = solve_dense_semidual(x, y, cost_scale=0.5, max_iter=100, lr=1.0)
print(result.mass_error_l1, result.transport_cost)
```

Flash CUDA solver:

```python
import torch
from flash_wasserstein import solve_flash_wasserstein

x = torch.randn(4096, 64, device="cuda")
y = torch.randn(4096, 64, device="cuda")
result = solve_flash_wasserstein(x, y, cost_scale=0.5, max_iter=100, lr=1.0)
print(result.mass_error_l1, result.transport_cost)
```

OT Flow Matching compatible balanced pairs:

```python
import torch
from flash_wasserstein import solve_flash_otfm_pairs

x0 = torch.randn(1024, 2, device="cuda")
x1 = torch.randn(1024, 2, device="cuda")
pairs = solve_flash_otfm_pairs(
    x0,
    x1,
    cost_scale=0.5,
    epsilon=1e-2,
    epsilon_schedule=[0.5, 0.2, 0.1, 0.05, 0.01],
    fused_bids=True,
    fused_accept=True,
)

# Direct OT-FM ingredients:
paired_x0 = pairs.paired_x
paired_x1 = pairs.paired_y
t = torch.rand(x0.shape[0], 1, device=x0.device)
x_t = (1.0 - t) * paired_x0 + t * paired_x1
u_t = paired_x1 - paired_x0
```

`solve_flash_otfm_pairs` is distinct from `solve_flash_wasserstein`: it returns
a balanced permutation coupling for equal-size uniform minibatches. This is the
mode intended to replace POT-style minibatch OT pairing in OT Flow Matching.
It raises on non-convergence by default rather than returning invalid pairs.
The `epsilon_schedule` option performs auction epsilon-scaling: coarse balanced
pairs warm-start the prices before the final, tighter pairing pass. Later
epsilon stages keep the previous assignment and only re-auction rows that
violate the tighter epsilon-complementary-slackness certificate.

OT-FM integration:

```python
from torchcfm.conditional_flow_matching import FlashOptimalTransportConditionalFlowMatcher

fm = FlashOptimalTransportConditionalFlowMatcher(
    sigma=0.0,
    flash_epsilon=1e-2,
    flash_epsilon_schedule=[0.5, 0.2, 0.1, 0.05, 0.01],
)
t, x_t, u_t = fm.sample_location_and_conditional_flow(x0, x1)
```

Equivalently, use `OTPlanSampler(method="flash")`. Its training path samples
pair indices directly from the sparse permutation coupling and avoids
materializing the dense `n x n` plan. Calling `get_map()` still returns a dense
permutation plan for debugging/API compatibility.

Evaluate the induced Monge assignment:

```python
from flash_wasserstein import monge_map

mapped = monge_map(x, y, result.psi, backend="flash")
print(mapped.mapped_y.shape, mapped.mass_error_l1)
```

## Benchmarks

```bash
python3 FlashWasserstein/benchmarks/bench_semidual.py \
  --sizes 256,512,4096 \
  --dims 8,64 \
  --methods dense,flash,pot \
  --device cuda
```

Outputs are written to `FlashWasserstein/output/` as JSON and CSV.

2D Gaussian benchmark:

```bash
CUDA_VISIBLE_DEVICES=9 python3 FlashWasserstein/benchmarks/bench_gaussian_2d.py \
  --sizes 256,512,2048,8192 \
  --cases shift,anisotropic,near \
  --device cuda \
  --lr-mass-scale 0.25 \
  --warmup-iter 2
```

`--lr-mass-scale alpha` uses `lr = alpha * n` for each `n`. This is often a
better starting point than a fixed `lr` because the default uniform residuals
are normalized masses of size `1/n`.

For the named 2D Gaussian cases, the benchmark also writes a
`gaussian_closed_form` row containing the closed-form population Gaussian W2
cost. POT remains the finite empirical exact-OT reference on small sizes.

OT-FM pairing benchmark:

```bash
CUDA_VISIBLE_DEVICES=9 python3 FlashWasserstein/benchmarks/bench_otfm_flash.py \
  --sizes 128 512 1024 2048 4096 \
  --dim 16 \
  --ablate-top2
```

This compares the OT-FM POT sampler, the fully fused Flash auction sampler, and
the older non-fused Flash top-2 auction loop.

## Tests

```bash
PYTHONPATH="$PWD/FlashWasserstein:$PWD/code/src:$PYTHONPATH" \
  python3 -m pytest FlashWasserstein/tests
```

Tests skip cleanly when optional dependencies are missing:

- no `torch`: all runtime tests skip;
- no CUDA/Triton/FlashSinkhorn: Flash tests skip;
- no POT (`ot`): POT baseline tests skip.

## Limitations

- The OT-FM path is an epsilon-accurate auction solver, not exact network
  simplex. If the final assignment satisfies epsilon-CS, its total assignment
  cost is at most `n * epsilon` above the exact linear-assignment optimum under
  the chosen cost scale.
- The solver is nonsmooth subgradient ascent, not a polished production OT
  optimizer.
- `solve_flash_otfm_pairs` currently targets the equal-size uniform minibatch
  setting. Unequal sizes or nonuniform weights need a capacitated assignment or
  general coupling extension.
- POT can still be faster for small low-dimensional minibatches because it uses
  optimized CPU network-simplex code. The Flash path is intended for dense-free
  GPU pairing and larger minibatches where materializing a full plan is the
  bottleneck.
- Exact Monge maps need not exist for arbitrary finite empirical weights.
  Inspect `mass_error_l1` rather than assuming the target marginal is matched.
- The scalable Flash backend supports squared Euclidean / half-squared
  Euclidean costs through the additive-dot-product structure.
