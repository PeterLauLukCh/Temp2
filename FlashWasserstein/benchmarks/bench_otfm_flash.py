"""Benchmark OT-FM minibatch pairing with FlashWasserstein.

This benchmark times the actual OT-FM sampling interface:

    OTPlanSampler(method="exact").sample_plan(...)
    OTPlanSampler(method="flash").sample_plan(...)

It also reports a direct FlashWasserstein ablation with ``fused_bids=False`` to
measure the benefit of the fused auction bid oracle over the older top-2 oracle
loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OTFM_ROOT = ROOT / "conditional-flow-matching-main"
for path in (ROOT, REPO_ROOT / "code" / "src", OTFM_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flash_wasserstein import solve_flash_otfm_pairs  # noqa: E402
from torchcfm.optimal_transport import OTPlanSampler  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, *, device, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    last = None
    for _ in range(repeat):
        _sync(device)
        start = time.perf_counter()
        last = fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times), last


def _make_gaussian(n, d, *, device, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x0 = torch.randn(n, d, generator=gen, device=device)
    shift = torch.linspace(-0.5, 0.75, d, device=device)
    x1 = torch.randn(n, d, generator=gen, device=device) + shift
    return x0, x1


def _pair_cost(x0, x1):
    return float((x0.float() - x1.float()).pow(2).sum(dim=1).mean().item())


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    schedule = [float(v) for v in args.epsilon_schedule.split(",") if v]

    for n in args.sizes:
        x0, x1 = _make_gaussian(n, args.dim, device=device, seed=args.seed + n)

        flash_sampler = OTPlanSampler(
            method="flash",
            flash_backend="cuda" if device.type == "cuda" else "dense",
            flash_epsilon=args.epsilon,
            flash_epsilon_schedule=schedule,
            flash_max_rounds=args.max_rounds,
            flash_verify=False,
            flash_fused_bids=True,
            flash_allow_tf32=not args.no_tf32,
        )

        def flash_sample():
            return flash_sampler.sample_plan(x0, x1, replace=False)

        flash_min, flash_mean, flash_pairs = _time_call(
            flash_sample,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        rows.append(
            {
                "n": n,
                "d": args.dim,
                "method": "otfm_flash_fused",
                "time_min_s": flash_min,
                "time_mean_s": flash_mean,
                "mean_pair_cost": _pair_cost(*flash_pairs),
            }
        )

        if args.ablate_top2 and device.type == "cuda":
            def flash_top2_direct():
                result = solve_flash_otfm_pairs(
                    x0.reshape(n, -1),
                    x1.reshape(n, -1),
                    cost_scale=1.0,
                    epsilon=args.epsilon,
                    epsilon_schedule=schedule,
                    max_rounds=args.max_rounds,
                    verify=False,
                    allow_tf32=not args.no_tf32,
                    fused_bids=False,
                    fused_accept=False,
                )
                return x0, x1[result.permutation]

            top2_min, top2_mean, top2_pairs = _time_call(
                flash_top2_direct,
                device=device,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            rows.append(
                {
                    "n": n,
                    "d": args.dim,
                    "method": "flash_top2_loop",
                    "time_min_s": top2_min,
                    "time_mean_s": top2_mean,
                    "mean_pair_cost": _pair_cost(*top2_pairs),
                }
            )

        if n <= args.max_pot_n:
            pot_sampler = OTPlanSampler(method="exact")

            def pot_sample():
                return pot_sampler.sample_plan(x0, x1, replace=False)

            pot_min, pot_mean, pot_pairs = _time_call(
                pot_sample,
                device=device,
                warmup=0,
                repeat=max(1, min(args.repeat, 2)),
            )
            rows.append(
                {
                    "n": n,
                    "d": args.dim,
                    "method": "otfm_pot_exact",
                    "time_min_s": pot_min,
                    "time_mean_s": pot_mean,
                    "mean_pair_cost": _pair_cost(*pot_pairs),
                }
            )

    csv_path = out_dir / "otfm_flash_bench.csv"
    json_path = out_dir / "otfm_flash_bench.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "d", "method", "time_min_s", "time_mean_s", "mean_pair_cost"],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))

    print(json.dumps(rows, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--epsilon-schedule", default="0.5,0.2,0.1,0.05,0.01")
    parser.add_argument("--max-rounds", type=int, default=20000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-pot-n", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default=str(ROOT / "output" / "otfm_flash_bench"))
    parser.add_argument("--ablate-top2", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
