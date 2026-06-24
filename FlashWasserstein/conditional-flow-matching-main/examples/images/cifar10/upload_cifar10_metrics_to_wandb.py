"""Upload CIFAR-10 training/validation metrics.jsonl files to Weights & Biases.

Each run directory under --run_root is expected to contain:
  - args.json
  - metrics.jsonl

The script also writes a combined CSV so the raw per-step timing/validation data
is easy to archive independently of W&B.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, bool) or isinstance(value, bool)


def safe_wandb_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value[:120].strip("-_.") or "cifar10-run"


def selected_run_dirs(run_root: Path, include: str) -> list[Path]:
    needles = [item.strip() for item in include.split(",") if item.strip()]
    dirs = [p for p in sorted(run_root.iterdir()) if p.is_dir() and not p.name.startswith("_")]
    if not needles:
        return dirs
    return [p for p in dirs if any(needle in p.name for needle in needles)]


def write_combined_csv(rows_by_run: dict[str, list[dict[str, Any]]], csv_out: Path) -> None:
    field_order = [
        "run",
        "step",
        "loss",
        "val_loss",
        "step_s",
        "step_time_s",
        "ot_s",
        "ot_time_s",
        "images/s",
        "images_s",
        "images_per_s",
        "peak_mem_gb",
        "lr",
        "sample_cost",
        "duplicate_fraction",
        "context_size",
        "source_context_size",
        "target_context_size",
        "eps",
        "sinkhorn_iters",
        "val_time_s",
        "val_ot_time_s",
        "val_batches",
        "val_images",
    ]
    extras = set()
    for rows in rows_by_run.values():
        for row in rows:
            extras.update(row)
    fields = field_order + sorted(k for k in extras if k not in set(field_order))

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run_name, rows in rows_by_run.items():
            for row in rows:
                out = {"run": run_name, **row}
                writer.writerow(out)


def make_wandb_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in row.items():
        if key == "step":
            continue
        if is_scalar(value):
            metrics[key] = value
    if "step_time_s" in metrics and "step_s" not in metrics:
        metrics["step_s"] = metrics["step_time_s"]
    if "ot_time_s" in metrics and "ot_s" not in metrics:
        metrics["ot_s"] = metrics["ot_time_s"]
    return metrics


def upload_run(
    wandb,
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    project: str,
    entity: str | None,
    group: str,
    tags: list[str],
    resume: str,
    stable_ids: bool,
    dry_run: bool,
) -> None:
    config = read_json(run_dir / "args.json")
    config.update(
        {
            "metrics_path": str(run_dir / "metrics.jsonl"),
            "run_dir": str(run_dir),
            "uploaded_rows": len(rows),
        }
    )
    run_id = safe_wandb_id(f"{group}-{run_dir.name}") if stable_ids else None
    print(f"{run_dir.name}: rows={len(rows)} id={run_id or '<new>'}")
    if dry_run:
        return

    wb_run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=run_dir.name,
        id=run_id,
        resume=resume,
        tags=tags,
        config=config,
        reinit=True,
    )
    try:
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        for row in rows:
            step = int(row.get("step", 0))
            metrics = make_wandb_metrics(row)
            metrics["step"] = step
            wandb.log(metrics, step=step)
        val_rows = [r for r in rows if "val_loss" in r]
        if val_rows:
            best = min(val_rows, key=lambda r: float(r["val_loss"]))
            wb_run.summary["best_val_loss"] = float(best["val_loss"])
            wb_run.summary["best_val_step"] = int(best["step"])
        if rows:
            wb_run.summary["final_step"] = int(rows[-1].get("step", 0))
            wb_run.summary["final_loss"] = float(rows[-1].get("loss", 0.0))
    finally:
        wb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "flashsinkhorn-cifar10"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--group", default=os.environ.get("WANDB_GROUP", "cifar10_flash_eps_context_sweep_400k"))
    parser.add_argument("--include", default="")
    parser.add_argument("--csv_out", default="")
    parser.add_argument("--tags", default="cifar10,flashsinkhorn,validation")
    parser.add_argument("--resume", choices=["allow", "must", "never"], default="allow")
    parser.add_argument("--no_stable_ids", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    run_dirs = selected_run_dirs(run_root, args.include)
    if not run_dirs:
        raise SystemExit(f"No run directories matched under {run_root}")

    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            print(f"skip {run_dir.name}: missing metrics.jsonl")
            continue
        rows = read_jsonl(metrics_path)
        if not rows:
            print(f"skip {run_dir.name}: empty metrics.jsonl")
            continue
        rows_by_run[run_dir.name] = rows

    if not rows_by_run:
        raise SystemExit("No metrics rows found.")

    csv_out = Path(args.csv_out).expanduser() if args.csv_out else run_root / "wandb_metrics_export.csv"
    write_combined_csv(rows_by_run, csv_out)
    print(f"wrote {csv_out}")

    if args.dry_run:
        wandb = None
    else:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing wandb. Install it with: pip install wandb") from exc

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    for run_dir in run_dirs:
        if run_dir.name not in rows_by_run:
            continue
        upload_run(
            wandb,
            run_dir=run_dir,
            rows=rows_by_run[run_dir.name],
            project=args.project,
            entity=args.entity or None,
            group=args.group,
            tags=tags,
            resume=args.resume,
            stable_ids=not args.no_stable_ids,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
