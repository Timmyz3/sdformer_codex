"""Rapid train/profile screening for H9+ neuron attention variants.

This entrypoint intentionally lives outside third_party/SDformerFlow. It
creates temporary short-run configs from existing experiment configs, trains for
a small number of steps, profiles the resulting checkpoint, and writes a compact
ranking table. The goal is to reject bad neuron/attention/threshold choices
before spending a full training run on them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = EXP_ROOT / "configs"
RESULTS_DIR = EXP_ROOT / "results"
DEFAULT_BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor: Any = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if isinstance(cursor, dict):
            cursor = cursor.setdefault(part, {})
        else:
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
    if not isinstance(cursor, dict):
        raise ValueError(f"Cannot set {dotted_key}: parent is not a mapping")
    cursor[parts[-1]] = value


def apply_setters(config: dict[str, Any], setters: list[str]) -> None:
    for item in setters:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        set_nested(config, key, parse_scalar(value))


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[rapid-screen] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    return checkpoints[-1] if checkpoints else None


def compact_summary(name: str, steps: int, summary_path: Path, train_seconds: float, profile_seconds: float) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    aee = float(metrics.get("AEE", math.inf))
    aae = float(metrics.get("AAE", math.inf))
    sops_g = float(data.get("estimated_total_sops", math.inf)) / 1e9
    firing = float(data.get("global_firing_rate", math.inf))
    score = aee + 0.02 * aae + 0.04 * max(0.0, sops_g - 3.10)
    return {
        "name": name,
        "steps": steps,
        "samples": int(data.get("samples", 0)),
        "AEE": aee,
        "AAE": aae,
        "PE1": float(metrics.get("AEE_PE1", math.inf)),
        "PE2": float(metrics.get("AEE_PE2", math.inf)),
        "PE3": float(metrics.get("AEE_PE3", metrics.get("AEE_outliers", math.inf))),
        "SOPs_G": sops_g,
        "firing": firing,
        "score": score,
        "train_seconds": train_seconds,
        "profile_seconds": profile_seconds,
        "summary": str(summary_path),
    }


def make_short_config(base_config: Path, out_config: Path, experiment: str, steps: int, args: argparse.Namespace) -> None:
    config = deepcopy(load_yaml(base_config))
    config["experiment"] = experiment
    runtime = config.setdefault("runtime", {})
    runtime["max_train_steps"] = steps
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False
    loader = config.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = args.batch_size
    loader["n_workers"] = args.workers
    loader["persistent_workers"] = args.workers > 0
    loader["prefetch_factor"] = args.prefetch_factor
    loader["pin_memory"] = args.pin_memory
    optimizer = config.setdefault("optimizer", {})
    if args.lr is not None:
        optimizer["lr"] = args.lr
    if args.amp is not None:
        optimizer["use_amp"] = args.amp
    config.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    config.setdefault("test", {})["sample"] = args.valid_samples
    apply_setters(config, args.set)
    dump_yaml(out_config, config)


def profile_checkpoint(config_path: Path, checkpoint: Path, out_dir: Path, samples: int, args: argparse.Namespace) -> tuple[int, float]:
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_sops.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(out_dir),
        "--split",
        "valid",
        "--num-samples",
        str(samples),
        "--batch-size",
        str(args.profile_batch_size),
        "--num-workers",
        str(args.profile_workers),
        "--metric",
        "AEE",
        "--metric",
        "AAE",
    ]
    start = time.time()
    exit_code = run_command(command, out_dir / "profile.log")
    return exit_code, time.time() - start


def write_tables(rows: list[dict[str, Any]], out_root: Path) -> None:
    rows = sorted(rows, key=lambda item: item["score"])
    csv_path = out_root / "summary.csv"
    md_path = out_root / "summary.md"
    fields = [
        "name",
        "steps",
        "samples",
        "AEE",
        "AAE",
        "PE1",
        "PE2",
        "PE3",
        "SOPs_G",
        "firing",
        "score",
        "train_seconds",
        "profile_seconds",
        "summary",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Rapid Screen Summary\n\n")
        handle.write("| rank | name | steps | samples | AEE | AAE | SOPs(G) | firing | score | summary |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for rank, row in enumerate(rows, 1):
            handle.write(
                f"| {rank} | {row['name']} | {row['steps']} | {row['samples']} | "
                f"{row['AEE']:.4f} | {row['AAE']:.4f} | {row['SOPs_G']:.4f} | "
                f"{row['firing']:.5f} | {row['score']:.4f} | `{row['summary']}` |\n"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, help="Config filename under configs/ or an absolute path.")
    parser.add_argument("--steps", type=int, action="append", default=None, help="Train steps per candidate. Repeatable.")
    parser.add_argument("--prev-runid", type=Path, default=DEFAULT_BASELINE_CKPT, help="Checkpoint to fine-tune from.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--set", action="append", default=[], help="Override config value, e.g. atlif_ternary_psn.target_rate=0.035")
    parser.add_argument("--valid-samples", type=int, default=10, help="Fast profile samples.")
    parser.add_argument("--promote-samples", type=int, default=40, help="Second-stage profile samples.")
    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--profile-workers", type=int, default=4)
    parser.add_argument("--promote-aee", type=float, default=1.62)
    parser.add_argument("--promote-aae", type=float, default=8.2)
    parser.add_argument("--promote-sops-g", type=float, default=3.90)
    parser.add_argument("--no-promote-valid40", action="store_true")
    parser.add_argument("--tag", default="rapid_screen")
    return parser.parse_args(argv)


def resolve_config(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps is None:
        args.steps = [40, 120]
    run_stamp = stamp()
    out_root = RESULTS_DIR / f"{args.tag}_{run_stamp}"
    generated_dir = out_root / "configs"
    rows: list[dict[str, Any]] = []
    for raw_config in args.config:
        base_config = resolve_config(raw_config)
        base_stem = base_config.stem
        for steps in args.steps:
            name = f"{base_stem}_steps{steps}"
            config_path = generated_dir / f"{name}.yml"
            make_short_config(base_config, config_path, name, steps, args)
            run_dir = out_root / "runs" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-u",
                str(EXP_ROOT / "entrypoints/train.py"),
                "--config",
                str(config_path),
                "--prev_runid",
                str(args.prev_runid.resolve()),
                "--save_path",
                str(run_dir / "checkpoint_epoch{}.pth"),
            ]
            start = time.time()
            train_exit = run_command(command, run_dir / "train.log")
            train_seconds = time.time() - start
            checkpoint = latest_checkpoint(run_dir)
            if train_exit != 0 or checkpoint is None:
                continue
            profile_dir = out_root / "profiles" / f"{name}_valid{args.valid_samples}"
            profile_exit, profile_seconds = profile_checkpoint(config_path, checkpoint, profile_dir, args.valid_samples, args)
            summary_path = profile_dir / "sops_summary.json"
            if profile_exit != 0 or not summary_path.exists():
                continue
            row = compact_summary(name, steps, summary_path, train_seconds, profile_seconds)
            rows.append(row)
            should_promote = (
                not args.no_promote_valid40
                and row["AEE"] <= args.promote_aee
                and row["AAE"] <= args.promote_aae
                and row["SOPs_G"] <= args.promote_sops_g
            )
            if should_promote:
                promote_dir = out_root / "profiles" / f"{name}_valid{args.promote_samples}"
                promote_exit, promote_seconds = profile_checkpoint(config_path, checkpoint, promote_dir, args.promote_samples, args)
                promote_summary = promote_dir / "sops_summary.json"
                if promote_exit == 0 and promote_summary.exists():
                    promoted = compact_summary(
                        f"{name}_valid{args.promote_samples}",
                        steps,
                        promote_summary,
                        train_seconds,
                        promote_seconds,
                    )
                    rows.append(promoted)
            write_tables(rows, out_root)
    write_tables(rows, out_root)
    print(f"rapid screen dir: {out_root}")
    print(f"summary: {out_root / 'summary.md'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
