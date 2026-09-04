"""Rapid train/profile screening for H9+ neuron attention variants.

This entrypoint intentionally lives outside third_party/SDformerFlow. It
creates temporary short-run configs from existing experiment configs, trains for
a small number of steps, profiles the resulting checkpoint, and writes a compact
ranking table. The goal is to reject bad neuron/attention/threshold choices
before spending a full training run on them.
"""

from __future__ import annotations

import argparse
import ast
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

H9A_AEE = 1.5044
H9A_AAE = 7.6365
H9A_SOPS_G = 3.0847
BASELINE_FIRING = 0.084961


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


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


def extract_train_health(train_log: Path) -> dict[str, float | int]:
    health: dict[str, float | int] = {
        "threshold_mean": math.inf,
        "threshold_min": math.inf,
        "threshold_max": math.inf,
        "ternary_activity_mean": math.inf,
        "ternary_pos_neg_ratio": math.inf,
        "ternary_worst_pos_neg_ratio": math.inf,
        "ternary_zero_pos_modules": math.inf,
        "ternary_zero_neg_modules": math.inf,
        "target_rate_control_modules": 0,
        "target_rate_bidirectional_modules": 0,
        "raw_update_mean": 0.0,
        "guarded_update_mean": 0.0,
        "quantile_guard_mean": 1.0,
        "importance_guard_mean": 1.0,
        "quantile_value_mean": 0.0,
        "importance_ema_mean": 0.0,
    }
    if not train_log.exists():
        return health
    for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "[H9] ATLIFTernaryPSN summary:" in line:
            payload = line.split("[H9] ATLIFTernaryPSN summary:", 1)[1].strip()
        elif " update: " in line:
            payload = line.split(" update: ", 1)[1].strip()
        else:
            continue
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, dict) or "threshold_mean" not in parsed:
            continue
        for key in health:
            if key in parsed:
                value = parsed[key]
                if isinstance(value, (int, float)):
                    health[key] = value
    return health


def candidate_score(aee: float, aae: float, sops_g: float, firing: float, health: dict[str, float | int]) -> float:
    score = aee + 0.025 * aae
    score += 0.30 * max(0.0, sops_g - H9A_SOPS_G)
    score += 0.90 * max(0.0, aee - 1.58)
    score += 0.08 * max(0.0, aae - 7.90)
    score += 0.25 * max(0.0, sops_g - 3.50)
    score += 1.2 * max(0.0, firing - BASELINE_FIRING)
    score += 0.015 * max(0.0, float(health.get("ternary_zero_neg_modules", 0)) - 2.0)
    score += 0.002 * max(0.0, float(health.get("ternary_worst_pos_neg_ratio", 1.0)) - 20.0)
    if sops_g <= 3.25:
        score -= 0.05
    if aee <= H9A_AEE and aae <= H9A_AAE:
        score -= 0.04
    return score


def gate_reason(row: dict[str, Any], args: argparse.Namespace) -> str:
    if row["AEE"] > args.promote_aee:
        return f"AEE>{args.promote_aee}"
    if row["AAE"] > args.promote_aae:
        return f"AAE>{args.promote_aae}"
    if row["SOPs_G"] > args.promote_sops_g:
        return f"SOPs>{args.promote_sops_g}G"
    if row["ternary_zero_neg_modules"] > args.max_zero_neg_modules:
        return f"zero_neg>{args.max_zero_neg_modules}"
    if row["ternary_worst_pos_neg_ratio"] > args.max_worst_pos_neg_ratio:
        return f"pos_neg_ratio>{args.max_worst_pos_neg_ratio}"
    return "pass"


def compact_summary(
    name: str,
    steps: int,
    summary_path: Path,
    train_log: Path,
    train_seconds: float,
    profile_seconds: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    aee = float(metrics.get("AEE", math.inf))
    aae = float(metrics.get("AAE", math.inf))
    sops_g = float(data.get("estimated_total_sops", math.inf)) / 1e9
    firing = float(data.get("global_firing_rate", math.inf))
    health = extract_train_health(train_log)
    row: dict[str, Any] = {
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
        "score": candidate_score(aee, aae, sops_g, firing, health),
        "stage": "confirm" if steps >= args.confirm_steps and int(data.get("samples", 0)) >= args.promote_samples else "screen",
        "train_seconds": train_seconds,
        "profile_seconds": profile_seconds,
        "summary": str(summary_path),
    }
    row.update(health)
    row["gate"] = gate_reason(row, args)
    return row


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
    rows = sorted(
        rows,
        key=lambda item: (
            item.get("stage") != "confirm",
            item.get("gate") != "pass",
            float(item["score"]),
        ),
    )
    csv_path = out_root / "summary.csv"
    md_path = out_root / "summary.md"
    fields = [
        "name",
        "stage",
        "gate",
        "steps",
        "samples",
        "AEE",
        "AAE",
        "PE1",
        "PE2",
        "PE3",
        "SOPs_G",
        "firing",
        "threshold_mean",
        "threshold_min",
        "threshold_max",
        "ternary_activity_mean",
        "ternary_pos_neg_ratio",
        "ternary_worst_pos_neg_ratio",
        "ternary_zero_pos_modules",
        "ternary_zero_neg_modules",
        "target_rate_control_modules",
        "target_rate_bidirectional_modules",
        "raw_update_mean",
        "guarded_update_mean",
        "quantile_guard_mean",
        "importance_guard_mean",
        "quantile_value_mean",
        "importance_ema_mean",
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
        handle.write("# 快速短训筛选汇总\n\n")
        handle.write(
            "排序逻辑：valid40 确认结果优先，综合考虑 AEE、AAE、SOPs、firing、"
            "三值负发放塌缩和正负比例。valid10 只用于早筛，不能单独决定全量。\n\n"
        )
        handle.write(
            "| rank | name | stage | gate | steps | samples | AEE | AAE | SOPs(G) | firing | "
            "zero_neg | worst_pos/neg | threshold | score | summary |\n"
        )
        handle.write("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for rank, row in enumerate(rows, 1):
            handle.write(
                f"| {rank} | {row['name']} | {row['stage']} | {row['gate']} | "
                f"{row['steps']} | {row['samples']} | "
                f"{row['AEE']:.4f} | {row['AAE']:.4f} | {row['SOPs_G']:.4f} | "
                f"{row['firing']:.5f} | {row['ternary_zero_neg_modules']:.0f} | "
                f"{row['ternary_worst_pos_neg_ratio']:.2f} | {row['threshold_mean']:.4f} | "
                f"{row['score']:.4f} | `{row['summary']}` |\n"
            )
        handle.write("\n## ATLIF 控制指标\n\n")
        handle.write("| name | raw_update | guarded_update | quantile_guard | importance_guard | quantile_value | importance_ema |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['name']} | {row['raw_update_mean']:.3e} | {row['guarded_update_mean']:.3e} | "
                f"{row['quantile_guard_mean']:.4f} | {row['importance_guard_mean']:.4f} | "
                f"{row['quantile_value_mean']:.4f} | {row['importance_ema_mean']:.4f} |\n"
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
    parser.add_argument("--promote-aee", type=float, default=1.58)
    parser.add_argument("--promote-aae", type=float, default=7.90)
    parser.add_argument("--promote-sops-g", type=float, default=3.35)
    parser.add_argument("--max-zero-neg-modules", type=float, default=4.0)
    parser.add_argument("--max-worst-pos-neg-ratio", type=float, default=40.0)
    parser.add_argument("--confirm-steps", type=int, default=360)
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel configs to run simultaneously.")
    parser.add_argument("--no-promote-valid40", action="store_true")
    parser.add_argument("--tag", default="rapid_screen")
    return parser.parse_args(argv)


def resolve_config(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    return path.resolve()


import concurrent.futures
import multiprocessing


def _run_one_config(
    raw_config: str, steps: int, args: argparse.Namespace, out_root: Path
) -> tuple[list[dict[str, Any]], Path | None]:
    """Run train+profile for a single config, returning rows and any promoted row dir."""
    from copy import deepcopy as _deepcopy
    local_args = _deepcopy(args)
    local_args.config = [raw_config]
    local_args.steps = [steps]
    local_args.parallel = 1  # prevent recursive parallel

    base_config = resolve_config(raw_config)
    base_stem = base_config.stem
    generated_dir = out_root / "configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    name = f"{base_stem}_steps{steps}"
    config_path = generated_dir / f"{name}.yml"
    make_short_config(base_config, config_path, name, steps, local_args)
    run_dir = out_root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config", str(config_path),
        "--prev_runid", str(local_args.prev_runid.resolve()),
        "--save_path", str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    start = time.time()
    train_exit = run_command(command, run_dir / "train.log")
    train_seconds = time.time() - start
    checkpoint = latest_checkpoint(run_dir)
    rows: list[dict[str, Any]] = []
    if train_exit != 0 or checkpoint is None:
        return rows, None
    profile_dir = out_root / "profiles" / f"{name}_valid{local_args.valid_samples}"
    profile_exit, profile_seconds = profile_checkpoint(config_path, checkpoint, profile_dir, local_args.valid_samples, local_args)
    summary_path = profile_dir / "sops_summary.json"
    if profile_exit != 0 or not summary_path.exists():
        return rows, None
    train_log = run_dir / "train.log"
    row = compact_summary(name, steps, summary_path, train_log, train_seconds, profile_seconds, local_args)
    rows.append(row)
    promote_dir = None
    if not local_args.no_promote_valid40 and row["gate"] == "pass" and steps >= local_args.confirm_steps:
        promote_dir = out_root / "profiles" / f"{name}_valid{local_args.promote_samples}"
        promote_exit, promote_seconds = profile_checkpoint(config_path, checkpoint, promote_dir, local_args.promote_samples, local_args)
        promote_summary = promote_dir / "sops_summary.json"
        if promote_exit == 0 and promote_summary.exists():
            promoted = compact_summary(f"{name}_valid{local_args.promote_samples}", steps, promote_summary, train_log, train_seconds, promote_seconds, local_args)
            rows.append(promoted)
    return rows, promote_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps is None:
        args.steps = [120, 360]
    run_stamp = stamp()
    out_root = RESULTS_DIR / f"{args.tag}_{run_stamp}"
    generated_dir = out_root / "configs"
    rows: list[dict[str, Any]] = []
    tasks = [(cfg, step) for cfg in args.config for step in args.steps]
    out_root.mkdir(parents=True, exist_ok=True)

    if args.parallel > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_run_one_config, cfg, step, args, out_root): (cfg, step) for cfg, step in tasks}
            for future in concurrent.futures.as_completed(futures):
                cfg, step = futures[future]
                try:
                    new_rows, _ = future.result()
                    rows.extend(new_rows)
                    if new_rows:
                        write_tables(rows, out_root)
                except Exception as e:
                    print(f"[rapid_screen] {cfg} step={step} failed: {e}", file=sys.stderr)
    else:
        for cfg, step in tasks:
            new_rows, _ = _run_one_config(cfg, step, args, out_root)
            rows.extend(new_rows)
            write_tables(rows, out_root)
    write_tables(rows, out_root)
    print(f"rapid screen dir: {out_root}")
    print(f"summary: {out_root / 'summary.md'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
