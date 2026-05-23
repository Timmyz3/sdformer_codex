"""H42 hyper-parameter sweep around the H41 SN S02 C result.

H41 proved that the sparse story is viable, but the full run became too sparse
after several epochs. This script profiles that curve, then tests milder
threshold/LR/angular settings around the same SN S02 C topology.
"""

from __future__ import annotations

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
GENERATED_DIR = CONFIG_DIR / "generated"
RESULTS_DIR = EXP_ROOT / "results"
BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"

H41_RUN_DIR = RESULTS_DIR / "h41_sns02c_dlr_20260522_183050_h41_full_20260522_183050_bs8_20260522_183050_setsid"
H41_CONFIG = CONFIG_DIR / "h41_sns02c_dlr_20260522_183050_h41_full_20260522_183050.yml"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def env() -> dict[str, str]:
    merged = os.environ.copy()
    merged["SDFORMER_USE_MLFLOW"] = "0"
    merged["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    merged["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return merged


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env(), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[h42] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def append_status(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {text.rstrip()}\n")


def set_deep(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    cursor: Any = mapping
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def apply_changes(config: dict[str, Any], changes: dict[str, Any]) -> None:
    for key, value in changes.items():
        set_deep(config, key, value)


def make_config(name: str, changes: dict[str, Any], *, epochs: int, force_epochs: list[int]) -> Path:
    config = deepcopy(read_yaml(GENERATED_DIR / "h40_p3_SNS02_C.yml"))
    config["experiment"] = name
    runtime = config.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = force_epochs
    runtime["use_mlflow_model_logging"] = False
    loader = config.setdefault("loader", {})
    loader["n_epochs"] = epochs
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True
    config.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    config.setdefault("test", {})["sample"] = 10
    config.setdefault("optimizer", {})["use_amp"] = True
    apply_changes(config, changes)
    config["note"] = (
        "H42 SN S02 C hyper-parameter sweep. "
        "Goal: keep the sparsity win near 2.8-3.2G SOPs while preventing the "
        "epoch9+ firing collapse seen in H41. "
        f"changes={changes}"
    )
    out = GENERATED_DIR / f"{name}.yml"
    write_yaml(out, config)
    return out


def parse_last_health(train_log: Path) -> dict[str, Any]:
    health: dict[str, Any] = {}
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
        if isinstance(parsed, dict):
            health = parsed
    return health


def profile_checkpoint(config: Path, checkpoint: Path, out_dir: Path, status: Path, label: str) -> dict[str, Any] | None:
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_sops.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(out_dir),
        "--split",
        "valid",
        "--num-samples",
        "40",
        "--batch-size",
        "1",
        "--num-workers",
        "4",
        "--metric",
        "AEE",
        "--metric",
        "AAE",
    ]
    code = run_command(command, out_dir / "profile.log")
    append_status(status, f"profile {label} exit={code}")
    summary = out_dir / "sops_summary.json"
    if code != 0 or not summary.exists():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "label": label,
        "AEE": float(metrics.get("AEE", math.nan)),
        "AAE": float(metrics.get("AAE", math.nan)),
        "SOPs_G": float(data.get("estimated_total_sops", math.nan)) / 1e9,
        "firing": float(data.get("global_firing_rate", math.nan)),
    }


def run_train(config: Path, run_dir: Path, status: Path, label: str) -> int:
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(config),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    code = run_command(command, run_dir / "train.log")
    append_status(status, f"train {label} exit={code}")
    return code


def score(row: dict[str, Any]) -> float:
    value = float(row["AEE"]) + 0.03 * float(row["AAE"])
    value += 0.65 * max(0.0, float(row["AEE"]) - 1.80)
    value += 0.06 * max(0.0, float(row["AAE"]) - 8.80)
    value += 0.25 * max(0.0, float(row["SOPs_G"]) - 3.20)
    value += 0.25 * max(0.0, 2.70 - float(row["SOPs_G"]))
    # Reward useful sparsity, but penalize near-silent training health.
    ternary = float(row.get("ternary_activity_mean", math.nan))
    if math.isfinite(ternary):
        value += 0.80 * max(0.0, 0.006 - ternary)
    if float(row["SOPs_G"]) <= 3.10:
        value -= 0.04
    return value


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "phase",
        "variant",
        "epoch",
        "AEE",
        "AAE",
        "SOPs_G",
        "firing",
        "threshold_mean",
        "ternary_activity_mean",
        "ternary_pos_neg_ratio",
        "ternary_zero_neg_modules",
        "score",
    ]
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# H42 SN S02 C 超参数调整实验\n\n")
        handle.write("目标：在 SOPs 约 2.8-3.2G 的前提下，抑制 H41 后期三值发放塌缩。\n\n")
        handle.write("| phase | variant | epoch | AEE | AAE | SOPs(G) | firing | threshold | ternary | pos/neg | zero_neg | score |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row.get('phase')} | {row.get('variant')} | {int(row.get('epoch', -1))} | "
                f"{float(row.get('AEE', math.nan)):.4f} | {float(row.get('AAE', math.nan)):.4f} | "
                f"{float(row.get('SOPs_G', math.nan)):.4f} | {float(row.get('firing', math.nan)):.5f} | "
                f"{float(row.get('threshold_mean', math.nan)):.4f} | "
                f"{float(row.get('ternary_activity_mean', math.nan)):.5f} | "
                f"{float(row.get('ternary_pos_neg_ratio', math.nan)):.2f} | "
                f"{int(float(row.get('ternary_zero_neg_modules', -1)))} | "
                f"{float(row.get('score', math.nan)):.4f} |\n"
            )


def add_row(rows: list[dict[str, Any]], phase: str, variant: str, epoch: int, profile: dict[str, Any], health: dict[str, Any]) -> None:
    row = {
        "phase": phase,
        "variant": variant,
        "epoch": epoch,
        "AEE": profile["AEE"],
        "AAE": profile["AAE"],
        "SOPs_G": profile["SOPs_G"],
        "firing": profile["firing"],
        "threshold_mean": health.get("threshold_mean", math.nan),
        "ternary_activity_mean": health.get("ternary_activity_mean", math.nan),
        "ternary_pos_neg_ratio": health.get("ternary_pos_neg_ratio", math.nan),
        "ternary_zero_neg_modules": health.get("ternary_zero_neg_modules", math.nan),
    }
    row["score"] = score(row)
    rows.append(row)


def main() -> int:
    run_stamp = stamp()
    out_dir = RESULTS_DIR / f"h42_sn_hparam_sweep_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = out_dir / "status.log"
    rows: list[dict[str, Any]] = []
    append_status(status, "H42 started")

    # First, profile the existing H41 trajectory so the hyper-parameter changes
    # are grounded in the actual failure curve.
    h41_epochs = [0, 3, 6, 9, 12, 15, 18]
    h41_health = parse_last_health(H41_RUN_DIR / "train.log")
    for epoch in h41_epochs:
        checkpoint = H41_RUN_DIR / f"checkpoint_epoch{epoch}.pth"
        if not checkpoint.exists():
            continue
        profile = profile_checkpoint(
            H41_CONFIG,
            checkpoint,
            out_dir / "profiles" / f"h41_epoch{epoch}_valid40",
            status,
            f"H41 epoch{epoch}",
        )
        if profile:
            add_row(rows, "h41_curve", "h41_sns02c_dlr", epoch, profile, h41_health)
            write_summary(rows, out_dir)

    candidates: list[tuple[str, dict[str, Any]]] = [
        (
            "h42a_mild_theta",
            {
                "optimizer.param_groups.threshold_lr": 1.0e-6,
                "optimizer.param_groups.neuron_lr": 1.0e-5,
                "atlif_ternary_psn.threshold_base_lr": 1.0e-6,
                "atlif_ternary_psn.threshold_eta": 0.00025,
                "atlif_ternary_psn.threshold_lr_scale": 20000.0,
                "atlif_ternary_psn.target_rate_eta": 0.03,
                "atlif_ternary_psn.max_threshold": 1.45,
            },
        ),
        (
            "h42b_keepfire_target08",
            {
                "optimizer.param_groups.threshold_lr": 8.0e-7,
                "optimizer.param_groups.neuron_lr": 1.0e-5,
                "atlif_ternary_psn.threshold_base_lr": 8.0e-7,
                "atlif_ternary_psn.threshold_eta": 0.00018,
                "atlif_ternary_psn.threshold_lr_scale": 16000.0,
                "atlif_ternary_psn.target_rate": 0.08,
                "atlif_ternary_psn.target_rate_eta": 0.025,
                "atlif_ternary_psn.max_threshold": 1.35,
            },
        ),
        (
            "h42c_mild_theta_ang005",
            {
                "optimizer.param_groups.threshold_lr": 1.0e-6,
                "optimizer.param_groups.neuron_lr": 1.0e-5,
                "atlif_ternary_psn.threshold_base_lr": 1.0e-6,
                "atlif_ternary_psn.threshold_eta": 0.00025,
                "atlif_ternary_psn.threshold_lr_scale": 20000.0,
                "atlif_ternary_psn.target_rate_eta": 0.03,
                "atlif_ternary_psn.max_threshold": 1.45,
                "loss.use_angular_loss": True,
                "loss.lambda_ang": 0.05,
            },
        ),
        (
            "h42d_fast_backbone_mild",
            {
                "optimizer.param_groups.backbone_lr": 1.0e-6,
                "optimizer.param_groups.norm_lr": 1.0e-6,
                "optimizer.param_groups.threshold_lr": 1.0e-6,
                "optimizer.param_groups.neuron_lr": 1.0e-5,
                "atlif_ternary_psn.threshold_base_lr": 1.0e-6,
                "atlif_ternary_psn.threshold_eta": 0.00025,
                "atlif_ternary_psn.threshold_lr_scale": 20000.0,
                "atlif_ternary_psn.target_rate_eta": 0.03,
                "atlif_ternary_psn.max_threshold": 1.45,
            },
        ),
        (
            "h42e_slowbb_ang01",
            {
                "optimizer.param_groups.backbone_lr": 2.0e-7,
                "optimizer.param_groups.norm_lr": 2.0e-7,
                "optimizer.param_groups.threshold_lr": 1.0e-6,
                "optimizer.param_groups.neuron_lr": 1.0e-5,
                "atlif_ternary_psn.threshold_base_lr": 1.0e-6,
                "atlif_ternary_psn.threshold_eta": 0.00025,
                "atlif_ternary_psn.threshold_lr_scale": 20000.0,
                "atlif_ternary_psn.target_rate_eta": 0.03,
                "atlif_ternary_psn.max_threshold": 1.45,
                "loss.use_angular_loss": True,
                "loss.lambda_ang": 0.10,
            },
        ),
    ]

    short_rows: list[dict[str, Any]] = []
    config_by_variant: dict[str, Path] = {}
    for variant, changes in candidates:
        config = make_config(f"{variant}_{run_stamp}", changes, epochs=3, force_epochs=[0, 1, 2])
        config_by_variant[variant] = config
        run_dir = out_dir / "runs" / variant
        code = run_train(config, run_dir, status, variant)
        if code != 0:
            continue
        health = parse_last_health(run_dir / "train.log")
        for epoch in [0, 1, 2]:
            checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
            if not checkpoint.exists():
                continue
            profile = profile_checkpoint(
                config,
                checkpoint,
                out_dir / "profiles" / f"{variant}_epoch{epoch}_valid40",
                status,
                f"{variant} epoch{epoch}",
            )
            if profile:
                add_row(rows, "short3", variant, epoch, profile, health)
                short_rows.append(rows[-1])
                write_summary(rows, out_dir)

    eligible = [
        row
        for row in short_rows
        if row["epoch"] == 2
        and float(row["SOPs_G"]) <= 3.25
        and float(row["AEE"]) <= 1.92
        and float(row["AAE"]) <= 9.20
        and float(row["ternary_activity_mean"]) >= 0.006
    ]
    if not eligible:
        eligible = [row for row in short_rows if row["epoch"] == 2] or short_rows
    if not eligible:
        append_status(status, "No eligible short rows; stop before extended run.")
        return 1
    chosen = min(eligible, key=score)
    chosen_variant = str(chosen["variant"])
    append_status(status, f"extended chosen={chosen_variant}")

    # Re-run chosen candidate for 10 epochs from baseline, saving every epoch,
    # to test whether the epoch9 collapse is fixed before spending a full run.
    chosen_base_changes = next(changes for variant, changes in candidates if variant == chosen_variant)
    ext_config = make_config(
        f"{chosen_variant}_extended10_{run_stamp}",
        chosen_base_changes,
        epochs=10,
        force_epochs=list(range(10)),
    )
    ext_dir = out_dir / "runs" / f"{chosen_variant}_extended10"
    code = run_train(ext_config, ext_dir, status, f"{chosen_variant}_extended10")
    if code != 0:
        return code
    ext_health = parse_last_health(ext_dir / "train.log")
    for epoch in [0, 3, 6, 9]:
        checkpoint = ext_dir / f"checkpoint_epoch{epoch}.pth"
        if not checkpoint.exists():
            continue
        profile = profile_checkpoint(
            ext_config,
            checkpoint,
            out_dir / "profiles" / f"{chosen_variant}_extended_epoch{epoch}_valid40",
            status,
            f"{chosen_variant} extended epoch{epoch}",
        )
        if profile:
            add_row(rows, "extended10", chosen_variant, epoch, profile, ext_health)
            write_summary(rows, out_dir)
    append_status(status, "H42 done")
    print(f"h42 dir: {out_dir}")
    print(f"summary: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
