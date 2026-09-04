"""Run the H40 stage-3 priority queue.

This queue uses the short names from neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md:

- TX S02 A + D
- SN S02 C + SB
- SC S012 C + D

Each candidate is trained for three epochs from the baseline checkpoint, saves
every epoch, profiles valid40 for each saved checkpoint, and writes a compact
Chinese summary. It intentionally stays outside third_party/SDformerFlow.
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
RESULTS_DIR = EXP_ROOT / "results"
BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


CANDIDATES = [
    {
        "short_name": "TX S02 A + D",
        "slug": "tx_s02_a_d",
        "base_config": "generated/h40_p3_TXS02_A.yml",
        "lr_strategy": "D",
        "note": "三值 alpha-XNOR shiftmax，S02 FFN，A preset，默认 differential LR。",
    },
    {
        "short_name": "SN S02 C + SB",
        "slug": "sn_s02_c_sb",
        "base_config": "generated/h40_p3_SNS02_C.yml",
        "lr_strategy": "SB",
        "note": "signed-consensus shiftnorm，S02 FFN，C preset，slow backbone LR。",
    },
    {
        "short_name": "SC S012 C + D",
        "slug": "sc_s012_c_d",
        "base_config": "generated/h40_p3_SCS012_C.yml",
        "lr_strategy": "D",
        "note": "signed-consensus shiftmax，S012 FFN，C preset，默认 differential LR。",
    },
]


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def result_env() -> dict[str, str]:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def apply_lr_strategy(config: dict[str, Any], strategy: str) -> None:
    optimizer = config.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    if strategy == "D":
        return
    if strategy == "SB":
        optimizer.setdefault("param_groups", {}).update(
            {
                "backbone_lr": 2.0e-7,
                "norm_lr": 2.0e-7,
                "neuron_lr": 1.2e-5,
                "threshold_lr": 3.0e-6,
            }
        )
        config.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = 3.0e-6
        return
    if strategy == "W":
        optimizer["lr_warmup"] = {"enabled": True, "steps": 80, "start_factor": 0.2}
        return
    raise ValueError(f"Unknown LR strategy: {strategy}")


def make_train_config(candidate: dict[str, str], out_dir: Path) -> Path:
    base = CONFIG_DIR / candidate["base_config"]
    config = deepcopy(read_yaml(base))
    experiment = f"h40_stage3_{candidate['slug']}"
    config["experiment"] = experiment
    runtime = config.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0, 1, 2]
    runtime["use_mlflow_model_logging"] = False
    loader = config.setdefault("loader", {})
    loader["n_epochs"] = 3
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True
    config.setdefault("test", {})["sample"] = 10
    config.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    apply_lr_strategy(config, candidate["lr_strategy"])
    config["note"] = (
        "H40 stage3 priority queue. "
        f"short_name={candidate['short_name']}; {candidate['note']} "
        "3 epoch staged training from baseline epoch59; profile valid40 per checkpoint."
    )
    out_path = out_dir / "configs" / f"{experiment}.yml"
    write_yaml(out_path, config)
    return out_path


def parse_last_health(train_log: Path) -> dict[str, Any]:
    health: dict[str, Any] = {}
    if not train_log.exists():
        return health
    for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        marker = None
        if "[H9] step " in line and " update: " in line:
            marker = " update: "
        elif "[H9] ATLIFTernaryPSN summary:" in line:
            marker = "[H9] ATLIFTernaryPSN summary:"
        if marker is None:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            health = parsed
    return health


def should_abort(health: dict[str, Any]) -> tuple[bool, str]:
    if not health:
        return False, ""
    zero_neg = float(health.get("ternary_zero_neg_modules", 0) or 0)
    worst = float(health.get("ternary_worst_pos_neg_ratio", 1) or 1)
    global_ratio = float(health.get("ternary_pos_neg_ratio", 1) or 1)
    activity = float(health.get("ternary_activity_mean", 0) or 0)
    if zero_neg > 4:
        return True, f"zero_neg={zero_neg} > 4"
    # A single module can make the worst ratio explode when its denominator is
    # numerically tiny while global pos/neg balance remains healthy. Treat that
    # as a warning unless multiple modules lose negative firing or global balance
    # also collapses.
    if math.isfinite(worst) and worst > 80 and (zero_neg > 1 or global_ratio > 10):
        return True, f"worst_pos_neg_ratio={worst:.2f}, global_ratio={global_ratio:.2f}, zero_neg={zero_neg}"
    if activity < 0.005:
        return True, f"ternary_activity_mean={activity:.5f} < 0.005"
    return False, ""


def monitor_process(proc: subprocess.Popen[Any], log_path: Path, status_path: Path, short_name: str) -> int:
    last_size = -1
    while True:
        code = proc.poll()
        health = parse_last_health(log_path)
        if log_path.exists():
            size = log_path.stat().st_size
        else:
            size = 0
        if size != last_size or code is not None:
            last_size = size
            bits = [f"[{datetime.now().isoformat(timespec='seconds')}] {short_name}"]
            if health:
                bits.append(
                    "threshold={:.4f} ternary={:.5f} posneg={:.2f} worst={:.2f} zero_neg={}".format(
                        float(health.get("threshold_mean", math.nan)),
                        float(health.get("ternary_activity_mean", math.nan)),
                        float(health.get("ternary_pos_neg_ratio", math.nan)),
                        float(health.get("ternary_worst_pos_neg_ratio", math.nan)),
                        int(health.get("ternary_zero_neg_modules", -1)),
                    )
                )
            if code is not None:
                bits.append(f"exit={code}")
            status_path.parent.mkdir(parents=True, exist_ok=True)
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(" | ".join(bits) + "\n")
        abort, reason = should_abort(health)
        if abort and code is None:
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[ABORT] {short_name}: {reason}\n")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 99
        if code is not None:
            return int(code)
        time.sleep(30)


def run_train(config_path: Path, run_dir: Path, status_path: Path, short_name: str) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(config_path),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=REPO_ROOT, env=result_env(), stdout=log, stderr=subprocess.STDOUT)
        code = monitor_process(proc, log_path, status_path, short_name)
        log.write(f"\n[h40-stage3] exit_code={code}\n")
    return code


def run_profile(config_path: Path, checkpoint: Path, out_dir: Path) -> tuple[int, dict[str, Any] | None]:
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
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "profile.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=result_env(), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[h40-stage3-profile] exit_code={proc.returncode}\n")
    summary_path = out_dir / "sops_summary.json"
    if proc.returncode != 0 or not summary_path.exists():
        return int(proc.returncode), None
    return int(proc.returncode), json.loads(summary_path.read_text(encoding="utf-8"))


def collect_row(candidate: dict[str, str], epoch: int, profile: dict[str, Any], run_dir: Path, train_log: Path) -> dict[str, Any]:
    metrics = profile.get("metrics", {})
    health = parse_last_health(train_log)
    return {
        "short_name": candidate["short_name"],
        "epoch": epoch,
        "AEE": float(metrics.get("AEE", math.nan)),
        "AAE": float(metrics.get("AAE", math.nan)),
        "SOPs_G": float(profile.get("estimated_total_sops", math.nan)) / 1e9,
        "firing": float(profile.get("global_firing_rate", math.nan)),
        "threshold_mean": health.get("threshold_mean", math.nan),
        "ternary_activity_mean": health.get("ternary_activity_mean", math.nan),
        "ternary_pos_neg_ratio": health.get("ternary_pos_neg_ratio", math.nan),
        "ternary_worst_pos_neg_ratio": health.get("ternary_worst_pos_neg_ratio", math.nan),
        "ternary_zero_neg_modules": health.get("ternary_zero_neg_modules", math.nan),
        "run_dir": str(run_dir),
    }


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "short_name",
        "epoch",
        "AEE",
        "AAE",
        "SOPs_G",
        "firing",
        "threshold_mean",
        "ternary_activity_mean",
        "ternary_pos_neg_ratio",
        "ternary_worst_pos_neg_ratio",
        "ternary_zero_neg_modules",
        "run_dir",
    ]
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# H40 Stage3 Priority 阶段式短训\n\n")
        handle.write("串行候选：`TX S02 A + D`、`SN S02 C + SB`、`SC S012 C + D`。每个候选 3 epoch，逐 epoch valid40。\n\n")
        handle.write("| 方案 | epoch | AEE | AAE | SOPs(G) | firing | threshold | ternary | pos/neg | worst | zero_neg |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['short_name']} | {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['SOPs_G']:.4f} | {row['firing']:.5f} | "
                f"{float(row['threshold_mean']):.4f} | {float(row['ternary_activity_mean']):.5f} | "
                f"{float(row['ternary_pos_neg_ratio']):.2f} | {float(row['ternary_worst_pos_neg_ratio']):.2f} | "
                f"{int(row['ternary_zero_neg_modules'])} |\n"
            )


def main() -> int:
    run_stamp = stamp()
    out_dir = RESULTS_DIR / f"h40_stage3_priority_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.log"
    rows: list[dict[str, Any]] = []
    with status_path.open("w", encoding="utf-8") as handle:
        handle.write(f"H40 stage3 priority started at {run_stamp}\n")
        handle.write("Parallel note: bs8 single run uses ~42GiB; this queue keeps official results serial for comparability.\n")

    for candidate in CANDIDATES:
        config_path = make_train_config(candidate, out_dir)
        run_dir = out_dir / "runs" / candidate["slug"]
        code = run_train(config_path, run_dir, status_path, candidate["short_name"])
        if code != 0:
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[STOP] {candidate['short_name']} train failed/aborted with code {code}\n")
            continue
        for epoch in range(3):
            checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
            if not checkpoint.exists():
                continue
            profile_dir = out_dir / "profiles" / f"{candidate['slug']}_epoch{epoch}_valid40"
            profile_code, profile = run_profile(config_path, checkpoint, profile_dir)
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[PROFILE] {candidate['short_name']} epoch{epoch} exit={profile_code}\n")
            if profile is not None:
                rows.append(collect_row(candidate, epoch, profile, run_dir, run_dir / "train.log"))
                write_summary(rows, out_dir)

    write_summary(rows, out_dir)
    print(f"stage3 priority dir: {out_dir}")
    print(f"status: {status_path}")
    print(f"summary: {out_dir / 'summary.md'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
