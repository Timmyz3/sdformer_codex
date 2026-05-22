"""Continue H40 automatically, then promote one candidate to full training.

Workflow:
1. wait for the running H40 stage3 priority queue to finish;
2. read its valid40 summary;
3. run a small LR/sparsity micro-sweep around the best families;
4. pick one candidate and launch a 30-epoch full run with every epoch saved;
5. profile each saved checkpoint with valid40 so intermediate epochs can be
   compared without rerunning training.

This entrypoint is intentionally outside third_party/SDformerFlow.
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
GENERATED_DIR = CONFIG_DIR / "generated"
RESULTS_DIR = EXP_ROOT / "results"
REVIEWS_DIR = EXP_ROOT / "reviews"
BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"

BASELINE_AEE = 1.58
H9A_AAE = 7.64
TARGET_SOPS_G = 3.10


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


def run_command(command: list[str], log_path: Path, *, cwd: Path = REPO_ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=cwd, env=env(), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[h41-autopilot] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {text.rstrip()}\n")


def read_h40_rows(summary_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for key in (
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
            ):
                parsed[key] = float(parsed[key])
            parsed["epoch"] = int(parsed["epoch"])
            rows.append(parsed)
    return rows


def score_row(row: dict[str, Any]) -> float:
    score = float(row["AEE"]) + 0.03 * float(row["AAE"])
    score += 0.35 * max(0.0, float(row["SOPs_G"]) - TARGET_SOPS_G)
    score += 0.90 * max(0.0, float(row["AEE"]) - BASELINE_AEE)
    score += 0.05 * max(0.0, float(row["AAE"]) - H9A_AAE)
    score += 0.25 * max(0.0, 0.015 - float(row["ternary_activity_mean"]))
    score += 0.03 * max(0.0, float(row["ternary_zero_neg_modules"]) - 2.0)
    if float(row["SOPs_G"]) <= 3.0:
        score -= 0.06
    if row["epoch"] == 0:
        # We prefer early-stop style candidates when later epochs drift.
        score -= 0.03
    return score


def best_h40_family(rows: list[dict[str, Any]]) -> str:
    healthy = [
        row
        for row in rows
        if row["SOPs_G"] <= 3.20
        and row["AEE"] <= 1.90
        and row["AAE"] <= 8.90
        and row["ternary_activity_mean"] >= 0.015
        and row["ternary_zero_neg_modules"] <= 2
    ]
    if not healthy:
        healthy = rows
    return min(healthy, key=score_row)["short_name"]


def set_deep(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    cursor: Any = mapping
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def make_micro_config(source_name: str, variant: str, changes: dict[str, Any], run_stamp: str) -> str:
    cfg = deepcopy(read_yaml(GENERATED_DIR / source_name))
    experiment = f"h41_{variant}_{run_stamp}"
    cfg["experiment"] = experiment
    cfg.setdefault("loss", {})["use_angular_loss"] = False
    cfg.setdefault("loss", {})["lambda_ang"] = 0
    cfg.setdefault("optimizer", {})["use_amp"] = True
    for key, value in changes.items():
        set_deep(cfg, key, value)
    cfg["note"] = (
        "H41 micro-sweep：围绕 H40 valid40 排名靠前方案做学习率/稀疏强度微调。"
        f"source={source_name}; variant={variant}; changes={changes}"
    )
    out = GENERATED_DIR / f"{experiment}.yml"
    write_yaml(out, cfg)
    return f"generated/{out.name}"


def micro_candidates(best_family: str, run_stamp: str) -> list[str]:
    # A/C presets are kept canonical; the changes below only tune LR and
    # threshold dynamics, not the replacement topology.
    candidates = [
        make_micro_config(
            "h40_p3_TXS02_A.yml",
            "txs02a_slowbb_softtheta",
            {
                "optimizer.param_groups.backbone_lr": 2.0e-7,
                "optimizer.param_groups.norm_lr": 2.0e-7,
                "optimizer.param_groups.neuron_lr": 1.2e-5,
                "optimizer.param_groups.threshold_lr": 2.0e-6,
                "atlif_ternary_psn.threshold_base_lr": 2.0e-6,
                "atlif_ternary_psn.threshold_eta": 0.0006,
                "atlif_ternary_psn.target_rate_eta": 0.05,
            },
            run_stamp,
        ),
        make_micro_config(
            "h40_p3_TXS02_C.yml",
            "txs02c_dlr",
            {},
            run_stamp,
        ),
        make_micro_config(
            "h40_p3_TXS02_C.yml",
            "txs02c_slowbb",
            {
                "optimizer.param_groups.backbone_lr": 2.0e-7,
                "optimizer.param_groups.norm_lr": 2.0e-7,
                "optimizer.param_groups.neuron_lr": 1.2e-5,
                "optimizer.param_groups.threshold_lr": 3.0e-6,
                "atlif_ternary_psn.threshold_base_lr": 3.0e-6,
            },
            run_stamp,
        ),
        make_micro_config(
            "h40_p3_SNS02_C.yml",
            "sns02c_dlr",
            {},
            run_stamp,
        ),
        make_micro_config(
            "h40_p3_SCS012_C.yml",
            "scs012c_slowbb",
            {
                "optimizer.param_groups.backbone_lr": 2.0e-7,
                "optimizer.param_groups.norm_lr": 2.0e-7,
                "optimizer.param_groups.neuron_lr": 1.2e-5,
                "optimizer.param_groups.threshold_lr": 3.0e-6,
                "atlif_ternary_psn.threshold_base_lr": 3.0e-6,
            },
            run_stamp,
        ),
    ]
    if best_family.startswith("SC"):
        candidates.insert(
            0,
            make_micro_config(
                "h40_p3_SCS012_A.yml",
                "scs012a_softtheta",
                {
                    "atlif_ternary_psn.threshold_eta": 0.0006,
                    "atlif_ternary_psn.target_rate_eta": 0.05,
                    "optimizer.param_groups.threshold_lr": 2.0e-6,
                    "atlif_ternary_psn.threshold_base_lr": 2.0e-6,
                },
                run_stamp,
            ),
        )
    return candidates


def run_rapid_valid40(configs: list[str], run_stamp: str, log_dir: Path) -> Path:
    tag = f"h41_micro360_valid40_{run_stamp}"
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--tag",
        tag,
        "--steps",
        "360",
        "--valid-samples",
        "40",
        "--batch-size",
        "8",
        "--workers",
        "8",
        "--prefetch-factor",
        "4",
        "--pin-memory",
        "--amp",
        "--no-promote-valid40",
        "--promote-aee",
        "1.90",
        "--promote-aae",
        "8.90",
        "--promote-sops-g",
        "3.25",
        "--max-zero-neg-modules",
        "4",
        "--max-worst-pos-neg-ratio",
        "80",
    ]
    for config in configs:
        command.extend(["--config", config])
    code = run_command(command, log_dir / f"{tag}.log")
    if code != 0:
        raise RuntimeError(f"H41 micro rapid_screen failed: {log_dir / (tag + '.log')}")
    summaries = sorted(RESULTS_DIR.glob(f"{tag}_*/summary.csv"), key=lambda path: path.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"No H41 summary for tag {tag}")
    return summaries[-1]


def read_rapid_rows(summary_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for key in (
                "AEE",
                "AAE",
                "SOPs_G",
                "firing",
                "threshold_mean",
                "ternary_activity_mean",
                "ternary_pos_neg_ratio",
                "ternary_worst_pos_neg_ratio",
                "ternary_zero_neg_modules",
                "score",
            ):
                parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def config_from_rapid_name(name: str) -> Path:
    base = name
    if base.endswith("_valid40"):
        base = base.removesuffix("_valid40")
    for suffix in ("_steps360", "_steps160", "_steps120"):
        if base.endswith(suffix):
            base = base.removesuffix(suffix)
    path = GENERATED_DIR / f"{base}.yml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def choose_rapid_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("gate") == "pass"
        and int(float(row.get("samples", 0))) >= 40
        and row["AEE"] <= 1.90
        and row["AAE"] <= 8.90
        and row["SOPs_G"] <= 3.25
        and row["ternary_activity_mean"] >= 0.015
        and row["ternary_zero_neg_modules"] <= 4
    ]
    if not eligible:
        eligible = [
            row
            for row in rows
            if int(float(row.get("samples", 0))) >= 40
            and row["SOPs_G"] <= 3.30
            and row["ternary_activity_mean"] >= 0.015
        ]
    if not eligible:
        eligible = rows
    return min(eligible, key=lambda row: float(row["score"]))


def make_full_config(base_config: Path, chosen: dict[str, Any], run_stamp: str) -> Path:
    cfg = deepcopy(read_yaml(base_config))
    full_name = f"{base_config.stem}_h41_full_{run_stamp}"
    cfg["experiment"] = full_name
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = list(range(30))
    runtime["use_mlflow_model_logging"] = False
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True
    cfg.setdefault("optimizer", {})["use_amp"] = True
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    cfg["note"] = (
        "H41 自动推进全量：从 H40+H41 valid40 短训筛选后选择。"
        f"valid40 AEE={chosen['AEE']:.4f}, AAE={chosen['AAE']:.4f}, "
        f"SOPs={chosen['SOPs_G']:.4f}G, firing={chosen['firing']:.5f}, "
        "全量 30 epoch，每个 epoch 保存 checkpoint 便于中途推理。"
    )
    out = CONFIG_DIR / f"{full_name}.yml"
    write_yaml(out, cfg)
    return out


def write_review(h40_rows: list[dict[str, Any]], micro_summary: Path, chosen: dict[str, Any], full_config: Path, run_stamp: str) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    review = REVIEWS_DIR / f"FULL_REVIEW_{full_config.stem}_{run_stamp}.md"
    best_h40 = min(h40_rows, key=score_row)
    text = f"""# H41 全量训练前 Review：{full_config.stem}

## 选择结论

- 全量配置：`{full_config.relative_to(REPO_ROOT)}`
- 选择短测：`{chosen['name']}`
- H41 valid40：AEE={chosen['AEE']:.4f}, AAE={chosen['AAE']:.4f}, SOPs={chosen['SOPs_G']:.4f}G, firing={chosen['firing']:.5f}
- H40 最优参考：{best_h40['short_name']} epoch{best_h40['epoch']}，AEE={best_h40['AEE']:.4f}, AAE={best_h40['AAE']:.4f}, SOPs={best_h40['SOPs_G']:.4f}G
- H41 短测汇总：`{micro_summary.relative_to(REPO_ROOT)}`

## 范式检查

- 神经元主线仍为 PSN + ATLIF：Q/K 是三值 PSN+ATLIF，高 SOP FFN 目标层是二值 PSN+official ATLIF。
- 没有改 `third_party/SDformerFlow`；训练入口仍通过 H9 实验 overlay 调用 baseline 训练逻辑。
- 没有启用 angular loss，因为它在前序 H9/i14 与 QK/compat 类注意力上持续恶化 AAE。
- 本轮重点微调学习率和 ATLIF 阈值动态，避免 H40 的 `TX S02 A` 在 epoch1/2 继续训练后漂移变差。

## 风险和监控

- 全量 30 epoch 每轮保存，后续可直接 profile epoch0...29；若早期 epoch 最好，可做 early-stop 叙事。
- 若三值负发放塌缩，优先检查 `ternary_zero_neg_modules`、`ternary_pos_neg_ratio` 和 `ternary_activity_mean`。
- 若 SOPs 低但 AAE 变差，下一轮应降低 target-rate/threshold 增长强度，而不是继续扩大替换范围。
"""
    review.write_text(text, encoding="utf-8")
    return review


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    return checkpoints[-1] if checkpoints else None


def run_full(full_config: Path, run_stamp: str, log_dir: Path) -> Path:
    run_dir = RESULTS_DIR / f"{full_config.stem}_bs8_{run_stamp}_setsid"
    train_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(full_config),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    code = run_command(train_cmd, run_dir / "train.log")
    if code != 0:
        raise RuntimeError(f"full training failed: {run_dir / 'train.log'}")
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    profile_root = run_dir / "profile_valid40_by_epoch"
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        epoch_text = checkpoint.stem.replace("checkpoint_epoch", "")
        profile_dir = profile_root / f"epoch{epoch_text}"
        profile_cmd = [
            sys.executable,
            "-u",
            str(EXP_ROOT / "entrypoints/profile_sops.py"),
            "--config",
            str(full_config),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(profile_dir),
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
        code = run_command(profile_cmd, profile_dir / "profile.log")
        summary = profile_dir / "sops_summary.json"
        if code == 0 and summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            rows.append(
                {
                    "epoch": int(epoch_text),
                    "AEE": float(metrics.get("AEE", math.nan)),
                    "AAE": float(metrics.get("AAE", math.nan)),
                    "SOPs_G": float(data.get("estimated_total_sops", math.nan)) / 1e9,
                    "firing": float(data.get("global_firing_rate", math.nan)),
                }
            )
        write_full_profile_summary(rows, run_dir)
    return run_dir


def write_full_profile_summary(rows: list[dict[str, Any]], run_dir: Path) -> None:
    csv_path = run_dir / "profile_valid40_summary.csv"
    md_path = run_dir / "profile_valid40_summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "AEE", "AAE", "SOPs_G", "firing"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["epoch"]))
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# H41 全量逐 epoch valid40 推理汇总\n\n")
        handle.write("| epoch | AEE | AAE | SOPs(G) | firing |\n")
        handle.write("|---:|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda item: item["epoch"]):
            handle.write(
                f"| {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['SOPs_G']:.4f} | {row['firing']:.5f} |\n"
            )


def wait_for_h40(h40_dir: Path, status_log: Path, max_wait_hours: float) -> None:
    deadline = time.time() + max_wait_hours * 3600
    summary = h40_dir / "summary.csv"
    while time.time() < deadline:
        running = subprocess.run(
            ["bash", "-lc", "ps -eo cmd | rg 'run_h40_stage3_priority.py' | rg -v rg >/dev/null"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if summary.exists() and running.returncode != 0:
            return
        append_log(status_log, "等待 H40 stage3 priority 队列结束...")
        time.sleep(60)
    raise TimeoutError(f"H40 did not finish within {max_wait_hours} hours")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h40-dir", type=Path, required=True)
    parser.add_argument("--max-wait-hours", type=float, default=8.0)
    args = parser.parse_args(argv)

    run_stamp = stamp()
    log_dir = RESULTS_DIR / f"h41_after_h40_autopilot_{run_stamp}"
    status_log = log_dir / "status.log"
    append_log(status_log, f"H41 autopilot started; h40_dir={args.h40_dir}")

    wait_for_h40(args.h40_dir, status_log, args.max_wait_hours)
    h40_summary = args.h40_dir / "summary.csv"
    h40_rows = read_h40_rows(h40_summary)
    if not h40_rows:
        raise RuntimeError(f"No H40 rows in {h40_summary}")

    family = best_h40_family(h40_rows)
    append_log(status_log, f"H40 最优族={family}; 开始 H41 微调短测")
    configs = micro_candidates(family, run_stamp)
    append_log(status_log, "H41 configs: " + ", ".join(configs))
    micro_summary = run_rapid_valid40(configs, run_stamp, log_dir)
    rapid_rows = read_rapid_rows(micro_summary)
    chosen = choose_rapid_row(rapid_rows)
    base_config = config_from_rapid_name(chosen["name"])
    full_config = make_full_config(base_config, chosen, run_stamp)
    review = write_review(h40_rows, micro_summary, chosen, full_config, run_stamp)
    append_log(status_log, f"选择 full={full_config}; review={review}")
    run_dir = run_full(full_config, run_stamp, log_dir)
    append_log(status_log, f"H41 full 完成：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
