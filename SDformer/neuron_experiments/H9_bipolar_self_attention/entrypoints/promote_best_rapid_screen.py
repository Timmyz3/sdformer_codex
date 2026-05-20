"""从一组 rapid_screen 结果里自动选择一个候选并启动全量训练。

选择原则是面向当前论文故事的：精度不能炸，AAE 要守住，同时 SOPs
越接近/低于 H9a 越好。若没有候选完全达标，也会选综合分最低的一个
继续全量训练，避免 GPU 空转。
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

H9A_AEE = 1.5044
H9A_AAE = 7.6365
H9A_SOPS_G = 3.0847


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def run_command(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[promote-full] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def latest_dirs_for_tag(tag: str) -> list[Path]:
    return sorted(RESULTS_DIR.glob(f"{tag}_20*"), key=lambda path: path.stat().st_mtime)


def read_rows(root: Path) -> list[dict[str, Any]]:
    csv_path = root / "summary.csv"
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                samples = int(float(row.get("samples", "0")))
                aee = float(row.get("AEE", "inf"))
                aae = float(row.get("AAE", "inf"))
                sops_g = float(row.get("SOPs_G", "inf"))
                firing = float(row.get("firing", "inf"))
            except ValueError:
                continue
            if samples < 40 or not all(math.isfinite(x) for x in (aee, aae, sops_g, firing)):
                continue
            item = dict(row)
            item.update({"root": str(root), "samples": samples, "AEE": aee, "AAE": aae, "SOPs_G": sops_g, "firing": firing})
            rows.append(item)
    return rows


def candidate_score(row: dict[str, Any]) -> float:
    aee = float(row["AEE"])
    aae = float(row["AAE"])
    sops_g = float(row["SOPs_G"])
    score = aee + 0.025 * aae + 0.28 * max(0.0, sops_g - H9A_SOPS_G)
    score += 0.80 * max(0.0, aee - 1.58)
    score += 0.08 * max(0.0, aae - 7.90)
    score += 0.20 * max(0.0, sops_g - 3.50)
    if sops_g <= 3.25:
        score -= 0.05
    if aee <= H9A_AEE and aae <= H9A_AAE:
        score -= 0.04
    return score


def generated_config_for_row(row: dict[str, Any]) -> Path:
    root = Path(str(row["root"]))
    name = str(row["name"])
    if name.endswith("_valid40"):
        name = name[: -len("_valid40")]
    return root / "configs" / f"{name}.yml"


def make_full_config(short_config: Path, out_config: Path, experiment: str, args: argparse.Namespace) -> None:
    cfg = deepcopy(load_yaml(short_config))
    cfg["experiment"] = experiment
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [9, 19, 29]
    runtime["use_mlflow_model_logging"] = False
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = args.epochs
    loader["batch_size"] = args.batch_size
    loader["n_workers"] = args.workers
    loader["persistent_workers"] = args.workers > 0
    loader["prefetch_factor"] = args.prefetch_factor
    loader["pin_memory"] = args.pin_memory
    loader["non_blocking"] = True
    optimizer = cfg.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    optimizer["milestones"] = [20, 25]
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    cfg["note"] = str(cfg.get("note", "")) + "\n自动 promotion：由 rapid_screen valid40 综合分选出后全量续训。"
    dump_yaml(out_config, cfg)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    return checkpoints[-1] if checkpoints else None


def checkpoints_for_profile(run_dir: Path, preferred_epochs: list[int]) -> list[Path]:
    selected: list[Path] = []
    for epoch in preferred_epochs:
        checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
        if checkpoint.exists():
            selected.append(checkpoint)
    latest = latest_checkpoint(run_dir)
    if latest is not None and latest not in selected:
        selected.append(latest)
    return selected


def profile_checkpoint(config_path: Path, checkpoint: Path, out_dir: Path, samples: int) -> dict[str, Any] | None:
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
        "1",
        "--num-workers",
        "4",
        "--metric",
        "AEE",
        "--metric",
        "AAE",
    ]
    exit_code = run_command(command, out_dir / "profile.log")
    summary_path = out_dir / "sops_summary.json"
    if exit_code != 0 or not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def compact_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("metrics", {})
    return {
        "AEE": float(metrics.get("AEE", math.inf)),
        "AAE": float(metrics.get("AAE", math.inf)),
        "SOPs_G": float(summary.get("estimated_total_sops", math.inf)) / 1.0e9,
        "firing": float(summary.get("global_firing_rate", math.inf)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", action="append", required=True, help="rapid_screen tag prefix to include.")
    parser.add_argument("--prev-runid", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--profile-samples", type=int, default=40)
    parser.add_argument(
        "--profile-epoch",
        action="append",
        type=int,
        default=[],
        help="Epoch checkpoint to profile after full training. Repeatable. Defaults to runtime.force_save_epochs plus latest.",
    )
    parser.add_argument("--log", type=Path, default=RESULTS_DIR / f"promote_best_rapid_screen_{now_stamp()}.log")
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Substring of summary row name to exclude from promotion. Repeatable.",
    )
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as log:
        def record(message: str) -> None:
            line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        rows: list[dict[str, Any]] = []
        for tag in args.tag:
            dirs = latest_dirs_for_tag(tag)
            if not dirs:
                record(f"没有找到 tag={tag} 的 rapid_screen 结果")
                continue
            root = dirs[-1]
            tag_rows = read_rows(root)
            record(f"读取 {root}: valid40 候选 {len(tag_rows)} 个")
            rows.extend(tag_rows)

        if not rows:
            record("没有 valid40 候选，无法 promotion")
            return 1

        if args.exclude_name:
            before = len(rows)
            rows = [
                row
                for row in rows
                if not any(excluded in str(row.get("name", "")) for excluded in args.exclude_name)
            ]
            record(f"按 exclude-name 过滤候选: {before} -> {len(rows)}")
            if not rows:
                record("过滤后没有 valid40 候选，无法 promotion")
                return 1

        rows = sorted(rows, key=candidate_score)
        best = rows[0]
        short_config = generated_config_for_row(best)
        if not short_config.exists():
            record(f"候选配置不存在: {short_config}")
            return 1

        full_stem = str(best["name"]).replace("_valid40", "").replace("_guard120_steps120", "_auto_full")
        if not full_stem.endswith("_auto_full"):
            full_stem += "_auto_full"
        stamp = now_stamp()
        out_config = CONFIG_DIR / f"{full_stem}_{stamp}.yml"
        run_dir = RESULTS_DIR / f"{full_stem}_bs{args.batch_size}_{stamp}_setsid"
        run_dir.mkdir(parents=True, exist_ok=True)
        make_full_config(short_config, out_config, full_stem, args)

        ranking_path = run_dir / "promotion_ranking.md"
        with ranking_path.open("w", encoding="utf-8") as handle:
            handle.write("# 自动全量候选排序\n\n")
            handle.write("| rank | name | AEE | AAE | SOPs(G) | firing | score | root |\n")
            handle.write("|---:|---|---:|---:|---:|---:|---:|---|\n")
            for rank, row in enumerate(rows, 1):
                handle.write(
                    f"| {rank} | {row['name']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                    f"{row['SOPs_G']:.4f} | {row['firing']:.5f} | {candidate_score(row):.4f} | "
                    f"`{row['root']}` |\n"
                )

        record(
            "选中全量候选: "
            f"{best['name']} AEE={best['AEE']:.4f} AAE={best['AAE']:.4f} "
            f"SOPs={best['SOPs_G']:.4f}G firing={best['firing']:.5f} score={candidate_score(best):.4f}"
        )
        record(f"full config: {out_config}")
        record(f"full run dir: {run_dir}")

        command = [
            sys.executable,
            "-u",
            str(EXP_ROOT / "entrypoints/train.py"),
            "--config",
            str(out_config),
            "--prev_runid",
            str(args.prev_runid.resolve()),
            "--save_path",
            str(run_dir / "checkpoint_epoch{}.pth"),
        ]
        (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        exit_code = run_command(command, run_dir / "train.log")
        (run_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        record(f"full training exit_code={exit_code}")
        checkpoint = latest_checkpoint(run_dir)
        if exit_code != 0 or checkpoint is None:
            record("full training 没有可用 checkpoint，停止后续推理")
            return exit_code or 1

        preferred_epochs = args.profile_epoch or list(load_yaml(out_config).get("runtime", {}).get("force_save_epochs", []) or [])
        profiled: list[tuple[Path, dict[str, float]]] = []
        for checkpoint in checkpoints_for_profile(run_dir, [int(item) for item in preferred_epochs]):
            summary = profile_checkpoint(
                out_config,
                checkpoint,
                RESULTS_DIR / f"profile_{full_stem}_{checkpoint.stem}_valid{args.profile_samples}_{stamp}",
                args.profile_samples,
            )
            if summary is None:
                record(f"{checkpoint.name} profile 失败，跳过")
                continue
            metrics = compact_metrics(summary)
            profiled.append((checkpoint, metrics))
            record(f"{checkpoint.name} valid{args.profile_samples}: {metrics}")

        if not profiled:
            record("full checkpoint profile 全部失败")
            return 1
        best_checkpoint, best_metrics = min(
            profiled,
            key=lambda item: item[1]["AEE"] + 0.025 * item[1]["AAE"] + 0.35 * max(0.0, item[1]["SOPs_G"] - H9A_SOPS_G),
        )
        best_path = run_dir / f"best_profile_valid{args.profile_samples}.md"
        with best_path.open("w", encoding="utf-8") as handle:
            handle.write("# Full Checkpoint Profile Ranking\n\n")
            handle.write("| rank | checkpoint | AEE | AAE | SOPs(G) | firing |\n")
            handle.write("|---:|---|---:|---:|---:|---:|\n")
            for rank, (ckpt, metrics) in enumerate(
                sorted(
                    profiled,
                    key=lambda item: item[1]["AEE"] + 0.025 * item[1]["AAE"] + 0.35 * max(0.0, item[1]["SOPs_G"] - H9A_SOPS_G),
                ),
                1,
            ):
                handle.write(
                    f"| {rank} | `{ckpt.name}` | {metrics['AEE']:.4f} | {metrics['AAE']:.4f} | "
                    f"{metrics['SOPs_G']:.4f} | {metrics['firing']:.5f} |\n"
                )
        record(f"best full checkpoint: {best_checkpoint.name} {best_metrics}; ranking={best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
