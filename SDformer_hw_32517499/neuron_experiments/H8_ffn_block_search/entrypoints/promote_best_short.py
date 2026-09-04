#!/usr/bin/env python
"""Promote effective H8 short probes to full-run configs and commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_summary(profile_dir: Path) -> dict[str, Any] | None:
    summary_file = profile_dir / "sops_summary.json"
    if not summary_file.exists():
        return None
    summary = json.loads(summary_file.read_text())
    return {
        "profile_dir": str(profile_dir),
        "firing": float(summary["global_firing_rate"]),
        "sops": float(summary["estimated_total_sops"]),
        "aee": float(summary["metrics"]["AEE"]),
        "aae": float(summary["metrics"].get("AAE", 0.0)),
    }


def _short_name_from_profile(profile_dir: Path, stamp: str) -> str:
    name = profile_dir.name
    prefix = "profile_"
    suffix = f"_valid10_{stamp}"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected profile dir name: {name}")
    return name[len(prefix) : -len(suffix)]


def _base_config_name(short_name: str) -> str:
    # Short-run output names often drop the trailing `_120` from config stems.
    return short_name if short_name.endswith("_120") else f"{short_name}_120"


def _collect(results_dir: Path, stamps: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stamp in stamps:
        for profile_dir in sorted(results_dir.glob(f"profile_*_valid10_{stamp}")):
            summary = _load_summary(profile_dir)
            if summary is None:
                continue
            short_name = _short_name_from_profile(profile_dir, stamp)
            summary.update(
                {
                    "stamp": stamp,
                    "short_name": short_name,
                    "config_stem": _base_config_name(short_name),
                }
            )
            rows.append(summary)
    return rows


def _make_full_config(short_cfg: Path, full_cfg: Path, experiment: str) -> None:
    cfg = yaml.safe_load(short_cfg.read_text())
    cfg["experiment"] = experiment
    cfg.setdefault("runtime", {})["max_train_steps"] = 0
    cfg.setdefault("runtime", {})["skip_state_save"] = True
    cfg.setdefault("loader", {})["n_epochs"] = 30
    cfg.setdefault("optimizer", {})["milestones"] = [20, 25]
    full_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _write_markdown(path: Path, rows: list[dict[str, Any]], promoted: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        f"# H8 Promotion Summary {args.stamp}",
        "",
        f"Selection thresholds: AEE <= {args.aee_max}, AAE <= {args.aae_max}, SOPs <= {args.sops_max_g}G.",
        "",
        "| run | stamp | status | SOPs | firing | AEE | AAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    promoted_names = {row["short_name"] for row in promoted}
    for row in rows:
        status = "promoted" if row["short_name"] in promoted_names else row["status"]
        lines.append(
            f"| `{row['short_name']}` | `{row['stamp']}` | {status} | "
            f"{row['sops'] / 1e9:.4f}G | {row['firing']:.6f} | {row['aee']:.6f} | {row['aae']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/root/private_data/work/sdformer_codex/SDformer")
    parser.add_argument("--exp", default="neuron_experiments/H8_ffn_block_search")
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--extra-stamp", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--aee-max", type=float, default=1.07)
    parser.add_argument("--aae-max", type=float, default=6.35)
    parser.add_argument("--sops-max-g", type=float, default=3.60)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    base = Path(args.base)
    exp = base / args.exp
    results_dir = exp / "results"
    configs_dir = exp / "configs"
    full_configs_dir = configs_dir / "generated_full"
    full_configs_dir.mkdir(parents=True, exist_ok=True)

    stamps = [args.stamp, *args.extra_stamp]
    rows = _collect(results_dir, stamps)
    for row in rows:
        row["status"] = "candidate"
        if row["aee"] > args.aee_max:
            row["status"] = "reject_aee"
        elif row["aae"] > args.aae_max:
            row["status"] = "reject_aae"
        elif row["sops"] > args.sops_max_g * 1e9:
            row["status"] = "reject_sops"

    candidates = [row for row in rows if row["status"] == "candidate"]
    candidates.sort(key=lambda row: (row["aee"], row["sops"], row["aae"]))
    promoted = candidates[: max(0, args.top_k)]

    commands = []
    ckpt = args.ckpt or str(base / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth")
    for row in promoted:
        short_cfg = configs_dir / f"{row['config_stem']}.yml"
        if not short_cfg.exists():
            row["status"] = "missing_config"
            continue
        full_stem = f"{row['config_stem']}_full_from_{args.stamp}"
        full_cfg = full_configs_dir / f"{full_stem}.yml"
        _make_full_config(short_cfg, full_cfg, f"{row['config_stem']}_full_from_{args.stamp}")
        full_run_dir = results_dir / f"{full_stem}_setsid"
        full_run_dir.mkdir(parents=True, exist_ok=True)
        command = (
            f"SDFORMER_USE_MLFLOW=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"/opt/conda/envs/sdformerflow/bin/python -u {exp / 'entrypoints/train.py'} "
            f"--config {full_cfg} "
            f"--prev_runid {ckpt} "
            f"--save_path {full_run_dir / 'checkpoint_epoch{}.pth'} "
            f"> {full_run_dir / 'train.log'} 2>&1"
        )
        (full_run_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")
        commands.append((row["short_name"], command, full_run_dir))

    summary_md = results_dir / f"promotion_summary_{args.stamp}.md"
    _write_markdown(summary_md, rows, promoted, args)
    summary_json = results_dir / f"promotion_summary_{args.stamp}.json"
    summary_json.write_text(json.dumps({"rows": rows, "promoted": promoted}, indent=2), encoding="utf-8")

    run_script = results_dir / f"run_promoted_full_{args.stamp}.sh"
    lines = ["#!/usr/bin/env bash", "set -u", f"echo '[H8 promote] summary: {summary_md}'"]
    for short_name, command, full_run_dir in commands:
        lines.extend(
            [
                f"echo '===== FULL TRAIN {short_name} ====='",
                f"mkdir -p {full_run_dir}",
                command,
            ]
        )
    if not commands:
        lines.append("echo '[H8 promote] no candidate satisfied thresholds; no full run launched'")
    run_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_script.chmod(0o755)

    print(f"summary_md={summary_md}")
    print(f"run_script={run_script}")
    print(f"promoted={','.join(row['short_name'] for row in promoted) if promoted else 'none'}")


if __name__ == "__main__":
    main()
