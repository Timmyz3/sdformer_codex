"""NTS-11 phase-5 autopilot: short screen → best → full30/finetune → valid825.

Post-11aah sweep over scope × recipe × attention × resume track.
Two-neuron deployment story unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = EXP_ROOT / "results"
CONFIG_DIR = EXP_ROOT / "configs"
RUNS_MD = EXP_ROOT / "RUNS.md"
MANIFEST = CONFIG_DIR / "generated/nts11_phase5_manifest.json"
MAKE_CONFIGS = EXP_ROOT / "entrypoints/make_nts11_phase5_configs.py"
TAG = "nts11_phase5_short"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_status(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], log_path: Path, *, check: bool = True) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[nts11-phase5] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): log={log_path}")
    return int(proc.returncode)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def short_score(row: dict[str, Any]) -> float:
    aee = float(row["AEE"])
    aae = float(row["AAE"])
    sops_g = float(row.get("SOPs_G", "inf"))
    if not all(math.isfinite(x) for x in (aee, aae, sops_g)):
        return math.inf
    return aee + 0.025 * aae + 0.15 * max(0.0, sops_g - 1.55)


def load_manifest() -> list[dict[str, Any]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def group_by_resume(manifest: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in manifest:
        groups.setdefault(str(item["resume"]), []).append(item)
    return groups


def run_short_screens(manifest: list[dict[str, Any]], driver_dir: Path, status: Path) -> Path:
    groups = group_by_resume(manifest)
    short_dirs: list[Path] = []
    for resume, items in groups.items():
        cmd = [
            sys.executable,
            "-u",
            str(EXP_ROOT / "entrypoints/rapid_screen.py"),
            "--steps",
            "1224",
            "--prev-runid",
            resume,
            "--batch-size",
            "8",
            "--workers",
            "8",
            "--prefetch-factor",
            "4",
            "--valid-samples",
            "10",
            "--confirm-steps",
            "1224",
            "--promote-samples",
            "40",
            "--no-promote-valid40",
            "--tag",
            TAG,
        ]
        for item in items:
            cmd.extend(["--config", str(item["config"])])
        wave = Path(resume).stem
        append_status(status, f"rapid screen wave={wave} n={len(items)}")
        run(cmd, driver_dir / f"rapid_screen_{wave}.log")
        dirs = sorted(RESULTS_DIR.glob(f"{TAG}_20*"), key=lambda p: p.stat().st_mtime)
        short_dirs.append(dirs[-1])
        append_status(status, f"rapid screen wave done dir={dirs[-1]}")
    combined = driver_dir / "short_combined"
    combined.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for short_dir in short_dirs:
        csv_path = short_dir / "summary.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stem = str(row.get("name", "")).replace("_steps1224", "")
                match = next((m for m in manifest if m["name"] == stem or stem.startswith(m["name"])), None)
                if match:
                    row["resume"] = match["resume"]
                    row["track"] = match["track"]
                    row["full_epochs"] = str(match["full_epochs"])
                    row["scope_policy"] = match["scope_policy"]
                    row["recipe"] = match["recipe"]
                rows.append(row)
    fields = list(rows[0].keys()) if rows else []
    out_csv = combined / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    append_status(status, f"combined short rows={len(rows)} csv={out_csv}")
    return combined


def pick_best(manifest: list[dict[str, Any]], combined_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    csv_path = combined_dir / "summary.csv"
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                aee = float(row["AEE"])
                aae = float(row["AAE"])
            except (KeyError, ValueError):
                continue
            if not math.isfinite(aee) or not math.isfinite(aae):
                continue
            stem = str(row.get("name", "")).replace("_steps1224", "")
            match = next((m for m in manifest if m["name"] == stem or stem.startswith(m["name"])), None)
            if not match:
                continue
            item = dict(row)
            item.update(match)
            rows.append(item)
    if not rows:
        raise RuntimeError("no valid short rows")
    best = min(rows, key=short_score)
    meta = next(m for m in manifest if m["name"] == best["name"])
    return best, meta


def make_full_config(short_config: Path, meta: dict[str, Any], out_config: Path) -> None:
    cfg = deepcopy(load_yaml(short_config))
    cfg["experiment"] = meta["name"] + "_full"
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = int(meta["full_epochs"])
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    n = int(meta["full_epochs"])
    if meta["track"] == "finetune":
        runtime["force_save_epochs"] = sorted({0, min(1, n - 1), n - 1})
    else:
        runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    optimizer = cfg.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    if meta["track"] == "finetune":
        optimizer["milestones"] = [max(1, n - 2)]
    else:
        optimizer["milestones"] = [20, 25]
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    cfg["note"] = str(cfg.get("note", "")) + f"\nPhase-5 autopilot winner ({meta['track']}) from short screen."
    dump_yaml(out_config, cfg)


def standard_eval(config: Path, run_dir: Path, epochs: list[int], status: Path) -> list[dict[str, Any]]:
    cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/run_h9_standard_valid825_eval.py"),
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
    ]
    for epoch in epochs:
        cmd.extend(["--epoch", str(epoch)])
    append_status(status, f"standard valid825 epochs={epochs}")
    run(cmd, run_dir / "standard_valid825_launch.log")
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        profile = run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
        if not profile.exists():
            continue
        data = json.loads(profile.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        rows.append(
            {
                "epoch": epoch,
                "AEE": float(metrics.get("AEE", math.nan)),
                "AAE": float(metrics.get("AAE", math.nan)),
                "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
                "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
                "energy_uj": float(data.get("energy_uj", 0.0) or 0.0),
            }
        )
    return rows


def append_runs_md(
    driver_dir: Path,
    best: dict[str, Any],
    meta: dict[str, Any],
    full_config: Path,
    run_dir: Path,
    eval_rows: list[dict[str, Any]],
) -> None:
    block = [
        "",
        "## NTS-11 Phase-5 短测 → 全量（自动追加）",
        "",
        f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 驱动目录：`{driver_dir}`",
        f"- 短测最优：`{meta['name']}`（valid10 AEE `{float(best['AEE']):.4f}`、track `{meta['track']}`、resume `{Path(meta['resume']).name}`）",
        f"- scope：`{meta['scope_policy']}` | recipe：`{meta['recipe']}`",
        f"- 全量配置：`{full_config}`",
        f"- 全量目录：`{run_dir}`",
        "",
        "### valid825",
        "",
        "| epoch | AEE | AAE | total_spikes(G) | firing | energy_uj |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_rows:
        block.append(
            f"| {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
            f"{row['spikes_g']:.4f} | {row['firing'] * 100:.4f}% | {row['energy_uj']:.2f} |"
        )
    if eval_rows:
        winner = min(eval_rows, key=lambda r: r["AEE"])
        block.append(f"\nPhase-5 当前最优：epoch{winner['epoch']} AEE `{winner['AEE']:.4f}`。\n")
    with RUNS_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))


def main() -> int:
    driver_dir = RESULTS_DIR / f"nts11_phase5_autopilot_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"driver_dir={driver_dir}")

    run([sys.executable, str(MAKE_CONFIGS)], driver_dir / "make_configs.log")
    manifest = load_manifest()
    append_status(status, f"manifest entries={len(manifest)}")

    combined = run_short_screens(manifest, driver_dir, status)
    best_row, meta = pick_best(manifest, combined)
    append_status(
        status,
        f"best={meta['name']} AEE={float(best_row['AEE']):.4f} track={meta['track']}",
    )

    short_config = Path(meta["config"])
    full_stamp = stamp()
    full_stem = f"{meta['name']}_full_{full_stamp}"
    full_config = CONFIG_DIR / f"{full_stem}.yml"
    run_dir = RESULTS_DIR / f"{full_stem}_bs8_{full_stamp}_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    make_full_config(short_config, meta, full_config)
    (run_dir / "launch.txt").write_text(
        f"config={full_config}\nresume={meta['resume']}\ntrack={meta['track']}\n",
        encoding="utf-8",
    )

    train_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(full_config),
        "--prev_runid",
        str(meta["resume"]),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    append_status(status, "full training start")
    run(train_cmd, run_dir / "train.log")
    append_status(status, "full training done")

    n = int(meta["full_epochs"])
    if meta["track"] == "finetune":
        eval_epochs = sorted({0, min(1, n - 1), n - 1})
    else:
        eval_epochs = [9, 14, 19, 24, 28, 29]
    eval_rows = standard_eval(full_config, run_dir, eval_epochs, status)
    append_runs_md(driver_dir, best_row, meta, full_config, run_dir, eval_rows)
    append_status(status, "Phase-5 autopilot complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())