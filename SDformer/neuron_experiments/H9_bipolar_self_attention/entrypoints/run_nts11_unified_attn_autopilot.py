"""NTS-11bd autopilot: unified h60 attention sweep → winner → full30 → valid825."""

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
MANIFEST = CONFIG_DIR / "generated/nts11bd_unified_attn_manifest.json"
MAKE_CONFIGS = EXP_ROOT / "entrypoints/make_nts11bd_unified_attn_sweep_configs.py"
VERIFY = EXP_ROOT / "entrypoints/verify_nts11_chain.py"
TAG = "nts11bd_unified_short"
PY = os.environ.get("UNIFIED_ATTN_PYTHON", "/opt/conda/bin/python3")


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
        handle.write(f"\n[nts11bd-unified] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): log={log_path}")
    return int(proc.returncode)


def short_score(row: dict[str, Any]) -> float:
    aee = float(row["AEE"])
    aae = float(row["AAE"])
    sops_g = float(row.get("SOPs_G", row.get("total_spikes_G", "inf")))
    if not all(math.isfinite(x) for x in (aee, aae, sops_g)):
        return math.inf
    return aee + 0.025 * aae + 0.12 * max(0.0, sops_g - 1.55)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_manifest() -> list[dict[str, Any]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def run_short_screens(manifest: list[dict[str, Any]], driver_dir: Path, status: Path) -> Path:
    cmd = [
        PY,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--steps",
        "1224",
        "--prev-runid",
        str(manifest[0]["resume"]),
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
    for item in manifest:
        cmd.extend(["--config", str(item["config"])])
    append_status(status, f"rapid screen n={len(manifest)} tag={TAG}")
    run(cmd, driver_dir / "rapid_screen.log")
    dirs = sorted(RESULTS_DIR.glob(f"{TAG}_20*"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(f"no short screen dir for tag={TAG}")
    short_dir = dirs[-1]
    append_status(status, f"rapid screen done dir={short_dir}")
    return short_dir


def pick_best(manifest: list[dict[str, Any]], short_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    csv_path = short_dir / "summary.csv"
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
            stem = str(row.get("name", "")).replace("_steps1224", "").replace("_s1224", "")
            match = next((m for m in manifest if m["name"] == stem or stem.startswith(m["name"])), None)
            if not match:
                continue
            item = dict(row)
            item.update(match)
            rows.append(item)
    if not rows:
        raise RuntimeError(f"no valid short rows in {csv_path}")
    best = min(rows, key=short_score)
    meta = next(m for m in manifest if m["name"] == best["name"])
    return best, meta


def make_full_config(short_config: Path, meta: dict[str, Any], out_config: Path) -> None:
    cfg = deepcopy(load_yaml(short_config))
    cfg["experiment"] = meta["name"] + "_full30"
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = int(meta["full_epochs"])
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    optimizer = cfg.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    optimizer["milestones"] = [20, 25]
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    cfg["note"] = (
        str(cfg.get("note", ""))
        + "\n11bd unified-attn autopilot winner: h60 all12 blocks, two-neuron scope."
    )
    dump_yaml(out_config, cfg)


def standard_eval(config: Path, run_dir: Path, epochs: list[int], status: Path) -> list[dict[str, Any]]:
    cmd = [
        PY,
        "-u",
        str(EXP_ROOT / "entrypoints/run_h9_standard_valid825_eval.py"),
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
    ]
    for epoch in epochs:
        cmd.extend(["--epoch", str(epoch)])
    append_status(status, f"valid825 epochs={epochs}")
    run(cmd, run_dir / "standard_valid825_launch.log", check=False)
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        profile = run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
        if not profile.exists():
            continue
        data = json.loads(profile.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        dense = float(data.get("dense_flops", 0.0) or 0.0)
        effective = float(data.get("effective_flops", 0.0) or 0.0)
        rows.append(
            {
                "epoch": epoch,
                "AEE": float(metrics.get("AEE", math.nan)),
                "AAE": float(metrics.get("AAE", math.nan)),
                "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
                "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
                "effective_g": effective / 1e9,
                "sparsity": 1.0 - effective / dense if dense else 0.0,
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
        "## NTS-11bd 统一注意力 短测 → 全量（自动追加）",
        "",
        f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 驱动目录：`{driver_dir}`",
        f"- 注意力：**h60 Shiftmax 全 12 block**（S0–S3 统一，无 Legacy）",
        f"- 短测最优：`{meta['name']}`（valid10 AEE `{float(best['AEE']):.4f}`）",
        f"- scope：`{meta['scope_policy']}` | recipe：`{meta['recipe']}` | resume NB0",
        f"- 全量配置：`{full_config}`",
        f"- 全量目录：`{run_dir}`",
        "",
        "### valid825",
        "",
        "| epoch | AEE | AAE | spikes(G) | firing | effective(G) | sparsity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_rows:
        block.append(
            f"| {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
            f"{row['spikes_g']:.4f} | {row['firing'] * 100:.4f}% | "
            f"{row['effective_g']:.2f} | {row['sparsity'] * 100:.2f}% |"
        )
    if eval_rows:
        winner = min(eval_rows, key=lambda r: r["AEE"])
        block.append(
            f"\n11bd 当前最优：epoch{winner['epoch']} AEE `{winner['AEE']:.4f}` "
            f"effective `{winner['effective_g']:.2f}G`。\n"
        )
    with RUNS_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))


def main() -> int:
    driver_dir = RESULTS_DIR / f"nts11bd_unified_autopilot_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"driver={driver_dir} python={PY}")

    append_status(status, "step1: generate 11bd unified-attn sweep configs")
    run([PY, str(MAKE_CONFIGS)], driver_dir / "make_configs.log")

    manifest = load_manifest()
    append_status(status, f"manifest entries={len(manifest)}")

    append_status(status, f"step2: verify chain ({Path(manifest[0]['config']).name})")
    run([PY, str(VERIFY), str(manifest[0]["config"])], driver_dir / "verify.log")

    short_dir = run_short_screens(manifest, driver_dir, status)
    best_row, meta = pick_best(manifest, short_dir)
    append_status(
        status,
        f"step3 winner: {meta['name']} score={short_score(best_row):.4f} "
        f"AEE={float(best_row['AEE']):.4f} scope={meta['scope_policy']} recipe={meta['recipe']}",
    )

    short_config = Path(meta["config"])
    full_stamp = stamp()
    full_stem = f"{meta['name']}_full30_{full_stamp}"
    full_config = CONFIG_DIR / f"{full_stem}.yml"
    run_dir = RESULTS_DIR / f"{full_stem}_bs8_{full_stamp}_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    make_full_config(short_config, meta, full_config)
    (driver_dir / "full_run_dir.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    append_status(status, f"step4 full30 -> {run_dir}")

    run(
        [
            PY,
            "-u",
            str(EXP_ROOT / "entrypoints/train.py"),
            "--config",
            str(full_config),
            "--prev_runid",
            str(meta["resume"]),
            "--save_path",
            str(run_dir / "checkpoint_epoch{}.pth"),
        ],
        run_dir / "train.log",
        check=False,
    )

    ckpts = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    ckpts = [p for p in ckpts if "state_dict" not in p.name]
    if not ckpts:
        append_status(status, "step4: no checkpoints — stopping before valid825")
        return 1

    eval_epochs = [ep for ep in (9, 14, 19, 24, 28, 29) if (run_dir / f"checkpoint_epoch{ep}.pth").is_file()]
    if not eval_epochs:
        last = int(ckpts[-1].stem.replace("checkpoint_epoch", ""))
        eval_epochs = [last]

    eval_rows = standard_eval(full_config, run_dir, eval_epochs, status)
    append_runs_md(driver_dir, best_row, meta, full_config, run_dir, eval_rows)
    append_status(status, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())