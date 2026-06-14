"""NTS-11 scope sweep: binary/ternary combinatorics short-test → full30 → valid825."""

from __future__ import annotations

import csv
import json
import math
import os
import re
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
BASELINE = REPO_ROOT / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
RUNS_MD = EXP_ROOT / "RUNS.md"
TAG = "nts11_scope_short"
MAKE_CONFIGS = EXP_ROOT / "entrypoints/make_nts11_phase4_scope_configs.py"
VERIFY = EXP_ROOT / "entrypoints/verify_nts11_chain.py"


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
        handle.write(f"\n[nts11-scope-autopilot] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}; log={log_path}")
    return int(proc.returncode)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def generated_configs() -> list[Path]:
    patterns = [
        "nts11q*_scope_*_s1224.yml",
        "nts11r*_scope_*_s1224.yml",
        "nts11s*_scope_*_s1224.yml",
        "nts11t*_scope_*_s1224.yml",
        "nts11u*_scope_*_s1224.yml",
        "nts11v*_scope_*_s1224.yml",
        "nts11w*_scope_*_s1224.yml",
        "nts11x*_scope_*_s1224.yml",
        "nts11y*_scope_*_s1224.yml",
        "nts11z*_scope_*_s1224.yml",
        "nts11aa*_scope_*_s1224.yml",
        "nts11ab*_scope_*_s1224.yml",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend((CONFIG_DIR / "generated").glob(pattern))
    return sorted(set(found))


def latest_short_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob(f"{TAG}_20*"), key=lambda item: item.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(f"no rapid screen dir for tag={TAG}")
    return dirs[-1]


def short_score(row: dict[str, Any]) -> float:
    aee = float(row["AEE"])
    aae = float(row["AAE"])
    sops_g = float(row.get("SOPs_G", "inf"))
    firing = float(row.get("firing", "inf"))
    if not all(math.isfinite(x) for x in (aee, aae, sops_g, firing)):
        return math.inf
    score = aee + 0.025 * aae + 0.15 * max(0.0, sops_g - 1.55)
    if "vanilla_decoder" in str(row.get("name", "")):
        score += 100.0
    return score


def pick_best_short_row(short_dir: Path) -> dict[str, Any]:
    csv_path = short_dir / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing summary.csv in {short_dir}")
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("name", ""))
            if "vanilla_decoder" in name or "vdec" in name:
                continue
            try:
                samples = int(float(row.get("samples", "0")))
                aee = float(row["AEE"])
                aae = float(row["AAE"])
            except (KeyError, ValueError):
                continue
            if samples < 10 or not math.isfinite(aee) or not math.isfinite(aae):
                continue
            item = dict(row)
            item["samples"] = samples
            item["AEE"] = aee
            item["AAE"] = aae
            item["root"] = str(short_dir)
            rows.append(item)
    if not rows:
        raise RuntimeError(f"no valid short rows in {csv_path}")
    return min(rows, key=short_score)


def short_config_for_row(short_dir: Path, row: dict[str, Any]) -> Path:
    name = str(row["name"])
    if name.endswith("_valid40"):
        name = name[: -len("_valid40")]
    path = short_dir / "configs" / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"short config missing: {path}")
    return path


def make_full_config(short_config: Path, out_config: Path, experiment: str) -> None:
    cfg = deepcopy(load_yaml(short_config))
    cfg["experiment"] = experiment
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = False
    loader["non_blocking"] = True
    optimizer = cfg.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    optimizer["milestones"] = [20, 25]
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    cfg["note"] = str(cfg.get("note", "")) + "\nNTS-11 scope sweep winner: auto-promoted to full30 from valid10 short screen."
    dump_yaml(out_config, cfg)


def standard_eval(config: Path, run_dir: Path, epochs: list[int], status: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
        if not checkpoint.exists():
            append_status(status, f"standard eval skip missing {checkpoint.name}")
            continue
        out_dir = run_dir / "standard_valid825" / f"epoch{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stale = out_dir / "spike_profile.json"
        if stale.exists():
            stale.unlink()
        append_status(status, f"standard valid825 eval epoch{epoch} start")
        run(
            [
                sys.executable,
                "-u",
                "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--path_results",
                str(out_dir),
                "--mode",
                "valid",
            ],
            out_dir / "eval.log",
        )
        summary = out_dir / "spike_profile.json"
        if not summary.exists():
            append_status(status, f"standard eval epoch{epoch} missing spike_profile.json")
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        row = {
            "epoch": epoch,
            "AEE": float(metrics.get("AEE", math.nan)),
            "AAE": float(metrics.get("AAE", math.nan)),
            "PE1": float(metrics.get("AEE_PE1", math.nan)),
            "PE2": float(metrics.get("AEE_PE2", math.nan)),
            "outlier": float(metrics.get("AEE_outliers", math.nan)),
            "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
            "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
            "energy_uj": float(data.get("energy_uj", 0.0) or 0.0),
        }
        rows.append(row)
        append_status(status, f"standard eval epoch{epoch} done AEE={row['AEE']:.4f} AAE={row['AAE']:.4f}")
    return rows


def append_runs_md(
    short_dir: Path,
    best_row: dict[str, Any],
    full_config: Path,
    run_dir: Path,
    eval_rows: list[dict[str, Any]],
) -> None:
    block = [
        "",
        "## NTS-11 Phase-4 二值/三值范围短测 → 全量（自动追加）",
        "",
        f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 短测目录：`{short_dir}`",
        f"- 选中短测：`{best_row['name']}`（valid10 AEE `{float(best_row['AEE']):.4f}`、AAE `{float(best_row['AAE']):.4f}`）",
        f"- 全量配置：`{full_config}`",
        f"- 全量目录：`{run_dir}`",
        "- 方法：两神经元线 scope sweep（11q–11ab），统一 fastlr+freeze816。",
        "- 标准推理：`eval_DSEC_flow_SNN.py` full valid825。",
        "",
        "### 短测排名（valid10）",
        "",
        "见短测目录 `summary.md`。",
        "",
        "### 全量 valid825",
        "",
        "| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes(G) | firing | energy_uj |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_rows:
        block.append(
            f"| {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | {row['PE1']:.4f} | "
            f"{row['PE2']:.4f} | {row['outlier']:.4f} | {row['spikes_g']:.4f} | "
            f"{row['firing'] * 100:.4f}% | {row['energy_uj']:.2f} |"
        )
    if eval_rows:
        best = min(eval_rows, key=lambda item: item["AEE"])
        block.append(
            f"\n当前全量最佳：epoch{best['epoch']}，AEE `{best['AEE']:.4f}`、AAE `{best['AAE']:.4f}`。\n"
        )
    with RUNS_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))


def main() -> int:
    driver_dir = RESULTS_DIR / f"nts11_scope_autopilot_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"driver_dir={driver_dir}")

    run([sys.executable, str(MAKE_CONFIGS)], driver_dir / "make_configs.log")
    run([sys.executable, "-m", "py_compile", str(MAKE_CONFIGS)], driver_dir / "py_compile_make.log")
    run([sys.executable, "-m", "py_compile", str(Path(__file__))], driver_dir / "py_compile_autopilot.log")

    configs = generated_configs()
    append_status(status, f"generated {len(configs)} scope configs")
    verify_cfg = CONFIG_DIR / "generated/nts11r_hw_h60_s23_scope_sn2q_binary_s1224.yml"
    run([sys.executable, str(VERIFY), str(verify_cfg)], driver_dir / "verify_11r.log")

    rapid_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--steps",
        "1224",
        "--prev-runid",
        str(BASELINE),
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
        "--promote-aee",
        "4.50",
        "--promote-aae",
        "90.0",
        "--promote-sops-g",
        "8.0",
        "--no-promote-valid40",
        "--tag",
        TAG,
    ]
    for config in configs:
        rapid_cmd.extend(["--config", str(config)])
    append_status(status, "rapid screen start")
    run(rapid_cmd, driver_dir / "rapid_screen.log")
    short_dir = latest_short_dir()
    append_status(status, f"rapid screen done short_dir={short_dir}")

    best_row = pick_best_short_row(short_dir)
    append_status(
        status,
        f"best short row: {best_row['name']} AEE={float(best_row['AEE']):.4f} AAE={float(best_row['AAE']):.4f}",
    )

    short_config = short_config_for_row(short_dir, best_row)
    full_stem = re.sub(r"_s1224$", "_scope_full30", short_config.stem.replace("_steps1224", ""))
    full_stamp = stamp()
    full_config = CONFIG_DIR / f"{full_stem}_{full_stamp}.yml"
    run_dir = RESULTS_DIR / f"{full_stem}_bs8_{full_stamp}_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    make_full_config(short_config, full_config, full_stem)
    append_status(status, f"full config={full_config} run_dir={run_dir}")

    train_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(full_config),
        "--prev_runid",
        str(BASELINE),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    append_status(status, "full training start")
    run(train_cmd, run_dir / "train.log")
    append_status(status, "full training done")

    eval_rows = standard_eval(full_config, run_dir, [9, 14, 19, 24, 28, 29], status)
    append_runs_md(short_dir, best_row, full_config, run_dir, eval_rows)
    append_status(status, "RUNS.md appended; NTS-11 scope autopilot complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())