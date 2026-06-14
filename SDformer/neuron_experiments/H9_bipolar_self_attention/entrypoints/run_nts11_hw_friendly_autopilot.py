"""HW-friendly NTS-11 autopilot: verify → short screen → promote winner → full30 → valid825."""

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
MANIFEST = EXP_ROOT / "configs/generated/nts11_hw_friendly_manifest.json"
MAKE_CFG = EXP_ROOT / "entrypoints/make_nts11_hw_friendly_configs.py"
VERIFY = EXP_ROOT / "entrypoints/verify_nts11_chain.py"
RAPID = EXP_ROOT / "entrypoints/rapid_screen.py"
TRAIN = EXP_ROOT / "entrypoints/train.py"
STD_EVAL = EXP_ROOT / "entrypoints/run_h9_standard_valid825_eval.py"
TAG = "nts11_hw_friendly_short"
PY = os.environ.get("HW_FRIENDLY_PYTHON", "/opt/conda/bin/python3")


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
        handle.write(f"\n[hw-friendly-autopilot] exit_code={proc.returncode}\n")
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


def load_manifest() -> list[dict[str, Any]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def pick_winner(summary_csv: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with summary_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty summary: {summary_csv}")
    best = min(rows, key=short_score)
    return best


def base_name_from_row(row: dict[str, Any]) -> str:
    raw = str(row.get("name", row.get("experiment", ""))).strip()
    for suffix in ("_s1224_steps1224", "_steps1224", "_s1224"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    return raw


def promote_full_config(short_name: str) -> Path:
    full = EXP_ROOT / "configs/generated" / f"{short_name}_scope_full30.yml"
    if not full.is_file():
        raise FileNotFoundError(full)
    return full


def find_latest_short_summary() -> Path | None:
    dirs = sorted(RESULTS_DIR.glob(f"{TAG}_*"), key=lambda p: p.stat().st_mtime)
    for short_dir in reversed(dirs):
        summary = short_dir / "summary.csv"
        if summary.is_file():
            return summary
    return None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-short",
        action="store_true",
        help="Skip short screen; use latest nts11_hw_friendly_short_*/summary.csv",
    )
    parser.add_argument("--winner", default="", help="Force promote base name (e.g. nts11aw_hw_h60_s23_sn2qbin_w720_stdlr)")
    args = parser.parse_args()

    driver_dir = RESULTS_DIR / f"nts11_hw_friendly_autopilot_{stamp()}"
    status = driver_dir / "status.log"
    append_status(status, f"driver={driver_dir} python={PY}")

    append_status(status, "step1: generate hw-friendly configs")
    run([PY, str(MAKE_CFG)], driver_dir / "make_config.log")

    manifest = load_manifest()
    short_configs = [Path(item["config_short"]) for item in manifest]
    resume = manifest[0]["resume"]

    if args.resume_short:
        summary = find_latest_short_summary()
        if summary is None:
            raise FileNotFoundError(f"no prior short summary for tag={TAG}")
        append_status(status, f"step2-3: resume short results from {summary.parent}")
    else:
        append_status(status, f"step2: verify chain ({short_configs[0].name})")
        run([PY, str(VERIFY), str(short_configs[0])], driver_dir / "verify.log")

        append_status(status, "step3: rapid_screen 1224 steps (11aw/11ax/11ay)")
        cmd = [
            PY,
            "-u",
            str(RAPID),
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
        for cfg in short_configs[:3]:
            cmd.extend(["--config", str(cfg)])
        run(cmd, driver_dir / "rapid_screen.log")
        summary = find_latest_short_summary()
        if summary is None:
            raise FileNotFoundError("rapid_screen finished but summary.csv missing")

    winner = pick_winner(summary)
    winner_name = args.winner.strip() or base_name_from_row(winner)
    append_status(
        status,
        f"step3 winner: {winner_name} score={short_score(winner):.4f} "
        f"AEE={winner.get('AEE')} SOPs_G={winner.get('SOPs_G')}",
    )

    full_cfg = promote_full_config(winner_name)
    run_stamp = stamp()
    run_dir = RESULTS_DIR / f"{winner_name.replace('_s1224', '')}_bs8_{run_stamp}_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    (driver_dir / "full_run_dir.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    save_path = str(run_dir / "checkpoint_epoch{}.pth")
    append_status(status, f"step4: full30 train -> {run_dir} save_path={save_path}")

    run(
        [
            PY,
            "-u",
            str(TRAIN),
            "--config",
            str(full_cfg),
            "--prev_runid",
            resume,
            "--save_path",
            save_path,
        ],
        run_dir / "train.log",
        check=False,
    )

    ckpts = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    ckpts = [p for p in ckpts if "state_dict" not in p.name]
    if not ckpts:
        append_status(status, "step4: training failed or no checkpoints — stopping before valid825")
        return 1

    eval_epochs = []
    for ep in (9, 14, 19, 24, 28, 29):
        if (run_dir / f"checkpoint_epoch{ep}.pth").is_file():
            eval_epochs.append(ep)
    if not eval_epochs:
        eval_epochs = [int(ckpts[-1].stem.replace("checkpoint_epoch", ""))]

    append_status(status, f"step5: valid825 epochs {eval_epochs}")
    eval_cmd = [PY, "-u", str(STD_EVAL), "--config", str(full_cfg), "--run-dir", str(run_dir)]
    for ep in eval_epochs:
        eval_cmd.extend(["--epoch", str(ep)])
    run(eval_cmd, driver_dir / "valid825.log", check=False)

    append_status(status, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())