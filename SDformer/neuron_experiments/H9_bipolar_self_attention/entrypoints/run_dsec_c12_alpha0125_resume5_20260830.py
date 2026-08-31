#!/usr/bin/env python3
"""Run promoted C12 alpha=1/8 resume5 and standard valid825 evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.yml"
MANIFEST = EXP / "configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.json"
PARENT = EXP / "results/date_two_contribution_full30_20260826/c12_binary_motion_ttx/checkpoint_epoch29.pth"
RUN = EXP / "results/dsec_c12_alpha0125_ep29_resume5_20260830"
C10_PROFILE = EXP / "results/date_two_contribution_full30_20260826/c10_binary_original/standard_valid825/epoch29/spike_profile.json"
C12_PROFILE = EXP / "results/date_two_contribution_full30_20260826/c12_binary_motion_ttx/standard_valid825/epoch29/spike_profile.json"
STATUS = RUN / "status.log"
LOCK = Path("/tmp/sdformer_dsec_c12_alpha0125_resume5_20260830.lock")
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
RESULT_MARKER = "<!-- DSEC_C12_ALPHA0125_RESUME5_RESULT_20260830 -->"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def wait_for_idle_gpu() -> None:
    stable = 0
    while stable < 2:
        pid_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        memory_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if pid_result.returncode or memory_result.returncode:
            raise RuntimeError((pid_result.stderr + memory_result.stderr).strip())
        pids = [line.strip() for line in pid_result.stdout.splitlines() if line.strip().isdigit()]
        memory_mib = max(
            [int(line.strip()) for line in memory_result.stdout.splitlines() if line.strip().isdigit()],
            default=0,
        )
        if pids or memory_mib > 1024:
            stable = 0
            record(f"WAIT GPU compute pids={pids}, memory_mib={memory_mib}")
        else:
            stable += 1
            record(f"GPU idle confirmation {stable}/2 memory_mib={memory_mib}")
        if stable < 2:
            time.sleep(30)


def run(command: list[str], log: Path, label: str) -> None:
    record(f"START {label}: {' '.join(command)}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {log}")


def parse_profile(
    path: Path,
    *,
    expected_atlif: int = 105,
    expected_shiftmax: int = 12,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw["metrics"]
    counts = raw.get("module_counts") or {}
    audit = raw.get("checkpoint_load_audit") or {}
    if int(raw.get("samples", 0)) != 825:
        raise RuntimeError(f"not valid825: {path}")
    if (
        int(counts.get("ATLIFTernaryPSN", -1)) != expected_atlif
        or int(counts.get("ShiftmaxAttention", -1)) != expected_shiftmax
    ):
        raise RuntimeError(f"module count audit failed: {path}")
    for key, expected in (("checkpoint_overlay_keys", 210), ("missing_count", 0), ("unexpected_count", 0)):
        if int(audit.get(key, -1)) != expected:
            raise RuntimeError(f"checkpoint audit {key} failed: {path}")
    return {
        "AEE": float(metrics["AEE"]),
        "AAE_2D": float(metrics["AAE"]),
        "AE_3D": float(metrics["AAE_Benchmark"]),
        "Fl_percent": float(metrics["AEE_outliers"]) * 100.0,
        "spikes_g": float(raw["total_spikes"]) / 1e9,
        "energy_proxy_uj": float(raw["energy_uj"]),
        "profile": str(path.resolve()),
        "profile_sha256": sha256(path),
    }


def main() -> int:
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("C12 alpha=1/8 resume5 already active", flush=True)
            return 0
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for path in (CONFIG, PARENT, PARENT.with_name(PARENT.stem + "_state_dict.pth"), C10_PROFILE, C12_PROFILE):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(CONFIG) != manifest["config_sha256"] or sha256(PARENT) != manifest["parent_checkpoint_sha256"]:
            raise RuntimeError("C12 alpha=1/8 resume identity mismatch")
        wait_for_idle_gpu()
        final_checkpoint = RUN / "checkpoint_epoch34.pth"
        if not final_checkpoint.is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/train.py"),
                    "--config",
                    str(CONFIG),
                    "--prev_runid",
                    str(PARENT),
                    "--save_path",
                    str(RUN / "checkpoint_epoch{}.pth"),
                    "--finetune",
                    "1",
                    "--resume",
                    "1",
                ],
                RUN / "train.log",
                "train C12 alpha=1/8 resume ep30-34",
            )
        train_text = (RUN / "train.log").read_text(encoding="utf-8", errors="replace")
        required = (
            "installed ATLIFTernaryPSN before load: 105 modules",
            "installed attention before load: 12 modules",
            "checkpoint_overlay_keys=210, missing=0, unexpected=0",
            "Training state resumed from local checkpoint",
        )
        missing = [marker for marker in required if marker not in train_text]
        if missing:
            raise RuntimeError(f"C12 alpha=1/8 resume audit failed: {missing}")
        epochs = manifest["evaluation_epochs"]
        profiles = [RUN / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json" for epoch in epochs]
        if not all(path.is_file() for path in profiles):
            command = [
                sys.executable,
                "-u",
                str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config",
                str(CONFIG),
                "--run-dir",
                str(RUN),
                "--ranking-mode",
                "aee",
            ]
            for epoch in epochs:
                command.extend(["--epoch", str(epoch)])
            run(command, RUN / "valid825.log", "valid825 C12 alpha=1/8 resume5")
        rows = [{"epoch": epoch, **parse_profile(profile)} for epoch, profile in zip(epochs, profiles)]
        best = min(rows, key=lambda row: row["AEE"])
        c10 = parse_profile(C10_PROFILE, expected_shiftmax=0)
        c12 = parse_profile(C12_PROFILE)
        payload = {
            "schema": "dsec_c12_alpha0125_resume5_result_v1",
            "manifest": str(MANIFEST.resolve()),
            "manifest_sha256": sha256(MANIFEST),
            "strict_c10_ep29": c10,
            "strict_c12_alpha025_ep29": c12,
            "rows": rows,
            "best": best,
            "beats_c10": best["AEE"] < c10["AEE"],
            "claim_boundary": manifest["claim_boundary"],
        }
        (RUN / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# DSEC optimized C12 alpha=1/8 resume5",
            "",
            "| epoch | AEE | AAE-2D | AE-3D | Fl (%) | spikes (G) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['epoch']} | {row['AEE']:.6f} | {row['AAE_2D']:.6f} | "
                f"{row['AE_3D']:.6f} | {row['Fl_percent']:.4f} | {row['spikes_g']:.4f} |"
            )
        lines += ["", f"Best epoch: `{best['epoch']}`; beats strict C10: `{payload['beats_c10']}`.", ""]
        (RUN / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        if RESULT_MARKER not in REDESIGN.read_text(encoding="utf-8"):
            with REDESIGN.open("a", encoding="utf-8") as handle:
                handle.write("\n" + RESULT_MARKER + "\n\n")
                handle.write("### DSEC optimized C12 alpha=1/8 resume5 结果（2026-08-30）\n\n")
                handle.write("\n".join(lines[2:]) + "\n")
                handle.write("\n该行是优化部署主线，不替代 strict same-parent full30 因果消融。\n")
        record(f"ALL COMPLETE best=ep{best['epoch']} AEE={best['AEE']:.6f} beats_c10={payload['beats_c10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
