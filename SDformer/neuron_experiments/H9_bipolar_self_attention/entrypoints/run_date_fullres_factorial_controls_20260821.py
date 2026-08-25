#!/usr/bin/env python3
"""Run same-parent full-resolution DATE factorial controls sequentially."""

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
GENERATED = EXP / "configs/generated"
MANIFEST = GENERATED / "date_fullres_factorial_controls_20260821.json"
PARENT = (
    EXP
    / "results/dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/checkpoint_epoch29.pth"
)
ROOT = EXP / "results/date_fullres_factorial_controls_20260821"
STATUS = ROOT / "status.log"
LOCK = Path("/tmp/sdformer_date_fullres_factorial_controls_20260821.lock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
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


def audit_initial_load(log: Path, row: dict[str, Any]) -> None:
    text = log.read_text(encoding="utf-8", errors="replace")
    atlif = bool(row["atlif_enabled"])
    attention = bool(row["attention_enabled"])
    expected_missing = 210 if atlif else 0
    required = [
        f"checkpoint_overlay_keys=0, missing={expected_missing}, unexpected=0",
    ]
    if atlif:
        required.append("installed ATLIFTernaryPSN before load: 105 modules")
    if attention:
        required.extend(
            (
                "installed attention before load: 12 modules",
                "installed Shiftmax attention: 12 modules",
            )
        )
    missing = [marker for marker in required if marker not in text]
    forbidden = []
    if not atlif and "installed ATLIFTernaryPSN before load:" in text:
        forbidden.append("ATLIF unexpectedly installed")
    if not attention and "installed attention before load:" in text:
        forbidden.append("Shiftmax unexpectedly installed")
    if missing or forbidden:
        raise RuntimeError(
            f"{row['cell']} initial load audit failed: missing={missing}, "
            f"forbidden={forbidden}"
        )


def parse_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    return {
        "AEE": float(metrics["AEE"]),
        "AAE_2D": float(metrics["AAE"]),
        "AAE_3D": float(metrics["AAE_Benchmark"]),
        "Fl_percent": float(metrics["AEE_outliers"]) * 100.0,
        "total_spikes_g": float(data["total_spikes"]) / 1e9,
        "energy_proxy_uj": float(data["energy_uj"]),
        "samples": int(data["samples"]),
        "module_counts": data.get("module_counts"),
        "checkpoint_load_audit": data.get("checkpoint_load_audit"),
        "artifact_identity": data.get("artifact_identity"),
    }


def write_summary(manifest: dict[str, Any]) -> None:
    rows = []
    for row in manifest["controls"]:
        run_dir = ROOT / row["cell"]
        candidates = []
        for epoch in (4, 9):
            profile = run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
            if profile.is_file():
                candidates.append({"epoch": epoch, **parse_profile(profile)})
        if len(candidates) != 2:
            raise RuntimeError(f"{row['cell']} does not have both valid825 profiles")
        best = min(candidates, key=lambda value: value["AEE"])
        rows.append({**row, "run_dir": str(run_dir.resolve()), "best": best, "epochs": candidates})
    summary = {
        "schema": "date_fullres_factorial_controls_result_v1",
        "parent_checkpoint": str(PARENT.resolve()),
        "parent_checkpoint_sha256": sha256(PARENT),
        "protocol": manifest["shared_protocol"],
        "rows": rows,
        "claim_boundary": (
            "same-parent fresh-optimizer 10-epoch causal controls; local valid825; "
            "not official DSEC hidden-test"
        ),
    }
    json_path = ROOT / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DATE same-parent full-resolution factorial controls",
        "",
        "| cell | ATLIF | TTX | motion | epoch | AEE | AAE-2D | AE-3D | Fl (%) | spikes (G) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = row["best"]
        lines.append(
            f"| {row['cell']} | {int(row['atlif_enabled'])} | "
            f"{int(row['attention_enabled'])} | {row['motion_alpha']} | "
            f"{best['epoch']} | {best['AEE']:.6f} | {best['AAE_2D']:.6f} | "
            f"{best['AAE_3D']:.6f} | {best['Fl_percent']:.4f} | "
            f"{best['total_spikes_g']:.4f} |"
        )
    lines += [
        "",
        "All cells start from the same frozen NB0 ep29 model with a fresh optimizer, "
        "seed 0, 10 full-resolution epochs, and standard local valid825 selection over "
        "predeclared epochs 4 and 9. This is not an official DSEC hidden-test result.",
    ]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("factorial control queue already active", flush=True)
            return 0
        if not MANIFEST.is_file() or not PARENT.is_file():
            raise FileNotFoundError(MANIFEST if not MANIFEST.is_file() else PARENT)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest["parent_checkpoint_sha256"] != sha256(PARENT):
            raise RuntimeError("factorial parent checkpoint SHA changed")
        for row in manifest["controls"]:
            config = Path(row["config"])
            if sha256(config) != row["config_sha256"]:
                raise RuntimeError(f"config SHA changed: {config}")
            run_dir = ROOT / row["cell"]
            ranking = run_dir / "profile_ranking_valid825.md"
            if ranking.is_file():
                record(f"SKIP complete {row['cell']}")
                continue
            checkpoint = run_dir / "checkpoint_epoch9.pth"
            if not checkpoint.is_file():
                run(
                    [
                        sys.executable,
                        "-u",
                        str(EXP / "entrypoints/train.py"),
                        "--config",
                        str(config),
                        "--prev_runid",
                        str(PARENT),
                        "--save_path",
                        str(run_dir / "checkpoint_epoch{}.pth"),
                        "--finetune",
                        "1",
                    ],
                    run_dir / "train.log",
                    f"train {row['cell']}",
                )
            audit_initial_load(run_dir / "train.log", row)
            run(
                [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                    "--config",
                    str(config),
                    "--run-dir",
                    str(run_dir),
                    "--ranking-mode",
                    "aee",
                    "--epoch",
                    "4",
                    "--epoch",
                    "9",
                ],
                run_dir / "valid825.log",
                f"valid825 {row['cell']} epochs 4/9",
            )
        write_summary(manifest)
        record("ALL COMPLETE DATE same-parent full-resolution factorial controls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
