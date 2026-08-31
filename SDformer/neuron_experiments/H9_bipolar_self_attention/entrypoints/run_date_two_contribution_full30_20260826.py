#!/usr/bin/env python3
"""Run clean same-parent full30 DATE two-contribution controls sequentially."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
MANIFEST = GENERATED / "date_two_contribution_full30_20260826.json"
ROOT = EXP / "results/date_two_contribution_full30_20260826"
STATUS = ROOT / "status.log"
LOCK = Path("/tmp/sdformer_date_two_contribution_full30_20260826.lock")


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
    required = [f"checkpoint_overlay_keys=0, missing={expected_missing}, unexpected=0"]
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
        "AE_3D": float(metrics["AAE_Benchmark"]),
        "Fl_percent": float(metrics["AEE_outliers"]) * 100.0,
        "total_spikes_g": float(data["total_spikes"]) / 1e9,
        "energy_proxy_uj": float(data["energy_uj"]),
        "samples": int(data["samples"]),
        "module_counts": data.get("module_counts"),
        "checkpoint_load_audit": data.get("checkpoint_load_audit"),
        "artifact_identity": data.get("artifact_identity"),
    }


def write_summary(manifest: dict[str, Any]) -> None:
    eval_epochs = manifest["shared_protocol"]["evaluation_epochs"]
    rows = []
    for row in manifest["controls"]:
        run_dir = ROOT / row["cell"]
        candidates = []
        for epoch in eval_epochs:
            profile = run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
            if not profile.is_file():
                raise RuntimeError(f"missing profile: {profile}")
            candidates.append({"epoch": epoch, **parse_profile(profile)})
        best = min(candidates, key=lambda value: value["AEE"])
        rows.append(
            {
                **row,
                "run_dir": str(run_dir.resolve()),
                "best": best,
                "epochs": candidates,
                "last_minus_best_AEE": candidates[-1]["AEE"] - best["AEE"],
            }
        )
    summary = {
        "schema": "date_two_contribution_full30_result_v1",
        "parent_checkpoint": manifest["parent_checkpoint"],
        "parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
        "protocol": manifest["shared_protocol"],
        "rows": rows,
        "claim_boundary": manifest["claim_boundary"],
    }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# DATE same-parent full30 two-contribution controls",
        "",
        "| cell | ATLIF | Complete TTX | alpha | rank1 epoch | AEE | AAE-2D | AE-3D | Fl (%) | spikes (G) | energy proxy (uJ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = row["best"]
        lines.append(
            f"| {row['cell']} | {int(row['atlif_enabled'])} | "
            f"{int(row['attention_enabled'])} | {row['motion_alpha']} | "
            f"{best['epoch']} | {best['AEE']:.6f} | {best['AAE_2D']:.6f} | "
            f"{best['AE_3D']:.6f} | {best['Fl_percent']:.4f} | "
            f"{best['total_spikes_g']:.4f} | {best['energy_proxy_uj']:.2f} |"
        )
    lines += [
        "",
        "All rows use the same frozen NB0 ep29 parent, fresh optimizer, seed 0, "
        "30 full-resolution epochs, and local valid825 selection over epochs "
        f"{eval_epochs}. These are not official DSEC hidden-test results.",
    ]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("two-contribution full30 queue already active", flush=True)
            return 0
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        parent = Path(manifest["parent_checkpoint"])
        if sha256(parent) != manifest["parent_checkpoint_sha256"]:
            raise RuntimeError("parent checkpoint SHA changed")
        eval_epochs = [str(epoch) for epoch in manifest["shared_protocol"]["evaluation_epochs"]]
        for row in manifest["controls"]:
            config = Path(row["config"])
            if sha256(config) != row["config_sha256"]:
                raise RuntimeError(f"config SHA changed: {config}")
            run_dir = ROOT / row["cell"]
            final_checkpoint = run_dir / "checkpoint_epoch29.pth"
            if not final_checkpoint.is_file():
                run(
                    [
                        sys.executable,
                        "-u",
                        str(EXP / "entrypoints/train.py"),
                        "--config",
                        str(config),
                        "--prev_runid",
                        str(parent),
                        "--save_path",
                        str(run_dir / "checkpoint_epoch{}.pth"),
                        "--finetune",
                        "1",
                    ],
                    run_dir / "train.log",
                    f"train {row['cell']}",
                )
            audit_initial_load(run_dir / "train.log", row)
            profile_paths = [
                run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
                for epoch in eval_epochs
            ]
            if not all(path.is_file() for path in profile_paths):
                command = [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                    "--config",
                    str(config),
                    "--run-dir",
                    str(run_dir),
                    "--ranking-mode",
                    "aee",
                ]
                for epoch in eval_epochs:
                    command.extend(["--epoch", epoch])
                run(command, run_dir / "valid825.log", f"valid825 {row['cell']}")
        write_summary(manifest)
        record("ALL COMPLETE DATE same-parent full30 two-contribution controls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
