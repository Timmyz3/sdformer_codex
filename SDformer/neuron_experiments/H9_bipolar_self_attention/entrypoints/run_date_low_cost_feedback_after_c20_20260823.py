#!/usr/bin/env python3
"""Evaluate low-cost Motion/Local5 deployment constants after C20 completes."""

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
MANIFEST = EXP / "configs/generated/date_low_cost_feedback_sensitivity_20260823.json"
FACTORIAL = EXP / "results/date_fullres_factorial_controls_20260821"
LOCAL_CONTROL = EXP / "results/date_fullres_local5_same_parent_control_20260821"
ROOT = EXP / "results/date_low_cost_feedback_sensitivity_20260823"
STATUS = ROOT / "status.log"
LOCK = Path("/tmp/sdformer_date_low_cost_feedback_sensitivity_20260823.lock")


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


def parse_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    counts = data.get("module_counts") or {}
    load = data.get("checkpoint_load_audit") or {}
    if counts.get("ATLIFTernaryPSN") != 105 or counts.get("ShiftmaxAttention") != 12:
        raise RuntimeError(f"module-count audit failed: {path}")
    expected_load = {
        "checkpoint_overlay_keys": 210,
        "missing_count": 0,
        "unexpected_count": 0,
    }
    for key, expected in expected_load.items():
        if load.get(key) != expected:
            raise RuntimeError(f"load audit {key} failed in {path}: {load.get(key)}")
    return {
        "AEE": float(metrics["AEE"]),
        "AAE_2D": float(metrics["AAE"]),
        "AAE_3D": float(metrics["AAE_Benchmark"]),
        "Fl_percent": float(metrics["AEE_outliers"]) * 100.0,
        "total_spikes_g": float(data["total_spikes"]) / 1e9,
        "energy_proxy_uj": float(data["energy_uj"]),
        "samples": int(data["samples"]),
        "profile": str(path.resolve()),
        "profile_sha256": sha256(path),
    }


def percent_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100.0


def percent_increase(baseline: float, candidate: float) -> float:
    return (candidate - baseline) / baseline * 100.0


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("DATE low-cost feedback watcher already active", flush=True)
            return 0

        prerequisites = (
            FACTORIAL / "c12_binary_motion_ttx/profile_ranking_valid825.md",
            LOCAL_CONTROL / "profile_ranking_valid825.md",
        )
        while not all(path.is_file() for path in prerequisites):
            record("WAIT C12 and C20 standard valid825")
            time.sleep(300)

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for source in manifest["source_configs"].values():
            path = Path(source["path"])
            if sha256(path) != source["sha256"]:
                raise RuntimeError(f"source config SHA changed: {path}")

        source_checkpoints = {
            "motion": FACTORIAL / "c12_binary_motion_ttx/checkpoint_epoch9.pth",
            "local5": LOCAL_CONTROL / "checkpoint_epoch9.pth",
        }
        source_profiles = {
            "motion": FACTORIAL / "c12_binary_motion_ttx/standard_valid825/epoch9/spike_profile.json",
            "local5": LOCAL_CONTROL / "standard_valid825/epoch9/spike_profile.json",
        }
        rows: list[dict[str, Any]] = []
        baselines = {
            family: {
                "id": f"{family}_baseline",
                "family": family,
                "value": manifest["source_configs"][family]["baseline_value"],
                "checkpoint": str(source_checkpoints[family].resolve()),
                "checkpoint_sha256": sha256(source_checkpoints[family]),
                "source_baseline": True,
                **parse_profile(source_profiles[family]),
            }
            for family in ("motion", "local5")
        }

        for variant in manifest["variants"]:
            config = Path(variant["config"])
            if sha256(config) != variant["config_sha256"]:
                raise RuntimeError(f"variant config SHA changed: {config}")
            family = variant["family"]
            checkpoint = source_checkpoints[family]
            run_dir = ROOT / variant["id"]
            linked_checkpoint = run_dir / "checkpoint_epoch9.pth"
            run_dir.mkdir(parents=True, exist_ok=True)
            if not linked_checkpoint.exists():
                os.link(checkpoint, linked_checkpoint)
            if sha256(linked_checkpoint) != sha256(checkpoint):
                raise RuntimeError(f"checkpoint link mismatch: {linked_checkpoint}")
            profile = run_dir / "standard_valid825/epoch9/spike_profile.json"
            if not profile.is_file():
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
                        "9",
                    ],
                    run_dir / "valid825.log",
                    f"valid825 {variant['id']}",
                )
            rows.append(
                {
                    **variant,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                    "source_baseline": False,
                    **parse_profile(profile),
                }
            )

        all_rows = [baselines["motion"], baselines["local5"], *rows]
        decisions = {}
        for family in ("motion", "local5"):
            baseline = baselines[family]
            candidates = [row for row in rows if row["family"] == family]
            best = min(candidates, key=lambda row: row["AEE"])
            aee_gain = percent_improvement(baseline["AEE"], best["AEE"])
            aae_gain = percent_improvement(baseline["AAE_2D"], best["AAE_2D"])
            spike_increase = percent_increase(baseline["total_spikes_g"], best["total_spikes_g"])
            if family == "motion":
                passed = (aee_gain >= 0.3 or aae_gain >= 0.5) and spike_increase <= 2.0
            else:
                passed = aee_gain >= 0.3 and spike_increase <= 2.0
            decisions[family] = {
                "baseline": baseline["id"],
                "best_candidate": best["id"],
                "best_value": best["value"],
                "aee_improvement_percent": aee_gain,
                "aae_2d_improvement_percent": aae_gain,
                "spikes_increase_percent": spike_increase,
                "promotion_gate_passed": passed,
                "next_action": "short_finetune" if passed else "stop_constant_sweep",
            }

        summary = {
            "schema": "date_low_cost_feedback_sensitivity_result_v1",
            "manifest": str(MANIFEST.resolve()),
            "manifest_sha256": sha256(MANIFEST),
            "rows": all_rows,
            "decisions": decisions,
            "claim_boundary": manifest["protocol"]["claim_boundary"],
        }
        (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# DATE low-cost algorithm feedback sensitivity",
            "",
            "| family | id | value | AEE | AAE-2D | AE-3D | Fl (%) | spikes (G) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in all_rows:
            lines.append(
                f"| {row['family']} | {row['id']} | {row['value']:.7f} | "
                f"{row['AEE']:.6f} | {row['AAE_2D']:.6f} | {row['AAE_3D']:.6f} | "
                f"{row['Fl_percent']:.4f} | {row['total_spikes_g']:.4f} |"
            )
        lines += ["", "## Decisions", ""]
        for family, decision in decisions.items():
            lines.append(f"- `{family}`: `{json.dumps(decision, sort_keys=True)}`")
        lines += [
            "",
            "These are frozen-checkpoint deployment-constant sensitivity results, not trained causal ablations.",
        ]
        (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        record("ALL COMPLETE DATE low-cost Motion/Local5 feedback sensitivity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
