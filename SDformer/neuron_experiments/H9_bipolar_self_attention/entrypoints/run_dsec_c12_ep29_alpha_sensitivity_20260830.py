#!/usr/bin/env python3
"""Run standard valid825 for frozen C12 ep29 dyadic-alpha candidates."""

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
MANIFEST = EXP / "configs/generated/dsec_c12_ep29_alpha_sensitivity_20260830.json"
SOURCE_RUN = EXP / "results/date_two_contribution_full30_20260826/c12_binary_motion_ttx"
C10_RUN = EXP / "results/date_two_contribution_full30_20260826/c10_binary_original"
CHECKPOINT = SOURCE_RUN / "checkpoint_epoch29.pth"
SOURCE_PROFILE = SOURCE_RUN / "standard_valid825/epoch29/spike_profile.json"
C10_PROFILE = C10_RUN / "standard_valid825/epoch29/spike_profile.json"
ROOT = EXP / "results/dsec_c12_ep29_alpha_sensitivity_20260830"
STATUS = ROOT / "status.log"
LOCK = Path("/tmp/sdformer_dsec_c12_ep29_alpha_sensitivity_20260830.lock")
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
RESULT_MARKER = "<!-- DSEC_C12_EP29_ALPHA_SENSITIVITY_RESULT_20260830 -->"


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


def parse_profile(path: Path, *, expected_shiftmax: int) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw["metrics"]
    counts = raw.get("module_counts") or {}
    audit = raw.get("checkpoint_load_audit") or {}
    expected = {
        "ATLIFTernaryPSN": 105,
        "ShiftmaxAttention": expected_shiftmax,
    }
    for key, value in expected.items():
        if int(counts.get(key, -1)) != value:
            raise RuntimeError(f"module count {key} failed in {path}: {counts}")
    for key, value in (
        ("checkpoint_overlay_keys", 210),
        ("missing_count", 0),
        ("unexpected_count", 0),
    ):
        if int(audit.get(key, -1)) != value:
            raise RuntimeError(f"load audit {key} failed in {path}: {audit}")
    if int(raw.get("samples", 0)) != 825:
        raise RuntimeError(f"not a valid825 profile: {path}")
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
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("C12 ep29 alpha sensitivity is already active", flush=True)
            return 0
        for path in (MANIFEST, CHECKPOINT, SOURCE_PROFILE, C10_PROFILE):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source_config = Path(manifest["source_config"])
        if sha256(source_config) != manifest["source_config_sha256"]:
            raise RuntimeError("source config SHA mismatch")

        rows = [
            {
                "id": "alpha0250",
                "alpha": 0.25,
                "source_baseline": True,
                **parse_profile(SOURCE_PROFILE, expected_shiftmax=12),
            }
        ]
        c10 = parse_profile(C10_PROFILE, expected_shiftmax=0)
        for variant in manifest["variants"]:
            config = Path(variant["config"])
            if sha256(config) != variant["config_sha256"]:
                raise RuntimeError(f"variant config SHA mismatch: {config}")
            run_dir = ROOT / variant["id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            linked = run_dir / "checkpoint_epoch29.pth"
            if not linked.exists():
                os.link(CHECKPOINT, linked)
            if sha256(linked) != sha256(CHECKPOINT):
                raise RuntimeError(f"checkpoint hardlink mismatch: {linked}")
            profile = run_dir / "standard_valid825/epoch29/spike_profile.json"
            if not profile.is_file():
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
                    "--epoch",
                    "29",
                ]
                record(f"START {variant['id']}: {' '.join(command)}")
                with (run_dir / "valid825.log").open("w", encoding="utf-8") as handle:
                    result = subprocess.run(
                        command,
                        cwd=REPO,
                        env=environment(),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                record(f"END {variant['id']}: exit_code={result.returncode}")
                if result.returncode:
                    raise RuntimeError(f"valid825 failed: {run_dir / 'valid825.log'}")
            rows.append(
                {
                    **variant,
                    "source_baseline": False,
                    **parse_profile(profile, expected_shiftmax=12),
                }
            )

        baseline = rows[0]
        best = min(rows, key=lambda row: row["AEE"])
        spike_increase = (best["spikes_g"] - baseline["spikes_g"]) / baseline["spikes_g"] * 100.0
        ae3d_increase = (best["AE_3D"] - baseline["AE_3D"]) / baseline["AE_3D"] * 100.0
        decision = {
            "best_id": best["id"],
            "best_alpha": best["alpha"],
            "best_aee": best["AEE"],
            "c10_aee": c10["AEE"],
            "beats_c10": best["AEE"] < c10["AEE"],
            "aee_improvement_vs_alpha025_percent": (
                (baseline["AEE"] - best["AEE"]) / baseline["AEE"] * 100.0
            ),
            "ae3d_increase_vs_alpha025_percent": ae3d_increase,
            "spikes_increase_vs_alpha025_percent": spike_increase,
            "promotion_gate_passed": (
                best["AEE"] < c10["AEE"] and ae3d_increase <= 0.2 and spike_increase <= 1.0
            ),
        }
        payload = {
            "schema": "dsec_c12_ep29_alpha_sensitivity_result_v1",
            "manifest": str(MANIFEST.resolve()),
            "manifest_sha256": sha256(MANIFEST),
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "c10": c10,
            "rows": rows,
            "decision": decision,
            "claim_boundary": manifest["protocol"]["claim_boundary"],
        }
        (ROOT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# DSEC C12 ep29 dyadic-alpha sensitivity",
            "",
            "| alpha | AEE | AAE-2D | AE-3D | Fl (%) | spikes (G) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(rows, key=lambda item: item["alpha"]):
            lines.append(
                f"| {row['alpha']:.3f} | {row['AEE']:.6f} | {row['AAE_2D']:.6f} | "
                f"{row['AE_3D']:.6f} | {row['Fl_percent']:.4f} | {row['spikes_g']:.4f} |"
            )
        lines += ["", f"Decision: `{json.dumps(decision, sort_keys=True)}`", ""]
        (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        if RESULT_MARKER not in REDESIGN.read_text(encoding="utf-8"):
            with REDESIGN.open("a", encoding="utf-8") as handle:
                handle.write("\n" + RESULT_MARKER + "\n\n")
                handle.write("### DSEC C12 ep29 dyadic-alpha 冻结敏感性结果（2026-08-30）\n\n")
                handle.write("\n".join(lines[2:]) + "\n")
                handle.write(
                    "\n以上仅为同一 C12 ep29 checkpoint 的部署常数敏感性，不是训练因果消融。\n"
                )
        record(f"ALL COMPLETE best={best['id']} AEE={best['AEE']:.6f} beats_c10={decision['beats_c10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
