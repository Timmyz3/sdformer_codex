#!/usr/bin/env python3
"""Queue H67 QF5-QF8 sensitivity after the H81 control releases the GPU."""

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


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
ENTRY = EXP / "entrypoints"
GEN = EXP / "configs/generated"
CHECKPOINT = RESULTS / (
    "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
)
MANIFEST = GEN / "h67_ep35_score_precision_qf5_qf8_manifest.json"
UPSTREAM = REPO / "neuron_autoresearch/H67_H81_NOMOTION_RESULT_20260812.json"
ROOT = RESULTS / "h67_ep35_score_precision_qf5_qf8_20260813"
STATUS = ROOT / "status.log"
OUTPUT = ROOT / "summary.json"
OUTPUT_MD = ROOT / "summary.md"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
LOCK = Path("/tmp/sdformer_h67_score_precision_qf5_qf8.lock")


def record(message: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def stage_checkpoint(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    staged = run_dir / "checkpoint_epoch35.pth"
    if not staged.exists():
        os.link(CHECKPOINT, staged)
    if staged.stat().st_ino != CHECKPOINT.stat().st_ino:
        raise RuntimeError(f"precision sweep checkpoint is not the canonical hardlink: {staged}")


def parse_valid_profile(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw["metrics"]
    identity = raw["artifact_identity"]
    load = raw["checkpoint_load_audit"]
    counts = raw["module_counts"]
    checks = {
        "samples825": raw.get("samples") == 825,
        "checkpoint_sha": identity.get("checkpoint_sha256") == sha256(CHECKPOINT),
        "overlay210": load.get("checkpoint_overlay_keys") == 210,
        "missing0": load.get("missing_count") == 0,
        "unexpected0": load.get("unexpected_count") == 0,
        "ATLIF105": counts.get("ATLIFTernaryPSN") == 105,
        "Shiftmax12": counts.get("ShiftmaxAttention") == 12,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"precision valid825 contract failed: {failed}")
    return {
        "AEE": float(metrics["AEE"]),
        "AAE_2D": float(metrics["AAE"]),
        "AE_3D": float(metrics["AAE_Benchmark"]),
        "DSEC_Fl": float(metrics["DSEC_Fl"]),
        "total_spikes_g": float(raw["total_spikes"]) / 1e9,
    }


def write_summary(manifest: dict[str, object], profile_path: Path) -> None:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    pair = profile["summary"]["binary_temporal_pairs"]
    pair_total = int(pair["pair_total"])
    rows = []
    for item in manifest["rows"]:
        bits = int(item["fractional_bits"])
        run_dir = Path(item["result_dir"])
        metrics = parse_valid_profile(run_dir / "standard_valid825/epoch35/spike_profile.json")
        equal = int(pair[f"pair_score_equal_h67_qf{bits}"])
        rows.append(
            {
                "fractional_bits": bits,
                "score_step": float(item["score_step"]),
                **metrics,
                "profile_samples_for_equality": int(profile["samples"]),
                "temporal_pairs": pair_total,
                "score_pair_equal": equal,
                "score_pair_equal_ratio": equal / pair_total,
                "ideal_dual_slot_reduction_ratio": equal / (2 * pair_total),
                "config": item["config"],
                "config_sha256": item["config_sha256"],
                "result_dir": item["result_dir"],
                "rtl_claim": "none; algorithm sensitivity only",
            }
        )
    q7 = next(row for row in rows if row["fractional_bits"] == 7)
    for row in rows:
        row["AEE_change_vs_QF7"] = row["AEE"] - q7["AEE"]
    payload = {
        "schema": "h67_ep35_score_precision_qf5_qf8_result_v1",
        "status": "PASS",
        "scope": "generic Shiftmax algorithm sensitivity; QF5/QF6/QF8 are not RTL implementations",
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "profile": str(profile_path.resolve()),
        "profile_sha256": sha256(profile_path),
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# H67 ep35 score fractional-precision sensitivity",
        "",
        "QF denotes fractional bits. QF5/QF6/QF8 use generic quantized Shiftmax and carry no RTL-exact claim.",
        "",
        "| score | AEE | delta vs QF7 | AAE-2D | AE-3D | Fl(%) | spikes(G) | pair equal | ideal dual-slot reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| QF{row['fractional_bits']} | {row['AEE']:.6f} | "
            f"{row['AEE_change_vs_QF7']:+.6f} | {row['AAE_2D']:.6f} | "
            f"{row['AE_3D']:.6f} | {row['DSEC_Fl']:.4f} | "
            f"{row['total_spikes_g']:.4f} | {row['score_pair_equal_ratio']:.4%} | "
            f"{row['ideal_dual_slot_reduction_ratio']:.4%} |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker = "H67_SCORE_PRECISION_QF5_QF8_RESULT_20260813"
    text = REDESIGN.read_text(encoding="utf-8")
    if marker not in text:
        with REDESIGN.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n"
                f"<!-- {marker} -->\n\n"
                "### H67 ep35 score QF5-QF8 sensitivity\n\n"
                + "\n".join(lines[4:])
                + "\n\n- 该表是算法位宽敏感性，不把 QF5/QF6/QF8 写成已有 RTL。\n"
                f"- 机器结果：`{OUTPUT.relative_to(REPO)}`。\n"
            )


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("H67 score precision sweep watcher already active", flush=True)
            return 0
        while not UPSTREAM.is_file():
            record("WAIT H67/H81 no-motion result and existing GPU queue")
            time.sleep(300)
        if OUTPUT.is_file():
            record("ALL COMPLETE H67 QF5-QF8 score precision sweep already exists")
            return 0
        run(
            [sys.executable, "-u", str(ENTRY / "make_h67_score_precision_sweep.py")],
            ROOT / "generate.log",
            "generate H67 score precision configs",
        )
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for item in manifest["rows"]:
            bits = int(item["fractional_bits"])
            run_dir = Path(item["result_dir"])
            stage_checkpoint(run_dir)
            run(
                [
                    sys.executable,
                    "-u",
                    str(ENTRY / "run_h9_standard_valid825_eval.py"),
                    "--config",
                    str(item["config"]),
                    "--run-dir",
                    str(run_dir),
                    "--ranking-mode",
                    "aee",
                    "--epoch",
                    "35",
                ],
                run_dir / "valid825_runner.log",
                f"H67 QF{bits} valid825",
            )
        q7 = next(item for item in manifest["rows"] if int(item["fractional_bits"]) == 7)
        profile_dir = ROOT / "qf5_qf8_equality_profile20"
        profile_path = profile_dir / "nts11_hardware_p0_profile.json"
        if not profile_path.is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    str(ENTRY / "profile_nts11_hardware_p0.py"),
                    "--config",
                    str(q7["config"]),
                    "--checkpoint",
                    str(CHECKPOINT),
                    "--output-dir",
                    str(profile_dir),
                    "--samples",
                    "20",
                ],
                ROOT / "profile20.log",
                "H67 QF5-QF8 equality profile20",
            )
        write_summary(manifest, profile_path)
        record(f"ALL COMPLETE H67 QF5-QF8 score precision sweep: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
