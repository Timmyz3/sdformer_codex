#!/usr/bin/env python3
"""Extend Local5 full-resolution budget 40->50 after the H81 control finishes."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
SOURCE_CONFIG = GEN / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40.yml"
CONFIG = GEN / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml"
SOURCE_ROOT = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40_20260809"
ROOT = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812"
QF_SUMMARY = (
    RESULTS / "h67_ep35_score_precision_qf5_qf8_20260813/summary.json"
)
STATUS = ROOT / "status.log"
LOCK = Path("/tmp/sdformer_local5_fullres_plus20_ep50.lock")
SOURCE_LABEL = 39
FINAL_LABEL = 49
EVAL_LABELS = (39, 44, 49)
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
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


def generate_config() -> None:
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(SOURCE_CONFIG)
    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    source["experiment"] = "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50"
    source["loader"]["n_epochs"] = 50
    runtime = source["runtime"]
    runtime["force_save_epochs"] = [44, 49]
    runtime["state_save_epochs"] = [44, 49]
    runtime["full_resolution_protocol"] = (
        "paper_480x640_window2x15x15_local5_bb1e4_equal_plus20_ep50"
    )
    runtime["convergence_extension"] = (
        "local5_only_40_to_50_after_equal_plus10_and_h81_control"
    )
    runtime["resume_protocol"] = (
        "audited_model_optimizer_scheduler_scaler_local5_40_to_50"
    )
    runtime["resume_source_budget"] = 40
    runtime["resume_source_checkpoint_label"] = SOURCE_LABEL
    source["note"] = (
        "Local5-only right-censor closure after the fair 30/35/40 comparison and H81 "
        "no-motion control. Resume model, AdamW, AMP scaler and fixed optimizer LR from "
        "ep39; evaluate ep39/44/49. This does not inherit ep29 hardware provenance."
    )
    rendered = yaml.safe_dump(source, sort_keys=False, width=100)
    if CONFIG.is_file():
        current = CONFIG.read_text(encoding="utf-8")
        if current != rendered:
            raise RuntimeError(f"generated config drift: {CONFIG}")
        return
    CONFIG.write_text(rendered, encoding="utf-8")


def stage_resume() -> None:
    source_model = SOURCE_ROOT / f"checkpoint_epoch{SOURCE_LABEL}.pth"
    source_state = SOURCE_ROOT / f"checkpoint_epoch{SOURCE_LABEL}_state_dict.pth"
    for path in (source_model, source_state):
        if not path.is_file():
            raise FileNotFoundError(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    model = ROOT / source_model.name
    state_path = ROOT / source_state.name
    audit_path = ROOT / "resume_stage_audit.json"
    if not model.exists():
        os.link(source_model, model)
    if not state_path.exists():
        os.link(source_state, state_path)

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    scheduler = state.get("scheduler") or {}
    optimizer = state.get("optimizer") or {}
    lrs = [float(group["lr"]) for group in optimizer.get("param_groups", [])]
    checks = {
        "model hardlink": model.stat().st_ino == source_model.stat().st_ino,
        "state hardlink": state_path.stat().st_ino == source_state.stat().st_ino,
        "state epoch39": int(state.get("epoch", -1)) == SOURCE_LABEL,
        # The historical trainer serializes forced continuation checkpoints before
        # scheduler.step(); ep34/ep39 therefore carry last_epoch=label-1. With no
        # remaining milestones this preserves the fixed optimizer LR exactly.
        "scheduler pre-step epoch38": int(scheduler.get("last_epoch", -1))
        == SOURCE_LABEL - 1,
        "scheduler milestones empty": not dict(scheduler.get("milestones", {})),
        "optimizer groups present": bool(lrs),
        "fixed source LR": bool(lrs) and abs(lrs[0] - 2.5e-5) <= 1e-12,
        "scaler present": bool(state.get("scaler")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Local5 ep39 resume contract failed: {failed}")
    audit = {
        "schema": "local5_fullres_40_to_50_resume_audit_v1",
        "status": "PASS",
        "scope": "model_optimizer_scheduler_scaler_resume_not_rng_bit_exact",
        "source_model": str(source_model.resolve()),
        "source_model_sha256": sha256(source_model),
        "source_state": str(source_state.resolve()),
        "source_state_sha256": sha256(source_state),
        "staged_model": str(model.resolve()),
        "staged_state": str(state_path.resolve()),
        "config": str(CONFIG.resolve()),
        "config_sha256": sha256(CONFIG),
        "checks": checks,
        "optimizer_lrs": lrs,
        "scheduler_save_order": (
            "historical_forced_checkpoint_before_scheduler_step_label_minus_one"
        ),
        "rng_state_present": any("rng" in key.lower() for key in state),
        "does_not_inherit_ep29_hardware_provenance": True,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    record("PASS Local5 ep39 model/optimizer/scheduler/scaler resume audit")


def verify_load_chain() -> None:
    text = (ROOT / "train.log").read_text(encoding="utf-8", errors="replace")
    audits = re.findall(
        r"\[H9\] load audit: checkpoint_overlay_keys=(\d+), missing=(\d+), unexpected=(\d+)",
        text,
    )
    checks = {
        "load audit": bool(audits) and tuple(map(int, audits[-1])) == (210, 0, 0),
        "ATLIF105": "ATLIFTernaryPSN summary: {'num_modules': 105," in text,
        "Shiftmax12": "Shiftmax attention summary: {'num_modules': 12," in text,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Local5 40->50 load chain failed: {failed}")
    record("PASS Local5 load chain overlay=210 missing=0 unexpected=0 ATLIF=105 Shiftmax=12")


def load_profile(label: int) -> dict[str, object]:
    checkpoint = ROOT / f"checkpoint_epoch{label}.pth"
    path = ROOT / f"standard_valid825/epoch{label}/spike_profile.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    protocol = raw.get("eval_protocol") or {}
    audit = raw.get("checkpoint_load_audit") or {}
    counts = raw.get("module_counts") or {}
    identity = raw.get("artifact_identity") or {}
    metrics = raw.get("metrics") or {}
    checks = {
        "resolution": protocol.get("resolution") == [480, 640],
        "crop": protocol.get("crop") is None,
        "window": protocol.get("window_size") == [2, 15, 15],
        "samples": int(raw.get("samples", 0)) == 825,
        "checkpoint SHA": identity.get("checkpoint_sha256") == sha256(checkpoint),
        "config SHA": identity.get("config_sha256") == sha256(CONFIG),
        "overlay": audit.get("checkpoint_overlay_keys") == 210,
        "missing": audit.get("missing_count") == 0,
        "unexpected": audit.get("unexpected_count") == 0,
        "ATLIF105": counts.get("ATLIFTernaryPSN") == 105,
        "Shiftmax12": counts.get("ShiftmaxAttention") == 12,
        "metrics": all(
            key in metrics for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Local5 ep{label} valid825 contract failed: {failed}")
    return {
        "checkpoint_label": label,
        "AEE": float(metrics["AEE"]),
        "AAE": float(metrics["AAE"]),
        "AAE_Benchmark": float(metrics["AAE_Benchmark"]),
        "DSEC_Fl": float(metrics["DSEC_Fl"]),
        "total_spikes_g": float(raw["total_spikes"]) / 1e9,
        "checkpoint_sha256": sha256(checkpoint),
        "profile": str(path.resolve()),
        "profile_sha256": sha256(path),
    }


def write_summary() -> None:
    points = [load_profile(label) for label in EVAL_LABELS]
    rank1 = min(points, key=lambda point: float(point["AEE"]))
    decision = (
        "not_plateaued"
        if int(rank1["checkpoint_label"]) == FINAL_LABEL
        else "operationally_plateaued_or_overfit"
    )
    output = {
        "schema": "local5_fullres_40_to_50_convergence_v1",
        "status": "PASS",
        "criterion": (
            "not_plateaued iff the largest observed budget is AEE rank1; "
            "last-five slope is descriptive only"
        ),
        "budgets": [40, 45, 50],
        "points": points,
        "rank1_checkpoint_label": int(rank1["checkpoint_label"]),
        "decision": decision,
        "hardware_provenance": (
            "none_for_ep44_or_ep49; existing component RTL remains bound to ep29"
        ),
        "resume_audit": str((ROOT / "resume_stage_audit.json").resolve()),
        "resume_audit_sha256": sha256(ROOT / "resume_stage_audit.json"),
    }
    summary_json = ROOT / "convergence_summary.json"
    summary_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Local5 full-resolution budget 40 to 50 convergence audit",
        "",
        "| budget | checkpoint | AEE | AAE-2D | AE-3D | DSEC Fl(%) | spikes(G) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, point in zip((40, 45, 50), points, strict=True):
        lines.append(
            f"| {budget} | {point['checkpoint_label']} | {point['AEE']:.6f} | "
            f"{point['AAE']:.6f} | {point['AAE_Benchmark']:.6f} | "
            f"{point['DSEC_Fl']:.4f} | {point['total_spikes_g']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{decision}`; rank-1 checkpoint is ep{rank1['checkpoint_label']}.",
            "",
            "No ep44/ep49 hardware provenance is claimed; ep29 remains the frozen "
            "Local5 checkpoint-bound component-RTL anchor.",
        ]
    )
    (ROOT / "convergence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker = "LOCAL5_FULLRES_40_TO_50_FINAL_RESULT_20260812"
    for path in (REDESIGN,):
        text = path.read_text(encoding="utf-8")
        if marker in text:
            continue
        doc_lines = [
            "",
            f"<!-- {marker} -->",
            "",
            "### Local5 full-resolution 40→50 收敛结果",
            "",
            "| budget | checkpoint | AEE | AAE-2D | AE-3D | Fl(%) | spikes(G) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for budget, point in zip((40, 45, 50), points, strict=True):
            doc_lines.append(
                f"| {budget} | {point['checkpoint_label']} | {point['AEE']:.6f} | "
                f"{point['AAE']:.6f} | {point['AAE_Benchmark']:.6f} | "
                f"{point['DSEC_Fl']:.4f} | {point['total_spikes_g']:.4f} |"
            )
        doc_lines.extend(
            [
                "",
                f"- 收敛判定=`{decision}`，AEE rank-1=ep{rank1['checkpoint_label']}。",
                "- ep44/49 没有硬件 provenance；Local5 现有 component RTL 仍只绑 ep29。",
                f"- 机器审计：`{summary_json.relative_to(REPO)}`。",
                "",
            ]
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(doc_lines))
    record(f"ALL COMPLETE Local5 40->50 convergence audit: {decision}")


def main() -> int:
    generate_config()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Local5 40->50 watcher already active", flush=True)
            return 0

        while not QF_SUMMARY.is_file():
            record("WAIT H67 QF5-QF8 mainline sensitivity before Local5 extension")
            time.sleep(300)

        if (ROOT / "convergence_summary.json").is_file():
            record("ALL COMPLETE Local5 40->50 convergence audit already exists")
            return 0

        stage_resume()
        if not (ROOT / f"checkpoint_epoch{FINAL_LABEL}.pth").is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/train.py"),
                    "--config",
                    str(CONFIG),
                    "--prev_runid",
                    str(ROOT / f"checkpoint_epoch{SOURCE_LABEL}.pth"),
                    "--save_path",
                    str(ROOT / "checkpoint_epoch{}.pth"),
                    "--finetune",
                    "1",
                    "--resume",
                    "1",
                ],
                ROOT / "train.log",
                "Local5 DSEC full-resolution budget 40->50",
            )
        verify_load_chain()
        run(
            [
                sys.executable,
                "-u",
                str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config",
                str(CONFIG),
                "--run-dir",
                str(ROOT),
                "--ranking-mode",
                "aee",
                *[
                    item
                    for label in EVAL_LABELS
                    for item in ("--epoch", str(label))
                ],
            ],
            ROOT / "valid825.log",
            "Local5 DSEC full-resolution valid825 ep39/44/49",
        )
        write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
