#!/usr/bin/env python3
"""Local5 joint-head profile完成后运行纯CPU Relation Memo联合评估。"""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROFILE = ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
UPSTREAM_STATUS = ROOT / "results/local5_joint_head_profile_watcher_20260809.log"
UPSTREAM_MARKER = "ALL COMPLETE Local5 epoch29 same-window all-head profile100"
OUTPUT = ROOT / "results/local5_joint_relation_memo_final_ep29_20260809"
STATUS = ROOT / "results/local5_joint_relation_memo_watcher_20260809.log"
LOCK = ROOT / "results/local5_joint_relation_memo_watcher_20260809.lock"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_local5_joint_relation_memo as analyzer  # noqa: E402
import model_local5_relation_vault as vault_model  # noqa: E402
import run_local5_joint_head_profile as upstream  # noqa: E402


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def output_complete() -> bool:
    report_path = OUTPUT / "report.json"
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    manifest = PROFILE / "ordered_term_manifest.json"
    plan = PROFILE / "joint_window_selection_plan.json"
    if not manifest.is_file() or not plan.is_file():
        return False
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    payload = PROFILE / str(manifest_value.get("payload_file", ""))
    cohort = PROFILE / str(manifest_value.get("cohort_file", ""))
    gpu_audit = PROFILE / "gpu_exclusivity_audit.json"
    bindings = report.get("source_bindings") or {}
    return (
        report.get("status") == "PROFILE_MODEL_COMPLETE_NOT_RTL"
        and report.get("input", {}).get("sampling_id")
        == "uniform_plan_window_all_heads_v1"
        and report.get("input", {}).get("joint_windows") == 1200
        and report.get("input", {}).get("head_groups") == 13800
        and report.get("input", {}).get("manifest_sha256") == sha256(manifest)
        and payload.is_file()
        and report.get("input", {}).get("payload_sha256") == sha256(payload)
        and report.get("input", {}).get("selection_plan_sha256") == sha256(plan)
        and cohort.is_file()
        and report.get("input", {}).get("cohort_file_sha256") == sha256(cohort)
        and gpu_audit.is_file()
        and report.get("input", {}).get("gpu_exclusivity_audit_sha256")
        == sha256(gpu_audit)
        and report.get("input", {}).get("checkpoint_sha256")
        == manifest_value.get("checkpoint_sha256")
        and report.get("input", {}).get("cohort_sha256")
        == manifest_value.get("cohort_sha256")
        and report.get("method", {}).get("trials") == 20000
        and report.get("method", {}).get("seed") == 20260809
        and bindings.get("analyzer", {}).get("sha256")
        == sha256(Path(analyzer.__file__).resolve())
        and bindings.get("vault_model", {}).get("sha256")
        == sha256(Path(vault_model.__file__).resolve())
    )


def upstream_complete() -> bool:
    return (
        UPSTREAM_STATUS.is_file()
        and UPSTREAM_MARKER
        in UPSTREAM_STATUS.read_text(encoding="utf-8", errors="replace")
        and upstream.output_complete()
    )


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("EXIT another joint Relation Memo watcher owns the lock")
            return 0
        if output_complete() and upstream_complete():
            record("REUSE completed Local5 joint Relation Memo analysis")
            return 0
        while not upstream_complete():
            record("WAIT Local5 same-window all-head profile")
            time.sleep(300)
        command = [
            PYTHON,
            "hw_autoresearch_nts07/scripts/analyze_local5_joint_relation_memo.py",
            "--manifest",
            str(PROFILE / "ordered_term_manifest.json"),
            "--selection-plan",
            str(PROFILE / "joint_window_selection_plan.json"),
            "--output-dir",
            str(OUTPUT),
            "--trials",
            "20000",
            "--seed",
            "20260809",
        ]
        record("START " + " ".join(command))
        result = subprocess.run(command, cwd=REPO)
        record(f"END joint Relation Memo exit_code={result.returncode}")
        if result.returncode or not output_complete():
            raise RuntimeError("Local5 joint Relation Memo分析失败")
        record("ALL COMPLETE Local5 joint Relation Memo exact-head profile model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
