#!/usr/bin/env python3
"""等待 Local5 joint-head trace 后运行纯 CPU 硬件合同分析。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROFILE = ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
OUTPUT = ROOT / "results/local5_joint_hardware_contract_20260810"
STATUS = ROOT / "results/local5_joint_hardware_contract_watcher_20260810.log"
LOCK = ROOT / "results/local5_joint_hardware_contract_watcher_20260810.lock"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
sys.path.insert(0, str(ROOT / "scripts"))

import run_local5_joint_head_profile as upstream  # noqa: E402
import analyze_local5_joint_hardware_contract as analyzer  # noqa: E402
import analyze_local5_active_tcfm5_postg0 as tcfm5  # noqa: E402


def sha256(path: Path) -> str:
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
    report_md_path = OUTPUT / "report.md"
    commit_path = OUTPUT / "commit.json"
    if not report_path.is_file() or not report_md_path.is_file() or not commit_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    manifest_path = PROFILE / "ordered_term_manifest.json"
    plan_path = PROFILE / "joint_window_selection_plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    payload_path = PROFILE / str(manifest.get("payload_file", ""))
    bindings = report.get("source_bindings") or {}
    report_input = report.get("input") or {}
    bound_files = (
        ("run_identity_file", "run_identity_file_sha256"),
        ("cohort_file", "cohort_file_sha256"),
        ("gpu_exclusivity_audit", "gpu_exclusivity_audit_sha256"),
    )
    for path_key, sha_key in bound_files:
        bound_path = Path(str(report_input.get(path_key, "")))
        if not bound_path.is_file() or report_input.get(sha_key) != sha256(bound_path):
            return False
    return (
        report.get("schema") == "local5_joint_hardware_contract_v1"
        and report.get("status") == "PROFILE_CONTRACT_COMPLETE_NOT_RTL"
        and report.get("input", {}).get("samples") == 100
        and report.get("input", {}).get("joint_windows") == 1200
        and report.get("input", {}).get("head_groups") == 13800
        and report.get("input", {}).get("manifest_sha256") == sha256(manifest_path)
        and payload_path.is_file()
        and report.get("input", {}).get("payload_sha256") == sha256(payload_path)
        and report.get("input", {}).get("selection_plan_sha256") == sha256(plan_path)
        and bindings.get("analyzer", {}).get("sha256")
        == sha256(Path(analyzer.__file__).resolve())
        and bindings.get("tcfm5_model", {}).get("sha256")
        == sha256(Path(tcfm5.__file__).resolve())
        and commit.get("schema") == "local5_joint_hardware_contract_commit_v1"
        and commit.get("status") == "COMMITTED"
        and commit.get("report_json_sha256") == sha256(report_path)
        and commit.get("report_md_sha256") == sha256(report_md_path)
        and commit.get("manifest_sha256") == report["input"]["manifest_sha256"]
        and commit.get("payload_sha256") == report["input"]["payload_sha256"]
        and commit.get("selection_plan_sha256")
        == report["input"]["selection_plan_sha256"]
        and commit.get("identity_sha256")
        == report["input"]["run_identity_file_sha256"]
        and commit.get("cohort_file_sha256")
        == report["input"]["cohort_file_sha256"]
        and commit.get("gpu_exclusivity_audit_sha256")
        == report["input"]["gpu_exclusivity_audit_sha256"]
    )


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("EXIT another Local5 hardware-contract watcher owns the lock")
            return 0
        if output_complete() and upstream.output_complete():
            record("REUSE completed Local5 joint hardware contract")
            return 0
        while not upstream.output_complete():
            record("WAIT Local5 same-window all-head profile")
            time.sleep(300)
        command = [
            PYTHON,
            "hw_autoresearch_nts07/scripts/analyze_local5_joint_hardware_contract.py",
            "--manifest",
            str(PROFILE / "ordered_term_manifest.json"),
            "--selection-plan",
            str(PROFILE / "joint_window_selection_plan.json"),
            "--output-dir",
            str(OUTPUT),
        ]
        record("START " + " ".join(command))
        result = subprocess.run(command, cwd=REPO)
        record(f"END Local5 joint hardware contract exit_code={result.returncode}")
        if result.returncode or not output_complete():
            raise RuntimeError("Local5 joint hardware contract 分析失败")
        record("ALL COMPLETE Local5 joint hardware contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
