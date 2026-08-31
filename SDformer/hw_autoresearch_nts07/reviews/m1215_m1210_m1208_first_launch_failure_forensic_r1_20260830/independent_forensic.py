#!/usr/bin/env python3
"""Local read-only validator for the M1210/M1208 first-launch forensic."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTER = HW / "scripts/run_m1210_m1208_motion_ep29_unified_capture_secure_remote_one_shot_source.py"
REMOTE_LAUNCHER = HW / "scripts/run_m1208_motion_ep29_unified_capture_remote_one_shot_source.py"
LAUNCH_CONTRACT = HW / "contracts/m1210_m1208_motion_ep29_unified_capture_launch_release_r1_20260830.json"
M1211 = HW / "reviews/m1211_m1210_m1208_motion_ep29_unified_capture_release_hammer_r1_20260830"
MARKER = HW / "results/.m1210_m1208_secure_transfer_and_launch_r1_attempt_consumed"
OBS = HERE / "remote_read_only_observation.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    OUTER: "d3ce4b1e7aa1243b266053ed4f26ba452d25505791a9d328b03b5edda6e8432d",
    REMOTE_LAUNCHER: "273447a1de9708a066e7356a66ef346213cf723c96c7b505a441acc4532dcfae",
    LAUNCH_CONTRACT: "5aeeaf9cab836f32e025f0c329ef1fe90caa4ee3acae691514f4793c1d143829",
    M1211 / "review.json": "813eec1d3fe025a21001c03d8394f32cb646674a1687648e1be2eefb54bb6567",
    M1211 / "SHA256SUMS": "2c02a12fd8f64340306af6b709cab624ed492cd6771e82d6aa21b22088f3730b",
    M1211 / "SHA256SUMS.seal.sha256": "8e62c4a3667540cd797736867b02ba86f9353b92f49c94d4351b775418f9b2fa",
    MARKER: "b60af667912eae9f19fb93aaf201fc342cfdd22e9add4bfeac0e55c09268e5f6",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
ACTUAL_STATUS = "PASS_M1210_SECURE_TRANSFER_AND_ONE_M1208_REMOTE_LAUNCH_AUTHORIZED"
EXPECTED_CAPTURE_SOURCE_CONTRACT_SHA = "dad36c0a264e3e0d3a478929549431453ced60cba84fc24b2d9de442d29faa20"
checks = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_m1211() -> None:
    sums, outer = M1211 / "SHA256SUMS", M1211 / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(sums), "SHA256SUMS"], "M1211 outer seal")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in listed and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "safe M1211 member")
        listed[name] = digest
    actual: set[str] = set()
    for root, dirs, files in os.walk(M1211, followlinks=False):
        base = Path(root); dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            member = base / name; rel = member.relative_to(M1211).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or member.is_symlink():
                continue
            if stat.S_ISREG(member.lstat().st_mode): actual.add(rel)
    require(actual == set(listed), "M1211 complete membership")
    for name, digest in listed.items(): require(sha(M1211 / name) == digest, "M1211 drift " + name)


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift " + str(path))
    verify_m1211()
    review = json.loads((M1211 / "review.json").read_text())
    contract = json.loads(LAUNCH_CONTRACT.read_text())
    launcher = REMOTE_LAUNCHER.read_text()
    outer = OUTER.read_text()
    obs = json.loads(OBS.read_text())

    require(MARKER.read_text(encoding="ascii") ==
            "M1210_TRANSFER_COMPLETE__M1208_REMOTE_LAUNCH_ATTEMPT_CONSUMED__NO_RETRY\n",
            "exact local consumed marker")
    require(stat.S_IMODE(MARKER.stat().st_mode) == 0o400 and MARKER.stat().st_size == 72,
            "marker mode and size")
    require(outer.index("os.open(LOCAL_ATTEMPT") < outer.index("launched = runner("),
            "outer marker precedes sole remote launch")
    require(outer.count("launched = runner(") == 1 and
            "single M1208 remote launch failed; no retry authorized" in outer,
            "outer one launch and no retry")

    require(review["schema"] ==
            "m1211_m1210_m1208_motion_ep29_unified_capture_release_hammer_r1_v1",
            "sealed review schema")
    require(review["status"] == ACTUAL_STATUS, "sealed actual status")
    require(contract["release_hammer_gate"]["required_status"] == ACTUAL_STATUS,
            "launch contract agrees with sealed actual status")
    expected = re.findall(
        r'review\.get\("schema"\)\s*==\s*RELEASE_HAMMER_SCHEMA\s+and\s+'
        r'review\.get\("status"\)\s*==\s*"([^"]+)"', launcher)
    require(expected == ["PASS"], "remote launcher exact expected status")
    require(expected[0] != review["status"], "deterministic status mismatch")
    # The status comparison is the dynamically observed first fault.  Preserve
    # the next static fault separately: this M1211 review authorizes the outer
    # secure publisher and does not expose the inner-launcher binding schema.
    expected_binding_fields = {
        "launcher_sha256", "launch_contract_sha256", "capture_source_sha256",
        "source_contract_sha256", "source_test_sha256",
        "source_hammer_manifest_sha256", "source_hammer_outer_file_sha256",
    }
    actual_binding_fields = set(review["bindings"])
    require(expected_binding_fields - actual_binding_fields == {
        "launcher_sha256", "capture_source_sha256", "source_test_sha256",
        "source_hammer_manifest_sha256", "source_hammer_outer_file_sha256",
    }, "exact latent missing binding fields")
    for field in expected_binding_fields:
        require(('bindings.get("' + field + '")') in launcher,
                "launcher binding expectation " + field)
    require(review["bindings"]["source_contract_sha256"] !=
            EXPECTED_CAPTURE_SOURCE_CONTRACT_SHA,
            "latent same-name source-contract identity mismatch")
    require(review["bindings"]["launch_contract_sha256"] == sha(LAUNCH_CONTRACT),
            "launch-contract binding is the one shared identity")
    require(launcher.index("validate_release_hammer(capture, contract_path)") <
            launcher.index("capture.validate_launch_contract(contract, contract_path)") <
            launcher.index("selected = capture.selected_samples"),
            "failure occurs before contract/sample/GPU workload preflight")
    require(launcher.index("validate_release_hammer(capture, contract_path)") <
            launcher.index("require(gpu_compute_pids() == []"),
            "failure occurs before GPU query and child command construction")

    require(obs["collection_policy"] == {
        "ssh_read_only": True, "remote_writes": 0, "main_calls": 0,
        "execute_once_calls": 0, "child_capture_calls": 0,
        "gpu_jobs_launched": 0, "python_dont_write_bytecode": True},
        "read-only observation policy")
    require(obs["remote_identity"]["launcher_sha256"] == sha(REMOTE_LAUNCHER)
            and obs["remote_identity"]["release_hammer_review_sha256"] == sha(M1211 / "review.json")
            and obs["remote_identity"]["release_hammer_actual_status"] == ACTUAL_STATUS,
            "remote/local exact identity")
    for key in ("m1208_attempt_exists", "m1208_result_exists", "m1208_log_exists"):
        require(obs["namespace_before_preflight_reproduction"][key] is False,
                "remote namespace absent before " + key)
        require(obs["post_reproduction_state"][key] is False,
                "remote namespace absent after " + key)
    reproduction = obs["preflight_only_reproduction"]
    require(reproduction == {
        "called": "run_m1208_motion_ep29_unified_capture_remote_one_shot_source.preflight",
        "main_called": False, "execute_once_called": False, "child_capture_called": False,
        "exception_type": "ReleaseError",
        "exception_message": "M1211 release hammer semantic admission mismatch",
        "unexpected_return": False,
        "namespace_before": [False, False, False],
        "namespace_after": [False, False, False],
        "namespace_unchanged": True}, "exact preflight-only reproduction")
    require(obs["post_reproduction_state"]["capture_related_processes"] == [],
            "no capture/model process")
    require(obs["post_reproduction_state"]["nvidia_smi_compute_app_rows"] == [],
            "no GPU compute application")
    require(obs["namespace_before_preflight_reproduction"]["m1180_attempt_exists"] is True
            and obs["namespace_before_preflight_reproduction"]["m1180_result_exists"] is False
            and obs["namespace_before_preflight_reproduction"]["m1180_log_exists"] is False,
            "prior M1180 read-only state preserved")

    print(json.dumps({
        "schema": "m1215_m1210_m1208_first_launch_failure_forensic_mechanical_v1",
        "status": "PASS_FORENSIC__LOCAL_M1210_CONSUMED__REMOTE_M1208_UNCONSUMED__STATUS_MISMATCH_REPRODUCED",
        "checks_passed": checks,
        "local_m1210_marker_consumed": True,
        "remote_m1208_attempt_result_log_absent": True,
        "remote_preflight_only_exception": "M1211 release hammer semantic admission mismatch",
        "expected_status": "PASS",
        "sealed_actual_status": ACTUAL_STATUS,
        "latent_binding_mismatch_after_status_fix": True,
        "latent_missing_binding_fields": 5,
        "latent_source_contract_identity_mismatch": True,
        "dynamic_first_fault_is_status": True,
        "gpu_compute_apps": 0,
        "capture_related_processes": 0,
        "model_or_capture_workload_started": False,
        "remote_writes_during_forensic": 0,
        "main_calls": 0,
        "execute_once_calls": 0,
        "child_capture_calls": 0,
        "successor_release_input": True,
        "capture_result": False,
        "paper_result": False,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
