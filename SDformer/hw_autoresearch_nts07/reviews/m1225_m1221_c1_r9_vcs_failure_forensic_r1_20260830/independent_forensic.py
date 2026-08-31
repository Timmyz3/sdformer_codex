#!/usr/bin/env python3
"""Read-only M1225 forensic for the consumed M1221/R9 VCS attempt."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
ATTEMPT = HW / "results/.m1221_m1219r9_m1162_c1_common_charge_protocol_vcs_r9_attempt_consumed"
QUARANTINE = HW / "results/m1221_m1219r9_m1162_c1_common_charge_protocol_unit_delay_vcs_r9_20260830.failed_or_incomplete.983909.quarantine"
TB = HW / "verif_m1219r9_c1_common_charge_protocol/tb_m1219r9_m1162_common_charge_protocol_unit_delay_r9.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
M1162 = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
EXPECTED = {
    ATTEMPT / "identity.txt": "603aee66d300d67e99ab1c5c1dc7428a374549b788310dc8da11c488234d42b9",
    QUARANTINE / "SHA256SUMS": "3aece26a5f503c38c22edddd8b39bda970c096252ef03af0bedd56c7b3ac3156",
    QUARANTINE / "SHA256SUMS.seal.sha256": "e5be966c47771fa11987fe0bde5b5a6669270ffafd70a81d293879dacefa37f1",
    QUARANTINE / "compile.log": "d9c0a35f3b872a8de9bbdb2fd856572dc6e032bb98c4b8e00cceeecc159e3468",
    QUARANTINE / "sim.log": "90e44f850115fe81a22ef5224b1544c5d8150cf43ae47fafa0687f33bdb756a7",
    QUARANTINE / "phase_watchdog_timeout_dump.txt": "599116163205ea593fcfb162f634c5d8274b755bac64a71525a70801822dafd8",
    QUARANTINE / "RUN_FAILED_OR_INCOMPLETE.txt": "d1c98db4f844198832802b4246a11b49ebfd5d5cd4180f2b323eb34f36c92ce7",
    QUARANTINE / "compile.exit_codes": "0ccdb5a77ba5bf7687f2565a8ed97dfb9c1af45503c496fb646312239fab5101",
    QUARANTINE / "sim.exit_codes": "0ccdb5a77ba5bf7687f2565a8ed97dfb9c1af45503c496fb646312239fab5101",
    TB: "9666e086c69ecda4670622e063e9d54c89f94f2c77cd5eb012da54ca23492a75",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    M1162: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    HW / "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recursive_seal(root: Path) -> List[str]:
    errors: List[str] = []
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        return ["outer seal mismatch"]
    listed: Dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1); name = name.lstrip("*")
        if name in listed or Path(name).is_absolute() or ".." in Path(name).parts:
            errors.append("unsafe manifest member " + name); continue
        listed[name] = digest
    actual = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text); dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            path = base / name; rel = path.relative_to(root).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or path.is_symlink(): continue
            if stat.S_ISREG(path.lstat().st_mode): actual.add(rel)
    if actual != set(listed): errors.append("recursive membership drift")
    for name, digest in listed.items():
        if sha(root / name) != digest: errors.append("member drift " + name)
    if len(listed) != 94: errors.append("manifest count")
    return errors


def task(text: str, name: str) -> str:
    match = re.search(r"task\s+automatic\s+" + re.escape(name) + r"\b(.*?)endtask", text, re.S)
    return match.group(0) if match else ""


def main() -> int:
    errors: List[str] = []
    for path, digest in EXPECTED.items():
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            errors.append("identity drift " + str(path))
    errors.extend(recursive_seal(QUARANTINE))
    identity = (ATTEMPT / "identity.txt").read_text()
    for token in (
        "runner_sha256=e018fa988cdf5f1a60033884b3ea5e95e4c4985ac799ead140aaba61e52df1d1",
        "release_sha256=9a1b09c7270c83f795a3fb7a2493941eae24e43601eccc4ee9802f7c3bcc28c0",
        "hammer_review_sha256=4f12ae895a9f806f3e02ff21d2154594dec425dc129f07fd1dc8d2534301593f",
        "hammer_manifest_sha256=eab0b1d680a5ad29e934d747b67eaa1be3af049969d6a69e1937fb09ae0c5a8f",
        "hammer_outer_file_sha256=3ed4c528042c32961b550b7425cd29436f9f1a5e1f3799eb1a07b4fb185162cb",
        "automatic_retry=false"):
        if token not in identity: errors.append("attempt identity " + token)

    compile_codes = (QUARANTINE / "compile.exit_codes").read_text().split()
    sim_codes = (QUARANTINE / "sim.exit_codes").read_text().split()
    sim = (QUARANTINE / "sim.log").read_text()
    dump = (QUARANTINE / "phase_watchdog_timeout_dump.txt").read_text()
    failed = (QUARANTINE / "RUN_FAILED_OR_INCOMPLETE.txt").read_text()
    if compile_codes != ["0", "0"]: errors.append("compile did not pass")
    if sim_codes != ["0", "0"]: errors.append("sim process code")
    if "exit_code=31" not in failed or "functional_vcs_verified=false" not in failed:
        errors.append("runner failure classification")

    phases = ("DIRECTED", "RESET_PENDING", "STICKY_ATTACKS", "SERVICE_ATTACKS")
    for phase in phases:
        if sim.count("PHASE_M1219R9_{}_ENTER".format(phase)) != 1 or \
                sim.count("PHASE_M1219R9_{}_COMPLETE".format(phase)) != 1:
            errors.append("phase incomplete " + phase)
    if sim.count("PHASE_M1219R9_RANDOM_ENTER count=24") != 1 or \
            sim.count("PHASE_M1219R9_RANDOM_COMPLETE count=24") != 1:
        errors.append("random phase incomplete")
    for index in range(24):
        if len(re.findall(r"^PHASE_M1219R9_RANDOM_TRANSACTION_ENTER index={}$".format(index),
                          sim, re.M)) != 1 or \
                len(re.findall(r"^PHASE_M1219R9_RANDOM_TRANSACTION_COMPLETE index={}$".format(index),
                               sim, re.M)) != 1:
            errors.append("random transaction incomplete " + str(index))
    if sim.count("PHASE_M1219R9_CLEAN_RESET_PREP_ENTER limit=16") != 1 or \
            sim.count("PHASE_M1219R9_CLEAN_RESET_PREP_COMPLETE cycles=0") != 1:
        errors.append("clean reset prep proof")
    if sim.count("PHASE_M1219R9_NORMAL_M935_ENTER") != 1 or \
            "PHASE_M1219R9_NORMAL_M935_COMPLETE" in sim:
        errors.append("normal phase boundary")
    if "normal M935 issue request timeout" not in sim or "at time 8113500 ps" not in sim:
        errors.append("terminal timeout identity")
    if "TIMEOUT_M1219R9 " in sim:
        errors.append("unexpected R9 internal timeout token")

    failures = re.findall(r"u_protocol_sva\.(ap_[a-z_]+): started at", sim)
    counts = {name: failures.count(name) for name in sorted(set(failures))}
    expected_counts = {"ap_weight_request_hold": 12, "ap_weight_response_hold": 13,
                       "ap_psum_response_hold": 1}
    if counts != expected_counts: errors.append("SVA failure counts " + repr(counts))

    tb = TB.read_text(); m935 = M935.read_text(); wrapper = M1162.read_text()
    load = task(tb, "load_normal_task"); serve = task(tb, "serve_normal_beat")
    normal = task(tb, "normal_m935_completion")
    structural = {
        "loads_all_64_rows": "row < 64" in load,
        "row0_mask_has_two_sources": "(row == 0) ? 16'h0003 : 16'h0000" in load,
        "calls_two_normal_beats": normal.count("serve_normal_beat(") == 2,
        "first_beat_reached_response_drive": (
            sim.count("ap_weight_response_hold") >= 1 and sim.count("ap_psum_response_hold") >= 1),
        "unsafe_extra_posedge_before_response_retire": (
            "while (!dut.response_accept_w" in serve and
            "@(posedge clk_core); #1ps;\n            @(negedge clk_core);\n            weight_rsp_valid = 1'b0;" in serve),
        "service_ready_not_retired_after_request_fire": (
            "weight_req_ready = 1'b1;" in serve and
            serve.find("weight_req_ready = 1'b1;") < serve.find("weight_rsp_valid = 1'b1;") and
            "weight_req_ready = 1'b0;" not in serve),
        "m935_two_source_issue_rule": (
            "issue_work_mask_w = active_ctx_residual_q;" in m935 and
            "issue_remaining_after_w[issue_source_index_w] = 1'b0;" in m935 and
            "issue_request_valid = exec_active_q && active_ctx_valid_q" in m935),
        "wrapper_spurious_response_sets_boundary_fault": (
            "weight_read_response_valid" in wrapper and
            "&& (!request_active_q || !weight_request_accepted_q)" in wrapper and
            "boundary_fault_q <= 1'b1;" in wrapper),
    }
    if not all(structural.values()): errors.append("structural proof incomplete")

    result = {
        "schema": "m1225_m1221_c1_r9_vcs_failure_forensic_mechanical_r1_v1",
        "status": "PASS_FORENSIC" if not errors else "FAIL_FORENSIC",
        "score": 99 if not errors else 0,
        "p0_count": len(errors), "p1_count": 0, "p2_count": 0,
        "attempt_identity_sha256": sha(ATTEMPT / "identity.txt"),
        "quarantine_manifest_sha256": sha(QUARANTINE / "SHA256SUMS"),
        "quarantine_outer_file_sha256": sha(QUARANTINE / "SHA256SUMS.seal.sha256"),
        "quarantine_manifest_members": 94,
        "compile_exit_codes": compile_codes,
        "sim_exit_codes": sim_codes,
        "runner_exit_code": 31,
        "random_transactions_enter_complete": 24,
        "clean_reset_prep_cycles": 0,
        "normal_issue_timeout_watchdog_cycles": 2000,
        "normal_timeout_sim_time_ps": 8113500,
        "sva_failure_counts": counts,
        "structural_evidence": structural,
        "root_cause": "TB_SERVICE_READY_RESPONSE_RETIRE_ORDER",
        "not_root_cause": ["NO_ISSUABLE_PRODUCT_ROW", "PRIMARY_DUT_BOUNDARY_DEFECT"],
        "boundary_fault_assessment": "LIKELY_SECONDARY_EFFECT_OF_TB_HELD_READY_RESPONSE; NOT_DIRECTLY_DUMPED",
        "authorization": {"source_only_repair_authoring": True, "vcs": False,
                          "eda": False, "rtl_mutation": False, "retry_m1221": False},
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
