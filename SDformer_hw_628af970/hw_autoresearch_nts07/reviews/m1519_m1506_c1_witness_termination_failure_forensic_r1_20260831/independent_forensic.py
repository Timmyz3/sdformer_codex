#!/usr/bin/env python3
"""Read-only M1519 audit of the consumed M1506 evidence.

This script never launches VCS/simv/EDA and never writes evidence. It verifies
the two existing seals, hashes the source and logs, and emits a compact JSON
finding to stdout. A non-zero exit means the reviewed evidence drifted.
"""

import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ATTEMPT = ROOT / "results/.m1506_c1_active_next_oracle_vcs_attempt_consumed"
QUARANTINE = ROOT / (
    "results/m1506_c1_active_next_oracle_unit_delay_vcs_r1_20260831."
    "failed_or_incomplete.quarantine"
)
RAW = ROOT / "results/.m1506_c1_active_next_oracle_raw_build.2572547"
TB = ROOT / (
    "verif_m1497_c1_active_next_oracle_successor/"
    "tb_m1497_m1270r13_m1162_real_m935_protocol_unit_delay.sv"
)
WITNESS = ROOT / (
    "verif_m1337r15_c1_real_m935_runtime_witness/"
    "m1337r15_m935_runtime_witness.sv"
)
SVA = ROOT / (
    "verif_m1168r3_c1_common_charge_protocol/"
    "m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
)
RUNNER = ROOT / (
    "dc_handoff/scripts/"
    "run_m1506_m1497_c1_active_next_oracle_release_safe_successor_one_shot.py"
)
DOC359 = ROOT / "docs/359_DATE终局冻结_20260813.md"


EXPECTED = {
    "attempt_json": "9edc4e2872c5f64c4b395a6c6725a8ce9913506db689067b97947b8ad66354e8",
    "attempt_manifest": "4b0a934bb00569b4f11fbfa3d5c2c32c1873a3feb3c6407a21dc082fa3799757",
    "attempt_outer": "9d8492ae35b61254565f59d11759c99e7ce5a12ed1aec49252a16e624b64564a",
    "compile_log": "87b9a31b46eff9b426dee8aa172abadfd7128a248aa1463ad322c67e934ff275",
    "sim_log": "3cea9ce5685d17e1ef74ec3b7207c7da9a85a36095d8fa98843beb9f75a849eb",
    "identity": "39e7fc74be19dbe0f3acc13a32ed7dcab67c9d986303830444ec5f7b3d98b085",
    "receipt": "63fcea2467beb03d59976c4cee3eab09c7e720dc33fe847400d04adbd7dfc214",
    "quarantine_manifest": "fb1843867c31f9cae154e64ee68b3869729d292c5e7e31678022a38765075cdd",
    "quarantine_outer": "60e0527a166008cb3c84f7fbf60051c7414927fb5b756a9f1df5a47a059dca6c",
    "runner": "9613922eb3aec2c7fe0efa69cafb4fb8337009b26686435f44cc139c774317cc",
    "testbench": "e5604300f3e6cfcbdadfdafa8fae6a2faa6cdc1c18446fa8c48ba6ea10632526",
    "witness": "0ec7179e36f9af09e3020f76a5a927298d877b3cc20c6ac9ab4686bf465d18af",
    "sva": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        chunk = stream.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = stream.read(1024 * 1024)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_manifest(directory):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    checked = []
    for line in manifest.read_text().splitlines():
        expected, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        target = directory / rel
        require(target.is_file() and not target.is_symlink(), f"bad payload {target}")
        require(sha(target) == expected, f"payload hash drift {target}")
        checked.append(rel)
    outer_expected, outer_rel = outer.read_text().split(None, 1)
    require(outer_rel.strip().lstrip("*") == "SHA256SUMS",
            "outer seal target drift")
    require(sha(manifest) == outer_expected, "outer seal mismatch")
    return {"payloads": checked, "inner": True, "outer": True}


def unique_index(lines, predicate, label):
    indices = [i for i, line in enumerate(lines) if predicate(line)]
    require(len(indices) == 1, f"{label} cardinality={len(indices)}")
    return indices[0]


def main():
    attempt_seal = verify_manifest(ATTEMPT)
    quarantine_seal = verify_manifest(QUARANTINE)

    measured_hashes = {
        "attempt_json": sha(ATTEMPT / "attempt.json"),
        "attempt_manifest": sha(ATTEMPT / "SHA256SUMS"),
        "attempt_outer": sha(ATTEMPT / "SHA256SUMS.seal.sha256"),
        "compile_log": sha(QUARANTINE / "compile.log"),
        "sim_log": sha(QUARANTINE / "sim.log"),
        "identity": sha(QUARANTINE / "m1506_c1_active_next_oracle_identity_r1.json"),
        "receipt": sha(QUARANTINE / "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json"),
        "quarantine_manifest": sha(QUARANTINE / "SHA256SUMS"),
        "quarantine_outer": sha(QUARANTINE / "SHA256SUMS.seal.sha256"),
        "runner": sha(RUNNER),
        "testbench": sha(TB),
        "witness": sha(WITNESS),
        "sva": sha(SVA),
        "docs359": sha(DOC359),
    }
    require(measured_hashes == EXPECTED, "reviewed identity drift")
    require(sha(RAW / "compile.log") == measured_hashes["compile_log"],
            "raw/quarantine compile.log differs")
    require(sha(RAW / "sim.log") == measured_hashes["sim_log"],
            "raw/quarantine sim.log differs")

    attempt = json.loads((ATTEMPT / "attempt.json").read_text())
    receipt = json.loads((QUARANTINE /
        "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json").read_text())
    require(attempt["status"] == "M1506_ATTEMPT_CONSUMED", "attempt status drift")
    require(attempt["maximum_vcs_compiles"] == 1, "compile budget drift")
    require(attempt["maximum_simv_runs"] == 1, "sim budget drift")
    require(attempt["automatic_retry"] is False, "retry policy drift")
    require(receipt["status"] == "FAILED_OR_INCOMPLETE", "failure status drift")
    require(receipt["phase"] == "LOG_ADMISSION", "failure phase drift")
    require(receipt["exception"] == "RuntimeError: required log token cardinality",
            "failure exception drift")
    require(receipt["one_shot"]["vcs_compiles"] == 1, "receipt compile count drift")
    require(receipt["one_shot"]["simv_runs"] == 1, "receipt sim count drift")

    compile_log = (QUARANTINE / "compile.log").read_text(errors="replace")
    require("../simv up to date" in compile_log, "compile completion token absent")
    require("CPU time:" in compile_log, "compile timing footer absent")

    sim = (QUARANTINE / "sim.log").read_text(errors="replace")
    lines = sim.splitlines()
    oracle_lines = [line for line in lines if line.startswith("ORACLE_M1270R13 ")]
    require(len(oracle_lines) == 90, f"oracle records={len(oracle_lines)}")
    require(all(" pass=1 " in line for line in oracle_lines), "non-pass oracle exists")

    src_idx = unique_index(lines,
        lambda x: x.startswith("PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE "),
        "source pass")
    finish_idx = unique_index(lines, lambda x: x.startswith("$finish called from file "),
                              "source finish")
    witness_idx = unique_index(lines,
        lambda x: x.startswith("M1337R15_WITNESS_OPERANDS pass=0 stage=3 "),
        "witness fail operands")
    fatal_idx = unique_index(lines,
        lambda x: x.startswith("Fatal: ") and "m1337r15" in x,
        "witness fatal")
    require(src_idx < finish_idx < witness_idx < fatal_idx, "terminal log order drift")
    require("design_issue=2 design_commit=1 design_rows=1 masks=0 faults=0"
            in lines[witness_idx], "witness/design operand drift")
    coverage_line = next(line for line in lines
                         if line.startswith("COVERAGE_M1270R13_REAL_M935 "))
    require("response_cycle_gap=3" in coverage_line, "response gap drift")
    cp_nonfirst = next(line for line in lines if ".cp_nonfirst," in line)
    cp_ii2 = next(line for line in lines if ".cp_ii2," in line)
    require(re.search(r", 147 attempts, 1 match$", cp_nonfirst) is not None,
            "cp_nonfirst result drift")
    require(re.search(r", 147 attempts, 0 match$", cp_ii2) is not None,
            "cp_ii2 result drift")

    tb = TB.read_text()
    witness = WITNESS.read_text()
    sva = SVA.read_text()
    runner = RUNNER.read_text()
    require(tb.index("PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE")
            < tb.index("$finish;"), "TB PASS/finish source order drift")
    require("second_response_cycle - first_response_cycle >= 2" in tb,
            "TB lower-bound II predicate absent")
    require("stage_q === W_TASK_DONE" in witness and
            '$fatal(1, "M1337R15 runtime witness incomplete, unknown, or attacked")'
            in witness, "witness final gate drift")
    require("response_accept ##1 !response_accept ##1 response_accept" in sva,
            "exact-II2 cover drift")
    admission_order = [
        runner.index('raise RuntimeError("required log token cardinality")'),
        runner.index('raise RuntimeError("cp_nonfirst/cp_ii2 coverage missing or duplicate")'),
        runner.index('raise RuntimeError("error/fatal/assertion-failure line")'),
    ]
    require(admission_order == sorted(admission_order), "runner admission order drift")

    result = {
        "schema": "m1519_independent_forensic_checks_v1",
        "status": "PASS_READ_ONLY_FORENSIC",
        "seals": {"attempt": attempt_seal, "quarantine": quarantine_seal},
        "identity_sha256": measured_hashes,
        "raw_log_copy": {
            "compile_byte_exact": True,
            "sim_byte_exact": True,
            "compile_bytes": (RAW / "compile.log").stat().st_size,
            "sim_bytes": (RAW / "sim.log").stat().st_size,
        },
        "execution": {
            "vcs_compiles": 1,
            "simv_runs": 1,
            "automatic_retry": False,
            "runner_reached_log_admission": True,
            "sim_process_return_accepted_by_runner": True,
        },
        "log_evidence": {
            "oracle_records": 90,
            "oracle_pass_records": 90,
            "response_cycle_gap": 3,
            "cp_nonfirst_matches": 1,
            "cp_exact_ii2_matches": 0,
            "source_pass_before_finish": True,
            "witness_after_finish": True,
            "witness_pass": False,
            "witness_stage": 3,
            "witness_responses": 1,
            "witness_core_accepts": 1,
            "witness_psum_commits": 0,
            "witness_rows": 0,
            "witness_tasks": 0,
            "design_issue_accepts": 2,
            "design_psum_commits": 1,
            "design_row_completions": 1,
            "design_faults": 0,
        },
        "classification": {
            "dut_functional_failure_proven": False,
            "dut_functional_pass_admitted": False,
            "primary": "WITNESS_OBSERVATION_AND_TERMINATION_ORDER_MISMATCH",
            "independent_blocker": "EXACT_II2_COVER_EXPECTED_BUT_GAP3_OBSERVED",
            "confidence": "high for admission classification; medium for the exact simulator scheduling edge without waveform evidence",
        },
        "claim_boundary": {
            "m1506_failed_or_incomplete": True,
            "paper_citable": False,
            "functional_vcs": False,
            "timing_verified": False,
            "cycles_measured": False,
            "speedup": False,
            "ppa": False,
            "energy": False,
            "headline": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
