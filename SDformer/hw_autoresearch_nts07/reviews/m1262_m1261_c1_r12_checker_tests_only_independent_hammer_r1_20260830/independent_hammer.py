#!/usr/bin/env python3
"""Read-only mutation hammer for M1261.  Never invokes VCS/EDA."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIF = ROOT / "verif_m1258r12_c1_common_charge_protocol"
TB = VERIF / "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"
CHECKER = VERIF / "check_m1261r12_source.py"
TESTS = VERIF / "test_m1261r12_source.py"

EXPECTED = {
    TB: "e13d630f4cf2e2f7e0264dc2325218aee4cc580497be3b37deb1ff7a641ad302",
    CHECKER: "8d5a9cce22b9b3e0ab7907efdc659a4af48e3b87a7e0f9619ecfd52b7777c484",
    TESTS: "9c6514c7d4a40f25693848b0a888ce303da2eb5154e25751f7893beb73c0618d",
    ROOT / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv":
        "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    ROOT / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv":
        "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    ROOT / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv":
        "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    ROOT / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv":
        "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    ROOT / "docs/359_DATE终局冻结_20260813.md":
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_candidate(text):
    with tempfile.NamedTemporaryFile("w", suffix=".sv", delete=False) as handle:
        handle.write(text)
        candidate = Path(handle.name)
    try:
        proc = subprocess.run(
            ["python3", str(CHECKER), "--candidate", str(candidate)],
            cwd=str(ROOT), universal_newlines=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
        payload = json.loads(proc.stdout)
        return proc.returncode, payload
    finally:
        candidate.unlink()


def main():
    findings = []
    checks = []
    for path, expected in EXPECTED.items():
        actual = sha(path)
        checks.append({"name": "frozen_sha:" + str(path.relative_to(ROOT)),
                       "pass": actual == expected, "actual": actual})

    canonical = TB.read_text()
    rc, payload = run_candidate(canonical)
    checks.append({"name": "canonical_accept", "pass": rc == 0 and not payload["errors"]})

    suite = subprocess.run(
        ["python3", "-m", "unittest", "-v", str(TESTS)], cwd=str(ROOT),
        universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    checks.append({"name": "declared_18_tests", "pass": suite.returncode == 0 and
                   "Ran 18 tests" in suite.stdout and "OK" in suite.stdout})

    def attack(name, mutant, should_reject=True):
        rc2, pay2 = run_candidate(mutant)
        rejected = rc2 != 0 and bool(pay2["errors"])
        checks.append({"name": name, "pass": rejected == should_reject,
                       "checker_rejected": rejected, "errors": pay2["errors"]})
        if should_reject and not rejected:
            findings.append(name)

    # Requested nearest-neighbour attacks that the checker correctly rejects.
    attack("valid_shadow", canonical.replace(
        "force dut.u_frozen_m935.issue_request_valid = 1'b1;",
        "force dut.u_frozen_m935.issue_request_valid_shadow = 1'b1;", 1))
    attack("normal_call_commented", canonical.replace(
        "        normal_m935_completion();", "        // normal_m935_completion();", 1))
    attack("normal_call_multiple", canonical.replace(
        "        normal_m935_completion();",
        "        normal_m935_completion();\n        normal_m935_completion();", 1))
    attack("claim_false_with_comment_decoy", canonical.replace(
        "integrated_normal_m935_evidence=true",
        "integrated_normal_m935_evidence=false", 1).replace(
        "// M1258/R12 additive",
        "// integrated_normal_m935_evidence=true\n// M1258/R12 additive", 1))

    # A comment is inert and must remain accepted (positive control).
    comment_control = canonical.replace(
        "// M1258/R12 additive",
        "// force dut.issue_request_valid = 1'b1;\n// M1258/R12 additive", 1)
    rc2, pay2 = run_candidate(comment_control)
    checks.append({"name": "comment_force_inert_positive_control",
                   "pass": rc2 == 0 and not pay2["errors"]})

    # Adversarial cases M1261 must reject but currently accepts.
    phase = "PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_ENTER"
    attack("phase_string_not_display_decoy", canonical.replace(
        '$display("' + phase + '");', '$display("DECOY_PHASE");\n'
        '        string phase_shadow = "' + phase + '";', 1))
    attack("pass_string_not_display_decoy", canonical.replace(
        '$display("PASS_M1258R12_', '$fatal(1, "PASS_M1258R12_', 1).replace(
        '");\n        $finish;', '");\n        $finish;', 1))
    attack("phase_prefix_suffix_near_neighbor", canonical.replace(
        phase, phase + "_SHADOW", 1))
    attack("pass_prefix_suffix_near_neighbor", canonical.replace(
        "PASS_M1258R12_M1162_BOUNDARY_ONLY_SOURCE_CANDIDATE ",
        "PASS_M1258R12_M1162_BOUNDARY_ONLY_SOURCE_CANDIDATE_SHADOW ", 1))
    force_line = "            force dut.u_frozen_m935.issue_request_valid = 1'b1;"
    attack("duplicate_allowed_force_in_helper", canonical.replace(
        force_line, force_line + "\n" + force_line, 1))
    attack("extra_allowed_force_outside_helper", canonical.replace(
        "        $finish;", force_line + "\n        $finish;", 1))

    payload_out = {
        "schema": "m1262_m1261_c1_r12_checker_tests_only_independent_hammer_r1_v1",
        "status": "FAIL_CLOSED_M1261_CHECKER_P1_HOLES__NO_RELEASE_NO_VCS",
        "score": 92,
        "p0_count": 0,
        "p1_count": 4,
        "p2_count": 0,
        "declared_suite_18_pass": next(c["pass"] for c in checks if c["name"] == "declared_18_tests"),
        "unexpected_accepts": findings,
        "unexpected_accept_count": len(findings),
        "checks": checks,
        "authorization": {"release_authoring": False, "vcs": False, "simv": False,
                          "eda": False, "source_mutation": False},
    }
    print(json.dumps(payload_out, indent=2, sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
