#!/usr/bin/env python3
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HW = Path(__file__).resolve().parents[1]
WRAPPER = HW / "dc_handoff/scripts/run_m1302_m1288_fixed_t10_ptsta_exact_closed_one_shot.sh"
ADMISSION = HW / "contracts/m1289_m1288_c3_m917_fixed_t10_ptsta_launch_admission_r1_20260830.json"
CONTRACT = HW / "contracts/m1302_m1288_c3_fixed_t10_ptsta_launch_wrapper_source_contract_r1_20260830.json"
M1288 = HW / "dc_handoff/scripts/run_m1288_m917_fixed_t10_ptsta_inert_exact_sha.sh"
M1299 = HW / "reviews/m1299_m1288_c3_m917_fixed_t10_ptsta_receipt_blind_hammer_r1_20260830"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(obj, keys):
    assert type(obj) is dict and set(obj) == set(keys)


def sealed_payload(path):
    digest_line = Path(str(path) + ".sha256").read_text().split()
    seal_line = Path(str(path) + ".sha256.seal.sha256").read_text().split()
    assert digest_line == [sha(path), path.name]
    assert seal_line == [sha(Path(str(path) + ".sha256")), path.name + ".sha256"]


def extract_adjudicator(source):
    marker = 'python3 - "${m1302_m1288_canonical}" "${m1302_work}" <<\'PY\'\n'
    assert marker in source
    return source.split(marker, 1)[1].split("\nPY\n", 1)[0]


def coverage_row(name, total=10, met=10, violated=0, untested=0):
    return "%s %d %d (100.00%%) %d (0.00%%) %d (0.00%%)\n" % (
        name, total, met, violated, untested)


def run_mock(code, hold=-0.01, unconstrained=0, untested=0):
    with tempfile.TemporaryDirectory(prefix="m1302_mock.") as tmp:
        root = Path(tmp)
        src = root / "m1288"
        out = root / "m1302"
        reports = src / "reports"
        reports.mkdir(parents=True)
        out.mkdir()
        (reports / "timing_setup_slow.rpt").write_text("slack (MET) 0.100000\n")
        hs = "MET" if hold >= 0 else "VIOLATED"
        (reports / "timing_hold_fast.rpt").write_text("slack (%s) %.6f\n" % (hs, hold))
        cov = "".join(coverage_row(n, untested=untested) for n in (
            "setup", "hold", "out_setup", "out_hold"))
        (reports / "analysis_coverage.rpt").write_text(cov)
        if unconstrained:
            check = ("Warning: There are %d input ports with no input delay specified. "
                     "The paths from such ports will be unconstrained.\n" % unconstrained)
        else:
            check = "No unconstrained endpoint diagnostics.\n"
        (reports / "check_timing.rpt").write_text(check)
        (reports / "constraint_violators.rpt").write_text("No violators.\n")
        (src / "m1288_m917_fixed_t10_prelayout_ptsta_receipt_r1.json").write_text(
            json.dumps({"status": "fixture"}))
        completed = subprocess.run(
            [sys.executable, "-", str(src), str(out)], input=code,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert completed.returncode == 0, completed.stderr
        receipt = json.loads((out / "m1302_adjudication_receipt_r1.json").read_text())
        return receipt, (out / "GATE_EXIT_CODE.txt").read_text().strip()


def main():
    checks = {}
    source = WRAPPER.read_text()
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    checks["bash_syntax"] = True

    admission = json.loads(ADMISSION.read_text())
    contract = json.loads(CONTRACT.read_text())
    exact(admission, ("schema", "date", "milestone", "status", "objective",
        "identity", "exact_files", "tool", "preflight", "authorization",
        "result_adjudication", "claim_boundary"))
    assert admission["schema"] == "m1302_m1288_c3_fixed_t10_ptsta_exact_closed_launch_admission_v1"
    assert admission["status"] == "AUTHORIZED_ONE_M1288_M917_FIXED_T10_PTSTA_ATTEMPT"
    exact(admission["authorization"], ("launch_now", "launch_after_independent_hammer",
        "max_attempts", "run_pt", "run_dc", "run_vcs", "run_formality",
        "run_ptpx", "run_remote", "query_license", "result_adjudication"))
    assert admission["authorization"] == {
        "launch_now": False, "launch_after_independent_hammer": True,
        "max_attempts": 1, "run_pt": True, "run_dc": False, "run_vcs": False,
        "run_formality": False, "run_ptpx": False, "run_remote": False,
        "query_license": True, "result_adjudication": True}
    exact(admission["claim_boundary"], ("launch_admission_only", "pt_executed",
        "setup_completed", "hold_closed", "coverage_closed",
        "unconstrained_paths_zero", "automatic_hold_fix", "power", "energy",
        "speedup", "system", "paper_ppa_ready", "headline"))
    assert admission["claim_boundary"]["launch_admission_only"] is True
    assert all(v is False for k, v in admission["claim_boundary"].items()
               if k != "launch_admission_only")
    checks["admission_exact_closed"] = True

    assert admission["identity"]["wrapper_sha256"] == sha(WRAPPER)
    assert admission["identity"]["runner_sha256"] == sha(M1288)
    assert admission["identity"]["m1302_contract_sha256"] == sha(CONTRACT)
    assert admission["identity"]["m1299_outer_seal_sha256"] == sha(M1299 / "SHA256SUMS.seal.sha256")
    for relative, digest in admission["exact_files"].items():
        path = HW / relative
        assert path.is_file() and not path.is_symlink() and sha(path) == digest
    assert sha(DOC359) == DOC359_SHA
    sealed_payload(ADMISSION)
    sealed_payload(CONTRACT)
    checks["identities_and_double_seals"] = True

    exact(contract, ("schema", "date", "status", "objective", "identity",
        "frozen_inputs", "preflight", "result_gate", "authorization",
        "claim_boundary"))
    assert contract["status"] == "M1302_SOURCE_ONLY__NO_PT_EDA_LICENSE_QUERY_EXECUTED"
    assert contract["identity"]["wrapper_sha256"] == sha(WRAPPER)
    assert contract["identity"]["test_sha256"] == sha(Path(__file__))
    assert contract["authorization"] == {
        "launch_now": False, "max_attempts_now": 0, "run_pt_now": False,
        "run_dc_now": False, "run_vcs_now": False, "run_formality_now": False,
        "run_ptpx_now": False, "run_remote_now": False,
        "query_license_now": False, "independent_receipt_blind_hammer_required": True}
    checks["source_contract_exact_closed"] = True

    order = [
        source.index('m1302_sealed_payload_ok "${m1302_admission}"'),
        source.index('[[ -z "$(m1302_collisions)" ]]'),
        source.index('m1302_mem_available="$(awk'),
        source.index('"${m1302_lmutil}" lmstat'),
        source.index('mkdir "${m1302_attempt}"'),
        source.index('/usr/bin/bash "${m1302_m1288_runner}"')]
    assert order == sorted(order)
    assert "! -e \"${m1302_m1288_attempt}\"" in source
    assert "! -e \"${m1302_attempt}\"" in source
    assert "m1302_double_seal_ok \"${m1302_m1288_canonical}\"" in source
    checks["preflight_attempt_launch_order"] = True

    assert "fix_eco_timing" not in source and "set_fix_hold" not in source
    assert "report_power" not in source and "update_power" not in source
    assert source.count('"${m1302_pt}"') == 1  # exact identity check, never direct launch
    assert '"${m1302_lmutil}" lmstat' in source
    checks["no_eco_power_or_direct_pt"] = True

    adjudicator = extract_adjudicator(source)
    passed, rc = run_mock(adjudicator, hold=0.0, unconstrained=0, untested=0)
    assert passed["strict_timing_gate_pass"] is True and rc == "0"
    negative, rc = run_mock(adjudicator, hold=-0.05, unconstrained=0, untested=0)
    assert negative["strict_timing_gate_pass"] is False and rc == "10"
    unconst, rc = run_mock(adjudicator, hold=0.0, unconstrained=2, untested=0)
    assert unconst["strict_timing_gate_pass"] is False and unconst["unconstrained_paths"] == 2 and rc == "10"
    uncovered, rc = run_mock(adjudicator, hold=0.0, unconstrained=0, untested=1)
    assert uncovered["strict_timing_gate_pass"] is False and rc == "10"
    for payload in (passed, negative, unconst, uncovered):
        assert payload["claim_boundary"] == {
            "fresh_result_hammer_required": True, "power": False, "energy": False,
            "speedup": False, "system": False, "paper_ppa_ready": False,
            "headline": False}
    checks["strict_result_mocks"] = True

    assert not (HW / "dc_handoff/runs/m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830").exists()
    assert not (HW / "dc_handoff/runs/.m1288_m917_fixed_t10_ptsta_attempt_consumed").exists()
    assert not (HW / "dc_handoff/runs/m1302_m1288_fixed_t10_ptsta_adjudication_r1_20260830").exists()
    assert not (HW / "dc_handoff/runs/.m1302_m1288_fixed_t10_ptsta_attempt_consumed").exists()
    checks["future_namespaces_fresh"] = True

    print(json.dumps({"schema": "m1302_author_static_test_v1", "status": "PASS",
                      "checks": checks, "pt_eda_license_calls": 0},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
