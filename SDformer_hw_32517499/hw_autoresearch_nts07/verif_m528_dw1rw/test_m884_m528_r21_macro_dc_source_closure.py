#!/usr/bin/env python3
"""No-EDA source closure for the M884 C1 macro-aware product point."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_exact_sha_r1.sh"
TCL = ROOT / "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_candidate.tcl"
FILELIST = ROOT / "dc_handoff/filelists/date_m884_m528_r21_macro_aware_product_dc.f"
SDC = ROOT / "dc_handoff/constraints/date_m884_m528_r21_macro_aware_product_3ns.sdc"
CONTRACT = ROOT / "contracts/m884_m528_r21_macro_aware_product_dc_source_only_contract_r1_20260829.json"
CANDIDATE = ROOT / "contracts/m884_m528_r21_macro_aware_product_dc_launch_candidate_source_only_r1_20260829.json"
CANONICAL = ROOT / "dc_handoff/runs/m884_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829"
ATTEMPT = ROOT / "dc_handoff/runs/.m884_m528_r21_macro_aware_product_dc_attempt_consumed"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def strict_load_bytes(payload):
    return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def strict_load(path):
    return strict_load_bytes(Path(path).read_bytes())


def verify_file_seal(path):
    path = Path(path)
    subprocess.check_call(["sha256sum", "-c", path.name + ".sha256"],
                          cwd=str(path.parent), stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", path.name + ".sha256.seal.sha256"],
                          cwd=str(path.parent), stdout=subprocess.DEVNULL)


def validate_contract(contract):
    require(set(contract) == {
        "authorization", "claim_boundary", "date", "docs359_sha256",
        "exact_files", "fairness", "foundry_views", "frozen_authorities",
        "future_release_chain", "physical_point", "schema", "status",
        "tool_identity",
    }, "contract top-level keys not closed")
    require(contract["schema"] ==
            "m884_m528_r21_macro_aware_product_dc_source_only_contract_v1",
            "contract schema drift")
    require(contract["authorization"] == {
        "author_ran_eda": False, "run_dc_now": False,
        "run_formality_now": False, "run_pt_now": False,
        "run_ptpx_now": False, "run_remote_now": False,
        "run_saif_now": False, "run_vcs_now": False,
    }, "contract authorization drift")
    require(contract["fairness"] == {
        "bit_rtl_baseline_present": False, "candidate_point_only": True,
        "fair_K_zero_bit": False, "zero_rtl_baseline_present": False,
    }, "fairness drift")
    for key in ["throughput_per_mm2", "speedup", "system_speedup",
                "paper_ppa_ready", "ppa", "energy", "power"]:
        require(contract["claim_boundary"][key] is False,
                "forbidden claim became true: " + key)
    require(contract["physical_point"]["clock_period_ns"] == 3.0,
            "clock drift")
    require(contract["physical_point"]["macro_count"] == 9,
            "macro count drift")
    require(contract["physical_point"]["hold_diagnostic_only"] is True,
            "hold boundary drift")
    require(contract["physical_point"]["macro_slow_fast_min_pair"] is True,
            "macro min-pair absent")


def validate_candidate(contract, candidate):
    require(set(candidate) == {
        "authorization", "claim_boundary", "date", "docs359_sha256",
        "fairness", "frozen_authorities", "future_release_chain", "identity",
        "launch_now", "prospective_attempt", "schema", "status",
    }, "candidate top-level keys not closed")
    require(candidate["schema"] ==
            "m884_m528_r21_macro_aware_product_dc_launch_candidate_source_only_v1",
            "candidate schema drift")
    require(candidate["status"] ==
            "READY_FOR_FRESH_M884_SOURCE_HAMMER__NO_EDA_AUTHORIZED",
            "candidate status drift")
    require(candidate["launch_now"] is False, "candidate launch drift")
    require(candidate["authorization"] == {
        "max_attempts": 0, "run_dc": False, "run_formality": False,
        "run_pt": False, "run_ptpx": False, "run_remote": False,
        "run_saif": False, "run_vcs": False,
    }, "candidate authorization drift")
    require(candidate["identity"]["runner_sha256"] == sha(RUNNER),
            "runner SHA binding drift")
    require(candidate["identity"]["source_contract_sha256"] == sha(CONTRACT),
            "contract SHA binding drift")
    require(candidate["fairness"] == contract["fairness"],
            "contract/candidate fairness mismatch")
    require(candidate["frozen_authorities"] == contract["frozen_authorities"],
            "authority mismatch")
    require(candidate["future_release_chain"] == contract["future_release_chain"],
            "release-chain mismatch")


def mutation_rejected(contract, candidate, mutator):
    trial = copy.deepcopy(candidate)
    mutator(trial)
    try:
        validate_candidate(contract, trial)
    except (KeyError, RuntimeError, TypeError):
        return True
    return False


REQUIRED_OUTPUTS = [
    "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v",
    "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.sdc",
    "netlist/m528_dead_write_only_1rw_product_capture_island_r2.ddc",
    "netlist/m528_dead_write_only_1rw_product_capture_island_r2.svf",
    "reports/area_hierarchy.rpt", "reports/qor.rpt",
    "reports/timing_setup.rpt", "reports/timing_hold_diagnostic.rpt",
    "reports/macro_binding_audit.txt", "m884_dc_receipt.json",
]


def validate_fake_artifacts(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "artifact root invalid")
    actual = []
    for path in root.rglob("*"):
        require(not path.is_symlink(), "artifact symlink")
        if path.is_file():
            require(path.stat().st_size > 0, "zero-byte artifact")
            actual.append(path.relative_to(root).as_posix())
    require(set(actual) == set(REQUIRED_OUTPUTS), "artifact set not closed")
    receipt = strict_load(root / "m884_dc_receipt.json")
    require(set(receipt) == {"claim_boundary", "schema", "status"},
            "receipt keys not closed")
    require(receipt["claim_boundary"] == {
        "fair_K_zero_bit": False, "paper_ppa_ready": False,
        "speedup": False, "system_speedup": False,
        "throughput_per_mm2": False,
    }, "receipt claim boundary drift")


def artifact_negative_suite(tmp):
    base = Path(tmp) / "artifacts"
    for relative in REQUIRED_OUTPUTS:
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload\n")
    (base / "m884_dc_receipt.json").write_text(json.dumps({
        "claim_boundary": {"fair_K_zero_bit": False,
                           "paper_ppa_ready": False, "speedup": False,
                           "system_speedup": False,
                           "throughput_per_mm2": False},
        "schema": "fixture", "status": "fixture",
    }, sort_keys=True) + "\n")
    validate_fake_artifacts(base)
    negatives = 0
    mutations = ["missing", "zero", "symlink", "unknown", "receipt_unknown",
                 "receipt_nan", "receipt_duplicate"]
    for mutation in mutations:
        trial = Path(tmp) / ("artifact_" + mutation)
        shutil.copytree(base, trial)
        if mutation == "missing":
            (trial / REQUIRED_OUTPUTS[0]).unlink()
        elif mutation == "zero":
            (trial / REQUIRED_OUTPUTS[1]).write_bytes(b"")
        elif mutation == "symlink":
            target = trial / REQUIRED_OUTPUTS[2]
            target.unlink()
            target.symlink_to("/etc/passwd")
        elif mutation == "unknown":
            (trial / "unexpected.txt").write_text("x\n")
        elif mutation == "receipt_unknown":
            payload = strict_load(trial / "m884_dc_receipt.json")
            payload["unknown"] = 1
            (trial / "m884_dc_receipt.json").write_text(json.dumps(payload) + "\n")
        elif mutation == "receipt_nan":
            (trial / "m884_dc_receipt.json").write_text(
                '{"claim_boundary":{},"schema":"x","status":"x","x":NaN}\n')
        elif mutation == "receipt_duplicate":
            (trial / "m884_dc_receipt.json").write_text(
                '{"claim_boundary":{},"schema":"x","schema":"y","status":"x"}\n')
        try:
            validate_fake_artifacts(trial)
        except (OSError, RuntimeError, ValueError):
            negatives += 1
    require(negatives == len(mutations), "artifact negative escaped")
    return negatives


def main():
    require(not CANONICAL.exists(), "canonical result exists")
    require(not ATTEMPT.exists(), "attempt sentinel exists")
    require(not list((ROOT / "dc_handoff/runs").glob(
        ".m884_m528_r21_macro_aware_product_dc_work.*")), "work residue exists")
    require(not list((ROOT / "dc_handoff/runs").glob(
        "m884_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829.failed_or_incomplete.*")),
        "quarantine residue exists")
    for source in [RUNNER, TCL, FILELIST, SDC, CONTRACT, CANDIDATE]:
        require(source.is_file() and not source.is_symlink(),
                "source missing/symlink: " + str(source))
        verify_file_seal(source)
    contract = strict_load(CONTRACT)
    candidate = strict_load(CANDIDATE)
    validate_contract(contract)
    validate_candidate(contract, candidate)
    for relative, expected in contract["exact_files"].items():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(),
                "exact file absent/symlink: " + relative)
        require(sha(path) == expected, "exact SHA drift: " + relative)
    require(mutation_rejected(contract, candidate,
                              lambda x: x.update({"unknown": 1})),
            "unknown candidate key escaped")
    require(mutation_rejected(contract, candidate,
                              lambda x: x["authorization"].update({"run_dc": True})),
            "launch mutation escaped")
    require(mutation_rejected(contract, candidate,
                              lambda x: x["fairness"].update({"fair_K_zero_bit": True})),
            "fairness mutation escaped")
    strict_negatives = 0
    for payload in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}']:
        try:
            strict_load_bytes(payload)
        except ValueError:
            strict_negatives += 1
    require(strict_negatives == 3, "strict JSON negative escaped")
    tcl = TCL.read_text()
    for token in [
        "analyze -format sverilog -define SYNTHESIS", "set_min_library $std_slow_db",
        "set_min_library $macro_slow_db", "report_area -hierarchy",
        "report_timing -delay_type max", "report_timing -delay_type min",
        "constraint_setup.rpt", "constraint_hold_diagnostic.rpt",
        "constraint_max_capacitance.rpt", "constraint_max_transition.rpt",
        "constraint_max_fanout.rpt", "write_file -format verilog", "write_sdc",
        "write -format ddc", "set_svf", "macro_count_pre",
        "macro_count_post", "get_lib_cells -quiet */$macro_cell",
    ]:
        require(token in tcl, "Tcl token absent: " + token)
    require(tcl.count("compile_ultra") == 2,
            "Tcl should contain one command and one flow receipt token")
    require("set_fix_hold" not in tcl and "-only_hold_time" not in tcl,
            "hold optimization entered diagnostic-only flow")

    tmp = tempfile.mkdtemp(prefix="m884_source_closure.", dir="/tmp")
    try:
        artifact_negatives = artifact_negative_suite(tmp)
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "M884_NO_EDA_FULL_PATH_SELFTEST": "1",
            "M884_NO_EDA_SELFTEST_ROOT": tmp,
            "M884_EXPECTED_DC_RUNNER_SHA256": sha(RUNNER),
            "M884_EXPECTED_DC_ADMISSION_SHA256": sha(CANDIDATE),
        }
        completed = subprocess.run([str(RUNNER)], cwd=str(ROOT), env=env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        require(completed.returncode == 0,
                "runner full-path no-EDA failed: " +
                completed.stderr.decode("utf-8", errors="replace"))
        marker = (Path(tmp) / "FULL_PATH_PASS.txt").read_text()
        require("status=PASS_M884_FULL_ADMISSION_CONTRACT_PATH_NO_EDA" in marker,
                "full-path marker absent")
        require("attempt_consumed=false" in marker and
                "license_query_started=false" in marker and
                "dc_shell_started=false" in marker,
                "no-EDA boundary drift")
    finally:
        shutil.rmtree(tmp)
    require(not CANONICAL.exists() and not ATTEMPT.exists(),
            "source selftest changed canonical coordinate")
    print("PASS M884 source closure; strict_json_negatives=3; "
          "artifact_negatives=%d; full_path_no_eda=1" % artifact_negatives)


if __name__ == "__main__":
    main()
