#!/usr/bin/env python3
"""Different-author read-only hammer for M1695 C1 fast-min hold source."""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
TCL = HW / "dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_candidate.tcl"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_exact_one_shot.sh"
TEST = HW / "system_simulator/tests/test_m1695_c1_fastmin_hold_closure_source.py"
CONTRACT = HW / "contracts/m1695_m1665_c1_fastmin_hold_closure_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1695_m1665_c1_fastmin_hold_closure_source_author_receipt_r1_20260901"
M1665 = HW / "dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901"
ORIGINAL = M1665 / "original_quarantine"
M1678 = HW / "dc_handoff/runs/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_r1_20260901.failed_or_incomplete.1991841.quarantine"
FUTURE_REVIEW = HW / "reviews/m1696_m1695_c1_fastmin_hold_closure_source_hammer_r1_20260901"
FUTURE_RELEASE = HW / "contracts/m1697_m1696_m1695_c1_fastmin_hold_closure_launch_release_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1695_m1665_c1_fastmin_hold_closure_dc_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1695_m1665_c1_fastmin_hold_closure_dc_attempt_consumed"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    TCL: "cb05b053078c7ab9d084cddf5028802aeff52ef1a4aef6d1b026ba6da2f41ad8",
    RUNNER: "f470eee1f4f68be76d4d680522efca4157472582e9f442721ef836bd5957ca5d",
    TEST: "8b70e57e538e9c45b75e6fa91d1affc1918381bc0571573059a695f0799ea74b",
    CONTRACT: "6a22bc25666de314760eb88cb5843690ad70dd98ba061d2798a5590fb6b30064",
    Path(str(CONTRACT) + ".sha256"): "ab6b21d6e976db1c7b70e0b1d76ddfe5a3bd0cf1170ba75250c78cfafdd34dc2",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "a8bee8c0d784bec7887b02ee09ed9c5832cbbcc7575fcfcaf0e2fbe11873fa80",
    AUTHOR / "author_receipt.json": "9e6875911f3b346ecf64d6cdf796257166c60591271e7d6d0b5b41a3cd84bf15",
    AUTHOR / "SHA256SUMS": "27510eb6b5987b9345b5ef9b4ac5d55fc513ff886666dbd191bf837ee47b5707",
    AUTHOR / "SHA256SUMS.seal.sha256": "2d19864994609ec073ed3dfa4626fda22a3e5fd5d5bc7669fbd16f39d0e6b1a4",
    M1665 / "SHA256SUMS": "a16b9fb100bf7f1b3c6e7453035a5bf89a8f2ffbbeeca1d373038f6e899dba72",
    M1665 / "SHA256SUMS.seal.sha256": "12d87acb439b0cc171d3f42cd4f169fa6a531946c9c3c120cc9babc9c36fbc08",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc": "2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc": "5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198",
    M1678 / "SHA256SUMS": "9556e3bfab30af74326473f6cb9e492d41d3b782d0f23fabb6564626ce6fc675",
    M1678 / "SHA256SUMS.seal.sha256": "7b90352dd62288415f12903cbc4c2745cf2f2fa574080b37f63871015bc77602",
    M1678 / "rtl_to_m993/FORMALITY_INTERNAL_COMPLETE.txt": "9eee52aa958d835e9b682d99e5b52cfed515bacee74854fb8f0a4a8ddfab7eb9",
    M1678 / "m993_to_m1665/FORMALITY_INTERNAL_COMPLETE.txt": "b27aeb9e49081c6fbc238a082dfe7c364270e25ca11579e7ee73c717d0a12fd8",
    M1678 / "ptsta/reports/global_timing.rpt": "c323bdd22a6f9137ee02f85aba0ed9c7792cf1febd6d8c3b11fb2650d41f7557",
    M1678 / "ptsta/reports/timing_setup_slow.rpt": "c0dc0bce139cdf1f8be3058c43bc40ed5b67fa8c2c82292b7265f0f232f35495",
    M1678 / "ptsta/reports/timing_hold_fast.rpt": "eeacd609124059018fdc1bbdafd460342adcc524473d0769c4d43daa43aa3445",
    M1678 / "ptsta/reports/constraint_violators.rpt": "d974d269d592fe02ea04db0c062c8061bba1f8d6e67fd479bb929a1da97526eb",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise HammerError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def no_duplicates(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=no_duplicates,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HammerError("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def commands(text):
    return "\n".join(row.split("#", 1)[0] for row in text.splitlines())


def verify_tree(root, strict_population=True):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "tree absent/symlink " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii") ==
            sha256(manifest) + "  SHA256SUMS\n", "outer mismatch " + str(root))
    sealed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "malformed manifest row")
        name = fields[1].strip().lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and
                name not in sealed, "unsafe/duplicate manifest row")
        member = root / rel
        require(stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink(),
                "nonregular sealed member " + name)
        require(sha256(member) == fields[0], "sealed SHA drift " + name)
        sealed[name] = fields[0]
    if strict_population:
        actual = set()
        for base, dirs, files in os.walk(str(root), followlinks=False):
            base = Path(base)
            dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
            for name in files:
                path = base / name
                if path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                    require(stat.S_ISREG(path.lstat().st_mode) and
                            not path.is_symlink(), "nonregular actual member")
                    actual.add(path.relative_to(root).as_posix())
        require(actual == set(sealed), "recursive population mismatch " + str(root))
    return {"entries": len(sealed), "manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer)}


def validate_tcl(text):
    cmd = commands(text)
    forbidden = ("read_verilog", "compile_ultra", "set_false_path",
                 "set_multicycle_path", "set_min_delay", "set_max_delay",
                 "set_disable_timing", "set_case_analysis")
    for token in forbidden:
        require(re.search(r"(?m)^\s*" + token + r"\b", cmd) is None,
                "forbidden TCL command " + token)
    require(len(re.findall(r"(?m)^\s*read_ddc\s+\$input_ddc\s*$", cmd)) == 1,
            "M1665 DDC not unique")
    require(len(re.findall(r"(?m)^\s*read_sdc\s+\$input_sdc\s*$", cmd)) == 1,
            "mapped SDC not unique")
    require(text.count("set_min_library $std_slow_db -min_version $std_fast_db") == 1 and
            text.count("set_min_library $macro_slow_db -min_version $macro_fast_db") == 1,
            "slow-to-fast min library binding drift")
    require(len(re.findall(r"(?m)^\s*set_fix_hold\s+\$core_clock\s*$", cmd)) == 1,
            "set_fix_hold count drift")
    require(len(re.findall(r"(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$", cmd)) == 1 and
            len(re.findall(r"(?m)^\s*compile\b", cmd)) == 1,
            "hold-only compile count/options drift")
    require(text.count("set optimization_hold_uncertainty_ns 0.081") == 1 and
            text.count("set reported_hold_uncertainty_ns 0.050") == 1,
            "optimization/reported uncertainty constant drift")
    guard = text.index("set_clock_uncertainty -hold $optimization_hold_uncertainty_ns $core_clock")
    fix = text.index("set_fix_hold $core_clock")
    compile_at = text.index("compile -incremental_mapping -only_hold_time")
    restore = text.index("set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock")
    update = text.index("update_timing", restore)
    report = text.index('report_qor > "$output_dir/reports/qor_posthold.rpt"')
    write_sdc = text.index('write_sdc "$output_dir/netlist/')
    require(guard < fix < compile_at < restore < update < report < write_sdc,
            "optimization-only guard/final restore order drift")
    require(text.count("expected=3.000") == 2 and
            "set expected_macro_count 9" in text and
            "set area_ceiling_um2 168188.4885824" in text and
            text.count("macro_count_pre=$macro_count_pre expected=$expected_macro_count") == 1 and
            text.count("macro_count_post=$macro_count_post expected=$expected_macro_count") == 1,
            "period/macro/area constants drift")
    for artifact in ("fastmin_hold_closed.ddc", "fastmin_hold_closed.svf",
                     "fastmin_hold_closed_mapped.sdc",
                     "fastmin_hold_closed_mapped.v"):
        require(artifact in text, "missing TCL output " + artifact)
    return {"read_ddc": 1, "set_fix_hold": 1, "hold_compile": 1,
            "std_min_bind": 1, "macro_min_bind": 1,
            "optimization_hold_ns": 0.081, "reported_hold_ns": 0.050}


def validate_runner(text):
    cmd = commands(text)
    require('INPUT_DDC="${M1665_ORIGINAL}/netlist/' in text and
            not re.search(r"(?m)^INPUT_DDC=.*M1678", text),
            "M1665 DDC is not sole mapped design input")
    for digest in (
            "2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0",
            "9556e3bfab30af74326473f6cb9e492d41d3b782d0f23fabb6564626ce6fc675",
            "9eee52aa958d835e9b682d99e5b52cfed515bacee74854fb8f0a4a8ddfab7eb9",
            "b27aeb9e49081c6fbc238a082dfe7c364270e25ca11579e7ee73c717d0a12fd8",
            "c323bdd22a6f9137ee02f85aba0ed9c7792cf1febd6d8c3b11fb2650d41f7557",
            "c0dc0bce139cdf1f8be3058c43bc40ed5b67fa8c2c82292b7265f0f232f35495",
            "eeacd609124059018fdc1bbdafd460342adcc524473d0769c4d43daa43aa3445",
            "d974d269d592fe02ea04db0c062c8061bba1f8d6e67fd479bb929a1da97526eb"):
        require(text.count(digest) >= 1, "required evidence SHA absent " + digest)
    require(text.count('SHARED_QUEUE="/tmp/date_dual_synopsys_same_uid_eda_queue.lock"') == 1,
            "shared queue identity drift")
    acquire = text.index('"${FLOCK}" -x 9')
    post_lock = text.index('fail "same-UID DC collision after shared lock"')
    prelaunch = text.index('fail "same-UID DC collision immediately before launch"')
    launch = text.index('"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"')
    attempt = text.index('mkdir -- "${ATTEMPT}"')
    seal = text.index('seal_dir "${WORK}"', launch)
    publish = text.index('mv -- "${WORK}" "${RESULT}"', seal)
    require(acquire < post_lock < attempt < prelaunch < launch < seal < publish,
            "lock/collision/attempt/launch/publish order drift")
    require("ancestry=set(); pid=os.getpid()" in text and
            text.count('[[ -z "$(same_uid_dc)" ]]') == 3 and
            "flock -u" not in text,
            "ancestry-aware collision or lock lifetime drift")
    require(text.count('mkdir -- "${ATTEMPT}"') == 1 and
            text.count('max_dc_runs=1') == 1 and
            text.count('retry=false') >= 3 and
            'M1695_EXPECTED_DC_RUNNER_SHA256' in text and
            'M1695_EXPECTED_DC_RELEASE_SHA256' in text,
            "one attempt/no retry/caller pin drift")
    require(cmd.count('"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"') == 1,
            "dc_shell invocation count/flags drift")
    for token in ('"${FM_SHELL}"', '"${PT_SHELL}"', "vcs -full64", "ptpx"):
        require(token not in cmd, "other EDA invocation present")
    require('"${mem_available}" -ge 16777216' in text and
            '"${swap_free}" -ge 8388608' in text and
            '"${headroom}" -ge 25165824' in text,
            "resource gate drift")
    for artifact in ("fastmin_hold_closed.ddc", "fastmin_hold_closed.svf",
                     "fastmin_hold_closed_mapped.sdc",
                     "fastmin_hold_closed_mapped.v"):
        require(artifact in text, "required result artifact gate absent " + artifact)
    require("area<=ceiling" in text and
            "post_setup['status']=='MET' and post_hold['status']=='MET'" in text and
            "macro_ok and drc_count==0" in text,
            "timing/area/macro/DRC positive gate drift")
    require("set_false_path','set_multicycle_path','set_min_delay','set_max_delay','set_disable_timing','set_case_analysis" in text and
            "optimization uncertainty leaked" in text,
            "output SDC exception/0.081 leak gate drift")
    require(text.index('verify_dir_seal "${HAMMER_DIR}"') < attempt and
            text.index('verify_file_seal "${RELEASE}"') < attempt,
            "review/release gate occurs after attempt")
    return {"dc_invocations": 1, "collision_rechecks_after_lock": 3,
            "attempts": 1, "retry": False, "required_outputs": 4}


def expect_reject(label, function, payload):
    try:
        function(payload)
    except Exception:
        return label
    raise HammerError("mutation accepted: " + label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(),
                "exact input absent/symlink " + str(path))
        require(sha256(path) == digest, "exact SHA drift " + str(path))
    author_seal = verify_tree(AUTHOR, strict_population=True)
    m1665_seal = verify_tree(M1665, strict_population=True)
    original_seal = verify_tree(ORIGINAL, strict_population=True)
    m1678_seal = verify_tree(M1678, strict_population=False)

    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "author_receipt.json")
    require(contract["status"] ==
            "SOURCE_ONLY_M1695_C1_FASTMIN_HOLD_CLOSURE__NO_EDA_AUTHORIZED" and
            contract["identity"]["runner_sha256"] == EXPECTED[RUNNER] and
            contract["identity"]["tcl_sha256"] == EXPECTED[TCL] and
            contract["identity"]["author_test_sha256"] == EXPECTED[TEST],
            "contract source identity drift")
    require(author["status"] ==
            "PASS_M1695_C1_FASTMIN_HOLD_CLOSURE_SOURCE_AUTHOR_HANDOFF__NO_EDA" and
            author["artifacts"]["contract_sha256"] == EXPECTED[CONTRACT] and
            author["execution_boundary"]["dc"] is False,
            "author receipt identity/boundary drift")
    require(contract["frozen_m1665"]["input_ddc_sha256"] ==
                EXPECTED[ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc"] and
            contract["m1678_sealed_evidence"]["pt_fastmin_hold_wns_ns"] ==
                -0.028168444 and
            contract["m1678_sealed_evidence"]["macro_hold_check_delta_ns"] ==
                0.029174,
            "M1665/M1678 evidence semantics drift")

    tcl = TCL.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tcl_summary = validate_tcl(tcl)
    runner_summary = validate_runner(runner)
    shell = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                           cwd=str(ROOT), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True,
                           timeout=10, check=False)
    require(shell.returncode == 0, "bash -n failed: " + shell.stdout)

    spec = importlib.util.spec_from_file_location("m1696_m1695_author_tests", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    test_stream = io.StringIO()
    result = unittest.TextTestRunner(stream=test_stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module))
    require(result.wasSuccessful() and result.testsRun == 16,
            "author tests failed: " + test_stream.getvalue())

    mutations = []
    tcl_mutations = [
        ("tcl_extra_read_ddc", tcl.replace("read_ddc $input_ddc", "read_ddc $input_ddc\nread_ddc $input_ddc", 1)),
        ("tcl_read_verilog", tcl.replace("read_ddc $input_ddc", "read_verilog fake.v\nread_ddc $input_ddc", 1)),
        ("tcl_wrong_std_min", tcl.replace(
            "set_min_library $std_slow_db -min_version $std_fast_db",
            "set_min_library $std_slow_db -min_version $std_slow_db", 1)),
        ("tcl_missing_macro_min", tcl.replace("set_min_library $macro_slow_db -min_version $macro_fast_db", "# removed", 1)),
        ("tcl_wrong_optimization_hold", tcl.replace("set optimization_hold_uncertainty_ns 0.081", "set optimization_hold_uncertainty_ns 0.080", 1)),
        ("tcl_reported_hold_relaxed", tcl.replace("set reported_hold_uncertainty_ns 0.050", "set reported_hold_uncertainty_ns 0.081", 1)),
        ("tcl_extra_fix_hold", tcl.replace("set_fix_hold $core_clock", "set_fix_hold $core_clock\nset_fix_hold $core_clock", 1)),
        ("tcl_extra_compile", tcl.replace("compile -incremental_mapping -only_hold_time", "compile -incremental_mapping -only_hold_time\ncompile -incremental_mapping", 1)),
        ("tcl_false_path", tcl.replace("set_fix_hold $core_clock", "set_false_path -from [all_inputs]\nset_fix_hold $core_clock", 1)),
        ("tcl_restore_after_report", tcl.replace("set_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock", "# delayed restore", 1).replace('report_qor > "$output_dir/reports/qor_posthold.rpt"', 'report_qor > "$output_dir/reports/qor_posthold.rpt"\nset_clock_uncertainty -hold $reported_hold_uncertainty_ns $core_clock', 1)),
        ("tcl_macro_count_eight", tcl.replace("set expected_macro_count 9", "set expected_macro_count 8", 1)),
        ("tcl_area_ceiling_relaxed", tcl.replace("set area_ceiling_um2 168188.4885824", "set area_ceiling_um2 200000.0", 1)),
    ]
    for label, payload in tcl_mutations:
        mutations.append(expect_reject(label, validate_tcl, payload))
    runner_mutations = [
        ("runner_m1678_design_input", runner.replace('INPUT_DDC="${M1665_ORIGINAL}/netlist/', 'INPUT_DDC="${M1678_NEGATIVE}/netlist/', 1)),
        ("runner_missing_failure_evidence_sha", runner.replace("eeacd609124059018fdc1bbdafd460342adcc524473d0769c4d43daa43aa3445", "0" * 64, 1)),
        ("runner_wrong_shared_lock", runner.replace("/tmp/date_dual_synopsys_same_uid_eda_queue.lock", "/tmp/private.lock", 1)),
        ("runner_no_ancestry", runner.replace("ancestry=set(); pid=os.getpid()", "ancestry=set(); pid=1", 1)),
        ("runner_prelaunch_collision_after_launch", runner.replace('[[ -z "$(same_uid_dc)" ]] || fail "same-UID DC collision immediately before launch"', '# moved collision', 1)),
        ("runner_attempt_after_launch", runner.replace('mkdir -- "${ATTEMPT}"', '# delayed attempt', 1)),
        ("runner_retry_true", runner.replace("retry=false", "retry=true")),
        ("runner_second_dc", runner.replace('"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"', '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"\n"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"', 1)),
        ("runner_missing_svf_gate", runner.replace("fastmin_hold_closed.svf", "fastmin_hold_closed.missing", 1)),
        ("runner_area_gate_removed", runner.replace("area<=ceiling", "area>0", 1)),
        ("runner_drc_gate_removed", runner.replace("macro_ok and drc_count==0", "macro_ok", 1)),
        ("runner_review_after_attempt", runner.replace('verify_dir_seal "${HAMMER_DIR}"', '# removed review gate', 1)),
    ]
    for label, payload in runner_mutations:
        mutations.append(expect_reject(label, validate_runner, payload))

    require(not FUTURE_REVIEW.exists() and not FUTURE_RELEASE.exists() and
            not RESULT.exists() and not ATTEMPT.exists(),
            "future review/release/result/attempt is not absent")
    output = {
        "schema": "m1696_m1695_c1_fastmin_hold_closure_source_hammer_r1_v1",
        "date_cst": "2026-09-01",
        "status": (
            "PASS_M1696_M1695_C1_FASTMIN_HOLD_CLOSURE_SOURCE_HAMMER__"
            "AUTHORIZE_ONE_FUTURE_DC_ATTEMPT"),
        "verdict": "PASS_SOURCE_ONLY_AUTHORIZE_M1697_AUTHORING_NOT_DC",
        "score": 99,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 1,
        "p0": [],
        "p1": [],
        "p2": [
            "M1695 can authorize only one DC candidate attempt after a separately sealed M1697 release. Fresh gate-to-gate Formality, independent slowmax/fastmin PrimeTime at reported 0.050 ns, and a result hammer remain mandatory before paper use."
        ],
        "identity": {
            "tcl_sha256": EXPECTED[TCL],
            "runner_sha256": EXPECTED[RUNNER],
            "author_test_sha256": EXPECTED[TEST],
            "source_contract_sha256": EXPECTED[CONTRACT],
            "author_receipt_sha256": EXPECTED[AUTHOR / "author_receipt.json"],
            "author_manifest_sha256": author_seal["manifest_sha256"],
            "author_outer_file_sha256": author_seal["outer_file_sha256"],
            "m1665_ddc_sha256": contract["frozen_m1665"]["input_ddc_sha256"],
            "m1665_manifest_sha256": m1665_seal["manifest_sha256"],
            "m1665_original_manifest_sha256": original_seal["manifest_sha256"],
            "m1678_manifest_sha256": m1678_seal["manifest_sha256"],
            "docs359_sha256": EXPECTED[DOCS359],
        },
        "static_contract": {
            "only_frozen_m1665_ddc_design_input": True,
            "m1678_failure_and_calibration_evidence_only": True,
            "std_slow_to_fast_count": tcl_summary["std_min_bind"],
            "macro_slow_to_fast_count": tcl_summary["macro_min_bind"],
            "optimization_hold_uncertainty_ns": 0.081,
            "reported_clock_period_ns": 3.000,
            "reported_setup_uncertainty_ns": 0.200,
            "reported_hold_uncertainty_ns": 0.050,
            "set_fix_hold_count": tcl_summary["set_fix_hold"],
            "hold_only_incremental_mapping_count": tcl_summary["hold_compile"],
            "timing_exception_count": 0,
            "macro_count": 9,
            "area_ceiling_um2": 168188.4885824,
            "future_setup_wns_min_ns": 0.0,
            "future_hold_wns_min_ns": 0.0,
            "future_drc_violating_nets": 0,
            "required_outputs": ["DDC", "SVF", "mapped SDC", "mapped Verilog"],
        },
        "runner_contract": {
            "shared_queue_lock": "/tmp/date_dual_synopsys_same_uid_eda_queue.lock",
            "ancestry_aware_collision_rechecks_after_lock":
                runner_summary["collision_rechecks_after_lock"],
            "shared_lock_held_through_result_publication": True,
            "dc_invocations": runner_summary["dc_invocations"],
            "attempts": runner_summary["attempts"],
            "automatic_retry": runner_summary["retry"],
            "result_or_negative_quarantine_double_sealed": True,
        },
        "regression": {
            "author_tests_passed": result.testsRun,
            "bash_n": "PASS",
            "mutations_rejected": len(mutations),
            "mutation_labels": mutations,
        },
        "authorization": {
            "m1697_release_authoring": True,
            "dc_runs_now": 0,
            "future_dc_runs_max_after_m1697": 1,
            "all_other_eda_runs": 0,
            "automatic_retry": False,
        },
        "claim_boundary": {
            "source_only": True,
            "dc_executed": False,
            "hold_closed": False,
            "formality_new_identity": False,
            "prime_time_new_identity": False,
            "power": False,
            "energy": False,
            "cycle_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "paper_citable": False,
        },
        "review_execution": {
            "eda_runs": 0,
            "attempt_writes": 0,
            "result_writes": 0,
            "release_created": False,
            "git_commit": False,
            "git_push": False,
        },
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output["status"])


if __name__ == "__main__":
    main()
