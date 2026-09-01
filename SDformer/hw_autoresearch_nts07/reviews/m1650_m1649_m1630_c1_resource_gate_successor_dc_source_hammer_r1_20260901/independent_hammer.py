#!/usr/bin/env python3
"""Different-author, no-EDA hammer for the M1649 C1 source successor."""
from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_dc_m1649_m1630_c1_resource_gate_successor_exact_sha_r1.sh"
OLD_RUNNER = HW / "dc_handoff/scripts/run_dc_m1630_m993_c1_residual_hold_guardband_exact_sha_r1.sh"
TCL = HW / "dc_handoff/scripts/run_dc_m1630_m993_c1_residual_hold_guardband_candidate.tcl"
TEST = HW / "system_simulator/tests/test_m1649_c1_resource_gate_successor_dc_source.py"
CONTRACT = HW / "contracts/m1649_m1630_c1_resource_gate_successor_dc_source_contract_r1_20260901.json"
OLD_CONTRACT = HW / "contracts/m1630_m993_c1_residual_hold_guardband_dc_source_contract_r1_20260901.json"
M1631 = HW / "reviews/m1631_m1630_c1_residual_hold_guardband_dc_source_hammer_r1_20260901"
M1632 = HW / "contracts/m1632_m1631_m1630_c1_residual_hold_guardband_dc_launch_release_r1_20260901.json"
AUTHOR = HW / "reviews/m1649_m1630_c1_resource_gate_successor_dc_source_author_receipt_r1_20260901"
M993 = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829"
ORIGINAL = M993 / "original_quarantine"
M1006 = HW / "reviews/m1006_m993_m989_m962_recovered_c1_component_result_hammer_r1_20260829"
M1614 = HW / "dc_handoff/runs/m1614_m993_c1_macro_aware_hold_only_incremental_dc_r1_20260901.failed_or_incomplete.4065447.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RELEASE = HW / "contracts/m1651_m1650_m1649_m1630_c1_resource_gate_successor_dc_launch_release_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1649_m1630_c1_resource_gate_successor_dc_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1649_m1630_c1_resource_gate_successor_dc_attempt_consumed"
LOCK = HW / "dc_handoff/runs/.m1649_m1630_c1_resource_gate_successor_dc_launch_lock"
WORK_GLOB = ".m1649_m1630_c1_resource_gate_successor_dc_work.*"

EXPECTED = {
    RUNNER: "8a1688206acf75ee0942c7bf6acb20b16c3017c7bf54451ab11d84953a4474e3",
    OLD_RUNNER: "ceb4dcdf1a90b9285e10b06a6ff9e91faf8f840373dc6d2095c50004e7493b81",
    TCL: "e1a138284a4e558c339d987fd4da8e248b0c2ea81858fe255e812c0a4ec592c1",
    TEST: "3a612109d14d72115e6583e3116e7567e067380f6d84b80d9da24c73548834e6",
    CONTRACT: "5ca134044f1e100c925785db8025b8a7dce3e23daf5c3964608ca039ace84fb3",
    Path(str(CONTRACT) + ".sha256"): "7e5c24a9a7465ea3dd45f095cb94f2c3489ec930d0fae69462fa8d7ecf8213d4",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "1f43ae286a1b5604182a8cfa45ca1a8ff0328860cb6d0b6f90a2eb5e088a47c3",
    OLD_CONTRACT: "7fd0d5792dfcbfeb9f7c715ec7b0264e5f4da9eeabcd5ca660d07eca1d38dc0a",
    Path(str(OLD_CONTRACT) + ".sha256"): "85a653bfc9afec9d0fd9119a86cca0d63ec57f8f5c653960185653a36ec5a198",
    Path(str(OLD_CONTRACT) + ".sha256.seal.sha256"): "203223d8bfd8dee9e263c63484b01ea9d450931418fa4fa41c56b011f9e65ac8",
    M1631 / "review.json": "5d13cc9410fce81fbe22ee1c5f2f4e81bebfa17eee634adfe24254fc75344b93",
    M1631 / "SHA256SUMS": "74fdad7a95b406c2721de4bd1f507bdb34673476320146ecd37aebc56b18239f",
    M1631 / "SHA256SUMS.seal.sha256": "d8cc841f55f68e11a2ca029841bcdbe5e98d2f0218f707759e7e535846761ec1",
    M1632: "5bdf1599362ab25e3faf86e4cbe64d1ed7bf640fde40209fa8b667c206616a46",
    Path(str(M1632) + ".sha256"): "f32c33d38bbc5c53cde1e1bddc015a25deeb6d19da928f45c772b0bea99a448c",
    Path(str(M1632) + ".sha256.seal.sha256"): "6c2945db3ba5b4a1f44ed80a04183eed2bd43534b523f267c42797970e0f2a7b",
    AUTHOR / "review.json": "e01f39a09e458636eaeb96f534fed5d15ce6478d455897f1e497f6c0a593608f",
    AUTHOR / "SHA256SUMS": "4f705d1c018eb3fd627f42665a99e8f05284a39cafe4153cf90da0231d986114",
    AUTHOR / "SHA256SUMS.seal.sha256": "0f7d10e7f39680776ff546d829cfc8bf97c3341edf1d6b15a8ae1ce1dc90bee6",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.ddc":
        "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.sdc":
        "cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_mapped.v":
        "9f96c10a6cc782b7d940bb5bafff15bcd016ac63ec56dd98cb1fae09b026e8cf",
    ORIGINAL / "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island.svf":
        "8775b57603cbd7f3b465386b0b587aa4af4e00354bc959d915cc1ec71cf967c7",
    M1006 / "review.json":
        "d7b30ff3a82a099c080f3aa3dd32c13c1d2d5b5e278112eb9e3b1c24588809ea",
    M1614 / "reports/setup_posthold_summary_machine.txt":
        "ff8f206815233c222c781a507d3ce504571148885ff7b113c5f94cb8824f639b",
    M1614 / "reports/hold_posthold_summary_machine.txt":
        "fb12725e8bc76cce0c9f8198cb8b915dc91cf8c0fbb4e185b4fae2da2414da8c",
    M1614 / "reports/area_posthold.rpt":
        "aa109ac641cbee88d6617e4b4f3008f6669a91d51e055bb151d7b3c324ec655e",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

FORBIDDEN = ("set_false_path", "set_multicycle_path", "set_min_delay",
             "set_max_delay", "set_disable_timing", "set_case_analysis")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def verify_file_seal(path):
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(manifest.read_text(encoding="ascii").split() ==
            [sha256(path), path.name], "file manifest drift " + path.name)
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), manifest.name], "file outer drift " + path.name)


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), "SHA256SUMS"], "tree outer drift " + str(root))
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and
                name not in expected, "unsafe tree row")
        expected[name] = digest
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        base_path = Path(base)
        dirs[:] = [name for name in dirs
                   if not (base_path / name).is_symlink()]
        for name in files:
            path = base_path / name
            if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
                    "nonregular tree member")
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(expected), "tree coverage drift " + str(root))
    for name, digest in expected.items():
        require(sha256(root / name) == digest, "tree member drift " + name)


def command_text(text, comment="#"):
    return "\n".join(row.split(comment, 1)[0] for row in text.splitlines())


def audit_semantics(runner, tcl, contract):
    commands = command_text(runner)
    tcl_commands = command_text(tcl)
    require(hashlib.sha256(tcl.encode("utf-8")).hexdigest() ==
            "e1a138284a4e558c339d987fd4da8e248b0c2ea81858fe255e812c0a4ec592c1" and
            contract["identity"]["tcl_sha256"] ==
            "e1a138284a4e558c339d987fd4da8e248b0c2ea81858fe255e812c0a4ec592c1",
            "exact M1630 Tcl identity drift")
    require(runner.count('"${headroom}" -ge 50331648') == 1 and
            '"${headroom}" -ge 67108864' not in runner,
            "commit headroom is not exactly 48 GiB")
    require(runner.count('"${mem_available}" -ge 100663296') == 1 and
            runner.count('"${swap_free}" -ge 16777216') == 1,
            "resident/swap gates drift")
    require("blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}"
            in runner and "same-UID DC collision" in runner,
            "collision gate drift")
    require(runner.count('"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler') == 1,
            "license gate drift")
    require(runner.count(
        '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"') == 1,
        "DC invocation drift")
    require(not re.search(r"(?m)^\s*(?:export\s+)?HOME=", commands),
            "HOME repurposed")
    require(runner.count('mkdir -- "${ATTEMPT}"') == 1 and
            runner.count('mkdir -- "${LOCK}"') == 1 and
            'rmdir -- "${ATTEMPT}"' not in runner and "rm -rf" not in runner,
            "one-shot/lock policy drift")
    require("retry=false" in runner and "'retry':False" in runner,
            "retry policy drift")
    require("M1649_EXPECTED_DC_RUNNER_SHA256" in runner and
            "M1649_EXPECTED_DC_RELEASE_SHA256" in runner,
            "caller pins missing")
    require("PASS_M1650_M1649_M1630_C1_RESOURCE_GATE_SOURCE_HAMMER" in runner and
            "AUTHORIZE_ONE_M1649_C1_RESOURCE_GATE_SUCCESSOR_DC_ATTEMPT" in runner,
            "future authority namespace drift")
    require("m1649_m1630_c1_resource_gate_successor_dc_r1_20260901" in runner and
            "m1630_m993_c1_residual_hold_guardband_dc_r1_20260901" not in runner,
            "runtime namespace drift")
    release_gate = runner.index('verify_file_seal "${RELEASE}"')
    require(runner.index('verify_dir_seal "${HAMMER_DIR}"') < release_gate <
            runner.index('mkdir -- "${LOCK}"') <
            runner.index('mkdir -- "${ATTEMPT}"'),
            "release/lock/attempt order drift")
    require(runner.index('M1649_EXPECTED_DC_RELEASE_SHA256') <
            runner.index('mkdir -- "${LOCK}"'), "release pin is too late")
    for digest in (
        "ceb4dcdf1a90b9285e10b06a6ff9e91faf8f840373dc6d2095c50004e7493b81",
        "7fd0d5792dfcbfeb9f7c715ec7b0264e5f4da9eeabcd5ca660d07eca1d38dc0a",
        "5d13cc9410fce81fbe22ee1c5f2f4e81bebfa17eee634adfe24254fc75344b93",
        "5bdf1599362ab25e3faf86e4cbe64d1ed7bf640fde40209fa8b667c206616a46",
        "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56",
        "cf7a0c4a6af76471de8ea4fa06017a8152bea6365e278dd92c3f0fb489c40aa5",
    ):
        require(digest in runner, "frozen chain/input digest missing " + digest)
    require(not re.search(r"(?mi)^INPUT_DDC=.*m1614", runner) and
            "m1614_hold_repaired.ddc" not in tcl,
            "failed M1614 output became an input")
    require(len(re.findall(r"(?m)^\s*read_ddc\b", tcl_commands)) == 1 and
            len(re.findall(r"(?m)^\s*set_fix_hold\b", tcl_commands)) == 1 and
            len(re.findall(r"(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$",
                           tcl_commands)) == 1 and
            len(re.findall(r"(?m)^\s*compile\b", tcl_commands)) == 1 and
            not re.search(r"(?m)^\s*compile_ultra\b", tcl_commands),
            "Tcl compile cardinality drift")
    require(tcl.count("set optimization_hold_guardband_ns 0.051") == 1 and
            tcl.count("set reported_hold_uncertainty_ns 0.050") == 1,
            "hold guardband/report point drift")
    require("set expected_macro_count 9" in tcl and
            "set_wire_load_model -name ZeroWireload" in tcl and
            "set_min_library $std_slow_db -min_version $std_fast_db" in tcl and
            "set_min_library $macro_slow_db -min_version $macro_fast_db" in tcl,
            "macro/corner/wireload contract drift")
    for token in FORBIDDEN:
        require(not re.search(r"(?m)^\s*" + token + r"\b", tcl_commands),
                "timing exception added " + token)
    require("baseline=147246.392090; ceiling=154608.7116945" in runner and
            "macro_ok=(int(macro['macro_count_pre'])==9 and int(macro['macro_count_post'])==9)"
                in runner and
            "timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')"
                in runner and
            "positive=timing_ok and area_ok and macro_ok and drc_count==0"
                in runner,
            "result admission predicate drift")
    require(contract["resource_gate"] == {
        "commit_headroom_min_kib": 50331648,
        "commit_headroom_min_gib": 48,
        "mem_available_min_kib": 100663296,
        "swap_free_min_kib": 16777216,
        "same_uid_dc_collision_tolerance": 0,
        "license_preflight_unchanged": True,
        "tool_and_input_identity_preflight_unchanged": True,
        "m1630_five_minute_observation": contract["resource_gate"][
            "m1630_five_minute_observation"],
        "first_principles_rationale": contract["resource_gate"][
            "first_principles_rationale"],
        "physical_or_result_condition_changed": False,
    }, "contract resource gate drift")
    require(contract["authorization"]["dc_runs_now"] == 0 and
            contract["authorization"]["future_dc_runs_max"] == 1 and
            contract["authorization"]["retry"] is False,
            "contract execution authorization drift")
    require(contract["claim_boundary"]["launch_authorized"] is False and
            contract["claim_boundary"]["paper_citable"] is False,
            "contract claim boundary opened")


def compare_predecessor_contract(old, new):
    exact_sections = ("input_policy", "frozen_reported_point",
                      "compile_contract", "failure_policy", "post_dc_gates")
    for section in exact_sections:
        require(new[section] == old[section],
                "M1630 contract section drift " + section)
    for key in ("setup", "hold", "area", "macros", "design_rules",
                "constraints"):
        require(new["dc_success_gate"][key] == old["dc_success_gate"][key],
                "M1630 success predicate drift " + key)
    require(new["frozen_predecessor"]["input_ddc"]["sha256"] ==
            "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56",
            "M993 DDC contract drift")


def mutation_hammer(runner, tcl, contract):
    runner_mutations = (
        ("HEADROOM_1K", '"${headroom}" -ge 50331648', '"${headroom}" -ge 1'),
        ("HEADROOM_OLD64G", '"${headroom}" -ge 50331648', '"${headroom}" -ge 67108864'),
        ("MEMAVAILABLE_1K", '"${mem_available}" -ge 100663296', '"${mem_available}" -ge 1'),
        ("SWAP_1K", '"${swap_free}" -ge 16777216', '"${swap_free}" -ge 1'),
        ("COLLISION_DISABLED", "blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}", "blocked=set()"),
        ("LICENSE_DISABLED", '"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler', ': # license disabled'),
        ("DC_INIT_FLAGS_REMOVED", '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"', '"${DC_SHELL}" -f "${TCL}"'),
        ("ATTEMPT_REMOVED", 'mkdir -- "${ATTEMPT}"', ': # attempt removed'),
        ("LOCK_REMOVED", 'mkdir -- "${LOCK}"', ': # lock removed'),
        ("RETRY_TRUE", "retry=false", "retry=true"),
        ("RUNNER_PIN_REMOVED", "M1649_EXPECTED_DC_RUNNER_SHA256", "M1649_UNPINNED_DC_RUNNER_SHA256"),
        ("RELEASE_PIN_REMOVED", "M1649_EXPECTED_DC_RELEASE_SHA256", "M1649_UNPINNED_DC_RELEASE_SHA256"),
        ("OLD_RESULT_NAMESPACE", "m1649_m1630_c1_resource_gate_successor_dc_r1_20260901", "m1630_m993_c1_residual_hold_guardband_dc_r1_20260901"),
        ("OLD_REVIEW_STATUS", "PASS_M1650_M1649_M1630_C1_RESOURCE_GATE_SOURCE_HAMMER", "PASS_M1631_M1630_C1_RESIDUAL_HOLD_SOURCE_HAMMER"),
        ("OLD_RELEASE_STATUS", "AUTHORIZE_ONE_M1649_C1_RESOURCE_GATE_SUCCESSOR_DC_ATTEMPT", "AUTHORIZE_ONE_M1630_C1_RESIDUAL_HOLD_GUARDBAND_DC_ATTEMPT"),
        ("M993_DDC_SHA_DRIFT", "d301d6b5e9f20c694721cae36a3363e13815129d322ec9423b05658b329afb56", "0" * 64),
        ("M1630_RUNNER_BINDING_DRIFT", "ceb4dcdf1a90b9285e10b06a6ff9e91faf8f840373dc6d2095c50004e7493b81", "2" * 64),
        ("FAILED_M1614_AS_INPUT", 'INPUT_DDC="${M993_ORIGINAL}/netlist/', 'INPUT_DDC="${M1614_NEGATIVE}/netlist/'),
        ("AREA_CEILING_OPEN", "baseline=147246.392090; ceiling=154608.7116945", "baseline=147246.392090; ceiling=999999.0"),
        ("MACRO_GATE_TRUE", "macro_ok=(int(macro['macro_count_pre'])==9 and int(macro['macro_count_post'])==9)", "macro_ok=True"),
        ("TIMING_GATE_TRUE", "timing_ok=(post_setup['status']=='MET' and post_hold['status']=='MET')", "timing_ok=True"),
        ("POSITIVE_GATE_TRUE", "positive=timing_ok and area_ok and macro_ok and drc_count==0", "positive=True"),
        ("SECOND_DC_INVOCATION", '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"', '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"\n"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"'),
        ("ATTEMPT_RECOVERABLE", 'rmdir -- "${LOCK}"', 'rmdir -- "${ATTEMPT}"\nrmdir -- "${LOCK}"'),
    )
    tcl_mutations = (
        ("TCL_GUARDBAND_050", "set optimization_hold_guardband_ns 0.051", "set optimization_hold_guardband_ns 0.050"),
        ("TCL_REPORT_HOLD_051", "set reported_hold_uncertainty_ns 0.050", "set reported_hold_uncertainty_ns 0.051"),
        ("TCL_MACROS_8", "set expected_macro_count 9", "set expected_macro_count 8"),
        ("TCL_SECOND_COMPILE", "compile -incremental_mapping -only_hold_time", "compile -incremental_mapping -only_hold_time\ncompile -incremental_mapping -only_hold_time"),
        ("TCL_FALSE_PATH", "set expected_macro_count 9", "set expected_macro_count 9\nset_false_path -from [all_inputs]"),
        ("TCL_COMPILE_ULTRA", "compile -incremental_mapping -only_hold_time", "compile_ultra -incremental"),
        ("TCL_WIRELOAD_REMOVED", "set_wire_load_model -name ZeroWireload", "# wireload removed"),
    )
    rejected = []
    escaped = []
    for name, old, new in runner_mutations:
        require(old in runner, "missing mutation anchor " + name)
        mutant = runner.replace(old, new)
        try:
            audit_semantics(mutant, tcl, contract)
        except AssertionError:
            rejected.append(name)
        else:
            escaped.append(name)
    for name, old, new in tcl_mutations:
        require(old in tcl, "missing mutation anchor " + name)
        mutant = tcl.replace(old, new)
        try:
            audit_semantics(runner, mutant, contract)
        except AssertionError:
            rejected.append(name)
        else:
            escaped.append(name)
    contract_mutations = (
        ("CONTRACT_DC_NOW", ("authorization", "dc_runs_now"), 1),
        ("CONTRACT_RETRY", ("authorization", "retry"), True),
        ("CONTRACT_HEADROOM", ("resource_gate", "commit_headroom_min_kib"), 1),
        ("CONTRACT_TCL_SHA", ("identity", "tcl_sha256"), "1" * 64),
        ("CONTRACT_PAPER", ("claim_boundary", "paper_citable"), True),
    )
    for name, keys, value in contract_mutations:
        mutant = json.loads(json.dumps(contract))
        mutant[keys[0]][keys[1]] = value
        try:
            audit_semantics(runner, tcl, mutant)
        except AssertionError:
            rejected.append(name)
        else:
            escaped.append(name)
    require(not escaped, "mutation escaped: " + repr(escaped))
    return rejected


def run():
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(),
                "missing/nonregular exact input " + str(path))
        require(sha256(path) == digest, "exact input SHA drift " + str(path))
    for path in (CONTRACT, OLD_CONTRACT, M1632):
        verify_file_seal(path)
    for path in (M1631, AUTHOR, M993, ORIGINAL, M1006, M1614):
        verify_tree(path)
    runner = RUNNER.read_text(encoding="utf-8")
    old_runner = OLD_RUNNER.read_text(encoding="utf-8")
    tcl = TCL.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    old_contract = json.loads(OLD_CONTRACT.read_text(encoding="utf-8"))
    audit_semantics(runner, tcl, contract)
    compare_predecessor_contract(old_contract, contract)
    require(old_runner.count('"${headroom}" -ge 67108864') == 1 and
            old_runner.count('"${mem_available}" -ge 100663296') == 1 and
            old_runner.count('"${swap_free}" -ge 16777216') == 1,
            "old resource-gate baseline drift")
    require(not RELEASE.exists() and not RESULT.exists() and
            not ATTEMPT.exists() and not LOCK.exists() and
            not list(RESULT.parent.glob(WORK_GLOB)),
            "fresh M1649/M1651 runtime namespace is not empty")
    rejected = mutation_hammer(runner, tcl, contract)
    return {
        "schema": "m1650_m1649_m1630_c1_resource_gate_source_independent_hammer_r1_v1",
        "status": "PASS_M1650_INDEPENDENT_STATIC_AND_MUTATION_HAMMER",
        "exact_files_checked": len(EXPECTED),
        "sealed_trees_checked": 6,
        "sealed_files_checked": 3,
        "m1630_contract_exact_sections_compared": 5,
        "m1630_success_predicates_compared": 6,
        "only_runtime_change": {
            "commit_headroom_kib": {"m1630": 67108864,
                                     "m1649": 50331648},
            "mem_available_kib": {"m1630": 100663296,
                                   "m1649": 100663296},
            "swap_free_kib": {"m1630": 16777216,
                               "m1649": 16777216},
            "authority_and_runtime_namespaces_fresh": True,
            "physical_or_result_predicate_changed": False},
        "mutations_rejected": len(rejected),
        "mutation_categories": rejected,
        "preflight_order": "M1650 seal then absent M1651 release then caller pins/collision/lock/attempt/license/DC",
        "release_absent": True, "attempt_absent": True,
        "result_absent": True, "lock_absent": True, "work_absent": True,
        "eda": False, "attempt_created": False, "result_created": False,
        "release_created": False, "docs359_modified": False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    args = parser.parse_args(argv)
    del args
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
