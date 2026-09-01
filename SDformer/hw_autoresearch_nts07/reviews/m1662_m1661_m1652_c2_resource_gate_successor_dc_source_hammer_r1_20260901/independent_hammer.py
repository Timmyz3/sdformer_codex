#!/usr/bin/env python3
"""Different-author, no-EDA hammer for the M1661 C2 DC successor source."""
from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_dc_m1661_m1652_c2_resource_gate_successor_exact_sha_r1.sh"
TEST = HW / "system_simulator/tests/test_m1661_m1652_c2_resource_gate_successor_dc_source.py"
CONTRACT = HW / "contracts/m1661_m1652_c2_resource_gate_successor_dc_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1661_m1652_c2_resource_gate_successor_dc_source_author_receipt_r1_20260901"
FILELIST = HW / "dc_handoff/filelists/date_m1634_c2_m1609_registered_fault_three_axis_logic_only_dc.f"
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
M1627 = HW / "reviews/m1627_m1613_c2_registered_fault_directed_vcs_result_independent_hammer_r1_20260901/review.json"
M903 = HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/review.json"
M1634_RUNNER = HW / "dc_handoff/scripts/run_dc_m1634_m1609_c2_registered_fault_three_axis_logic_only_exact_sha_r1.sh"
M1634_CONTRACT = HW / "contracts/m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_source_contract_r1_20260901.json"
M1635 = HW / "reviews/m1635_m1634_m1609_c2_three_axis_dc_source_hammer_r1_20260901"
M1636 = HW / "contracts/m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_release_r1_20260901.json"
M1641 = HW / "reviews/m1641_m1636_m1634_m1609_c2_three_axis_dc_release_hammer_r1_20260901"
M1652_RUNNER = HW / "dc_handoff/scripts/run_dc_m1652_m1634_c2_resource_gate_successor_exact_sha_r1.sh"
M1652_CONTRACT = HW / "contracts/m1652_m1634_c2_resource_gate_successor_dc_source_contract_r1_20260901.json"
M1653 = HW / "reviews/m1653_m1652_m1634_c2_resource_gate_successor_dc_source_hammer_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RELEASE = HW / "contracts/m1663_m1662_m1661_m1652_c2_resource_gate_successor_dc_launch_release_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_attempt_consumed"
LOCK = HW / "dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_launch_lock"
WORK_GLOB = ".m1661_m1652_c2_resource_gate_successor_three_axis_dc_work.*"

EXPECTED_ROWS = (
    "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
    "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
    "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
    "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
    "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
)

EXPECTED = {
    RUNNER: "9bf1e220054ff28e3c7bad27b07bc61f50a504625f4b7df0893b0e50162e80e6",
    TEST: "5f88093d0a24801c73173fddea7be1f3150b8f755f9e728495330e1c904bfbcf",
    CONTRACT: "1e2f04c6c46c69c58659e406b6c5d055f24c91429d6e2dcd9dd7bb1a53df03ed",
    Path(str(CONTRACT) + ".sha256"): "b04976ab2d4fc9fafc565a6c3e1e8b68bd6a9144627e7ef9fd58665d712fdb27",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "f7ceffc41f694dbecb8f673974ffeb3ffcc084e5ea24d8555ec78c04b460e4b2",
    AUTHOR / "review.json": "457ba8e3d8032368df8bf5953293f03ec0bd25acb0bb772cef08f4a075276ce7",
    AUTHOR / "SHA256SUMS": "7c0f882d60ec44e6f532ad95a173d78623c07a3bb0e79aa348d9afd0668174f3",
    AUTHOR / "SHA256SUMS.seal.sha256": "ae92b626d35586ef25ba0590fa67746321a10c4774658814d77fcf22cd49f4fe",
    FILELIST: "03c4dcd546da19d5de231fa80032473e7c365592012661e6ed77019d7bab4f3f",
    TCL: "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    SDC: "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    M1634_RUNNER: "da9cd0d118021eb85c8b548d93f6779ec6d25b6fec7ca5894bdae988a95840b7",
    M1634_CONTRACT: "9f5e5b1cb40da5cd403270ba48ceac9b5a7d6aecd79b7ad98cf3d644d0f8f030",
    M1635 / "review.json": "215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620",
    M1636: "0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088",
    M1641 / "review.json": "278df1d44232cccabc0c50e45beae9dee60adce834896f1be20f8fc7625bf1e6",
    M1652_RUNNER: "57f9b90642641215c801b0f61302636ddecb81e6b37523763f6523f2862dfdb3",
    M1652_CONTRACT: "01ee8cff796705c71a0b3c5875046ca32d08935936026315375da797d02d863c",
    M1653 / "review.json": "5e3e6c9974e26a28be3e6bae7efc93e661afafaf0ba8b5b9ebf35e5ad0855d6d",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

AUTH_MUTATIONS = (
    ("dc_runs_now", 1),
    ("future_dc_shell_runs_max", 4),
    ("all_other_eda_runs", 1),
    ("vcs_runs", 1),
    ("pt_runs", 1),
    ("formality_runs", 1),
    ("ptpx_runs", 1),
    ("gpu_runs", 1),
    ("remote_runs", 1),
    ("attempts_created_now", 1),
    ("retry", True),
)


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
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sidecar.read_text(encoding="ascii").split() ==
            [sha256(path), path.name], "file sidecar drift " + path.name)
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sidecar), sidecar.name], "file outer drift " + path.name)


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), "SHA256SUMS"], "tree outer drift " + str(root))
    expected = {}
    for row in manifest.read_text(encoding="ascii").splitlines():
        digest, name = row.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and
                name not in expected, "unsafe tree row")
        expected[name] = digest
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        bp = Path(base)
        for name in dirs:
            require(not (bp / name).is_symlink(), "symlink directory")
        for name in files:
            path = bp / name
            if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
                    "nonregular tree member")
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(expected), "tree coverage drift " + str(root))
    for name, digest in expected.items():
        require(sha256(root / name) == digest, "tree member drift " + name)


def embedded_authority_snippet(runner):
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", runner, re.S)
    selected = [text for text in snippets
                if "contract,runner,m1627,m903=map(Path,sys.argv[1:])" in text]
    require(len(selected) == 1, "embedded authorization cardinality drift")
    return selected[0]


def execute_embedded(snippet, contract_path):
    return subprocess.run(
        [os.environ.get("M1662_PYTHON", "/usr/bin/python3"), "-I", "-",
         str(contract_path), str(RUNNER), str(M1627), str(M903)],
        input=snippet, universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=15, check=False)


def audit_policy(runner, tcl, contract, rows):
    require(tuple(rows) == EXPECTED_ROWS, "M1634 12-row filelist drift")
    require(runner.count('"${headroom}" -ge 50331648') == 1 and
            '"${headroom}" -ge 67108864' not in runner,
            "commit headroom is not exactly 48 GiB")
    require(runner.count('"${mem_available}" -ge 100663296') == 1,
            "MemAvailable is not exactly 96 GiB")
    require(runner.count('"${swap_free}" -ge 16777216') == 1,
            "SwapFree is not exactly 16 GiB")
    require("blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}"
            in runner and "same-UID DC collision" in runner,
            "same-UID collision gate drift")
    require(runner.count(
        '"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler') == 1,
        "license gate drift")
    require("axis_names=(k1 k8 k1x8)" in runner and
            "axis_modes=(0 1 2)" in runner and
            "for index in 0 1 2" in runner, "three-axis loop drift")
    require(runner.count('"${DC_SHELL}" -f "${TCL}"') == 1,
            "DC call cardinality drift")
    require(len(re.findall(r"(?m)^\s*compile_ultra\s*$", tcl)) == 1 and
            tcl.count('puts $compile_fp "compile_ultra_count=1"') == 1,
            "fresh compile_ultra cardinality drift")
    require("fresh_all_axes=true" in runner and
            "old_netlist_reuse=false" in runner,
            "fresh/no-old-netlist contract drift")
    require(not re.search(r"cp[^\n]*(?:M872_RESULT|\.ddc|_mapped\.v)", runner),
            "old mapped artifact copy introduced")
    for token in (
        "TIM-209=0", "OPT-150=0", "compile_ultra_count=1",
        "incremental_compile_count=0", "hold_optimization_count=0",
        "slack (MET)", "This design has no violated constraints.",
        "PASS_RAW_M1661_M1609_C2_THREE_AXIS_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RESULT_HAMMER",
    ):
        require(token in runner, "result predicate drift " + token)
    for digest in (
        "215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620",
        "0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088",
        "278df1d44232cccabc0c50e45beae9dee60adce834896f1be20f8fc7625bf1e6",
        "57f9b90642641215c801b0f61302636ddecb81e6b37523763f6523f2862dfdb3",
        "01ee8cff796705c71a0b3c5875046ca32d08935936026315375da797d02d863c",
        "5e3e6c9974e26a28be3e6bae7efc93e661afafaf0ba8b5b9ebf35e5ad0855d6d",
    ):
        require(digest in runner, "predecessor binding drift " + digest)
    require("M1661_EXPECTED_DC_RUNNER_SHA256" in runner and
            "M1661_EXPECTED_DC_RELEASE_SHA256" in runner,
            "caller exact-SHA pins drift")
    require(runner.index('verify_dir_seal "${HAMMER_DIR}"') <
            runner.index('verify_file_seal "${RELEASE}"') <
            runner.index('mkdir -- "${LOCK}"') <
            runner.index('mkdir -- "${ATTEMPT}"') <
            runner.index('"${DC_SHELL}" -f "${TCL}"'),
            "authority/lock/attempt/DC order drift")
    require("rm -rf" not in runner and "retry=false" in runner and
            "automatic_retry':False" in runner,
            "destructive or retry policy drift")
    gate = contract["resource_gate"]
    require(gate["commit_headroom_min_kib"] == 50331648 and
            gate["mem_available_min_kib"] == 100663296 and
            gate["swap_free_min_kib"] == 16777216 and
            gate["same_uid_dc_collision_tolerance"] == 0 and
            gate["license_preflight_unchanged"] is True and
            gate["physical_or_result_condition_changed"] is False,
            "contract resource gate drift")
    require(contract["fair_three_axis_definition"]["axis_order"] ==
            ["k1", "k8", "k1x8"] and
            contract["fair_three_axis_definition"]["frozen_baseline_netlist_reuse"] is False,
            "contract fair-axis drift")
    auth = contract["authorization"]
    expected_auth = {
        "dc_runs_now": 0, "future_dc_shell_runs_max": 3,
        "all_other_eda_runs": 0, "vcs_runs": 0, "pt_runs": 0,
        "formality_runs": 0, "ptpx_runs": 0, "gpu_runs": 0,
        "remote_runs": 0, "attempts_created_now": 0, "retry": False,
    }
    require(auth == expected_auth, "sealed authorization dictionary drift")
    for key in ("dc_authorized", "dc_completed", "fresh_mapped_k8",
                "fresh_mapped_k1x8", "setup_area", "hold_closed", "power",
                "energy", "formality", "paper_ppa_ready", "system_speedup",
                "paper_headline"):
        require(contract["claim_boundary"][key] is False,
                "claim boundary opened " + key)


def static_mutation_hammer(runner, tcl, contract, rows):
    attacks = (
        ("HEADROOM", '"${headroom}" -ge 50331648', '"${headroom}" -ge 1'),
        ("MEMAVAILABLE", '"${mem_available}" -ge 100663296', '"${mem_available}" -ge 1'),
        ("SWAP", '"${swap_free}" -ge 16777216', '"${swap_free}" -ge 1'),
        ("COLLISION", "blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}", "blocked=set()"),
        ("LICENSE", '"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler', "true"),
        ("AXIS_NAMES", "axis_names=(k1 k8 k1x8)", "axis_names=(k1 k8)"),
        ("AXIS_MODES", "axis_modes=(0 1 2)", "axis_modes=(0 1 1)"),
        ("AXIS_LOOP", "for index in 0 1 2", "for index in 0 1"),
        ("DC_CALL", '"${DC_SHELL}" -f "${TCL}"', "true"),
        ("FRESH", "fresh_all_axes=true", "fresh_all_axes=false"),
        ("REUSE", "old_netlist_reuse=false", "old_netlist_reuse=true"),
        ("TIM209", "TIM-209=0", "TIM-209=1"),
        ("OPT150", "OPT-150=0", "OPT-150=1"),
        ("SETUP", "slack (MET)", "slack (VIOLATED)"),
        ("DRC", "This design has no violated constraints.", "violations ignored"),
        ("RETRY", "automatic_retry':False", "automatic_retry':True"),
        ("M1635", "215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620", "0" * 64),
        ("M1636", "0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088", "0" * 64),
        ("M1641", "278df1d44232cccabc0c50e45beae9dee60adce834896f1be20f8fc7625bf1e6", "0" * 64),
        ("M1652", "57f9b90642641215c801b0f61302636ddecb81e6b37523763f6523f2862dfdb3", "0" * 64),
        ("M1653", "5e3e6c9974e26a28be3e6bae7efc93e661afafaf0ba8b5b9ebf35e5ad0855d6d", "0" * 64),
        ("RUNNER_PIN", "M1661_EXPECTED_DC_RUNNER_SHA256", "UNPINNED_RUNNER"),
        ("RELEASE_PIN", "M1661_EXPECTED_DC_RELEASE_SHA256", "UNPINNED_RELEASE"),
    )
    rejected = []
    for label, old, new in attacks:
        require(old in runner, "missing mutation anchor " + label)
        try:
            audit_policy(runner.replace(old, new), tcl, contract, rows)
        except (AssertionError, KeyError, ValueError):
            rejected.append(label)
        else:
            raise AssertionError("static runner mutation escaped " + label)
    tcl_attacks = (
        ("TCL_COMPILE_REMOVED", "\n    compile_ultra\n", "\n    # compile removed\n"),
        ("TCL_SECOND_COMPILE", "\n    compile_ultra\n", "\n    compile_ultra\n    compile_ultra\n"),
    )
    for label, old, new in tcl_attacks:
        require(old in tcl, "missing Tcl mutation anchor " + label)
        try:
            audit_policy(runner, tcl.replace(old, new, 1), contract, rows)
        except (AssertionError, KeyError, ValueError):
            rejected.append(label)
        else:
            raise AssertionError("static Tcl mutation escaped " + label)
    contract_attacks = (
        ("CONTRACT_HEADROOM", ("resource_gate", "commit_headroom_min_kib"), 1),
        ("CONTRACT_MEM", ("resource_gate", "mem_available_min_kib"), 1),
        ("CONTRACT_SWAP", ("resource_gate", "swap_free_min_kib"), 1),
        ("CONTRACT_COLLISION", ("resource_gate", "same_uid_dc_collision_tolerance"), 1),
        ("CONTRACT_AXIS", ("fair_three_axis_definition", "axis_order"), ["k8"]),
        ("CONTRACT_REUSE", ("fair_three_axis_definition", "frozen_baseline_netlist_reuse"), True),
        ("CONTRACT_CLAIM", ("claim_boundary", "paper_headline"), True),
    )
    for label, keys, value in contract_attacks:
        mutant = json.loads(json.dumps(contract))
        mutant[keys[0]][keys[1]] = value
        try:
            audit_policy(runner, tcl, mutant, rows)
        except (AssertionError, KeyError, ValueError):
            rejected.append(label)
        else:
            raise AssertionError("static contract mutation escaped " + label)
    return rejected


def run():
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(),
                "missing/nonregular exact input " + str(path))
        require(sha256(path) == digest, "identity drift " + str(path))
    for path in (CONTRACT, M1634_CONTRACT, M1636, M1652_CONTRACT):
        verify_file_seal(path)
    for root in (AUTHOR, M1635, M1641, M1653):
        verify_tree(root)

    runner = RUNNER.read_text(encoding="utf-8")
    tcl = TCL.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rows = [row for row in FILELIST.read_text(encoding="utf-8").splitlines()
            if row.strip()]
    audit_policy(runner, tcl, contract, rows)

    completed = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, timeout=10, check=False)
    require(completed.returncode == 0, "bash syntax failure")

    snippet = embedded_authority_snippet(runner)
    require("c['authorization']==" not in snippet,
            "whole-dictionary authorization defect survived")
    canonical = execute_embedded(snippet, CONTRACT)
    require(canonical.returncode == 0, "canonical inline preflight failed: " + canonical.stdout)
    rejected_auth = []
    with tempfile.TemporaryDirectory(prefix="m1662_authority_") as directory:
        candidate_path = Path(directory) / "contract.json"
        for key, value in AUTH_MUTATIONS:
            candidate = json.loads(json.dumps(contract))
            candidate["authorization"][key] = value
            candidate_path.write_text(json.dumps(candidate, sort_keys=True) + "\n",
                                      encoding="utf-8")
            completed = execute_embedded(snippet, candidate_path)
            require(completed.returncode != 0 and "AssertionError" in completed.stdout,
                    "authorization mutation escaped " + key)
            rejected_auth.append(key)
    require(len(rejected_auth) == 11, "authorization mutation count drift")

    rejected_static = static_mutation_hammer(runner, tcl, contract, rows)
    old_fail = json.loads((M1653 / "review.json").read_text(encoding="utf-8"))
    require(old_fail["status"] ==
            "FAIL_M1653_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER__NO_RELEASE" and
            old_fail["p0_count"] == 0 and old_fail["p1_count"] == 1 and
            old_fail["authorization"]["m1654_release_authoring"] is False and
            old_fail["authorization"]["future_dc_attempts"] == 0,
            "M1653 negative authority drift")
    require(not RELEASE.exists() and not RESULT.exists() and
            not ATTEMPT.exists() and not LOCK.exists() and
            not list(RESULT.parent.glob(WORK_GLOB)),
            "future release/runtime namespace is not fresh")

    return {
        "schema": "m1662_m1661_m1652_c2_resource_gate_source_independent_hammer_r1_v1",
        "status": "PASS_M1662_INDEPENDENT_STATIC_EXECUTABLE_PREFLIGHT_AND_MUTATION_HAMMER",
        "python_runtime": os.environ.get("M1662_RUNTIME_LABEL", "unknown"),
        "runner_sha256": sha256(RUNNER),
        "contract_sha256": sha256(CONTRACT),
        "filelist_sha256": sha256(FILELIST),
        "tcl_sha256": sha256(TCL),
        "sdc_sha256": sha256(SDC),
        "m1652_runner_sha256": sha256(M1652_RUNNER),
        "m1652_contract_sha256": sha256(M1652_CONTRACT),
        "m1653_review_sha256": sha256(M1653 / "review.json"),
        "canonical_embedded_preflight_returncode": canonical.returncode,
        "executed_authorization_mutations_rejected": rejected_auth,
        "executed_authorization_mutation_count": len(rejected_auth),
        "static_mutations_rejected": rejected_static,
        "static_mutation_count": len(rejected_static),
        "filelist_rows": len(rows),
        "resource_gate_kib": {
            "commit_headroom": 50331648,
            "mem_available": 100663296,
            "swap_free": 16777216,
            "same_uid_dc_collision_tolerance": 0,
        },
        "fresh_three_axis_compile_ultra": True,
        "old_m1652_m1653_immutable": True,
        "eda_runs": 0,
        "attempts_created": 0,
        "results_created": 0,
        "releases_created": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
