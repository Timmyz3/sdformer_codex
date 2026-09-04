#!/usr/bin/python3.12
"""Independent, read-only M2165 hammer of the exact M2164 source.

No EDA executable, license client, GPU program, or production runner is invoked.
The review rechecks every M2146/M2154 repair and attacks the two repaired
predicates with synthetic native-file and process-graph mutations.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m2164_m2154_icc2_library_import_preflight_source_contract_r1_20260904.json"
RUNNER = HW / "dc_handoff/scripts/run_m2164_m2154_icc2_library_import_preflight_one_shot.sh"
TCL = HW / "dc_handoff/scripts/run_icc2_m2153_library_import_preflight.tcl"
MONITOR = HW / "dc_handoff/scripts/monitor_m2153_icc2_process_tree.py"
INVENTORY = HW / "dc_handoff/scripts/inventory_m2153_repo_root.py"
CHECKER = HW / "system_simulator/scripts/check_m2164_icc2_library_import_preflight.py"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
MW_MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
AUTHOR = HW / "reviews/m2164_m2154_icc2_library_import_preflight_source_author_receipt_r1_20260904"
M2146 = HW / "reviews/m2146_m2141_m2136_icc2_library_import_preflight_source_hammer_r1_20260904"
M2154 = HW / "reviews/m2154_m2153_m2146_icc2_library_import_preflight_source_hammer_r1_20260904"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "dc_handoff/runs/m2166_m2164_icc2_library_import_preflight_raw_r1_20260904"
ATTEMPT = HW / "dc_handoff/runs/.m2166_m2164_icc2_library_import_preflight_attempt_consumed"
MW_REF = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
KNOWN_NDM = Path("/opt/synopsys/icc2/V-2023.12-SP3/libraries/syn/gtech.nlib/reflib.ndm")

EXPECTED = {
    CONTRACT: "1669318c3033b222d3aa55d1f29efc1b2467e97ffc220f64528d5c0c86cc7f74",
    RUNNER: "164703323f9a7f639c753ce8a97708ffb793730ee48ba7e5d850bbcce5186d5e",
    TCL: "4df768db7385fe2c6d2807104f650c925310012a5c21d96e7d396086a3433e65",
    MONITOR: "a4d002f50b3fc45a31f98a2863f1dd39477f81bd219fbc55454b470ec1be56d1",
    INVENTORY: "351db733e16f15895c7f1658b21c16901ff907ed5613cb89c2f4a85ce8928f94",
    CHECKER: "b732130841808f67791eeb9907a327c5149dc94f119507dfcccc65240fbeabc5",
    MASTER_LIST: "e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b",
    MW_MANIFEST: "7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3",
    M2146 / "review.json": "c70e9ce4867d1cbd6010a2da0f403c5ee155a07ee0329888c226a7623ebdd51b",
    M2154 / "review.json": "ebc29bcbddaa0837241bd23a8a473fdbbd762340009f052644946e2590908b39",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    KNOWN_NDM: "56f9a2c14fc9ce7d3d7691146bbc89db35c58a4fe40543833a924a23e8ada829",
}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, raw = line.split(None, 1)
        name = raw.lstrip("* ")
        assert name not in result
        result[name] = digest
    return result


def exhaustive_seal(directory: Path) -> None:
    assert directory.is_dir() and not directory.is_symlink()
    assert not any(path.is_symlink() for path in directory.rglob("*"))
    listed = parse_manifest(directory / "SHA256SUMS")
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    assert set(listed) == actual, sorted(set(listed) ^ actual)
    for name, digest in listed.items():
        assert sha(directory / name) == digest
    assert parse_manifest(directory / "SHA256SUMS.seal.sha256") == {
        "SHA256SUMS": sha(directory / "SHA256SUMS")
    }


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checker = load("m2165_checker", CHECKER)
checks: list[str] = []


def need(condition: bool, label: str) -> None:
    assert condition, label
    checks.append(label)


def rejected(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        checks.append("rejected_" + label)
        return
    raise AssertionError(f"mutation survived: {label}")


# Exact source, predecessor, author-receipt, and protected-document identities.
for path, digest in EXPECTED.items():
    need(path.is_file() and not path.is_symlink(), "regular_" + path.name)
    need(sha(path) == digest, "sha_" + path.name)
for directory in (M2146, M2154, AUTHOR):
    exhaustive_seal(directory)
    checks.append("exhaustive_seal_" + directory.name)
need(not RESULT.exists() and not ATTEMPT.exists(), "m2166_unconsumed")

contract_hashes = parse_manifest(CONTRACT.with_suffix(CONTRACT.suffix + ".sha256"))
need(contract_hashes == {CONTRACT.name: sha(CONTRACT)}, "contract_inner_seal")
outer = parse_manifest(Path(str(CONTRACT) + ".sha256.seal.sha256"))
need(outer == {Path(str(CONTRACT) + ".sha256").name: sha(Path(str(CONTRACT) + ".sha256"))},
     "contract_outer_seal")

contract = json.loads(CONTRACT.read_text())
need(contract["status"] == "SOURCE_ONLY_PENDING_M2165_INDEPENDENT_HAMMER", "contract_source_only")
need(contract["author_authorization"] == {
    "m2165_independent_hammer": True, "m2166": False, "license_queries": 0,
    "top_level_icc2_shell_runs": 0, "pnr_runs": 0, "automatic_retry": False,
}, "contract_author_zero_execution")
need(contract["exact_runtime_budget_after_m2165_pass"] == {
    "license_queries": 1, "top_level_icc2_shell_runs": 1, "pnr_runs": 0,
    "automatic_retry": False,
    "tool_spawned_children": "observed and counted, not additional top-level launches",
}, "contract_exact_one_shot_budget")

# Static one-shot scope and all M2146 startup/isolation repairs.
runner = RUNNER.read_text()
tcl = TCL.read_text()
monitor = MONITOR.read_text()
checker_text = CHECKER.read_text()
need(subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=False).returncode == 0,
     "runner_bash_syntax")
need(runner.count('"${ICC2}" -no_init -f "${TCL}"') == 1, "one_exact_icc2_site")
need(runner.count('"${LMUTIL}" lmstat') == 1, "one_license_query_site")
for token in (
    "env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C",
    'HOME="${ISOLATED}/home"', 'TMPDIR="${ISOLATED}/tmp"',
    'XDG_CACHE_HOME="${ISOLATED}/cache/xdg"', 'cd -- "${ISOLATED}"',
    'M2164_EXPECTED_RUNNER_SHA256', 'M2164_EXPECTED_SOURCE_REVIEW_SHA256',
    "PASS_M2165_M2164_SOURCE_HAMMER__M2166_ONE_SHOT_AUTHORIZED",
    "automatic_retry': False", "'pnr_runs': 0", "mkdir -- \"${LOCK}\" \"${ATTEMPT}\" \"${WORK}\"",
    'sha_executable_exact 4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c "${ICC2_REAL}"',
):
    need(token in runner, "runner_anchor_" + hashlib.sha256(token.encode()).hexdigest()[:12])
need(runner.index('mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"') <
     runner.index('"${LMUTIL}" lmstat') <
     runner.index('"${ICC2}" -no_init -f "${TCL}"'), "attempt_precedes_license_and_icc2")
need("verify_dir_seal \"${M2165}\"" in runner, "m2165_exhaustive_runtime_seal")
need("trap on_exit EXIT INT TERM HUP" in runner and "FAILED_OR_INCOMPLETE_DO_NOT_CITE" in runner,
     "failure_quarantine")

need("get_site_defs -quiet -exact core" in tcl and "*core*" not in tcl, "exact_core_only")
need(tcl.count("generate_frame_from_mw $frame_name -mw_lib $mw_ref") == 1, "one_frame_conversion")
need(tcl.count("create_lib -ref_libs [list $frame_ndm] $design_lib") == 1, "one_design_library")
for command in ("read_verilog", "read_vhdl", "compile_fusion", "initialize_floorplan",
                "create_placement", "place_opt", "clock_opt", "route_auto", "route_opt",
                "report_timing", "report_area", "report_power"):
    need(not re.search(rf"^\s*{command}(?:\s|$)", tcl, re.MULTILINE), "no_" + command)

# M2146 parser, root-inventory, process-classifier, real-exec and master repairs.
for token in ('"icc2_exec"', '"dgcom_exec"', '"lm_shell_exec"', "starttime_ticks",
              "parent_links", "exec_observations", "selected_environment"):
    need(token in monitor, "monitor_" + hashlib.sha256(token.encode()).hexdigest()[:12])
for token in ("observed_masters == frozen_masters", "before == after",
              "actual ICC2 command lacks -no_init", "actual dgcom_exec never observed",
              "process identity count/list mismatch", "non-root identity has no observed parent",
              "root has an observed internal parent", "process identity graph has a disconnected component"):
    need(token in checker_text, "checker_" + hashlib.sha256(token.encode()).hexdigest()[:12])
masters = MASTER_LIST.read_text().splitlines()
need(len(masters) == 94 and masters == sorted(set(masters)), "union94_exact_sorted_unique")

# Rehash the complete immutable physical reference (1,051 regular members).
mw_manifest = parse_manifest(MW_MANIFEST)
mw_actual = {str(path.relative_to(MW_REF)) for path in MW_REF.rglob("*") if path.is_file()}
need(len(mw_manifest) == 1051 and set(mw_manifest) == mw_actual, "mw_inventory_exhaustive_1051")
need(not any(path.is_symlink() for path in MW_REF.rglob("*")), "mw_no_symlinks")
for name, digest in mw_manifest.items():
    need(sha(MW_REF / name) == digest, "mw_member_" + hashlib.sha256(name.encode()).hexdigest()[:16])

# M2154 repair 1: exact same-release 68-byte native header and legal object layout.
expected_header = bytes.fromhex(
    "b2bdea03be02010000104c696272617279204d616e61676572002a562d323032332e31322d53503320666f72206c696e75783634202d2d204d61792030372c2032303234"
)
need(len(expected_header) == 68, "native_header_length_68")
need(KNOWN_NDM.read_bytes()[:68] == expected_header, "native_header_same_release_exact")
need(checker.KNOWN_NDM_HEADER_BYTES == 68 and checker.KNOWN_NDM_SHA == EXPECTED[KNOWN_NDM],
     "checker_header_and_reference_pinned")

import tempfile
with tempfile.TemporaryDirectory(prefix="m2165_native_") as raw:
    root = Path(raw)
    good = root / "frame.ndm"
    good.write_bytes(KNOWN_NDM.read_bytes())
    need(checker.native_ndm_member(good, "known-copy")["size_bytes"] == KNOWN_NDM.stat().st_size,
         "native_known_copy_accepted")
    arbitrary_nul = root / "nul.ndm"
    arbitrary_nul.write_bytes(b"\0FORGED_NOT_LIBRARY_MANAGER")
    rejected("arbitrary_nul_file", lambda: checker.native_ndm_member(arbitrary_nul, "nul"))
    short_header = root / "short.ndm"
    short_header.write_bytes(expected_header[:-1])
    rejected("truncated_67_byte_header", lambda: checker.native_ndm_member(short_header, "short"))
    wrong_release = root / "wrong_release.ndm"
    changed = bytearray(expected_header)
    changed[31] ^= 1
    wrong_release.write_bytes(bytes(changed) + b"payload")
    rejected("wrong_lm_release", lambda: checker.native_ndm_member(wrong_release, "release"))
    directory = root / "directory.ndm"
    directory.mkdir()
    (directory / "native.db").write_bytes(KNOWN_NDM.read_bytes())
    rejected("directory_named_ndm", lambda: checker.native_ndm_member(directory, "dir"))
    wrong_suffix = root / "native.db"
    wrong_suffix.write_bytes(KNOWN_NDM.read_bytes())
    rejected("wrong_native_suffix", lambda: checker.native_ndm_member(wrong_suffix, "suffix"))

# M2154 repair 2: unique rooted, reachable, acyclic, time-ordered process graph.
isolated = Path("/tmp/m2165_graph/isolated_cwd")
env = {"HOME": str(isolated / "home"), "TMPDIR": str(isolated / "tmp"),
       "XDG_CACHE_HOME": str(isolated / "cache/xdg"), "M2153_ISOLATED_CWD": str(isolated)}
wrapper_obs = {"comm": "timeout", "exe_path": "/usr/bin/timeout",
               "cmdline": ["/usr/bin/timeout", str(checker.ICC2_WRAPPER), "-no_init", "-f", str(checker.TCL)],
               "selected_environment": env}
actual_obs = {"comm": "dgcom_exec", "exe_path": str(checker.ICC2_REAL),
              "cmdline": [str(checker.ICC2_REAL), "-no_init", "-f", str(checker.TCL)],
              "selected_environment": env}
identities = [
    {"pid": 700, "starttime_ticks": 1000, "first_ppid": 1,
     "parent_links": [{"ppid": 1, "parent_starttime_ticks": None}], "exec_observations": [wrapper_obs]},
    {"pid": 701, "starttime_ticks": 1001, "first_ppid": 700,
     "parent_links": [{"ppid": 700, "parent_starttime_ticks": 1000}], "exec_observations": [actual_obs]},
]
graph = {
    "schema": "m2153_icc2_process_tree_r1_v1", "root_pid": 700, "root_seen": True,
    "sample_count": 20, "unique_process_identity_count": 2, "exec_observation_count": 2,
    "icc2_wrapper_observation_count": 1,
    "icc2_wrapper_observations": [{"pid": 700, "starttime_ticks": 1000, **wrapper_obs}],
    "icc2_actual_exec_observation_count": 1,
    "icc2_actual_exec_observations": [{"pid": 701, "starttime_ticks": 1001, **actual_obs}],
    "tool_spawned_conversion_exec_observation_count": 0,
    "tool_spawned_conversion_exec_observations": [], "all_observed_processes": identities,
}
need(checker.validate_process_tree(copy.deepcopy(graph), isolated) ==
     {"identities": 2, "exec_observations": 2, "actual_exec_observations": 1}, "valid_process_graph")

def disconnected_cycle() -> None:
    value = copy.deepcopy(graph)
    value["all_observed_processes"][1]["first_ppid"] = 702
    value["all_observed_processes"][1]["parent_links"] = [{"ppid": 702, "parent_starttime_ticks": 1001}]
    value["all_observed_processes"].append({
        "pid": 702, "starttime_ticks": 1001, "first_ppid": 701,
        "parent_links": [{"ppid": 701, "parent_starttime_ticks": 1001}],
        "exec_observations": [{"comm": "helper", "exe_path": "/bin/true",
                               "cmdline": ["/bin/true"], "selected_environment": {}}],
    })
    value["unique_process_identity_count"] = 3
    value["exec_observation_count"] = 3
    checker.validate_process_tree(value, isolated)

rejected("disconnected_actual_cycle", disconnected_cycle)

def reachable_cycle() -> None:
    value = copy.deepcopy(graph)
    value["all_observed_processes"][0]["parent_links"].append(
        {"ppid": 701, "parent_starttime_ticks": 1001})
    checker.validate_process_tree(value, isolated)

rejected("reachable_extra_edge_cycle", reachable_cycle)

def parent_after_child() -> None:
    value = copy.deepcopy(graph)
    value["all_observed_processes"][1]["parent_links"][0]["parent_starttime_ticks"] = 1002
    checker.validate_process_tree(value, isolated)

rejected("parent_starts_after_child", parent_after_child)

def orphan() -> None:
    value = copy.deepcopy(graph)
    value["all_observed_processes"][1]["parent_links"] = [{"ppid": 999, "parent_starttime_ticks": 1}]
    checker.validate_process_tree(value, isolated)

rejected("orphan_nonroot", orphan)

def duplicate_root_pid() -> None:
    value = copy.deepcopy(graph)
    value["all_observed_processes"][1]["pid"] = 700
    checker.validate_process_tree(value, isolated)

rejected("duplicate_root_pid_identity", duplicate_root_pid)

def forged_summary() -> None:
    value = copy.deepcopy(graph)
    value["unique_process_identity_count"] = 99
    checker.validate_process_tree(value, isolated)

rejected("forged_process_summary", forged_summary)

def wrong_actual_env() -> None:
    value = copy.deepcopy(graph)
    value["all_observed_processes"][1]["exec_observations"][0]["selected_environment"]["HOME"] = "/root"
    value["icc2_actual_exec_observations"][0]["selected_environment"]["HOME"] = "/root"
    checker.validate_process_tree(value, isolated)

rejected("actual_exec_wrong_environment", wrong_actual_env)

need(not list(AUTHOR.rglob("__pycache__")), "author_receipt_pycache_free")
print("PASS_M2165_M2164_INDEPENDENT_SOURCE_HAMMER")
print(f"positive_checks={len(checks)}")
print("p0=0")
print("p1=0")
print("p2=0")
print("score=98")
print("m2166_authorized=true")
print("license_queries_authorized=1")
print("top_level_icc2_runs_authorized=1")
print("pnr_runs_authorized=0")
print("review_eda_runs=0")
