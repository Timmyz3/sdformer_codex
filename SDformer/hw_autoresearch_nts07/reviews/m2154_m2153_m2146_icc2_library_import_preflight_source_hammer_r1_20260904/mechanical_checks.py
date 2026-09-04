#!/usr/bin/python3.12
"""Independent fail-closed source hammer for M2153.

This audit invokes no ICC2/EDA executable, no license client, and no GPU.  It
checks the source identities and M2146 repairs, then attacks the M2153 raw
parser with synthetic evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import socket
import stat
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m2153_m2146_icc2_library_import_preflight_source_contract_r1_20260904.json"
RUNNER = HW / "dc_handoff/scripts/run_m2153_m2146_icc2_library_import_preflight_one_shot.sh"
TCL = HW / "dc_handoff/scripts/run_icc2_m2153_library_import_preflight.tcl"
MONITOR = HW / "dc_handoff/scripts/monitor_m2153_icc2_process_tree.py"
INVENTORY = HW / "dc_handoff/scripts/inventory_m2153_repo_root.py"
CHECKER = HW / "system_simulator/scripts/check_m2153_icc2_library_import_preflight.py"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
MW_MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
AUTHOR = HW / "reviews/m2153_m2146_icc2_library_import_preflight_source_author_receipt_r1_20260904"
M2146 = HW / "reviews/m2146_m2141_m2136_icc2_library_import_preflight_source_hammer_r1_20260904"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2155_RESULT = HW / "dc_handoff/runs/m2155_m2153_icc2_library_import_preflight_raw_r1_20260904"
M2155_ATTEMPT = HW / "dc_handoff/runs/.m2155_m2153_icc2_library_import_preflight_attempt_consumed"

M2029 = HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
NETLISTS = [
    M2029 / "ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
    M2029 / "tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
]

MW_REF = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140")
ICC2_WRAPPER = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell")
ICC2_REAL = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/dgcom_exec")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
KNOWN_NDM = Path("/opt/synopsys/icc2/V-2023.12-SP3/libraries/syn/gtech.nlib/reflib.ndm")
COLLATERAL = REPO / "icc2_output.txt"

EXPECTED = {
    CONTRACT: "69fdd225fcd4e81cfa76fc85885394241e8b9b69dfa74ebfd92ce2f26943752b",
    RUNNER: "b0479dc7fecd9cef9dcce38a878a2c0bf6c17d514b7dfa17d5092c175bc42dfa",
    TCL: "4df768db7385fe2c6d2807104f650c925310012a5c21d96e7d396086a3433e65",
    MONITOR: "a4d002f50b3fc45a31f98a2863f1dd39477f81bd219fbc55454b470ec1be56d1",
    INVENTORY: "351db733e16f15895c7f1658b21c16901ff907ed5613cb89c2f4a85ce8928f94",
    CHECKER: "e59fc54e9df9a35bc4980397d776813e7e58bf3aea21287d25c77cb077081a49",
    MASTER_LIST: "e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b",
    M2146 / "review.json": "c70e9ce4867d1cbd6010a2da0f403c5ee155a07ee0329888c226a7623ebdd51b",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    MW_MANIFEST: "7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3",
    ICC2_WRAPPER: "825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4",
    ICC2_REAL: "4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    COLLATERAL: "0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6",
    NETLISTS[0]: "f5847f355329a52511ab044ef458284a19ae424ac778418a4bc4778bb2d3a2b0",
    NETLISTS[1]: "739eb76dcb732ec0c66b75392c768cbe36027ecc5d458bd4b088f8488f67c9af",
}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def manifest(path: Path) -> dict[str, str]:
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
    listed = manifest(directory / "SHA256SUMS")
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    assert set(listed) == actual, (sorted(set(listed) ^ actual))
    for rel, digest in listed.items():
        assert sha(directory / rel) == digest
    assert manifest(directory / "SHA256SUMS.seal.sha256") == {
        "SHA256SUMS": sha(directory / "SHA256SUMS")
    }


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checker = load("m2154_checker", CHECKER)
inventory_module = load("m2154_inventory", INVENTORY)
checks: list[str] = []


def need(condition: bool, label: str) -> None:
    assert condition, label
    checks.append(label)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_forged_work(root: Path) -> Path:
    """Build structurally consistent evidence without invoking ICC2."""
    work = root.resolve()
    isolated = work / "isolated_cwd"
    reports = isolated / "reports"
    frame = isolated / "frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm"
    design = isolated / "m2153_disposable_design.nlib"
    reports.mkdir(parents=True)
    frame.mkdir(parents=True)
    design.mkdir(parents=True)
    # Deliberately not a Synopsys NDM: one NUL is enough for current parser.
    (frame / "native.db").write_bytes(b"\0FORGED_FRAME_NOT_AN_NDM\n")
    (design / "native.db").write_bytes(b"\0FORGED_DESIGN_NOT_AN_NDM\n")
    (reports / "reference_libraries.rpt").write_text("FORGED " + "R" * 96 + "\n")
    (reports / "design_library.rpt").write_text("FORGED " + "D" * 96 + "\n")
    frame_stats = checker.tree_stats(frame)
    design_stats = checker.tree_stats(design)
    tech = "forged_technology"
    log = [
        f"M2153_GATE1_OPTION_ROUND_TRIP_PASS cache={isolated / 'cache/library'}",
        f"M2153_GATE2_FRAME_CONVERSION_PASS status=1 frame={frame}",
        "M2153_GATE3_MASTER_COVERAGE_PASS count=94 views=4",
        f"M2153_GATE4_PHYSICAL_TECH_PASS site=core site_count=1 metals={checker.METALS} vias={checker.VIAS} tech={tech}",
        f"M2153_GATE5_RC_COMPATIBILITY_PASS name={checker.RC_NAME}",
        "M2153_GATE6_NONEMPTY_LIBRARY_OBJECTS_PASS "
        f"frame_files={frame_stats['regular_files']} frame_bytes={frame_stats['regular_bytes']} "
        f"design_files={design_stats['regular_files']} design_bytes={design_stats['regular_bytes']}",
        "RAW_PASS_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156_INDEPENDENT_RESULT_HAMMER",
    ]
    (work / "icc2_preflight.log").write_text("\n".join(log) + "\n")
    (work / "icc2_preflight.rc").write_text("0\n")
    facts = {
        "status": "RAW_PASS_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156",
        "application_option_value": str(isolated / "cache/library"),
        "conversion_status": "1", "frame_ndm": str(frame),
        "frame_regular_files": str(frame_stats["regular_files"]),
        "frame_regular_bytes": str(frame_stats["regular_bytes"]),
        "design_lib": str(design),
        "design_regular_files": str(design_stats["regular_files"]),
        "design_regular_bytes": str(design_stats["regular_bytes"]),
        "current_library": "forged_current", "tt_library": "forged_tt",
        "ss_library": "forged_ss", "ff_library": "forged_ff",
        "physical_library": "forged_physical", "mapped_master_union_count": "94",
        "tt_master_coverage": "94", "ss_master_coverage": "94",
        "ff_master_coverage": "94", "physical_master_coverage": "94",
        "core_site_name": "core", "core_site_count": "1",
        "routing_layers": checker.METALS, "via_layers": checker.VIAS,
        "current_technology": tech, "rc_technology_name": checker.RC_NAME,
        "rtl_imported": "false", "pnr_invoked": "false",
    }
    (reports / "machine_facts.txt").write_text("".join(f"{k}={v}\n" for k, v in facts.items()))
    masters = checker.MASTER_LIST.read_text().splitlines()
    (reports / "master_coverage.tsv").write_text(
        "master\ttt\tss\tff\tphysical\n" + "".join(f"{name}\t1\t1\t1\t1\n" for name in masters)
    )
    env = {"HOME": str(isolated / "home"), "TMPDIR": str(isolated / "tmp"),
           "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
           "M2153_ISOLATED_CWD": str(isolated)}
    wrapper = {"comm": "sh", "exe_path": "/usr/bin/dash",
               "cmdline": ["/bin/sh", str(checker.ICC2_WRAPPER), "-no_init", "-f", str(checker.TCL)],
               "selected_environment": env}
    actual = {"comm": "icc2_exec", "exe_path": str(checker.ICC2_REAL),
              "cmdline": [str(checker.ICC2_REAL), "-root_path", "/opt/synopsys/icc2/V-2023.12-SP3",
                          "-no_init", "-f", str(checker.TCL)],
              "selected_environment": env}
    identities = [
        {"pid": 100, "starttime_ticks": 1000, "first_ppid": 1,
         "parent_links": [{"ppid": 1, "parent_starttime_ticks": None}],
         "exec_observations": [wrapper]},
        {"pid": 101, "starttime_ticks": 1001, "first_ppid": 100,
         "parent_links": [{"ppid": 100, "parent_starttime_ticks": 1000}],
         "exec_observations": [actual]},
    ]
    process = {
        "schema": "m2153_icc2_process_tree_r1_v1", "root_pid": 100, "root_seen": True,
        "sample_count": 20, "unique_process_identity_count": 2, "exec_observation_count": 2,
        "icc2_wrapper_observation_count": 1,
        "icc2_wrapper_observations": [{"pid": 100, "starttime_ticks": 1000, **wrapper}],
        "icc2_actual_exec_observation_count": 1,
        "icc2_actual_exec_observations": [{"pid": 101, "starttime_ticks": 1001, **actual}],
        "tool_spawned_conversion_exec_observation_count": 0,
        "tool_spawned_conversion_exec_observations": [], "all_observed_processes": identities,
    }
    write_json(work / "process_tree.json", process)
    execution = {
        "schema": "m2155_m2153_execution_contract_r1_v1", "scope": "library_import_only",
        "license_queries": 1, "top_level_icc2_shell_runs": 1, "pnr_runs": 0,
        "automatic_retry": False,
        "icc2_invocation": [str(checker.ICC2_WRAPPER), "-no_init", "-f", str(checker.TCL)],
        "icc2_wrapper_sha256": checker.WRAPPER_SHA,
        "icc2_real_exec_path": str(checker.ICC2_REAL),
        "icc2_real_exec_sha256": checker.REAL_EXEC_SHA,
        "isolated_home": str(isolated / "home"), "isolated_tmpdir": str(isolated / "tmp"),
        "isolated_xdg_cache": str(isolated / "cache/xdg"),
        "isolated_library_cache": str(isolated / "cache/library"),
        "prior_m2135_collateral_action": "copied_byte_exact_original_preserved",
        "prior_m2135_collateral_sha256": checker.COLLATERAL_SHA,
    }
    write_json(work / "execution_contract.json", execution)
    root_inventory = inventory_module.inventory(checker.REPO)
    write_json(work / "repo_root_before.json", root_inventory)
    write_json(work / "repo_root_after.json", root_inventory)
    prior = work / "prior_m2135_collateral/icc2_output.txt"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(COLLATERAL.read_bytes())
    return work


# Exact additive-source identities and predecessor seals.
for path, digest in EXPECTED.items():
    need(path.is_file() and not path.is_symlink(), f"regular_identity_{path.name}")
    need(sha(path) == digest, f"sha_identity_{path.name}")
exhaustive_seal(M2146)
exhaustive_seal(AUTHOR)
need(not M2155_RESULT.exists() and not M2155_ATTEMPT.exists(), "m2155_unconsumed")

# M2146 P0/P2 source repairs.
runner = RUNNER.read_text()
tcl = TCL.read_text()
monitor = MONITOR.read_text()
checker_text = CHECKER.read_text()
inventory_text = INVENTORY.read_text()
need(subprocess.run(["bash", "-n", str(RUNNER)], check=False).returncode == 0, "runner_bash_syntax")
need(runner.count('"${ICC2}" -no_init -f "${TCL}"') == 1, "one_exact_no_init_icc2_site")
need(runner.count('"${LMUTIL}" lmstat') == 1, "one_license_query_site")
for token in ('env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C',
              'HOME="${ISOLATED}/home"', 'TMPDIR="${ISOLATED}/tmp"',
              'XDG_CACHE_HOME="${ISOLATED}/cache/xdg"', 'cd -- "${ISOLATED}"',
              'sha_executable_exact 4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c "${ICC2_REAL}"'):
    need(token in runner, "runner_anchor_" + hashlib.sha256(token.encode()).hexdigest()[:12])
need("get_site_defs -quiet -exact core" in tcl and "*core*" not in tcl, "exact_core_no_wildcard")
need(tcl.count("generate_frame_from_mw $frame_name -mw_lib $mw_ref") == 1, "one_frame_conversion")
need("-overwrite" not in "\n".join(line for line in tcl.splitlines() if not line.lstrip().startswith("#")),
     "no_frame_overwrite")
for command in ("read_verilog", "read_vhdl", "compile_fusion", "initialize_floorplan",
                "create_placement", "place_opt", "clock_opt", "route_auto", "route_opt",
                "report_timing", "report_area", "report_power"):
    need(not re.search(rf"^\s*{command}(?:\s|$)", tcl, re.MULTILINE), f"no_tcl_{command}")

# Complete union94 and Milkyway identities.
masters = MASTER_LIST.read_text().splitlines()
cell_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\s*\(", re.MULTILINE)
union: set[str] = set()
for path in NETLISTS:
    union.update(match.group(1) for match in cell_re.finditer(path.read_text())
                 if match.group(1).endswith("BWP35P140"))
need(len(masters) == 94 and masters == sorted(set(masters)), "union94_sorted_unique")
need(masters == sorted(union), "union94_equals_netlists")
mw_listed = manifest(MW_MANIFEST)
mw_actual = {str(path.relative_to(MW_REF)) for path in MW_REF.rglob("*")
             if path.is_file() and not path.is_symlink()}
need(len(mw_actual) == 1051 and set(mw_listed) == mw_actual, "milkyway_1051_path_exhaustive")
need(not any(path.is_symlink() for path in MW_REF.rglob("*")), "milkyway_no_symlink")
for rel, digest in mw_listed.items():
    need(sha(MW_REF / rel) == digest, "mw_sha_" + hashlib.sha256(rel.encode()).hexdigest()[:16])

# Process/root/parser repair anchors.
for token in ('"icc2_exec"', '"dgcom_exec"', '"lm_shell_exec"', 'starttime_ticks',
              'parent_links', 'exec_observations', 'selected_environment'):
    need(token in monitor, "monitor_anchor_" + hashlib.sha256(token.encode()).hexdigest()[:12])
for token in ('observed_masters == frozen_masters', 'before == after',
              'actual ICC2 command lacks -no_init', 'actual dgcom_exec never observed',
              'process identity count/list mismatch', 'non-root identity has no observed parent'):
    need(token in checker_text, "checker_anchor_" + hashlib.sha256(token.encode()).hexdigest()[:12])
for token in ('stat.S_ISREG', 'stat.S_ISDIR', 'stat.S_ISLNK', 'stat.S_ISFIFO',
              'stat.S_ISSOCK', 'stat.S_ISBLK', 'stat.S_ISCHR'):
    need(token in inventory_text, "inventory_anchor_" + hashlib.sha256(token.encode()).hexdigest()[:12])

# The current native-database predicate is only a NUL-byte heuristic.  Show
# that a fully synthetic NUL-prefixed file and invented reports still pass.
with tempfile.TemporaryDirectory(prefix="m2154_fake_native_") as raw:
    work = make_forged_work(Path(raw) / "work")
    frame_fake = work / "isolated_cwd/frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm/native.db"
    need(frame_fake.read_bytes()[:4] != KNOWN_NDM.read_bytes()[:4], "forged_header_differs_from_installed_ndm")
    forged = checker.validate(work, work / "receipt.json")
    need(forged["gates"]["frame_binary_files"] == 1, "finding_nul_prefixed_fake_native_survives")

# Show that local-parent checks do not prove all observations descend from the
# monitored root: a disconnected two-node cycle carrying dgcom_exec survives.
with tempfile.TemporaryDirectory(prefix="m2154_process_cycle_") as raw:
    work = make_forged_work(Path(raw) / "work")
    process_path = work / "process_tree.json"
    process = json.loads(process_path.read_text())
    process["all_observed_processes"][1]["first_ppid"] = 102
    process["all_observed_processes"][1]["parent_links"] = [
        {"ppid": 102, "parent_starttime_ticks": 1002}
    ]
    process["all_observed_processes"].append({
        "pid": 102, "starttime_ticks": 1002, "first_ppid": 101,
        "parent_links": [{"ppid": 101, "parent_starttime_ticks": 1001}],
        "exec_observations": [{"comm": "sleep", "exe_path": "/usr/bin/sleep",
                               "cmdline": ["sleep", "1"], "selected_environment": {}}],
    })
    process["unique_process_identity_count"] = 3
    process["exec_observation_count"] = 3
    write_json(process_path, process)
    forged = checker.validate(work, work / "receipt.json")
    need(forged["process_census"]["identities"] == 3, "finding_disconnected_parent_cycle_survives")

# Confirm several repaired semantic mutations are rejected.
def rejected(name: str, mutate) -> None:
    with tempfile.TemporaryDirectory(prefix=f"m2154_reject_{name}_") as raw:
        work = make_forged_work(Path(raw) / "work")
        mutate(work)
        try:
            checker.validate(work, work / "receipt.json")
        except Exception:
            checks.append("rejected_" + name)
            return
        raise AssertionError(f"mutation survived unexpectedly: {name}")


rejected("wrong_master", lambda work: (work / "isolated_cwd/reports/master_coverage.tsv").write_text(
    (work / "isolated_cwd/reports/master_coverage.tsv").read_text().replace(masters[0], "FORGED_MASTER", 1)))
rejected("pure_text_db", lambda work: (work / "isolated_cwd/frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm/native.db").write_text("text only\n"))
rejected("changed_root_inventory", lambda work: (work / "repo_root_after.json").write_text("{}\n"))

def missing_no_init(work: Path) -> None:
    path = work / "process_tree.json"
    payload = json.loads(path.read_text())
    payload["all_observed_processes"][1]["exec_observations"][0]["cmdline"].remove("-no_init")
    payload["icc2_actual_exec_observations"][0]["cmdline"].remove("-no_init")
    write_json(path, payload)


rejected("actual_missing_no_init", missing_no_init)

# Runtime inventory covers the user-creatable special node classes.
with tempfile.TemporaryDirectory(prefix="m2154_inventory_") as raw:
    root = Path(raw) / "root"
    root.mkdir()
    (root / "regular").write_text("x")
    (root / "directory").mkdir()
    (root / "symlink").symlink_to("regular")
    os.mkfifo(root / "fifo")
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.bind(str(root / "socket"))
    try:
        types = {item["node_type"] for item in inventory_module.inventory(root)["nodes"]}
    finally:
        endpoint.close()
    need({"regular", "directory", "symlink", "fifo", "socket"} <= types,
         "runtime_root_node_types")

need(not list(AUTHOR.rglob("__pycache__")), "author_receipt_no_pycache")
print("PASS_M2154_HAMMER_COMPLETED_WITH_FAIL_VERDICT")
print(f"positive_checks={len(checks)}")
print("p0=0")
print("p1=2")
print("p2=0")
print("score=91")
print("m2155_authorized=false")
print("icc2_runs=0")
print("license_queries=0")
print("pnr_runs=0")
