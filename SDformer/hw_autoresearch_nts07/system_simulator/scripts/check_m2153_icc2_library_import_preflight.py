#!/usr/bin/python3.12
"""Fail-closed semantic parser for one raw M2153/M2155 ICC2 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
ICC2_WRAPPER = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell")
ICC2_REAL = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/dgcom_exec")
TCL = HW / "dc_handoff/scripts/run_icc2_m2153_library_import_preflight.tcl"
COLLATERAL_SHA = "0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6"
WRAPPER_SHA = "825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4"
REAL_EXEC_SHA = "4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c"
RC_NAME = "crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical"
METALS = "M1,M2,M3,M4,M5,M6,M7,M8,M9"
VIAS = "VIA1,VIA2,VIA3,VIA4,VIA5,VIA6,VIA7,VIA8"
TOOL_CHILD_NAMES = {
    "icc2_lm_shell",
    "icc2_lm_shell_exec",
    "lm_shell",
    "lm_shell_exec",
    "milkyway_exec",
    "icc_shell_exec",
    "icc2_shell_exec",
    "common_shell_exec",
    "common_shell_exe",
}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"M2153_CHECK_FAIL: {message}")


def parse_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        need("=" in line, f"non-key/value line in {path}: {line!r}")
        key, value = line.split("=", 1)
        need(key and key not in result, f"duplicate/empty key {key!r} in {path}")
        result[key] = value
    return result


def tree_stats(root: Path) -> dict[str, int]:
    need(root.exists() and not root.is_symlink(), f"missing/symlink library object {root}")
    nodes = [root]
    files = 0
    bytes_total = 0
    binary_files = 0
    while nodes:
        node = nodes.pop()
        info = node.lstat()
        need(not stat.S_ISLNK(info.st_mode), f"symlink in library object {node}")
        if stat.S_ISREG(info.st_mode):
            files += 1
            bytes_total += info.st_size
            with node.open("rb") as stream:
                if b"\0" in stream.read(65536):
                    binary_files += 1
        elif stat.S_ISDIR(info.st_mode):
            nodes.extend(sorted(node.iterdir(), key=lambda p: p.name, reverse=True))
        else:
            need(False, f"unsupported node in library object {node}")
    need(files > 0 and bytes_total > 0, f"empty library object {root}")
    return {"regular_files": files, "regular_bytes": bytes_total, "binary_files": binary_files}


def load_root_inventory(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    need(payload.get("schema") == "m2153_repo_root_inventory_r1_v1", "root inventory schema")
    need(payload.get("root") == str(REPO.resolve()), "root inventory path")
    nodes = payload.get("nodes")
    need(isinstance(nodes, list), "root inventory nodes is not a list")
    need(payload.get("node_count") == len(nodes), "root inventory count mismatch")
    names = [item.get("name") for item in nodes]
    need(all(isinstance(name, str) and name and "/" not in name for name in names), "root node name")
    need(names == sorted(set(names)), "root inventory names not unique sorted")
    allowed = {"regular", "directory", "symlink", "fifo", "socket", "block_device", "character_device", "unknown"}
    for item in nodes:
        need(item.get("node_type") in allowed, "root inventory unknown type encoding")
        need(re.fullmatch(r"[0-7]{4}", str(item.get("mode_octal", ""))) is not None, "root node mode")
        if item["node_type"] == "regular":
            need(isinstance(item.get("size_bytes"), int) and item["size_bytes"] >= 0, "root regular size")
            need(re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is not None, "root regular SHA")
        if item["node_type"] == "symlink":
            need(isinstance(item.get("target"), str), "root symlink target")
    return payload


def validate_process_tree(process: dict[str, object], isolated: Path) -> dict[str, int]:
    need(process.get("schema") == "m2153_icc2_process_tree_r1_v1", "process schema")
    need(process.get("root_seen") is True, "process root not observed")
    identities = process.get("all_observed_processes")
    need(isinstance(identities, list) and len(identities) >= 2, "process identity list too small")
    need(process.get("unique_process_identity_count") == len(identities), "process identity count/list mismatch")
    identity_keys: set[tuple[int, int]] = set()
    flattened: list[dict[str, object]] = []
    for identity in identities:
        need(isinstance(identity, dict), "process identity not object")
        pid = identity.get("pid")
        start = identity.get("starttime_ticks")
        need(isinstance(pid, int) and pid > 0 and isinstance(start, int) and start > 0, "invalid pid/starttime")
        key = (pid, start)
        need(key not in identity_keys, "duplicate pid/starttime identity")
        identity_keys.add(key)
        links = identity.get("parent_links")
        observations = identity.get("exec_observations")
        first_ppid = identity.get("first_ppid")
        need(isinstance(first_ppid, int) and first_ppid >= 0, "invalid first_ppid")
        need(isinstance(links, list) and links, "missing parent-link list")
        need(any(link.get("ppid") == first_ppid for link in links), "first_ppid absent from parent links")
        need(isinstance(observations, list) and observations, "missing exec-observation list")
        for observation in observations:
            need(isinstance(observation, dict), "exec observation not object")
            need(isinstance(observation.get("comm"), str), "exec comm")
            need(isinstance(observation.get("exe_path"), str), "exec path")
            need(isinstance(observation.get("cmdline"), list), "exec cmdline")
            need(isinstance(observation.get("selected_environment"), dict), "exec environment")
            flattened.append({"pid": pid, "starttime_ticks": start, **observation})
    root_pid = process.get("root_pid")
    need(isinstance(root_pid, int) and sum(1 for pid, _ in identity_keys if pid == root_pid) == 1, "root identity mismatch")
    for identity in identities:
        key = (identity["pid"], identity["starttime_ticks"])
        if identity["pid"] == root_pid:
            continue
        linked = False
        for link in identity["parent_links"]:
            parent_start = link.get("parent_starttime_ticks")
            if isinstance(link.get("ppid"), int) and isinstance(parent_start, int):
                linked |= (link["ppid"], parent_start) in identity_keys
        need(linked, f"non-root identity has no observed parent {key}")
    need(process.get("exec_observation_count") == len(flattened), "exec observation count/list mismatch")

    wrappers = [item for item in flattened if str(ICC2_WRAPPER) in item["cmdline"]]
    actuals = [item for item in flattened if item["exe_path"] == str(ICC2_REAL)]
    wrapper_summary = process.get("icc2_wrapper_observations")
    need(isinstance(wrapper_summary, list) and wrapper_summary == wrappers, "wrapper list differs from census")
    need(process.get("icc2_wrapper_observation_count") == len(wrapper_summary), "wrapper summary mismatch")
    need(process.get("icc2_wrapper_observation_count") == len(wrappers), "wrapper census mismatch")
    actual_summary = process.get("icc2_actual_exec_observations")
    need(isinstance(actual_summary, list) and actual_summary == actuals, "actual-exec list differs from census")
    need(process.get("icc2_actual_exec_observation_count") == len(actual_summary), "actual-exec summary mismatch")
    need(process.get("icc2_actual_exec_observation_count") == len(actuals), "actual-exec census mismatch")
    need(actuals, "actual dgcom_exec never observed")
    expected_env = {
        "HOME": str(isolated / "home"),
        "TMPDIR": str(isolated / "tmp"),
        "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
        "M2153_ISOLATED_CWD": str(isolated),
    }
    for item in actuals:
        cmdline = item["cmdline"]
        need("-no_init" in cmdline, "actual ICC2 command lacks -no_init")
        need("-f" in cmdline and str(TCL) in cmdline, "actual ICC2 command lacks exact -f Tcl")
        need(item["selected_environment"] == expected_env, "actual ICC2 isolation environment mismatch")
    expected_children = [
        item
        for item in flattened
        if Path(str(item["exe_path"])).name in TOOL_CHILD_NAMES
        or str(item["comm"]) in TOOL_CHILD_NAMES
        or any(Path(arg).name in TOOL_CHILD_NAMES for arg in item["cmdline"])
    ]
    children = process.get("tool_spawned_conversion_exec_observations")
    need(isinstance(children, list) and children == expected_children,
         "conversion-child list differs from census")
    need(process.get("tool_spawned_conversion_exec_observation_count") == len(children), "conversion-child count/list mismatch")
    return {"identities": len(identities), "exec_observations": len(flattened), "actual_exec_observations": len(actuals)}


def validate(work: Path, output: Path) -> dict[str, object]:
    work = work.resolve(strict=True)
    need(work.is_dir() and not work.is_symlink(), "work is not a real directory")
    need(not output.exists(), "receipt output already exists")
    isolated = work / "isolated_cwd"
    reports = isolated / "reports"
    log = work / "icc2_preflight.log"
    rc_file = work / "icc2_preflight.rc"
    facts_file = reports / "machine_facts.txt"
    coverage = reports / "master_coverage.tsv"
    frame = isolated / "frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm"
    design_lib = isolated / "m2153_disposable_design.nlib"
    process_path = work / "process_tree.json"
    execution_path = work / "execution_contract.json"
    before_path = work / "repo_root_before.json"
    after_path = work / "repo_root_after.json"
    ref_report = reports / "reference_libraries.rpt"
    design_report = reports / "design_library.rpt"
    prior = work / "prior_m2135_collateral/icc2_output.txt"
    required = (log, rc_file, facts_file, coverage, frame, design_lib, process_path,
                execution_path, before_path, after_path, ref_report, design_report, prior)
    for path in required:
        need(path.exists() and not path.is_symlink(), f"missing/symlink output {path}")
    need(rc_file.read_text().strip() == "0", "ICC2 return code is nonzero")

    execution = json.loads(execution_path.read_text())
    expected_execution = {
        "schema": "m2155_m2153_execution_contract_r1_v1",
        "scope": "library_import_only",
        "license_queries": 1,
        "top_level_icc2_shell_runs": 1,
        "pnr_runs": 0,
        "automatic_retry": False,
        "icc2_invocation": [str(ICC2_WRAPPER), "-no_init", "-f", str(TCL)],
        "icc2_wrapper_sha256": WRAPPER_SHA,
        "icc2_real_exec_path": str(ICC2_REAL),
        "icc2_real_exec_sha256": REAL_EXEC_SHA,
        "isolated_home": str(isolated / "home"),
        "isolated_tmpdir": str(isolated / "tmp"),
        "isolated_xdg_cache": str(isolated / "cache/xdg"),
        "isolated_library_cache": str(isolated / "cache/library"),
        "prior_m2135_collateral_action": "copied_byte_exact_original_preserved",
        "prior_m2135_collateral_sha256": COLLATERAL_SHA,
    }
    need(execution == expected_execution, "execution contract semantic mismatch")

    text = log.read_text(errors="replace")
    terminal = "RAW_PASS_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156_INDEPENDENT_RESULT_HAMMER"
    need(len(re.findall(rf"^{re.escape(terminal)}$", text, re.MULTILINE)) == 1, "terminal token count")
    expected_lines = [
        f"M2153_GATE1_OPTION_ROUND_TRIP_PASS cache={isolated / 'cache/library'}",
        f"M2153_GATE2_FRAME_CONVERSION_PASS status=1 frame={frame}",
        "M2153_GATE3_MASTER_COVERAGE_PASS count=94 views=4",
        f"M2153_GATE5_RC_COMPATIBILITY_PASS name={RC_NAME}",
    ]
    for line in expected_lines:
        need(len(re.findall(rf"^{re.escape(line)}$", text, re.MULTILINE)) == 1, f"exact gate line {line}")
    gate4 = re.findall(
        rf"^M2153_GATE4_PHYSICAL_TECH_PASS site=core site_count=1 metals={re.escape(METALS)} vias={re.escape(VIAS)} tech=([^\s]+)$",
        text,
        re.MULTILINE,
    )
    need(len(gate4) == 1 and gate4[0], "exact gate4 semantics")
    gate6 = re.findall(
        r"^M2153_GATE6_NONEMPTY_LIBRARY_OBJECTS_PASS frame_files=([1-9][0-9]*) frame_bytes=([1-9][0-9]*) design_files=([1-9][0-9]*) design_bytes=([1-9][0-9]*)$",
        text,
        re.MULTILINE,
    )
    need(len(gate6) == 1, "exact gate6 semantics")
    need(not re.search(r"^M2153_FATAL_FAIL_CLOSED:", text, re.MULTILINE), "runtime fatal token")
    for token in ("CMD-104", "LIB-117", "FILE-001", "LIB-027"):
        need(token not in text, f"runtime failure diagnostic {token}")

    frame_stats = tree_stats(frame)
    design_stats = tree_stats(design_lib)
    need(frame_stats["binary_files"] > 0, "frame NDM has no binary database member")
    need(design_stats["binary_files"] > 0, "design library has no binary database member")
    gate_values = tuple(int(value) for value in gate6[0])
    need(gate_values == (
        frame_stats["regular_files"], frame_stats["regular_bytes"],
        design_stats["regular_files"], design_stats["regular_bytes"],
    ), "gate6 tree statistics mismatch")
    for report in (ref_report, design_report):
        need(report.is_file() and report.stat().st_size >= 64, f"empty ICC2 object report {report}")

    facts = parse_kv(facts_file)
    expected_facts = {
        "status": "RAW_PASS_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156",
        "application_option_value": str(isolated / "cache/library"),
        "conversion_status": "1",
        "frame_ndm": str(frame),
        "frame_regular_files": str(frame_stats["regular_files"]),
        "frame_regular_bytes": str(frame_stats["regular_bytes"]),
        "design_lib": str(design_lib),
        "design_regular_files": str(design_stats["regular_files"]),
        "design_regular_bytes": str(design_stats["regular_bytes"]),
        "mapped_master_union_count": "94",
        "tt_master_coverage": "94",
        "ss_master_coverage": "94",
        "ff_master_coverage": "94",
        "physical_master_coverage": "94",
        "core_site_name": "core",
        "core_site_count": "1",
        "routing_layers": METALS,
        "via_layers": VIAS,
        "current_technology": gate4[0],
        "rc_technology_name": RC_NAME,
        "rtl_imported": "false",
        "pnr_invoked": "false",
    }
    for key, value in expected_facts.items():
        need(facts.get(key) == value, f"machine fact {key} mismatch")
    for key in ("current_library", "tt_library", "ss_library", "ff_library", "physical_library"):
        need(bool(facts.get(key)), f"missing legal library object fact {key}")

    frozen_masters = MASTER_LIST.read_text().splitlines()
    need(len(frozen_masters) == 94 and frozen_masters == sorted(set(frozen_masters)), "frozen union94 malformed")
    lines = coverage.read_text().splitlines()
    need(lines and lines[0] == "master\ttt\tss\tff\tphysical", "coverage header")
    need(len(lines) == 95, "coverage row count")
    observed_masters: list[str] = []
    for line in lines[1:]:
        fields = line.split("\t")
        need(len(fields) == 5, "coverage field count")
        need(all(re.fullmatch(r"[1-9][0-9]*", value) for value in fields[1:]), "coverage positive views")
        observed_masters.append(fields[0])
    need(observed_masters == frozen_masters, "coverage names differ from frozen union94")

    process = json.loads(process_path.read_text())
    process_summary = validate_process_tree(process, isolated)
    before = load_root_inventory(before_path)
    after = load_root_inventory(after_path)
    need(before == after, "repository-root all-node inventory changed")
    collateral_nodes = [item for item in before["nodes"] if item["name"] == "icc2_output.txt"]
    need(len(collateral_nodes) == 1, "M2135 collateral absent/repeated in root inventory")
    need(collateral_nodes[0].get("node_type") == "regular" and collateral_nodes[0].get("sha256") == COLLATERAL_SHA,
         "M2135 collateral root identity mismatch")
    need(prior.is_file() and sha(prior) == COLLATERAL_SHA, "M2135 collateral copy identity mismatch")

    payload = {
        "schema": "m2155_m2153_icc2_library_import_preflight_raw_r1_v1",
        "milestone_source": "M2153",
        "milestone_execution": "M2155",
        "status": "RAW_PASS_M2155_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156_INDEPENDENT_RESULT_HAMMER",
        "claim_boundary": {
            "library_import_preflight_only": True,
            "rtl_imported": False,
            "pnr": False,
            "timing": False,
            "area": False,
            "power": False,
            "paper_ppa_ready": False,
            "authorizes_full_pnr": False,
        },
        "gates": {
            "option_round_trip": True,
            "frame_conversion_status": 1,
            "mapped_master_union_count": 94,
            "logical_physical_views_per_master": 4,
            "exact_core_site": "core",
            "routing_metal_layers": 9,
            "via_layers": 8,
            "rc_technology_name": RC_NAME,
            "frame_binary_files": frame_stats["binary_files"],
            "design_binary_files": design_stats["binary_files"],
        },
        "process_census": process_summary,
        "identity": {
            "icc2_log_sha256": sha(log),
            "machine_facts_sha256": sha(facts_file),
            "master_coverage_sha256": sha(coverage),
            "process_tree_sha256": sha(process_path),
            "root_inventory_sha256": sha(before_path),
            "prior_m2135_collateral_sha256": sha(prior),
            "icc2_wrapper_sha256": WRAPPER_SHA,
            "icc2_real_exec_sha256": REAL_EXEC_SHA,
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate(args.work, args.output)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(str(error))
    print("PASS_RAW_M2155_M2153_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2156_INDEPENDENT_RESULT_HAMMER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
