#!/usr/bin/python3.12
"""Fail-closed checker for one raw M2182 LM-only conversion preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
LM_SHELL = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/lm_shell")
LM_EXEC = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/lm_shell_exec")
MILKYWAY = Path("/opt/synopsys/starrc/V-2023.12-SP3/linux64_starrc/bin/Milkyway")
TCL = HW / "dc_handoff/scripts/run_lm_m2180_library_conversion_preflight.tcl"
NATIVE_HEADER = bytes.fromhex(
    "b2bdea03be02010000104c696272617279204d616e61676572002a562d323032332e31322d53503320666f72206c696e75783634202d2d204d61792030372c2032303234")


class Failure(RuntimeError):
    pass


def need(ok: bool, message: str) -> None:
    if not ok:
        raise Failure(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    need(path.is_file() and not path.is_symlink(), f"missing/symlink {path}")
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f"not object {path}")
    return value


def parse_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        need(bool(sep) and key not in result, f"invalid/duplicate fact {line}")
        result[key] = value
    return result


def validate_native_frame(path: Path) -> dict[str, object]:
    need(path.is_file() and not path.is_symlink() and path.suffix == ".ndm",
         "frame is not one regular nonsymlink .ndm")
    blob = path.read_bytes()
    need(len(blob) > len(NATIVE_HEADER) and blob[:len(NATIVE_HEADER)] == NATIVE_HEADER,
         "frame native Library Manager header mismatch")
    return {"regular_files": 1, "regular_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def validate_process_tree(payload: dict, isolated: Path) -> dict[str, int]:
    need(payload.get("schema") == "m2180_lm_conversion_process_tree_r1_v1", "process schema")
    need(payload.get("root_seen") is True, "root unseen")
    identities = payload.get("all_observed_processes")
    need(isinstance(identities, list) and identities, "identity list")
    need(payload.get("unique_process_identity_count") == len(identities), "identity count")
    keys: set[tuple[int, int]] = set()
    flat: list[dict] = []
    by_key: dict[tuple[int, int], dict] = {}
    for ident in identities:
        need(isinstance(ident, dict), "identity object")
        key = (ident.get("pid"), ident.get("starttime_ticks"))
        need(all(isinstance(x, int) and x > 0 for x in key) and key not in keys,
             "identity key")
        keys.add(key)
        by_key[key] = ident
        need(isinstance(ident.get("parent_links"), list) and ident["parent_links"], "parent links")
        need(isinstance(ident.get("exec_observations"), list) and ident["exec_observations"], "exec list")
        for obs in ident["exec_observations"]:
            need(isinstance(obs.get("cmdline"), list) and isinstance(obs.get("exe_path"), str), "exec obs")
            flat.append({"pid": key[0], "starttime_ticks": key[1], **obs})
    need(payload.get("exec_observation_count") == len(flat), "exec count")
    root_pid = payload.get("root_pid")
    roots = [key for key in keys if key[0] == root_pid]
    need(len(roots) == 1, "root identity")
    root = roots[0]
    parents: dict[tuple[int, int], set[tuple[int, int]]] = {key: set() for key in keys}
    children: dict[tuple[int, int], set[tuple[int, int]]] = {key: set() for key in keys}
    for key, ident in by_key.items():
        for link in ident["parent_links"]:
            parent = (link.get("ppid"), link.get("parent_starttime_ticks"))
            if parent in keys:
                need(parent != key and parent[1] <= key[1], "process cycle/time")
                parents[key].add(parent)
                children[parent].add(key)
    need(not parents[root], "root internal parent")
    need(all(parents[key] for key in keys - {root}), "orphan process")
    colors: dict[tuple[int, int], int] = {key: 0 for key in keys}
    def visit(node: tuple[int, int]) -> None:
        need(colors[node] != 1, "process graph cycle")
        if colors[node] == 2:
            return
        colors[node] = 1
        for child in children[node]:
            visit(child)
        colors[node] = 2
    visit(root)
    reachable = {root}
    stack = [root]
    while stack:
        for child in children[stack.pop()]:
            if child not in reachable:
                reachable.add(child)
                stack.append(child)
    need(reachable == keys, "disconnected process")
    wrappers = [x for x in flat if str(LM_SHELL) in x["cmdline"]]
    actuals = [x for x in flat if x["exe_path"] == str(LM_EXEC)]
    milkyways = [x for x in flat if x["exe_path"] == str(MILKYWAY)]
    need(payload.get("lm_wrapper_observations") == wrappers and wrappers, "wrapper census")
    need(payload.get("lm_actual_exec_observations") == actuals and actuals, "actual census")
    need(payload.get("milkyway_child_observations") == milkyways and milkyways, "Milkyway census")
    need(payload.get("unexpected_tool_observations") == [], "unexpected tool")
    need(len({(x["pid"], x["starttime_ticks"]) for x in actuals}) == 1, "actual identity count")
    need(len({(x["pid"], x["starttime_ticks"]) for x in milkyways}) == 1, "Milkyway identity count")
    expected_env = {"HOME": str(isolated / "home"), "TMPDIR": str(isolated / "tmp"),
                    "XDG_CACHE_HOME": str(isolated / "cache/xdg"),
                    "M2180_ISOLATED_CWD": str(isolated)}
    for obs in actuals + milkyways:
        need(obs["selected_environment"] == expected_env, "tool isolation environment")
    actual_key = next((x["pid"], x["starttime_ticks"]) for x in actuals)
    descendants = {actual_key}
    stack = [actual_key]
    while stack:
        for child in children[stack.pop()]:
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    need(all((x["pid"], x["starttime_ticks"]) in descendants for x in milkyways),
         "Milkyway not spawned below lm_shell_exec")
    return {"identities": len(keys), "observations": len(flat),
            "actual_identities": 1, "milkyway_identities": 1}


def validate(work: Path, output: Path) -> dict[str, object]:
    work = work.resolve(strict=True)
    need(work.is_dir() and not work.is_symlink() and not output.exists(), "work/output")
    isolated = work / "isolated_cwd"
    frame = isolated / "frame_output/m2180_tcbn28hpcplusbwp35p140_frame.ndm"
    facts_path = isolated / "reports/machine_facts.txt"
    log = work / "lm_preflight.log"
    rc = work / "lm_preflight.rc"
    process_path = work / "process_tree.json"
    execution_path = work / "execution_contract.json"
    for path in (frame, facts_path, log, rc, process_path, execution_path,
                 work / "repo_root_before.json", work / "repo_root_after.json"):
        need(path.exists() and not path.is_symlink(), f"missing/symlink {path}")
    need(rc.read_text().strip() == "0", "LM return code")
    frame_stats = validate_native_frame(frame)
    need(not list(isolated.rglob("*.nlib")), "design .nlib unexpectedly created")
    need([p for p in isolated.rglob("*.ndm") if p != frame] == [], "extra NDM output")
    text = log.read_text(errors="replace")
    exact_lines = [
        f"M2180_GATE1_LOCAL_OUTPUT_ROUND_TRIP_PASS cache={isolated / 'cache/library'}",
        f"M2180_GATE2_MILKYWAY_EXEC_ROUND_TRIP_PASS exec={MILKYWAY}",
        f"M2180_GATE3_FRAME_CONVERSION_PASS status=1 frame={frame}",
        f"M2180_GATE4_NONEMPTY_FRAME_PASS files=1 bytes={frame_stats['regular_bytes']}",
        "RAW_PASS_M2182_M2180_LM_LIBRARY_CONVERSION_PENDING_M2183_INDEPENDENT_RESULT_HAMMER",
    ]
    for line in exact_lines:
        need(len(re.findall(rf"^{re.escape(line)}$", text, re.MULTILINE)) == 1,
             f"exact log line {line}")
    need("M2180_FATAL_FAIL_CLOSED:" not in text and "CMD-005" not in text, "LM fatal diagnostic")
    facts = parse_kv(facts_path)
    expected_facts = {
        "status": "RAW_PASS_M2182_M2180_LM_LIBRARY_CONVERSION_PENDING_M2183",
        "shell": "lm_shell", "local_output_dir": str(isolated / "cache/library"),
        "milkyway_exec": str(MILKYWAY), "conversion_status": "1",
        "frame_ndm": str(frame), "frame_regular_files": "1",
        "frame_regular_bytes": str(frame_stats["regular_bytes"]),
        "design_library_created": "false", "rtl_imported": "false", "pnr_invoked": "false",
    }
    need(facts == expected_facts, "machine facts")
    execution = read_json(execution_path)
    expected_execution = {
        "schema": "m2182_m2180_lm_execution_contract_r1_v1",
        "scope": "lm_library_conversion_only", "license_queries": 1,
        "top_level_lm_shell_runs": 1, "milkyway_children": 1,
        "pnr_runs": 0, "automatic_retry": False,
        "lm_invocation": [str(LM_SHELL), "-no_init", "-f", str(TCL)],
        "lm_shell_sha256": "1b0ce5fb11a8b5b803415c15ebc7395e60df3c921dbf1006aef17e19d086a942",
        "lm_shell_exec_path": str(LM_EXEC),
        "lm_shell_exec_sha256": "3ebfe918bf64fd6d095f29765df5bda01b0d7d3fbfc74027a69fbaf48c8a23ab",
        "milkyway_exec_path": str(MILKYWAY),
        "milkyway_exec_sha256": "09dc7b34acb60b0078be27345db3e1c457f0891c596afe6c27ab2cf02a50c3ec",
        "isolated_root": str(isolated),
    }
    need(execution == expected_execution, "execution contract")
    need(read_json(work / "repo_root_before.json") == read_json(work / "repo_root_after.json"),
         "repository root inventory drift")
    process = validate_process_tree(read_json(process_path), isolated)
    result = {
        "schema": "m2182_m2180_lm_library_conversion_preflight_result_r1_v1",
        "status": "RAW_PASS_M2182_M2180_LM_LIBRARY_CONVERSION_PENDING_M2183_INDEPENDENT_RESULT_HAMMER",
        "frame": frame_stats, "process": process,
        "claim_boundary": {"library_conversion_only": True, "design_library": False,
                           "pnr": False, "timing": False, "area": False,
                           "power": False, "paper_ppa_ready": False},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    value = validate(args.work, args.output)
    print(value["status"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2180_CHECK_FAIL_CLOSED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
