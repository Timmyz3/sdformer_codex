#!/usr/bin/env python3
"""Strictly audit fresh M31-r4 Formality evidence and diagnostic population."""

import argparse
import hashlib
import json
import re
from pathlib import Path


DESIGN = "qfit_atlif_unified_t10_t2_stream_core"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("missing M31 Formality {}".format(label))
    return path


def read_json_no_duplicates(path):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate Formality manifest key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=hook)


def unique_pairs(rows, label):
    values = set((int(a), int(b)) for a, b in rows)
    if len(values) != 1:
        raise ValueError("M31 Formality {} population drift".format(label))
    return list(values)[0]


def build(run_dir, attempt, expected_passing=None):
    run = Path(run_dir).resolve()
    log_path = require(run / "formality_{}.log".format(attempt), "attempt log")
    exit_path = require(run / "formality_{}.exit_status".format(attempt),
                        "attempt exit status")
    status_path = require(run / "reports/formality_status.txt", "status")
    unmatched_path = require(run / "reports/formality_unmatched.rpt",
                             "unmatched report")
    verify_path = require(run / "reports/formality_verify.rpt", "failing report")
    manifest_path = require(run / "formality_run_manifest.json", "manifest")
    if exit_path.read_text(encoding="utf-8").strip() != "0":
        raise ValueError("M31 Formality attempt exit status is nonzero")
    if status_path.read_text(encoding="utf-8").strip() != "PASS":
        raise ValueError("M31 Formality status is not PASS")
    log = log_path.read_text(encoding="utf-8")
    unmatched = unmatched_path.read_text(encoding="utf-8")
    verify = verify_path.read_text(encoding="utf-8")
    if log.splitlines().count("Verification SUCCEEDED") != 1:
        raise ValueError("M31 Formality success marker population drift")
    if re.search(r"^\s*(Error|Fatal):", log, re.MULTILINE):
        raise ValueError("M31 Formality log contains an error or fatal")
    fmr_diagnostics = re.findall(
        r"^\s*(?:Warning|Error):.*\(FMR_ELAB-147\)\s*$", log,
        re.MULTILINE,
    )
    fmr_summaries = [int(value) for value in re.findall(
        r"^\s*(\d+) FMR_ELAB-147 messages produced\s*$", log,
        re.MULTILINE,
    )]
    if fmr_diagnostics or any(value != 0 for value in fmr_summaries):
        raise ValueError("M31 Formality FMR_ELAB-147 diagnostic population is nonzero")
    disagree_count = len(re.findall(
        r"Verification results may disagree with a logic simulator\.", log))
    if disagree_count:
        raise ValueError("M31 Formality logic-simulator disagreement warning found")
    passing_rows = [int(value) for value in re.findall(
        r"^\s+(\d+) Passing compare points\s*$", log, re.MULTILINE)]
    if len(passing_rows) != 1 or passing_rows[0] <= 0:
        raise ValueError("M31 Formality passing compare-point population drift")
    passing = passing_rows[0]
    if expected_passing is not None and passing != int(expected_passing):
        raise ValueError("M31 Formality passing compare-point count drift")
    failing_rows = re.findall(
        r"^Failing \(not equivalent\)\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", log, re.MULTILINE)
    if len(failing_rows) != 1 or any(int(value) for value in failing_rows[0]):
        raise ValueError("M31 Formality failing compare-point population is nonzero")
    unmatched_compare = unique_pairs(re.findall(
        r"^\s*(\d+)\((\d+)\) Unmatched reference\(implementation\) "
        r"compare points\s*$", log, re.MULTILINE), "unmatched compare points")
    unmatched_primary = unique_pairs(re.findall(
        r"^\s*(\d+)\((\d+)\) Unmatched reference\(implementation\) "
        r"primary inputs, black-box outputs\s*$", log, re.MULTILINE),
        "unmatched primary inputs")
    unread = unique_pairs(re.findall(
        r"^\s*(\d+)\((\d+)\) Unmatched reference\(implementation\) "
        r"unread points\s*$", log, re.MULTILINE), "unread points")
    if unmatched_compare != (0, 0) or unmatched_primary != (0, 0):
        raise ValueError("M31 Formality unmatched compare/input population is nonzero")
    if unmatched.splitlines().count("No unmatched points.") != 1:
        raise ValueError("M31 Formality unmatched report is not exactly closed")
    if verify.splitlines().count("No failing compare points.") != 1:
        raise ValueError("M31 Formality failing report is not exactly closed")

    manifest = read_json_no_duplicates(manifest_path)
    if manifest.get("mode") != "formality" or manifest.get("design_name") != DESIGN:
        raise ValueError("M31 Formality manifest mode/design drift")
    paths = manifest.get("paths", {})
    if set(paths) != {"LIB_DB", "RTL_FILELIST", "MAPPED_NETLIST"}:
        raise ValueError("M31 Formality manifest path population drift")
    live_paths = {}
    for name, item in paths.items():
        if set(item) != {"path", "sha256"}:
            raise ValueError("M31 Formality manifest item schema drift")
        path = Path(item["path"]).resolve()
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError("M31 Formality manifest live path drift: {}".format(name))
        live_paths[name] = path
    if live_paths["MAPPED_NETLIST"].parent.parent != run:
        raise ValueError("M31 Formality mapped netlist is outside run directory")

    return {
        "schema": "m31_r4_fresh_formality_audit_v1",
        "status": "PASS_M31_R4_RTL_TO_FRESH_MAPPED_NETLIST_FORMALITY_STRICT",
        "identity": {
            "run_directory": str(run), "attempt": attempt,
            "log_sha256": sha256(log_path),
            "exit_status_sha256": sha256(exit_path),
            "status_sha256": sha256(status_path),
            "unmatched_report_sha256": sha256(unmatched_path),
            "failing_report_sha256": sha256(verify_path),
            "manifest_sha256": sha256(manifest_path),
            "mapped_netlist_sha256": sha256(live_paths["MAPPED_NETLIST"]),
        },
        "verification": {
            "passing_compare_points": passing,
            "failing_compare_points": 0,
            "unmatched_reference_compare_points": 0,
            "unmatched_implementation_compare_points": 0,
            "unmatched_reference_primary_or_blackbox_points": 0,
            "unmatched_implementation_primary_or_blackbox_points": 0,
            "unread_reference_points": unread[0],
            "unread_implementation_points": unread[1],
            "fmr_elab_147_diagnostics": 0,
            "logic_simulator_disagreement_warnings": 0,
        },
        "admission": {
            "rtl_to_exact_mapped_netlist_equivalence_admitted": True,
            "dc_sta_identity_inherited_from_external_machine_audit": True,
            "ppa_power_energy_admitted": False,
            "system_speedup_admitted": False,
            "headline_admitted": False,
        },
    }


def write_output(path, result):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite M31 Formality machine audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--expected-passing", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.run_dir, args.attempt, args.expected_passing)
    write_output(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
