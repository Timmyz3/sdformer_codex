#!/usr/bin/env python3
"""Pre-input two-row dry run for M459R3 CSV/JSON/double-seal output paths."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("refusing micro output overwrite")
    args.output_dir.mkdir(parents=True)

    spec = importlib.util.spec_from_file_location("m459r3_wrapper", str(args.wrapper.resolve()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base, patched = module.build_patched_source()
    compile(patched, str(base) + "<M459R3_PREINPUT_COMPILE>", "exec")
    namespace = {"__file__": str(args.wrapper.resolve()), "__name__": "m459r3_preinput"}
    exec(compile(patched, str(base) + "<M459R3_PREINPUT_DEFS>", "exec"), namespace)

    tree = ast.parse(patched)
    phase_fields = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "phase_fields"
                for target in node.targets):
            phase_fields = ast.literal_eval(node.value)
            break
    if phase_fields is None:
        raise SystemExit("cannot extract patched phase_fields")
    for width in (1, 2, 4, 8):
        phase_fields.extend(("zero_issues_b{}".format(width),
                             "pwp_issues_b{}".format(width),
                             "correction_issues_b{}".format(width)))
    required_diagnostics = {
        "reconstruction_mismatches", "residual_count_mismatches",
        "plus_minus_overlap_mismatches"}
    if not required_diagnostics <= set(phase_fields):
        raise SystemExit("patched phase_fields still misses diagnostics")
    if len(phase_fields) != len(set(phase_fields)):
        raise SystemExit("patched phase_fields contains duplicates")

    rows = []
    for index in range(2):
        row = {field: index for field in phase_fields}
        row["group_boundary_key"] = "0:0:{}".format(index)
        for field in required_diagnostics:
            row[field] = 0
        rows.append(row)
    csv_path = args.output_dir / "two_row_phase.csv"
    namespace["write_csv"](csv_path, rows, phase_fields)
    with csv_path.open(newline="") as handle:
        parsed = list(csv.DictReader(handle))
    if len(parsed) != 2 or list(parsed[0]) != phase_fields:
        raise SystemExit("two-row CSV roundtrip failed")

    json_path = args.output_dir / "two_row_result.json"
    json_payload = {
        "status": "PASS_M459R3_PREINPUT_TWO_ROW_MICRO",
        "rows": len(parsed),
        "phase_fields": phase_fields,
        "representative_per_B": [
            {"B": width, "cycles": 100 // width,
             "utilization": 1.0 - width / 100.0,
             "wasted_slots_per_block": width - 1}
            for width in (1, 2, 4, 8)],
        "patched_source_compiles": True,
        "csv_writer_roundtrip": True,
        "json_roundtrip": True,
        "seal_path_roundtrip": True,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n")
    json.loads(json_path.read_text())

    manifest = args.output_dir / "PREINPUT_SHA256SUMS"
    manifest.write_text("{}  two_row_phase.csv\n{}  two_row_result.json\n".format(
        sha256(csv_path), sha256(json_path)))
    seal = args.output_dir / "PREINPUT_SHA256SUMS.seal.sha256"
    seal.write_text("{}  PREINPUT_SHA256SUMS\n".format(sha256(manifest)))
    for line in manifest.read_text().splitlines():
        digest, target = line.split(maxsplit=1)
        if sha256(args.output_dir / target) != digest:
            raise SystemExit("micro inner seal mismatch")
    outer_digest, outer_target = seal.read_text().strip().split(maxsplit=1)
    if outer_target != "PREINPUT_SHA256SUMS" or sha256(manifest) != outer_digest:
        raise SystemExit("micro outer seal mismatch")
    print("PASS_M459R3_PREINPUT_TWO_ROW_MICRO fields={} rows=2 seal={}".format(
        len(phase_fields), outer_digest))


if __name__ == "__main__":
    main()
