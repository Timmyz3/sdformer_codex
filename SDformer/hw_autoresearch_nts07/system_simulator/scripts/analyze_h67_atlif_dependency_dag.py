#!/usr/bin/env python3
"""Audit real tensor ancestry at each H67 ATLIF boundary.

This is a dependency classifier, not a cycle simulator.  It walks backward from
logical ATLIF module-entry tensors through storage-overlapping functional ops
and leaf-module exits.  Event order disambiguates allocator reuse.  The output
is intended to decide which operator->ATLIF edges are eligible for a causal
partial-retirement simulator and which require residual/join handling.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


PRODUCER_TYPES = {"Conv2d", "Conv3d", "ConvTranspose2d", "Linear"}
PASS_MODULE_TYPES = {
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "Dropout", "DropPath",
    "Identity", "ReLU", "LeakyReLU",
}
PASS_FUNCTION_PREFIXES = (
    "aten.view.", "aten.reshape.", "aten._unsafe_view.", "aten.permute.",
    "aten.transpose.", "aten.flatten.", "aten.contiguous.", "aten.clone.",
    "aten.squeeze.", "aten.unsqueeze.", "aten.slice.", "aten.select.",
)
JOIN_FUNCTION_PREFIXES = (
    "aten.add.Tensor", "aten.sub.Tensor", "aten.cat.", "aten.stack.",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_events(path):
    events = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["event_index"]) != len(events):
                raise ValueError("non-contiguous event_index at line {}".format(line_number))
            events.append(row)
    return events


def dtype_bytes(dtype):
    table = {
        "torch.bool": 1, "torch.int8": 1, "torch.uint8": 1,
        "torch.float16": 2, "torch.bfloat16": 2, "torch.int16": 2,
        "torch.float32": 4, "torch.int32": 4,
        "torch.float64": 8, "torch.int64": 8,
    }
    return table.get(dtype, 1)


def element_span(ref):
    low = int(ref.get("storage_offset", 0))
    high = low
    for shape, stride in zip(ref.get("shape", []), ref.get("stride", [])):
        delta = max(0, int(shape) - 1) * int(stride)
        if delta >= 0:
            high += delta
        else:
            low += delta
    width = dtype_bytes(str(ref.get("dtype", "")))
    return low * width, (high + 1) * width


def same_storage(left, right):
    left_cdata = int(left.get("storage_cdata", 0))
    right_cdata = int(right.get("storage_cdata", 0))
    # data_ptr can be recycled by the CUDA allocator.  StorageImpl identity is
    # the hard allocation epoch; different cdata values never alias here.
    return bool(left_cdata and left_cdata == right_cdata)


def refs_overlap(left, right):
    if not same_storage(left, right):
        return False
    left_low, left_high = element_span(left)
    right_low, right_high = element_span(right)
    return max(left_low, right_low) < min(left_high, right_high)


def exact_tensor(left, right):
    return (
        int(left.get("python_id", -1)) == int(right.get("python_id", -2))
        and same_storage(left, right)
    )


def exact_view(left, right):
    return (
        same_storage(left, right)
        and int(left.get("storage_offset", 0)) == int(right.get("storage_offset", 0))
        and list(left.get("shape", [])) == list(right.get("shape", []))
        and list(left.get("stride", [])) == list(right.get("stride", []))
        and str(left.get("dtype", "")) == str(right.get("dtype", ""))
    )


def build_output_index(events):
    by_storage = defaultdict(list)
    for event in events:
        if event.get("kind") not in {
            "functional_op", "leaf_module_exit", "leaf_module", "persistent_tensor"
        }:
            continue
        for ref in event.get("outputs", []):
            cdata = int(ref.get("storage_cdata", 0))
            key = (
                int(event.get("sample_id", -1)), str(event.get("sequence_key", "")), cdata
            )
            if cdata:
                by_storage[key].append((int(event["event_index"]), event, ref))
    return by_storage


def find_latest_producer(ref, upper_event_index, output_index, sample_id, sequence_key):
    cdata = int(ref.get("storage_cdata", 0))
    key = (int(sample_id), str(sequence_key), cdata)
    candidates = []
    seen = set()
    if not cdata:
        return None
    consumer_version = int(ref.get("version", -1))
    for item in output_index.get(key, []):
        identity = (item[0], id(item[2]))
        if identity in seen or item[0] >= upper_event_index:
            continue
        seen.add(identity)
        producer_version = int(item[2].get("version", -1))
        if producer_version > consumer_version:
            continue
        version_equal = producer_version == consumer_version
        if exact_tensor(ref, item[2]):
            candidates.append((4 if version_equal else 2, item[0], item[1], item[2]))
        elif exact_view(ref, item[2]):
            candidates.append((3 if version_equal else 2, item[0], item[1], item[2]))
        elif refs_overlap(ref, item[2]):
            candidates.append((1, item[0], item[1], item[2]))
    if not candidates:
        return None
    # Tensor/view identity dominates a broad storage-overlap fallback.  Event
    # order then fences allocator reuse among candidates of equal strength.
    candidates.sort(key=lambda item: (item[0], item[1]))
    quality, _, event, output = candidates[-1]
    names = {
        4: "exact_tensor_version", 3: "exact_view_version",
        2: "version_drift", 1: "storage_overlap",
    }
    return event, output, names[quality]


def tensor_identity(ref):
    return {
        "storage_cdata": int(ref.get("storage_cdata", 0)),
        "storage_offset": int(ref.get("storage_offset", 0)),
        "shape": list(ref.get("shape", [])),
        "stride": list(ref.get("stride", [])),
        "dtype": str(ref.get("dtype", "")),
        "version": int(ref.get("version", -1)),
    }


def resolve_nearest_boundaries(ref, upper, output_index, sample_id, sequence_key, max_depth=48):
    frontier = [(ref, upper, 0)]
    visited = set()
    endpoints = []
    while frontier:
        item_ref, item_upper, depth = frontier.pop()
        key = (
            int(item_ref.get("storage_cdata", 0)), int(item_ref.get("storage_offset", 0)),
            tuple(item_ref.get("shape", [])), tuple(item_ref.get("stride", [])),
            int(item_ref.get("version", -1)), item_upper,
        )
        if key in visited:
            continue
        visited.add(key)
        if depth > max_depth:
            endpoints.append({"kind": "depth_limit", "tensor": tensor_identity(item_ref)})
            continue
        match = find_latest_producer(
            item_ref, item_upper, output_index, sample_id, sequence_key
        )
        if match is None:
            endpoints.append({"kind": "unknown", "tensor": tensor_identity(item_ref)})
            continue
        event, output_ref, quality = match
        event_kind = event.get("kind")
        module_type = str(event.get("module_type", ""))
        function_name = str(event.get("name", ""))
        is_join = event_kind == "functional_op" and function_name.startswith(
            JOIN_FUNCTION_PREFIXES
        )
        is_boundary = (
            (event_kind in {"leaf_module_exit", "leaf_module"}
             and (module_type in PRODUCER_TYPES or module_type == "ATLIFTernaryPSN"))
            or is_join
            or event_kind == "persistent_tensor"
        )
        if is_boundary:
            endpoints.append({
                "kind": (
                    "persistent" if event_kind == "persistent_tensor"
                    else ("join" if is_join else "module")
                ),
                "event_index": int(event["event_index"]),
                "name": function_name,
                "module_type": module_type or None,
                "module_call_index": event.get("module_call_index"),
                "match_quality": quality,
                "tensor": tensor_identity(output_ref),
            })
            continue
        passthrough = (
            (event_kind in {"leaf_module_exit", "leaf_module"}
             and module_type in PASS_MODULE_TYPES)
            or (event_kind == "functional_op"
                and function_name.startswith(PASS_FUNCTION_PREFIXES))
        )
        if not passthrough:
            endpoints.append({
                "kind": "transform",
                "event_index": int(event["event_index"]),
                "name": function_name,
                "module_type": module_type or None,
                "match_quality": quality,
                "tensor": tensor_identity(output_ref),
            })
            continue
        for input_ref in event.get("inputs", []):
            frontier.append((input_ref, int(event["event_index"]), depth + 1))
    return endpoints


def trace_atlif(enter_event, output_index, max_depth=96):
    frontier = [(ref, int(enter_event["event_index"]), 0, 0) for ref in enter_event.get("inputs", [])]
    visited = set()
    producers = set()
    upstream_atlif = set()
    persistent_inputs = set()
    functions = set()
    modules = set()
    match_quality = defaultdict(int)
    unmatched_refs = 0
    join_seen = False
    transformed_seen = False
    traversed_events = set()
    join_boundaries = set()
    join_operands = []
    sample_id = int(enter_event.get("sample_id", -1))
    sequence_key = str(enter_event.get("sequence_key", ""))
    while frontier:
        ref, upper, depth, join_depth = frontier.pop()
        key = (
            int(ref.get("storage_cdata", 0)), int(ref.get("storage_data_ptr", 0)),
            int(ref.get("storage_offset", 0)), tuple(ref.get("shape", [])),
            tuple(ref.get("stride", [])), str(ref.get("dtype", "")), upper,
        )
        if key in visited:
            continue
        visited.add(key)
        if depth > max_depth:
            unmatched_refs += 1
            continue
        match = find_latest_producer(
            ref, upper, output_index, sample_id=sample_id, sequence_key=sequence_key
        )
        if match is None:
            unmatched_refs += 1
            continue
        event, _, quality = match
        event_index = int(event["event_index"])
        match_quality[quality] += 1
        traversed_events.add(event_index)
        kind = event.get("kind")
        if kind in {"leaf_module_exit", "leaf_module"}:
            name = str(event.get("name", ""))
            module_type = str(event.get("module_type", ""))
            modules.add(name)
            if module_type in PRODUCER_TYPES:
                producers.add(name)
                continue
            if module_type == "ATLIFTernaryPSN":
                upstream_atlif.add(name)
                continue
            if module_type not in PASS_MODULE_TYPES:
                transformed_seen = True
            inputs = event.get("inputs", [])
        elif kind == "persistent_tensor":
            persistent_inputs.add(str(event.get("name", "")))
            continue
        elif kind == "functional_op":
            name = str(event.get("name", ""))
            functions.add(name)
            if name.startswith(JOIN_FUNCTION_PREFIXES):
                join_seen = True
                join_boundaries.add(event_index)
                join_operands.append({
                    "join_event_index": event_index,
                    "join_name": name,
                    "output_tensor": tensor_identity(match[1]),
                    "operands": [
                        {
                            "operand_index": index,
                            "input_tensor": tensor_identity(input_ref),
                            "nearest_boundaries": resolve_nearest_boundaries(
                                input_ref, event_index, output_index,
                                sample_id, sequence_key,
                            ),
                        }
                        for index, input_ref in enumerate(event.get("inputs", []))
                    ],
                })
                # A materialized/resident join is itself a hardware-ready
                # boundary.  Expand its immediate operands once, but do not
                # flatten an entire residual network through older joins.
                if join_depth >= 1:
                    continue
                join_depth += 1
            elif not name.startswith(PASS_FUNCTION_PREFIXES):
                transformed_seen = True
            inputs = event.get("inputs", [])
        else:
            unmatched_refs += 1
            continue
        for input_ref in inputs:
            frontier.append((input_ref, event_index, depth + 1, join_depth))

    if join_seen:
        category = "join"
    elif not producers:
        category = "unknown"
    elif transformed_seen or len(producers) != 1:
        category = "transformed"
    else:
        category = "direct"
    return {
        "category": category,
        "producers": sorted(producers),
        "upstream_atlif": sorted(upstream_atlif),
        "persistent_inputs": sorted(persistent_inputs),
        "functional_ops": sorted(functions),
        "modules": sorted(modules),
        "join_seen": join_seen,
        "transformed_seen": transformed_seen,
        "unmatched_refs": unmatched_refs,
        "match_quality": dict(sorted(match_quality.items())),
        "traversed_events": len(traversed_events),
        "join_boundary_events": sorted(join_boundaries),
        "join_operands": join_operands,
    }


def load_real_tile_names(path):
    if path is None:
        return set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {row["name"] for row in csv.DictReader(handle)}


def load_ordered_atlif(path):
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("kind") == "atlif"]


def atlif_service_cycles(elements, temporal_steps, calls, lanes=16, frame_steps=10):
    if temporal_steps <= 0 or frame_steps % temporal_steps:
        raise ValueError("temporal_steps must be a positive divisor of frame_steps")
    if calls <= 0:
        raise ValueError("calls must be positive")
    elements_per_call = int(math.ceil(float(elements) / calls))
    neurons = int(math.ceil(float(elements_per_call) / temporal_steps))
    packed_contexts = frame_steps // temporal_steps
    return int(math.ceil(float(neurons) / (lanes * packed_contexts))) * temporal_steps * calls


def load_ledger(path, lanes):
    if path is None:
        return {}
    result = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dead = str(row.get("deployment_dead_result", "")).strip().lower() == "true"
            result[row["name"]] = {
                "live": not dead,
                "temporal_steps": int(row["temporal_steps"]),
                "elements_per_frame": int(row["elements_per_frame"]),
                "calls_per_frame": int(row["calls_per_frame"]),
                "service_cycles": atlif_service_cycles(
                    int(row["elements_per_frame"]), int(row["temporal_steps"]),
                    int(row["calls_per_frame"]), lanes=lanes,
                ),
            }
    return result


def analyze(events, ordered_atlif, real_tile_names, ledger, lanes=16):
    output_index = build_output_index(events)
    enters = [
        event for event in events
        if event.get("kind") == "leaf_module_enter"
        and event.get("module_type") == "ATLIFTernaryPSN"
    ]
    rows = []
    for enter in enters:
        traced = trace_atlif(enter, output_index)
        producers = traced["producers"]
        if traced["category"] == "direct":
            traced["category"] = (
                "direct_m4" if producers and producers[0] in real_tile_names
                else "direct_non_m4"
            )
        elif traced["category"] == "join":
            traced["category"] = (
                "join_with_m4" if any(name in real_tile_names for name in producers)
                else "join_non_m4"
            )
        name = str(enter["name"])
        ledger_row = ledger.get(name, {})
        traced.update({
            "name": name,
            "sample_id": int(enter.get("sample_id", -1)),
            "sequence_key": str(enter.get("sequence_key", "")),
            "module_call_index": int(enter.get("module_call_index", 0)),
            "enter_event_index": int(enter["event_index"]),
            "input_shapes": [ref.get("shape", []) for ref in enter.get("inputs", [])],
            "live": ledger_row.get("live"),
            "temporal_steps": ledger_row.get("temporal_steps"),
            "service_cycles_l{}".format(lanes): ledger_row.get("service_cycles", 0),
        })
        uncertain_qualities = sum(
            traced["match_quality"].get(name, 0)
            for name in ("storage_overlap", "version_drift")
        )
        traced["uncertain_matches"] = uncertain_qualities
        traced["admitted_for_overlap"] = (
            traced["category"] == "direct_m4"
            and ledger_row.get("live") is True
            and traced["unmatched_refs"] == 0
            and uncertain_qualities == 0
        )
        rows.append(traced)

    ordered_occurrence = defaultdict(int)
    ordered_keys = set()
    for row in ordered_atlif:
        prefix = (
            int(row.get("sample_id", -1)), str(row.get("sequence_key", "")), row["name"]
        )
        occurrence = ordered_occurrence[prefix]
        ordered_occurrence[prefix] += 1
        ordered_keys.add(prefix + (occurrence,))
    traced_keys = {
        (
            row["sample_id"], row["sequence_key"], row["name"], row["module_call_index"],
        )
        for row in rows
    }
    expected_names = {row["name"] for row in ordered_atlif}
    traced_names = {row["name"] for row in rows}
    summary = defaultdict(lambda: {"calls": 0, "live_calls": 0, "service_cycles": 0})
    for row in rows:
        item = summary[row["category"]]
        item["calls"] += 1
        if row["live"] is True:
            item["live_calls"] += 1
            item["service_cycles"] += int(row["service_cycles_l{}".format(lanes)])
    unknown_live = [row for row in rows if row["live"] is True and row["category"] == "unknown"]
    live_unmatched = [row for row in rows if row["live"] is True and row["unmatched_refs"]]
    uncertain = [row for row in rows if row["live"] is True and row["uncertain_matches"]]
    all_unmatched = [row for row in rows if row["unmatched_refs"]]
    all_uncertain = [row for row in rows if row["uncertain_matches"]]
    all_unknown = [row for row in rows if row["category"] == "unknown"]
    admitted = [row for row in rows if row["admitted_for_overlap"]]
    return rows, {
        "logical_atlif_enters": len(enters),
        "ordered_atlif_calls": len(ordered_atlif),
        "ordered_names_missing_from_dependency": sorted(expected_names - traced_names),
        "dependency_names_missing_from_ordered": sorted(traced_names - expected_names),
        "ordered_call_keys_missing_from_dependency": sorted(ordered_keys - traced_keys),
        "dependency_call_keys_missing_from_ordered": sorted(traced_keys - ordered_keys),
        "ledger_live_names_missing_from_dependency": sorted(
            name for name, row in ledger.items() if row["live"] and name not in traced_names
        ),
        "categories": dict(sorted(summary.items())),
        "live_service_cycles_l{}".format(lanes): sum(
            row["service_cycles_l{}".format(lanes)] for row in rows if row["live"] is True
        ),
        "unknown_live_calls": len(unknown_live),
        "unknown_live_service_cycles": sum(
            row["service_cycles_l{}".format(lanes)] for row in unknown_live
        ),
        "live_calls_with_unmatched_refs": len(live_unmatched),
        "live_calls_with_uncertain_matches": len(uncertain),
        "all_calls_with_unmatched_refs": len(all_unmatched),
        "all_calls_with_uncertain_matches": len(all_uncertain),
        "all_unknown_calls": len(all_unknown),
        "admitted_direct_m4_calls": len(admitted),
        "admitted_direct_m4_service_cycles": sum(
            row["service_cycles_l{}".format(lanes)] for row in admitted
        ),
    }


def write_csv(path, rows, lanes):
    fields = [
        "sample_id", "sequence_key", "name", "module_call_index",
        "enter_event_index", "category", "live", "admitted_for_overlap",
        "uncertain_matches",
        "temporal_steps", "service_cycles_l{}".format(lanes), "producers",
        "upstream_atlif", "join_seen", "transformed_seen", "unmatched_refs",
        "traversed_events", "match_quality", "functional_ops", "input_shapes",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for field in ("producers", "upstream_atlif", "functional_ops", "input_shapes", "match_quality"):
                row[field] = json.dumps(row[field], separators=(",", ":"), sort_keys=True)
            writer.writerow({field: row.get(field) for field in fields})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--dependency-manifest", type=Path, required=True)
    parser.add_argument("--ordered-trace", type=Path)
    parser.add_argument("--real-tiles", type=Path)
    parser.add_argument("--atlif-ledger", type=Path)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.lanes <= 0:
        raise ValueError("lanes must be positive")
    dependency_manifest = json.loads(args.dependency_manifest.read_text(encoding="utf-8"))
    if dependency_manifest.get("schema") != "h67_tensor_dependency_trace_v2":
        raise ValueError("dependency manifest schema mismatch")
    events_sha256 = sha256(args.events)
    if dependency_manifest.get("dependency_events_sha256") != events_sha256:
        raise ValueError("dependency manifest/event hash mismatch")
    events = load_events(args.events)
    ordered = load_ordered_atlif(args.ordered_trace)
    real_names = load_real_tile_names(args.real_tiles)
    ledger = load_ledger(args.atlif_ledger, args.lanes)
    rows, summary = analyze(events, ordered, real_names, ledger, args.lanes)
    completeness_fail = bool(
        summary["ordered_call_keys_missing_from_dependency"]
        or summary["dependency_call_keys_missing_from_ordered"]
        or summary["ledger_live_names_missing_from_dependency"]
    )
    uncertainty_fail = bool(
        summary["unknown_live_calls"]
        or summary["live_calls_with_unmatched_refs"]
        or summary["live_calls_with_uncertain_matches"]
    )
    if completeness_fail:
        status = "FAIL_INCOMPLETE_ATLIF_CAPTURE"
    elif uncertainty_fail:
        status = "PARTIAL_DEPENDENCY_CENSUS_OVERLAP_NOT_ADMITTED"
    else:
        status = "PASS_CAUSAL_DEPENDENCY_CLASSIFICATION"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "atlif_dependency_edges.csv"
    write_csv(csv_path, rows, args.lanes)
    evidence = {
        "schema": "h67_atlif_dependency_audit_v2",
        "status": status,
        "summary": summary,
        "rows": rows,
        "identities": {
            "events": {"path": str(args.events), "sha256": events_sha256},
            "dependency_manifest": {
                "path": str(args.dependency_manifest),
                "sha256": sha256(args.dependency_manifest),
                "schema": dependency_manifest["schema"],
                "samples": dependency_manifest["samples"],
                "artifact_identity": dependency_manifest["run_context"]["artifact_identity"],
                "eval_protocol": dependency_manifest["run_context"]["eval_protocol"],
                "writer_source_sha256": dependency_manifest["run_context"]["source_sha256"]["dependency_writer"],
            },
            "ordered_trace": None if args.ordered_trace is None else {"path": str(args.ordered_trace), "sha256": sha256(args.ordered_trace)},
            "real_tiles": None if args.real_tiles is None else {"path": str(args.real_tiles), "sha256": sha256(args.real_tiles)},
            "atlif_ledger": None if args.atlif_ledger is None else {"path": str(args.atlif_ledger), "sha256": sha256(args.atlif_ledger)},
            "edge_csv_sha256": sha256(csv_path),
            "analyzer_source_sha256": sha256(Path(__file__).resolve()),
            "analyzer_test_sha256": sha256(
                Path(__file__).resolve().parent.parent / "tests" / "test_h67_atlif_dependency_dag.py"
            ),
            "python_version": sys.version,
            "argv": list(sys.argv),
        },
        "claim_boundary": (
            "Storage-overlap/event-order ancestry and L{} ATLIF service classification only; "
            "unknown/unmatched/version-drift/storage-overlap rows are not admitted for timing; "
            "not tile-ready timing, overlap cycles, functional equivalence, SRAM-port timing, "
            "speedup, energy, or PPA. Storage-overlap-only matches remain auditable uncertainty."
        ).format(args.lanes),
    }
    json_path = args.output_dir / "atlif_dependency_audit.json"
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("{} logical_atlif={} ordered_atlif={} live_cycles={}".format(
        evidence["status"], summary["logical_atlif_enters"], summary["ordered_atlif_calls"],
        summary["live_service_cycles_l{}".format(args.lanes)],
    ))
    return 1 if evidence["status"].startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
