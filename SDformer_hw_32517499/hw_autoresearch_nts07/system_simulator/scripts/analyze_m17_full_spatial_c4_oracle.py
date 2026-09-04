#!/usr/bin/env python3
"""Reconcile M17 exact C4 costs against the same-sample source-work ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def temporal_ints(value: Any, steps: int, label: str, *, minimum: int = 0) -> list[int]:
    if not isinstance(value, list) or len(value) != steps:
        raise ValueError(f"{label} must contain exactly {steps} temporal entries")
    return [exact_int(item, f"{label}[{index}]", minimum=minimum) for index, item in enumerate(value)]


def validate_prototype(prototype: dict[str, Any], expected_id: int) -> None:
    if exact_int(prototype.get("prototype_id"), "prototype_id") != expected_id:
        raise ValueError("M17 prototype IDs are not dense")
    stored_hash = prototype.get("cost_source_sha256")
    unhashed = {key: value for key, value in prototype.items() if key != "cost_source_sha256"}
    if not isinstance(stored_hash, str) or stored_hash != canonical_sha256(unhashed):
        raise ValueError("M17 prototype cost-source identity mismatch")
    contexts = exact_int(prototype.get("contexts"), "prototype contexts", minimum=1)
    if contexts > 4:
        raise ValueError("M17 prototype contexts exceed frozen C4 geometry")
    chunks = exact_int(prototype.get("chunks"), "prototype chunks", minimum=1)
    fanout = exact_int(prototype.get("fanout"), "prototype fanout", minimum=1)
    lane_tiles = exact_int(prototype.get("lane_tiles"), "prototype lane tiles", minimum=1)
    if lane_tiles != math.ceil(fanout / 96):
        raise ValueError("M17 prototype fanout/lane-tile contract mismatch")
    descriptors = prototype.get("descriptor_cycles_by_t")
    local = prototype.get("local_lane_compute_cycles_by_t")
    hybrid = prototype.get("hybrid_lane_compute_cycles_by_t")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("M17 prototype temporal arrays are empty")
    steps = len(descriptors)
    descriptors = temporal_ints(descriptors, steps, "prototype descriptor cycles", minimum=1)
    local = temporal_ints(local, steps, "prototype Local cycles", minimum=1)
    hybrid = temporal_ints(hybrid, steps, "prototype Hybrid cycles", minimum=1)
    if any(value != contexts * chunks for value in descriptors):
        raise ValueError("M17 prototype descriptor/context/chunk contract mismatch")
    if any(value < 2 * chunks for value in local + hybrid):
        raise ValueError("M17 prototype omits frozen PREP/DRAIN chunk cycles")


def load_ledger(path: Path) -> dict[tuple[int, str, str, int, int], dict[str, str]]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "status", "sample_id", "sequence_key", "name", "operator_call_index",
            "temporal_step", "local_work", "selected_work", "selector_rows",
            "motion_selected_rows",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("same-sample source ledger schema is incomplete")
        for row in reader:
            if row["status"] != "PASS_EXACT_SOURCE_WORK":
                continue
            if not row["sequence_key"] or not row["name"]:
                raise ValueError("same-sample source-ledger identity is incomplete")
            for field in (
                "sample_id", "operator_call_index", "temporal_step", "local_work",
                "selected_work", "selector_rows", "motion_selected_rows",
            ):
                try:
                    parsed = int(row[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid integer ledger field: {field}") from exc
                if parsed < 0:
                    raise ValueError(f"negative ledger field: {field}")
            key = (
                int(row["sample_id"]), row["sequence_key"], row["name"],
                int(row["operator_call_index"]), int(row["temporal_step"]),
            )
            if key in result:
                raise ValueError("duplicate same-sample source-ledger identity")
            result[key] = row
    return result


def analyze(oracle_dir: Path, ledger_path: Path) -> dict[str, Any]:
    manifest_path = oracle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "h67_full_spatial_adjacent_c4_oracle_v2"
        or manifest.get("status")
        != "PASS_EXACT_FULL_SPATIAL_C4_SUFFICIENT_STATISTICS_NOT_SYSTEM_SPEEDUP"
    ):
        raise ValueError("M17 full-spatial oracle is not admitted")
    architecture = manifest.get("architecture", {})
    if architecture != {
        "tile_bits": 256, "contexts": 4, "issue_width": 16, "reduce_slots": 4,
        "output_lanes": 96, "scheduler": "deterministic_bank_first_context_first",
    }:
        raise ValueError("M17 manifest architecture is not the frozen C4 geometry")
    run_context = manifest.get("run_context")
    if not isinstance(run_context, dict):
        raise ValueError("M17 run context is absent")
    artifact_identity = run_context.get("artifact_identity", {})
    for key in ("config_sha256", "checkpoint_sha256"):
        if not isinstance(artifact_identity.get(key), str) or len(artifact_identity[key]) != 64:
            raise ValueError("M17 config/checkpoint identity is incomplete")
    if exact_int(artifact_identity.get("checkpoint_size"), "checkpoint size", minimum=1) <= 0:
        raise ValueError("M17 checkpoint identity is incomplete")
    load_audit = run_context.get("checkpoint_load_audit", {})
    if any(exact_int(load_audit.get(key), f"checkpoint {key}") != 0 for key in (
        "missing_count", "unexpected_count", "overlay_missing_count", "overlay_unexpected_count",
    )):
        raise ValueError("M17 checkpoint load was not exact")
    if not isinstance(run_context.get("eval_protocol"), dict):
        raise ValueError("M17 evaluation protocol identity is absent")
    for key in (
        "dependency_audit_sha256", "dependency_manifest_sha256", "dependency_events_sha256",
    ):
        if not isinstance(run_context.get(key), str) or len(run_context[key]) != 64:
            raise ValueError(f"M17 run context is missing {key}")
    repo = Path(__file__).resolve().parents[3]
    expected_sources = {
        "profiler": repo / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
        "full_spatial_c4_writer": repo / "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_full_spatial_c4_oracle.py",
    }
    recorded_sources = run_context.get("source_sha256", {})
    if any(recorded_sources.get(label) != sha256(path) for label, path in expected_sources.items()):
        raise ValueError("M17 profiler/writer source identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"prototypes.json", "ordered_stream.npz"}:
        raise ValueError("M17 manifest file population is incomplete or unexpected")
    for name, entry in files.items():
        if Path(name).name != name or not isinstance(entry, dict):
            raise ValueError("unsafe M17 artifact name")
        path = oracle_dir / name
        if (
            not path.is_file() or sha256(path) != entry.get("sha256")
            or path.stat().st_size != exact_int(entry.get("bytes"), f"{name} bytes", minimum=1)
        ):
            raise ValueError(f"M17 artifact identity mismatch: {name}")
    prototypes = json.loads((oracle_dir / "prototypes.json").read_text(encoding="utf-8"))
    if not isinstance(prototypes, list) or not prototypes:
        raise ValueError("M17 prototype population is empty")
    for expected_id, prototype in enumerate(prototypes):
        if not isinstance(prototype, dict):
            raise ValueError("M17 prototypes must be objects")
        validate_prototype(prototype, expected_id)
    if exact_int(manifest.get("prototypes"), "manifest prototypes", minimum=1) != len(prototypes):
        raise ValueError("M17 manifest prototype population mismatch")
    with np.load(oracle_dir / "ordered_stream.npz", allow_pickle=False) as stream:
        if set(stream.files) != {"operator_index", "population_cluster_id", "prototype_id"}:
            raise ValueError("M17 ordered stream arrays are incomplete or unexpected")
        operator_index = stream["operator_index"].copy()
        population = stream["population_cluster_id"].copy()
        prototype_id = stream["prototype_id"].copy()
    if not (operator_index.ndim == population.ndim == prototype_id.ndim == 1):
        raise ValueError("M17 ordered stream must use one-dimensional arrays")
    if not all(np.issubdtype(value.dtype, np.integer) for value in (operator_index, population, prototype_id)):
        raise ValueError("M17 ordered stream identities must use integer arrays")
    ordered_clusters = exact_int(manifest.get("ordered_clusters"), "ordered clusters", minimum=1)
    if not (len(operator_index) == len(population) == len(prototype_id) == ordered_clusters):
        raise ValueError("M17 ordered stream cardinality mismatch")
    if np.any(operator_index < 0) or np.any(population < 0) or np.any(prototype_id < 0):
        raise ValueError("M17 ordered stream contains negative identities")
    if int(prototype_id.max()) >= len(prototypes):
        raise ValueError("M17 ordered stream references an invalid prototype")
    operators = manifest.get("operators")
    allowed_calls = manifest.get("allowed_calls")
    if not isinstance(operators, list) or len(operators) != 13:
        raise ValueError("M17 requires exactly 13 operator calls")
    if not isinstance(allowed_calls, list) or len(allowed_calls) != len(operators):
        raise ValueError("M17 call-contract population mismatch")
    expected_call_keys = {
        (
            exact_int(item.get("sample_id"), "allowed sample ID"), str(item.get("sample_key", "")),
            str(item.get("sequence_key", "")), str(item.get("name", "")),
            exact_int(item.get("operator_call_index"), "allowed call index"),
        )
        for item in allowed_calls
    }
    if len(expected_call_keys) != len(allowed_calls) or any(not all(key[1:4]) for key in expected_call_keys):
        raise ValueError("M17 allowed name+call identities are duplicate or incomplete")
    if sum(exact_int(item.get("population_clusters"), "operator population", minimum=1) for item in operators) != ordered_clusters:
        raise ValueError("M17 manifest operator population does not cover the stream")
    if int(operator_index.max()) >= len(operators):
        raise ValueError("M17 ordered stream references an invalid operator")
    ledger = load_ledger(ledger_path)
    rows = []
    local_mismatches = 0
    hybrid_mismatches = 0
    row_mismatches = 0
    motion_mismatches = 0
    wall_mismatches = 0
    observed_call_keys = set()
    for expected_operator_index, operator in enumerate(operators):
        if exact_int(operator.get("operator_index"), "operator index") != expected_operator_index:
            raise ValueError("M17 operator IDs are not dense")
        operator_key = (
            exact_int(operator.get("sample_id"), "operator sample ID"),
            str(operator.get("sample_key", "")), str(operator.get("sequence_key", "")),
            str(operator.get("name", "")),
            exact_int(operator.get("operator_call_index"), "operator call index"),
        )
        if operator_key not in expected_call_keys or operator_key in observed_call_keys:
            raise ValueError("M17 observed name+call identity violates its contract")
        observed_call_keys.add(operator_key)
        temporal_steps = exact_int(operator.get("temporal_steps"), "temporal steps", minimum=1)
        row_count = exact_int(operator.get("row_count"), "row count", minimum=1)
        population_clusters = exact_int(
            operator.get("population_clusters"), "population clusters", minimum=1,
        )
        source_width = exact_int(operator.get("source_width"), "source width", minimum=1)
        chunks = exact_int(operator.get("chunks"), "chunks", minimum=1)
        fanout = exact_int(operator.get("fanout"), "fanout", minimum=1)
        lane_tiles = exact_int(operator.get("lane_tiles"), "lane tiles", minimum=1)
        if chunks != math.ceil(source_width / 256) or lane_tiles != math.ceil(fanout / 96):
            raise ValueError("M17 operator source/chunk or fanout/lane-tile contract mismatch")
        positions = np.flatnonzero(operator_index == expected_operator_index)
        if len(positions) != population_clusters:
            raise ValueError("M17 per-operator stream cardinality mismatch")
        if not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions))):
            raise ValueError("M17 per-operator stream must be contiguous")
        expected_population = np.arange(population_clusters, dtype=population.dtype)
        if not np.array_equal(population[positions], expected_population):
            raise ValueError("M17 population cluster identities are not dense and ordered")
        referenced = [prototypes[int(prototype_id[position])] for position in positions]
        if sum(int(item["contexts"]) for item in referenced) != row_count:
            raise ValueError("M17 prototype contexts do not cover the operator row population")
        for prototype in referenced:
            if (
                int(prototype["chunks"]) != chunks or int(prototype["fanout"]) != fanout
                or int(prototype["lane_tiles"]) != lane_tiles
                or len(prototype["descriptor_cycles_by_t"]) != temporal_steps
            ):
                raise ValueError("M17 referenced prototype/operator geometry mismatch")
        motion_observed = temporal_ints(
            operator.get("motion_selected_rows_by_t"), temporal_steps,
            "operator motion-selected rows",
        )
        if any(value > row_count for value in motion_observed):
            raise ValueError("M17 motion-selected rows exceed the row population")
        positive_observed = temporal_ints(
            operator.get("positive_selected_product_terms_by_t"), temporal_steps,
            "operator positive product terms",
        )
        negative_observed = temporal_ints(
            operator.get("negative_selected_product_terms_by_t"), temporal_steps,
            "operator negative product terms",
        )
        gate_histograms = operator.get("motion_gate_mask_histogram_by_t")
        if not isinstance(gate_histograms, list) or len(gate_histograms) != temporal_steps:
            raise ValueError("M17 motion gate histograms are temporally truncated")
        for histogram in gate_histograms:
            if not isinstance(histogram, dict) or any(
                not str(key).isdigit() or int(key) > 15 or exact_int(value, "gate count") < 0
                for key, value in histogram.items()
            ) or sum(histogram.values()) != population_clusters:
                raise ValueError("M17 motion gate histogram population mismatch")
        local_exact = []
        hybrid_exact = []
        selector_rows = []
        motion_rows = []
        for timestep in range(temporal_steps):
            key = (
                operator["sample_id"], operator["sequence_key"], operator["name"],
                operator["operator_call_index"], timestep,
            )
            if key not in ledger:
                raise ValueError("same-sample source ledger is missing an M17 operator step")
            ledger_row = ledger[key]
            local_exact.append(int(ledger_row["local_work"]))
            hybrid_exact.append(int(ledger_row["selected_work"]))
            selector_rows.append(int(ledger_row["selector_rows"]))
            motion_rows.append(int(ledger_row["motion_selected_rows"]))
        lines = operator.get("lines")
        if not isinstance(lines, dict) or set(lines) != {"local", "hybrid"}:
            raise ValueError("M17 operator line population mismatch")
        local_observed = temporal_ints(
            lines["local"].get("selected_product_terms_by_t"), temporal_steps,
            "Local product terms",
        )
        hybrid_observed = temporal_ints(
            lines["hybrid"].get("selected_product_terms_by_t"), temporal_steps,
            "Hybrid product terms",
        )
        local_mismatches += sum(local_observed[index] != local_exact[index] for index in range(temporal_steps))
        hybrid_mismatches += sum(hybrid_observed[index] != hybrid_exact[index] for index in range(temporal_steps))
        row_mismatches += sum(value != row_count for value in selector_rows)
        motion_mismatches += sum(motion_observed[index] != motion_rows[index] for index in range(temporal_steps))
        if any(positive_observed[index] + negative_observed[index] > hybrid_observed[index] for index in range(temporal_steps)):
            raise ValueError("M17 selected transition terms exceed Hybrid terms")
        reconstructed = {}
        for line, field in (
            ("local", "local_lane_compute_cycles_by_t"),
            ("hybrid", "hybrid_lane_compute_cycles_by_t"),
        ):
            descriptor = 0
            compute = 0
            output = 0
            compact_issue = 0
            reconstructed_histograms = [dict() for _ in range(temporal_steps)]
            for stream_position in positions:
                prototype = prototypes[int(prototype_id[stream_position])]
                descriptor += sum(prototype["descriptor_cycles_by_t"])
                compute += sum(prototype[field]) * lane_tiles
                output += (
                    int(prototype["contexts"]) * temporal_steps * lane_tiles
                )
                for timestep, value in enumerate(prototype[field]):
                    compact_issue += (int(value) - 2 * chunks) * lane_tiles
                    histogram = reconstructed_histograms[timestep]
                    histogram[str(int(value))] = histogram.get(str(int(value)), 0) + 1
            reconstructed[line] = descriptor + compute + output
            line_payload = lines[line]
            observed_histograms = line_payload.get("lane_compute_cycle_histogram_by_t")
            if not isinstance(observed_histograms, list) or len(observed_histograms) != temporal_steps:
                raise ValueError(f"M17 {line} lane-cycle histograms are temporally truncated")
            for histogram in observed_histograms:
                if not isinstance(histogram, dict) or any(
                    not str(key).isdigit() or exact_int(value, f"{line} histogram count", minimum=1) < 1
                    for key, value in histogram.items()
                ) or sum(histogram.values()) != population_clusters:
                    raise ValueError(f"M17 {line} lane-cycle histogram population mismatch")
            if observed_histograms != reconstructed_histograms:
                raise ValueError("M17 lane-cycle histogram/prototype mismatch")
            expected_descriptor = temporal_steps * row_count * chunks
            expected_control = temporal_steps * population_clusters * 2 * chunks * lane_tiles
            expected_output = temporal_steps * row_count * lane_tiles
            expected_line_fields = {
                "selected_product_terms": sum(local_observed if line == "local" else hybrid_observed),
                "descriptor_load_cycles": expected_descriptor,
                "compact_issue_cycles": compact_issue,
                "chunk_control_cycles": expected_control,
                "output_cycles": expected_output,
                "m4_wall_cycles": reconstructed[line],
            }
            for key, expected in expected_line_fields.items():
                if exact_int(line_payload.get(key), f"{line} {key}") != expected:
                    raise ValueError(f"M17 {line} aggregate {key} mismatch")
            if reconstructed[line] != int(line_payload["m4_wall_cycles"]):
                wall_mismatches += 1
        if sum(local_observed) % fanout or sum(hybrid_observed) % fanout:
            raise ValueError("M17 product terms are not divisible by operator fanout")
        rows.append({
            "name": operator["name"], "operator_call_index": operator["operator_call_index"],
            "row_count": row_count,
            "population_clusters": population_clusters,
            "local_product_terms": sum(local_observed),
            "hybrid_product_terms": sum(hybrid_observed),
            "local_m4_wall_cycles": reconstructed["local"],
            "hybrid_m4_wall_cycles": reconstructed["hybrid"],
            "local_p1_sparse_wall_cycles": (
                int(lines["local"]["descriptor_load_cycles"])
                + int(lines["local"]["chunk_control_cycles"])
                + int(lines["local"]["output_cycles"])
                + sum(local_observed) // fanout * lane_tiles
            ),
            "hybrid_p1_sparse_wall_cycles": (
                int(lines["hybrid"]["descriptor_load_cycles"])
                + int(lines["hybrid"]["chunk_control_cycles"])
                + int(lines["hybrid"]["output_cycles"])
                + sum(hybrid_observed) // fanout * lane_tiles
            ),
            "same_width_dense_wall_cycles": (
                int(lines["local"]["descriptor_load_cycles"])
                + int(lines["local"]["chunk_control_cycles"])
                + int(lines["local"]["output_cycles"])
                + temporal_steps * row_count
                * sum(
                    math.ceil(
                        min(256, source_width - 256 * chunk) / 16
                    )
                    for chunk in range(chunks)
                ) * lane_tiles
            ),
        })
    if observed_call_keys != expected_call_keys:
        raise ValueError("M17 call contract was not completely observed")
    mismatches = {
        "local_product_term_steps": local_mismatches,
        "hybrid_product_term_steps": hybrid_mismatches,
        "selector_row_steps": row_mismatches,
        "motion_selector_steps": motion_mismatches,
        "prototype_wall_lines": wall_mismatches,
    }
    if any(mismatches.values()):
        raise ValueError("M17 exact reconciliation mismatch: " + repr(mismatches))
    totals = {
        "local_product_terms": sum(row["local_product_terms"] for row in rows),
        "hybrid_product_terms": sum(row["hybrid_product_terms"] for row in rows),
        "local_m4_wall_cycles": sum(row["local_m4_wall_cycles"] for row in rows),
        "hybrid_m4_wall_cycles": sum(row["hybrid_m4_wall_cycles"] for row in rows),
        "local_p1_sparse_wall_cycles": sum(row["local_p1_sparse_wall_cycles"] for row in rows),
        "hybrid_p1_sparse_wall_cycles": sum(row["hybrid_p1_sparse_wall_cycles"] for row in rows),
        "same_width_dense_wall_cycles": sum(row["same_width_dense_wall_cycles"] for row in rows),
    }
    totals.update({
        "local_speedup_vs_p1_sparse_source_kernel": (
            totals["local_p1_sparse_wall_cycles"] / totals["local_m4_wall_cycles"]
        ),
        "hybrid_speedup_vs_p1_sparse_source_kernel": (
            totals["hybrid_p1_sparse_wall_cycles"] / totals["hybrid_m4_wall_cycles"]
        ),
        "local_speedup_vs_same_width_dense_source_kernel": (
            totals["same_width_dense_wall_cycles"] / totals["local_m4_wall_cycles"]
        ),
        "hybrid_speedup_vs_same_width_dense_source_kernel": (
            totals["same_width_dense_wall_cycles"] / totals["hybrid_m4_wall_cycles"]
        ),
        "hybrid_wall_reduction_vs_local": (
            1.0 - totals["hybrid_m4_wall_cycles"] / totals["local_m4_wall_cycles"]
        ),
    })
    return {
        "schema": "m17_full_spatial_c4_reconciliation_v2",
        "status": "PASS_SAME_SAMPLE_EXACT_C4_SOURCE_AND_ORDERED_WALL_RECONCILIATION",
        "summary": {
            "operators": len(rows), "ordered_clusters": len(operator_index),
            "prototypes": len(prototypes), "mismatches": mismatches,
            **totals,
        },
        "rows": rows,
        "source_kernel_baselines": {
            "p1_sparse": (
                "same 96 output lanes; one selected source per issue cycle per lane tile, "
                "plus identical descriptor, PREP/DRAIN, and output cycles"
            ),
            "same_width_dense": (
                "same 16 source banks and 96 output lanes; every valid source position is issued, "
                "plus identical descriptor, PREP/DRAIN, and output cycles"
            ),
        },
        "identities": {
            "oracle_manifest_sha256": sha256(manifest_path),
            "same_sample_source_ledger_sha256": sha256(ledger_path),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "Exact same-sample direct-M4 source-work and ordered prototype wall-cycle "
            "reconciliation. Not ATLIF overlap, physical memory timing, VCS equivalence, "
            "system speedup, energy, or PPA."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--same-sample-source-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.oracle_dir.resolve(), args.same_sample_source_ledger.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "PASS_M17_FULL_SPATIAL_C4 operators={} clusters={} prototypes={}".format(
            result["summary"]["operators"], result["summary"]["ordered_clusters"],
            result["summary"]["prototypes"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
