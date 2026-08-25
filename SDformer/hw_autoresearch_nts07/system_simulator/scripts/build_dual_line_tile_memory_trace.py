#!/usr/bin/env python3
"""Build a fail-closed compressed address/timed trace from real tile descriptors.

This is a sampled tile memory schedule, not a full-network latency result.  It
compares Local-only with Local+Motion while charging Motion for previous bitmap
and Acc32 state.  Weight source addresses remain bitmap-indexed and are linked
to the packed NPZ by record id so a later DRAMsim adapter can expand them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def integer(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing {key} in record {row.get('record_id')}")
    return int(value)


def truth(value: str) -> bool:
    return value.lower() == "true"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AddressArena:
    def __init__(self, base: int = 0x1000_0000, alignment: int = 64) -> None:
        self.base = base
        self.cursor = base
        self.alignment = alignment
        self.objects: dict[str, dict[str, int]] = {}

    def allocate(self, name: str, size: int) -> int:
        if size <= 0:
            raise ValueError(f"nonpositive allocation {name}={size}")
        if name in self.objects:
            if self.objects[name]["bytes"] != size:
                raise ValueError(f"object size changed: {name}")
            return self.objects[name]["base_address"]
        base = math.ceil(self.cursor / self.alignment) * self.alignment
        self.objects[name] = {"base_address": base, "bytes": size}
        self.cursor = base + size
        return base


def validate(directory: Path) -> tuple[dict[str, Any], list[dict[str, str]], np.ndarray, np.ndarray]:
    manifest_path = directory / "manifest.json"
    csv_path = directory / "tile_records.csv"
    npz_path = directory / "packed_tiles.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    admitted_status = {
        "dual_line_real_tile_trace_v1": "PASS_REAL_BITMAPS_ROW_SELECTOR_TILE_EXECUTION_NOT_ACC32_ORACLE",
        "dual_line_real_tile_trace_v2": "PASS_REAL_BITMAPS_STRATIFIED_ADJACENT_C4_NOT_ACC32_ORACLE",
    }
    if schema not in admitted_status:
        raise ValueError("unsupported tile trace schema")
    if manifest.get("status") != admitted_status[schema]:
        raise ValueError("tile trace status is not admitted")
    load_audit = manifest.get("run_context", {}).get("checkpoint_load_audit", {})
    for field in ("missing_count", "unexpected_count", "overlay_missing_count", "overlay_unexpected_count"):
        if load_audit.get(field) != 0:
            raise ValueError(f"checkpoint load audit is not clean: {field}")
    artifact_identity = manifest.get("run_context", {}).get("artifact_identity", {})
    if not artifact_identity.get("checkpoint_sha256") or not artifact_identity.get("config_sha256"):
        raise ValueError("checkpoint/config identity is incomplete")
    for path in (csv_path, npz_path):
        if manifest["sha256"].get(path.name) != sha256(path):
            raise ValueError(f"tile trace SHA mismatch: {path}")
    records = read_csv(csv_path)
    with np.load(npz_path) as arrays:
        current = arrays["packed_current_bits"].copy()
        previous = arrays["packed_previous_bits"].copy()
    if len(records) != int(manifest["records"]) or len(records) != len(current) or current.shape != previous.shape:
        raise ValueError("tile record/bitmap cardinality mismatch")
    tile_bits = int(manifest["tile_bits"])
    if current.shape[1] * 8 != tile_bits:
        raise ValueError("packed tile width mismatch")
    temporal_records: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        if integer(row, "record_id") != index:
            raise ValueError("record ids are not dense and ordered")
        chunks = math.ceil(integer(row, "source_width") / tile_bits)
        chunk_index = integer(row, "chunk_index")
        source_base = integer(row, "source_base")
        valid_bits = integer(row, "valid_bits")
        if integer(row, "chunks_per_row") != chunks or source_base != chunk_index * tile_bits:
            raise ValueError(f"source/chunk geometry mismatch: {index}")
        expected_valid = min(tile_bits, integer(row, "source_width") - source_base)
        if valid_bits != expected_valid or not 0 < valid_bits <= tile_bits:
            raise ValueError(f"valid extent mismatch: {index}")
        weight_group = integer(row, "weight_group")
        fanout = integer(row, "output_channel_fanout")
        if weight_group < 0 or fanout <= 0:
            raise ValueError(f"invalid weight group/fanout: {index}")
        if integer(row, "output_lane_tile_count_96") != math.ceil(fanout / 96):
            raise ValueError(f"96-lane output tile count mismatch: {index}")
        if schema == "dual_line_real_tile_trace_v2":
            cluster_rows = integer(row, "sample_cluster_rows")
            cluster_lane = integer(row, "sample_cluster_lane")
            population = integer(row, "stratum_population_clusters")
            samples = integer(row, "stratum_sample_clusters")
            weight = float(row["cluster_inverse_probability_weight"])
            if not 0 < cluster_rows <= int(manifest["cluster_contexts"]):
                raise ValueError(f"invalid cluster context count: {index}")
            if not 0 <= cluster_lane < cluster_rows:
                raise ValueError(f"invalid cluster lane: {index}")
            if not 0 < samples <= population or not math.isclose(
                weight, population / samples, rel_tol=1e-12
            ):
                raise ValueError(f"invalid cluster design weight: {index}")
        bits = np.unpackbits(current[index], bitorder="little")[: integer(row, "valid_bits")]
        old = np.unpackbits(previous[index], bitorder="little")[: integer(row, "valid_bits")]
        if int(bits.sum()) != integer(row, "tile_current_count"):
            raise ValueError(f"current popcount mismatch: {index}")
        if int(np.logical_and(bits, np.logical_not(old)).sum()) != integer(row, "tile_positive_count"):
            raise ValueError(f"positive popcount mismatch: {index}")
        if int(np.logical_and(np.logical_not(bits), old).sum()) != integer(row, "tile_negative_count"):
            raise ValueError(f"negative popcount mismatch: {index}")
        state_valid = truth(row["state_valid"])
        timestep = integer(row, "temporal_step")
        if state_valid != (timestep > 0):
            raise ValueError(f"state-valid/timestep mismatch: {index}")
        expected_motion = state_valid and integer(row, "row_transition_count") < integer(row, "row_current_count")
        if truth(row["row_use_motion"]) != expected_motion:
            raise ValueError(f"row selector formula mismatch: {index}")
        temporal_key = (
            row["sample_id"], row["sequence_key"], row["name"], row["operator_call_index"],
            row["row_id"], row["chunk_index"],
        )
        temporal_records[temporal_key].append(index)
    if len(temporal_records) != int(manifest["row_chunk_identities"]):
        raise ValueError("row/chunk identity cardinality mismatch")
    for indices in temporal_records.values():
        indices.sort(key=lambda item: integer(records[item], "temporal_step"))
        steps = [integer(records[item], "temporal_step") for item in indices]
        if steps != list(range(len(steps))):
            raise ValueError("temporal steps are not contiguous from t0")
        for order, item in enumerate(indices):
            if order == 0:
                if np.any(previous[item]):
                    raise ValueError(f"t0 previous bitmap is nonzero: {item}")
            elif not np.array_equal(previous[item], current[indices[order - 1]]):
                raise ValueError(f"previous/current temporal chain mismatch: {item}")
    if schema == "dual_line_real_tile_trace_v2":
        cluster_members: dict[tuple[str, ...], dict[int, int]] = defaultdict(dict)
        for row in records:
            key = (
                row["sample_id"], row["sequence_key"], row["name"],
                row["operator_call_index"], row["weight_group"], row["sample_cluster_id"],
            )
            lane = integer(row, "sample_cluster_lane")
            row_id = integer(row, "row_id")
            previous_id = cluster_members[key].setdefault(lane, row_id)
            if previous_id != row_id:
                raise ValueError("cluster lane changed row identity across time/chunks")
        for key, lanes in cluster_members.items():
            ordered = [lanes[index] for index in range(len(lanes))]
            if sorted(lanes) != list(range(len(lanes))) or ordered != list(
                range(ordered[0], ordered[0] + len(ordered))
            ):
                raise ValueError(f"cluster is not a contiguous physical C4: {key}")
    return manifest, records, current, previous


def group_rows(records: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        key = (
            row["sample_id"], row["sequence_key"], row["name"], row["operator_call_index"],
            row["row_id"], row["temporal_step"],
        )
        groups[key].append(row)
    result = []
    for rows in groups.values():
        rows.sort(key=lambda row: integer(row, "chunk_index"))
        chunks = integer(rows[0], "chunks_per_row")
        if len(rows) != chunks or [integer(row, "chunk_index") for row in rows] != list(range(chunks)):
            raise ValueError(
                f"sampled row is not chunk-complete: {rows[0]['name']} row={rows[0]['row_id']} t={rows[0]['temporal_step']}"
            )
        invariant = (rows[0]["row_current_count"], rows[0]["row_transition_count"], rows[0]["row_use_motion"])
        if any((row["row_current_count"], row["row_transition_count"], row["row_use_motion"]) != invariant for row in rows):
            raise ValueError("row selector metadata changed across chunks")
        current_sum = sum(integer(row, "tile_current_count") for row in rows)
        transition_sum = sum(
            integer(row, "tile_positive_count") + integer(row, "tile_negative_count") for row in rows
        )
        if current_sum != integer(rows[0], "row_current_count"):
            raise ValueError("tile current counts do not conserve to row count")
        if transition_sum != integer(rows[0], "row_transition_count"):
            raise ValueError("tile transition counts do not conserve to row count")
        result.append(rows)
    result.sort(key=lambda rows: (
        integer(rows[0], "sample_id"), rows[0]["name"], integer(rows[0], "operator_call_index"),
        integer(rows[0], "row_id"), integer(rows[0], "temporal_step"),
    ))
    return result


def schedule_variant(
    groups: list[list[dict[str, str]]],
    *,
    variant: str,
    lanes: int,
    sources_per_cycle: int,
    command_overhead: int,
    bitmap_bytes_per_cycle: int,
    acc_bytes_per_cycle: int,
    weight_bytes_per_cycle: int,
    arena: AddressArena,
    motion_enabled: bool | None = None,
    state_storage_model: str = "none",
    emit_transactions: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if motion_enabled is None:
        motion_enabled = variant != "local_only"
    if state_storage_model not in {"none", "shared_output_state", "explicit_copy_state"}:
        raise ValueError(f"unsupported state storage model: {state_storage_model}")
    if motion_enabled != (state_storage_model != "none"):
        raise ValueError("motion/state-storage configuration is inconsistent")
    cycle = 0
    transactions: list[dict[str, Any]] = []
    totals: dict[str, int] = defaultdict(int)

    def transaction(
        *, phase: str, object_id: str, base: int, total_bytes: int,
        service_cycles: int, row: dict[str, str], pattern: str,
    ) -> None:
        nonlocal cycle
        start = cycle
        cycle += service_cycles
        if emit_transactions:
            transactions.append({
            "variant": variant,
            "cycle_start": start,
            "cycle_end_exclusive": cycle,
            "phase": phase,
            "read_or_write": "write" if "write" in phase else "read",
            "base_address": f"0x{base:016x}",
            "total_bytes": total_bytes,
            "object_id": object_id,
            "address_pattern": pattern,
            "sample_id": row["sample_id"],
            "name": row["name"],
            "operator_call_index": row["operator_call_index"],
            "row_id": row["row_id"],
            "temporal_step": row["temporal_step"],
            })
        totals[f"{phase}_bytes"] += total_bytes
        totals[f"{phase}_cycles"] += service_cycles

    last_timestep: dict[str, int] = {}
    for rows in groups:
        row = rows[0]
        key = f"s{row['sample_id']}:{row['name']}:c{row['operator_call_index']}:r{row['row_id']}"
        last_timestep[key] = max(last_timestep.get(key, -1), integer(row, "temporal_step"))

    for rows in groups:
        head = rows[0]
        source_width = integer(head, "source_width")
        fanout = integer(head, "output_channel_fanout")
        weight_group = integer(head, "weight_group")
        timestep = integer(head, "temporal_step")
        row_motion = truth(head["row_use_motion"]) and motion_enabled
        segments = math.ceil(fanout / lanes)
        row_key = (
            f"s{head['sample_id']}:{head['name']}:c{head['operator_call_index']}:r{head['row_id']}"
        )
        current_object = f"activation:{row_key}:t{timestep}"
        current_bytes = math.ceil(source_width / 8)
        current_base = arena.allocate(current_object, current_bytes)
        transaction(
            phase="current_bitmap_read", object_id=current_object, base=current_base,
            total_bytes=current_bytes, service_cycles=math.ceil(current_bytes / bitmap_bytes_per_cycle),
            row=head, pattern="LINEAR_PACKED_BYTES",
        )

        acc_bytes = fanout * 4
        totals["peak_row_retained_state_bytes"] = max(
            totals["peak_row_retained_state_bytes"], (current_bytes + acc_bytes) if motion_enabled else 0
        )
        totals["peak_row_incremental_state_bytes"] = max(
            totals["peak_row_incremental_state_bytes"],
            (current_bytes + acc_bytes) if state_storage_model == "explicit_copy_state" else 0,
        )
        state_bits_object = f"state_bits:{row_key}"
        acc_object = f"state_acc32:{row_key}"
        if motion_enabled and timestep > 0:
            if state_storage_model == "shared_output_state":
                previous_bits_object = f"activation:{row_key}:t{timestep - 1}"
                previous_bits_base = arena.allocate(previous_bits_object, current_bytes)
            else:
                previous_bits_object = state_bits_object
                previous_bits_base = arena.allocate(state_bits_object, current_bytes)
            transaction(
                phase="previous_bitmap_read", object_id=previous_bits_object, base=previous_bits_base,
                total_bytes=current_bytes, service_cycles=math.ceil(current_bytes / bitmap_bytes_per_cycle),
                row=head, pattern="LINEAR_PACKED_BYTES",
            )
            if row_motion:
                if state_storage_model == "shared_output_state":
                    previous_acc_object = f"output_acc32:{row_key}:t{timestep - 1}"
                    previous_acc_base = arena.allocate(previous_acc_object, acc_bytes)
                else:
                    previous_acc_object = acc_object
                    previous_acc_base = arena.allocate(acc_object, acc_bytes)
                transaction(
                    phase="previous_acc32_read", object_id=previous_acc_object, base=previous_acc_base,
                    total_bytes=acc_bytes, service_cycles=math.ceil(acc_bytes / acc_bytes_per_cycle),
                    row=head, pattern="CONTIGUOUS_ACC32",
                )

        selected_counts = []
        record_ids = []
        for row in rows:
            record_ids.append(integer(row, "record_id"))
            if row_motion:
                selected_counts.append(integer(row, "tile_positive_count") + integer(row, "tile_negative_count"))
            else:
                selected_counts.append(integer(row, "tile_current_count"))
        commands = len(rows) * segments
        issue_cycles = sum(math.ceil(count / sources_per_cycle) for count in selected_counts) * segments
        selected_sources = sum(selected_counts) * segments
        transferred_weight_bytes = selected_sources * lanes
        weight_cycles = math.ceil(transferred_weight_bytes / weight_bytes_per_cycle) if transferred_weight_bytes else 0
        compute_cycles = command_overhead * commands + max(issue_cycles, weight_cycles)
        weight_object = f"weight:{head['name']}:{head['operator']}:g{weight_group}"
        weight_base = arena.allocate(weight_object, source_width * fanout)
        start = cycle
        cycle += compute_cycles
        if emit_transactions:
            transactions.append({
            "variant": variant,
            "cycle_start": start,
            "cycle_end_exclusive": cycle,
            "phase": "weight_read_and_accumulate",
            "read_or_write": "read",
            "base_address": f"0x{weight_base:016x}",
            "total_bytes": transferred_weight_bytes,
            "object_id": weight_object,
            "address_pattern": (
                "group_base + (source_base+source_index)*group_fanout + output_lane_segment; "
                "group=" + str(weight_group) + "; record_ids=" + ",".join(map(str, record_ids))
            ),
            "sample_id": head["sample_id"],
            "name": head["name"],
            "operator_call_index": head["operator_call_index"],
            "row_id": head["row_id"],
            "temporal_step": head["temporal_step"],
            })
        totals["weight_read_and_accumulate_bytes"] += transferred_weight_bytes
        totals["weight_read_and_accumulate_cycles"] += compute_cycles
        totals["commands"] += commands
        totals["selected_source_segments"] += selected_sources

        output_object = f"output_acc32:{row_key}:t{timestep}"
        output_base = arena.allocate(output_object, acc_bytes)
        transaction(
            phase="output_acc32_write", object_id=output_object, base=output_base,
            total_bytes=acc_bytes, service_cycles=math.ceil(acc_bytes / acc_bytes_per_cycle),
            row=head, pattern="CONTIGUOUS_ACC32",
        )
        if motion_enabled and timestep < last_timestep[row_key] and state_storage_model == "explicit_copy_state":
            state_bits_base = arena.allocate(state_bits_object, current_bytes)
            transaction(
                phase="state_bitmap_write", object_id=state_bits_object, base=state_bits_base,
                total_bytes=current_bytes, service_cycles=math.ceil(current_bytes / bitmap_bytes_per_cycle),
                row=head, pattern="LINEAR_PACKED_BYTES",
            )
            acc_base = arena.allocate(acc_object, acc_bytes)
            transaction(
                phase="state_acc32_write", object_id=acc_object, base=acc_base,
                total_bytes=acc_bytes, service_cycles=math.ceil(acc_bytes / acc_bytes_per_cycle),
                row=head, pattern="CONTIGUOUS_ACC32",
            )
        totals["rows"] += 1
        totals["motion_rows"] += int(row_motion)

    totals["cycles"] = cycle
    totals["transactions"] = len(transactions)
    return transactions, dict(totals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"), required=True)
    parser.add_argument("--lanes", type=int, default=96)
    parser.add_argument("--sources-per-cycle", type=int, default=1)
    parser.add_argument("--command-overhead", type=int, default=5)
    parser.add_argument("--bitmap-bytes-per-cycle", type=int, default=32)
    parser.add_argument("--acc-bytes-per-cycle", type=int, default=64)
    parser.add_argument("--weight-bytes-per-cycle", type=int, default=96)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.lanes, args.sources_per_cycle, args.bitmap_bytes_per_cycle, args.acc_bytes_per_cycle, args.weight_bytes_per_cycle) <= 0:
        raise ValueError("all widths/bandwidths must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "dual_line_sampled_address_timed_trace_v2",
        "status": "PASS_SAMPLED_TILE_MEMORY_SCHEDULE_NOT_FULL_NETWORK",
        "config": vars(args) | {"output": str(args.output)},
        "identities": {},
    }
    all_transactions = []
    for label, path in args.identity:
        directory = Path(path)
        manifest, records, _current, _previous = validate(directory)
        groups = group_rows(records)
        identity: dict[str, Any] = {
            "tile_manifest": manifest,
            "chunk_complete_row_timesteps": len(groups),
            "variants": {},
        }
        variants = (
            ("local_only", False, "none"),
            ("local_motion_shared_state", True, "shared_output_state"),
            ("local_motion_explicit_copy", True, "explicit_copy_state"),
        )
        for variant, motion_enabled, state_storage_model in variants:
            arena = AddressArena(base=0x1000_0000 if variant == "local_only" else 0x8000_0000)
            transactions, totals = schedule_variant(
                groups, variant=variant, lanes=args.lanes,
                sources_per_cycle=args.sources_per_cycle, command_overhead=args.command_overhead,
                bitmap_bytes_per_cycle=args.bitmap_bytes_per_cycle,
                acc_bytes_per_cycle=args.acc_bytes_per_cycle,
                weight_bytes_per_cycle=args.weight_bytes_per_cycle, arena=arena,
                motion_enabled=motion_enabled, state_storage_model=state_storage_model,
                emit_transactions=not args.summary_only,
            )
            for row in transactions:
                row["identity"] = label
            all_transactions.extend(transactions)
            identity["variants"][variant] = {
                **totals,
                "address_object_count": len(arena.objects),
                "address_span_bytes": arena.cursor - arena.base,
                "address_map_sha256": hashlib.sha256(
                    json.dumps(arena.objects, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
        local = identity["variants"]["local_only"]["cycles"]
        for variant in ("local_motion_shared_state", "local_motion_explicit_copy"):
            motion = identity["variants"][variant]["cycles"]
            identity[variant + "_comparison"] = {
                "local_only_over_variant": local / motion,
                "cycle_change_vs_local_only": motion / local - 1.0,
            }
        payload["identities"][label] = identity
    if not args.summary_only:
        csv_path = args.output / "compressed_memory_transactions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_transactions[0]))
            writer.writeheader()
            writer.writerows(all_transactions)
        payload["transactions_sha256"] = sha256(csv_path)
    (args.output / "memory_trace_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Sampled real-tile address/timed memory trace\n\n",
        "| identity | state model | Local-only cycles | Local+Motion cycles | Local/Motion | Motion cycle change |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for label, identity in payload["identities"].items():
        local = identity["variants"]["local_only"]["cycles"]
        for variant, state_model in (
            ("local_motion_shared_state", "shared output/activation"),
            ("local_motion_explicit_copy", "explicit state copy"),
        ):
            motion = identity["variants"][variant]["cycles"]
            lines.append(
                f"| {label} | {state_model} | {local} | {motion} | {local / motion:.6f}x | {motion / local - 1.0:+.4%} |\n"
            )
    lines.append(
        "\nThis is a chunk-complete sampled tile schedule with deterministic addresses and compressed timestamps. "
        "It is not a full-network trace, CACTI result, DRAMsim3 result, or paper latency.\n"
    )
    (args.output / "REPORT.md").write_text("".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
