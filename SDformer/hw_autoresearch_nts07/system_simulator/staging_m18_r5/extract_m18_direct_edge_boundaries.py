#!/usr/bin/env python3
"""Certify the exact pass-through paths hidden behind r8 ``direct_m4`` edges.

The r8 dependency classifier called BatchNorm a pass-through module.  That is
topologically useful, but it is not a producer-retirement boundary when the
deployment policy is ``no_running``: BatchNorm output depends on statistics
from its complete reduction domain.  M18 therefore emits evidence-only path
certificates and explicitly blocks every such edge from M15.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analyze_h67_atlif_dependency_dag import (
    PASS_FUNCTION_PREFIXES,
    PASS_MODULE_TYPES,
    PRODUCER_TYPES,
    build_output_index,
    find_latest_producer,
    load_events,
    sha256,
    tensor_identity,
)


EXACT_MATCH_QUALITIES = {"exact_tensor_version", "exact_view_version"}
BN_MODULE_TYPES = {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}
HARDWARE_TAG_FIELD_ORDER = (
    "static_edge_id", "sample_epoch", "population_cluster_id", "t",
    "line_id", "context", "lane_tile",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def unsigned_width(maximum: int) -> int:
    if maximum < 0:
        raise ValueError("hardware-tag fields must be unsigned")
    return max(1, int(maximum).bit_length())


def _compact_event(event: dict[str, Any], matched_output: dict[str, Any],
                   downstream_tensor: dict[str, Any], match_quality: str) -> dict[str, Any]:
    return {
        "event_index": int(event["event_index"]),
        "kind": str(event.get("kind", "")),
        "name": str(event.get("name", "")),
        "module_type": str(event.get("module_type", "")) or None,
        "module_call_index": (
            int(event["module_call_index"])
            if event.get("module_call_index") is not None else None
        ),
        "match_quality": match_quality,
        "input_tensors": [tensor_identity(item) for item in event.get("inputs", [])],
        "matched_output_tensor": tensor_identity(matched_output),
        "downstream_tensor": tensor_identity(downstream_tensor),
    }


def trace_exact_passthrough_path(
    input_tensor: dict[str, Any], enter_event_index: int,
    output_index: dict[Any, Any], sample_id: int, sequence_key: str,
    expected_producer: str, max_depth: int = 48,
) -> list[dict[str, Any]]:
    """Return one complete producer-to-ATLIF path or fail closed.

    Every intermediate node must be a one-input allow-listed pass-through and
    every link must preserve an exact tensor version.  Broad storage overlap,
    version drift, fan-in, cycles, and unsupported transforms are rejected.
    """
    current_ref = input_tensor
    upper = int(enter_event_index)
    reverse_hops: list[dict[str, Any]] = []
    visited: set[tuple[Any, ...]] = set()
    for _ in range(max_depth + 1):
        visit_key = (
            int(current_ref.get("storage_cdata", 0)),
            int(current_ref.get("storage_offset", 0)),
            tuple(current_ref.get("shape", [])), tuple(current_ref.get("stride", [])),
            int(current_ref.get("version", -1)), upper,
        )
        if visit_key in visited:
            raise ValueError("cycle in direct edge pass-through ancestry")
        visited.add(visit_key)
        match = find_latest_producer(
            current_ref, upper, output_index, sample_id, sequence_key
        )
        if match is None:
            raise ValueError("direct edge path has an unmatched tensor version")
        event, matched_output, quality = match
        if quality not in EXACT_MATCH_QUALITIES:
            raise ValueError(
                "direct edge path contains non-exact match quality: {}".format(quality)
            )
        hop = _compact_event(event, matched_output, current_ref, quality)
        reverse_hops.append(hop)
        event_kind = str(event.get("kind", ""))
        module_type = str(event.get("module_type", ""))
        name = str(event.get("name", ""))
        if event_kind in {"leaf_module_exit", "leaf_module"} and module_type in PRODUCER_TYPES:
            if name != expected_producer:
                raise ValueError("direct edge does not terminate at the exact audited producer")
            path = list(reversed(reverse_hops))
            indices = [item["event_index"] for item in path]
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError("pass-through certificate event order is not strictly increasing")
            for position, item in enumerate(path):
                item["producer_to_consumer_position"] = position
            return path
        is_passthrough = (
            event_kind in {"leaf_module_exit", "leaf_module"}
            and module_type in PASS_MODULE_TYPES
        ) or (
            event_kind == "functional_op"
            and name.startswith(PASS_FUNCTION_PREFIXES)
        )
        if not is_passthrough:
            raise ValueError("direct edge path contains an unsupported transform")
        inputs = event.get("inputs", [])
        if len(inputs) != 1:
            raise ValueError("direct edge pass-through is not a one-input path")
        current_ref = inputs[0]
        upper = int(event["event_index"])
    raise ValueError("direct edge pass-through exceeds the depth limit")


def _index_atlif_enters(events: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    enters: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        if not (
            event.get("kind") == "leaf_module_enter"
            and event.get("module_type") == "ATLIFTernaryPSN"
        ):
            continue
        key = (
            int(event.get("sample_id", -1)), str(event.get("sequence_key", "")),
            str(event.get("name", "")), int(event.get("module_call_index", 0)),
        )
        if key in enters:
            raise ValueError("duplicate ATLIF enter call identity")
        enters[key] = event
    return enters


def _validate_audit_closure(audit: dict[str, Any], admitted: list[dict[str, Any]]) -> None:
    if any(
        row.get("category") != "direct_m4"
        or row.get("live") is not True
        or row.get("admitted_for_overlap") is not True
        for row in admitted
    ):
        raise ValueError("r8 admitted row is not a live direct_m4 call")
    keys = [(
        int(row["sample_id"]), str(row["sequence_key"]), str(row["name"]),
        int(row["module_call_index"]),
    ) for row in admitted]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate admitted call identity in r8 audit")
    live_direct = [
        row for row in audit.get("rows", [])
        if row.get("category") == "direct_m4" and row.get("live") is True
    ]
    live_keys = [(
        int(row["sample_id"]), str(row["sequence_key"]), str(row["name"]),
        int(row["module_call_index"]),
    ) for row in live_direct]
    if len(live_keys) != len(set(live_keys)) or set(live_keys) != set(keys):
        raise ValueError("r8 live direct_m4 population differs from its admitted population")
    summary = audit.get("summary", {})
    required = ("admitted_direct_m4_calls", "admitted_direct_m4_service_cycles")
    if any(field not in summary for field in required):
        raise ValueError("r8 audit lacks direct-M4 cardinality/service closure")
    if int(summary["admitted_direct_m4_calls"]) != len(admitted):
        raise ValueError("r8 audit admitted call total is internally inconsistent")
    service = sum(int(row["service_cycles_l16"]) for row in admitted)
    if int(summary["admitted_direct_m4_service_cycles"]) != service:
        raise ValueError("r8 audit admitted service total is internally inconsistent")
    category = summary.get("categories", {}).get("direct_m4")
    if not isinstance(category, dict):
        raise ValueError("r8 audit lacks summary.categories.direct_m4 closure")
    if (
        int(category.get("calls", -1)) != len(admitted)
        or int(category.get("live_calls", -1)) != len(admitted)
        or int(category.get("service_cycles", -1)) != service
    ):
        raise ValueError("r8 summary.categories.direct_m4 is internally inconsistent")


def _producer_fanout(row: dict[str, Any]) -> int:
    shape = list(row["producer_output_tensor"].get("shape", []))
    module_type = row["producer_module_type"]
    if module_type == "Linear":
        fanout = int(shape[-1]) if shape else 0
    elif module_type in {"Conv2d", "ConvTranspose2d"}:
        # Sequence-wrapped Conv2d tensors are [T,B,C,H,W]; ordinary tensors
        # are [N,C,H,W].  Channel is consistently the third-from-last axis.
        fanout = int(shape[-3]) if len(shape) >= 3 else 0
    elif module_type == "Conv3d":
        fanout = int(shape[-4]) if len(shape) >= 4 else 0
    else:
        fanout = 0
    if fanout <= 0:
        raise ValueError("cannot derive a positive producer fanout for hardware tag ledger")
    return fanout


def build_hardware_tag_ledger(
    rows: list[dict[str, Any]], *, sample_epoch_bits: int = 16,
    population_cluster_bits: int = 17, contexts: int = 4,
    producer_lanes: int = 96,
) -> dict[str, Any]:
    """Build the finite, synthesis-facing tag contract (never a trace SHA)."""
    if not rows or min(sample_epoch_bits, population_cluster_bits, contexts, producer_lanes) <= 0:
        raise ValueError("hardware tag configuration must be positive")
    static_keys = sorted({(
        str(row["producer"]), int(row["producer_call_index"]),
        str(row["edge"]), int(row["edge_call_index"]),
    ) for row in rows})
    static_id = {key: index for index, key in enumerate(static_keys)}
    max_t = max(int(row["temporal_steps"]) for row in rows) - 1
    max_lane_tile = max(
        int(math.ceil(float(_producer_fanout(row)) / producer_lanes)) - 1
        for row in rows
    )
    fields = {
        "static_edge_id": {
            "bits": unsigned_width(len(static_keys) - 1), "minimum": 0,
            "maximum": len(static_keys) - 1, "encoding": "compile_time_edge_dictionary",
        },
        "sample_epoch": {
            "bits": sample_epoch_bits, "minimum": 0,
            "maximum": (1 << sample_epoch_bits) - 1, "encoding": "runtime_modulo_counter",
        },
        "population_cluster_id": {
            "bits": population_cluster_bits, "minimum": 0,
            "maximum": (1 << population_cluster_bits) - 1,
            "encoding": "downstream_exact_population_ordinal",
        },
        "t": {
            "bits": unsigned_width(max_t), "minimum": 0, "maximum": max_t,
            "encoding": "temporal_step",
        },
        "line_id": {
            "bits": 1, "minimum": 0, "maximum": 1,
            "encoding": "selected_source_line_local0_motion1",
        },
        "context": {
            "bits": unsigned_width(contexts - 1), "minimum": 0,
            "maximum": contexts - 1, "encoding": "resident_context_slot",
        },
        "lane_tile": {
            "bits": unsigned_width(max_lane_tile), "minimum": 0,
            "maximum": max_lane_tile, "encoding": "producer_lane_tile_ordinal",
        },
    }
    for row in rows:
        key = (
            str(row["producer"]), int(row["producer_call_index"]),
            str(row["edge"]), int(row["edge_call_index"]),
        )
        row["static_edge_id"] = static_id[key]
        row["producer_fanout"] = _producer_fanout(row)
        row["maximum_lane_tile"] = int(math.ceil(float(row["producer_fanout"]) / producer_lanes)) - 1
        row["hardware_tag_binding"] = {
            "static_edge_id": row["static_edge_id"],
            "sample_epoch": "allocated_at_runtime_after_sample_fence",
            "population_cluster_id": "bound_by_exact_population_stream",
            "t": "bound_by_temporal_step",
            "line_id": "bound_by_exclusive_local_or_motion_source_selector",
            "context": "bound_by_context_allocator",
            "lane_tile": "bound_by_producer_lane_tile",
        }
    dictionary = [{
        "static_edge_id": index, "producer": key[0], "producer_call_index": key[1],
        "edge": key[2], "edge_call_index": key[3],
    } for key, index in sorted(static_id.items(), key=lambda item: item[1])]
    return {
        "schema": "m18_finite_hardware_tag_ledger_v2",
        "configuration": {
            "sample_epoch_bits": sample_epoch_bits,
            "population_cluster_bits": population_cluster_bits,
            "contexts": contexts,
            "producer_lanes": producer_lanes,
        },
        "field_order": list(HARDWARE_TAG_FIELD_ORDER),
        "fields": fields,
        "required_bits": sum(int(fields[name]["bits"]) for name in HARDWARE_TAG_FIELD_ORDER),
        "static_edge_dictionary": dictionary,
        "static_edge_dictionary_sha256": canonical_sha256(dictionary),
        "lifecycle_contract": {
            "context_release": (
                "A context is exclusively owned by one complete "
                "(static_edge_id,sample_epoch,population_cluster_id) group. Multiple "
                "t/line_id/lane_tile tokens from that owner may coexist; the context is released only "
                "after consumer-finish retires the final token. Producer completion or FIFO "
                "dequeue cannot release it."
            ),
            "line_namespace": (
                "The source selector issues exactly one Local-or-Motion line for each "
                "(static_edge_id,sample_epoch,population_cluster_id,t,context) decision; "
                "the selected line may legally change at another temporal step. "
                "line_id remains part of the full tag so a shared FIFO/tag namespace cannot "
                "alias the two source definitions. Context slots are globally shared."
            ),
            "sample_fence": (
                "Before sample_epoch advances, all producer tokens, FIFOs, consumers, and "
                "context owners from the current epoch must drain."
            ),
            "sample_replay": (
                "Replaying the same dataset sample allocates the next sample_epoch; sample_id "
                "and sequence strings are evidence metadata, not hardware tag fields."
            ),
            "epoch_wrap_fence": (
                "Wrapping sample_epoch to zero requires a global drain, explicit tag-state "
                "invalidation, and acknowledgement before any wrapped tag is issued."
            ),
        },
        "excluded_from_hardware_tag": [
            "trace_sha256", "path_certificate_sha256", "storage_cdata",
            "storage_data_ptr", "python_id", "sequence_key", "sample_id",
        ],
        "collision_contract": (
            "All seven finite fields compare in full; no hash truncation or runtime pointer is "
            "permitted as a hardware identity."
        ),
    }


def bind_hardware_tag_ledger_to_m17(
    ledger: dict[str, Any], rows: list[dict[str, Any]], manifest_path: Path,
) -> dict[str, Any]:
    """Bind every finite field bound to the exact M17 ordered population stream."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_EXACT_FULL_SPATIAL_C4_SUFFICIENT_STATISTICS_NOT_SYSTEM_SPEEDUP":
        raise ValueError("M17 exact full-spatial manifest is not admitted")
    stream_path = manifest_path.parent / "ordered_stream.npz"
    prototypes_path = manifest_path.parent / "prototypes.json"
    entry = manifest.get("files", {}).get("ordered_stream.npz")
    prototypes_entry = manifest.get("files", {}).get("prototypes.json")
    if not isinstance(entry, dict):
        raise ValueError("M17 manifest lacks ordered_stream.npz identity")
    if not isinstance(prototypes_entry, dict):
        raise ValueError("M17 manifest lacks prototypes.json identity")
    if sha256(stream_path) != entry.get("sha256") or stream_path.stat().st_size != int(entry.get("bytes", -1)):
        raise ValueError("M17 ordered stream hash/size mismatch")
    if (
        sha256(prototypes_path) != prototypes_entry.get("sha256")
        or prototypes_path.stat().st_size != int(prototypes_entry.get("bytes", -1))
    ):
        raise ValueError("M17 prototype hash/size mismatch")
    prototypes = json.loads(prototypes_path.read_text(encoding="utf-8"))
    if not isinstance(prototypes, list) or not prototypes:
        raise ValueError("M17 prototype population is empty")
    with np.load(stream_path, allow_pickle=False) as stream:
        required = {"operator_index", "population_cluster_id", "prototype_id"}
        if set(stream.files) != required:
            raise ValueError("M17 ordered stream fields mismatch")
        cardinalities = {len(stream[name]) for name in required}
        if cardinalities != {int(manifest.get("ordered_clusters", -1))}:
            raise ValueError("M17 ordered stream cardinality mismatch")
        if not len(stream["population_cluster_id"]):
            raise ValueError("M17 ordered stream is empty")
        operator_index = stream["operator_index"].copy()
        population = stream["population_cluster_id"].copy()
        prototype_id = stream["prototype_id"].copy()
        observed_population_max = int(population.max())
        observed_operator_max = int(stream["operator_index"].max())
    operators = manifest.get("operators", [])
    if observed_operator_max >= len(operators):
        raise ValueError("M17 ordered stream operator index is out of range")
    if set(int(value) for value in np.unique(operator_index)) != set(range(len(operators))):
        raise ValueError("M17 ordered stream does not cover every operator")
    for operator_id in range(len(operators)):
        values = population[operator_index == operator_id]
        if not np.array_equal(values, np.arange(len(values), dtype=values.dtype)):
            raise ValueError("M17 per-operator population IDs are not dense and ordered")
    m17_keys = {
        (
            int(item["sample_id"]), str(item["sequence_key"]), str(item["name"]),
            int(item["operator_call_index"]),
        )
        for item in operators
    }
    boundary_keys = {
        (
            int(item["sample_id"]), str(item["sequence_key"]), str(item["producer"]),
            int(item["producer_call_index"]),
        )
        for item in rows
    }
    if m17_keys != boundary_keys:
        raise ValueError("M17 operator set and M18 producer boundary set differ")
    row_by_key = {
        (
            int(item["sample_id"]), str(item["sequence_key"]), str(item["producer"]),
            int(item["producer_call_index"]),
        ): item
        for item in rows
    }
    any_motion = False
    bound_rows = 0
    bound_populations = 0
    for item in operators:
        key = (
            int(item["sample_id"]), str(item["sequence_key"]), str(item["name"]),
            int(item["operator_call_index"]),
        )
        row = row_by_key[key]
        temporal_steps = int(item.get("temporal_steps", -1))
        fanout = int(item.get("fanout", -1))
        lane_tiles = int(item.get("lane_tiles", -1))
        population_clusters = item.get("population_clusters")
        row_count = item.get("row_count")
        if (
            isinstance(population_clusters, bool)
            or not isinstance(population_clusters, int)
            or population_clusters < 1
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
        ):
            raise ValueError("M17 operator row/population cardinality is invalid")
        positions = np.flatnonzero(operator_index == int(item["operator_index"]))
        if len(positions) != population_clusters:
            raise ValueError("M17 operator stream count differs from population_clusters")
        referenced_contexts = 0
        for position in positions:
            prototype_index = int(prototype_id[position])
            if not 0 <= prototype_index < len(prototypes):
                raise ValueError("M17 ordered stream references an invalid prototype")
            prototype_contexts = prototypes[prototype_index].get("contexts")
            if (
                isinstance(prototype_contexts, bool)
                or not isinstance(prototype_contexts, int)
                or not 1 <= prototype_contexts <= int(ledger["configuration"]["contexts"])
            ):
                raise ValueError("M17 prototype context population is invalid")
            referenced_contexts += prototype_contexts
        if referenced_contexts != row_count:
            raise ValueError("M17 operator row_count differs from referenced prototype contexts")
        expected_lane_tiles = int(math.ceil(
            float(fanout) / int(ledger["configuration"]["producer_lanes"])
        )) if fanout > 0 else -1
        if (
            temporal_steps != int(row["temporal_steps"])
            or fanout != int(row["producer_fanout"])
            or lane_tiles != expected_lane_tiles
            or lane_tiles != int(row["maximum_lane_tile"]) + 1
        ):
            raise ValueError("M17 operator temporal/fanout/lane geometry differs from M18")
        motion_by_t = item.get("motion_selected_rows_by_t")
        if not isinstance(motion_by_t, list) or len(motion_by_t) != temporal_steps:
            raise ValueError("M17 operator Motion selector timeline is incomplete")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            or value < 0 or value > row_count
            for value in motion_by_t
        ):
            raise ValueError("M17 operator Motion selector count is outside the row population")
        if motion_by_t[0] != 0:
            raise ValueError("M17 operator t0 Motion selector count must be zero")
        any_motion = any_motion or any(value > 0 for value in motion_by_t)
        bound_rows += row_count
        bound_populations += population_clusters
    observed = {
        "static_edge_id": max(int(item["static_edge_id"]) for item in rows),
        "sample_epoch": 0,
        "population_cluster_id": observed_population_max,
        "t": max(int(item["temporal_steps"]) - 1 for item in operators),
        "line_id": 1 if any_motion else 0,
        "context": int(ledger["configuration"]["contexts"]) - 1,
        "lane_tile": max(int(item["lane_tiles"]) - 1 for item in operators),
    }
    for name in HARDWARE_TAG_FIELD_ORDER:
        if observed[name] > int(ledger["fields"][name]["maximum"]):
            raise ValueError("hardware tag field is too narrow for M17: {}".format(name))
    ledger["observed_m17_bounds"] = observed
    ledger["m17_binding"] = {
        "manifest_sha256": sha256(manifest_path),
        "ordered_stream_sha256": sha256(stream_path),
        "ordered_clusters": int(manifest["ordered_clusters"]),
        "bound_population_clusters": bound_populations,
        "bound_rows": bound_rows,
        "operators": len(operators),
        "operator_set_sha256": canonical_sha256(sorted(m17_keys)),
        "boundary_set_sha256": canonical_sha256(sorted(boundary_keys)),
        "bound_check": "PASS_ALL_SEVEN_FIELDS_WITHIN_FINITE_MAXIMUM",
    }
    return manifest


class HardwareTagLifecycle:
    """Small executable contract used by synthetic duplicate/wrap tests."""

    def __init__(self, ledger: dict[str, Any]) -> None:
        self.ledger = ledger
        self.epoch: int | None = None
        self.active: set[tuple[int, ...]] = set()
        self.seen: set[tuple[int, ...]] = set()
        self.context_owners: dict[int, tuple[int, ...]] = {}
        self.line_decisions: dict[tuple[int, ...], int] = {}

    def _validate_tag(self, tag: tuple[int, ...]) -> None:
        if len(tag) != len(HARDWARE_TAG_FIELD_ORDER):
            raise ValueError("hardware tag field count mismatch")
        for value, name in zip(tag, HARDWARE_TAG_FIELD_ORDER):
            field = self.ledger["fields"][name]
            if not field["minimum"] <= int(value) <= field["maximum"]:
                raise ValueError("hardware tag field is out of range: {}".format(name))
        if self.epoch is None or int(tag[1]) != self.epoch:
            raise ValueError("stale or future sample_epoch tag")

    def begin_epoch(self, epoch: int, *, wrap_fence: bool = False) -> None:
        maximum = int(self.ledger["fields"]["sample_epoch"]["maximum"])
        if not 0 <= int(epoch) <= maximum:
            raise ValueError("sample_epoch is out of range")
        if self.active or self.context_owners:
            raise ValueError("sample fence cannot advance with live context owners")
        if self.epoch is not None:
            expected = (self.epoch + 1) % (maximum + 1)
            if int(epoch) != expected:
                raise ValueError("sample replay must allocate the next epoch")
            if expected == 0 and not wrap_fence:
                raise ValueError("sample_epoch wrap requires an explicit global fence")
        self.epoch = int(epoch)
        self.seen.clear()
        self.line_decisions.clear()

    def issue(self, tag: tuple[int, ...]) -> None:
        self._validate_tag(tag)
        if tag in self.seen:
            raise ValueError("duplicate hardware tag in one sample epoch")
        context_index = HARDWARE_TAG_FIELD_ORDER.index("context")
        context = int(tag[context_index])
        owner = (
            int(tag[HARDWARE_TAG_FIELD_ORDER.index("static_edge_id")]),
            int(tag[HARDWARE_TAG_FIELD_ORDER.index("sample_epoch")]),
            int(tag[HARDWARE_TAG_FIELD_ORDER.index("population_cluster_id")]),
        )
        current_owner = self.context_owners.get(context)
        if current_owner is not None and current_owner != owner:
            raise ValueError("context already owned by a different population group")
        decision_key = owner + (
            int(tag[HARDWARE_TAG_FIELD_ORDER.index("t")]), context,
        )
        line_id = int(tag[HARDWARE_TAG_FIELD_ORDER.index("line_id")])
        selected_line = self.line_decisions.get(decision_key)
        if selected_line is not None and selected_line != line_id:
            raise ValueError("source line already selected for this population timestep")
        self.seen.add(tag)
        self.active.add(tag)
        self.context_owners[context] = owner
        self.line_decisions[decision_key] = line_id

    def consumer_finish(self, tag: tuple[int, ...]) -> None:
        self._validate_tag(tag)
        if tag not in self.active:
            raise ValueError("stale or duplicate consumer finish")
        self.active.remove(tag)
        context_index = HARDWARE_TAG_FIELD_ORDER.index("context")
        context = int(tag[context_index])
        if not any(int(item[context_index]) == context for item in self.active):
            if context not in self.context_owners:
                raise ValueError("context owner disappeared before final consumer finish")
            del self.context_owners[context]


def extract_boundaries(
    events: list[dict[str, Any]], audit: dict[str, Any], *, bn_policy: str = "no_running",
) -> list[dict[str, Any]]:
    if bn_policy != "no_running":
        raise ValueError("M18 BN-blocked certificate requires the frozen no_running policy")
    output_index = build_output_index(events)
    enters = _index_atlif_enters(events)
    admitted = [row for row in audit["rows"] if row.get("admitted_for_overlap") is True]
    _validate_audit_closure(audit, admitted)
    rows = []
    for row in admitted:
        key = (
            int(row["sample_id"]), str(row["sequence_key"]), str(row["name"]),
            int(row["module_call_index"]),
        )
        if key not in enters:
            raise ValueError("direct edge ATLIF call is missing from dependency events")
        enter = enters[key]
        if int(row["enter_event_index"]) != int(enter["event_index"]):
            raise ValueError("r8 audit enter_event_index is stale or mismatched")
        if len(enter.get("inputs", [])) != 1 or len(row.get("producers", [])) != 1:
            raise ValueError("direct edge does not have one ATLIF input and one producer")
        path = trace_exact_passthrough_path(
            enter["inputs"][0], int(enter["event_index"]), output_index,
            int(row["sample_id"]), str(row["sequence_key"]), str(row["producers"][0]),
        )
        producer = path[0]
        barriers = [{
            "event_index": hop["event_index"], "name": hop["name"],
            "module_type": hop["module_type"], "bn_policy": bn_policy,
            "barrier_kind": "GLOBAL_REDUCTION_STATISTICS_BARRIER",
            "reason": (
                "track_running_stats=False/no_running uses input-dependent batch statistics; "
                "exact normalized outputs are not ready at producer completion."
            ),
            "reduction_input_tensors": hop["input_tensors"],
            "normalized_output_tensor": hop["matched_output_tensor"],
        } for hop in path if hop["module_type"] in BN_MODULE_TYPES]
        if not barriers:
            raise ValueError("historical direct-M4 edge is not explicitly BN-blocked")
        certificate = {
            "producer_to_atlif_path": path,
            "atlif_enter": {
                "event_index": int(enter["event_index"]), "name": str(enter["name"]),
                "module_type": str(enter["module_type"]),
                "module_call_index": int(enter.get("module_call_index", 0)),
                "input_tensor": tensor_identity(enter["inputs"][0]),
            },
        }
        identity = {
            "sample_id": int(row["sample_id"]),
            "sequence_key": str(row["sequence_key"]),
            "producer": str(producer["name"]),
            "producer_module_type": str(producer["module_type"]),
            "producer_call_index": int(producer["module_call_index"]),
            "producer_event_index": int(producer["event_index"]),
            "edge": str(row["name"]),
            "edge_call_index": int(row["module_call_index"]),
            "edge_enter_event_index": int(row["enter_event_index"]),
            "temporal_steps": int(row["temporal_steps"]),
            "service_cycles_l16": int(row["service_cycles_l16"]),
            "producer_output_tensor": producer["matched_output_tensor"],
            "atlif_input_tensor": tensor_identity(enter["inputs"][0]),
            "path_certificate": certificate,
            "path_certificate_sha256": canonical_sha256(certificate),
            "bn_barriers": barriers,
            "causal_classification": "BN_BLOCKED",
            "readiness_boundary": "GLOBAL_REDUCTION_STATISTICS_BARRIER",
            "m15_admitted": False,
            "m15_rejection_reason": "DYNAMIC_BN_OUTPUT_NOT_READY_AT_PRODUCER_P_DONE",
            "r8_historical_overlap_admission_reinterpreted": True,
        }
        rows.append(identity)
    rows.sort(key=lambda item: (
        item["sample_id"], item["sequence_key"], item["producer_event_index"],
        item["edge_enter_event_index"],
    ))
    keys = [(
        row["sample_id"], row["sequence_key"], row["producer"],
        row["producer_call_index"], row["edge"], row["edge_call_index"],
    ) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate direct edge call identity")
    if any(row["m15_admitted"] or row["causal_classification"] != "BN_BLOCKED" for row in rows):
        raise ValueError("M18 fail-close violated: every certified edge must be BN-blocked")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--dependency-manifest", type=Path, required=True)
    parser.add_argument("--dependency-audit-r8", type=Path, required=True)
    parser.add_argument("--m17-manifest", type=Path, required=True)
    parser.add_argument("--sample-epoch-bits", type=int, default=16)
    parser.add_argument("--population-cluster-bits", type=int, default=17)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--producer-lanes", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.dependency_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "h67_tensor_dependency_trace_v2":
        raise ValueError("dependency manifest schema mismatch")
    events_sha = sha256(args.events)
    if manifest.get("dependency_events_sha256") != events_sha:
        raise ValueError("dependency event/manifest hash mismatch")
    audit = json.loads(args.dependency_audit_r8.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_CAUSAL_DEPENDENCY_CLASSIFICATION":
        raise ValueError("r8 dependency audit is not admitted")
    expected_manifest_sha = audit["identities"]["dependency_manifest"]["sha256"]
    if expected_manifest_sha != sha256(args.dependency_manifest):
        raise ValueError("r8 audit/dependency manifest identity mismatch")
    bn_policy = str(manifest.get("run_context", {}).get("eval_protocol", {}).get("bn_policy", ""))
    rows = extract_boundaries(load_events(args.events), audit, bn_policy=bn_policy)
    if len(rows) != int(audit["summary"]["admitted_direct_m4_calls"]):
        raise ValueError("direct boundary cardinality does not match r8 audit")
    service = sum(row["service_cycles_l16"] for row in rows)
    if service != int(audit["summary"]["admitted_direct_m4_service_cycles"]):
        raise ValueError("direct boundary service cycles do not close to r8 audit")
    tag_ledger = build_hardware_tag_ledger(
        rows, sample_epoch_bits=args.sample_epoch_bits,
        population_cluster_bits=args.population_cluster_bits, contexts=args.contexts,
        producer_lanes=args.producer_lanes,
    )
    m17_manifest = bind_hardware_tag_ledger_to_m17(
        tag_ledger, rows, args.m17_manifest.resolve()
    )
    dependency_artifact = manifest["run_context"]["artifact_identity"]
    m17_artifact = m17_manifest["run_context"]["artifact_identity"]
    for field in ("checkpoint_sha256", "config_sha256"):
        if dependency_artifact[field] != m17_artifact[field]:
            raise ValueError("dependency/M17 artifact identity mismatch: {}".format(field))
    test_path = Path(__file__).resolve().parent.parent / "tests" / "test_m18_direct_edge_boundaries.py"
    analyzer_path = Path(__file__).resolve().with_name("analyze_h67_atlif_dependency_dag.py")
    payload = {
        "schema": "m18_direct_m4_bn_blocked_path_certificates_v2",
        "revision": 4,
        "status": "PASS_EXACT_PATH_CERTIFICATES_ALL_BN_BLOCKED_M15_PROHIBITED",
        "configuration": {
            "sample_epoch_bits": args.sample_epoch_bits,
            "population_cluster_bits": args.population_cluster_bits,
            "contexts": args.contexts,
            "producer_lanes": args.producer_lanes,
            "source_lines": {"local": 0, "motion": 1},
        },
        "summary": {
            "historical_r8_direct_m4_edges": len(rows),
            "path_certified_edges": len(rows),
            "bn_blocked_edges": sum(row["causal_classification"] == "BN_BLOCKED" for row in rows),
            "global_reduction_barrier_edges": sum(
                row["readiness_boundary"] == "GLOBAL_REDUCTION_STATISTICS_BARRIER" for row in rows
            ),
            "m15_admitted_edges": sum(bool(row["m15_admitted"]) for row in rows),
            "service_cycles_l16": service,
            "audit_service_cycles_l16": int(audit["summary"]["admitted_direct_m4_service_cycles"]),
        },
        "hardware_tag_ledger": tag_ledger,
        "rows": rows,
        "identities": {
            "events_sha256": events_sha,
            "dependency_manifest_sha256": sha256(args.dependency_manifest),
            "dependency_audit_r8_sha256": sha256(args.dependency_audit_r8),
            "m17_manifest_sha256": sha256(args.m17_manifest),
            "m17_ordered_stream_sha256": sha256(args.m17_manifest.parent / "ordered_stream.npz"),
            "checkpoint_sha256": manifest["run_context"]["artifact_identity"]["checkpoint_sha256"],
            "config_sha256": manifest["run_context"]["artifact_identity"]["config_sha256"],
            "source_sha256": sha256(Path(__file__).resolve()),
            "dependency_analyzer_source_sha256": sha256(analyzer_path),
            "test_source_sha256": sha256(test_path),
        },
        "run_context": {
            "argv": list(sys.argv),
            "python_version": sys.version,
            "source_path": str(Path(__file__).resolve()),
            "dependency_analyzer_source_path": str(analyzer_path),
            "test_source_path": str(test_path),
        },
        "claim_boundary": (
            "Exact call-qualified producer-to-ATLIF pass-through certificates under the frozen "
            "no_running BatchNorm policy. Every historical r8 direct-M4 edge is BN_BLOCKED and "
            "prohibited from M15. Path/SHA/storage identities are trace evidence only; the finite "
            "seven-field hardware tag is separately specified. No BN cycle model, overlap speedup, "
            "memory timing, RTL, energy, or PPA is claimed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS_M18_BN_BLOCKED_PATHS edges={} m15=0".format(len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
