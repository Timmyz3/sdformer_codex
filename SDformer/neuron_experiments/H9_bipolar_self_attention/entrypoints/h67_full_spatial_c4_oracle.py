#!/usr/bin/env python3
"""Exact full-spatial adjacent-C4 sufficient statistics for the M4 kernel.

The writer stores integer wall-cycle totals plus a compact ordered prototype
stream.  It never extrapolates from sampled rows and never calls a profile100
average a same-sample oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_issue_cycles_batched(bank_counts: np.ndarray, reduce_slots: int = 4) -> np.ndarray:
    """Vectorized equivalent of the RTL's bank-first/context-first scheduler."""
    if bank_counts.ndim != 3 or bank_counts.shape[2] != 16 or reduce_slots <= 0:
        raise ValueError("bank_counts must be [groups,contexts,16]")
    if np.any(bank_counts < 0):
        raise ValueError("bank counts must be non-negative")
    remaining = bank_counts.astype(np.int16, copy=True)
    cycles = np.zeros(remaining.shape[0], dtype=np.uint32)
    while np.any(remaining):
        active = np.any(remaining, axis=(1, 2))
        cycles[active] += 1
        used = np.zeros(remaining.shape[:2], dtype=np.uint8)
        issued = 0
        for bank in range(16):
            eligible = (remaining[:, :, bank] > 0) & (used < reduce_slots)
            has_work = np.any(eligible, axis=1)
            rows = np.flatnonzero(has_work)
            if not rows.size:
                continue
            contexts = np.argmax(eligible[rows], axis=1)
            remaining[rows, contexts, bank] -= 1
            used[rows, contexts] += 1
            issued += int(rows.size)
        if issued == 0:
            raise RuntimeError("batched compact scheduler made no progress")
    return cycles


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


EXPECTED_DIRECT_M4_CALLS = 13


def _identity_projection(value: dict[str, Any]) -> dict[str, Any]:
    """Content identity only; absolute paths and mtimes are transport metadata."""
    required = ("config_sha256", "checkpoint_sha256", "checkpoint_size")
    if any(key not in value for key in required):
        raise ValueError("dependency artifact identity is incomplete")
    return {key: value[key] for key in required}


def _load_dependency_sample_identities(events_path: Path) -> tuple[set[tuple[int, str, str]], list[dict[str, Any]]]:
    identities: set[tuple[int, str, str]] = set()
    producer_enters: list[dict[str, Any]] = []
    with events_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid dependency event JSON at line {line_number}") from exc
            if not isinstance(event, dict):
                raise ValueError("dependency events must be JSON objects")
            if "sample_id" not in event:
                continue
            identity = (
                int(event["sample_id"]), str(event.get("sample_key", "")),
                str(event.get("sequence_key", "")),
            )
            if not identity[1] or not identity[2]:
                raise ValueError("dependency event sample identity is incomplete")
            identities.add(identity)
            if event.get("kind") == "leaf_module_enter":
                producer_enters.append(event)
    if not identities:
        raise ValueError("dependency events contain no sample identities")
    return identities, producer_enters


def validate_dependency_contract(
    dependency_manifest_path: Path, dependency_audit_path: Path,
    dependency_events_path: Path, *, artifact_identity: dict[str, Any],
    eval_protocol: dict[str, Any], checkpoint_load_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fail-close cross-binding for the exact M17 producer calls and sample."""
    manifest_path = Path(dependency_manifest_path)
    audit_path = Path(dependency_audit_path)
    events_path = Path(dependency_events_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "h67_tensor_dependency_trace_v2"
        or manifest.get("status") != "PASS_PRE_POST_VERSION_MUTATION_DAG_METADATA_ONLY"
        or int(manifest.get("samples", -1)) != 1
        or int(manifest.get("sample_limit", -1)) != 1
    ):
        raise ValueError("M17 requires one admitted v2 dependency sample")
    if (
        audit.get("schema") != "h67_atlif_dependency_audit_v2"
        or audit.get("status") != "PASS_CAUSAL_DEPENDENCY_CLASSIFICATION"
    ):
        raise ValueError("M17 dependency audit is not admitted v2 evidence")
    audit_manifest = audit.get("identities", {}).get("dependency_manifest", {})
    if audit_manifest.get("sha256") != file_sha256(manifest_path):
        raise ValueError("dependency audit/manifest SHA mismatch")
    if audit_manifest.get("schema") != manifest["schema"]:
        raise ValueError("dependency audit/manifest schema mismatch")
    if int(audit_manifest.get("samples", -1)) != int(manifest["samples"]):
        raise ValueError("dependency audit/manifest sample count mismatch")
    if audit_manifest.get("writer_source_sha256") != manifest.get("run_context", {}).get(
        "source_sha256", {}
    ).get("dependency_writer"):
        raise ValueError("dependency audit/manifest writer identity mismatch")
    if manifest.get("dependency_events_sha256") != file_sha256(events_path):
        raise ValueError("dependency event/manifest SHA mismatch")
    run_context = manifest.get("run_context", {})
    manifest_artifact = run_context.get("artifact_identity", {})
    if _identity_projection(manifest_artifact) != _identity_projection(artifact_identity):
        raise ValueError("dependency/current checkpoint or config identity mismatch")
    if audit_manifest.get("artifact_identity") != manifest_artifact:
        raise ValueError("dependency audit embedded artifact identity mismatch")
    if run_context.get("eval_protocol") != eval_protocol:
        raise ValueError("dependency/current evaluation protocol mismatch")
    if audit_manifest.get("eval_protocol") != run_context["eval_protocol"]:
        raise ValueError("dependency audit embedded evaluation protocol mismatch")
    manifest_load = dict(run_context.get("checkpoint_load_audit", {}))
    current_load = dict(checkpoint_load_audit)
    manifest_load.pop("checkpoint", None)
    current_load.pop("checkpoint", None)
    if manifest_load != current_load:
        raise ValueError("dependency/current checkpoint load audit mismatch")
    if any(int(current_load.get(key, -1)) != 0 for key in (
        "missing_count", "unexpected_count", "overlay_missing_count", "overlay_unexpected_count",
    )):
        raise ValueError("M17 requires an exact checkpoint load")

    sample_identities, producer_enters = _load_dependency_sample_identities(events_path)
    if len(sample_identities) != 1:
        raise ValueError("dependency trace does not contain exactly one sample identity")
    (sample_identity,) = tuple(sample_identities)
    if sample_identity[0] != 0:
        raise ValueError("M17 dependency sample must be sample zero")
    admitted_rows = [
        row for row in audit.get("rows", [])
        if row.get("category") == "direct_m4" and row.get("live") is True
        and row.get("admitted_for_overlap") is True
    ]
    if (
        len(admitted_rows) != EXPECTED_DIRECT_M4_CALLS
        or len(admitted_rows) != int(audit.get("summary", {}).get("admitted_direct_m4_calls", -1))
    ):
        raise ValueError("M17 requires exactly 13 audit-admitted direct-M4 calls")
    contracts = []
    seen: set[tuple[int, str, str, str, int]] = set()
    for row in admitted_rows:
        producers = row.get("producers")
        if not isinstance(producers, list) or len(producers) != 1:
            raise ValueError("direct-M4 audit row must have exactly one producer")
        if int(row.get("sample_id", -1)) != sample_identity[0]:
            raise ValueError("dependency audit/event sample ID mismatch")
        if str(row.get("sequence_key", "")) != sample_identity[2]:
            raise ValueError("dependency audit/event sequence mismatch")
        producer_name = str(producers[0])
        matches = [
            event for event in producer_enters
            if int(event.get("sample_id", -1)) == sample_identity[0]
            and str(event.get("sample_key", "")) == sample_identity[1]
            and str(event.get("sequence_key", "")) == sample_identity[2]
            and str(event.get("name", "")) == producer_name
        ]
        if len(matches) != 1:
            raise ValueError("direct-M4 producer is not one call-qualified dependency event")
        producer_call_index = int(matches[0].get("module_call_index", -1))
        if producer_call_index < 0:
            raise ValueError("direct-M4 producer dependency call index is missing")
        key = sample_identity + (producer_name, producer_call_index)
        if key in seen:
            raise ValueError("duplicate direct-M4 name+call identity")
        seen.add(key)
        contracts.append({
            "sample_id": key[0], "sample_key": key[1], "sequence_key": key[2],
            "name": key[3], "operator_call_index": key[4],
            "atlif_name": str(row.get("name", "")),
            "atlif_module_call_index": int(row.get("module_call_index", -1)),
        })
    return sorted(contracts, key=lambda item: (
        item["sample_id"], item["sequence_key"], item["name"], item["operator_call_index"],
    ))


class H67FullSpatialC4OracleWriter:
    """Forward-hook sink for exact Local/Hybrid M4 row-cluster costs."""

    def __init__(
        self, output_dir: Path, *, allowed_calls: list[dict[str, Any]], tile_bits: int = 256,
        contexts: int = 4, issue_width: int = 16, reduce_slots: int = 4,
        output_lanes: int = 96, cluster_block: int = 2048,
    ) -> None:
        if tile_bits != 256 or contexts != 4 or issue_width != 16 or reduce_slots != 4:
            raise ValueError("M17 v2 freezes the synthesized M4 C4/256b/16-bank/4-slot geometry")
        if output_lanes != 96 or cluster_block <= 0 or not allowed_calls:
            raise ValueError("invalid M17 output lanes, block size, or direct-M4 allowlist")
        self.output_dir = Path(output_dir)
        self.allowed_calls = [dict(item) for item in allowed_calls]
        self.allowed_call_keys = {
            (
                int(item["sample_id"]), str(item["sample_key"]), str(item["sequence_key"]),
                str(item["name"]), int(item["operator_call_index"]),
            )
            for item in self.allowed_calls
        }
        if len(self.allowed_call_keys) != len(self.allowed_calls):
            raise ValueError("direct-M4 call contract contains duplicate identities")
        self.allowed_names = {item[3] for item in self.allowed_call_keys}
        self.tile_bits = tile_bits
        self.contexts = contexts
        self.issue_width = issue_width
        self.reduce_slots = reduce_slots
        self.output_lanes = output_lanes
        self.cluster_block = cluster_block
        self.run_context: dict[str, Any] = {}
        self.operators: list[dict[str, Any]] = []
        self.prototypes: list[dict[str, Any]] = []
        self.prototype_ids: dict[tuple[Any, ...], int] = {}
        self.stream_operator: list[np.ndarray] = []
        self.stream_population: list[np.ndarray] = []
        self.stream_prototype: list[np.ndarray] = []
        self.skipped: Counter[str] = Counter()
        self.seen_allowlist: Counter[tuple[int, str, str, str, int]] = Counter()

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def bind_run_context(self, value: dict[str, Any]) -> None:
        self.run_context = dict(value)

    @staticmethod
    def _linear_rows(value: torch.Tensor, module: torch.nn.Linear, timestep: int) -> torch.Tensor:
        return value[timestep].eq(1).reshape(-1, module.in_features)

    @staticmethod
    def _conv_rows(value: torch.Tensor, module: torch.nn.Conv2d, timestep: int) -> torch.Tensor:
        current = value[timestep]
        if current.ndim == 3:
            current = current.unsqueeze(0)
        patches = F.unfold(
            current.to(torch.float32), kernel_size=module.kernel_size,
            dilation=module.dilation, padding=module.padding, stride=module.stride,
        ).round().to(torch.bool)
        batch, _all_sources, locations = patches.shape
        source_width = (module.in_channels // module.groups) * math.prod(module.kernel_size)
        return (
            patches.reshape(batch, module.groups, source_width, locations)
            .permute(0, 1, 3, 2)
            .reshape(-1, source_width)
        )

    @staticmethod
    def _bank_counts(bits: torch.Tensor) -> np.ndarray:
        counts = torch.stack(
            [bits[:, bank::16].sum(dim=1, dtype=torch.int16) for bank in range(16)],
            dim=1,
        )
        return counts.detach().cpu().numpy().astype(np.int16, copy=False)

    def _rows_for_timestep(
        self, value: torch.Tensor, module: torch.nn.Module, timestep: int,
    ) -> torch.Tensor:
        if isinstance(module, torch.nn.Linear):
            return self._linear_rows(value, module, timestep)
        if isinstance(module, torch.nn.Conv2d):
            return self._conv_rows(value, module, timestep)
        raise TypeError("M17 supports Linear and Conv2d only")

    def _clusters_for_timestep(
        self, value: torch.Tensor, module: torch.nn.Module, timestep: int,
    ) -> tuple[torch.Tensor, np.ndarray]:
        rows = self._rows_for_timestep(value, module, timestep)
        source_width = int(rows.shape[1])
        if isinstance(module, torch.nn.Linear):
            clusters = math.ceil(int(rows.shape[0]) / self.contexts)
            padding = clusters * self.contexts - int(rows.shape[0])
            if padding:
                rows = torch.cat((rows, torch.zeros(
                    (padding, source_width), dtype=torch.bool, device=rows.device,
                )), dim=0)
            contexts = np.full(clusters, self.contexts, dtype=np.uint8)
            if padding:
                contexts[-1] = self.contexts - padding
            return rows.reshape(clusters, self.contexts, source_width), contexts
        if not isinstance(module, torch.nn.Conv2d):
            raise TypeError("unsupported M17 operator")
        current = value[timestep]
        if current.ndim == 3:
            current = current.unsqueeze(0)
        batch, _channels, height, width = (int(item) for item in current.shape)
        out_h = (
            height + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1
        ) // module.stride[0] + 1
        out_w = (
            width + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
        ) // module.stride[1] + 1
        x_clusters = math.ceil(out_w / self.contexts)
        rows = rows.reshape(batch, module.groups, out_h, out_w, source_width)
        x_padding = x_clusters * self.contexts - out_w
        if x_padding:
            rows = torch.cat((rows, torch.zeros(
                (batch, module.groups, out_h, x_padding, source_width),
                dtype=torch.bool, device=rows.device,
            )), dim=3)
        clusters = rows.reshape(
            batch, module.groups, out_h, x_clusters, self.contexts, source_width,
        ).reshape(-1, self.contexts, source_width)
        contexts_one_row = np.full(x_clusters, self.contexts, dtype=np.uint8)
        if x_padding:
            contexts_one_row[-1] = self.contexts - x_padding
        contexts = np.tile(contexts_one_row, batch * module.groups * out_h)
        return clusters, contexts

    def _analyze_rows(
        self, value: torch.Tensor, module: torch.nn.Module, *, state_hash: dict[str, Any],
    ) -> dict[str, Any]:
        temporal_steps = int(value.shape[0])
        if isinstance(module, torch.nn.Linear):
            source_width = int(module.in_features)
            fanout = int(module.out_features)
        elif isinstance(module, torch.nn.Conv2d):
            source_width = int(module.in_channels // module.groups) * math.prod(module.kernel_size)
            fanout = int(module.out_channels // module.groups)
        else:
            raise TypeError("unsupported M17 operator")
        chunks = math.ceil(source_width / self.tile_bits)
        lane_tiles = math.ceil(fanout / self.output_lanes)
        first_rows, contexts_by_cluster = self._clusters_for_timestep(value, module, 0)
        population_clusters = int(first_rows.shape[0])
        row_count = int(contexts_by_cluster.sum(dtype=np.uint64))
        local_cost = np.zeros((population_clusters, temporal_steps), dtype=np.uint32)
        hybrid_cost = np.zeros((population_clusters, temporal_steps), dtype=np.uint32)
        local_work_by_t = np.zeros(temporal_steps, dtype=np.uint64)
        hybrid_work_by_t = np.zeros(temporal_steps, dtype=np.uint64)
        motion_rows_by_t = np.zeros(temporal_steps, dtype=np.uint64)
        positive_selected_by_t = np.zeros(temporal_steps, dtype=np.uint64)
        negative_selected_by_t = np.zeros(temporal_steps, dtype=np.uint64)
        motion_gate_histograms: list[Counter[int]] = [Counter() for _ in range(temporal_steps)]
        compact_local_by_t = np.zeros(temporal_steps, dtype=np.uint64)
        compact_hybrid_by_t = np.zeros(temporal_steps, dtype=np.uint64)
        previous_rows: torch.Tensor | None = None
        for timestep in range(temporal_steps):
            if timestep == 0:
                current_rows, current_contexts = first_rows, contexts_by_cluster
            else:
                current_rows, current_contexts = self._clusters_for_timestep(value, module, timestep)
            if (
                tuple(current_rows.shape) != (population_clusters, self.contexts, source_width)
                or not np.array_equal(current_contexts, contexts_by_cluster)
            ):
                raise ValueError("M17 row geometry changed across time")
            previous = torch.zeros_like(current_rows) if previous_rows is None else previous_rows
            transition = torch.logical_xor(current_rows, previous)
            positive = torch.logical_and(current_rows, torch.logical_not(previous))
            negative = torch.logical_and(torch.logical_not(current_rows), previous)
            choose_motion = (
                transition.sum(dim=2, dtype=torch.int64)
                < current_rows.sum(dim=2, dtype=torch.int64)
            )
            if timestep == 0:
                choose_motion.zero_()
            selected_rows = torch.where(choose_motion[:, :, None], transition, current_rows)
            selected_positive = torch.logical_and(positive, choose_motion[:, :, None])
            selected_negative = torch.logical_and(negative, choose_motion[:, :, None])
            local_work_by_t[timestep] = int(current_rows.sum(dtype=torch.int64).item()) * fanout
            hybrid_work_by_t[timestep] = int(selected_rows.sum(dtype=torch.int64).item()) * fanout
            motion_rows_by_t[timestep] = int(choose_motion.sum(dtype=torch.int64).item())
            positive_selected_by_t[timestep] = int(selected_positive.sum(dtype=torch.int64).item()) * fanout
            negative_selected_by_t[timestep] = int(selected_negative.sum(dtype=torch.int64).item()) * fanout
            gate_masks = sum(
                choose_motion[:, context].to(torch.uint8) << context
                for context in range(self.contexts)
            ).detach().cpu().numpy()
            gate_values, gate_counts = np.unique(gate_masks, return_counts=True)
            motion_gate_histograms[timestep].update({
                int(value): int(count) for value, count in zip(gate_values, gate_counts)
            })
            local_compact_all = np.zeros(population_clusters, dtype=np.uint32)
            hybrid_compact_all = np.zeros(population_clusters, dtype=np.uint32)
            state_hash["metadata"].update(
                np.asarray([timestep, population_clusters], dtype=np.uint64).tobytes()
            )
            state_hash["metadata"].update(contexts_by_cluster.tobytes(order="C"))
            state_hash["metadata"].update(gate_masks.tobytes(order="C"))
            for chunk in range(chunks):
                source_start = chunk * self.tile_bits
                source_stop = min(source_width, source_start + self.tile_bits)
                state_hash["metadata"].update(
                    np.asarray([chunk, source_stop - source_start], dtype=np.uint64).tobytes()
                )
                for cluster_start in range(0, population_clusters, self.cluster_block):
                    cluster_stop = min(population_clusters, cluster_start + self.cluster_block)
                    groups = cluster_stop - cluster_start
                    local_bits = current_rows[
                        cluster_start:cluster_stop, :, source_start:source_stop
                    ].reshape(groups * self.contexts, -1)
                    hybrid_bits = selected_rows[
                        cluster_start:cluster_stop, :, source_start:source_stop
                    ].reshape(groups * self.contexts, -1)
                    local_counts = self._bank_counts(local_bits).reshape(groups, self.contexts, 16)
                    hybrid_counts = self._bank_counts(hybrid_bits).reshape(groups, self.contexts, 16)
                    positive_counts = self._bank_counts(selected_positive[
                        cluster_start:cluster_stop, :, source_start:source_stop
                    ].reshape(groups * self.contexts, -1)).reshape(groups, self.contexts, 16)
                    negative_counts = self._bank_counts(selected_negative[
                        cluster_start:cluster_stop, :, source_start:source_stop
                    ].reshape(groups * self.contexts, -1)).reshape(groups, self.contexts, 16)
                    local_compact_all[cluster_start:cluster_stop] += compact_issue_cycles_batched(
                        local_counts, self.reduce_slots
                    )
                    hybrid_compact_all[cluster_start:cluster_stop] += compact_issue_cycles_batched(
                        hybrid_counts, self.reduce_slots
                    )
                    state_hash["local"].update(local_counts.tobytes(order="C"))
                    state_hash["positive"].update(positive_counts.tobytes(order="C"))
                    state_hash["negative"].update(negative_counts.tobytes(order="C"))
            local_cost[:, timestep] = local_compact_all + 2 * chunks
            hybrid_cost[:, timestep] = hybrid_compact_all + 2 * chunks
            compact_local_by_t[timestep] = int(local_compact_all.sum(dtype=np.uint64))
            compact_hybrid_by_t[timestep] = int(hybrid_compact_all.sum(dtype=np.uint64))
            previous_rows = current_rows

        descriptor_cycles = int(temporal_steps * contexts_by_cluster.sum(dtype=np.uint64) * chunks)
        output_cycles = int(temporal_steps * row_count * lane_tiles)
        control_cycles = int(temporal_steps * population_clusters * 2 * chunks * lane_tiles)
        lines = {}
        for line, work, compact, costs in (
            ("local", local_work_by_t, compact_local_by_t, local_cost),
            ("hybrid", hybrid_work_by_t, compact_hybrid_by_t, hybrid_cost),
        ):
            compact_cycles = int(compact.sum(dtype=np.uint64)) * lane_tiles
            totals = {
                "selected_product_terms": int(work.sum(dtype=np.uint64)),
                "selected_product_terms_by_t": [int(item) for item in work],
                "descriptor_load_cycles": descriptor_cycles,
                "compact_issue_cycles": compact_cycles,
                "chunk_control_cycles": control_cycles,
                "output_cycles": output_cycles,
                "m4_wall_cycles": descriptor_cycles + compact_cycles + control_cycles + output_cycles,
                "lane_compute_cycle_histogram_by_t": [],
            }
            for timestep in range(temporal_steps):
                values, counts = np.unique(costs[:, timestep], return_counts=True)
                totals["lane_compute_cycle_histogram_by_t"].append({
                    str(int(value)): int(count) for value, count in zip(values, counts)
                })
            lines[line] = totals
        return {
            "temporal_steps": temporal_steps,
            "source_width": source_width,
            "chunks": chunks,
            "fanout": fanout,
            "lane_tiles": lane_tiles,
            "row_count": row_count,
            "population_clusters": population_clusters,
            "contexts_by_cluster": contexts_by_cluster,
            "motion_selected_rows_by_t": [int(item) for item in motion_rows_by_t],
            "positive_selected_product_terms_by_t": [int(item) for item in positive_selected_by_t],
            "negative_selected_product_terms_by_t": [int(item) for item in negative_selected_by_t],
            "motion_gate_mask_histogram_by_t": [
                {str(key): value for key, value in sorted(histogram.items())}
                for histogram in motion_gate_histograms
            ],
            "local_cost": local_cost,
            "hybrid_cost": hybrid_cost,
            "lines": lines,
        }

    def _encode_prototypes(self, analysis: dict[str, Any]) -> np.ndarray:
        contexts = analysis["contexts_by_cluster"].astype(np.uint32)[:, None]
        feature = np.concatenate((contexts, analysis["local_cost"], analysis["hybrid_cost"]), axis=1)
        unique, inverse = np.unique(feature, axis=0, return_inverse=True)
        local_ids = np.zeros(unique.shape[0], dtype=np.uint32)
        steps = int(analysis["temporal_steps"])
        for index, row in enumerate(unique):
            key = (
                int(row[0]), int(analysis["chunks"]), int(analysis["fanout"]),
                tuple(int(item) for item in row[1:1 + steps]),
                tuple(int(item) for item in row[1 + steps:1 + 2 * steps]),
            )
            prototype_id = self.prototype_ids.get(key)
            if prototype_id is None:
                prototype_id = len(self.prototypes)
                self.prototype_ids[key] = prototype_id
                payload = {
                    "prototype_id": prototype_id,
                    "contexts": key[0], "chunks": key[1], "fanout": key[2],
                    "lane_tiles": math.ceil(key[2] / self.output_lanes),
                    "descriptor_cycles_by_t": [key[0] * key[1]] * steps,
                    "local_lane_compute_cycles_by_t": list(key[3]),
                    "hybrid_lane_compute_cycles_by_t": list(key[4]),
                }
                payload["cost_source_sha256"] = _canonical_sha256(payload)
                self.prototypes.append(payload)
            local_ids[index] = prototype_id
        return local_ids[inverse]

    def record_operator(
        self, module: torch.nn.Module, input_tensor: torch.Tensor, *, name: str,
        sample_id: int, sample_key: str, sequence_key: str, operator_call_index: int,
        temporal_steps: int = 10,
    ) -> None:
        call_key = (
            int(sample_id), str(sample_key), str(sequence_key), str(name), int(operator_call_index),
        )
        if name not in self.allowed_names:
            return
        if call_key not in self.allowed_call_keys:
            raise RuntimeError("observed an uncontracted call of a direct-M4 producer: " + repr(call_key))
        self.seen_allowlist[call_key] += 1
        if self.seen_allowlist[call_key] != 1:
            raise RuntimeError("direct-M4 call identity was captured more than once: " + repr(call_key))
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            self.skipped["UNSUPPORTED_OPERATOR"] += 1
            return
        if isinstance(module, torch.nn.Conv2d) and module.padding_mode != "zeros":
            self.skipped["NONZERO_PADDING_MODE"] += 1
            return
        value = input_tensor.detach()
        if value.ndim < 2 or int(value.shape[0]) != temporal_steps:
            self.skipped["TEMPORAL_AXIS_UNQUALIFIED"] += 1
            return
        if not bool(torch.logical_or(value.eq(0), value.eq(1)).all().item()):
            self.skipped["NON_BINARY_BYPASS"] += 1
            return
        operator_index = len(self.operators)
        identity_header = (
            f"{sample_id}|{sample_key}|{sequence_key}|{name}|{operator_call_index}\n"
        ).encode("utf-8")
        state_hash = {
            component: hashlib.sha256(identity_header + component.encode("ascii") + b"\n")
            for component in ("metadata", "local", "positive", "negative")
        }
        analysis = self._analyze_rows(value, module, state_hash=state_hash)
        prototype_ids = self._encode_prototypes(analysis)
        populations = np.arange(analysis["population_clusters"], dtype=np.uint64)
        self.stream_operator.append(np.full(populations.shape, operator_index, dtype=np.uint16))
        self.stream_population.append(populations)
        self.stream_prototype.append(prototype_ids.astype(np.uint32, copy=False))
        self.operators.append({
            "operator_index": operator_index,
            "sample_id": int(sample_id), "sample_key": sample_key,
            "sequence_key": sequence_key, "name": name,
            "operator": module.__class__.__name__,
            "operator_call_index": int(operator_call_index),
            "temporal_steps": analysis["temporal_steps"],
            "source_width": analysis["source_width"], "chunks": analysis["chunks"],
            "fanout": analysis["fanout"], "lane_tiles": analysis["lane_tiles"],
            "row_count": analysis["row_count"],
            "population_clusters": analysis["population_clusters"],
            "motion_selected_rows_by_t": analysis["motion_selected_rows_by_t"],
            "positive_selected_product_terms_by_t": analysis["positive_selected_product_terms_by_t"],
            "negative_selected_product_terms_by_t": analysis["negative_selected_product_terms_by_t"],
            "motion_gate_mask_histogram_by_t": analysis["motion_gate_mask_histogram_by_t"],
            "ordered_scheduler_sufficient_statistics_sha256": hashlib.sha256(b"".join(
                bytes.fromhex(state_hash[component].hexdigest())
                for component in ("metadata", "local", "positive", "negative")
            )).hexdigest(),
            "lines": analysis["lines"],
        })

    def close(self) -> None:
        missing = sorted(self.allowed_call_keys - set(self.seen_allowlist))
        if missing:
            raise RuntimeError("direct-M4 contracted calls were not observed: " + repr(missing))
        if self.skipped:
            raise RuntimeError("direct-M4 full-spatial oracle skipped calls: " + repr(dict(self.skipped)))
        if not self.operators:
            raise RuntimeError("full-spatial oracle captured no operators")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prototypes_path = self.output_dir / "prototypes.json"
        prototypes_path.write_text(
            json.dumps(self.prototypes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stream_path = self.output_dir / "ordered_stream.npz"
        np.savez_compressed(
            stream_path,
            operator_index=np.concatenate(self.stream_operator),
            population_cluster_id=np.concatenate(self.stream_population),
            prototype_id=np.concatenate(self.stream_prototype),
        )
        payload = {
            "schema": "h67_full_spatial_adjacent_c4_oracle_v2",
            "status": "PASS_EXACT_FULL_SPATIAL_C4_SUFFICIENT_STATISTICS_NOT_SYSTEM_SPEEDUP",
            "architecture": {
                "tile_bits": self.tile_bits, "contexts": self.contexts,
                "issue_width": self.issue_width, "reduce_slots": self.reduce_slots,
                "output_lanes": self.output_lanes,
                "scheduler": "deterministic_bank_first_context_first",
            },
            "run_context": self.run_context,
            "allowed_calls": self.allowed_calls,
            "operators": self.operators,
            "prototypes": len(self.prototypes),
            "ordered_clusters": int(sum(len(item) for item in self.stream_population)),
            "files": {
                "prototypes.json": {"sha256": file_sha256(prototypes_path), "bytes": prototypes_path.stat().st_size},
                "ordered_stream.npz": {"sha256": file_sha256(stream_path), "bytes": stream_path.stat().st_size},
            },
            "claim_boundary": (
                "Exact same-sample full-spatial Local/row-selected-Hybrid adjacent-C4 source and "
                "M4 compact-scheduler sufficient statistics. Not ATLIF overlap, SRAM/DRAM timing, "
                "RTL equivalence, system speedup, energy, or PPA."
            ),
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
