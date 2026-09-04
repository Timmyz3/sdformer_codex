#!/usr/bin/env python3
"""Extract deterministic real 256-bit Local/Motion tile descriptors.

The writer samples row/chunk identities and emits all temporal steps for each
identity consecutively.  It records both whole-row selector counts and the
chunk bitmap.  This makes the distinction between row-level selection and
tile-level execution explicit; it does not pretend that a single-slot RTL
state represents every row in a full layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _pair_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    count = min(total, count)
    return [min(total - 1, ((2 * index + 1) * total) // (2 * count)) for index in range(count)]


def _axis_indices(total: int, count: int) -> list[int]:
    """Boundary-inclusive deterministic samples on one axis."""
    if total <= 0 or count <= 0:
        return []
    count = min(total, count)
    if count == 1:
        return [total // 2]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def _spatial_grid(height: int, width: int, count: int) -> list[tuple[int, int]]:
    """Farthest-point subset of a boundary-inclusive Cartesian grid."""
    total = height * width
    if total <= 0 or count <= 0:
        return []
    count = min(total, count)
    if count == 1:
        return [(height // 2, width // 2)]
    y_count = min(height, max(2, round(math.sqrt(count * height / width))))
    x_count = min(width, max(2, math.ceil(count / y_count)))
    while y_count * x_count < count:
        if x_count < width and (x_count / max(1, y_count)) <= (width / height):
            x_count += 1
        elif y_count < height:
            y_count += 1
        elif x_count < width:
            x_count += 1
        else:
            break
    candidates = [
        (y, x)
        for y in _axis_indices(height, y_count)
        for x in _axis_indices(width, x_count)
    ]
    candidates = list(dict.fromkeys(candidates))
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    selected = [point for point in dict.fromkeys(corners) if point in candidates][:count]
    remaining = [point for point in candidates if point not in selected]
    while len(selected) < count and remaining:
        def distance(point: tuple[int, int]) -> tuple[float, int, int]:
            y, x = point
            nearest = min(
                ((y - sy) / max(1, height - 1)) ** 2
                + ((x - sx) / max(1, width - 1)) ** 2
                for sy, sx in selected
            )
            return nearest, -y, -x

        point = max(remaining, key=distance)
        selected.append(point)
        remaining.remove(point)
    return sorted(selected)


def _conv_row_indices(
    batches: int, groups: int, height: int, width: int, count: int
) -> list[int]:
    total = batches * groups * height * width
    if count >= total:
        return list(range(total))
    outer_total = batches * groups
    outer_count = min(outer_total, count)
    outer_indices = _pair_indices(outer_total, outer_count)
    base, remainder = divmod(count, outer_count)
    result = []
    for rank, outer in enumerate(outer_indices):
        quota = base + (1 if rank < remainder else 0)
        for y, x in _spatial_grid(height, width, quota):
            result.append((outer * height + y) * width + x)
    if len(result) != min(total, count) or len(set(result)) != len(result):
        raise RuntimeError("conv spatial sampler cardinality/uniqueness failure")
    return result


def _allocate_stratum_samples(populations: dict[str, int], budget: int) -> dict[str, int]:
    """Allocate a bounded cluster budget while retaining rare boundary strata."""
    active = [name for name in sorted(populations) if populations[name] > 0]
    allocation = {name: 0 for name in active}
    if not active or budget <= 0:
        return allocation
    if budget >= len(active):
        for name in active:
            allocation[name] = 1
    else:
        for name in sorted(active, key=lambda item: (-populations[item], item))[:budget]:
            allocation[name] = 1
    # At P32 and above, capture all four spatial corners when a single outer
    # identity is present.  The corresponding inverse-probability weight keeps
    # these coverage sentinels from being treated as representative interior.
    if budget >= 8:
        for name in active:
            if name.startswith("corner"):
                while allocation[name] < min(4, populations[name]) and sum(allocation.values()) < budget:
                    allocation[name] += 1
    while sum(allocation.values()) < budget:
        eligible = [name for name in active if allocation[name] < populations[name]]
        if not eligible:
            break
        # Greedily equalize sampling fractions N/n without losing determinism.
        chosen = max(
            eligible,
            key=lambda name: (populations[name] / (allocation[name] + 1), populations[name], name),
        )
        allocation[chosen] += 1
    return allocation


def _sample_cluster_strata(
    candidates: dict[str, list[dict[str, Any]]], budget: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    populations = {name: len(items) for name, items in candidates.items()}
    allocation = _allocate_stratum_samples(populations, min(budget, sum(populations.values())))
    selected = []
    for stratum in sorted(candidates):
        items = candidates[stratum]
        sample_count = allocation.get(stratum, 0)
        for index in _pair_indices(len(items), sample_count):
            item = dict(items[index])
            item.update({
                "sampling_stratum": stratum,
                "stratum_population_clusters": len(items),
                "stratum_sample_clusters": sample_count,
                "cluster_inverse_probability_weight": len(items) / sample_count,
            })
            selected.append(item)
    return sorted(selected, key=lambda item: item["sample_cluster_id"]), {
        "population_clusters_by_stratum": dict(sorted(populations.items())),
        "sample_clusters_by_stratum": dict(sorted(allocation.items())),
    }


def _linear_cluster_samples(
    row_count: int, row_budget: int, contexts: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    cluster_width = min(max(1, contexts), max(1, row_budget))
    candidates = []
    for start in range(0, row_count, cluster_width):
        candidates.append({
            "sample_cluster_id": start // cluster_width,
            "rows": list(range(start, min(start + cluster_width, row_count))),
        })
    cluster_budget = min(len(candidates), math.ceil(min(row_count, row_budget) / cluster_width))
    return _sample_cluster_strata({"flat": candidates}, cluster_budget)


def _conv_cluster_samples(
    module: torch.nn.Conv2d, *, batches: int, height: int, width: int,
    out_h: int, out_w: int, row_budget: int, contexts: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    cluster_width = min(max(1, contexts), max(1, row_budget))
    candidates: dict[str, list[dict[str, Any]]] = {}
    cluster_id = 0
    for batch in range(batches):
        for group in range(module.groups):
            for output_y in range(out_h):
                for output_x_start in range(0, out_w, cluster_width):
                    output_x_stop = min(output_x_start + cluster_width, out_w)
                    rows = [
                        ((batch * module.groups + group) * out_h + output_y) * out_w + output_x
                        for output_x in range(output_x_start, output_x_stop)
                    ]
                    y_edge = output_y in {0, out_h - 1}
                    x_edge = output_x_start == 0 or output_x_stop == out_w
                    corner = y_edge and x_edge
                    halo = False
                    for output_x in range(output_x_start, output_x_stop):
                        start_y = output_y * module.stride[0] - module.padding[0]
                        start_x = output_x * module.stride[1] - module.padding[1]
                        end_y = start_y + module.dilation[0] * (module.kernel_size[0] - 1)
                        end_x = start_x + module.dilation[1] * (module.kernel_size[1] - 1)
                        halo = halo or start_y < 0 or start_x < 0 or end_y >= height or end_x >= width
                    prefix = "corner" if corner else ("edge" if y_edge or x_edge else "interior")
                    stratum = prefix + ("_padding_halo" if halo else "")
                    candidates.setdefault(stratum, []).append({
                        "sample_cluster_id": cluster_id,
                        "rows": rows,
                        "batch": batch,
                        "group": group,
                        "output_y": output_y,
                        "output_x_start": output_x_start,
                        "output_x_stop": output_x_stop,
                        "padding_halo": halo,
                    })
                    cluster_id += 1
    total_rows = batches * module.groups * out_h * out_w
    cluster_budget = min(cluster_id, math.ceil(min(total_rows, row_budget) / cluster_width))
    return _sample_cluster_strata(candidates, cluster_budget)


def _pack(bits: torch.Tensor, tile_bits: int) -> np.ndarray:
    values = bits.detach().to(torch.uint8).cpu().numpy()
    padded = np.zeros(tile_bits, dtype=np.uint8)
    padded[: values.size] = values
    return np.packbits(padded, bitorder="little")


def _conv_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class DualLineTileTraceWriter:
    def __init__(
        self, output_dir: Path, *, tile_bits: int = 256, pairs_per_call: int = 4,
        cluster_contexts: int = 4,
    ) -> None:
        if tile_bits <= 0 or tile_bits % 8 or pairs_per_call <= 0 or cluster_contexts <= 0:
            raise ValueError("tile_bits, pairs_per_call, and cluster_contexts must be positive")
        self.output_dir = Path(output_dir)
        self.tile_bits = int(tile_bits)
        self.pairs_per_call = int(pairs_per_call)
        self.cluster_contexts = int(cluster_contexts)
        self.records: list[dict[str, Any]] = []
        self.current: list[np.ndarray] = []
        self.previous: list[np.ndarray] = []
        self.run_context: dict[str, Any] = {}
        self.skipped: dict[str, int] = {}
        self.conv_sampling: list[dict[str, Any]] = []
        self.cluster_sampling: list[dict[str, Any]] = []

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def bind_run_context(self, value: dict[str, Any]) -> None:
        self.run_context = dict(value)

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def _append_temporal_pair(
        self,
        *,
        name: str,
        operator: str,
        sample_id: int,
        sample_key: str,
        sequence_key: str,
        operator_call_index: int,
        row_id: int,
        chunk_index: int,
        source_width: int,
        fanout: int,
        weight_group: int,
        sampling: dict[str, Any],
        rows: list[torch.Tensor],
    ) -> None:
        start = chunk_index * self.tile_bits
        stop = min(start + self.tile_bits, source_width)
        valid_bits = stop - start
        for timestep, current_row in enumerate(rows):
            previous_row = rows[timestep - 1] if timestep else torch.zeros_like(current_row)
            current_count = int(current_row.sum().item())
            transition_count = int(torch.logical_xor(current_row, previous_row).sum().item())
            use_motion = timestep > 0 and transition_count < current_count
            current_tile = current_row[start:stop]
            previous_tile = previous_row[start:stop]
            self.current.append(_pack(current_tile, self.tile_bits))
            self.previous.append(_pack(previous_tile, self.tile_bits))
            self.records.append({
                "record_id": len(self.records),
                "sample_id": sample_id,
                "sample_key": sample_key,
                "sequence_key": sequence_key,
                "name": name,
                "operator": operator,
                "operator_call_index": operator_call_index,
                "row_id": row_id,
                "chunk_index": chunk_index,
                "chunks_per_row": math.ceil(source_width / self.tile_bits),
                "source_base": start,
                "source_width": source_width,
                "valid_bits": valid_bits,
                "output_channel_fanout": fanout,
                "weight_group": weight_group,
                "sample_cluster_id": sampling["sample_cluster_id"],
                "sample_cluster_lane": sampling["sample_cluster_lane"],
                "sample_cluster_rows": sampling["sample_cluster_rows"],
                "sampling_stratum": sampling["sampling_stratum"],
                "stratum_population_clusters": sampling["stratum_population_clusters"],
                "stratum_sample_clusters": sampling["stratum_sample_clusters"],
                "cluster_inverse_probability_weight": sampling["cluster_inverse_probability_weight"],
                "output_lane_tile_count_96": math.ceil(fanout / 96),
                "temporal_step": timestep,
                "state_valid": timestep > 0,
                "row_current_count": current_count,
                "row_transition_count": transition_count,
                "row_use_motion": use_motion,
                "tile_current_count": int(current_tile.sum().item()),
                "tile_positive_count": int(torch.logical_and(current_tile, ~previous_tile).sum().item()),
                "tile_negative_count": int(torch.logical_and(~current_tile, previous_tile).sum().item()),
                "schedule_contract": "ROW_CHUNK_IDENTITY_TEMPORAL_INNER_REQUIRES_STATE_TABLE",
            })

    def _record_linear(self, module: torch.nn.Linear, value: torch.Tensor, **context: Any) -> None:
        if value.shape[-1] != module.in_features:
            self._skip("LINEAR_FEATURE_MISMATCH")
            return
        temporal = int(value.shape[0])
        rows = value.eq(1).reshape(temporal, -1, module.in_features)
        row_count = int(rows.shape[1])
        chunks = math.ceil(module.in_features / self.tile_bits)
        sampled_clusters, sampling_summary = _linear_cluster_samples(
            row_count, self.pairs_per_call, self.cluster_contexts
        )
        self.cluster_sampling.append({
            "name": context["name"],
            "operator_call_index": context["operator_call_index"],
            "operator": "Linear",
            "population_rows": row_count,
            "sampled_rows": sum(len(item["rows"]) for item in sampled_clusters),
            "population_clusters": math.ceil(row_count / min(self.cluster_contexts, self.pairs_per_call)),
            "sampled_clusters": len(sampled_clusters),
            **sampling_summary,
        })
        for cluster in sampled_clusters:
            for cluster_lane, row_id in enumerate(cluster["rows"]):
                temporal_rows = [rows[timestep, row_id] for timestep in range(temporal)]
                sampling = {
                    **cluster,
                    "sample_cluster_lane": cluster_lane,
                    "sample_cluster_rows": len(cluster["rows"]),
                }
                for chunk_index in range(chunks):
                    self._append_temporal_pair(
                        operator="Linear", row_id=row_id, chunk_index=chunk_index,
                        source_width=module.in_features, fanout=module.out_features,
                        weight_group=0, sampling=sampling, rows=temporal_rows, **context,
                    )

    def _conv_row(
        self,
        value: torch.Tensor,
        *,
        timestep: int,
        batch: int,
        group: int,
        output_y: int,
        output_x: int,
        module: torch.nn.Conv2d,
    ) -> torch.Tensor:
        channels = module.in_channels // module.groups
        kernel_h, kernel_w = module.kernel_size
        source_width = channels * kernel_h * kernel_w
        source = torch.arange(source_width, device=value.device)
        channel = source // (kernel_h * kernel_w)
        rem = source % (kernel_h * kernel_w)
        kernel_y = rem // kernel_w
        kernel_x = rem % kernel_w
        input_y = output_y * module.stride[0] - module.padding[0] + kernel_y * module.dilation[0]
        input_x = output_x * module.stride[1] - module.padding[1] + kernel_x * module.dilation[1]
        valid = (input_y >= 0) & (input_y < value.shape[-2]) & (input_x >= 0) & (input_x < value.shape[-1])
        result = torch.zeros(source_width, dtype=torch.bool, device=value.device)
        absolute_channel = group * channels + channel[valid]
        result[valid] = value[timestep, batch, absolute_channel, input_y[valid], input_x[valid]].eq(1)
        return result

    def _record_conv2d(self, module: torch.nn.Conv2d, value: torch.Tensor, **context: Any) -> None:
        if value.ndim == 4:
            value = value.unsqueeze(1)
        if value.ndim != 5 or value.shape[2] != module.in_channels:
            self._skip("CONV2D_SHAPE_MISMATCH")
            return
        temporal, batches, _channels, height, width = (int(item) for item in value.shape)
        out_h = _conv_output_size(height, module.kernel_size[0], module.stride[0], module.padding[0], module.dilation[0])
        out_w = _conv_output_size(width, module.kernel_size[1], module.stride[1], module.padding[1], module.dilation[1])
        source_width = (module.in_channels // module.groups) * math.prod(module.kernel_size)
        chunks = math.ceil(source_width / self.tile_bits)
        row_count = batches * module.groups * out_h * out_w
        sampled_clusters, sampling_summary = _conv_cluster_samples(
            module, batches=batches, height=height, width=width,
            out_h=out_h, out_w=out_w, row_budget=self.pairs_per_call,
            contexts=self.cluster_contexts,
        )
        sampled_rows = [row_id for cluster in sampled_clusters for row_id in cluster["rows"]]
        sampled_coordinates = []
        for row_id in sampled_rows:
            index = row_id
            output_x = index % out_w
            index //= out_w
            output_y = index % out_h
            index //= out_h
            group = index % module.groups
            batch = index // module.groups
            start_y = output_y * module.stride[0] - module.padding[0]
            start_x = output_x * module.stride[1] - module.padding[1]
            end_y = start_y + module.dilation[0] * (module.kernel_size[0] - 1)
            end_x = start_x + module.dilation[1] * (module.kernel_size[1] - 1)
            sampled_coordinates.append({
                "row_id": row_id,
                "batch": batch,
                "group": group,
                "y": output_y,
                "x": output_x,
                "padding_halo": start_y < 0 or start_x < 0 or end_y >= height or end_x >= width,
            })
        self.conv_sampling.append({
            "name": context["name"],
            "operator_call_index": context["operator_call_index"],
            "total_rows": row_count,
            "output_height": out_h,
            "output_width": out_w,
            "sampled_rows": len(sampled_rows),
            "distinct_batches": len({item["batch"] for item in sampled_coordinates}),
            "distinct_groups": len({item["group"] for item in sampled_coordinates}),
            "distinct_y": len({item["y"] for item in sampled_coordinates}),
            "distinct_x": len({item["x"] for item in sampled_coordinates}),
            "y_edge_rows": sum(item["y"] in {0, out_h - 1} for item in sampled_coordinates),
            "x_edge_rows": sum(item["x"] in {0, out_w - 1} for item in sampled_coordinates),
            "corner_rows": sum(
                item["y"] in {0, out_h - 1} and item["x"] in {0, out_w - 1}
                for item in sampled_coordinates
            ),
            "padding_halo_rows": sum(item["padding_halo"] for item in sampled_coordinates),
            "interior_rows": sum(not item["padding_halo"] for item in sampled_coordinates),
        })
        self.cluster_sampling.append({
            "name": context["name"],
            "operator_call_index": context["operator_call_index"],
            "operator": "Conv2d",
            "population_rows": row_count,
            "sampled_rows": len(sampled_rows),
            "population_clusters": sum(sampling_summary["population_clusters_by_stratum"].values()),
            "sampled_clusters": len(sampled_clusters),
            **sampling_summary,
        })
        for cluster in sampled_clusters:
            for cluster_lane, row_id in enumerate(cluster["rows"]):
                index = row_id
                output_x = index % out_w
                index //= out_w
                output_y = index % out_h
                index //= out_h
                group = index % module.groups
                batch = index // module.groups
                temporal_rows = [
                    self._conv_row(
                        value, timestep=timestep, batch=batch, group=group,
                        output_y=output_y, output_x=output_x, module=module,
                    )
                    for timestep in range(temporal)
                ]
                sampling = {
                    **cluster,
                    "sample_cluster_lane": cluster_lane,
                    "sample_cluster_rows": len(cluster["rows"]),
                }
                for chunk_index in range(chunks):
                    self._append_temporal_pair(
                        operator="Conv2d", row_id=row_id, chunk_index=chunk_index,
                        source_width=source_width, fanout=module.out_channels // module.groups,
                        weight_group=group, sampling=sampling, rows=temporal_rows, **context,
                    )

    def record_operator(
        self,
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        *,
        name: str,
        sample_id: int,
        sample_key: str,
        sequence_key: str,
        operator_call_index: int,
        temporal_steps: int = 10,
    ) -> None:
        value = input_tensor.detach()
        if value.ndim < 2 or int(value.shape[0]) != temporal_steps:
            self._skip("TEMPORAL_AXIS_UNQUALIFIED")
            return
        if not bool(torch.logical_or(value.eq(0), value.eq(1)).all().item()):
            self._skip("NON_BINARY_BYPASS")
            return
        context = {
            "name": name,
            "sample_id": int(sample_id),
            "sample_key": sample_key,
            "sequence_key": sequence_key,
            "operator_call_index": int(operator_call_index),
        }
        if isinstance(module, torch.nn.Linear):
            self._record_linear(module, value, **context)
        elif isinstance(module, torch.nn.Conv2d):
            self._record_conv2d(module, value, **context)
        else:
            self._skip("UNSUPPORTED_OPERATOR")

    def close(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if len(self.current) != len(self.records) or len(self.previous) != len(self.records):
            raise RuntimeError("tile bitmap/metadata cardinality mismatch")
        current = np.stack(self.current) if self.current else np.zeros((0, self.tile_bits // 8), dtype=np.uint8)
        previous = np.stack(self.previous) if self.previous else np.zeros_like(current)
        npz_path = self.output_dir / "packed_tiles.npz"
        np.savez_compressed(npz_path, packed_current_bits=current, packed_previous_bits=previous)
        csv_path = self.output_dir / "tile_records.csv"
        if self.records:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows(self.records)
        else:
            csv_path.write_text("", encoding="utf-8")
        hashes = {}
        for path in (npz_path, csv_path):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "schema": "dual_line_real_tile_trace_v2",
            "status": "PASS_REAL_BITMAPS_STRATIFIED_ADJACENT_C4_NOT_ACC32_ORACLE",
            "tile_bits": self.tile_bits,
            "pairs_per_operator_call": self.pairs_per_call,
            "cluster_contexts": self.cluster_contexts,
            "records": len(self.records),
            "row_chunk_identities": len({
                (row["sample_id"], row["name"], row["operator_call_index"], row["row_id"], row["chunk_index"])
                for row in self.records
            }),
            "operators": len({row["name"] for row in self.records}),
            "motion_records": sum(bool(row["row_use_motion"]) for row in self.records),
            "skipped_calls": self.skipped,
            "schedule_contract": "temporal-inner samples require a state table keyed by row/chunk; one-slot RTL is only a tile miter",
            "sampling_contract": (
                "Real adjacent C4 row clusters; Conv2d clusters are stratified by "
                "corner/edge/interior and padding-halo with explicit population/sample weights"
            ),
            "conv_sampling": self.conv_sampling,
            "cluster_sampling": self.cluster_sampling,
            "claim_boundary": "real checkpoint bitmap descriptors; weights and Acc32 oracle are not included",
            "sha256": hashes,
            "run_context": self.run_context,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
