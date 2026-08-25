#!/usr/bin/env python3
"""Exact streaming shift-compensated residual opportunity census (M28A).

This producer is deliberately an opportunity census, not a performance model.
For a binary T10 Conv2d input it searches an output-space displacement per
``(time,batch,group,output-tile)``.  A displacement ``q`` maps to the exact
input displacement ``stride*q``.  Rows whose shifted previous output is outside
the output extent fall back to the unshifted Local expression.

Only aggregate histograms and a per-call CSV ledger are written.  Activations,
residual bitmaps, and tile-selection maps are never persisted.
"""

import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import string
from collections import Counter
from pathlib import Path

import torch


PASS_STATUS = "PASS_EXACT_STREAMING_T10_SHIFT_RESIDUAL_CENSUS"
EXACT_STATUS = "PASS_EXACT_SHIFT_RESIDUAL_CONSERVATION"
REJECTION_STATUSES = {
    "NON_BINARY_BYPASS",
    "TEMPORAL_AXIS_UNQUALIFIED",
    "UNSUPPORTED_OPERATOR",
    "UNSUPPORTED_PADDING",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair(value, field):
    if isinstance(value, int):
        pair = (int(value), int(value))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError("{} must be an integer or integer pair".format(field))
    if pair[0] < 0 or pair[1] < 0:
        raise ValueError("{} must be nonnegative".format(field))
    return pair


def _conv_output_extent(size, kernel, stride, padding, dilation):
    numerator = int(size) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1
    result = numerator // int(stride) + 1
    if result <= 0:
        raise ValueError("Conv2d output extent is nonpositive")
    return int(result)


def canonical_shift_candidates(radius, candidate_shifts=None):
    """Return a content-independent search order used as the tie-break order."""
    radius = int(radius)
    if radius < 0:
        raise ValueError("shift radius must be nonnegative")
    if candidate_shifts is None:
        candidates = [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
    else:
        candidates = []
        for item in candidate_shifts:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("candidate shift must be an integer pair")
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in item):
                raise ValueError("candidate shift must be an integer pair")
            dy, dx = int(item[0]), int(item[1])
            if abs(dy) > radius or abs(dx) > radius:
                raise ValueError("candidate shift exceeds configured radius")
            candidates.append((dy, dx))
    if not candidates or (0, 0) not in candidates:
        raise ValueError("candidate shifts must include (0,0)")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate shifts must be unique")
    # Zero displacement wins equal-cost choices.  Remaining choices are stable
    # by Manhattan distance and then signed lexicographic order.
    return tuple(sorted(candidates, key=lambda q: (q != (0, 0), abs(q[0]) + abs(q[1]), q[0], q[1])))


def _conv_geometry(module, value):
    if not isinstance(module, torch.nn.Conv2d):
        raise TypeError("exact shift census geometry requires Conv2d")
    if str(getattr(module, "padding_mode", "zeros")) != "zeros":
        raise ValueError("exact shift census requires padding_mode='zeros'")
    if isinstance(module.padding, str):
        raise ValueError("symbolic Conv2d padding is not admitted")
    if value.ndim == 4:
        value = value.unsqueeze(1)
    if value.ndim != 5 or int(value.shape[0]) != 10:
        raise ValueError("Conv2d input must be [T,B,C,H,W] or [T,C,H,W] with T=10")
    _, batches, channels, height, width = [int(item) for item in value.shape]
    if channels != int(module.in_channels):
        raise ValueError("Conv2d input channel mismatch")
    kernel_h, kernel_w = _pair(module.kernel_size, "kernel_size")
    stride_h, stride_w = _pair(module.stride, "stride")
    padding_h, padding_w = _pair(module.padding, "padding")
    dilation_h, dilation_w = _pair(module.dilation, "dilation")
    if not all(value > 0 for value in (kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w)):
        raise ValueError("kernel, stride, and dilation must be positive")
    groups = int(module.groups)
    if groups <= 0 or channels % groups or int(module.out_channels) % groups:
        raise ValueError("Conv2d group geometry is invalid")
    output_h = _conv_output_extent(height, kernel_h, stride_h, padding_h, dilation_h)
    output_w = _conv_output_extent(width, kernel_w, stride_w, padding_w, dilation_w)
    return value, {
        "batches": batches,
        "channels": channels,
        "height": height,
        "width": width,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "padding_h": padding_h,
        "padding_w": padding_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
        "groups": groups,
        "channels_per_group": channels // groups,
        "fanout": int(module.out_channels) // groups,
        "output_h": output_h,
        "output_w": output_w,
        "source_width": (channels // groups) * kernel_h * kernel_w,
    }


def _tile_positions(geometry, y_start, y_stop, x_start, x_stop, device):
    if not (
        0 <= int(y_start) < int(y_stop) <= geometry["output_h"]
        and 0 <= int(x_start) < int(x_stop) <= geometry["output_w"]
    ):
        raise ValueError("output tile is outside Conv2d output extent")
    ys = torch.arange(int(y_start), int(y_stop), dtype=torch.int64, device=device)
    xs = torch.arange(int(x_start), int(x_stop), dtype=torch.int64, device=device)
    grid_y = ys.reshape(-1, 1).expand(-1, xs.numel()).reshape(-1)
    grid_x = xs.reshape(1, -1).expand(ys.numel(), -1).reshape(-1)
    return grid_y, grid_x


def _gather_source_chunk(value_t, geometry, batch, group, output_y, output_x, shift, source_start, source_stop):
    """Gather zero-padded logical Conv2d sources for one output tile."""
    dy, dx = int(shift[0]), int(shift[1])
    source = torch.arange(int(source_start), int(source_stop), dtype=torch.int64, device=value_t.device)
    kernel_area = geometry["kernel_h"] * geometry["kernel_w"]
    local_channel = source // kernel_area
    kernel_offset = source % kernel_area
    kernel_y = kernel_offset // geometry["kernel_w"]
    kernel_x = kernel_offset % geometry["kernel_w"]
    shifted_y = output_y + dy
    shifted_x = output_x + dx
    input_y = (
        shifted_y.reshape(-1, 1) * geometry["stride_h"]
        - geometry["padding_h"]
        + kernel_y.reshape(1, -1) * geometry["dilation_h"]
    )
    input_x = (
        shifted_x.reshape(-1, 1) * geometry["stride_w"]
        - geometry["padding_w"]
        + kernel_x.reshape(1, -1) * geometry["dilation_w"]
    )
    valid = (
        (input_y >= 0) & (input_y < geometry["height"])
        & (input_x >= 0) & (input_x < geometry["width"])
    )
    absolute_channel = group * geometry["channels_per_group"] + local_channel
    gathered = value_t[
        int(batch),
        absolute_channel.reshape(1, -1).expand(input_y.shape[0], -1),
        input_y.clamp(0, geometry["height"] - 1),
        input_x.clamp(0, geometry["width"] - 1),
    ]
    return gathered & valid


def analyze_conv_tile(
    module, binary_value, timestep, batch, group,
    y_start, y_stop, x_start, x_stop,
    candidates, source_chunk_size=256,
):
    """Analyze one tile and return scalar conservation fields, never raw masks."""
    value, geometry = _conv_geometry(module, binary_value)
    if value.dtype != torch.bool:
        raise ValueError("analyze_conv_tile requires Boolean input")
    timestep = int(timestep)
    if not 0 <= timestep < 10:
        raise ValueError("timestep outside T10")
    if not 0 <= int(batch) < geometry["batches"] or not 0 <= int(group) < geometry["groups"]:
        raise ValueError("batch/group outside Conv2d extent")
    if int(source_chunk_size) <= 0:
        raise ValueError("source_chunk_size must be positive")
    output_y, output_x = _tile_positions(
        geometry, y_start, y_stop, x_start, x_stop, value.device
    )
    rows = int(output_y.numel())
    local_current = 0
    for source_start in range(0, geometry["source_width"], int(source_chunk_size)):
        source_stop = min(geometry["source_width"], source_start + int(source_chunk_size))
        current = _gather_source_chunk(
            value[timestep], geometry, batch, group, output_y, output_x,
            (0, 0), source_start, source_stop,
        )
        local_current += int(current.sum().item())
    if timestep == 0:
        return {
            "selected_dy": 0,
            "selected_dx": 0,
            "candidate_rank": 0,
            "rows": rows,
            "base_valid_rows": 0,
            "border_fallback_rows": rows,
            "local_current_source_count": local_current,
            "valid_current_source_count": 0,
            "shifted_previous_source_count": 0,
            "positive_residual_source_count": local_current,
            "negative_residual_source_count": 0,
            "selected_source_count": local_current,
            "search_bit_comparisons": 0,
            "search_current_input_bit_reads": 0,
            "search_previous_input_bit_reads": 0,
            "search_candidate_costs": 0,
            "search_selector_comparisons": 0,
        }

    if not candidates:
        raise ValueError("candidate list is empty")
    candidate_stats = []
    for rank, shift in enumerate(candidates):
        dy, dx = shift
        base_valid = (
            (output_y + int(dy) >= 0)
            & (output_y + int(dy) < geometry["output_h"])
            & (output_x + int(dx) >= 0)
            & (output_x + int(dx) < geometry["output_w"])
        )
        positive = 0
        negative = 0
        valid_current = 0
        shifted_previous = 0
        bit_comparisons = 0
        for source_start in range(0, geometry["source_width"], int(source_chunk_size)):
            source_stop = min(geometry["source_width"], source_start + int(source_chunk_size))
            current = _gather_source_chunk(
                value[timestep], geometry, batch, group, output_y, output_x,
                (0, 0), source_start, source_stop,
            )
            previous = _gather_source_chunk(
                value[timestep - 1], geometry, batch, group, output_y, output_x,
                shift, source_start, source_stop,
            )
            valid_2d = base_valid.reshape(-1, 1)
            positive += int(((current & ~previous) & valid_2d).sum().item())
            negative += int(((previous & ~current) & valid_2d).sum().item())
            # Border rows have no old output base and therefore execute Local.
            positive += int((current & ~valid_2d).sum().item())
            valid_current += int((current & valid_2d).sum().item())
            shifted_previous += int((previous & valid_2d).sum().item())
            bit_comparisons += int(base_valid.sum().item()) * (source_stop - source_start)
        # The fallback positive terms must be removed before the valid-row
        # signed conservation identity is checked.
        fallback_current = local_current - valid_current
        if valid_current - shifted_previous != (positive - fallback_current) - negative:
            raise ValueError("signed input conservation failed")
        candidate_stats.append({
            "selected_dy": int(dy),
            "selected_dx": int(dx),
            "candidate_rank": int(rank),
            "rows": rows,
            "base_valid_rows": int(base_valid.sum().item()),
            "border_fallback_rows": rows - int(base_valid.sum().item()),
            "local_current_source_count": local_current,
            "valid_current_source_count": valid_current,
            "shifted_previous_source_count": shifted_previous,
            "positive_residual_source_count": positive,
            "negative_residual_source_count": negative,
            "selected_source_count": positive + negative,
            "search_bit_comparisons": bit_comparisons,
            "search_current_input_bit_reads": rows * geometry["source_width"],
            "search_previous_input_bit_reads": bit_comparisons,
        })
    # Canonical order is supplied by canonical_shift_candidates; rank resolves
    # every equal-work case without looking at tensor content or runtime order.
    selected = min(candidate_stats, key=lambda item: (item["selected_source_count"], item["candidate_rank"]))
    selected = dict(selected)
    selected["search_bit_comparisons"] = sum(
        item["search_bit_comparisons"] for item in candidate_stats
    )
    selected["search_current_input_bit_reads"] = sum(
        item["search_current_input_bit_reads"] for item in candidate_stats
    )
    selected["search_previous_input_bit_reads"] = sum(
        item["search_previous_input_bit_reads"] for item in candidate_stats
    )
    selected["search_candidate_costs"] = len(candidate_stats)
    selected["search_selector_comparisons"] = len(candidate_stats) - 1
    fallback_current = selected["local_current_source_count"] - selected["valid_current_source_count"]
    if (
        selected["valid_current_source_count"]
        - selected["shifted_previous_source_count"]
        != selected["positive_residual_source_count"]
        - fallback_current
        - selected["negative_residual_source_count"]
    ):
        raise ValueError("selected signed input conservation failed")
    if selected["selected_source_count"] != (
        selected["positive_residual_source_count"]
        + selected["negative_residual_source_count"]
    ):
        raise ValueError("selected source partition failed")
    return selected


def _sparse_counter(counter):
    return [[str(key), int(counter[key])] for key in sorted(counter, key=lambda item: str(item))]


class StreamingShiftResidualCensusWriter(object):
    """Atomic aggregate-only writer for exact Conv2d shift residual census."""

    def __init__(
        self, output_dir, temporal_steps=10, shift_radius=1,
        output_tile=(16, 16), source_chunk_size=256,
        accumulator_bits=24, candidate_shifts=None,
        expected_samples=0, expected_operator_calls=0, expected_exact_calls=0,
    ):
        self.output_dir = Path(output_dir)
        self.temporal_steps = int(temporal_steps)
        self.shift_radius = int(shift_radius)
        self.output_tile = _pair(output_tile, "output_tile")
        self.source_chunk_size = int(source_chunk_size)
        self.accumulator_bits = int(accumulator_bits)
        self.expected_samples = int(expected_samples)
        self.expected_operator_calls = int(expected_operator_calls)
        self.expected_exact_calls = int(expected_exact_calls)
        self.candidates = canonical_shift_candidates(self.shift_radius, candidate_shifts)
        if self.temporal_steps != 10:
            raise ValueError("M28A requires temporal_steps=10")
        if min(self.output_tile) <= 0 or self.source_chunk_size <= 0 or self.accumulator_bits <= 0:
            raise ValueError("tile, source chunk, and accumulator width must be positive")
        if min(
            self.expected_samples,
            self.expected_operator_calls,
            self.expected_exact_calls,
        ) < 0:
            raise ValueError("M28A expected coverage counts must be nonnegative")
        if self.output_dir.exists():
            raise ValueError("refusing to overwrite shift census output: {}".format(self.output_dir))
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.output_dir.parent / (
            ".{}.m28a.lock".format(self.output_dir.name)
        )
        self._lock_handle = self.lock_path.open("a+")
        try:
            fcntl.flock(
                self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except OSError:
            self._lock_handle.close()
            raise ValueError(
                "another M28A writer owns output lock: {}".format(self.lock_path)
            )
        self.staging_dir = self.output_dir.parent / (
            ".{}.m28a_incomplete_{}".format(self.output_dir.name, os.getpid())
        )
        if self.staging_dir.exists():
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            raise ValueError("stale M28A staging directory exists: {}".format(self.staging_dir))
        self.staging_dir.mkdir()
        self.histogram_path = self.staging_dir / "shift_residual_histograms.jsonl"
        self.calls_path = self.staging_dir / "call_reconciliation.csv"
        self._histogram_handle = self.histogram_path.open("w", encoding="utf-8")
        self._calls_handle = self.calls_path.open("w", newline="", encoding="utf-8")
        self._fields = [
            "call_id", "sample_id", "sample_key", "sequence_key", "name", "operator",
            "operator_call_index", "status", "temporal_steps", "batches", "groups",
            "output_height", "output_width", "output_tiles", "source_width", "fanout",
            "input_shift_stride_y", "input_shift_stride_x",
            "local_current_source_count", "positive_residual_source_count",
            "negative_residual_source_count", "selected_source_count",
            "local_product_terms", "selected_product_terms", "product_term_delta",
            "base_output_scalar_reads", "output_scalar_writes", "bias_scalar_additions",
            "border_fallback_rows", "shift_reused_rows", "search_bit_comparisons",
            "search_candidate_costs", "search_selector_comparisons",
            "local_current_input_bit_reads",
            "search_current_input_bit_reads", "search_previous_input_bit_reads",
            "selected_current_input_bit_reads", "selected_previous_input_bit_reads",
            "previous_input_state_bits", "previous_output_state_bits",
            "selector_live_state_bits", "search_cost_counter_live_state_bits",
            "selector_trace_bits", "search_cost_trace_bits", "input_conservation_mismatches",
            "headline_admitted",
        ]
        self._calls_writer = csv.DictWriter(self._calls_handle, fieldnames=self._fields)
        self._calls_writer.writeheader()
        self.run_context = None
        self.calls = 0
        self.exact_calls = 0
        self.rejected_calls = 0
        self.status_counts = Counter()
        self.totals = Counter()
        self.peak_live_state_bits = Counter()
        self.sample_ids = set()
        self.call_keys = set()
        self._closed = False
        self._aborted = False
        self._published = False

    @property
    def manifest_path(self):
        return self.output_dir / "manifest.json"

    def bind_run_context(self, context):
        if self.run_context is not None or self.calls:
            raise ValueError("M28A context must be bound exactly once before calls")
        required = ("artifact_identity", "eval_protocol", "checkpoint_load_audit", "source_sha256")
        if not all(field in context for field in required):
            raise ValueError("incomplete M28A run context")
        if set(context["source_sha256"]) != {"profiler", "shift_residual_census_writer"}:
            raise ValueError("M28A source identity is incomplete")
        hexadecimal = set(string.hexdigits)
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in hexadecimal for character in value)
            for value in context["source_sha256"].values()
        ):
            raise ValueError("invalid M28A source SHA")
        identity = context["artifact_identity"]
        if not all(
            isinstance(identity.get(field), str)
            and len(identity[field]) == 64
            and all(character in hexadecimal for character in identity[field])
            for field in ("checkpoint_sha256", "config_sha256")
        ):
            raise ValueError("M28A checkpoint/config SHA is missing")
        protocol = context["eval_protocol"]
        if (
            int(protocol.get("temporal_steps", -1)) != self.temporal_steps
            or int(protocol.get("requested_profile_samples", -1))
            != self.expected_samples
            or int(protocol.get("expected_operator_calls", -1))
            != self.expected_operator_calls
            or int(protocol.get("expected_exact_calls", -1))
            != self.expected_exact_calls
            or int(protocol.get("eval_batch_size", -1)) <= 0
            or protocol.get("temporal_axis_contract")
            != "hook_input_dim0_is_T10_and_dim1_is_eval_batch"
        ):
            raise ValueError("M28A evaluation/coverage protocol drift")
        audit = context["checkpoint_load_audit"]
        missing = audit.get("missing_count", audit.get("missing", -1))
        unexpected = audit.get("unexpected_count", audit.get("unexpected", -1))
        if int(missing) != 0 or int(unexpected) != 0:
            raise ValueError("M28A checkpoint load audit is not clean")
        self.run_context = json.loads(json.dumps(context))

    def _observe_call_identity(self, context):
        key = (
            int(context["sample_id"]),
            str(context["sample_key"]),
            str(context["sequence_key"]),
            str(context["name"]),
            int(context["operator_call_index"]),
        )
        if key in self.call_keys:
            raise ValueError("duplicate M28A operator call identity")
        self.call_keys.add(key)
        self.sample_ids.add(int(context["sample_id"]))

    @staticmethod
    def _binary(value):
        # Bounded scans avoid allocating one full-size validation mask.
        flattened = value.reshape(-1)
        chunk = 1 << 20
        for start in range(0, int(flattened.numel()), chunk):
            part = flattened[start:start + chunk]
            if not bool(torch.logical_or(part.eq(0), part.eq(1)).all().item()):
                return False
        return True

    def _reject(self, context, status):
        if status not in REJECTION_STATUSES:
            raise ValueError("unsupported M28A rejection status")
        self._observe_call_identity(context)
        row = {field: 0 for field in self._fields}
        row.update({
            "call_id": self.calls,
            "sample_id": int(context["sample_id"]),
            "sample_key": context["sample_key"],
            "sequence_key": context["sequence_key"],
            "name": context["name"],
            "operator": context["operator"],
            "operator_call_index": int(context["operator_call_index"]),
            "status": status,
            "temporal_steps": self.temporal_steps,
            "headline_admitted": False,
        })
        self._calls_writer.writerow(row)
        self.calls += 1
        self.rejected_calls += 1
        self.status_counts[status] += 1

    def record_operator(
        self, module, input_tensor, name, sample_id, sample_key,
        sequence_key, operator_call_index, temporal_steps=10,
    ):
        if self._closed or self._aborted or self._published:
            raise ValueError("M28A writer is not open")
        if self.run_context is None:
            raise ValueError("M28A run context is not bound")
        if int(temporal_steps) != self.temporal_steps:
            raise ValueError("M28A temporal-step configuration drift")
        context = {
            "sample_id": int(sample_id),
            "sample_key": str(sample_key),
            "sequence_key": str(sequence_key),
            "name": str(name),
            "operator": module.__class__.__name__,
            "operator_call_index": int(operator_call_index),
        }
        value = input_tensor.detach()
        if not isinstance(module, torch.nn.Conv2d):
            # Linear is intentionally rejected: the profiler has no strict
            # spatial-layout identity for its flattened token/feature axes.
            status = (
                "TEMPORAL_AXIS_UNQUALIFIED"
                if isinstance(module, torch.nn.Linear)
                else "UNSUPPORTED_OPERATOR"
            )
            self._reject(context, status)
            return
        if str(getattr(module, "padding_mode", "zeros")) != "zeros" or isinstance(module.padding, str):
            self._reject(context, "UNSUPPORTED_PADDING")
            return
        if value.ndim != 5 or int(value.shape[0]) != self.temporal_steps:
            self._reject(context, "TEMPORAL_AXIS_UNQUALIFIED")
            return
        if int(value.shape[1]) != int(self.run_context["eval_protocol"]["eval_batch_size"]):
            self._reject(context, "TEMPORAL_AXIS_UNQUALIFIED")
            return
        if not self._binary(value):
            self._reject(context, "NON_BINARY_BYPASS")
            return

        self._observe_call_identity(context)

        binary, geometry = _conv_geometry(module, value.eq(1))
        totals = Counter()
        shift_hist = Counter()
        selected_hist = Counter()
        positive_hist = Counter()
        negative_hist = Counter()
        fallback_hist = Counter()
        temporal_shift_hist = Counter()
        tile_count = 0
        for timestep in range(self.temporal_steps):
            for batch in range(geometry["batches"]):
                for group in range(geometry["groups"]):
                    for y_start in range(0, geometry["output_h"], self.output_tile[0]):
                        y_stop = min(geometry["output_h"], y_start + self.output_tile[0])
                        for x_start in range(0, geometry["output_w"], self.output_tile[1]):
                            x_stop = min(geometry["output_w"], x_start + self.output_tile[1])
                            record = analyze_conv_tile(
                                module, binary, timestep, batch, group,
                                y_start, y_stop, x_start, x_stop,
                                self.candidates, self.source_chunk_size,
                            )
                            tile_count += 1
                            key = "{},{}".format(record["selected_dy"], record["selected_dx"])
                            shift_hist[key] += 1
                            temporal_shift_hist["t{}:{}".format(timestep, key)] += 1
                            selected_hist[record["selected_source_count"]] += 1
                            positive_hist[record["positive_residual_source_count"]] += 1
                            negative_hist[record["negative_residual_source_count"]] += 1
                            fallback_hist[record["border_fallback_rows"]] += 1
                            for field in (
                                "local_current_source_count", "positive_residual_source_count",
                                "negative_residual_source_count", "selected_source_count",
                                "base_valid_rows", "border_fallback_rows", "search_bit_comparisons",
                                "search_current_input_bit_reads", "search_previous_input_bit_reads",
                                "search_candidate_costs", "search_selector_comparisons",
                            ):
                                totals[field] += int(record[field])
        output_rows = (
            self.temporal_steps * geometry["batches"] * geometry["groups"]
            * geometry["output_h"] * geometry["output_w"]
        )
        if totals["base_valid_rows"] + totals["border_fallback_rows"] != output_rows:
            raise ValueError("M28A base/fallback row partition failed")
        if totals["selected_source_count"] != (
            totals["positive_residual_source_count"] + totals["negative_residual_source_count"]
        ):
            raise ValueError("M28A signed residual partition failed")
        fanout = geometry["fanout"]
        selector_bits = max(1, int(math.ceil(math.log(len(self.candidates), 2)))) if len(self.candidates) > 1 else 0
        maximum_tile_rows = min(self.output_tile[0], geometry["output_h"]) * min(
            self.output_tile[1], geometry["output_w"]
        )
        search_cost_bits = max(
            1,
            int(math.ceil(math.log(maximum_tile_rows * geometry["source_width"] + 1, 2))),
        )
        spatial_tiles = (
            geometry["batches"] * geometry["groups"]
            * ((geometry["output_h"] + self.output_tile[0] - 1) // self.output_tile[0])
            * ((geometry["output_w"] + self.output_tile[1] - 1) // self.output_tile[1])
        )
        selected_valid_rows = totals["base_valid_rows"]
        selected_current_reads = (
            selected_valid_rows + totals["border_fallback_rows"]
        ) * geometry["source_width"]
        selected_previous_reads = selected_valid_rows * geometry["source_width"]
        row = {
            "call_id": self.calls,
            "sample_id": context["sample_id"],
            "sample_key": context["sample_key"],
            "sequence_key": context["sequence_key"],
            "name": context["name"],
            "operator": context["operator"],
            "operator_call_index": context["operator_call_index"],
            "status": EXACT_STATUS,
            "temporal_steps": self.temporal_steps,
            "batches": geometry["batches"],
            "groups": geometry["groups"],
            "output_height": geometry["output_h"],
            "output_width": geometry["output_w"],
            "output_tiles": tile_count,
            "source_width": geometry["source_width"],
            "fanout": fanout,
            "input_shift_stride_y": geometry["stride_h"],
            "input_shift_stride_x": geometry["stride_w"],
            "local_current_source_count": totals["local_current_source_count"],
            "positive_residual_source_count": totals["positive_residual_source_count"],
            "negative_residual_source_count": totals["negative_residual_source_count"],
            "selected_source_count": totals["selected_source_count"],
            "local_product_terms": totals["local_current_source_count"] * fanout,
            "selected_product_terms": totals["selected_source_count"] * fanout,
            "product_term_delta": (
                totals["selected_source_count"] - totals["local_current_source_count"]
            ) * fanout,
            "base_output_scalar_reads": selected_valid_rows * fanout,
            "output_scalar_writes": output_rows * fanout,
            # Reused rows read a previous output that already contains bias;
            # only Local/fallback rows add it again.
            "bias_scalar_additions": (
                totals["border_fallback_rows"] * fanout
                if module.bias is not None else 0
            ),
            "border_fallback_rows": totals["border_fallback_rows"],
            "shift_reused_rows": selected_valid_rows,
            "search_bit_comparisons": totals["search_bit_comparisons"],
            "search_candidate_costs": totals["search_candidate_costs"],
            "search_selector_comparisons": totals["search_selector_comparisons"],
            "local_current_input_bit_reads": output_rows * geometry["source_width"],
            "search_current_input_bit_reads": totals["search_current_input_bit_reads"],
            "search_previous_input_bit_reads": totals["search_previous_input_bit_reads"],
            "selected_current_input_bit_reads": selected_current_reads,
            "selected_previous_input_bit_reads": selected_previous_reads,
            "previous_input_state_bits": (
                geometry["batches"] * geometry["channels"]
                * geometry["height"] * geometry["width"]
            ),
            "previous_output_state_bits": (
                geometry["batches"] * int(module.out_channels)
                * geometry["output_h"] * geometry["output_w"] * self.accumulator_bits
            ),
            "selector_live_state_bits": spatial_tiles * selector_bits,
            # Minimal sequential-search datapath state, separate from the
            # persisted per-spatial-tile selector map.
            "search_cost_counter_live_state_bits": 2 * search_cost_bits + selector_bits,
            "selector_trace_bits": (self.temporal_steps - 1) * spatial_tiles * selector_bits,
            "search_cost_trace_bits": (
                (self.temporal_steps - 1) * spatial_tiles
                * len(self.candidates) * search_cost_bits
            ),
            "input_conservation_mismatches": 0,
            "headline_admitted": False,
        }
        self._calls_writer.writerow(row)
        histogram_record = {
            "schema": "m28a_shift_residual_histograms_v1",
            "sample_id": context["sample_id"],
            "sample_key": context["sample_key"],
            "sequence_key": context["sequence_key"],
            "name": context["name"],
            "operator_call_index": context["operator_call_index"],
            "canonical_shift_order": [list(item) for item in self.candidates],
            "input_shift_mapping": "input_delta_y=stride_y*output_shift_y; input_delta_x=stride_x*output_shift_x",
            "shift_selection_histogram": _sparse_counter(shift_hist),
            "temporal_shift_selection_histogram": _sparse_counter(temporal_shift_hist),
            "selected_source_count_histogram": _sparse_counter(selected_hist),
            "positive_residual_count_histogram": _sparse_counter(positive_hist),
            "negative_residual_count_histogram": _sparse_counter(negative_hist),
            "border_fallback_rows_histogram": _sparse_counter(fallback_hist),
            "raw_tensor_or_tile_map_saved": False,
        }
        self._histogram_handle.write(json.dumps(histogram_record, sort_keys=True, separators=(",", ":")) + "\n")
        self.calls += 1
        self.exact_calls += 1
        self.status_counts[EXACT_STATUS] += 1
        for field in (
            "local_product_terms", "selected_product_terms", "product_term_delta",
            "base_output_scalar_reads", "output_scalar_writes", "search_bit_comparisons",
            "bias_scalar_additions",
        ):
            self.totals[field] += int(row[field])
        for field in (
            "previous_input_state_bits", "previous_output_state_bits",
            "selector_live_state_bits", "search_cost_counter_live_state_bits",
        ):
            self.peak_live_state_bits[field] = max(
                int(self.peak_live_state_bits[field]), int(row[field])
            )

    @staticmethod
    def _flush_fsync(handle):
        handle.flush()
        os.fsync(handle.fileno())

    def _close_streams(self):
        if not self._histogram_handle.closed:
            self._flush_fsync(self._histogram_handle)
            self._histogram_handle.close()
        if not self._calls_handle.closed:
            self._flush_fsync(self._calls_handle)
            self._calls_handle.close()

    def _release_lock(self):
        if not self._lock_handle.closed:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()

    @staticmethod
    def _write_json_fsync(path, payload):
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path):
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def abort(self, reason="profiling interrupted"):
        if self._published:
            raise ValueError("cannot abort a published M28A census")
        if self._aborted:
            return
        self._close_streams()
        self._write_json_fsync(self.staging_dir / "INTERRUPTED.json", {
            "schema": "m28a_interrupted_run_v1",
            "status": "INTERRUPTED_NOT_ADMITTED",
            "reason": str(reason),
            "calls_before_abort": self.calls,
            "exact_calls_before_abort": self.exact_calls,
            "rejected_calls_before_abort": self.rejected_calls,
            "headline_admitted": False,
            "pass_manifest_written": False,
        })
        self._fsync_directory(self.staging_dir)
        self._aborted = True
        self._closed = True
        self._release_lock()

    def close(self):
        if self._published:
            return
        if self._aborted:
            raise ValueError("cannot close an aborted M28A census")
        if self.run_context is None or self.calls <= 0:
            self.abort("no calls were observed")
            raise ValueError("M28A cannot publish without context and calls")
        if self.exact_calls <= 0:
            self.abort("all calls were rejected")
            raise ValueError("M28A all-rejected census cannot PASS")
        if self.exact_calls + self.rejected_calls != self.calls:
            self.abort("call status accounting mismatch")
            raise ValueError("M28A call accounting mismatch")
        expected_sample_ids = set(range(self.expected_samples))
        if self.expected_samples <= 0 or self.sample_ids != expected_sample_ids:
            self.abort("sample coverage mismatch")
            raise ValueError("M28A sample coverage mismatch")
        if self.expected_operator_calls <= 0 or self.calls != self.expected_operator_calls:
            self.abort("operator-call coverage mismatch")
            raise ValueError("M28A operator-call coverage mismatch")
        if self.expected_exact_calls <= 0 or self.exact_calls != self.expected_exact_calls:
            self.abort("exact-call coverage mismatch")
            raise ValueError("M28A exact-call coverage mismatch")
        self._close_streams()
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (self.histogram_path, self.calls_path)
        }
        self._write_json_fsync(self.staging_dir / "manifest.staged.json", {
            "schema": "m28a_shift_residual_census_manifest_v1",
            "status": "STAGING_NOT_ADMITTED",
            "files": files,
            "pass_manifest_written": False,
        })
        self._fsync_directory(self.staging_dir)
        os.replace(str(self.staging_dir), str(self.output_dir))
        self._fsync_directory(self.output_dir.parent)
        (self.output_dir / "manifest.staged.json").unlink()
        manifest = {
            "schema": "m28a_shift_residual_census_manifest_v1",
            "status": PASS_STATUS,
            "headline_admitted": False,
            "temporal_steps": self.temporal_steps,
            "shift_radius": self.shift_radius,
            "canonical_shift_order": [list(item) for item in self.candidates],
            "output_tile": list(self.output_tile),
            "source_chunk_size": self.source_chunk_size,
            "accumulator_bits": self.accumulator_bits,
            "calls": self.calls,
            "exact_calls": self.exact_calls,
            "rejected_calls": self.rejected_calls,
            "status_counts": dict(sorted(self.status_counts.items())),
            "workload_totals": dict(sorted(self.totals.items())),
            "peak_live_capacity_bits": dict(sorted(self.peak_live_state_bits.items())),
            "coverage": {
                "expected_samples": self.expected_samples,
                "observed_sample_ids": sorted(self.sample_ids),
                "expected_operator_calls": self.expected_operator_calls,
                "observed_operator_calls": self.calls,
                "expected_exact_calls": self.expected_exact_calls,
                "observed_exact_calls": self.exact_calls,
            },
            "input_conservation_mismatches": 0,
            "algebraic_binary_source_conservation_exact": True,
            "fixed_point_bit_exact": False,
            "fixed_point_vcs_miter_status": "PENDING_REAL_QUANTIZED_WEIGHT_RANGE_AND_SATURATION_MITER",
            "accumulator_bits_role": "candidate_width_not_yet_range_admitted",
            "raw_activation_residual_or_tile_map_saved": False,
            "files": files,
            "run_context": self.run_context,
            "claim_boundary": (
                "Algebraically exact binary-source T10 Conv2d shift/residual opportunity "
                "census with explicit complete-call coverage and "
                "search, state, traffic, border-fallback, and signed-source ledgers. "
                "Fixed-point bit exactness remains pending a real-weight VCS miter. It is "
                "not a cycle, speedup, energy, accuracy, PPA, or headline claim."
            ),
        }
        temporary = self.output_dir / ".manifest.json.tmp"
        self._write_json_fsync(temporary, manifest)
        os.replace(str(temporary), str(self.manifest_path))
        self._fsync_directory(self.output_dir)
        self._published = True
        self._closed = True
        self._release_lock()

    def discard_staging_for_test_only(self):
        if self._published:
            raise ValueError("cannot discard a published M28A census")
        self._close_streams()
        if self.staging_dir.exists():
            shutil.rmtree(str(self.staging_dir))
        self._closed = True
        self._release_lock()
