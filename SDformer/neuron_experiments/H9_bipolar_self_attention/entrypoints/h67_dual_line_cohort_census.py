#!/usr/bin/env python3
"""Streaming exact T10 Local/Motion coefficient-cohort census.

The writer processes every output row and every logical source identity in
bounded blocks.  It emits histograms only; it never stores input activations or
per-row bitmaps.  A run is admitted only after the staging directory is renamed
and a PASS manifest is atomically installed by :meth:`close`.
"""

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import torch


PASS_STATUS = "PASS_EXACT_STREAMING_T10_COHORT_CENSUS_REFERENCE_RECONCILED"
EXACT_STATUS = "PASS_EXACT_SOURCE_WORK"
REJECTION_STATUSES = {"NON_BINARY_BYPASS", "TEMPORAL_AXIS_UNQUALIFIED"}
HISTOGRAM_BINS = 1024
SIGNED_PAIR_BINS = 1 << 20
WORKING_SET_FIXED_BYTES = 1 << 16
WORKING_SET_BYTES_PER_SOURCE = 128
WORKING_SET_BYTES_PER_ROW = 1024
WORKING_SET_BYTES_PER_ROW_SOURCE = 128
BINARY_SCAN_BYTES_PER_ELEMENT = 64


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ceil_div(value, divisor):
    if value < 0 or divisor <= 0:
        raise ValueError("invalid ceil-div operands")
    return (int(value) + int(divisor) - 1) // int(divisor)


def _pair(value):
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("expected scalar or pair")


def _conv_output_extent(size, kernel, stride, padding, dilation):
    numerator = int(size) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1
    result = numerator // int(stride) + 1
    if result <= 0:
        raise ValueError("Conv2d output extent is nonpositive")
    return result


def _popcount10(value):
    return bin(int(value) & 0x3FF).count("1")


def _sparse_histogram(values, bins):
    flattened = values.reshape(-1).to(dtype=torch.int64)
    if flattened.numel() == 0:
        return [], 0
    if int(flattened.min().item()) < 0 or int(flattened.max().item()) >= int(bins):
        raise ValueError("histogram value outside admitted domain")
    if bins <= HISTOGRAM_BINS:
        counts = torch.bincount(flattened, minlength=int(bins)).detach().cpu().tolist()
        sparse = [[index, int(count)] for index, count in enumerate(counts) if int(count)]
    else:
        unique, counts = torch.unique(flattened, sorted=True, return_counts=True)
        sparse = [
            [int(value), int(count)]
            for value, count in zip(unique.detach().cpu().tolist(), counts.detach().cpu().tolist())
        ]
    total = sum(item[1] for item in sparse)
    if total != int(flattened.numel()):
        raise ValueError("sparse histogram population mismatch")
    return sparse, total


def _histogram_event_count(sparse):
    return sum(_popcount10(mask) * int(count) for mask, count in sparse)


def _histogram_nonzero_population(sparse):
    return sum(int(count) for mask, count in sparse if int(mask) != 0)


def _signed_pair_event_counts(sparse):
    positive = 0
    negative = 0
    nonzero = 0
    for code, count in sparse:
        code = int(code)
        count = int(count)
        positive_mask = code & 0x3FF
        negative_mask = (code >> 10) & 0x3FF
        if positive_mask & negative_mask:
            raise ValueError("positive and negative masks overlap")
        positive += _popcount10(positive_mask) * count
        negative += _popcount10(negative_mask) * count
        if positive_mask or negative_mask:
            nonzero += count
    return positive, negative, nonzero


class StreamingCohortCensusWriter(object):
    """Write exact per-call cohort histograms without retaining raw tensors."""

    def __init__(
        self, output_dir, temporal_steps=10, source_chunk_size=256,
        requested_row_block_size=512, max_working_set_mib=256,
    ):
        self.output_dir = Path(output_dir)
        self.temporal_steps = int(temporal_steps)
        self.source_chunk_size = int(source_chunk_size)
        self.requested_row_block_size = int(requested_row_block_size)
        self.max_working_set_bytes = int(max_working_set_mib) * (1 << 20)
        if (
            self.temporal_steps != 10
            or self.source_chunk_size <= 0
            or self.requested_row_block_size <= 0
            or self.max_working_set_bytes <= 0
        ):
            raise ValueError("M24C requires T10 and positive chunk/block/memory bounds")
        if self.output_dir.exists():
            raise ValueError("refusing to overwrite cohort census output: {}".format(self.output_dir))
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.output_dir.parent / (
            ".{}.m24c_incomplete_{}".format(self.output_dir.name, os.getpid())
        )
        if self.staging_dir.exists():
            raise ValueError("stale M24C staging directory exists: {}".format(self.staging_dir))
        self.staging_dir.mkdir()
        self.histogram_path = self.staging_dir / "cohort_histograms.jsonl"
        self.reconciliation_path = self.staging_dir / "call_reconciliation.csv"
        self._histogram_handle = self.histogram_path.open("w", encoding="utf-8")
        self._reconciliation_handle = self.reconciliation_path.open(
            "w", newline="", encoding="utf-8"
        )
        self._reconciliation_fields = [
            "call_id", "sample_id", "sample_key", "sequence_key", "name", "operator",
            "operator_call_index", "status", "temporal_steps", "rows", "source_width",
            "source_chunks", "weight_groups", "output_channel_fanout",
            "valid_source_count", "current_source_count", "positive_transition_source_count",
            "negative_transition_source_count", "selected_positive_source_count",
            "selected_negative_source_count", "selected_source_count",
            "local_cohort_coefficient_vectors", "motion_cohort_coefficient_vectors",
            "local_cohort_coefficient_scalar_reads", "motion_cohort_coefficient_scalar_reads",
            "destination_scalar_updates", "histogram_records", "maximum_row_block_size",
            "estimated_peak_working_set_bytes", "reference_reconciled",
        ]
        self._reconciliation_writer = csv.DictWriter(
            self._reconciliation_handle, fieldnames=self._reconciliation_fields
        )
        self._reconciliation_writer.writeheader()
        self.run_context = None
        self.calls = 0
        self.histogram_records = 0
        self.status_counts = Counter()
        self.operator_status_counts = Counter()
        self.total_counts = Counter()
        self.maximum_row_block_size = 0
        self.binary_scan_chunk_elements = max(
            1, min(1 << 20, self.max_working_set_bytes // BINARY_SCAN_BYTES_PER_ELEMENT)
        )
        self.binary_scan_estimated_working_set_bytes = (
            self.binary_scan_chunk_elements * BINARY_SCAN_BYTES_PER_ELEMENT
        )
        self.maximum_estimated_working_set_bytes = (
            self.binary_scan_estimated_working_set_bytes
        )
        self._closed = False
        self._aborted = False
        self._published = False

    @property
    def manifest_path(self):
        return self.output_dir / "manifest.json"

    def bind_run_context(self, context):
        if self.run_context is not None or self.calls:
            raise ValueError("M24C run context must be bound exactly once before calls")
        required = ("artifact_identity", "eval_protocol", "checkpoint_load_audit", "source_sha256")
        if not all(field in context for field in required):
            raise ValueError("incomplete M24C run context")
        sources = context.get("source_sha256", {})
        if set(sources) != {
            "profiler", "cohort_census_writer", "dual_line_reference",
        }:
            raise ValueError(
                "M24C source identity must bind profiler, writer, and dual-line reference"
            )
        for digest in sources.values():
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("invalid M24C source SHA")
        identity = context.get("artifact_identity", {})
        if not all(
            isinstance(identity.get(field), str) and len(identity[field]) == 64
            for field in ("checkpoint_sha256", "config_sha256")
        ):
            raise ValueError("M24C checkpoint/config SHA is missing")
        self.run_context = json.loads(json.dumps(context))

    def _working_row_block(self, source_width):
        source_width = int(source_width)
        if source_width <= 0:
            raise ValueError("source width must be positive")
        # Conservative tensor-payload contract: Conv2d gather geometry, T10
        # current/transition state, three int64 masks, pair codes, count vectors,
        # histogram temporaries, and slack for framework tensor payloads.
        fixed_bytes = (
            WORKING_SET_FIXED_BYTES
            + source_width * WORKING_SET_BYTES_PER_SOURCE
        )
        bytes_per_row = (
            WORKING_SET_BYTES_PER_ROW
            + source_width * WORKING_SET_BYTES_PER_ROW_SOURCE
        )
        maximum = (self.max_working_set_bytes - fixed_bytes) // bytes_per_row
        if maximum < 1:
            raise ValueError(
                "configured M24C memory bound cannot hold one source row: width={}".format(
                    source_width
                )
            )
        block = min(self.requested_row_block_size, int(maximum))
        estimated = fixed_bytes + block * bytes_per_row
        if estimated > self.max_working_set_bytes:
            raise ValueError("M24C working-set bound calculation failed")
        self.maximum_row_block_size = max(self.maximum_row_block_size, block)
        self.maximum_estimated_working_set_bytes = max(
            self.maximum_estimated_working_set_bytes, estimated
        )
        return block, estimated

    @staticmethod
    def _logical_block(value, row_start, row_stop, source_start, source_stop):
        """Gather a bounded logical row/source block without materializing reshape copies."""
        prefix_shape = tuple(int(item) for item in value.shape[:-1])
        if not prefix_shape:
            if int(row_start) != 0 or int(row_stop) != 1:
                raise ValueError("logical scalar-row range is invalid")
            return value[int(source_start):int(source_stop)].reshape(1, -1)
        logical_rows = 1
        for extent in prefix_shape:
            logical_rows *= extent
        if not (0 <= int(row_start) < int(row_stop) <= logical_rows):
            raise ValueError("logical row block is outside tensor extent")
        remainder = torch.arange(
            int(row_start), int(row_stop), dtype=torch.int64, device=value.device
        )
        coordinates = []
        for extent in reversed(prefix_shape):
            coordinates.append(remainder % extent)
            remainder = remainder // extent
        coordinates.reverse()
        return value[tuple(coordinates) + (slice(int(source_start), int(source_stop)),)]

    def _binary_status(self, value):
        width = int(value.shape[-1])
        rows = int(value.numel()) // width
        source_block = min(width, self.binary_scan_chunk_elements)
        row_block = max(1, self.binary_scan_chunk_elements // source_block)
        for row_start in range(0, rows, row_block):
            row_stop = min(rows, row_start + row_block)
            for source_start in range(0, width, source_block):
                source_stop = min(width, source_start + source_block)
                part = self._logical_block(
                    value, row_start, row_stop, source_start, source_stop
                )
                if not bool(torch.logical_or(part.eq(0), part.eq(1)).all().item()):
                    return False
        return True

    def _reject_call(self, context, status, reference_work):
        if status not in REJECTION_STATUSES:
            raise ValueError("unsupported M24C rejection status: {}".format(status))
        if not (
            isinstance(reference_work, list)
            and len(reference_work) == 1
            and reference_work[0].get("status") == status
        ):
            raise ValueError("M24C rejection disagrees with dual-line reference")
        row = {
            "call_id": self.calls,
            "sample_id": int(context["sample_id"]),
            "sample_key": context["sample_key"],
            "sequence_key": context["sequence_key"],
            "name": context["name"],
            "operator": context["operator"],
            "operator_call_index": int(context["operator_call_index"]),
            "status": status,
            "temporal_steps": self.temporal_steps,
            "rows": 0,
            "source_width": 0,
            "source_chunks": 0,
            "weight_groups": 0,
            "output_channel_fanout": 0,
            "valid_source_count": 0,
            "current_source_count": 0,
            "positive_transition_source_count": 0,
            "negative_transition_source_count": 0,
            "selected_positive_source_count": 0,
            "selected_negative_source_count": 0,
            "selected_source_count": 0,
            "local_cohort_coefficient_vectors": 0,
            "motion_cohort_coefficient_vectors": 0,
            "local_cohort_coefficient_scalar_reads": 0,
            "motion_cohort_coefficient_scalar_reads": 0,
            "destination_scalar_updates": 0,
            "histogram_records": 0,
            "maximum_row_block_size": 0,
            "estimated_peak_working_set_bytes": 0,
            "reference_reconciled": True,
        }
        self._reconciliation_writer.writerow(row)
        self.calls += 1
        self.status_counts[status] += 1
        self.operator_status_counts[(context["operator"], status)] += 1

    def _validate_reference(self, reference_work, per_t, fanout):
        if not isinstance(reference_work, list) or len(reference_work) != self.temporal_steps:
            raise ValueError("M24C exact call lacks ten reference rows")
        fields = (
            ("selector_rows", "selector_rows", 1),
            ("motion_selected_rows", "motion_selected_rows", 1),
            ("local_selected_rows", "local_selected_rows", 1),
            ("valid_source_work", "valid_source_count", fanout),
            ("current_source_count", "current_source_count", 1),
            ("positive_transition_source_count", "positive_transition_source_count", 1),
            ("negative_transition_source_count", "negative_transition_source_count", 1),
            ("local_work", "current_source_count", fanout),
            ("motion_work", "transition_source_count", fanout),
            ("selected_work", "selected_source_count", fanout),
            ("selector_saved_work", "selector_saved_source_count", fanout),
            ("output_channel_fanout", None, 1),
        )
        for timestep in range(self.temporal_steps):
            reference = reference_work[timestep]
            actual = per_t[timestep]
            if (
                reference.get("status") != EXACT_STATUS
                or int(reference.get("temporal_step", -1)) != timestep
                or bool(reference.get("state_valid")) != (timestep > 0)
            ):
                raise ValueError("M24C reference temporal identity mismatch")
            for reference_field, actual_field, scale in fields:
                expected = fanout if actual_field is None else int(actual[actual_field]) * int(scale)
                if int(reference.get(reference_field, -1)) != int(expected):
                    raise ValueError(
                        "M24C/reference mismatch t={} field={} expected={} actual={}".format(
                            timestep, reference_field, reference.get(reference_field), expected
                        )
                    )

    def _write_histogram_record(
        self, context, geometry, row_start, row_stop, block_index, source_base,
        source_valid_bits, current, choose_motion,
    ):
        rows = int(current.shape[1])
        source_count = int(current.shape[2])
        if source_count != int(source_valid_bits):
            raise ValueError("source chunk width mismatch")
        device = current.device
        local_mask = torch.zeros((rows, source_count), dtype=torch.int64, device=device)
        positive_mask = torch.zeros_like(local_mask)
        negative_mask = torch.zeros_like(local_mask)
        previous = torch.zeros((rows, source_count), dtype=torch.bool, device=device)
        for timestep in range(self.temporal_steps):
            now = current[timestep]
            raw_positive = now & ~previous
            raw_negative = previous & ~now
            selector = choose_motion[timestep].reshape(rows, 1)
            selected_positive = torch.where(selector, raw_positive, now)
            selected_negative = torch.where(
                selector, raw_negative, torch.zeros_like(raw_negative)
            )
            local_mask |= now.to(dtype=torch.int64) << timestep
            positive_mask |= selected_positive.to(dtype=torch.int64) << timestep
            negative_mask |= selected_negative.to(dtype=torch.int64) << timestep
            previous = now
        if bool((positive_mask & negative_mask).ne(0).any().item()):
            raise ValueError("M24C signed masks are not mutually exclusive")
        local_hist, population = _sparse_histogram(local_mask, HISTOGRAM_BINS)
        positive_hist, positive_population = _sparse_histogram(
            positive_mask, HISTOGRAM_BINS
        )
        negative_hist, negative_population = _sparse_histogram(
            negative_mask, HISTOGRAM_BINS
        )
        pair_code = positive_mask | (negative_mask << 10)
        signed_hist, signed_population = _sparse_histogram(pair_code, SIGNED_PAIR_BINS)
        if not (
            population == positive_population == negative_population == signed_population
            == rows * source_count
        ):
            raise ValueError("M24C histogram identity population mismatch")
        local_events = _histogram_event_count(local_hist)
        selected_positive_events, selected_negative_events, motion_coefficients = (
            _signed_pair_event_counts(signed_hist)
        )
        if (
            local_events != int(current.sum().item())
            or selected_positive_events + selected_negative_events
            != int(sum(
                _popcount10(code & 0x3FF) * count
                + _popcount10((code >> 10) & 0x3FF) * count
                for code, count in signed_hist
            ))
        ):
            raise ValueError("M24C histogram event reconciliation failed")
        record = {
            "schema": "m24c_streaming_cohort_histogram_record_v1",
            "histogram_domain_bins": HISTOGRAM_BINS,
            "signed_pair_domain_bins": SIGNED_PAIR_BINS,
            "sample_id": int(context["sample_id"]),
            "sample_key": context["sample_key"],
            "sequence_key": context["sequence_key"],
            "name": context["name"],
            "operator": context["operator"],
            "operator_call_index": int(context["operator_call_index"]),
            "row_block_index": int(block_index),
            "row_start": int(row_start),
            "row_stop_exclusive": int(row_stop),
            "source_chunk": int(source_base) // self.source_chunk_size,
            "source_base": int(source_base),
            "source_width": int(geometry["source_width"]),
            "valid_bits": int(source_valid_bits),
            "weight_group": int(geometry["weight_group"]),
            "output_channel_fanout": int(geometry["fanout"]),
            "row_source_identity_count": population,
            "local_presence_histogram_nonzero_bins": local_hist,
            "motion_positive_histogram_nonzero_bins": positive_hist,
            "motion_negative_histogram_nonzero_bins": negative_hist,
            "motion_signed_pair_histogram_nonzero_bins": signed_hist,
            "local_current_events": local_events,
            "selected_positive_events": selected_positive_events,
            "selected_negative_events": selected_negative_events,
            "local_cohort_coefficient_vectors": _histogram_nonzero_population(local_hist),
            "motion_cohort_coefficient_vectors": motion_coefficients,
        }
        self._histogram_handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self.histogram_records += 1
        return record

    def _process_current_block(
        self, context, geometry, current, valid_sources, row_start, block_index,
        per_t, call_totals,
    ):
        if current.dtype != torch.bool or current.ndim != 3:
            raise ValueError("M24C current block must be [T,rows,sources] Boolean")
        if int(current.shape[0]) != self.temporal_steps:
            raise ValueError("M24C current block is not T10")
        rows = int(current.shape[1])
        source_width = int(current.shape[2])
        if valid_sources.shape != current.shape[1:]:
            raise ValueError("M24C valid-source geometry mismatch")
        current_counts = current.sum(dim=2, dtype=torch.int64)
        previous = torch.zeros_like(current[0])
        transition_counts = []
        positive_counts = []
        negative_counts = []
        for timestep in range(self.temporal_steps):
            now = current[timestep]
            positive = now & ~previous
            negative = previous & ~now
            positive_counts.append(positive.sum(dim=1, dtype=torch.int64))
            negative_counts.append(negative.sum(dim=1, dtype=torch.int64))
            transition_counts.append(positive_counts[-1] + negative_counts[-1])
            previous = now
        positive_counts = torch.stack(positive_counts)
        negative_counts = torch.stack(negative_counts)
        transition_counts = torch.stack(transition_counts)
        choose_motion = transition_counts < current_counts
        choose_motion[0] = False
        selected_counts = torch.where(choose_motion, transition_counts, current_counts)
        valid_count = int(valid_sources.sum().item())
        for timestep in range(self.temporal_steps):
            current_total = int(current_counts[timestep].sum().item())
            positive_total = int(positive_counts[timestep].sum().item())
            negative_total = int(negative_counts[timestep].sum().item())
            selected_total = int(selected_counts[timestep].sum().item())
            per_t[timestep]["selector_rows"] += rows
            per_t[timestep]["motion_selected_rows"] += int(
                choose_motion[timestep].sum().item()
            )
            per_t[timestep]["local_selected_rows"] += rows - int(
                choose_motion[timestep].sum().item()
            )
            per_t[timestep]["valid_source_count"] += valid_count
            per_t[timestep]["current_source_count"] += current_total
            per_t[timestep]["positive_transition_source_count"] += positive_total
            per_t[timestep]["negative_transition_source_count"] += negative_total
            per_t[timestep]["transition_source_count"] += positive_total + negative_total
            per_t[timestep]["selected_source_count"] += selected_total
            per_t[timestep]["selector_saved_source_count"] += current_total - selected_total
        for source_base in range(0, source_width, self.source_chunk_size):
            stop = min(source_width, source_base + self.source_chunk_size)
            record = self._write_histogram_record(
                context, geometry, row_start, row_start + rows, block_index,
                source_base, stop - source_base, current[:, :, source_base:stop],
                choose_motion,
            )
            for field in (
                "local_current_events", "selected_positive_events",
                "selected_negative_events", "local_cohort_coefficient_vectors",
                "motion_cohort_coefficient_vectors",
            ):
                call_totals[field] += int(record[field])

    def _linear(self, module, value, context, per_t, call_totals):
        if int(value.shape[-1]) != int(module.in_features):
            raise ValueError("M24C Linear input feature dimension mismatch")
        rows = int(value.numel()) // (self.temporal_steps * int(module.in_features))
        if rows <= 0:
            raise ValueError("M24C Linear has no rows")
        source_width = int(module.in_features)
        fanout = int(module.out_features)
        block_rows, estimated = self._working_row_block(source_width)
        block_index = 0
        for start in range(0, rows, block_rows):
            stop = min(rows, start + block_rows)
            current = torch.empty(
                (self.temporal_steps, stop - start, source_width),
                dtype=torch.bool,
                device=value.device,
            )
            for timestep in range(self.temporal_steps):
                current[timestep] = self._logical_block(
                    value[timestep], start, stop, 0, source_width
                ).eq(1)
            valid = torch.ones(
                (stop - start, source_width), dtype=torch.bool, device=value.device
            )
            geometry = {
                "source_width": source_width, "weight_group": 0, "fanout": fanout,
            }
            self._process_current_block(
                context, geometry, current, valid, start, block_index, per_t, call_totals
            )
            block_index += 1
        return {
            "rows": rows, "source_width": source_width,
            "source_chunks": _ceil_div(source_width, self.source_chunk_size),
            "weight_groups": 1, "fanout": fanout,
            "maximum_row_block_size": min(rows, block_rows),
            "estimated_peak_working_set_bytes": estimated,
        }

    def _conv2d(self, module, value, context, per_t, call_totals):
        if value.ndim == 4:
            value = value.unsqueeze(1)
        if value.ndim != 5 or int(value.shape[0]) != self.temporal_steps:
            raise ValueError("M24C Conv2d input must be [T,B,C,H,W] or [T,C,H,W]")
        temporal, batches, channels, height, width = [int(item) for item in value.shape]
        del temporal
        if channels != int(module.in_channels):
            raise ValueError("M24C Conv2d input channel mismatch")
        kernel_h, kernel_w = _pair(module.kernel_size)
        stride_h, stride_w = _pair(module.stride)
        padding_h, padding_w = _pair(module.padding)
        dilation_h, dilation_w = _pair(module.dilation)
        groups = int(module.groups)
        if groups <= 0 or channels % groups or int(module.out_channels) % groups:
            raise ValueError("M24C Conv2d group geometry is invalid")
        output_h = _conv_output_extent(
            height, kernel_h, stride_h, padding_h, dilation_h
        )
        output_w = _conv_output_extent(
            width, kernel_w, stride_w, padding_w, dilation_w
        )
        source_width = (channels // groups) * kernel_h * kernel_w
        fanout = int(module.out_channels) // groups
        block_rows, estimated = self._working_row_block(source_width)
        spatial_rows = output_h * output_w
        block_index = 0
        for batch in range(batches):
            for group in range(groups):
                for spatial_start in range(0, spatial_rows, block_rows):
                    spatial_stop = min(spatial_rows, spatial_start + block_rows)
                    positions = torch.arange(
                        spatial_start, spatial_stop, dtype=torch.int64, device=value.device
                    )
                    output_y = positions // output_w
                    output_x = positions % output_w
                    sources = torch.arange(
                        source_width, dtype=torch.int64, device=value.device
                    )
                    kernel_area = kernel_h * kernel_w
                    input_channel = sources // kernel_area
                    kernel_offset = sources % kernel_area
                    kernel_y = kernel_offset // kernel_w
                    kernel_x = kernel_offset % kernel_w
                    input_y = (
                        output_y.reshape(-1, 1) * stride_h - padding_h
                        + kernel_y.reshape(1, -1) * dilation_h
                    )
                    input_x = (
                        output_x.reshape(-1, 1) * stride_w - padding_w
                        + kernel_x.reshape(1, -1) * dilation_w
                    )
                    valid = (
                        (input_y >= 0) & (input_y < height)
                        & (input_x >= 0) & (input_x < width)
                    )
                    absolute_channel = group * (channels // groups) + input_channel
                    current = torch.empty(
                        (self.temporal_steps, spatial_stop - spatial_start, source_width),
                        dtype=torch.bool,
                        device=value.device,
                    )
                    for timestep in range(self.temporal_steps):
                        gathered = value[
                            timestep,
                            batch,
                            absolute_channel.reshape(1, -1),
                            input_y.clamp(0, height - 1),
                            input_x.clamp(0, width - 1),
                        ]
                        current[timestep] = gathered.eq(1) & valid
                    del (
                        gathered, input_y, input_x, positions,
                        output_y, output_x, sources, input_channel,
                        kernel_offset, kernel_y, kernel_x, absolute_channel,
                    )
                    global_row_start = (
                        (batch * groups + group) * spatial_rows + spatial_start
                    )
                    geometry = {
                        "source_width": source_width,
                        "weight_group": group,
                        "fanout": fanout,
                    }
                    self._process_current_block(
                        context, geometry, current, valid, global_row_start,
                        block_index, per_t, call_totals,
                    )
                    block_index += 1
        return {
            "rows": batches * groups * spatial_rows,
            "source_width": source_width,
            "source_chunks": _ceil_div(source_width, self.source_chunk_size),
            "weight_groups": groups,
            "fanout": fanout,
            "maximum_row_block_size": min(spatial_rows, block_rows),
            "estimated_peak_working_set_bytes": estimated,
        }

    def record_operator(
        self, module, input_tensor, reference_work, name, sample_id, sample_key,
        sequence_key, operator_call_index, temporal_steps=10,
    ):
        if self._closed or self._aborted or self._published:
            raise ValueError("M24C writer is not open")
        if self.run_context is None:
            raise ValueError("M24C run context is not bound")
        if int(temporal_steps) != self.temporal_steps:
            raise ValueError("M24C temporal-step configuration drift")
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            raise TypeError("M24C supports only Linear and Conv2d")
        if (
            isinstance(module, torch.nn.Conv2d)
            and str(getattr(module, "padding_mode", "zeros")) != "zeros"
        ):
            raise ValueError(
                "M24C exact Conv2d requires padding_mode='zeros', got {!r}".format(
                    str(getattr(module, "padding_mode", "zeros"))
                )
            )
        value = input_tensor.detach()
        context = {
            "sample_id": int(sample_id), "sample_key": str(sample_key),
            "sequence_key": str(sequence_key), "name": str(name),
            "operator": module.__class__.__name__,
            "operator_call_index": int(operator_call_index),
        }
        if value.ndim < 2 or int(value.shape[0]) != self.temporal_steps:
            self._reject_call(
                context, "TEMPORAL_AXIS_UNQUALIFIED", reference_work
            )
            return
        if int(value.shape[-1]) <= 0:
            raise ValueError("M24C source dimension must be positive")
        if not self._binary_status(value):
            self._reject_call(context, "NON_BINARY_BYPASS", reference_work)
            return
        if not all(row.get("status") == EXACT_STATUS for row in reference_work):
            raise ValueError("M24C exact tensor disagrees with dual-line reference status")
        per_t = [Counter() for _ in range(self.temporal_steps)]
        call_totals = Counter()
        histogram_records_before = self.histogram_records
        if isinstance(module, torch.nn.Linear):
            geometry = self._linear(module, value, context, per_t, call_totals)
        else:
            geometry = self._conv2d(module, value, context, per_t, call_totals)
        self._validate_reference(reference_work, per_t, geometry["fanout"])
        current = sum(item["current_source_count"] for item in per_t)
        positive = sum(item["positive_transition_source_count"] for item in per_t)
        negative = sum(item["negative_transition_source_count"] for item in per_t)
        selected = sum(item["selected_source_count"] for item in per_t)
        if call_totals["local_current_events"] != current:
            raise ValueError("M24C Local histogram/current conservation failed")
        if (
            call_totals["selected_positive_events"]
            + call_totals["selected_negative_events"] != selected
        ):
            raise ValueError("M24C signed histogram/selected conservation failed")
        fanout = int(geometry["fanout"])
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
            "rows": geometry["rows"],
            "source_width": geometry["source_width"],
            "source_chunks": geometry["source_chunks"],
            "weight_groups": geometry["weight_groups"],
            "output_channel_fanout": fanout,
            "valid_source_count": sum(item["valid_source_count"] for item in per_t),
            "current_source_count": current,
            "positive_transition_source_count": positive,
            "negative_transition_source_count": negative,
            "selected_positive_source_count": call_totals["selected_positive_events"],
            "selected_negative_source_count": call_totals["selected_negative_events"],
            "selected_source_count": selected,
            "local_cohort_coefficient_vectors": call_totals["local_cohort_coefficient_vectors"],
            "motion_cohort_coefficient_vectors": call_totals["motion_cohort_coefficient_vectors"],
            "local_cohort_coefficient_scalar_reads": call_totals["local_cohort_coefficient_vectors"] * fanout,
            "motion_cohort_coefficient_scalar_reads": call_totals["motion_cohort_coefficient_vectors"] * fanout,
            "destination_scalar_updates": selected * fanout,
            "histogram_records": self.histogram_records - histogram_records_before,
            "maximum_row_block_size": geometry["maximum_row_block_size"],
            "estimated_peak_working_set_bytes": geometry["estimated_peak_working_set_bytes"],
            "reference_reconciled": True,
        }
        self._reconciliation_writer.writerow(row)
        self.calls += 1
        self.status_counts[EXACT_STATUS] += 1
        self.operator_status_counts[(context["operator"], EXACT_STATUS)] += 1
        for field in (
            "valid_source_count", "current_source_count",
            "positive_transition_source_count", "negative_transition_source_count",
            "selected_positive_source_count", "selected_negative_source_count",
            "selected_source_count", "local_cohort_coefficient_vectors",
            "motion_cohort_coefficient_vectors", "local_cohort_coefficient_scalar_reads",
            "motion_cohort_coefficient_scalar_reads", "destination_scalar_updates",
        ):
            self.total_counts[field] += int(row[field])

    @staticmethod
    def _flush_fsync(handle):
        handle.flush()
        os.fsync(handle.fileno())

    def _close_streams(self):
        if not self._histogram_handle.closed:
            self._flush_fsync(self._histogram_handle)
            self._histogram_handle.close()
        if not self._reconciliation_handle.closed:
            self._flush_fsync(self._reconciliation_handle)
            self._reconciliation_handle.close()

    @staticmethod
    def _write_json_fsync(path, payload):
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
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
            raise ValueError("cannot abort a published M24C census")
        if self._aborted:
            return
        self._close_streams()
        marker = {
            "schema": "m24c_interrupted_run_v1",
            "status": "INTERRUPTED_NOT_ADMITTED",
            "reason": str(reason),
            "calls_before_abort": self.calls,
            "exact_calls_before_abort": int(self.status_counts.get(EXACT_STATUS, 0)),
            "rejected_calls_before_abort": int(sum(
                self.status_counts.get(status, 0) for status in REJECTION_STATUSES
            )),
            "pass_manifest_written": False,
        }
        self._write_json_fsync(self.staging_dir / "INTERRUPTED.json", marker)
        self._fsync_directory(self.staging_dir)
        self._aborted = True
        self._closed = True

    def close(self):
        if self._published:
            return
        if self._aborted:
            raise ValueError("cannot close an aborted M24C census")
        if self.run_context is None or self.calls <= 0:
            raise ValueError("M24C cannot publish without context and calls")
        exact_calls = int(self.status_counts.get(EXACT_STATUS, 0))
        rejected_calls = int(sum(
            self.status_counts.get(status, 0) for status in REJECTION_STATUSES
        ))
        if exact_calls + rejected_calls != self.calls:
            self.abort("call status accounting mismatch")
            raise ValueError("M24C call status accounting mismatch")
        if exact_calls <= 0:
            self.abort("no exact calls were admitted")
            raise ValueError("M24C cannot publish without at least one exact call")
        self._close_streams()
        files = {}
        for path in (self.histogram_path, self.reconciliation_path):
            files[path.name] = {
                "bytes": path.stat().st_size, "sha256": file_sha256(path),
            }
        staged = {
            "schema": "m24c_streaming_cohort_census_manifest_v1",
            "status": "STAGING_NOT_ADMITTED",
            "files": files,
            "calls": self.calls,
            "exact_calls": exact_calls,
            "rejected_calls": rejected_calls,
            "pass_manifest_written": False,
        }
        staged_manifest = self.staging_dir / "manifest.staged.json"
        self._write_json_fsync(staged_manifest, staged)
        self._fsync_directory(self.staging_dir)
        os.replace(str(self.staging_dir), str(self.output_dir))
        self._fsync_directory(self.output_dir.parent)
        published_staged_manifest = self.output_dir / staged_manifest.name
        published_staged_manifest.unlink()
        manifest = {
            "schema": "m24c_streaming_cohort_census_manifest_v1",
            "status": PASS_STATUS,
            "temporal_steps": self.temporal_steps,
            "histogram_domain_bins": HISTOGRAM_BINS,
            "signed_pair_domain_bins": SIGNED_PAIR_BINS,
            "source_chunk_size": self.source_chunk_size,
            "requested_row_block_size": self.requested_row_block_size,
            "max_working_set_bytes": self.max_working_set_bytes,
            "working_set_model": {
                "scope": "additional tensor payloads allocated by the census producer",
                "fixed_bytes": WORKING_SET_FIXED_BYTES,
                "bytes_per_source": WORKING_SET_BYTES_PER_SOURCE,
                "bytes_per_row": WORKING_SET_BYTES_PER_ROW,
                "bytes_per_row_source": WORKING_SET_BYTES_PER_ROW_SOURCE,
                "binary_scan_bytes_per_element": BINARY_SCAN_BYTES_PER_ELEMENT,
                "binary_scan_chunk_elements": self.binary_scan_chunk_elements,
                "binary_scan_estimated_working_set_bytes": (
                    self.binary_scan_estimated_working_set_bytes
                ),
            },
            "maximum_row_block_size": self.maximum_row_block_size,
            "maximum_estimated_working_set_bytes": self.maximum_estimated_working_set_bytes,
            "working_set_bound_satisfied": (
                self.maximum_estimated_working_set_bytes <= self.max_working_set_bytes
            ),
            "calls": self.calls,
            "exact_calls": exact_calls,
            "rejected_calls": rejected_calls,
            "histogram_records": self.histogram_records,
            "status_counts": dict(sorted(self.status_counts.items())),
            "operator_status_counts": {
                "{}:{}".format(operator, status): count
                for (operator, status), count in sorted(self.operator_status_counts.items())
            },
            "exact_totals": dict(sorted(self.total_counts.items())),
            "raw_activation_or_npz_saved": False,
            "all_exact_calls_reference_reconciled": True,
            "files": files,
            "run_context": self.run_context,
            "claim_boundary": (
                "Exact per-call T10 coefficient-mask census for accepted binary Linear/Conv2d "
                "calls. Rejected calls remain explicit and contribute no exact histogram. "
                "Histograms are not cycle, latency, energy, accuracy, or PPA evidence."
            ),
        }
        temporary_manifest = self.output_dir / ".manifest.json.tmp"
        self._write_json_fsync(temporary_manifest, manifest)
        os.replace(str(temporary_manifest), str(self.manifest_path))
        self._fsync_directory(self.output_dir)
        self._published = True
        self._closed = True

    def discard_staging_for_test_only(self):
        """Remove only this writer's unpublished staging directory in tests."""
        if self._published:
            raise ValueError("cannot discard a published census")
        self._close_streams()
        if self.staging_dir.exists():
            shutil.rmtree(str(self.staging_dir))
        self._closed = True
