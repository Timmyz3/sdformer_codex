#!/usr/bin/env python3
"""Run the frozen M328 token-dynamic cumulative-budget CPU fast-kill."""

from __future__ import division

import argparse
from collections import defaultdict, OrderedDict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import platform
import sys
import time
import zipfile

import numpy as np
from numba import njit, prange, set_num_threads


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        answer = {}
        for key, value in items:
            require(key not in answer, "duplicate JSON key: " + key)
            answer[key] = value
        return answer

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def product(values):
    answer = 1
    for value in values:
        answer *= int(value)
    return answer


def normalize_tbc_hw(shape):
    shape = tuple(int(value) for value in shape)
    if len(shape) == 4:
        return (shape[0], 1, shape[1], shape[2], shape[3])
    require(len(shape) == 5, "unsupported Conv shape")
    return shape


def infer_stride(height, width, out_height, out_width):
    candidates = []
    for stride in (1, 2):
        got_height = (height + 2 - 3) // stride + 1
        got_width = (width + 2 - 3) // stride + 1
        if (got_height, got_width) == (out_height, out_width):
            candidates.append(stride)
    require(len(candidates) == 1, "ambiguous Conv stride")
    return candidates[0]


# This constrained loader reconstructs only tensor descriptors and module
# dictionaries from the trusted, SHA-pinned PyTorch zip checkpoint.  It never
# imports or executes checkpoint model classes and avoids a runtime torch
# dependency on the Synopsys host.
class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


class _Stub(metaclass=_StubMeta):
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


class _StorageType:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = np.dtype(dtype)


class _StorageRef:
    def __init__(self, storage_type, key, location, size):
        self.storage_type = storage_type
        self.key = str(key)
        self.location = location
        self.size = int(size)


class _TensorRef:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(value) for value in size)
        self.stride = tuple(int(value) for value in stride)


def _rebuild_tensor(storage, offset, size, stride, *unused):
    del unused
    return _TensorRef(storage, offset, size, stride)


def _rebuild_parameter(tensor, *unused):
    del unused
    return tensor


class _CheckpointUnpickler(pickle.Unpickler):
    def __init__(self, handle):
        super().__init__(handle, encoding="utf-8")
        self.stub_classes = {}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module in ("__builtin__", "builtins") and name == "set":
            return set
        if module in ("__builtin__", "builtins") and name == "getattr":
            return getattr
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in (
                "FloatStorage", "HalfStorage", "LongStorage"):
            dtype = {"FloatStorage": "<f4", "HalfStorage": "<f2",
                     "LongStorage": "<i8"}[name]
            return _StorageType(name, dtype)
        key = (module, name)
        if key not in self.stub_classes:
            self.stub_classes[key] = _StubMeta(name, (_Stub,), {})
        return self.stub_classes[key]

    def persistent_load(self, persistent_id):
        require(len(persistent_id) == 5 and persistent_id[0] == "storage",
                "unsupported checkpoint persistent ID")
        _, storage_type, key, location, size = persistent_id
        return _StorageRef(storage_type, key, location, size)


def _walk_parameters(module, prefix, answer):
    state = getattr(module, "__dict__", {})
    parameters = state.get("_parameters", {})
    if isinstance(parameters, dict):
        for name, tensor in parameters.items():
            if tensor is not None:
                answer[prefix + "." + name] = tensor
    children = state.get("_modules", {})
    if isinstance(children, dict):
        for name, child in children.items():
            if child is not None:
                child_prefix = prefix + "." + name if prefix else name
                _walk_parameters(child, child_prefix, answer)


def load_checkpoint_weights(checkpoint_path, requested):
    with zipfile.ZipFile(str(checkpoint_path), "r") as archive:
        pickle_members = [name for name in archive.namelist()
                          if name.endswith("/data.pkl")]
        require(len(pickle_members) == 1,
                "checkpoint must contain exactly one data.pkl")
        prefix = pickle_members[0][:-len("data.pkl")]
        root = _CheckpointUnpickler(
            io.BytesIO(archive.read(pickle_members[0]))).load()
        parameters = {}
        _walk_parameters(root, "", parameters)
        answer = {}
        for name in requested:
            key = name + ".weight"
            require(key in parameters, "checkpoint missing " + key)
            tensor = parameters[key]
            require(isinstance(tensor, _TensorRef),
                    "unsupported checkpoint tensor for " + key)
            storage = tensor.storage
            member = prefix + "data/" + storage.key
            raw = archive.read(member)
            dtype = storage.storage_type.dtype
            required_bytes = storage.size * dtype.itemsize
            require(len(raw) == required_bytes,
                    "checkpoint storage extent drift for " + key)
            array = np.ndarray(
                tensor.size,
                dtype=dtype,
                buffer=raw,
                offset=tensor.offset * dtype.itemsize,
                strides=tuple(value * dtype.itemsize
                              for value in tensor.stride),
            ).copy()
            answer[name] = array.astype(np.float64, copy=False)
    return answer


def quantize_weight(weight):
    require(weight.ndim in (2, 4), "unsupported weight rank")
    flat = weight.reshape(int(weight.shape[0]), -1)
    row_maximum = np.max(np.abs(flat), axis=1)
    scale = np.where(row_maximum == 0.0, 1.0, row_maximum / 127.0)
    quantized = np.clip(np.rint(flat / scale[:, None]),
                        -127, 127).astype(np.int16)
    require(not bool(np.any(quantized == -128)), "emitted INT8 -128")
    return quantized


def unpack_fc1(record, payload):
    shape = tuple(int(value) for value in record["input_shape"])
    channels = shape[-1]
    require(channels % 8 == 0 and product(shape) ==
            int(record["input_elements"]), "FC1 shape drift")
    packed = np.fromfile(str(payload), dtype=np.uint8)
    require(packed.size == int(record["packed_bytes"]) and
            packed.size * 8 == int(record["input_elements"]),
            "FC1 payload extent drift")
    bits = np.unpackbits(packed.reshape(-1, channels // 8), axis=1,
                         bitorder="little")[:, :channels]
    require(int(bits.sum(dtype=np.uint64)) ==
            int(record["active_elements"]), "FC1 active drift")
    return np.ascontiguousarray(bits, dtype=np.uint8)


def unpack_conv_sources(record, payload):
    shape = normalize_tbc_hw(record["input_shape"])
    out_shape = normalize_tbc_hw(record["output_shape"])
    require(product(shape) == int(record["input_elements"]),
            "Conv input extent drift")
    packed = np.fromfile(str(payload), dtype=np.uint8)
    require(packed.size == int(record["packed_bytes"]),
            "Conv payload byte drift")
    raw_bits = np.unpackbits(packed, bitorder="little")[:product(shape)]
    raw_bits = raw_bits.reshape(shape)
    require(int(raw_bits.sum(dtype=np.uint64)) ==
            int(record["active_elements"]), "Conv input active drift")
    stride = infer_stride(shape[-2], shape[-1],
                          out_shape[-2], out_shape[-1])
    padded = np.pad(raw_bits,
                    ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)))
    tokens = (shape[0] * shape[1] * out_shape[-2] * out_shape[-1])
    channels = shape[2]
    sources = np.empty((tokens, channels * 9), dtype=np.uint8)
    for ky in range(3):
        for kx in range(3):
            tap = ky * 3 + kx
            sampled = padded[
                :, :, :, ky:ky + stride * out_shape[-2]:stride,
                kx:kx + stride * out_shape[-1]:stride,
            ]
            require(sampled.shape[-2:] == out_shape[-2:],
                    "bad Conv receptive-field slice")
            sampled = sampled.transpose(0, 1, 3, 4, 2).reshape(
                tokens, channels)
            sources[:, tap::9] = sampled
    return np.ascontiguousarray(sources), stride


@njit(parallel=True, cache=False)
def analyze_bits(bits, base_banks, quantized, beta, order,
                 group_size, budgets):
    tokens, sources = bits.shape
    groups = beta.shape[0]
    policies = budgets.size
    issued = np.zeros((groups, policies), dtype=np.int64)
    dropped = np.zeros((groups, policies), dtype=np.int64)
    bound_sum = np.zeros((groups, policies), dtype=np.int64)
    bound_max = np.zeros((groups, policies), dtype=np.int64)
    raw_abs_sum = np.zeros((groups, policies), dtype=np.int64)
    raw_abs_max = np.zeros((groups, policies), dtype=np.int64)
    raw_nonzero = np.zeros((groups, policies), dtype=np.int64)
    violations = np.zeros((groups, policies), dtype=np.int64)
    drop_counts = np.zeros((groups, policies, sources), dtype=np.int64)
    for group in prange(groups):
        row_base = group * group_size
        for token in range(tokens):
            baseline = 0
            for bank in range(8):
                if base_banks[token, bank] > baseline:
                    baseline = base_banks[token, bank]
            issued[group, 0] += baseline
            cumulative = 0
            budget_index = 1
            dropped_banks = np.zeros(8, dtype=np.int64)
            raw_error = np.zeros(16, dtype=np.int64)
            for rank in range(sources):
                source = order[group, rank]
                if bits[token, source] == 0:
                    continue
                next_cumulative = cumulative + beta[group, source]
                while (budget_index < policies and
                       next_cumulative > budgets[budget_index]):
                    candidate_issue = 0
                    for bank in range(8):
                        value = (base_banks[token, bank] -
                                 dropped_banks[bank])
                        if value > candidate_issue:
                            candidate_issue = value
                    issued[group, budget_index] += candidate_issue
                    drop_total = 0
                    for bank in range(8):
                        drop_total += dropped_banks[bank]
                    dropped[group, budget_index] += drop_total
                    bound_sum[group, budget_index] += cumulative
                    if cumulative > bound_max[group, budget_index]:
                        bound_max[group, budget_index] = cumulative
                    for row in range(group_size):
                        absolute = raw_error[row]
                        if absolute < 0:
                            absolute = -absolute
                        raw_abs_sum[group, budget_index] += absolute
                        if absolute > raw_abs_max[group, budget_index]:
                            raw_abs_max[group, budget_index] = absolute
                        if absolute != 0:
                            raw_nonzero[group, budget_index] += 1
                        if (absolute > cumulative or
                                cumulative > budgets[budget_index]):
                            violations[group, budget_index] += 1
                    budget_index += 1
                if budget_index >= policies:
                    break
                cumulative = next_cumulative
                dropped_banks[source % 8] += 1
                for row in range(group_size):
                    raw_error[row] += quantized[row_base + row, source]
                for policy in range(budget_index, policies):
                    drop_counts[group, policy, source] += 1
            while budget_index < policies:
                candidate_issue = 0
                for bank in range(8):
                    value = base_banks[token, bank] - dropped_banks[bank]
                    if value > candidate_issue:
                        candidate_issue = value
                issued[group, budget_index] += candidate_issue
                drop_total = 0
                for bank in range(8):
                    drop_total += dropped_banks[bank]
                dropped[group, budget_index] += drop_total
                bound_sum[group, budget_index] += cumulative
                if cumulative > bound_max[group, budget_index]:
                    bound_max[group, budget_index] = cumulative
                for row in range(group_size):
                    absolute = raw_error[row]
                    if absolute < 0:
                        absolute = -absolute
                    raw_abs_sum[group, budget_index] += absolute
                    if absolute > raw_abs_max[group, budget_index]:
                        raw_abs_max[group, budget_index] = absolute
                    if absolute != 0:
                        raw_nonzero[group, budget_index] += 1
                    if (absolute > cumulative or
                            cumulative > budgets[budget_index]):
                        violations[group, budget_index] += 1
                budget_index += 1
    return (issued, dropped, bound_sum, bound_max, raw_abs_sum,
            raw_abs_max, raw_nonzero, violations, drop_counts)


def base_bank_counts(bits):
    counts = np.empty((bits.shape[0], 8), dtype=np.int16)
    for bank in range(8):
        counts[:, bank] = bits[:, bank::8].sum(axis=1, dtype=np.int16)
    return counts


def empty_accumulator(groups, policies, sources):
    return {
        "issued": np.zeros(policies, dtype=np.int64),
        "dropped": np.zeros(policies, dtype=np.int64),
        "bound_sum": np.zeros(policies, dtype=np.int64),
        "bound_max": np.zeros(policies, dtype=np.int64),
        "raw_abs_sum": np.zeros(policies, dtype=np.int64),
        "raw_abs_max": np.zeros(policies, dtype=np.int64),
        "raw_nonzero": np.zeros(policies, dtype=np.int64),
        "violations": np.zeros(policies, dtype=np.int64),
        "drop_counts": np.zeros((groups, policies, sources),
                                dtype=np.int64),
        "tokens": 0,
        "active_sources": 0,
        "active_source_tasks": 0,
        "accumulators": 0,
        "full_scan_overhead": 0,
        "active_list_overhead": 0,
        "metadata_overhead": 0,
    }


def add_record(accumulator, bits, quantized, beta, order, group_size,
               budgets, cost):
    banks = base_bank_counts(bits)
    outputs = analyze_bits(bits, banks, quantized, beta, order,
                           group_size, budgets)
    names = ("issued", "dropped", "bound_sum", "bound_max",
             "raw_abs_sum", "raw_abs_max", "raw_nonzero", "violations")
    for index, name in enumerate(names):
        reduced = (outputs[index].max(axis=0) if name in
                   ("bound_max", "raw_abs_max") else
                   outputs[index].sum(axis=0, dtype=np.int64))
        if name in ("bound_max", "raw_abs_max"):
            accumulator[name] = np.maximum(accumulator[name], reduced)
        else:
            accumulator[name] += reduced
    accumulator["drop_counts"] += outputs[-1]
    groups = beta.shape[0]
    active_per_token = banks.sum(axis=1, dtype=np.int32)
    active = int(active_per_token.sum(dtype=np.int64))
    tokens = int(bits.shape[0])
    accumulator["tokens"] += tokens
    accumulator["active_sources"] += active
    accumulator["active_source_tasks"] += active * groups
    accumulator["accumulators"] += tokens * int(quantized.shape[0])
    accumulator["full_scan_overhead"] += (
        tokens * groups * int(math.ceil(bits.shape[1] /
                                       float(cost["full_width"]))))
    accumulator["active_list_overhead"] += (
        int(((active_per_token + cost["active_width"] - 1) //
             cost["active_width"]).sum(dtype=np.int64)) * groups)
    accumulator["metadata_overhead"] += (
        tokens * groups * int(math.ceil(
            cost["metadata_bytes_per_pair"] * bits.shape[1] /
            float(cost["metadata_bytes_per_cycle"]))))
    return active_per_token


def prepare_group_tables(quantized, group_size):
    outputs, sources = quantized.shape
    require(outputs % group_size == 0,
            "destination group does not divide output")
    groups = outputs // group_size
    beta = np.abs(quantized).reshape(
        groups, group_size, sources).max(axis=1).astype(np.int16)
    order = np.argsort(beta, axis=1, kind="stable").astype(np.int32)
    # Stable sorting over source-ID-ordered columns implements the frozen
    # (beta, source_id) tie-break.
    for group in range(groups):
        row = order[group]
        require(bool(np.all(beta[group, row[:-1]] <=
                            beta[group, row[1:]])), "beta order drift")
        ties = beta[group, row[:-1]] == beta[group, row[1:]]
        require(bool(np.all(row[:-1][ties] < row[1:][ties])),
                "source-ID tie-break drift")
    return beta, order


def policy_rows(name, kind, group_size, budgets, accumulator,
                active_by_source, static_pairs, gates, cost):
    rows = []
    groups = accumulator["drop_counts"].shape[0]
    static_bytes = (static_pairs + 7) // 8
    naive_bytes = (cost["metadata_bytes_per_pair"] * static_pairs)
    metadata_ratio = naive_bytes / float(static_bytes)
    b0_exact = bool(
        int(accumulator["dropped"][0]) == 0 and
        int(accumulator["issued"][0]) > 0 and
        int(accumulator["raw_abs_max"][0]) == 0 and
        int(accumulator["violations"][0]) == 0)
    for policy, budget in enumerate(budgets):
        baseline = int(accumulator["issued"][0])
        candidate = int(accumulator["issued"][policy])
        overhead_full = (0 if int(budget) == 0 else
                         int(accumulator["full_scan_overhead"]))
        overhead_active = (0 if int(budget) == 0 else
                           int(accumulator["active_list_overhead"]))
        overhead_metadata = (0 if int(budget) == 0 else
                             int(accumulator["metadata_overhead"]))
        dropped_counts = accumulator["drop_counts"][:, policy, :]
        active_grid = np.broadcast_to(active_by_source[None, :],
                                      dropped_counts.shape)
        witness = ((dropped_counts > 0) &
                   (dropped_counts < active_grid))
        witness_count = int(witness.sum())
        examples = []
        for group, source in np.argwhere(witness)[:5]:
            examples.append({
                "destination_group": int(group),
                "source_id": int(source),
                "dropped_active_occurrences":
                    int(dropped_counts[group, source]),
                "total_active_occurrences": int(active_by_source[source]),
            })
        ideal_speedup = (baseline / float(candidate)
                         if candidate else None)
        full_total = candidate + overhead_full
        active_total = candidate + overhead_active
        metadata_total = candidate + overhead_metadata
        row = {
            "module": name,
            "kind": kind,
            "destination_group_size": int(group_size),
            "budget": int(budget),
            "tokens": int(accumulator["tokens"]),
            "destination_groups": int(groups),
            "active_source_group_tasks":
                int(accumulator["active_source_tasks"]),
            "dropped_source_group_tasks":
                int(accumulator["dropped"][policy]),
            "dropped_task_fraction": (
                int(accumulator["dropped"][policy]) /
                float(accumulator["active_source_tasks"])),
            "ideal_k8_baseline_issued_cycles": baseline,
            "ideal_k8_candidate_issued_cycles": candidate,
            "ideal_k8_speedup": ideal_speedup,
            "full_domain_scan96_overhead_cycles": overhead_full,
            "full_domain_scan96_total_cycles": full_total,
            "full_domain_scan96_net_speedup": baseline / float(full_total),
            "active_list_lookup8_overhead_cycles": overhead_active,
            "active_list_lookup8_total_cycles": active_total,
            "active_list_lookup8_net_speedup": baseline / float(active_total),
            "metadata_stream16B_overhead_cycles": overhead_metadata,
            "metadata_stream16B_total_cycles": metadata_total,
            "metadata_stream16B_net_speedup":
                baseline / float(metadata_total),
            "configured_budget": int(budget),
            "maximum_observed_cumulative_bound":
                int(accumulator["bound_max"][policy]),
            "sum_observed_cumulative_bound":
                int(accumulator["bound_sum"][policy]),
            "maximum_raw_signed_int8_error_absolute":
                int(accumulator["raw_abs_max"][policy]),
            "sum_raw_signed_int8_error_absolute":
                int(accumulator["raw_abs_sum"][policy]),
            "nonzero_raw_error_accumulators":
                int(accumulator["raw_nonzero"][policy]),
            "bound_violations": int(accumulator["violations"][policy]),
            "dynamic_witness_source_group_pairs": witness_count,
            "dynamic_witness_examples": examples,
            "static_source_group_pairs": int(static_pairs),
            "static_one_bit_mask_bytes": int(static_bytes),
            "naive_dynamic_metadata_bytes": int(naive_bytes),
            "naive_metadata_bytes_over_static_mask_bytes": metadata_ratio,
        }
        row["b0_exact_pass"] = bool(
            b0_exact and
            (int(budget) != 0 or
             (candidate == baseline and overhead_full == 0 and
              overhead_active == 0 and overhead_metadata == 0)))
        row["promotion_gate_pass"] = bool(
            int(budget) > 0 and row["b0_exact_pass"] and
            row["bound_violations"] <= gates["bound_max"] and
            witness_count > 0 and ideal_speedup is not None and
            ideal_speedup >= gates["ideal_min"] and
            row["full_domain_scan96_net_speedup"] >
                gates["full_min"] and
            row["metadata_stream16B_net_speedup"] >
                gates["metadata_min"] and
            metadata_ratio <= gates["metadata_ratio_max"])
        rows.append(row)
    return rows


def aggregate_rows(scope_name, kind, module_rows, group_size, budgets,
                   gates, cost):
    selected = [rows for rows in module_rows.values()
                if int(rows[0]["destination_group_size"]) == group_size]
    answer = []
    for policy, budget in enumerate(budgets):
        source = [rows[policy] for rows in selected]
        baseline = sum(row["ideal_k8_baseline_issued_cycles"]
                       for row in source)
        candidate = sum(row["ideal_k8_candidate_issued_cycles"]
                        for row in source)
        full_overhead = sum(row["full_domain_scan96_overhead_cycles"]
                            for row in source)
        active_overhead = sum(row["active_list_lookup8_overhead_cycles"]
                              for row in source)
        metadata_overhead = sum(row["metadata_stream16B_overhead_cycles"]
                                for row in source)
        static_pairs = sum(row["static_source_group_pairs"]
                           for row in source)
        static_bytes = (static_pairs + 7) // 8
        naive_bytes = cost["metadata_bytes_per_pair"] * static_pairs
        ratio = naive_bytes / float(static_bytes)
        row = {
            "module": scope_name,
            "kind": kind,
            "destination_group_size": int(group_size),
            "budget": int(budget),
            "tokens": sum(row["tokens"] for row in source),
            "destination_groups": sum(row["destination_groups"]
                                      for row in source),
            "active_source_group_tasks":
                sum(row["active_source_group_tasks"] for row in source),
            "dropped_source_group_tasks":
                sum(row["dropped_source_group_tasks"] for row in source),
            "ideal_k8_baseline_issued_cycles": baseline,
            "ideal_k8_candidate_issued_cycles": candidate,
            "ideal_k8_speedup": baseline / float(candidate),
            "full_domain_scan96_overhead_cycles": full_overhead,
            "full_domain_scan96_total_cycles": candidate + full_overhead,
            "full_domain_scan96_net_speedup":
                baseline / float(candidate + full_overhead),
            "active_list_lookup8_overhead_cycles": active_overhead,
            "active_list_lookup8_total_cycles": candidate + active_overhead,
            "active_list_lookup8_net_speedup":
                baseline / float(candidate + active_overhead),
            "metadata_stream16B_overhead_cycles": metadata_overhead,
            "metadata_stream16B_total_cycles":
                candidate + metadata_overhead,
            "metadata_stream16B_net_speedup":
                baseline / float(candidate + metadata_overhead),
            "maximum_observed_cumulative_bound":
                max(row["maximum_observed_cumulative_bound"]
                    for row in source),
            "sum_observed_cumulative_bound":
                sum(row["sum_observed_cumulative_bound"] for row in source),
            "maximum_raw_signed_int8_error_absolute":
                max(row["maximum_raw_signed_int8_error_absolute"]
                    for row in source),
            "sum_raw_signed_int8_error_absolute":
                sum(row["sum_raw_signed_int8_error_absolute"]
                    for row in source),
            "nonzero_raw_error_accumulators":
                sum(row["nonzero_raw_error_accumulators"] for row in source),
            "bound_violations":
                sum(row["bound_violations"] for row in source),
            "dynamic_witness_source_group_pairs":
                sum(row["dynamic_witness_source_group_pairs"]
                    for row in source),
            "static_source_group_pairs": static_pairs,
            "static_one_bit_mask_bytes": static_bytes,
            "naive_dynamic_metadata_bytes": naive_bytes,
            "naive_metadata_bytes_over_static_mask_bytes": ratio,
            "b0_exact_pass": all(row["b0_exact_pass"] for row in source),
        }
        row["dropped_task_fraction"] = (
            row["dropped_source_group_tasks"] /
            float(row["active_source_group_tasks"]))
        row["promotion_gate_pass"] = bool(
            int(budget) > 0 and row["b0_exact_pass"] and
            row["bound_violations"] <= gates["bound_max"] and
            row["dynamic_witness_source_group_pairs"] > 0 and
            row["ideal_k8_speedup"] >= gates["ideal_min"] and
            row["full_domain_scan96_net_speedup"] > gates["full_min"] and
            row["metadata_stream16B_net_speedup"] >
                gates["metadata_min"] and
            ratio <= gates["metadata_ratio_max"])
        answer.append(row)
    return answer


def write_csv(path, rows):
    fields = [
        "module", "kind", "destination_group_size", "budget", "tokens",
        "destination_groups", "active_source_group_tasks",
        "dropped_source_group_tasks", "dropped_task_fraction",
        "ideal_k8_baseline_issued_cycles",
        "ideal_k8_candidate_issued_cycles", "ideal_k8_speedup",
        "full_domain_scan96_overhead_cycles",
        "full_domain_scan96_total_cycles",
        "full_domain_scan96_net_speedup",
        "active_list_lookup8_overhead_cycles",
        "active_list_lookup8_total_cycles",
        "active_list_lookup8_net_speedup",
        "metadata_stream16B_overhead_cycles",
        "metadata_stream16B_total_cycles",
        "metadata_stream16B_net_speedup",
        "maximum_observed_cumulative_bound",
        "maximum_raw_signed_int8_error_absolute", "bound_violations",
        "dynamic_witness_source_group_pairs", "static_source_group_pairs",
        "static_one_bit_mask_bytes", "naive_dynamic_metadata_bytes",
        "naive_metadata_bytes_over_static_mask_bytes", "b0_exact_pass",
        "promotion_gate_pass",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m328_token_dynamic_cumulative_budget_cpu_fast_kill_contract_v1",
            "M328 contract schema drift")
    hw = contract_path.parents[1]
    paths = {}
    identity = {
        "contract": {
            "path": str(contract_path.relative_to(hw)),
            "sha256": sha256(contract_path),
        },
        "analyzer": {
            "path": str(source_path.relative_to(hw)),
            "sha256": source_start,
        },
    }
    for label, spec in contract["identity"].items():
        path = (hw / spec["path"]).resolve()
        require(path.is_file(), "missing M328 input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"],
                "M328 input SHA drift for {}: {}".format(label, digest))
        paths[label] = path
        identity[label] = {"path": spec["path"], "sha256": digest}

    threads = args.threads if args.threads > 0 else min(96, __import__("os").cpu_count())
    set_num_threads(threads)
    manifest = strict_json(paths["m51_manifest"])
    require(manifest["packing"]["layout"] == "C_ORDER_FLAT" and
            manifest["packing"]["bit_order"] == "LITTLE_WITHIN_BYTE" and
            "exact-binary" in manifest["claim_boundary"],
            "M328 M51 packing/scope drift")
    fc1_spec = contract["population"]["fc1"]
    conv_spec = contract["population"]["conv"]
    fc1_records = [record for record in manifest["records"]
                   if fc1_spec["name_substring"] in record["name"]]
    conv_records = [record for record in manifest["records"]
                    if record["name"] == conv_spec["name"]]
    require(len(fc1_records) == int(fc1_spec["records"]) and
            len(set(row["name"] for row in fc1_records)) ==
            int(fc1_spec["modules"]), "M328 FC1 record population drift")
    require(len(conv_records) == int(conv_spec["records"]),
            "M328 Conv record population drift")
    for record in fc1_records + conv_records:
        payload = paths["m51_manifest"].parent / record["relative_path"]
        require(payload.is_file(), "missing M328 payload: " + str(payload))
        require(sha256(payload) == record["file_sha256"],
                "M328 payload SHA drift: " + str(payload))

    module_names = sorted(set(record["name"] for record in fc1_records))
    module_names.append(conv_spec["name"])
    weights = load_checkpoint_weights(paths["checkpoint"], module_names)
    quantized = dict((name, quantize_weight(weights[name]))
                     for name in module_names)
    # Cross-check the torchless checkpoint path against sealed M287/M293 census.
    m287 = strict_json(paths["m287_result"])
    m293 = strict_json(paths["m293_result"])
    sealed_rows = dict((row["module"], row) for row in
                       m287["per_module"] + m293["per_module"])
    for name in module_names:
        for group_size in (4, 16):
            beta, unused = prepare_group_tables(quantized[name], group_size)
            del unused
            for threshold in (48, 96):
                got = int((beta <= threshold).sum())
                expected = int(sealed_rows[name]["groups"][str(group_size)]
                               [str(threshold)]
                               ["static_source_group_pairs_removed"])
                require(got == expected,
                        "M328 torchless quantization census drift: " + name)

    budgets = np.asarray(contract["mechanism"]["budgets"], dtype=np.int32)
    group_sizes = [int(value) for value in
                   contract["mechanism"]["destination_group_sizes"]]
    require(group_sizes == [4, 16] and budgets.tolist() ==
            [0, 16, 32, 64, 128, 256, 512, 1024],
            "M328 grid drift")
    cost_contract = contract["cost_models"]
    cost = {
        "full_width": int(cost_contract["full_domain_scan96"]
                          ["candidates_per_cycle"]),
        "active_width": int(cost_contract["active_list_lookup8"]
                            ["active_sources_per_cycle"]),
        "metadata_bytes_per_pair":
            int(cost_contract["metadata_stream16B"]
                ["bytes_per_source_group_pair"]),
        "metadata_bytes_per_cycle":
            int(cost_contract["metadata_stream16B"]["bytes_per_cycle"]),
    }
    gate_contract = contract["promotion_gates"]
    gates = {
        "bound_max": int(gate_contract["bound_violation_maximum"]),
        "ideal_min": float(gate_contract["minimum_ideal_k8_speedup"]),
        "full_min": float(
            gate_contract["minimum_full_domain_scan96_speedup"]),
        "metadata_min": float(
            gate_contract["minimum_metadata_stream16B_speedup"]),
        "metadata_ratio_max": float(
            gate_contract[
                "maximum_naive_metadata_bytes_over_static_mask_bytes"]),
    }

    records_by_module = defaultdict(list)
    for record in fc1_records:
        records_by_module[record["name"]].append(record)
    for name in records_by_module:
        records_by_module[name].sort(key=lambda row: int(row["sample_id"]))
    conv_records.sort(key=lambda row: int(row["sample_id"]))
    module_rows_by_group = dict((group, {}) for group in group_sizes)
    per_module = []
    fc1_tokens = 0
    fc1_active = 0
    conv_sources = 0
    conv_maximum = 0

    for module_index, name in enumerate(sorted(records_by_module)):
        qweight = quantized[name]
        channels = qweight.shape[1]
        active_by_source = np.zeros(channels, dtype=np.int64)
        accumulators = {}
        tables = {}
        for group_size in group_sizes:
            beta, order = prepare_group_tables(qweight, group_size)
            tables[group_size] = (beta, order)
            accumulators[group_size] = empty_accumulator(
                beta.shape[0], budgets.size, channels)
        for record in records_by_module[name]:
            bits = unpack_fc1(
                record, paths["m51_manifest"].parent /
                record["relative_path"])
            active_by_source += bits.sum(axis=0, dtype=np.int64)
            fc1_tokens += int(bits.shape[0])
            fc1_active += int(bits.sum(dtype=np.uint64))
            for group_size in group_sizes:
                beta, order = tables[group_size]
                add_record(accumulators[group_size], bits, qweight, beta,
                           order, group_size, budgets, cost)
            del bits
        module_entry = {
            "module": name,
            "kind": "fc1",
            "input_sources": int(channels),
            "output_channels": int(qweight.shape[0]),
            "records": len(records_by_module[name]),
            "active_sources": int(active_by_source.sum()),
            "policies": {},
        }
        for group_size in group_sizes:
            static_pairs = channels * (qweight.shape[0] // group_size)
            rows = policy_rows(name, "fc1", group_size, budgets,
                               accumulators[group_size], active_by_source,
                               static_pairs, gates, cost)
            module_entry["policies"][str(group_size)] = rows
            module_rows_by_group[group_size][name] = rows
        per_module.append(module_entry)
        print("M328 FC1 {}/{} {} complete".format(
            module_index + 1, len(records_by_module), name), flush=True)

    require(fc1_tokens == int(fc1_spec["tokens"]) and
            fc1_active == int(fc1_spec["active_sources"]),
            "M328 FC1 token/activity drift")

    conv_name = conv_spec["name"]
    qweight = quantized[conv_name]
    sources = qweight.shape[1]
    require(sources == int(conv_spec["input_channels"]) * 9 and
            qweight.shape[0] == int(conv_spec["output_channels"]),
            "M328 Conv weight geometry drift")
    active_by_source = np.zeros(sources, dtype=np.int64)
    conv_accumulators = {}
    conv_tables = {}
    for group_size in group_sizes:
        beta, order = prepare_group_tables(qweight, group_size)
        conv_tables[group_size] = (beta, order)
        conv_accumulators[group_size] = empty_accumulator(
            beta.shape[0], budgets.size, sources)
    conv_receipts = []
    for index, record in enumerate(conv_records):
        bits, stride = unpack_conv_sources(
            record, paths["m51_manifest"].parent / record["relative_path"])
        active = int(bits.sum(dtype=np.uint64))
        maximum = int(bits.sum(axis=1, dtype=np.int16).max())
        active_by_source += bits.sum(axis=0, dtype=np.int64)
        conv_sources += active
        conv_maximum = max(conv_maximum, maximum)
        conv_receipts.append({
            "sample_id": int(record["sample_id"]),
            "tokens": int(bits.shape[0]),
            "source_contributions": active,
            "maximum_active_sources_per_output": maximum,
            "stride": int(stride),
        })
        for group_size in group_sizes:
            beta, order = conv_tables[group_size]
            add_record(conv_accumulators[group_size], bits, qweight, beta,
                       order, group_size, budgets, cost)
        del bits
        print("M328 Conv {}/{} sample {} complete".format(
            index + 1, len(conv_records), record["sample_id"]), flush=True)
    require(conv_sources == int(conv_spec["source_contributions"]) and
            conv_maximum == int(conv_spec["maximum_active_sources_per_output"]),
            "M328 Conv source population drift")
    conv_entry = {
        "module": conv_name,
        "kind": "conv3x3",
        "input_sources": int(sources),
        "output_channels": int(qweight.shape[0]),
        "records": len(conv_records),
        "source_contributions": conv_sources,
        "maximum_active_sources_per_output": conv_maximum,
        "record_receipts": conv_receipts,
        "policies": {},
    }
    for group_size in group_sizes:
        static_pairs = sources * (qweight.shape[0] // group_size)
        rows = policy_rows(conv_name, "conv3x3", group_size, budgets,
                           conv_accumulators[group_size], active_by_source,
                           static_pairs, gates, cost)
        conv_entry["policies"][str(group_size)] = rows
        module_rows_by_group[group_size][conv_name] = rows
    per_module.append(conv_entry)

    aggregates = {"fc1": {}, "conv": {}, "combined": {}}
    flat_rows = []
    for module in per_module:
        for rows in module["policies"].values():
            flat_rows.extend(rows)
    for group_size in group_sizes:
        fc1_map = dict((name, rows) for name, rows in
                       module_rows_by_group[group_size].items()
                       if name != conv_name)
        conv_map = {conv_name: module_rows_by_group[group_size][conv_name]}
        combined_map = module_rows_by_group[group_size]
        aggregates["fc1"][str(group_size)] = aggregate_rows(
            "__FC1_10_MODULE_AGGREGATE__", "fc1_aggregate", fc1_map,
            group_size, budgets, gates, cost)
        aggregates["conv"][str(group_size)] = aggregate_rows(
            "__SELECTED_CONV_AGGREGATE__", "conv_aggregate", conv_map,
            group_size, budgets, gates, cost)
        aggregates["combined"][str(group_size)] = aggregate_rows(
            "__COMBINED_FC1_PLUS_SELECTED_CONV__", "combined", combined_map,
            group_size, budgets, gates, cost)
        flat_rows.extend(aggregates["fc1"][str(group_size)])
        flat_rows.extend(aggregates["conv"][str(group_size)])
        flat_rows.extend(aggregates["combined"][str(group_size)])

    combined_candidates = [row for rows in aggregates["combined"].values()
                           for row in rows if int(row["budget"]) > 0]
    require(all(row["bound_violations"] == 0 for row in flat_rows),
            "M328 cumulative bound violation")
    require(all(rows[0]["b0_exact_pass"] for rows in
                aggregates["combined"].values()), "M328 B0 exact failure")
    best_ideal = max(combined_candidates,
                     key=lambda row: row["ideal_k8_speedup"])
    best_full = max(combined_candidates,
                    key=lambda row: row["full_domain_scan96_net_speedup"])
    best_active = max(combined_candidates,
                      key=lambda row: row["active_list_lookup8_net_speedup"])
    best_metadata = max(
        combined_candidates,
        key=lambda row: row["metadata_stream16B_net_speedup"])
    passing = [row for row in combined_candidates
               if row["promotion_gate_pass"]]
    witness_total = sum(row["dynamic_witness_source_group_pairs"]
                        for row in combined_candidates)
    fail_reasons = []
    if witness_total == 0:
        fail_reasons.append("NO_DYNAMIC_KEEP_DROP_WITNESS")
    if best_ideal["ideal_k8_speedup"] < gates["ideal_min"]:
        fail_reasons.append("IDEAL_K8_BENEFIT_BELOW_GATE")
    if best_full["full_domain_scan96_net_speedup"] <= gates["full_min"]:
        fail_reasons.append("FULL_DOMAIN_SCAN96_NOT_FASTER")
    if (best_metadata["metadata_stream16B_net_speedup"] <=
            gates["metadata_min"]):
        fail_reasons.append("METADATA_STREAM16B_NOT_FASTER")
    if min(row["naive_metadata_bytes_over_static_mask_bytes"]
           for row in combined_candidates) > gates["metadata_ratio_max"]:
        fail_reasons.append("NAIVE_DYNAMIC_METADATA_RATIO_EXCEEDS_GATE")
    if not passing:
        fail_reasons.append("NO_POLICY_PASSES_ALL_FROZEN_GATES")
    decision = ("GO_CPU_FAST_KILL_PASS" if passing else
                "NO_GO_TOKEN_DYNAMIC_CUMULATIVE_B_AFTER_CPU_FAST_KILL")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / (
        "m328_token_dynamic_cumulative_budget_cpu_fast_kill_r1.json")
    csv_path = args.output_dir / "per_scope_policy_metrics.csv"
    write_csv(csv_path, flat_rows)
    require(sha256(source_path) == source_start,
            "M328 analyzer changed during execution")
    result = {
        "schema": "m328_token_dynamic_cumulative_budget_cpu_fast_kill_v1",
        "status": "PASS_CPU_EXECUTION_FAIL_CLOSED_DECISION",
        "decision": decision,
        "fail_reasons": fail_reasons,
        "identity": identity,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "numba": __import__("numba").__version__,
            "threads": int(threads),
            "elapsed_seconds": time.time() - started,
            "torch_dependency": False,
            "checkpoint_loader": "SHA-pinned constrained torchless tensor loader",
        },
        "mechanism": contract["mechanism"],
        "cost_models": contract["cost_models"],
        "promotion_gates": contract["promotion_gates"],
        "population": {
            "fc1_records": len(fc1_records),
            "fc1_modules": len(records_by_module),
            "fc1_tokens": fc1_tokens,
            "fc1_active_sources": fc1_active,
            "conv_module": conv_name,
            "conv_records": len(conv_records),
            "conv_source_contributions": conv_sources,
            "conv_maximum_active_sources_per_output": conv_maximum,
        },
        "correctness": {
            "b0_exact_all_scopes": True,
            "cumulative_bound_violations": 0,
            "raw_error_domain": "signed INT8 accumulator delta",
            "bound_domain": "sum of per-active-source destination-group max absolute INT8 weight",
        },
        "best_combined_diagnostics": {
            "ideal_k8": best_ideal,
            "full_domain_scan96": best_full,
            "active_list_lookup8": best_active,
            "metadata_stream16B": best_metadata,
            "dynamic_witness_pairs_across_nonzero_policies": witness_total,
            "passing_policy_count": len(passing),
        },
        "aggregates": aggregates,
        "per_module": per_module,
        "claim_boundary": contract["claim_boundary"],
        "admission": {
            "cpu_frozen_trace_fast_kill": True,
            "dynamic_witness": witness_total > 0,
            "integer_bound": True,
            "ideal_k8_proxy": True,
            "hardware_promotion": bool(passing),
            "accuracy": False,
            "modified_forward": False,
            "executable_hardware_cycles": False,
            "rtl": False,
            "synopsys": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    print("M328_RESULT {} decision={} elapsed={:.3f}s".format(
        result_path, decision, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
