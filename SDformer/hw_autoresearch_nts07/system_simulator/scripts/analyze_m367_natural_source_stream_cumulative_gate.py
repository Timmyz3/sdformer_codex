#!/usr/bin/env python3
"""M367 CPU predesign for a natural-source cumulative-budget gate.

The selector consumes active source IDs in ascending source order.  For each
active source, it drops the source iff adding its conservative four-bit beta
upper bound keeps the accumulated bound <= B; a source that does not fit is
kept, but later sources are still considered.  There is no sort, bucket
capture, or bucket drain.  B=0 is a hard bypass.
"""

from __future__ import division

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from numba import njit, prange, set_num_threads


MAX_ACTIVE = 448


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


def load_pinned_m328(path):
    spec = importlib.util.spec_from_file_location("m367_pinned_m328", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot load pinned M328 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_code_tables(quantized, endpoints):
    outputs, sources = quantized.shape
    require(outputs % 4 == 0, "group4 does not divide outputs")
    exact_beta = np.abs(quantized).reshape(
        outputs // 4, 4, sources).max(axis=1).astype(np.int16)
    codes = np.searchsorted(endpoints, exact_beta, side="left").astype(np.int16)
    require(bool(np.all(codes >= 0)) and bool(np.all(codes < 16)),
            "four-bit code overflow")
    upper = endpoints[codes].astype(np.int16)
    require(bool(np.all(upper >= exact_beta)),
            "conservative code understates exact beta")
    return exact_beta, codes, upper


NAMES = (
    "baseline", "active_sum", "scan8", "metadata_bank_conflict_extra",
    "issued", "dropped", "bound_sum", "bound_max", "exact_beta_sum",
    "raw_abs_sum", "raw_abs_max", "raw_nonzero", "violations",
    "ideal_comb8r_cycles", "registered_infinite8r_cycles",
    "registered_d8_8r_cycles", "registered_d16_8r_cycles",
    "registered_d8_banked1r_cycles", "d8_8r_stalls", "d16_8r_stalls",
    "d8_banked1r_stalls", "max_registered_queue", "max_d8_queue",
    "max_d16_queue", "max_banked_d8_queue", "drop_counts",
)


@njit(parallel=True, cache=False)
def analyze_record(bits, banks, quantized, exact_beta, upper, budgets):
    tokens, sources = bits.shape
    groups = upper.shape[0]
    policies = budgets.size
    baseline = np.zeros(groups, dtype=np.int64)
    active_sum = np.zeros(groups, dtype=np.int64)
    scan8 = np.zeros(groups, dtype=np.int64)
    metadata_bank_conflict_extra = np.zeros(groups, dtype=np.int64)
    issued = np.zeros((groups, policies), dtype=np.int64)
    dropped = np.zeros((groups, policies), dtype=np.int64)
    bound_sum = np.zeros((groups, policies), dtype=np.int64)
    bound_max = np.zeros((groups, policies), dtype=np.int64)
    exact_beta_sum = np.zeros((groups, policies), dtype=np.int64)
    raw_abs_sum = np.zeros((groups, policies), dtype=np.int64)
    raw_abs_max = np.zeros((groups, policies), dtype=np.int64)
    raw_nonzero = np.zeros((groups, policies), dtype=np.int64)
    violations = np.zeros((groups, policies), dtype=np.int64)
    ideal_comb8r = np.zeros((groups, policies), dtype=np.int64)
    reg_inf8r = np.zeros((groups, policies), dtype=np.int64)
    reg_d8_8r = np.zeros((groups, policies), dtype=np.int64)
    reg_d16_8r = np.zeros((groups, policies), dtype=np.int64)
    reg_d8_banked = np.zeros((groups, policies), dtype=np.int64)
    stalls_d8 = np.zeros((groups, policies), dtype=np.int64)
    stalls_d16 = np.zeros((groups, policies), dtype=np.int64)
    stalls_banked_d8 = np.zeros((groups, policies), dtype=np.int64)
    max_reg_queue = np.zeros((groups, policies), dtype=np.int64)
    max_d8_queue = np.zeros((groups, policies), dtype=np.int64)
    max_d16_queue = np.zeros((groups, policies), dtype=np.int64)
    max_banked_d8_queue = np.zeros((groups, policies), dtype=np.int64)
    drop_counts = np.zeros((groups, policies, sources), dtype=np.int64)

    for group in prange(groups):
        active_ids = np.empty(sources, dtype=np.int16)
        row_base = group * 4
        for token in range(tokens):
            active = 0
            base_issue = 0
            for bank in range(8):
                active += int(banks[token, bank])
                if int(banks[token, bank]) > base_issue:
                    base_issue = int(banks[token, bank])
            baseline[group] += base_issue
            active_sum[group] += active
            active = 0
            for source in range(sources):
                if bits[token, source] != 0:
                    active_ids[active] = source
                    active += 1
            words = (active + 7) // 8
            scan8[group] += words

            cumulative = np.zeros(policies, dtype=np.int64)
            dropped_exact = np.zeros(policies, dtype=np.int64)
            dropped_count = np.zeros(policies, dtype=np.int64)
            dropped_banks = np.zeros((policies, 8), dtype=np.int16)
            kept_banks = np.zeros((policies, 8), dtype=np.int16)
            raw_error = np.zeros((policies, 4), dtype=np.int64)
            q_comb = np.zeros((policies, 8), dtype=np.int16)
            q_reg = np.zeros((policies, 8), dtype=np.int16)
            q_d8 = np.zeros((policies, 8), dtype=np.int16)
            q_d16 = np.zeros((policies, 8), dtype=np.int16)
            q_banked = np.zeros((policies, 8), dtype=np.int16)
            cyc_comb = np.zeros(policies, dtype=np.int16)
            cyc_reg = np.zeros(policies, dtype=np.int16)
            cyc_d8 = np.zeros(policies, dtype=np.int16)
            cyc_d16 = np.zeros(policies, dtype=np.int16)
            cyc_banked = np.zeros(policies, dtype=np.int16)
            task_stall_d8 = np.zeros(policies, dtype=np.int16)
            task_stall_d16 = np.zeros(policies, dtype=np.int16)
            task_stall_banked = np.zeros(policies, dtype=np.int16)
            task_max_reg = np.zeros(policies, dtype=np.int16)
            task_max_d8 = np.zeros(policies, dtype=np.int16)
            task_max_d16 = np.zeros(policies, dtype=np.int16)
            task_max_banked = np.zeros(policies, dtype=np.int16)

            for word in range(words):
                arrivals = np.zeros((policies, 8), dtype=np.int16)
                metadata_reads = np.zeros(8, dtype=np.int16)
                start = word * 8
                end = start + 8
                if end > active:
                    end = active
                for active_index in range(start, end):
                    source = int(active_ids[active_index])
                    bank = source & 7
                    metadata_reads[bank] += 1
                    value_upper = int(upper[group, source])
                    value_exact = int(exact_beta[group, source])
                    for policy in range(1, policies):
                        if cumulative[policy] + value_upper <= budgets[policy]:
                            cumulative[policy] += value_upper
                            dropped_exact[policy] += value_exact
                            dropped_count[policy] += 1
                            dropped_banks[policy, bank] += 1
                            drop_counts[group, policy, source] += 1
                            for row in range(4):
                                raw_error[policy, row] += int(
                                    quantized[row_base + row, source])
                        else:
                            kept_banks[policy, bank] += 1
                            arrivals[policy, bank] += 1

                read_cycles = 0
                for bank in range(8):
                    if int(metadata_reads[bank]) > read_cycles:
                        read_cycles = int(metadata_reads[bank])
                metadata_bank_conflict_extra[group] += read_cycles - 1

                for policy in range(1, policies):
                    # Ideal 8R combinational accept-and-issue. New arrivals can
                    # consume the current issue slot; the rest remain queued.
                    for bank in range(8):
                        q_comb[policy, bank] += arrivals[policy, bank]
                        if q_comb[policy, bank] > 0:
                            q_comb[policy, bank] -= 1
                    cyc_comb[policy] += 1

                    # Registered infinite queue: issue old state, then accept.
                    for bank in range(8):
                        if q_reg[policy, bank] > 0:
                            q_reg[policy, bank] -= 1
                        q_reg[policy, bank] += arrivals[policy, bank]
                        if q_reg[policy, bank] > task_max_reg[policy]:
                            task_max_reg[policy] = q_reg[policy, bank]
                    cyc_reg[policy] += 1

                    # Registered atomic-word D8/D16. The held input word stalls
                    # until every bank can accept all of its kept lanes.
                    need8 = 0
                    need16 = 0
                    for bank in range(8):
                        if q_d8[policy, bank] > 0:
                            q_d8[policy, bank] -= 1
                        if q_d16[policy, bank] > 0:
                            q_d16[policy, bank] -= 1
                        over8 = int(q_d8[policy, bank]) + int(
                            arrivals[policy, bank]) - 8
                        over16 = int(q_d16[policy, bank]) + int(
                            arrivals[policy, bank]) - 16
                        if over8 > need8:
                            need8 = over8
                        if over16 > need16:
                            need16 = over16
                    if need8 < 0:
                        need8 = 0
                    if need16 < 0:
                        need16 = 0
                    for bank in range(8):
                        remaining8 = int(q_d8[policy, bank]) - need8
                        remaining16 = int(q_d16[policy, bank]) - need16
                        if remaining8 < 0:
                            remaining8 = 0
                        if remaining16 < 0:
                            remaining16 = 0
                        q_d8[policy, bank] = remaining8 + arrivals[policy, bank]
                        q_d16[policy, bank] = remaining16 + arrivals[policy, bank]
                        if q_d8[policy, bank] > task_max_d8[policy]:
                            task_max_d8[policy] = q_d8[policy, bank]
                        if q_d16[policy, bank] > task_max_d16[policy]:
                            task_max_d16[policy] = q_d16[policy, bank]
                    cyc_d8[policy] += 1 + need8
                    cyc_d16[policy] += 1 + need16
                    task_stall_d8[policy] += need8
                    task_stall_d16[policy] += need16

                    # Eight single-read beta banks: read-bank conflicts stretch
                    # the held word before it can be accepted into D8 FIFOs.
                    for bank in range(8):
                        remaining = int(q_banked[policy, bank]) - read_cycles
                        q_banked[policy, bank] = remaining if remaining > 0 else 0
                    need_banked = 0
                    for bank in range(8):
                        over = int(q_banked[policy, bank]) + int(
                            arrivals[policy, bank]) - 8
                        if over > need_banked:
                            need_banked = over
                    if need_banked < 0:
                        need_banked = 0
                    for bank in range(8):
                        remaining = int(q_banked[policy, bank]) - need_banked
                        if remaining < 0:
                            remaining = 0
                        q_banked[policy, bank] = remaining + arrivals[policy, bank]
                        if q_banked[policy, bank] > task_max_banked[policy]:
                            task_max_banked[policy] = q_banked[policy, bank]
                    cyc_banked[policy] += read_cycles + need_banked
                    task_stall_banked[policy] += need_banked

            for policy in range(1, policies):
                ideal_issue = 0
                tail_comb = 0
                tail_reg = 0
                tail_d8 = 0
                tail_d16 = 0
                tail_banked = 0
                for bank in range(8):
                    if kept_banks[policy, bank] > ideal_issue:
                        ideal_issue = int(kept_banks[policy, bank])
                    if q_comb[policy, bank] > tail_comb:
                        tail_comb = int(q_comb[policy, bank])
                    if q_reg[policy, bank] > tail_reg:
                        tail_reg = int(q_reg[policy, bank])
                    if q_d8[policy, bank] > tail_d8:
                        tail_d8 = int(q_d8[policy, bank])
                    if q_d16[policy, bank] > tail_d16:
                        tail_d16 = int(q_d16[policy, bank])
                    if q_banked[policy, bank] > tail_banked:
                        tail_banked = int(q_banked[policy, bank])
                issued[group, policy] += ideal_issue
                dropped[group, policy] += dropped_count[policy]
                bound_sum[group, policy] += cumulative[policy]
                exact_beta_sum[group, policy] += dropped_exact[policy]
                if cumulative[policy] > bound_max[group, policy]:
                    bound_max[group, policy] = cumulative[policy]
                for row in range(4):
                    absolute = int(raw_error[policy, row])
                    if absolute < 0:
                        absolute = -absolute
                    raw_abs_sum[group, policy] += absolute
                    if absolute > raw_abs_max[group, policy]:
                        raw_abs_max[group, policy] = absolute
                    if absolute != 0:
                        raw_nonzero[group, policy] += 1
                    if (absolute > dropped_exact[policy] or
                            dropped_exact[policy] > cumulative[policy] or
                            cumulative[policy] > budgets[policy]):
                        violations[group, policy] += 1
                ideal_comb8r[group, policy] += int(cyc_comb[policy]) + tail_comb
                reg_inf8r[group, policy] += int(cyc_reg[policy]) + tail_reg
                reg_d8_8r[group, policy] += int(cyc_d8[policy]) + tail_d8
                reg_d16_8r[group, policy] += int(cyc_d16[policy]) + tail_d16
                reg_d8_banked[group, policy] += int(
                    cyc_banked[policy]) + tail_banked
                stalls_d8[group, policy] += task_stall_d8[policy]
                stalls_d16[group, policy] += task_stall_d16[policy]
                stalls_banked_d8[group, policy] += task_stall_banked[policy]
                if task_max_reg[policy] > max_reg_queue[group, policy]:
                    max_reg_queue[group, policy] = task_max_reg[policy]
                if task_max_d8[policy] > max_d8_queue[group, policy]:
                    max_d8_queue[group, policy] = task_max_d8[policy]
                if task_max_d16[policy] > max_d16_queue[group, policy]:
                    max_d16_queue[group, policy] = task_max_d16[policy]
                if task_max_banked[policy] > max_banked_d8_queue[group, policy]:
                    max_banked_d8_queue[group, policy] = task_max_banked[policy]

    # B0 hard bypass reproduces the legacy K8 issue timeline.
    issued[:, 0] = baseline
    ideal_comb8r[:, 0] = baseline
    reg_inf8r[:, 0] = baseline
    reg_d8_8r[:, 0] = baseline
    reg_d16_8r[:, 0] = baseline
    reg_d8_banked[:, 0] = baseline
    return (baseline, active_sum, scan8, metadata_bank_conflict_extra,
            issued, dropped, bound_sum, bound_max, exact_beta_sum,
            raw_abs_sum, raw_abs_max, raw_nonzero, violations,
            ideal_comb8r, reg_inf8r, reg_d8_8r, reg_d16_8r,
            reg_d8_banked, stalls_d8, stalls_d16, stalls_banked_d8,
            max_reg_queue, max_d8_queue, max_d16_queue,
            max_banked_d8_queue, drop_counts)


def output_dict(outputs):
    return dict(zip(NAMES, outputs))


def empty_accumulator(groups, policies, sources):
    scalar_group = (groups,)
    policy_group = (groups, policies)
    return {
        "baseline": np.zeros(scalar_group, dtype=np.int64),
        "active_sum": np.zeros(scalar_group, dtype=np.int64),
        "scan8": np.zeros(scalar_group, dtype=np.int64),
        "metadata_bank_conflict_extra": np.zeros(scalar_group, dtype=np.int64),
        "issued": np.zeros(policy_group, dtype=np.int64),
        "dropped": np.zeros(policy_group, dtype=np.int64),
        "bound_sum": np.zeros(policy_group, dtype=np.int64),
        "bound_max": np.zeros(policy_group, dtype=np.int64),
        "exact_beta_sum": np.zeros(policy_group, dtype=np.int64),
        "raw_abs_sum": np.zeros(policy_group, dtype=np.int64),
        "raw_abs_max": np.zeros(policy_group, dtype=np.int64),
        "raw_nonzero": np.zeros(policy_group, dtype=np.int64),
        "violations": np.zeros(policy_group, dtype=np.int64),
        "ideal_comb8r_cycles": np.zeros(policy_group, dtype=np.int64),
        "registered_infinite8r_cycles": np.zeros(policy_group, dtype=np.int64),
        "registered_d8_8r_cycles": np.zeros(policy_group, dtype=np.int64),
        "registered_d16_8r_cycles": np.zeros(policy_group, dtype=np.int64),
        "registered_d8_banked1r_cycles": np.zeros(policy_group, dtype=np.int64),
        "d8_8r_stalls": np.zeros(policy_group, dtype=np.int64),
        "d16_8r_stalls": np.zeros(policy_group, dtype=np.int64),
        "d8_banked1r_stalls": np.zeros(policy_group, dtype=np.int64),
        "max_registered_queue": np.zeros(policy_group, dtype=np.int64),
        "max_d8_queue": np.zeros(policy_group, dtype=np.int64),
        "max_d16_queue": np.zeros(policy_group, dtype=np.int64),
        "max_banked_d8_queue": np.zeros(policy_group, dtype=np.int64),
        "drop_counts": np.zeros((groups, policies, sources), dtype=np.int64),
        "active_by_source": np.zeros(sources, dtype=np.int64),
        "tokens": 0,
        "records": 0,
    }


def add_outputs(accumulator, outputs):
    sum_names = (
        "baseline", "active_sum", "scan8", "metadata_bank_conflict_extra",
        "issued", "dropped", "bound_sum", "exact_beta_sum",
        "raw_abs_sum", "raw_nonzero", "violations", "ideal_comb8r_cycles",
        "registered_infinite8r_cycles", "registered_d8_8r_cycles",
        "registered_d16_8r_cycles", "registered_d8_banked1r_cycles",
        "d8_8r_stalls", "d16_8r_stalls", "d8_banked1r_stalls",
        "drop_counts")
    for name in sum_names:
        accumulator[name] += outputs[name]
    for name in ("bound_max", "raw_abs_max", "max_registered_queue",
                 "max_d8_queue", "max_d16_queue", "max_banked_d8_queue"):
        accumulator[name] = np.maximum(accumulator[name], outputs[name])


def witness_count(drop_counts, active_by_source):
    grid = np.broadcast_to(active_by_source[None, :], drop_counts.shape)
    return int(((drop_counts > 0) & (drop_counts < grid)).sum())


def prefetch_cycles(groups, sources):
    bytes_per_group = (sources + 1) // 2
    return int(groups * ((bytes_per_group + 31) // 32))


def policy_row(module, kind, accumulator, policy, budget, metadata_cycles):
    baseline = int(accumulator["baseline"].sum(dtype=np.int64))
    b0 = policy == 0
    metadata = 0 if b0 else int(metadata_cycles)
    def total(name):
        return int(accumulator[name][:, policy].sum(dtype=np.int64))

    pass_through = baseline + metadata
    ideal_issue = total("issued")
    comb = total("ideal_comb8r_cycles") + metadata
    reg_inf = total("registered_infinite8r_cycles") + metadata
    d8 = total("registered_d8_8r_cycles") + metadata
    d16 = total("registered_d16_8r_cycles") + metadata
    banked_d8 = total("registered_d8_banked1r_cycles") + metadata
    active_tasks = int(accumulator["active_sum"].sum(dtype=np.int64))
    dropped = total("dropped")
    static_pairs = accumulator["drop_counts"].shape[0] * accumulator[
        "drop_counts"].shape[2]
    beta_bytes = (static_pairs + 1) // 2
    mask_bytes = (static_pairs + 7) // 8
    witnesses = witness_count(accumulator["drop_counts"][:, policy, :],
                              accumulator["active_by_source"])
    return {
        "module": module,
        "kind": kind,
        "budget": int(budget),
        "destination_group_size": 4,
        "tokens": int(accumulator["tokens"]),
        "tasks": int(accumulator["tokens"] *
                     accumulator["drop_counts"].shape[0]),
        "active_source_group_tasks": active_tasks,
        "dropped_source_group_tasks": dropped,
        "dropped_fraction": dropped / float(active_tasks),
        "baseline_k8_cycles": baseline,
        "kept_bank_lower_bound_cycles": ideal_issue,
        "ideal_free_compaction_speedup": baseline / float(ideal_issue),
        "natural_scan8_cycles": int(accumulator["scan8"].sum(dtype=np.int64)),
        "passthrough_no_repack_total_cycles": pass_through,
        "passthrough_no_repack_speedup": baseline / float(pass_through),
        "ideal_combinational_8r_total_cycles": comb,
        "ideal_combinational_8r_speedup": baseline / float(comb),
        "registered_infinite_8r_total_cycles": reg_inf,
        "registered_infinite_8r_speedup": baseline / float(reg_inf),
        "registered_d8_8r_total_cycles": d8,
        "registered_d8_8r_speedup": baseline / float(d8),
        "registered_d16_8r_total_cycles": d16,
        "registered_d16_8r_speedup": baseline / float(d16),
        "registered_d8_banked1r_total_cycles": banked_d8,
        "registered_d8_banked1r_speedup": baseline / float(banked_d8),
        "d8_8r_atomic_word_stall_cycles": total("d8_8r_stalls"),
        "d16_8r_atomic_word_stall_cycles": total("d16_8r_stalls"),
        "banked1r_metadata_conflict_extra_cycles": int(
            accumulator["metadata_bank_conflict_extra"].sum(dtype=np.int64)),
        "d8_banked1r_atomic_word_stall_cycles": total(
            "d8_banked1r_stalls"),
        "maximum_infinite_registered_queue_ids_per_bank": int(
            accumulator["max_registered_queue"][:, policy].max()),
        "maximum_d8_queue_ids_per_bank": int(
            accumulator["max_d8_queue"][:, policy].max()),
        "maximum_d16_queue_ids_per_bank": int(
            accumulator["max_d16_queue"][:, policy].max()),
        "maximum_banked_d8_queue_ids_per_bank": int(
            accumulator["max_banked_d8_queue"][:, policy].max()),
        "maximum_conservative_bound": int(
            accumulator["bound_max"][:, policy].max()),
        "sum_conservative_bound": total("bound_sum"),
        "sum_exact_beta_of_dropped": total("exact_beta_sum"),
        "maximum_raw_signed_int8_error_absolute": int(
            accumulator["raw_abs_max"][:, policy].max()),
        "sum_raw_signed_int8_error_absolute": total("raw_abs_sum"),
        "nonzero_raw_error_accumulators": total("raw_nonzero"),
        "bound_violations": total("violations"),
        "dynamic_witness_source_group_pairs": witnesses,
        "metadata_prefetch_cycles": metadata,
        "persistent_beta_metadata_bytes": beta_bytes,
        "one_bit_mask_reference_bytes": mask_bytes,
        "persistent_metadata_ratio": beta_bytes / float(mask_bytes),
        "accuracy": True if b0 else False,
        "b0_exact": bool(not b0 or (
            dropped == 0 and pass_through == baseline and comb == baseline and
            d8 == baseline and total("violations") == 0)),
    }


def aggregate_rows(module_rows, budgets):
    answer = []
    sum_fields = (
        "tokens", "tasks", "active_source_group_tasks",
        "dropped_source_group_tasks", "baseline_k8_cycles",
        "kept_bank_lower_bound_cycles", "natural_scan8_cycles",
        "passthrough_no_repack_total_cycles",
        "ideal_combinational_8r_total_cycles",
        "registered_infinite_8r_total_cycles",
        "registered_d8_8r_total_cycles", "registered_d16_8r_total_cycles",
        "registered_d8_banked1r_total_cycles",
        "d8_8r_atomic_word_stall_cycles", "d16_8r_atomic_word_stall_cycles",
        "banked1r_metadata_conflict_extra_cycles",
        "d8_banked1r_atomic_word_stall_cycles", "sum_conservative_bound",
        "sum_exact_beta_of_dropped", "sum_raw_signed_int8_error_absolute",
        "nonzero_raw_error_accumulators", "bound_violations",
        "dynamic_witness_source_group_pairs", "metadata_prefetch_cycles",
        "persistent_beta_metadata_bytes", "one_bit_mask_reference_bytes")
    for policy, budget in enumerate(budgets):
        rows = [module[policy] for module in module_rows]
        row = {field: sum(int(item[field]) for item in rows)
               for field in sum_fields}
        baseline = row["baseline_k8_cycles"]
        active = row["active_source_group_tasks"]
        row.update({
            "module": "__COMBINED_FC1_PLUS_SELECTED_CONV__",
            "kind": "combined",
            "budget": int(budget),
            "destination_group_size": 4,
            "dropped_fraction": row["dropped_source_group_tasks"] / float(active),
            "ideal_free_compaction_speedup": baseline / float(
                row["kept_bank_lower_bound_cycles"]),
            "passthrough_no_repack_speedup": baseline / float(
                row["passthrough_no_repack_total_cycles"]),
            "ideal_combinational_8r_speedup": baseline / float(
                row["ideal_combinational_8r_total_cycles"]),
            "registered_infinite_8r_speedup": baseline / float(
                row["registered_infinite_8r_total_cycles"]),
            "registered_d8_8r_speedup": baseline / float(
                row["registered_d8_8r_total_cycles"]),
            "registered_d16_8r_speedup": baseline / float(
                row["registered_d16_8r_total_cycles"]),
            "registered_d8_banked1r_speedup": baseline / float(
                row["registered_d8_banked1r_total_cycles"]),
            "maximum_infinite_registered_queue_ids_per_bank": max(int(
                item["maximum_infinite_registered_queue_ids_per_bank"])
                for item in rows),
            "maximum_d8_queue_ids_per_bank": max(int(
                item["maximum_d8_queue_ids_per_bank"]) for item in rows),
            "maximum_d16_queue_ids_per_bank": max(int(
                item["maximum_d16_queue_ids_per_bank"]) for item in rows),
            "maximum_banked_d8_queue_ids_per_bank": max(int(
                item["maximum_banked_d8_queue_ids_per_bank"]) for item in rows),
            "maximum_conservative_bound": max(int(
                item["maximum_conservative_bound"]) for item in rows),
            "maximum_raw_signed_int8_error_absolute": max(int(
                item["maximum_raw_signed_int8_error_absolute"]) for item in rows),
            "persistent_metadata_ratio": row[
                "persistent_beta_metadata_bytes"] / float(
                    row["one_bit_mask_reference_bytes"]),
            "accuracy": policy == 0,
            "b0_exact": all(bool(item["b0_exact"]) for item in rows),
        })
        answer.append(row)
    return answer


def write_csv(path, rows):
    fields = list(rows[0].keys())
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
    require(not args.output_dir.exists(), "refusing M367 output overwrite")
    started = time.time()
    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m367_natural_source_stream_cumulative_gate_contract_v1",
            "M367 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_CPU_EXECUTION",
            "M367 contract not frozen")
    require(contract["analyzer"]["sha256"] == source_start,
            "M367 analyzer/contract SHA drift")
    hw = contract_path.parents[1]
    paths = {}
    identity = {
        "contract": {"path": str(contract_path.relative_to(hw)),
                     "sha256": sha256(contract_path)},
        "analyzer": {"path": str(source_path.relative_to(hw)),
                     "sha256": source_start},
    }
    for label, spec in contract["identity"].items():
        path = (hw / spec["path"]).resolve()
        require(path.is_file(), "missing M367 input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"],
                "M367 input SHA drift for {}: {}".format(label, digest))
        paths[label] = path
        identity[label] = {"path": spec["path"], "sha256": digest}

    base = load_pinned_m328(paths["m328_analyzer"])
    threads = args.threads if args.threads > 0 else min(
        96, __import__("os").cpu_count())
    set_num_threads(threads)
    manifest = strict_json(paths["m51_manifest"])
    spec_fc = contract["population"]["fc1"]
    spec_conv = contract["population"]["conv"]
    fc_records = [row for row in manifest["records"]
                  if spec_fc["name_substring"] in row["name"]]
    conv_records = [row for row in manifest["records"]
                    if row["name"] == spec_conv["name"]]
    require(len(fc_records) == 100 and len(conv_records) == 10,
            "M367 population drift")
    for record in fc_records + conv_records:
        payload = paths["m51_manifest"].parent / record["relative_path"]
        require(payload.is_file() and sha256(payload) == record["file_sha256"],
                "M367 payload SHA drift")
    records_by_module = defaultdict(list)
    for record in fc_records:
        records_by_module[record["name"]].append(record)
    for name in records_by_module:
        records_by_module[name].sort(key=lambda row: int(row["sample_id"]))
    conv_records.sort(key=lambda row: int(row["sample_id"]))
    module_names = sorted(records_by_module)
    module_names.append(spec_conv["name"])
    weights = base.load_checkpoint_weights(paths["checkpoint"], module_names)
    quantized = {name: base.quantize_weight(weights[name])
                 for name in module_names}
    endpoints = np.asarray(contract["mechanism"]["upper_bound_codebook"],
                           dtype=np.int16)
    budgets = np.asarray(contract["mechanism"]["budgets"], dtype=np.int32)
    require(endpoints.tolist() ==
            [0, 9, 17, 26, 34, 43, 51, 60, 68, 77, 85, 94, 102, 111, 119, 127],
            "M367 endpoint drift")
    require(budgets.tolist() == [0, 16, 32, 64, 128, 256, 512, 1024],
            "M367 budget drift")

    per_module = []
    module_policy_rows = []
    for module_index, name in enumerate(module_names):
        qweight = quantized[name]
        exact_beta, unused_codes, upper = prepare_code_tables(qweight, endpoints)
        del unused_codes
        groups, sources = upper.shape
        accumulator = empty_accumulator(groups, budgets.size, sources)
        records = conv_records if name == spec_conv["name"] else records_by_module[name]
        kind = "conv3x3" if name == spec_conv["name"] else "fc1"
        for record in records:
            payload = paths["m51_manifest"].parent / record["relative_path"]
            if kind == "conv3x3":
                bits, unused_stride = base.unpack_conv_sources(record, payload)
                del unused_stride
            else:
                bits = base.unpack_fc1(record, payload)
            banks = base.base_bank_counts(bits)
            outputs = output_dict(analyze_record(
                bits, banks, qweight, exact_beta, upper, budgets))
            add_outputs(accumulator, outputs)
            accumulator["active_by_source"] += bits.sum(axis=0, dtype=np.int64)
            accumulator["tokens"] += int(bits.shape[0])
            accumulator["records"] += 1
            del bits, banks, outputs
        metadata = prefetch_cycles(groups, sources)
        policies = [policy_row(name, kind, accumulator, policy, budget,
                               metadata)
                    for policy, budget in enumerate(budgets.tolist())]
        module_policy_rows.append(policies)
        per_module.append({
            "module": name, "kind": kind, "sources": int(sources),
            "outputs": int(qweight.shape[0]), "groups": int(groups),
            "records": int(accumulator["records"]),
            "tokens": int(accumulator["tokens"]),
            "tasks": int(accumulator["tokens"] * groups),
            "metadata_bytes": int((groups * sources + 1) // 2),
            "metadata_prefetch_cycles": int(metadata),
            "policies": policies,
        })
        print("M367 module {}/{} complete".format(
            module_index + 1, len(module_names)), flush=True)

    combined = aggregate_rows(module_policy_rows, budgets.tolist())
    m328 = strict_json(paths["m328_result"])
    m341 = strict_json(paths["m341_result"])
    expected_baseline = int(m328["aggregates"]["combined"]["4"][0]
                            ["ideal_k8_baseline_issued_cycles"])
    require(combined[0]["baseline_k8_cycles"] == expected_baseline ==
            int(m341["aggregates"]["combined"][0]["baseline_k8_cycles"]),
            "M367 B0 baseline mismatch")
    require(combined[0]["b0_exact"], "M367 B0 bypass failure")
    require(all(row["bound_violations"] == 0 for row in combined),
            "M367 bound violation")
    require(combined[0]["tasks"] == 1013760000,
            "M367 combined task-count drift")
    nonzero = combined[1:]
    best_ideal = max(nonzero, key=lambda row: row[
        "ideal_combinational_8r_speedup"])
    best_d8 = max(nonzero, key=lambda row: row[
        "registered_d8_8r_speedup"])
    best_banked = max(nonzero, key=lambda row: row[
        "registered_d8_banked1r_speedup"])
    bounded_candidates = [row for row in nonzero
                          if int(row["budget"]) in (256, 512)]
    a800_candidates = [row for row in bounded_candidates
                       if row["registered_d8_8r_speedup"] >= 1.15]
    decision = ("GO_A800_VALID_ONE_NATURAL_STREAM_CANDIDATE_ONLY"
                if a800_candidates else
                "NO_GO_A800_VALID_AFTER_NATURAL_STREAM_FAST_KILL")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.output_dir / "per_module_policy_metrics.csv"
    aggregate_csv = args.output_dir / "aggregate_policy_metrics.csv"
    write_csv(csv_path, [row for rows in module_policy_rows for row in rows])
    write_csv(aggregate_csv, combined)
    require(sha256(source_path) == source_start,
            "M367 analyzer changed during execution")
    result = {
        "schema": "m367_natural_source_stream_cumulative_gate_v1",
        "status": "PASS_CPU_PREDESIGN_FAIL_CLOSED_DECISION",
        "decision": decision,
        "identity": identity,
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "numba": __import__("numba").__version__,
            "threads": int(threads), "elapsed_seconds": time.time() - started,
            "torch_dependency": False,
            "checkpoint_loader": "pinned M328 constrained torchless loader",
        },
        "mechanism": contract["mechanism"],
        "hardware_models": contract["hardware_models"],
        "population": contract["population"],
        "correctness": {
            "b0_hard_bypass_reproduces_m328_m341": True,
            "cumulative_bound_violations": 0,
            "raw_error_proof": "abs(sum dropped Wq_j)<=sum exact beta<=sum conservative upper<=B",
        },
        "per_module": per_module,
        "aggregates": {"combined": combined},
        "best_diagnostics": {
            "ideal_combinational_8r": best_ideal,
            "registered_d8_8r": best_d8,
            "registered_d8_banked1r": best_banked,
            "a800_budget_256_512_candidates": a800_candidates,
        },
        "admission": {
            "cpu_frozen_trace_opportunity": True,
            "integer_raw_error_bound": True,
            "m341_uncovered_route": True,
            "accuracy": False,
            "a800_valid": bool(a800_candidates),
            "rtl": False, "vcs": False, "synopsys": False,
            "executable_hardware_cycle": False,
            "area_matched": False, "energy": False,
            "system_speedup": False, "headline": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "output_files": {
            "per_module_policy_metrics": csv_path.name,
            "aggregate_policy_metrics": aggregate_csv.name,
        },
    }
    result_path = args.output_dir / (
        "m367_natural_source_stream_cumulative_gate_r1.json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("M367_PASS decision={} B256_D8={:.6f}x B512_D8={:.6f}x".format(
        decision,
        next(row for row in combined if row["budget"] == 256)[
            "registered_d8_8r_speedup"],
        next(row for row in combined if row["budget"] == 512)[
            "registered_d8_8r_speedup"]), flush=True)


if __name__ == "__main__":
    main()
