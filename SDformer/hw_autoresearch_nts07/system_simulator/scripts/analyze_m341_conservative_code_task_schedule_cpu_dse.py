#!/usr/bin/env python3
"""Run the frozen M341 conservative-code task-schedule CPU DSE."""

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
MAX_WORDS = 70
MAX_STAGE2 = 518
SOURCE_ID_BITS = 10


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
    spec = importlib.util.spec_from_file_location("m341_pinned_m328", str(path))
    require(spec is not None and spec.loader is not None,
            "cannot load pinned M328 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile_from_histogram(histogram, percentile):
    total = int(np.sum(histogram, dtype=np.int64))
    if total == 0:
        return None
    target = int(math.ceil(percentile * total))
    running = 0
    for value, count in enumerate(histogram.tolist()):
        running += int(count)
        if running >= target:
            return int(value)
    raise RuntimeError("histogram percentile fell through")


def histogram_summary(histogram):
    nonzero = np.flatnonzero(histogram)
    return {
        "count": int(np.sum(histogram, dtype=np.int64)),
        "mean": (float(np.dot(np.arange(histogram.size, dtype=np.float64),
                              histogram.astype(np.float64)) /
                       np.sum(histogram, dtype=np.float64))
                 if nonzero.size else None),
        "p50": percentile_from_histogram(histogram, 0.50),
        "p90": percentile_from_histogram(histogram, 0.90),
        "p99": percentile_from_histogram(histogram, 0.99),
        "maximum": int(nonzero[-1]) if nonzero.size else None,
    }


def prepare_code_tables(quantized, group_size, endpoints):
    outputs, sources = quantized.shape
    require(outputs % group_size == 0, "group size does not divide output")
    exact_beta = np.abs(quantized).reshape(
        outputs // group_size, group_size, sources).max(axis=1).astype(np.int16)
    codes = np.searchsorted(endpoints, exact_beta, side="left").astype(np.int16)
    require(bool(np.all(codes >= 0)) and bool(np.all(codes < 16)),
            "four-bit code overflow")
    upper = endpoints[codes].astype(np.int16)
    require(bool(np.all(upper >= exact_beta)), "conservative code understates beta")
    source_ids = np.broadcast_to(
        np.arange(sources, dtype=np.int32)[None, :], codes.shape)
    composite = codes.astype(np.int32) * (sources + 1) + source_ids
    order = np.argsort(composite, axis=1, kind="stable").astype(np.int32)
    for group in range(order.shape[0]):
        row = order[group]
        require(bool(np.all(codes[group, row[:-1]] <=
                            codes[group, row[1:]])), "code order drift")
        ties = codes[group, row[:-1]] == codes[group, row[1:]]
        require(bool(np.all(row[:-1][ties] < row[1:][ties])),
                "source-ID stability drift")
    return exact_beta, codes, upper, order


def empty_schedule_state(groups, policies):
    return {
        "capture_end": np.zeros((groups, policies), dtype=np.int64),
        "stage2_end_prev": np.zeros((groups, policies), dtype=np.int64),
        "stage2_end_prev2": np.zeros((groups, policies), dtype=np.int64),
        "context_wait": np.zeros((groups, policies), dtype=np.int64),
        "stage2_wait": np.zeros((groups, policies), dtype=np.int64),
        "first_capture": np.full((groups, policies), -1, dtype=np.int64),
        "last_stage2": np.zeros((groups, policies), dtype=np.int64),
    }


@njit(parallel=True, cache=False)
def analyze_record(bits, banks, quantized, exact_beta, codes, upper, order,
                   budgets, capture_end, stage2_end_prev,
                   stage2_end_prev2, context_wait, stage2_wait,
                   first_capture, last_stage2):
    tokens, sources = bits.shape
    groups = upper.shape[0]
    policies = budgets.size

    baseline = np.zeros(groups, dtype=np.int64)
    active_sum = np.zeros(groups, dtype=np.int64)
    issued = np.zeros((groups, policies), dtype=np.int64)
    dropped = np.zeros((groups, policies), dtype=np.int64)
    bound_sum = np.zeros((groups, policies), dtype=np.int64)
    bound_max = np.zeros((groups, policies), dtype=np.int64)
    exact_beta_sum = np.zeros((groups, policies), dtype=np.int64)
    raw_abs_sum = np.zeros((groups, policies), dtype=np.int64)
    raw_abs_max = np.zeros((groups, policies), dtype=np.int64)
    raw_nonzero = np.zeros((groups, policies), dtype=np.int64)
    violations = np.zeros((groups, policies), dtype=np.int64)
    drop_counts = np.zeros((groups, policies, sources), dtype=np.int64)

    capture_sum = np.zeros(groups, dtype=np.int64)
    drain_sum = np.zeros(groups, dtype=np.int64)
    fragmentation_sum = np.zeros(groups, dtype=np.int64)
    stage2_sum = np.zeros((groups, policies), dtype=np.int64)
    reservoir_stall_sum = np.zeros((groups, policies), dtype=np.int64)
    tail_sum = np.zeros((groups, policies), dtype=np.int64)
    stream_penalty_sum = np.zeros((groups, policies), dtype=np.int64)
    lane_idle_sum = np.zeros((groups, policies), dtype=np.int64)
    pipeline_delta = np.zeros((groups, policies), dtype=np.int64)
    capacity_violations = np.zeros(groups, dtype=np.int64)
    max_bucket_occupancy = np.zeros(groups, dtype=np.int64)
    max_reservoir_occupancy = np.zeros((groups, policies), dtype=np.int64)

    active_hist = np.zeros((groups, MAX_ACTIVE + 1), dtype=np.int64)
    capture_hist = np.zeros((groups, MAX_ACTIVE // 8 + 1), dtype=np.int64)
    drain_hist = np.zeros((groups, MAX_WORDS + 1), dtype=np.int64)
    fragmentation_hist = np.zeros((groups, 15), dtype=np.int64)
    cutoff_hist = np.zeros((groups, policies, MAX_ACTIVE + 1), dtype=np.int64)
    issue_hist = np.zeros((groups, policies, MAX_ACTIVE + 1), dtype=np.int64)
    stage2_hist = np.zeros((groups, policies, MAX_STAGE2 + 1), dtype=np.int64)

    for group in prange(groups):
        active_ids = np.empty(sources, dtype=np.int16)
        word_ids = np.empty((128, 8), dtype=np.int16)
        word_lengths = np.empty(128, dtype=np.int16)
        word_starts = np.empty(128, dtype=np.int16)
        cutoffs = np.empty(policies, dtype=np.int16)
        snapshot_bound = np.empty(policies, dtype=np.int64)
        snapshot_exact_beta = np.empty(policies, dtype=np.int64)
        snapshot_banks = np.empty((policies, 8), dtype=np.int16)
        snapshot_raw = np.empty((policies, 4), dtype=np.int64)
        dropped_banks = np.empty(8, dtype=np.int16)
        raw_error = np.empty(4, dtype=np.int64)
        reservoir = np.empty(8, dtype=np.int16)
        add_banks = np.empty(8, dtype=np.int16)
        pipeline_before = stage2_end_prev[group].copy()
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
            if active > MAX_ACTIVE:
                capacity_violations[group] += 1
            else:
                active_hist[group, active] += 1

            word_count = 0
            word_length = 0
            current_code = -1
            active_index = 0
            bucket_occupancy = 0
            prior_code = -1
            for rank in range(sources):
                source = int(order[group, rank])
                if bits[token, source] == 0:
                    continue
                code = int(codes[group, source])
                active_ids[active_index] = source
                active_index += 1
                if code != prior_code:
                    bucket_occupancy = 0
                    prior_code = code
                bucket_occupancy += 1
                if bucket_occupancy > max_bucket_occupancy[group]:
                    max_bucket_occupancy[group] = bucket_occupancy
                if code != current_code:
                    if word_length != 0:
                        word_lengths[word_count] = word_length
                        word_count += 1
                        word_length = 0
                    current_code = code
                if word_length == 0:
                    word_starts[word_count] = active_index - 1
                word_ids[word_count, word_length] = source
                word_length += 1
                if word_length == 8:
                    word_lengths[word_count] = 8
                    word_count += 1
                    word_length = 0
            if word_length != 0:
                word_lengths[word_count] = word_length
                word_count += 1
            if active_index != active or word_count > MAX_WORDS:
                capacity_violations[group] += 1

            capture_cycles = (active + 7) // 8
            drain_cycles = word_count
            fragmentation = drain_cycles - capture_cycles
            capture_sum[group] += capture_cycles
            drain_sum[group] += drain_cycles
            fragmentation_sum[group] += fragmentation
            if capture_cycles < capture_hist.shape[1]:
                capture_hist[group, capture_cycles] += 1
            else:
                capacity_violations[group] += 1
            if drain_cycles <= MAX_WORDS:
                drain_hist[group, drain_cycles] += 1
            else:
                capacity_violations[group] += 1
            if fragmentation >= 0 and fragmentation < 15:
                fragmentation_hist[group, fragmentation] += 1
            else:
                capacity_violations[group] += 1

            issued[group, 0] += base_issue
            cutoff_hist[group, 0, 0] += 1
            issue_hist[group, 0, base_issue] += 1
            stage2_hist[group, 0, base_issue] += 1
            stage2_sum[group, 0] += base_issue
            lane_idle_sum[group, 0] += base_issue * 8 - active

            for bank in range(8):
                dropped_banks[bank] = 0
            for row in range(4):
                raw_error[row] = 0
            cumulative_upper = 0
            cumulative_exact = 0
            policy = 1
            for index in range(active):
                source = int(active_ids[index])
                next_upper = cumulative_upper + int(upper[group, source])
                while policy < policies and next_upper > int(budgets[policy]):
                    cutoffs[policy] = index
                    snapshot_bound[policy] = cumulative_upper
                    snapshot_exact_beta[policy] = cumulative_exact
                    for bank in range(8):
                        snapshot_banks[policy, bank] = dropped_banks[bank]
                    for row in range(4):
                        snapshot_raw[policy, row] = raw_error[row]
                    policy += 1
                if policy >= policies:
                    break
                cumulative_upper = next_upper
                cumulative_exact += int(exact_beta[group, source])
                dropped_banks[source % 8] += 1
                for row in range(4):
                    raw_error[row] += int(quantized[row_base + row, source])
                for pending in range(policy, policies):
                    drop_counts[group, pending, source] += 1
            while policy < policies:
                cutoffs[policy] = active
                snapshot_bound[policy] = cumulative_upper
                snapshot_exact_beta[policy] = cumulative_exact
                for bank in range(8):
                    snapshot_banks[policy, bank] = dropped_banks[bank]
                for row in range(4):
                    snapshot_raw[policy, row] = raw_error[row]
                policy += 1

            for policy in range(1, policies):
                cutoff = int(cutoffs[policy])
                candidate_issue = 0
                for bank in range(8):
                    value = int(banks[token, bank]) - int(snapshot_banks[policy, bank])
                    if value > candidate_issue:
                        candidate_issue = value
                kept = active - cutoff
                issued[group, policy] += candidate_issue
                dropped[group, policy] += cutoff
                bound = int(snapshot_bound[policy])
                exact_bound = int(snapshot_exact_beta[policy])
                bound_sum[group, policy] += bound
                exact_beta_sum[group, policy] += exact_bound
                if bound > bound_max[group, policy]:
                    bound_max[group, policy] = bound
                if cutoff <= MAX_ACTIVE:
                    cutoff_hist[group, policy, cutoff] += 1
                else:
                    capacity_violations[group] += 1
                issue_hist[group, policy, candidate_issue] += 1
                for row in range(4):
                    absolute = int(snapshot_raw[policy, row])
                    if absolute < 0:
                        absolute = -absolute
                    raw_abs_sum[group, policy] += absolute
                    if absolute > raw_abs_max[group, policy]:
                        raw_abs_max[group, policy] = absolute
                    if absolute != 0:
                        raw_nonzero[group, policy] += 1
                    if (absolute > exact_bound or exact_bound > bound or
                            bound > int(budgets[policy])):
                        violations[group, policy] += 1

                for bank in range(8):
                    reservoir[bank] = 0
                stage2_cycles = 0
                reservoir_stalls = 0
                maximum_reservoir = 0
                for word in range(word_count):
                    for bank in range(8):
                        add_banks[bank] = 0
                    word_start = int(word_starts[word])
                    length = int(word_lengths[word])
                    first_kept = cutoff - word_start
                    if first_kept < 0:
                        first_kept = 0
                    if first_kept > length:
                        first_kept = length
                    word_kept = 0
                    for slot in range(first_kept, length):
                        source = int(word_ids[word, slot])
                        add_banks[source % 8] += 1
                        word_kept += 1
                    delay = 1
                    while True:
                        occupancy_after_issue = 0
                        for bank in range(8):
                            remaining = int(reservoir[bank]) - delay
                            if remaining > 0:
                                occupancy_after_issue += remaining
                        if occupancy_after_issue + word_kept <= 16:
                            break
                        delay += 1
                        if delay > 16:
                            capacity_violations[group] += 1
                            break
                    stage2_cycles += delay
                    reservoir_stalls += delay - 1
                    occupancy = 0
                    for bank in range(8):
                        remaining = int(reservoir[bank]) - delay
                        if remaining < 0:
                            remaining = 0
                        reservoir[bank] = remaining + add_banks[bank]
                        occupancy += int(reservoir[bank])
                    if occupancy > maximum_reservoir:
                        maximum_reservoir = occupancy
                    if occupancy > 16:
                        capacity_violations[group] += 1
                tail = 0
                for bank in range(8):
                    if int(reservoir[bank]) > tail:
                        tail = int(reservoir[bank])
                stage2_cycles += tail
                if stage2_cycles > MAX_STAGE2:
                    capacity_violations[group] += 1
                else:
                    stage2_hist[group, policy, stage2_cycles] += 1
                if maximum_reservoir > max_reservoir_occupancy[group, policy]:
                    max_reservoir_occupancy[group, policy] = maximum_reservoir
                stage2_sum[group, policy] += stage2_cycles
                reservoir_stall_sum[group, policy] += reservoir_stalls
                tail_sum[group, policy] += tail
                reference = drain_cycles
                if candidate_issue > reference:
                    reference = candidate_issue
                stream_penalty_sum[group, policy] += stage2_cycles - reference
                lane_idle_sum[group, policy] += stage2_cycles * 8 - kept

                capture_start = int(capture_end[group, policy])
                if int(stage2_end_prev2[group, policy]) > capture_start:
                    capture_start = int(stage2_end_prev2[group, policy])
                context_wait[group, policy] += (
                    capture_start - int(capture_end[group, policy]))
                new_capture_end = capture_start + capture_cycles
                service_start = int(stage2_end_prev[group, policy])
                if new_capture_end > service_start:
                    service_start = new_capture_end
                stage2_wait[group, policy] += (
                    service_start - int(stage2_end_prev[group, policy]))
                new_stage2_end = service_start + stage2_cycles
                capture_end[group, policy] = new_capture_end
                stage2_end_prev2[group, policy] = stage2_end_prev[group, policy]
                stage2_end_prev[group, policy] = new_stage2_end
                if first_capture[group, policy] < 0:
                    first_capture[group, policy] = capture_cycles
                last_stage2[group, policy] = stage2_cycles

        for policy in range(1, policies):
            pipeline_delta[group, policy] = (
                stage2_end_prev[group, policy] - pipeline_before[policy])

    return (
        baseline, active_sum, issued, dropped, bound_sum, bound_max,
        exact_beta_sum, raw_abs_sum, raw_abs_max, raw_nonzero, violations,
        drop_counts, capture_sum, drain_sum, fragmentation_sum, stage2_sum,
        reservoir_stall_sum, tail_sum, stream_penalty_sum, lane_idle_sum,
        pipeline_delta, capacity_violations, max_bucket_occupancy,
        max_reservoir_occupancy, active_hist, capture_hist, drain_hist,
        fragmentation_hist, cutoff_hist, issue_hist, stage2_hist)


OUTPUT_NAMES = (
    "baseline", "active_sum", "issued", "dropped", "bound_sum",
    "bound_max", "exact_beta_sum", "raw_abs_sum", "raw_abs_max",
    "raw_nonzero", "violations", "drop_counts", "capture_sum",
    "drain_sum", "fragmentation_sum", "stage2_sum",
    "reservoir_stall_sum", "tail_sum", "stream_penalty_sum",
    "lane_idle_sum", "pipeline_delta", "capacity_violations",
    "max_bucket_occupancy", "max_reservoir_occupancy", "active_hist",
    "capture_hist", "drain_hist", "fragmentation_hist", "cutoff_hist",
    "issue_hist", "stage2_hist")


def output_dict(outputs):
    return dict(zip(OUTPUT_NAMES, outputs))


def empty_module_accumulator(groups, policies, sources):
    return {
        "baseline": np.zeros(groups, dtype=np.int64),
        "active_sum": np.zeros(groups, dtype=np.int64),
        "issued": np.zeros((groups, policies), dtype=np.int64),
        "dropped": np.zeros((groups, policies), dtype=np.int64),
        "bound_sum": np.zeros((groups, policies), dtype=np.int64),
        "bound_max": np.zeros((groups, policies), dtype=np.int64),
        "exact_beta_sum": np.zeros((groups, policies), dtype=np.int64),
        "raw_abs_sum": np.zeros((groups, policies), dtype=np.int64),
        "raw_abs_max": np.zeros((groups, policies), dtype=np.int64),
        "raw_nonzero": np.zeros((groups, policies), dtype=np.int64),
        "violations": np.zeros((groups, policies), dtype=np.int64),
        "drop_counts": np.zeros((groups, policies, sources), dtype=np.int64),
        "capture_sum": np.zeros(groups, dtype=np.int64),
        "drain_sum": np.zeros(groups, dtype=np.int64),
        "fragmentation_sum": np.zeros(groups, dtype=np.int64),
        "stage2_sum": np.zeros((groups, policies), dtype=np.int64),
        "reservoir_stall_sum": np.zeros((groups, policies), dtype=np.int64),
        "tail_sum": np.zeros((groups, policies), dtype=np.int64),
        "stream_penalty_sum": np.zeros((groups, policies), dtype=np.int64),
        "lane_idle_sum": np.zeros((groups, policies), dtype=np.int64),
        "capacity_violations": np.zeros(groups, dtype=np.int64),
        "max_bucket_occupancy": np.zeros(groups, dtype=np.int64),
        "max_reservoir_occupancy": np.zeros((groups, policies), dtype=np.int64),
        "active_hist": np.zeros(MAX_ACTIVE + 1, dtype=np.int64),
        "capture_hist": np.zeros(MAX_ACTIVE // 8 + 1, dtype=np.int64),
        "drain_hist": np.zeros(MAX_WORDS + 1, dtype=np.int64),
        "fragmentation_hist": np.zeros(15, dtype=np.int64),
        "cutoff_hist": np.zeros((policies, MAX_ACTIVE + 1), dtype=np.int64),
        "issue_hist": np.zeros((policies, MAX_ACTIVE + 1), dtype=np.int64),
        "stage2_hist": np.zeros((policies, MAX_STAGE2 + 1), dtype=np.int64),
        "tokens": 0,
        "records": 0,
        "active_by_source": np.zeros(sources, dtype=np.int64),
    }


def add_outputs(accumulator, outputs):
    for name in (
            "baseline", "active_sum", "issued", "dropped", "bound_sum",
            "exact_beta_sum", "raw_abs_sum", "raw_nonzero", "violations",
            "drop_counts", "capture_sum", "drain_sum", "fragmentation_sum",
            "stage2_sum", "reservoir_stall_sum", "tail_sum",
            "stream_penalty_sum", "lane_idle_sum", "capacity_violations"):
        accumulator[name] += outputs[name]
    for name in ("bound_max", "raw_abs_max", "max_bucket_occupancy",
                 "max_reservoir_occupancy"):
        accumulator[name] = np.maximum(accumulator[name], outputs[name])
    for name in ("active_hist", "capture_hist", "drain_hist",
                 "fragmentation_hist"):
        accumulator[name] += outputs[name].sum(axis=0, dtype=np.int64)
    for name in ("cutoff_hist", "issue_hist", "stage2_hist"):
        accumulator[name] += outputs[name].sum(axis=0, dtype=np.int64)


def witness_count(drop_counts, active_by_source):
    grid = np.broadcast_to(active_by_source[None, None, :], drop_counts.shape)
    return int(((drop_counts > 0) & (drop_counts < grid)).sum())


def prefetch_cycles(groups, sources):
    bytes_per_group = (sources + 1) // 2
    return int(groups * ((bytes_per_group + 31) // 32))


def policy_row(scope, kind, budget_index, budget, accumulator, state,
               metadata_cycles, static_pairs, task_count):
    baseline = int(accumulator["baseline"].sum(dtype=np.int64))
    if budget_index == 0:
        pipeline = baseline
        total = baseline
        capture = 0
        drain = 0
        fragmentation = 0
        stage2 = baseline
        reservoir_stalls = 0
        tail = 0
        stream_penalty = 0
        context_wait_cycles = 0
        stage2_wait_cycles = 0
        startup = 0
        final_drain = 0
        overlap_floor = baseline
        metadata = 0
    else:
        pipeline = int(state["stage2_end_prev"][:, budget_index].sum(
            dtype=np.int64))
        metadata = int(metadata_cycles)
        total = pipeline + metadata
        capture = int(accumulator["capture_sum"].sum(dtype=np.int64))
        drain = int(accumulator["drain_sum"].sum(dtype=np.int64))
        fragmentation = int(accumulator["fragmentation_sum"].sum(
            dtype=np.int64))
        stage2 = int(accumulator["stage2_sum"][:, budget_index].sum(
            dtype=np.int64))
        reservoir_stalls = int(
            accumulator["reservoir_stall_sum"][:, budget_index].sum(
                dtype=np.int64))
        tail = int(accumulator["tail_sum"][:, budget_index].sum(
            dtype=np.int64))
        stream_penalty = int(
            accumulator["stream_penalty_sum"][:, budget_index].sum(
                dtype=np.int64))
        context_wait_cycles = int(state["context_wait"][:, budget_index].sum(
            dtype=np.int64))
        stage2_wait_cycles = int(state["stage2_wait"][:, budget_index].sum(
            dtype=np.int64))
        startup = int(state["first_capture"][:, budget_index].sum(
            dtype=np.int64))
        final_drain = int(state["last_stage2"][:, budget_index].sum(
            dtype=np.int64))
        overlap_floor = int(np.maximum(
            accumulator["capture_sum"],
            accumulator["stage2_sum"][:, budget_index]).sum(dtype=np.int64))

    issued = int(accumulator["issued"][:, budget_index].sum(dtype=np.int64))
    dropped = int(accumulator["dropped"][:, budget_index].sum(dtype=np.int64))
    active_tasks = int(accumulator["active_sum"].sum(dtype=np.int64))
    violations = int(accumulator["violations"][:, budget_index].sum(
        dtype=np.int64))
    witnesses = witness_count(
        accumulator["drop_counts"][:, budget_index:budget_index + 1, :],
        accumulator["active_by_source"])
    static_mask_bytes = (static_pairs + 7) // 8
    beta_bytes = (static_pairs + 1) // 2
    row = {
        "scope": scope,
        "kind": kind,
        "budget": int(budget),
        "destination_group_size": 4,
        "tokens": int(accumulator["tokens"]),
        "tasks": int(task_count),
        "active_source_group_tasks": active_tasks,
        "dropped_source_group_tasks": dropped,
        "dropped_fraction": dropped / float(active_tasks),
        "baseline_k8_cycles": baseline,
        "kept_fixed_bank_lower_bound_cycles": issued,
        "capture_cycles": capture,
        "bucket_drain_cycles": drain,
        "bucket_fragmentation_cycles": fragmentation,
        "registered_reservoir_stage2_cycles": stage2,
        "reservoir_capacity_stall_cycles": reservoir_stalls,
        "reservoir_tail_cycles": tail,
        "stream_order_penalty_over_max_drain_issue": stream_penalty,
        "two_context_pipeline_cycles": pipeline,
        "two_context_overlap_floor_sum_per_group": overlap_floor,
        "two_context_excess_over_group_overlap_floor": pipeline - overlap_floor,
        "capture_context_wait_cycles": context_wait_cycles,
        "stage2_wait_for_capture_cycles": stage2_wait_cycles,
        "group_startup_first_capture_cycles": startup,
        "group_final_stage2_drain_cycles": final_drain,
        "metadata_prefetch_cycles": metadata,
        "total_cycles": total,
        "total_speedup": baseline / float(total),
        "maximum_conservative_bound": int(
            accumulator["bound_max"][:, budget_index].max()),
        "sum_conservative_bound": int(
            accumulator["bound_sum"][:, budget_index].sum(dtype=np.int64)),
        "sum_exact_beta_of_dropped": int(
            accumulator["exact_beta_sum"][:, budget_index].sum(
                dtype=np.int64)),
        "maximum_raw_signed_int8_error_absolute": int(
            accumulator["raw_abs_max"][:, budget_index].max()),
        "sum_raw_signed_int8_error_absolute": int(
            accumulator["raw_abs_sum"][:, budget_index].sum(dtype=np.int64)),
        "nonzero_raw_error_accumulators": int(
            accumulator["raw_nonzero"][:, budget_index].sum(dtype=np.int64)),
        "bound_violations": violations,
        "dynamic_witness_source_group_pairs": witnesses,
        "capacity_violations": int(accumulator["capacity_violations"].sum(
            dtype=np.int64)),
        "maximum_bucket_occupancy": int(
            accumulator["max_bucket_occupancy"].max()),
        "maximum_reservoir_occupancy": int(
            accumulator["max_reservoir_occupancy"][:, budget_index].max()),
        "lane_idle_slots": int(
            accumulator["lane_idle_sum"][:, budget_index].sum(dtype=np.int64)),
        "persistent_beta_metadata_bytes": int(beta_bytes),
        "one_bit_mask_reference_bytes": int(static_mask_bytes),
        "metadata_ratio": beta_bytes / float(static_mask_bytes),
        "accuracy": False if budget_index > 0 else True,
        "task_distributions": {
            "active_sources": histogram_summary(accumulator["active_hist"]),
            "capture_cycles": (histogram_summary(accumulator["capture_hist"])
                               if budget_index > 0 else
                               {"count": task_count, "mean": 0.0,
                                "p50": 0, "p90": 0, "p99": 0,
                                "maximum": 0}),
            "bucket_drain_cycles": (
                histogram_summary(accumulator["drain_hist"])
                if budget_index > 0 else
                {"count": task_count, "mean": 0.0, "p50": 0,
                 "p90": 0, "p99": 0, "maximum": 0}),
            "bucket_fragmentation_cycles": (
                histogram_summary(accumulator["fragmentation_hist"])
                if budget_index > 0 else
                {"count": task_count, "mean": 0.0, "p50": 0,
                 "p90": 0, "p99": 0, "maximum": 0}),
            "dropped_prefix_sources": histogram_summary(
                accumulator["cutoff_hist"][budget_index]),
            "fixed_bank_lower_bound_cycles": histogram_summary(
                accumulator["issue_hist"][budget_index]),
            "registered_reservoir_stage2_cycles": histogram_summary(
                accumulator["stage2_hist"][budget_index]),
        },
    }
    row["b0_exact"] = bool(
        budget_index != 0 or
        (dropped == 0 and issued == baseline and total == baseline and
         capture == 0 and drain == 0 and metadata == 0 and violations == 0))
    return row


def flat_row(row):
    return dict((key, value) for key, value in row.items()
                if key != "task_distributions")


def write_csv(path, rows):
    require(bool(rows), "cannot write empty CSV")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_policy_rows(scope, kind, modules, budget_index, budget,
                          combined_hist):
    answer = {}
    sum_fields = [
        "tokens", "tasks", "active_source_group_tasks",
        "dropped_source_group_tasks", "baseline_k8_cycles",
        "kept_fixed_bank_lower_bound_cycles", "capture_cycles",
        "bucket_drain_cycles", "bucket_fragmentation_cycles",
        "registered_reservoir_stage2_cycles",
        "reservoir_capacity_stall_cycles", "reservoir_tail_cycles",
        "stream_order_penalty_over_max_drain_issue",
        "two_context_pipeline_cycles",
        "two_context_overlap_floor_sum_per_group",
        "two_context_excess_over_group_overlap_floor",
        "capture_context_wait_cycles", "stage2_wait_for_capture_cycles",
        "group_startup_first_capture_cycles",
        "group_final_stage2_drain_cycles", "metadata_prefetch_cycles",
        "total_cycles", "sum_conservative_bound",
        "sum_exact_beta_of_dropped", "sum_raw_signed_int8_error_absolute",
        "nonzero_raw_error_accumulators", "bound_violations",
        "dynamic_witness_source_group_pairs", "capacity_violations",
        "lane_idle_slots", "persistent_beta_metadata_bytes",
        "one_bit_mask_reference_bytes"]
    for field in sum_fields:
        answer[field] = sum(int(row[field]) for row in modules)
    distributions = {
        name: histogram_summary(histogram[budget_index]
                                if histogram.ndim == 2 else histogram)
        for name, histogram in combined_hist.items()
    }
    if budget_index == 0:
        zero_selector = {"count": answer["tasks"], "mean": 0.0,
                         "p50": 0, "p90": 0, "p99": 0, "maximum": 0}
        distributions["capture_cycles"] = dict(zero_selector)
        distributions["bucket_drain_cycles"] = dict(zero_selector)
        distributions["bucket_fragmentation_cycles"] = dict(zero_selector)
    answer.update({
        "scope": scope,
        "kind": kind,
        "budget": int(budget),
        "destination_group_size": 4,
        "dropped_fraction": (answer["dropped_source_group_tasks"] /
                             float(answer["active_source_group_tasks"])),
        "total_speedup": (answer["baseline_k8_cycles"] /
                          float(answer["total_cycles"])),
        "maximum_conservative_bound": max(
            int(row["maximum_conservative_bound"]) for row in modules),
        "maximum_raw_signed_int8_error_absolute": max(
            int(row["maximum_raw_signed_int8_error_absolute"])
            for row in modules),
        "maximum_bucket_occupancy": max(
            int(row["maximum_bucket_occupancy"]) for row in modules),
        "maximum_reservoir_occupancy": max(
            int(row["maximum_reservoir_occupancy"]) for row in modules),
        "metadata_ratio": (
            answer["persistent_beta_metadata_bytes"] /
            float(answer["one_bit_mask_reference_bytes"])),
        "accuracy": False if budget_index > 0 else True,
        "b0_exact": all(bool(row["b0_exact"]) for row in modules),
        "task_distributions": distributions,
    })
    return answer


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
            "m341_conservative_code_task_schedule_cpu_dse_contract_v1",
            "M341 contract schema drift")
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
        require(path.is_file(), "missing M341 input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"],
                "M341 input SHA drift for {}: {}".format(label, digest))
        paths[label] = path
        identity[label] = {"path": spec["path"], "sha256": digest}

    base = load_pinned_m328(paths["m328_analyzer"])
    threads = args.threads if args.threads > 0 else min(
        96, __import__("os").cpu_count())
    set_num_threads(threads)
    manifest = strict_json(paths["m51_manifest"])
    require(manifest["packing"]["layout"] == "C_ORDER_FLAT" and
            manifest["packing"]["bit_order"] == "LITTLE_WITHIN_BYTE",
            "M341 M51 packing drift")
    fc1_spec = contract["population"]["fc1"]
    conv_spec = contract["population"]["conv"]
    fc1_records = [row for row in manifest["records"]
                   if fc1_spec["name_substring"] in row["name"]]
    conv_records = [row for row in manifest["records"]
                    if row["name"] == conv_spec["name"]]
    require(len(fc1_records) == int(fc1_spec["records"]) and
            len(set(row["name"] for row in fc1_records)) ==
            int(fc1_spec["modules"]), "M341 FC1 population drift")
    require(len(conv_records) == int(conv_spec["records"]),
            "M341 Conv population drift")
    for record in fc1_records + conv_records:
        payload = paths["m51_manifest"].parent / record["relative_path"]
        require(payload.is_file() and sha256(payload) == record["file_sha256"],
                "M341 payload SHA drift: " + str(payload))

    records_by_module = defaultdict(list)
    for record in fc1_records:
        records_by_module[record["name"]].append(record)
    for name in records_by_module:
        records_by_module[name].sort(key=lambda row: int(row["sample_id"]))
    conv_records.sort(key=lambda row: int(row["sample_id"]))
    module_names = sorted(records_by_module)
    module_names.append(conv_spec["name"])
    weights = base.load_checkpoint_weights(paths["checkpoint"], module_names)
    quantized = dict((name, base.quantize_weight(weights[name]))
                     for name in module_names)

    endpoints = np.asarray(contract["mechanism"]["upper_bound_codebook"],
                           dtype=np.int16)
    budgets = np.asarray(contract["mechanism"]["budgets"], dtype=np.int32)
    require(endpoints.tolist() ==
            [0, 9, 17, 26, 34, 43, 51, 60, 68, 77, 85, 94, 102, 111, 119, 127]
            and budgets.tolist() == [0, 16, 32, 64, 128, 256, 512, 1024],
            "M341 frozen codebook/grid drift")

    m328 = strict_json(paths["m328_result"])
    m328_rows = dict((entry["module"], entry["policies"]["4"][0])
                     for entry in m328["per_module"])
    module_policy_rows = []
    record_policy_rows = []
    module_details = []
    module_histograms = {}
    fc1_tokens = 0
    fc1_active = 0
    conv_active = 0
    conv_maximum = 0

    for module_index, name in enumerate(module_names):
        is_conv = name == conv_spec["name"]
        kind = "conv3x3" if is_conv else "fc1"
        records = conv_records if is_conv else records_by_module[name]
        qweight = quantized[name]
        sources = int(qweight.shape[1])
        exact_beta, codes, upper, order = prepare_code_tables(
            qweight, 4, endpoints)
        groups = int(upper.shape[0])
        state = empty_schedule_state(groups, budgets.size)
        accumulator = empty_module_accumulator(
            groups, budgets.size, sources)
        metadata_cycles = prefetch_cycles(groups, sources)
        require(metadata_cycles == groups * int(math.ceil(
            ((sources + 1) // 2) / 32.0)), "prefetch formula drift")

        for record_index, record in enumerate(records):
            payload = paths["m51_manifest"].parent / record["relative_path"]
            if is_conv:
                bits, stride = base.unpack_conv_sources(record, payload)
                del stride
                conv_active += int(bits.sum(dtype=np.uint64))
                conv_maximum = max(
                    conv_maximum,
                    int(bits.sum(axis=1, dtype=np.int16).max()))
            else:
                bits = base.unpack_fc1(record, payload)
                fc1_tokens += int(bits.shape[0])
                fc1_active += int(bits.sum(dtype=np.uint64))
            banks = base.base_bank_counts(bits)
            active_by_source_record = bits.sum(axis=0, dtype=np.int64)
            before_timeline = state["stage2_end_prev"].copy()
            outputs = output_dict(analyze_record(
                bits, banks, qweight, exact_beta, codes, upper, order,
                budgets, state["capture_end"], state["stage2_end_prev"],
                state["stage2_end_prev2"], state["context_wait"],
                state["stage2_wait"], state["first_capture"],
                state["last_stage2"]))
            after_timeline = state["stage2_end_prev"].copy()
            add_outputs(accumulator, outputs)
            accumulator["tokens"] += int(bits.shape[0])
            accumulator["records"] += 1
            accumulator["active_by_source"] += active_by_source_record

            task_count_record = int(bits.shape[0]) * groups
            for policy, budget in enumerate(budgets.tolist()):
                baseline_record = int(outputs["baseline"].sum(dtype=np.int64))
                issued_record = int(outputs["issued"][:, policy].sum(
                    dtype=np.int64))
                pipeline_record = (baseline_record if policy == 0 else
                    int((after_timeline[:, policy] -
                         before_timeline[:, policy]).sum(dtype=np.int64)))
                metadata_record = (metadata_cycles if
                                   policy > 0 and record_index == 0 else 0)
                total_record = pipeline_record + metadata_record
                dropped_record = int(outputs["dropped"][:, policy].sum(
                    dtype=np.int64))
                active_record_tasks = int(outputs["active_sum"].sum(
                    dtype=np.int64))
                witness_record = witness_count(
                    outputs["drop_counts"][:, policy:policy + 1, :],
                    active_by_source_record)
                record_policy_rows.append({
                    "module": name,
                    "kind": kind,
                    "sample_id": int(record["sample_id"]),
                    "budget": int(budget),
                    "tokens": int(bits.shape[0]),
                    "groups": groups,
                    "tasks": task_count_record,
                    "active_source_group_tasks": active_record_tasks,
                    "dropped_source_group_tasks": dropped_record,
                    "dropped_fraction": (dropped_record /
                                         float(active_record_tasks)),
                    "baseline_k8_cycles": baseline_record,
                    "kept_fixed_bank_lower_bound_cycles": issued_record,
                    "capture_cycles": (0 if policy == 0 else
                                       int(outputs["capture_sum"].sum(
                                           dtype=np.int64))),
                    "bucket_drain_cycles": (0 if policy == 0 else
                        int(outputs["drain_sum"].sum(dtype=np.int64))),
                    "bucket_fragmentation_cycles": (0 if policy == 0 else
                        int(outputs["fragmentation_sum"].sum(
                            dtype=np.int64))),
                    "registered_reservoir_stage2_cycles": (
                        baseline_record if policy == 0 else
                        int(outputs["stage2_sum"][:, policy].sum(
                            dtype=np.int64))),
                    "reservoir_capacity_stall_cycles": (0 if policy == 0 else
                        int(outputs["reservoir_stall_sum"][:, policy].sum(
                            dtype=np.int64))),
                    "reservoir_tail_cycles": (0 if policy == 0 else
                        int(outputs["tail_sum"][:, policy].sum(
                            dtype=np.int64))),
                    "stream_order_penalty_over_max_drain_issue": (
                        0 if policy == 0 else
                        int(outputs["stream_penalty_sum"][:, policy].sum(
                            dtype=np.int64))),
                    "two_context_attributed_timeline_cycles": pipeline_record,
                    "metadata_prefetch_cycles_attributed": metadata_record,
                    "total_attributed_cycles": total_record,
                    "attributed_speedup": baseline_record / float(total_record),
                    "maximum_conservative_bound": int(
                        outputs["bound_max"][:, policy].max()),
                    "maximum_raw_signed_int8_error_absolute": int(
                        outputs["raw_abs_max"][:, policy].max()),
                    "bound_violations": int(
                        outputs["violations"][:, policy].sum(dtype=np.int64)),
                    "dynamic_witness_source_group_pairs": witness_record,
                    "capacity_violations": int(
                        outputs["capacity_violations"].sum(dtype=np.int64)),
                    "accuracy": bool(policy == 0),
                })
            del outputs, bits, banks
            print("M341 {}/{} {} record {}/{} sample {} complete".format(
                module_index + 1, len(module_names), name,
                record_index + 1, len(records), record["sample_id"]),
                flush=True)

        require(int(accumulator["baseline"].sum(dtype=np.int64)) ==
                int(m328_rows[name]["ideal_k8_baseline_issued_cycles"]),
                "M341 B0 module baseline mismatch: " + name)
        static_pairs = sources * groups
        task_count = accumulator["tokens"] * groups
        rows = []
        for policy, budget in enumerate(budgets.tolist()):
            row = policy_row(name, kind, policy, budget, accumulator, state,
                             metadata_cycles, static_pairs, task_count)
            rows.append(row)
            module_policy_rows.append(flat_row(row))
        module_details.append({
            "module": name,
            "kind": kind,
            "sources": sources,
            "outputs": int(qweight.shape[0]),
            "groups": groups,
            "records": len(records),
            "tokens": int(accumulator["tokens"]),
            "tasks": int(task_count),
            "metadata_prefetch_cycles": metadata_cycles,
            "metadata_bytes": (static_pairs + 1) // 2,
            "policies": rows,
        })
        module_histograms[name] = {
            "kind": kind,
            "active_sources": accumulator["active_hist"].copy(),
            "capture_cycles": accumulator["capture_hist"].copy(),
            "bucket_drain_cycles": accumulator["drain_hist"].copy(),
            "bucket_fragmentation_cycles": accumulator[
                "fragmentation_hist"].copy(),
            "dropped_prefix_sources": accumulator["cutoff_hist"].copy(),
            "fixed_bank_lower_bound_cycles": accumulator["issue_hist"].copy(),
            "registered_reservoir_stage2_cycles": accumulator[
                "stage2_hist"].copy(),
        }

    require(fc1_tokens == int(fc1_spec["tokens"]) and
            fc1_active == int(fc1_spec["active_sources"]),
            "M341 FC1 population mismatch")
    require(conv_active == int(conv_spec["source_contributions"]) and
            conv_maximum == int(conv_spec["maximum_active_sources"]),
            "M341 Conv population mismatch")

    details_by_name = dict((entry["module"], entry) for entry in module_details)
    aggregate_rows = []
    aggregate_details = {"fc1": [], "conv": [], "combined": []}
    scope_members = {
        "fc1": [name for name in module_names if name != conv_spec["name"]],
        "conv": [conv_spec["name"]],
        "combined": module_names,
    }
    scope_labels = {
        "fc1": ("__FC1_10_MODULE_AGGREGATE__", "fc1_aggregate"),
        "conv": ("__SELECTED_CONV_AGGREGATE__", "conv_aggregate"),
        "combined": ("__COMBINED_FC1_PLUS_SELECTED_CONV__", "combined"),
    }
    for scope_key, members in scope_members.items():
        combined_hist = {}
        for hist_name in (
                "active_sources", "capture_cycles", "bucket_drain_cycles",
                "bucket_fragmentation_cycles", "dropped_prefix_sources",
                "fixed_bank_lower_bound_cycles",
                "registered_reservoir_stage2_cycles"):
            combined_hist[hist_name] = sum(
                (module_histograms[name][hist_name] for name in members),
                np.zeros_like(module_histograms[members[0]][hist_name]))
        for policy, budget in enumerate(budgets.tolist()):
            members_rows = [details_by_name[name]["policies"][policy]
                            for name in members]
            row = aggregate_policy_rows(
                scope_labels[scope_key][0], scope_labels[scope_key][1],
                members_rows, policy, budget, combined_hist)
            aggregate_details[scope_key].append(row)
            aggregate_rows.append(flat_row(row))

    combined = aggregate_details["combined"]
    require(combined[0]["b0_exact"] and
            int(combined[0]["baseline_k8_cycles"]) ==
            int(contract["population"]["m328_b0_baseline_cycles"]) and
            int(combined[0]["total_cycles"]) ==
            int(contract["population"]["m328_b0_baseline_cycles"]),
            "M341 combined B0 mismatch")
    require(int(combined[0]["tasks"]) ==
            int(contract["population"]["combined_group4_task_count"]),
            "M341 task count mismatch")
    require(sum(row["metadata_prefetch_cycles"] for row in
                [details_by_name[name]["policies"][1]
                 for name in module_names]) ==
            int(contract["hardware_model"]["metadata"]
                ["full_group_major_sweep_cycles"]),
            "M341 metadata sweep mismatch")
    require(all(row["bound_violations"] == 0 for row in combined),
            "M341 bound violation")
    require(all(row["capacity_violations"] == 0 for row in combined),
            "M341 capacity violation")
    require(all(abs(float(row["metadata_ratio"]) - 4.0) < 1e-12
                for row in combined), "M341 metadata ratio drift")

    gate = float(contract["promotion_gates"]
                 ["minimum_combined_total_speedup"])
    candidates = [row for row in combined[1:]
                  if row["total_speedup"] >= gate and
                  row["dynamic_witness_source_group_pairs"] > 0 and
                  row["bound_violations"] == 0 and
                  row["capacity_violations"] == 0]
    candidates.sort(key=lambda row: (-row["total_speedup"], row["budget"]))
    promoted = candidates[:1]
    decision = (contract["decision_policy"]
                ["one_or_more_nonzero_points_pass"] if promoted else
                contract["decision_policy"]["no_nonzero_point_passes"])

    combined_hist_json = {}
    for name, histogram in {
            key: sum((module_histograms[module][key] for module in module_names),
                     np.zeros_like(module_histograms[module_names[0]][key]))
            for key in module_histograms[module_names[0]] if key != "kind"
            }.items():
        if histogram.ndim == 1:
            combined_hist_json[name] = {
                "counts": [int(value) for value in histogram.tolist()],
                "summary": histogram_summary(histogram),
            }
        else:
            combined_hist_json[name] = dict((str(int(budgets[index])), {
                "counts": [int(value) for value in histogram[index].tolist()],
                "summary": histogram_summary(histogram[index]),
            }) for index in range(budgets.size))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / (
        "m341_conservative_code_task_schedule_cpu_dse_r1.json")
    record_csv = args.output_dir / "per_record_policy_metrics.csv"
    module_csv = args.output_dir / "per_module_policy_metrics.csv"
    aggregate_csv = args.output_dir / "aggregate_policy_metrics.csv"
    histogram_path = args.output_dir / "task_histograms.json"
    write_csv(record_csv, record_policy_rows)
    write_csv(module_csv, module_policy_rows)
    write_csv(aggregate_csv, aggregate_rows)
    with histogram_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "schema": "m341_task_histograms_v1",
            "scope": "combined FC1 plus selected Conv group4",
            "budgets": [int(value) for value in budgets.tolist()],
            "histograms": combined_hist_json,
        }, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    require(sha256(source_path) == source_start,
            "M341 analyzer changed during execution")
    result = {
        "schema": "m341_conservative_code_task_schedule_cpu_dse_v1",
        "status": "PASS_CPU_EXECUTION_FAIL_CLOSED_DECISION",
        "decision": decision,
        "promoted_gpu_candidate": (flat_row(promoted[0]) if promoted else None),
        "identity": identity,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "numba": __import__("numba").__version__,
            "threads": int(threads),
            "elapsed_seconds": time.time() - started,
            "torch_dependency": False,
            "checkpoint_loader": "pinned M328 constrained torchless loader",
        },
        "mechanism": contract["mechanism"],
        "hardware_model": contract["hardware_model"],
        "promotion_gates": contract["promotion_gates"],
        "population": {
            "fc1_modules": len(records_by_module),
            "fc1_records": len(fc1_records),
            "fc1_tokens": fc1_tokens,
            "fc1_active_sources": fc1_active,
            "conv_records": len(conv_records),
            "conv_source_contributions": conv_active,
            "conv_maximum_active_sources": conv_maximum,
            "combined_group4_tasks": int(combined[0]["tasks"]),
        },
        "correctness": {
            "b0_exact_m328_work_and_cycles": True,
            "m328_b0_baseline_cycles": int(combined[0]["baseline_k8_cycles"]),
            "m341_b0_total_cycles": int(combined[0]["total_cycles"]),
            "conservative_code_violations": 0,
            "integer_bound_violations": 0,
            "capacity_violations": 0,
            "all_nonzero_budget_accuracy": False,
        },
        "best_nonzero_combined": flat_row(max(
            combined[1:], key=lambda row: row["total_speedup"])),
        "aggregates": aggregate_details,
        "per_module": module_details,
        "output_files": {
            "per_record_policy_metrics": record_csv.name,
            "per_module_policy_metrics": module_csv.name,
            "aggregate_policy_metrics": aggregate_csv.name,
            "task_histograms": histogram_path.name,
        },
        "admission": {
            "cpu_frozen_trace_schedule": True,
            "gpu_modified_forward_candidate": bool(promoted),
            "accuracy": False,
            "modified_forward": False,
            "valid825": False,
            "rtl": False,
            "vcs": False,
            "synopsys": False,
            "energy": False,
            "system_speedup": False,
            "headline": False,
            "paper_contribution": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("M341_RESULT {} decision={} best_speedup={:.9f} elapsed={:.3f}s".format(
        result_path, decision,
        max(row["total_speedup"] for row in combined[1:]),
        time.time() - started), flush=True)


if __name__ == "__main__":
    main()
