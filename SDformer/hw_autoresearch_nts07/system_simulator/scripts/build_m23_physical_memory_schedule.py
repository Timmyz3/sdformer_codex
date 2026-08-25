#!/usr/bin/env python3
"""Map frozen M22 transactions to trace-live buffers and compressed bank service.

M23 is a deterministic physical-memory *envelope*.  It implements SRAM
lifetime reuse, fixed DRAM bursts, word-interleaved SRAM banks, explicit
trace/object dependencies and bounded port-service estimates.  It is not a
DRAMsim3 trace, a compute schedule, or a system-cycle/speedup result.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


M22_FILES = {
    "REPORT.md", "m22_ordered_transactions.csv", "m22_output_manifest.json",
    "m22_summary.json",
}
VARIANTS = {
    "local_line", "motion_selector_shared_state", "motion_selector_explicit_copy",
}
REQUIRED_TRANSACTION_FIELDS = {
    "transaction_id", "identity", "variant", "request_issue_order",
    "previous_in_trace_order", "sample_id", "call_index", "event_kind", "name",
    "temporal_step", "tier", "direction", "phase", "address", "object_id",
    "object_span_bytes", "byte_count", "serialized_service_start",
    "serialized_service_end_exclusive", "service_bytes_per_cycle",
    "address_pattern", "evidence_class",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ceil_div(value, divisor):
    if value < 0 or divisor <= 0:
        raise ValueError("invalid ceil_div operands")
    return (value + divisor - 1) // divisor


def integer(value, field):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError("invalid integer {}={!r}".format(field, value))
    if str(result) != str(value).strip() and not (
        isinstance(value, int) and result == value
    ):
        raise ValueError("non-canonical integer {}={!r}".format(field, value))
    return result


def load_contract(path, expected_sha256, repo_root):
    path = Path(path)
    actual = sha256(path)
    if len(expected_sha256) != 64 or actual != expected_sha256.lower():
        raise ValueError("M23 input contract SHA mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "m23_m22_input_contract_v1"
        or contract.get("status") != "FROZEN_M22_BINARYCLOSED2_INPUT"
        or set(contract.get("files_sha256", {})) != M22_FILES
        or set(contract.get("identities", {})) != {"h67_ep35", "local_ep44"}
    ):
        raise ValueError("M23 input contract schema/status/coverage is not admitted")
    relative = Path(contract.get("artifact_directory", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("M22 artifact directory must be repo-relative")
    artifact = Path(repo_root) / relative
    for name, expected in contract["files_sha256"].items():
        candidate = artifact / name
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError("M22 artifact identity mismatch: {}".format(name))
    receipt = contract.get("m22_receipt", {})
    receipt_path = Path(receipt.get("path", ""))
    if receipt_path.is_absolute() or ".." in receipt_path.parts:
        raise ValueError("M22 receipt path must be repo-relative")
    receipt_path = Path(repo_root) / receipt_path
    if not receipt_path.is_file() or sha256(receipt_path) != receipt.get("sha256"):
        raise ValueError("M22 receipt identity mismatch")
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt_payload.get("schema") != "m22_frozen_output_receipt_v2"
        or receipt_payload.get("transaction_records") != contract.get("transaction_records")
        or receipt_payload.get("files_sha256") != contract.get("files_sha256")
    ):
        raise ValueError("M22 receipt does not reconcile to M23 contract")
    return contract, actual, artifact


def load_m22(contract, artifact):
    output_manifest = json.loads(
        (artifact / "m22_output_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads((artifact / "m22_summary.json").read_text(encoding="utf-8"))
    if (
        output_manifest.get("schema") != "m22_ordered_transaction_output_manifest_v2"
        or output_manifest.get("status") != "FROZEN_REPRODUCIBLE_PARTIAL_LEDGER"
        or output_manifest.get("transaction_records") != contract["transaction_records"]
        or output_manifest.get("artifacts", {}).get("m22_ordered_transactions.csv", {}).get("sha256")
        != contract["transactions_sha256"]
        or summary.get("schema") != "m22_ordered_compressed_system_transactions_v2"
        or summary.get("status")
        != "PASS_FROZEN_INPUT_PARTIAL_TRANSACTION_LEDGER_NOT_DRAMSIM_OR_SPEEDUP"
        or summary.get("transaction_records") != contract["transaction_records"]
        or summary.get("transactions_sha256") != contract["transactions_sha256"]
    ):
        raise ValueError("M22 output manifest/summary identity mismatch")
    for identity, expected in contract["identities"].items():
        actual = summary.get("identities", {}).get(identity, {})
        if (
            actual.get("attention_coverage_status") != expected["attention_coverage_status"]
            or set(actual.get("variants", {})) != set(expected["variants"])
        ):
            raise ValueError("M22 identity/variant/attention contract mismatch")

    rows = []
    path = artifact / "m22_ordered_transactions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_TRANSACTION_FIELDS.issubset(reader.fieldnames):
            raise ValueError("M22 transaction CSV schema is incomplete")
        for record in reader:
            row = dict(record)
            for field in (
                "transaction_id", "request_issue_order", "previous_in_trace_order",
                "sample_id", "call_index", "temporal_step", "object_span_bytes",
                "byte_count", "serialized_service_start",
                "serialized_service_end_exclusive", "service_bytes_per_cycle",
            ):
                row[field] = integer(row[field], field)
            try:
                row["address_integer"] = int(row["address"], 16)
            except ValueError:
                raise ValueError("invalid M22 hexadecimal address")
            if (
                row["tier"] not in ("SRAM", "DRAM")
                or row["direction"] not in ("READ", "WRITE")
                or row["variant"] not in VARIANTS
                or row["object_span_bytes"] <= 0
                or row["byte_count"] <= 0
                or row["service_bytes_per_cycle"] <= 0
                or row["serialized_service_end_exclusive"]
                <= row["serialized_service_start"]
            ):
                raise ValueError("invalid M22 transaction extent/resource")
            rows.append(row)
    if len(rows) != contract["transaction_records"]:
        raise ValueError("M22 transaction cardinality mismatch")
    if [row["transaction_id"] for row in rows] != list(range(len(rows))):
        raise ValueError("M22 global transaction ids are not dense")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["identity"], row["variant"])].append(row)
    expected_groups = {
        (identity, variant) for identity in contract["identities"]
        for variant in contract["identities"][identity]["variants"]
    }
    if set(groups) != expected_groups:
        raise ValueError("M22 CSV identity/variant coverage mismatch")
    for key, records in groups.items():
        if [row["request_issue_order"] for row in records] != list(range(len(records))):
            raise ValueError("M22 request order is not dense: {}".format(key))
        expected_previous = [-1] + list(range(len(records) - 1))
        if [row["previous_in_trace_order"] for row in records] != expected_previous:
            raise ValueError("M22 trace predecessor chain is broken: {}".format(key))
        byte_totals = Counter()
        transport_ticks = 0
        for row in records:
            byte_totals["{}_{}_bytes".format(
                row["tier"].lower(), row["direction"].lower()
            )] += row["byte_count"]
            transport_ticks += (
                row["serialized_service_end_exclusive"]
                - row["serialized_service_start"]
            )
        expected = summary["identities"][key[0]]["variants"][key[1]]["totals"]
        for field, value in byte_totals.items():
            if expected.get(field, 0) != value:
                raise ValueError("M22 byte conservation failed: {} {}".format(key, field))
        if expected.get("serialized_byte_service_ticks") != transport_ticks:
            raise ValueError("M22 serialized transport tick conservation failed")
    return summary, rows


def object_category(object_id):
    if "attention_" in object_id:
        return "attention_abstract"
    if "motion_" in object_id:
        return "motion_state"
    if "atlif_state" in object_id:
        return "atlif_state"
    if "atlif_param" in object_id:
        return "atlif_parameter"
    if "atlif_acc_input" in object_id:
        return "atlif_input"
    if "atlif_bitmap_output" in object_id:
        return "atlif_output"
    if ":weight:" in object_id:
        return "weight"
    if "activation_input" in object_id:
        return "activation_input"
    if "operator_acc_output" in object_id:
        return "operator_output"
    return "other"


def collect_sram_lifetimes(records):
    objects = {}
    for row in records:
        if row["tier"] != "SRAM":
            continue
        # Repeated model weights and ATLIF parameters are re-filled per call.
        # The sample/call suffix therefore defines a residency generation even
        # when M22 intentionally retains one logical object id.
        key = (row["object_id"], row["sample_id"], row["call_index"])
        current = objects.get(key)
        if current is None:
            current = {
                "instance_key": key,
                "instance_id": "{}|s{}|c{}".format(*key),
                "object_id": row["object_id"],
                "sample_id": row["sample_id"],
                "call_index": row["call_index"],
                "event_kind": row["event_kind"],
                "category": object_category(row["object_id"]),
                "object_span_bytes": row["object_span_bytes"],
                "first_issue_order": row["request_issue_order"],
                "last_issue_order": row["request_issue_order"],
                "logical_base_address": row["address_integer"],
                "transaction_count": 0,
            }
            objects[key] = current
        if current["object_span_bytes"] != row["object_span_bytes"]:
            raise ValueError("M22 SRAM object span drift")
        current["first_issue_order"] = min(
            current["first_issue_order"], row["request_issue_order"]
        )
        current["last_issue_order"] = max(
            current["last_issue_order"], row["request_issue_order"]
        )
        current["logical_base_address"] = min(
            current["logical_base_address"], row["address_integer"]
        )
        current["transaction_count"] += 1
    for row in records:
        if row["tier"] != "SRAM":
            continue
        key = (row["object_id"], row["sample_id"], row["call_index"])
        obj = objects[key]
        offset = row["address_integer"] - obj["logical_base_address"]
        if offset < 0 or offset >= obj["object_span_bytes"]:
            raise ValueError("M22 SRAM transaction is outside logical object")
    return list(objects.values())


def _add_free_block(free_blocks, base, size, predecessor):
    if size > 0:
        free_blocks.append({"base": base, "size": size, "predecessor": predecessor})
        free_blocks.sort(key=lambda block: block["base"])
        merged = []
        for block in free_blocks:
            if merged and merged[-1]["base"] + merged[-1]["size"] == block["base"]:
                merged[-1]["size"] += block["size"]
                if merged[-1]["predecessor"] != block["predecessor"]:
                    merged[-1]["predecessor"] = "MULTIPLE_RELEASED_REGIONS"
            else:
                merged.append(dict(block))
        free_blocks[:] = merged


def allocate_lifetimes(objects, alignment, physical_base=0x80000040):
    if alignment <= 0:
        raise ValueError("allocator alignment must be positive")
    active = []
    free_blocks = []
    high_water = 0
    allocations = []
    for obj in sorted(objects, key=lambda item: (
        item["first_issue_order"], item["last_issue_order"], item["instance_id"]
    )):
        retained = []
        for prior in active:
            if prior["last_issue_order"] < obj["first_issue_order"]:
                _add_free_block(
                    free_blocks, prior["physical_offset"], prior["allocated_bytes"],
                    prior["instance_id"],
                )
            else:
                retained.append(prior)
        active = retained
        size = ceil_div(obj["object_span_bytes"], alignment) * alignment
        choices = [
            (block["size"], block["base"], index) for index, block in enumerate(free_blocks)
            if block["size"] >= size
        ]
        predecessor = ""
        reused = False
        if choices:
            _block_size, _block_base, index = min(choices)
            block = free_blocks.pop(index)
            offset = block["base"]
            predecessor = block["predecessor"]
            reused = True
            _add_free_block(
                free_blocks, offset + size, block["size"] - size, block["predecessor"]
            )
        else:
            offset = ceil_div(high_water, alignment) * alignment
            high_water = offset + size
        allocation = dict(obj)
        allocation.update({
            "physical_offset": offset,
            "physical_base_address": physical_base + offset,
            "allocated_bytes": size,
            "alignment_bytes": alignment,
            "reused_region": reused,
            "reuse_predecessor_instance": predecessor,
        })
        allocations.append(allocation)
        active.append(allocation)
    validate_allocator_nonoverlap(allocations)

    events = []
    for allocation in allocations:
        events.append((allocation["first_issue_order"], 1, allocation["allocated_bytes"]))
        events.append((allocation["last_issue_order"] + 1, -1, allocation["allocated_bytes"]))
    current = 0
    peak = 0
    for _position, delta, size in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta * size
        peak = max(peak, current)
    largest_stream_buffer = max(
        [row["allocated_bytes"] for row in allocations if row["category"] in (
            "activation_input", "operator_output", "atlif_input", "atlif_output"
        )] or [0]
    )
    two_buffer_bytes = 2 * largest_stream_buffer
    return allocations, {
        "allocator_capacity_bytes": high_water,
        "peak_live_aligned_bytes": peak,
        "high_water_slack_over_peak_bytes": high_water - peak,
        "allocator_policy": "COALESCED_BEST_FIT_BY_SIZE_THEN_BASE",
        "allocation_instances": len(allocations),
        "reused_allocation_instances": sum(row["reused_region"] for row in allocations),
        "max_single_object_bytes": max(
            [row["object_span_bytes"] for row in allocations] or [0]
        ),
        "two_copy_largest_stream_capacity_bound_bytes": two_buffer_bytes,
        "allocator_plus_extra_largest_stream_capacity_bound_bytes": (
            high_water + largest_stream_buffer
        ),
    }


def validate_allocator_nonoverlap(allocations):
    ordered = sorted(allocations, key=lambda row: row["first_issue_order"])
    for index, left in enumerate(ordered):
        left_end = left["physical_offset"] + left["allocated_bytes"]
        for right in ordered[index + 1:]:
            if right["first_issue_order"] > left["last_issue_order"]:
                break
            right_end = right["physical_offset"] + right["allocated_bytes"]
            time_overlap = not (
                left["last_issue_order"] < right["first_issue_order"]
                or right["last_issue_order"] < left["first_issue_order"]
            )
            address_overlap = not (
                left_end <= right["physical_offset"] or right_end <= left["physical_offset"]
            )
            if time_overlap and address_overlap:
                raise ValueError("allocator overlap: {} / {}".format(
                    left["instance_id"], right["instance_id"]
                ))
    return True


def consecutive_bank_counts(start_bank, count, banks):
    if count < 0 or banks <= 0 or not 0 <= start_bank < banks:
        raise ValueError("invalid bank-count arguments")
    quotient, remainder = divmod(count, banks)
    result = [quotient] * banks
    for index in range(remainder):
        result[(start_bank + index) % banks] += 1
    return result


def cyclic_bank_counts(start_bank, request_words, object_words, banks):
    if object_words <= 0:
        raise ValueError("cyclic object has no words")
    cycles, remainder = divmod(request_words, object_words)
    per_object = consecutive_bank_counts(start_bank, object_words, banks)
    tail = consecutive_bank_counts(start_bank, remainder, banks)
    return [cycles * per_object[index] + tail[index] for index in range(banks)]


def cyclic_visited_rows(physical_base, word_bytes, banks, start_object_word,
                        request_words, object_words):
    """Return the exact distinct-row envelope for a cyclic word stream."""
    if physical_base % (word_bytes * banks):
        raise ValueError("cyclic object base is not a complete bank-stripe boundary")
    if not 0 <= start_object_word < object_words:
        raise ValueError("cyclic start word is outside object")
    distinct_words = min(request_words, object_words)
    if distinct_words <= 0:
        raise ValueError("cyclic request has no visited words")
    stop = start_object_word + distinct_words
    intervals = []
    if stop <= object_words:
        intervals.append((start_object_word, stop - 1))
    else:
        intervals.append((start_object_word, object_words - 1))
        intervals.append((0, stop - object_words - 1))
    base_word = physical_base // word_bytes
    row_intervals = sorted(
        ((base_word + first) // banks, (base_word + last) // banks)
        for first, last in intervals
    )
    merged = []
    for first, last in row_intervals:
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return {
        "row_index_start": min(first for first, _last in merged),
        "row_index_end": max(last for _first, last in merged),
        "visited_row_count": sum(last - first + 1 for first, last in merged),
        "row_interval_rle": ";".join(
            str(first) if first == last else "{}-{}".format(first, last)
            for first, last in merged
        ),
    }


def encode_bank_counts(counts):
    if not counts:
        return ""
    runs = []
    start = 0
    for index in range(1, len(counts) + 1):
        if index == len(counts) or counts[index] != counts[start]:
            bank_range = str(start) if index - 1 == start else "{}-{}".format(start, index - 1)
            runs.append("{}={}".format(bank_range, counts[start]))
            start = index
    return ";".join(runs)


def schedule_transaction(row, allocation, config, trace_dependency, object_dependency):
    tier = row["tier"]
    direction = row["direction"].lower()
    if tier == "DRAM":
        quantum = config["dram_burst_bytes"]
        banks = config["dram_banks"]
        ports = config["dram_{}_ports_per_bank".format(direction)]
        address = row["address_integer"]
        offset = address % quantum
        request_count = ceil_div(offset + row["byte_count"], quantum)
        first_quantum = address // quantum
        counts = consecutive_bank_counts(first_quantum % banks, request_count, banks)
        lower_counts = counts
        upper_counts = counts
        mapping = "MODEL_EXACT_CONTIGUOUS_FIXED_BURST_INTERLEAVE_ASSUMPTION"
        row_bytes = config["dram_row_bytes"]
        first_transfer_address = address - offset
        last_transfer_address = first_transfer_address + request_count * quantum - 1
        row_start = first_transfer_address // row_bytes
        row_end = last_transfer_address // row_bytes
        object_row_start = row_start
        object_row_end = row_end
        visited_row_lower = row_end - row_start + 1
        visited_row_upper = visited_row_lower
        visited_row_rle = (
            str(row_start) if row_start == row_end
            else "{}-{}".format(row_start, row_end)
        )
        physical_instance = row["object_id"]
    else:
        quantum = config["sram_word_bytes"]
        banks = config["sram_banks"]
        ports = config["sram_{}_ports_per_bank".format(direction)]
        logical_offset = row["address_integer"] - allocation["logical_base_address"]
        if logical_offset < 0 or logical_offset >= allocation["object_span_bytes"]:
            raise ValueError("physical SRAM offset is outside allocation")
        address = allocation["physical_base_address"] + logical_offset
        request_count = ceil_div(address % quantum + row["byte_count"], quantum)
        start_bank = (address // quantum) % banks
        object_first = allocation["physical_base_address"]
        object_last = object_first + allocation["object_span_bytes"] - 1
        object_row_start = object_first // config["sram_row_bytes"]
        object_row_end = object_last // config["sram_row_bytes"]
        if row["address_pattern"] == "CONTIGUOUS":
            counts = consecutive_bank_counts(start_bank, request_count, banks)
            lower_counts = counts
            upper_counts = counts
            mapping = "EXACT_CONTIGUOUS_WORD_INTERLEAVED"
            first_word_address = address - address % quantum
            last_word_address = first_word_address + request_count * quantum - 1
            row_start = first_word_address // config["sram_row_bytes"]
            row_end = last_word_address // config["sram_row_bytes"]
            visited_row_lower = row_end - row_start + 1
            visited_row_upper = visited_row_lower
            visited_row_rle = (
                str(row_start) if row_start == row_end
                else "{}-{}".format(row_start, row_end)
            )
        elif row["address_pattern"] in (
            "CYCLIC_WEIGHT_OBJECT_COMPRESSED", "TEMPORAL_STATE_REVISIT_COMPRESSED"
        ):
            object_words = ceil_div(allocation["object_span_bytes"], quantum)
            counts = cyclic_bank_counts(start_bank, request_count, object_words, banks)
            lower_counts = counts
            upper_counts = counts
            mapping = "EXACT_CYCLIC_WORD_INTERLEAVED_COMPRESSED"
            start_object_word = logical_offset // quantum
            visited = cyclic_visited_rows(
                allocation["physical_base_address"], quantum, banks,
                start_object_word, request_count, object_words,
            )
            row_start = visited["row_index_start"]
            row_end = visited["row_index_end"]
            visited_row_lower = visited["visited_row_count"]
            visited_row_upper = visited_row_lower
            visited_row_rle = visited["row_interval_rle"]
        elif row["address_pattern"] == "ROW_SELECTED_WITHIN_PREVIOUS_TIMESTEP":
            counts = None
            lower_counts = consecutive_bank_counts(start_bank, request_count, banks)
            upper_counts = [0] * banks
            upper_counts[start_bank] = request_count
            mapping = "BOUNDED_UNKNOWN_ROW_SELECTION_BALANCED_TO_SINGLE_BANK"
            row_start = ""
            row_end = ""
            object_rows = object_row_end - object_row_start + 1
            visited_row_lower = 1
            visited_row_upper = min(request_count, object_rows)
            visited_row_rle = "UNKNOWN_WITHIN_OBJECT_ROW_ENVELOPE"
            address = None
        else:
            raise ValueError("unsupported SRAM address pattern")
        physical_instance = allocation["instance_id"]
    aggregate_requests_per_tick = (
        config["dram_global_bursts_per_tick"] if tier == "DRAM" else banks * ports
    )
    ideal_ticks = ceil_div(request_count, aggregate_requests_per_tick)
    lower_ticks = max(ideal_ticks, ceil_div(max(lower_counts), ports))
    upper_ticks = max(ideal_ticks, ceil_div(max(upper_counts), ports))
    if lower_ticks < ideal_ticks or upper_ticks < lower_ticks:
        raise ValueError("invalid port service envelope")
    return {
        "m22_transaction_id": row["transaction_id"],
        "identity": row["identity"],
        "variant": row["variant"],
        "request_issue_order": row["request_issue_order"],
        "trace_dependency_m22_id": trace_dependency,
        "object_hazard_dependency_m22_id": object_dependency,
        "sample_id": row["sample_id"],
        "call_index": row["call_index"],
        "event_kind": row["event_kind"],
        "name": row["name"],
        "phase": row["phase"],
        "tier": tier,
        "direction": row["direction"],
        "physical_instance_id": physical_instance,
        "physical_address": (
            "0x{:016x}".format(address) if address is not None
            else "UNKNOWN_WITHIN_PHYSICAL_OBJECT"
        ),
        "physical_object_row_index_start": object_row_start,
        "physical_object_row_index_end": object_row_end,
        "payload_bytes": row["byte_count"],
        "transfer_quantum_bytes": quantum,
        "compressed_request_count": request_count,
        "transferred_bytes_with_edge_padding": request_count * quantum,
        "bank_count": banks,
        "ports_per_bank": ports,
        "bank_request_count_rle": (
            encode_bank_counts(counts) if counts is not None else "UNKNOWN_ROW_SELECTION"
        ),
        "bank_mapping_evidence": mapping,
        "row_index_start": row_start,
        "row_index_end": row_end,
        "visited_row_count_lower_bound": visited_row_lower,
        "visited_row_count_upper_bound": visited_row_upper,
        "visited_row_interval_rle": visited_row_rle,
        "ideal_aggregate_port_ticks": ideal_ticks,
        "bank_service_ticks_lower_bound": lower_ticks,
        "bank_service_ticks_upper_bound": upper_ticks,
        "bank_conflict_stall_ticks_lower_bound": lower_ticks - ideal_ticks,
        "bank_conflict_stall_ticks_upper_bound": upper_ticks - ideal_ticks,
        "m22_transport_ticks": (
            row["serialized_service_end_exclusive"] - row["serialized_service_start"]
        ),
        "m22_address_pattern": row["address_pattern"],
        "m22_evidence_class": row["evidence_class"],
    }


def validate_config(config):
    required = (
        "dram_burst_bytes", "dram_global_bursts_per_tick", "dram_banks", "dram_row_bytes",
        "dram_read_ports_per_bank", "dram_write_ports_per_bank",
        "sram_word_bytes", "sram_banks", "sram_row_bytes",
        "sram_read_ports_per_bank", "sram_write_ports_per_bank",
        "sram_allocation_alignment_bytes", "sram_physical_base_address",
    )
    if set(config) != set(required) or any(
        not isinstance(config[key], int) or config[key] <= 0 for key in required
    ):
        raise ValueError("M23 resource configuration is incomplete or nonpositive")
    if config["sram_allocation_alignment_bytes"] % (
        config["sram_word_bytes"] * config["sram_banks"]
    ):
        raise ValueError("SRAM allocation alignment must be a complete bank stripe")
    if config["sram_row_bytes"] != config["sram_word_bytes"] * config["sram_banks"]:
        raise ValueError("SRAM row bytes must equal one complete word-interleaved bank stripe")
    if config["sram_physical_base_address"] % config["sram_allocation_alignment_bytes"]:
        raise ValueError("SRAM physical aperture base is not allocation aligned")


def build(summary, rows, config, input_meta):
    validate_config(config)
    m22_config = summary.get("config", {})
    if (
        config["dram_burst_bytes"] * config["dram_global_bursts_per_tick"]
        != m22_config.get("dram_bytes_per_cycle")
        or config["sram_word_bytes"] * config["sram_banks"]
        != m22_config.get("sram_bytes_per_cycle")
    ):
        raise ValueError("M23 aggregate transport resources do not match frozen M22")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["identity"], row["variant"])].append(row)
    payload = {
        "schema": "m23_trace_live_physical_memory_envelope_v1",
        "status": "PASS_PHYSICAL_REUSE_AND_BANK_PORT_ENVELOPE_NOT_SYSTEM_CYCLES",
        "generator": {"name": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "input": input_meta,
        "resource_config": config,
        "capacity_definition": {
            "peak_live_aligned_bytes": "MAXIMUM_SIMULTANEOUSLY_LIVE_M22_SRAM_OBJECTS_WITH_96B_ALIGNMENT",
            "allocator_capacity_bytes": "COALESCED_BEST_FIT_HIGH_WATER_FOR_ALL_M22_SRAM_OBJECT_CATEGORIES",
            "two_copy_largest_stream_capacity_bound_bytes": "ANALYTICAL_TWO_COPIES_OF_LARGEST_STREAM_TENSOR_NOT_A_SCHEDULED_PING_PONG",
            "allocator_plus_extra_largest_stream_capacity_bound_bytes": "ANALYTICAL_ALLOCATOR_HIGH_WATER_PLUS_ONE_EXTRA_LARGEST_STREAM_TENSOR",
            "included_categories": [
                "activation/input and operator/output", "weights", "ATLIF input/output/state/parameters",
                "Motion state/selector", "observed abstract H67 attention summaries",
            ],
            "qualification": "BOUNDARY_MATERIALIZED_TRACE_WORKING_SET_NOT_A_PROPOSED_ON_CHIP_SRAM_MACRO; REQUIRES_TILING; LOCAL_ATTENTION_UNKNOWN",
        },
        "identities": {},
        "claim_boundary": {
            "permitted": [
                "trace-live SRAM allocation capacity under M22 boundary materialization",
                "fixed-burst DRAM payload/wire byte and bank-service counts",
                "word-interleaved SRAM exact contiguous/cyclic and bounded row-selected port service",
                "same-resource Local, Motion shared-state, and Motion explicit-copy envelopes",
            ],
            "forbidden": [
                "DRAMsim3 timing, refresh, row-buffer hit rate, or calibrated latency",
                "compute overlap, system cycles, FPS, energy, or speedup",
                "physical macro PPA or place-and-route feasibility",
                "claiming missing Local attention or abstract H67 attention as zero-cost physical traffic",
            ],
        },
    }
    all_allocations = []
    all_schedule = []
    for (identity, variant), records in sorted(grouped.items()):
        lifetimes = collect_sram_lifetimes(records)
        allocations, allocator = allocate_lifetimes(
            lifetimes, config["sram_allocation_alignment_bytes"],
            config["sram_physical_base_address"],
        )
        allocation_map = {row["instance_key"]: row for row in allocations}
        previous_transaction = -1
        previous_object = {}
        counters = Counter()
        schedule = []
        for row in records:
            instance_key = (row["object_id"], row["sample_id"], row["call_index"])
            allocation = allocation_map.get(instance_key) if row["tier"] == "SRAM" else None
            object_key = (row["tier"], instance_key)
            scheduled = schedule_transaction(
                row, allocation, config, previous_transaction,
                previous_object.get(object_key, -1),
            )
            schedule.append(scheduled)
            previous_transaction = row["transaction_id"]
            previous_object[object_key] = row["transaction_id"]
            counters["{}_payload_bytes".format(row["tier"].lower())] += row["byte_count"]
            counters["{}_compressed_requests".format(row["tier"].lower())] += scheduled[
                "compressed_request_count"
            ]
            counters["{}_wire_bytes".format(row["tier"].lower())] += scheduled[
                "transferred_bytes_with_edge_padding"
            ]
            direction_prefix = "{}_{}".format(
                row["tier"].lower(), row["direction"].lower()
            )
            counters["{}_payload_bytes".format(direction_prefix)] += row["byte_count"]
            counters["{}_compressed_requests".format(direction_prefix)] += scheduled[
                "compressed_request_count"
            ]
            counters["{}_wire_bytes".format(direction_prefix)] += scheduled[
                "transferred_bytes_with_edge_padding"
            ]
            for field in (
                "ideal_aggregate_port_ticks", "bank_service_ticks_lower_bound",
                "bank_service_ticks_upper_bound", "bank_conflict_stall_ticks_lower_bound",
                "bank_conflict_stall_ticks_upper_bound",
            ):
                counters["{}_{}".format(direction_prefix, field)] += scheduled[field]
            for field in (
                "m22_transport_ticks", "ideal_aggregate_port_ticks",
                "bank_service_ticks_lower_bound", "bank_service_ticks_upper_bound",
                "bank_conflict_stall_ticks_lower_bound",
                "bank_conflict_stall_ticks_upper_bound",
            ):
                counters[field] += scheduled[field]
        expected_transport = summary["identities"][identity]["variants"][variant][
            "totals"
        ]["serialized_byte_service_ticks"]
        if counters["m22_transport_ticks"] != expected_transport:
            raise ValueError("M23/M22 transport tick conservation failed")
        category_bytes = Counter()
        for allocation in allocations:
            category_bytes[allocation["category"]] += allocation["object_span_bytes"]
            exported = dict(allocation)
            exported.pop("instance_key")
            exported["identity"] = identity
            exported["variant"] = variant
            exported["physical_base_address"] = "0x{:016x}".format(
                allocation["physical_base_address"]
            )
            exported["logical_base_address"] = "0x{:016x}".format(
                allocation["logical_base_address"]
            )
            all_allocations.append(exported)
        all_schedule.extend(schedule)
        identity_payload = payload["identities"].setdefault(identity, {
            "attention": {}, "capacity_completeness": "", "variants": {},
        })
        source_identity = summary["identities"][identity]
        if source_identity["attention_coverage_status"] == (
            "ABSTRACT_PACKED1_COUNTER_SUMMARY_NOT_PHYSICAL_TRAFFIC"
        ):
            attention_payload = sum(
                row["payload_bytes"] for row in schedule
                if row["event_kind"] == "attention"
            )
            attention = {
                "status": "NONZERO_ABSTRACT_PACKED1_LOWER_BOUND_NOT_RTL_PHYSICAL",
                "execution_records": source_identity["attention_execution_records"],
                "scheduled_abstract_payload_bytes": attention_payload,
                "unmodeled_physical_bytes": "UNKNOWN_NONZERO_ADDITIONAL_OR_EQUAL",
            }
            if attention_payload <= 0:
                raise ValueError("H67 abstract attention was silently made free")
        else:
            module_count = source_identity["profile_identity"]["module_counts"].get(
                "ShiftmaxAttention", 0
            )
            missing_events = module_count * source_identity["sample_count"]
            attention = {
                "status": "MISSING_TRACE_UNKNOWN_NONZERO_NOT_SCHEDULED",
                "execution_records": 0,
                "minimum_missing_module_calls": missing_events,
                "scheduled_abstract_payload_bytes": 0,
                "unmodeled_physical_bytes": "UNKNOWN_NONZERO",
            }
            if missing_events <= 0:
                raise ValueError("Local attention unknown boundary is not nonzero")
        if identity_payload["attention"] and identity_payload["attention"] != attention:
            raise ValueError("attention boundary changed across variants")
        identity_payload["attention"] = attention
        completeness = (
            "OBSERVED_M22_WITH_ABSTRACT_ATTENTION_LOWER_BOUND_NOT_FULL_PHYSICAL"
            if attention["status"].startswith("NONZERO_ABSTRACT")
            else "OBSERVED_M22_EXCLUDING_UNKNOWN_NONZERO_LOCAL_ATTENTION"
        )
        if identity_payload["capacity_completeness"] and (
            identity_payload["capacity_completeness"] != completeness
        ):
            raise ValueError("capacity completeness changed across variants")
        identity_payload["capacity_completeness"] = completeness
        identity_payload["variants"][variant] = {
            "m22_transaction_records": len(records),
            "allocation": allocator,
            "logical_sram_span_bytes_m22": source_identity["variants"][variant][
                "sram_logical_span_bytes"
            ],
            "logical_span_to_allocator_capacity_ratio": (
                source_identity["variants"][variant]["sram_logical_span_bytes"]
                / float(allocator["allocator_capacity_bytes"])
            ),
            "lifetime_payload_bytes_by_category": dict(category_bytes),
            "transport_and_bank_service": dict(counters),
            "claim": "BANK_PORT_SERVICE_ENVELOPE_NOT_SYSTEM_SPEEDUP",
        }
    for identity, identity_payload in payload["identities"].items():
        local = identity_payload["variants"]["local_line"]
        identity_payload["motion_models_vs_local"] = {}
        for variant in (
            "motion_selector_shared_state", "motion_selector_explicit_copy"
        ):
            if variant not in identity_payload["variants"]:
                continue
            motion = identity_payload["variants"][variant]
            local_service = local["transport_and_bank_service"]
            motion_service = motion["transport_and_bank_service"]
            identity_payload["motion_models_vs_local"][variant] = {
                "same_resource_config": True,
                "allocator_capacity_byte_delta": (
                    motion["allocation"]["allocator_capacity_bytes"]
                    - local["allocation"]["allocator_capacity_bytes"]
                ),
                "peak_live_aligned_byte_delta": (
                    motion["allocation"]["peak_live_aligned_bytes"]
                    - local["allocation"]["peak_live_aligned_bytes"]
                ),
                "dram_payload_byte_delta": (
                    motion_service["dram_payload_bytes"] - local_service["dram_payload_bytes"]
                ),
                "sram_payload_byte_delta": (
                    motion_service["sram_payload_bytes"] - local_service["sram_payload_bytes"]
                ),
                "dram_fixed_burst_delta": (
                    motion_service["dram_compressed_requests"]
                    - local_service["dram_compressed_requests"]
                ),
                "sram_word_request_delta": (
                    motion_service["sram_compressed_requests"]
                    - local_service["sram_compressed_requests"]
                ),
                "m22_transport_tick_delta": (
                    motion_service["m22_transport_ticks"]
                    - local_service["m22_transport_ticks"]
                ),
                "serialized_bank_service_lower_tick_delta": (
                    motion_service["bank_service_ticks_lower_bound"]
                    - local_service["bank_service_ticks_lower_bound"]
                ),
                "serialized_bank_service_upper_tick_delta": (
                    motion_service["bank_service_ticks_upper_bound"]
                    - local_service["bank_service_ticks_upper_bound"]
                ),
                "bank_conflict_stall_upper_tick_delta": (
                    motion_service["bank_conflict_stall_ticks_upper_bound"]
                    - local_service["bank_conflict_stall_ticks_upper_bound"]
                ),
                "claim": "ABSOLUTE_MEMORY_SERVICE_DELTA_NOT_SYSTEM_SPEEDUP",
            }
    payload["allocation_records"] = len(all_allocations)
    payload["schedule_records"] = len(all_schedule)
    if len(all_schedule) != len(rows):
        raise ValueError("M23 schedule does not preserve M22 transaction cardinality")
    return payload, all_allocations, all_schedule


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty M23 CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output, payload, allocations, schedule, contract_path, contract_sha):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    allocation_path = output / "m23_buffer_allocations.csv"
    schedule_path = output / "m23_compressed_bank_schedule.csv"
    summary_path = output / "m23_summary.json"
    report_path = output / "REPORT.md"
    write_csv(allocation_path, allocations)
    write_csv(schedule_path, schedule)
    payload["allocations_sha256"] = sha256(allocation_path)
    payload["schedule_sha256"] = sha256(schedule_path)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# M23 trace-live physical memory and bank/port envelope\n\n",
        "M23 replaces M22's logical SRAM address span with deterministic trace-live reuse, then maps every M22 transaction to a compressed fixed-quantum bank/port service record. These figures are not DRAMsim3 or system speedup.\n\n",
        "| identity | variant | M22 logical SRAM B | allocator B | peak live B | two-copy largest-stream bound B | allocator + extra stream B | DRAM bursts | SRAM words | M22 transport ticks | serialized port ticks lower..upper | conflict stalls lower..upper |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for identity, identity_payload in sorted(payload["identities"].items()):
        for variant, result in sorted(identity_payload["variants"].items()):
            allocation = result["allocation"]
            service = result["transport_and_bank_service"]
            report.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}..{} | {}..{} |\n".format(
                    identity, variant, result["logical_sram_span_bytes_m22"],
                    allocation["allocator_capacity_bytes"], allocation["peak_live_aligned_bytes"],
                    allocation["two_copy_largest_stream_capacity_bound_bytes"],
                    allocation["allocator_plus_extra_largest_stream_capacity_bound_bytes"],
                    service.get("dram_compressed_requests", 0),
                    service.get("sram_compressed_requests", 0),
                    service["m22_transport_ticks"],
                    service["bank_service_ticks_lower_bound"],
                    service["bank_service_ticks_upper_bound"],
                    service["bank_conflict_stall_ticks_lower_bound"],
                    service["bank_conflict_stall_ticks_upper_bound"],
                )
            )
    report.append("\nAttention boundaries remain fail-open for cost but fail-closed for claims: H67 is a nonzero abstract packed-summary lower bound; Local has at least the frozen module-call count with unknown nonzero bytes.\n")
    report.append("\nThe live peak and best-fit allocator include every observed M22 SRAM category (stream tensors, weights, ATLIF input/output/state/parameters, Motion metadata/state, and observed abstract H67 attention) at 96-byte alignment. They are a boundary-materialized trace working set, not a proposed on-chip SRAM macro. The two-copy and allocator-plus-extra-stream values are analytical capacity bounds, not a scheduled ping-pong placement; tiling is still mandatory, and missing Local attention can only increase its bound.\n")
    report.append("\nTransport ticks reproduce M22's serialized byte ledger. Bank-service ticks are a separate sequential port envelope. Neither is compute-overlapped system latency, FPS, energy, or speedup.\n")
    report_path.write_text("".join(report), encoding="utf-8")
    test_path = Path(__file__).resolve().parents[1] / "tests/test_m23_physical_memory_schedule.py"
    output_manifest = {
        "schema": "m23_physical_memory_output_manifest_v1",
        "status": "FROZEN_REPRODUCIBLE_PORT_SERVICE_ENVELOPE",
        "input_contract": {"path": str(contract_path), "sha256": contract_sha},
        "sources_sha256": {
            Path(__file__).name: sha256(Path(__file__)),
            test_path.name: sha256(test_path),
        },
        "artifacts": {},
        "allocation_records": len(allocations),
        "schedule_records": len(schedule),
        "claim": "PHYSICAL_REUSE_AND_BANK_PORT_ENVELOPE_NOT_DRAMSIM_OR_SYSTEM_SPEEDUP",
    }
    for path in (allocation_path, schedule_path, summary_path, report_path):
        output_manifest["artifacts"][path.name] = {
            "sha256": sha256(path), "bytes": path.stat().st_size,
        }
    (output / "m23_output_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--expected-input-contract-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dram-burst-bytes", type=int, default=64)
    parser.add_argument("--dram-global-bursts-per-tick", type=int, default=3)
    parser.add_argument("--dram-banks", type=int, default=16)
    parser.add_argument("--dram-row-bytes", type=int, default=8192)
    parser.add_argument("--dram-read-ports-per-bank", type=int, default=1)
    parser.add_argument("--dram-write-ports-per-bank", type=int, default=1)
    parser.add_argument("--sram-word-bytes", type=int, default=4)
    parser.add_argument("--sram-banks", type=int, default=24)
    parser.add_argument("--sram-row-bytes", type=int, default=96)
    parser.add_argument("--sram-read-ports-per-bank", type=int, default=1)
    parser.add_argument("--sram-write-ports-per-bank", type=int, default=1)
    parser.add_argument("--sram-allocation-alignment-bytes", type=int, default=96)
    parser.add_argument(
        "--sram-physical-base-address", type=lambda value: int(value, 0),
        default=0x80000040,
    )
    args = parser.parse_args()
    config = {
        key: getattr(args, key) for key in (
            "dram_burst_bytes", "dram_global_bursts_per_tick", "dram_banks", "dram_row_bytes",
            "dram_read_ports_per_bank", "dram_write_ports_per_bank",
            "sram_word_bytes", "sram_banks", "sram_row_bytes",
            "sram_read_ports_per_bank", "sram_write_ports_per_bank",
            "sram_allocation_alignment_bytes", "sram_physical_base_address",
        )
    }
    contract, contract_sha, artifact = load_contract(
        args.input_contract, args.expected_input_contract_sha256, args.repo_root
    )
    summary, rows = load_m22(contract, artifact)
    input_meta = {
        "contract_path": str(args.input_contract),
        "contract_sha256": contract_sha,
        "m22_artifact_directory": contract["artifact_directory"],
        "m22_transactions_sha256": contract["transactions_sha256"],
        "m22_transaction_records": contract["transaction_records"],
        "m22_output_manifest_sha256": contract["files_sha256"]["m22_output_manifest.json"],
    }
    payload, allocations, schedule = build(summary, rows, config, input_meta)
    write_outputs(
        args.output, payload, allocations, schedule, args.input_contract, contract_sha
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
