#!/usr/bin/env python3
"""M40 fail-closed conflict-aware event scheduler and real-trace audit."""

from __future__ import print_function

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import zlib


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = HW_ROOT / (
    "contracts/m40_conflict_aware_event_schedule_contract_r1_20260822.json")
EXPECTED_CONTRACT_SHA256 = (
    "1eeeea8f1778f45305226dbccf31a920586dff3eb14ee0bf684ef833728f9018")
TARGETS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
TARGET_TO_M35_PRODUCER = {
    "sttmultires_unet.resblocks.0.conv1.0":
        "sttmultires_unet.resblocks.0.sn1.spiking_neuron",
    "sttmultires_unet.resblocks.0.conv2.0":
        "sttmultires_unet.resblocks.0.sn2.spiking_neuron",
    "sttmultires_unet.resblocks.1.conv1.0":
        "sttmultires_unet.resblocks.1.sn1.spiking_neuron",
    "sttmultires_unet.resblocks.1.conv2.0":
        "sttmultires_unet.resblocks.1.sn2.spiking_neuron",
}
EVENT_FIELDS = (
    "event_id", "line", "invocation_id", "sample_id", "operator",
    "temporal_step", "flush_id", "event_order", "source_index",
    "destination_index", "activation_s8", "weight_s8", "contribution_s16",
    "accumulator_before_s32", "accumulator_after_s32",
    "input_sram_address", "weight_sram_address", "accumulator_sram_address",
    "weight_tile_id", "weight_tile_bytes", "is_last_for_destination",
    "threshold_raw_uq0p24", "expected_scaled_s56", "motion_delta_direction",
)
FORBIDDEN = (
    "real_four_bottleneck_executable_schedule_admitted",
    "real_local_motion_cycle_statistics_admitted",
    "real_fixed_point_m35_miter_admitted", "physical_sram_macro_admitted",
    "integrated_rtl_admitted", "integrated_vcs_dc_sta_formality_admitted",
    "system_speedup_admitted", "ppa_admitted", "power_energy_admitted",
    "external_accelerator_comparison_admitted", "headline_admitted",
    "best_paper_admitted",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(raw):
    raise ValueError("non-standard JSON numeric constant: {}".format(raw))


def read_json(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject_constant)


def mismatch(actual, expected, path="$"):
    if type(actual) is not type(expected):
        return "{} type {} != {}".format(
            path, type(actual).__name__, type(expected).__name__)
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return "{} key population differs".format(path)
        for key in sorted(actual):
            found = mismatch(actual[key], expected[key], "{}.{}".format(path, key))
            if found is not None:
                return found
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return "{} length differs".format(path)
        for index, pair in enumerate(zip(actual, expected)):
            found = mismatch(pair[0], pair[1], "{}[{}]".format(path, index))
            if found is not None:
                return found
        return None
    return None if actual == expected else "{} value differs".format(path)


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "module import failed: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract(path=DEFAULT_CONTRACT):
    require(sha256(DEFAULT_CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M40 canonical contract identity drift")
    canonical = read_json(DEFAULT_CONTRACT)
    contract = read_json(path)
    found = mismatch(contract, canonical)
    require(found is None, "M40 contract recursive type-strict drift: {}".format(found))
    payloads = {}
    for name, item in sorted(contract["inputs"].items()):
        require(type(item) is dict and set(item) == {"path", "sha256"},
                "M40 input descriptor population drift")
        source = resolve(item["path"])
        require(source.is_file(), "M40 input missing: {}".format(name))
        require(sha256(source) == item["sha256"],
                "M40 input identity drift: {}".format(name))
        payloads[name] = read_json(source) if source.suffix == ".json" else source
    return contract, payloads


def word_and_bank(address, banks, word_bytes):
    require(type(address) is int and address >= 0, "address domain/type violation")
    word = address // word_bytes
    return word, word % banks


def validate_event(event):
    require(type(event) is dict and set(event) == set(EVENT_FIELDS),
            "event field population drift")
    for key in ("event_id", "sample_id", "temporal_step", "flush_id",
                "event_order", "source_index", "destination_index",
                "activation_s8", "weight_s8", "contribution_s16",
                "accumulator_before_s32", "accumulator_after_s32",
                "input_sram_address", "weight_sram_address",
                "accumulator_sram_address", "weight_tile_id",
                "weight_tile_bytes", "threshold_raw_uq0p24",
                "expected_scaled_s56"):
        require(type(event[key]) is int, "event integer type drift: {}".format(key))
    require(type(event["is_last_for_destination"]) is bool,
            "event last flag type drift")
    require(event["line"] in ("Local", "Motion"), "event line drift")
    require(-128 <= event["activation_s8"] <= 127 and
            -128 <= event["weight_s8"] <= 127, "signed8 operand overflow")
    require(event["contribution_s16"] ==
            event["activation_s8"] * event["weight_s8"],
            "event product miter mismatch")
    require(event["accumulator_after_s32"] ==
            event["accumulator_before_s32"] + event["contribution_s16"],
            "event accumulator chain mismatch")
    require(-(1 << 31) <= event["accumulator_before_s32"] < (1 << 31) and
            -(1 << 31) <= event["accumulator_after_s32"] < (1 << 31),
            "signed32 accumulator overflow")
    require(event["weight_tile_bytes"] > 0, "weight tile extent invalid")
    require(type(event["motion_delta_direction"]) is int,
            "event integer type drift: motion_delta_direction")
    require(event["motion_delta_direction"] in (-2, -1, 0, 1, 2),
            "motion delta direction drift")
    if event["line"] == "Local":
        require(event["motion_delta_direction"] == 0,
                "Local event has Motion direction")
    if event["is_last_for_destination"]:
        require(0 <= event["threshold_raw_uq0p24"] < (1 << 24),
                "last event threshold outside UQ0.24")
        require(event["expected_scaled_s56"] ==
                event["accumulator_after_s32"] * event["threshold_raw_uq0p24"],
                "M35 late-scale integer miter mismatch")
    else:
        require(event["threshold_raw_uq0p24"] == 0 and
                event["expected_scaled_s56"] == 0,
                "non-last event carries late-scale payload")


def schedule_events(events, config):
    """Executable small-trace oracle with bank conflicts, credits and residency."""
    expected_config = {
        "banks", "word_bytes", "lanes", "queue_depth", "ingress_width",
        "weight_residency_bytes",
    }
    require(type(config) is dict and set(config) == expected_config,
            "scheduler config population drift")
    for key in expected_config:
        require(type(config[key]) is int and config[key] > 0,
                "scheduler config domain/type drift: {}".format(key))
    require(type(events) is list and len(events) > 0, "empty event trace")
    for event in events:
        validate_event(event)
    require([event["event_id"] for event in events] == list(range(len(events))),
            "event id/order population drift")
    require([event["event_order"] for event in events] == list(range(len(events))),
            "event order drift")
    seen_flush = []
    for event in events:
        if not seen_flush or seen_flush[-1] != event["flush_id"]:
            require(event["flush_id"] not in seen_flush,
                    "flush boundary is not contiguous")
            seen_flush.append(event["flush_id"])

    banks = config["banks"]
    word_bytes = config["word_bytes"]
    pending = []
    source_index = 0
    cycles = 0
    retired = []
    bank_conflict_deferrals = 0
    queue_credit_stall_events = 0
    input_words = 0
    weight_words = 0
    accumulator_words = 0
    residency = []
    resident_bytes = 0
    weight_load_bytes = 0
    weight_evictions = 0

    while source_index < len(events) or pending:
        active_flush = (pending[0]["flush_id"] if pending
                        else events[source_index]["flush_id"])
        offered = 0
        while (source_index < len(events) and offered < config["ingress_width"]
               and events[source_index]["flush_id"] == active_flush):
            if len(pending) >= config["queue_depth"]:
                queue_credit_stall_events += 1
                break
            event = events[source_index]
            tile = event["weight_tile_id"]
            if tile not in [item[0] for item in residency]:
                size = event["weight_tile_bytes"]
                require(size <= config["weight_residency_bytes"],
                        "weight tile exceeds residency capacity")
                while resident_bytes + size > config["weight_residency_bytes"]:
                    evicted = residency.pop(0)
                    resident_bytes -= evicted[1]
                    weight_evictions += 1
                residency.append((tile, size))
                resident_bytes += size
                weight_load_bytes += size
            else:
                position = [item[0] for item in residency].index(tile)
                residency.append(residency.pop(position))
            pending.append(event)
            source_index += 1
            offered += 1

        input_by_bank = {}
        weight_by_bank = {}
        acc_by_bank = {}
        selected = []
        blocked_destinations = set()
        for index, event in enumerate(pending):
            if len(selected) >= config["lanes"]:
                break
            destination = (event["invocation_id"], event["destination_index"])
            if destination in blocked_destinations:
                continue
            in_word, in_bank = word_and_bank(
                event["input_sram_address"], banks, word_bytes)
            wt_word, wt_bank = word_and_bank(
                event["weight_sram_address"], banks, word_bytes)
            ac_word, ac_bank = word_and_bank(
                event["accumulator_sram_address"], banks, word_bytes)
            conflict = ((in_bank in input_by_bank and input_by_bank[in_bank] != in_word)
                        or (wt_bank in weight_by_bank and
                            weight_by_bank[wt_bank] != wt_word)
                        or (ac_bank in acc_by_bank and acc_by_bank[ac_bank] != ac_word))
            if conflict:
                bank_conflict_deferrals += 1
                blocked_destinations.add(destination)
                continue
            input_by_bank[in_bank] = in_word
            weight_by_bank[wt_bank] = wt_word
            acc_by_bank[ac_bank] = ac_word
            selected.append(index)
        require(selected, "scheduler deadlock")
        input_words += len(input_by_bank)
        weight_words += len(weight_by_bank)
        accumulator_words += len(acc_by_bank)
        selected_set = set(selected)
        chosen = [event for index, event in enumerate(pending) if index in selected_set]
        pending = [event for index, event in enumerate(pending)
                   if index not in selected_set]
        retired.extend(event["event_id"] for event in chosen)
        cycles += 1

    require(sorted(retired) == list(range(len(events))) and len(retired) == len(events),
            "event conservation failure")
    return {
        "cycles": cycles,
        "input_word_reads": input_words,
        "weight_word_reads": weight_words,
        "accumulator_word_read_writes": accumulator_words,
        "bank_conflict_event_deferrals": bank_conflict_deferrals,
        "queue_credit_stall_events": queue_credit_stall_events,
        "weight_tile_load_bytes": weight_load_bytes,
        "weight_tile_evictions": weight_evictions,
        "events_offered": len(events),
        "events_retired": len(retired),
        "events_lost": len(events) - len(retired),
        "flushes": len(seen_flush),
        "m35_last_event_miters": sum(
            1 for event in events if event["is_last_for_destination"]),
        "m35_integer_mismatches": 0,
        "event_conservation_exact": True,
    }


def percentile_nearest_rank(values, numerator, denominator):
    require(values and numerator > 0 and denominator > 0 and numerator <= denominator,
            "percentile domain violation")
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[rank - 1]


def distribution(values):
    require(values, "empty distribution")
    return {
        "count": len(values), "minimum": min(values), "maximum": max(values),
        "mean_exact": {"numerator": sum(values), "denominator": len(values)},
        "p95_nearest_rank": percentile_nearest_rank(values, 95, 100),
        "p99_nearest_rank": percentile_nearest_rank(values, 99, 100),
    }


def bitmap_tables(height, width):
    spatial = height * width
    period = spatial // math.gcd(8, spatial)
    popcount = [bin(value).count("1") for value in range(256)]
    weighted = []
    for phase in range(period):
        base = (phase * 8) % spatial
        row = []
        for value in range(256):
            total = 0
            for bit in range(8):
                if value & (1 << bit):
                    position = (base + bit) % spatial
                    y, x = divmod(position, width)
                    total += (2 if y in (0, height - 1) else 3) * (
                        2 if x in (0, width - 1) else 3)
            row.append(total)
        weighted.append(row)
    return period, popcount, weighted


def audit_packed_record(trace_dir, record):
    require(record["shape"] == [10, 1, 768, 15, 20], "packed shape drift")
    require(record["output_shape"] == [10, 1, 768, 15, 20],
            "packed output shape drift")
    geometry = record["module_geometry"]
    require(geometry["kernel_size"] == [3, 3] and geometry["stride"] == [1, 1]
            and geometry["padding"] == [1, 1] and geometry["dilation"] == [1, 1]
            and geometry["groups"] == 1 and geometry["in_channels"] == 768
            and geometry["out_channels"] == 768,
            "packed Conv3x3 geometry drift")
    path = trace_dir / record["packed_file"]
    require(path.is_file() and sha256(path) == record["packed_file_sha256"],
            "packed bitmap identity drift")
    raw = path.read_bytes()
    plane_bytes = record["positive_plane_bytes"]
    require(len(raw) == 3 * plane_bytes == record["packed_file_bytes"],
            "packed bitmap extent drift")
    pos = raw[:plane_bytes]
    neg = raw[plane_bytes:2 * plane_bytes]
    changed_plane = raw[2 * plane_bytes:]
    require(not any(neg), "M40 amplitude code unexpectedly has negative values")
    codebook = record["value_bit_pattern_population"]
    require(codebook["unique_float32_bit_patterns"] == 2 and
            codebook["full_codebook_in_manifest"] is True and
            type(codebook["codebook"]) is list and len(codebook["codebook"]) == 2,
            "M40 source is not an exact two-code alphabet")
    code_by_bits = {item["float32_bits_hex"]: item["count"]
                    for item in codebook["codebook"]}
    require("00000000" in code_by_bits and len(code_by_bits) == 2,
            "M40 two-code alphabet lacks exact float32 zero")
    nonzero_bits = next(bits for bits in code_by_bits if bits != "00000000")
    nonzero_word = struct.pack("<I", int(nonzero_bits, 16))
    value_path = trace_dir / record["value_payload_file"]
    require(value_path.is_file() and sha256(value_path) ==
            record["value_payload_sha256"], "M40 value payload identity drift")
    value_raw = zlib.decompress(value_path.read_bytes())
    require(len(value_raw) == record["input_content_bytes"] and
            hashlib.sha256(value_raw).hexdigest() == record["input_content_sha256"],
            "M40 decompressed float32 payload identity drift")
    zero_word = b"\x00\x00\x00\x00"
    decode_table = tuple(b"".join(
        nonzero_word if byte & (1 << bit) else zero_word for bit in range(8))
                         for byte in range(256))
    reconstructed = b"".join(decode_table[byte] for byte in pos)
    require(len(reconstructed) == len(value_raw) and reconstructed == value_raw,
            "M40 bitmap plus amplitude codebook is not bit-exact to float payload")
    amplitude_float = struct.unpack("<f", nonzero_word)[0]
    amplitude_uq0p24_raw = int(round(amplitude_float * (1 << 24)))
    require(amplitude_float == amplitude_uq0p24_raw / float(1 << 24),
            "M40 amplitude is not exact UQ0.24")
    bytes_per_timestep = plane_bytes // 10
    require(bytes_per_timestep * 10 == plane_bytes, "temporal plane extent drift")
    period, popcount, weighted = bitmap_tables(15, 20)
    local_sources = []
    local_pairs = []
    motion_sources = []
    motion_pairs = []
    direction_sources = {str(value): [] for value in (-2, -1, 1, 2)}
    direction_pairs = {str(value): [] for value in (-2, -1, 1, 2)}
    for timestep in range(10):
        start = timestep * bytes_per_timestep
        stop = start + bytes_per_timestep
        previous_start = (timestep - 1) * bytes_per_timestep
        src_local = pair_local = src_motion = pair_motion = 0
        dir_src = {value: 0 for value in (-2, -1, 1, 2)}
        dir_pair = {value: 0 for value in (-2, -1, 1, 2)}
        for offset, (pbyte, nbyte) in enumerate(zip(pos[start:stop], neg[start:stop])):
            current = pbyte | nbyte
            previous_p = 0 if timestep == 0 else pos[previous_start + offset]
            previous_n = 0 if timestep == 0 else neg[previous_start + offset]
            previous = previous_p | previous_n
            current_zero = ~(pbyte | nbyte) & 0xff
            previous_zero = ~(previous_p | previous_n) & 0xff
            direction_mask = {
                2: pbyte & previous_n,
                1: (pbyte & previous_zero) | (current_zero & previous_n),
                -1: (nbyte & previous_zero) | (current_zero & previous_p),
                -2: nbyte & previous_p,
            }
            changed = changed_plane[start + offset]
            phase = offset % period
            src_local += popcount[current]
            pair_local += weighted[phase][current]
            src_motion += popcount[changed]
            pair_motion += weighted[phase][changed]
            for direction in (-2, -1, 1, 2):
                mask = direction_mask[direction]
                dir_src[direction] += popcount[mask]
                dir_pair[direction] += weighted[phase][mask]
        local_sources.append(src_local)
        local_pairs.append(pair_local)
        motion_sources.append(src_motion)
        motion_pairs.append(pair_motion)
        for direction in (-2, -1, 1, 2):
            direction_sources[str(direction)].append(dir_src[direction])
            direction_pairs[str(direction)].append(dir_pair[direction])
    require(sum(local_sources) == record["nonzero_count"],
            "local source support conservation mismatch")
    require(local_sources == record["local_nonzero_count_by_timestep"],
            "local timestep support mismatch")
    require(motion_sources == record["motion_numeric_transition_count_by_timestep"],
            "Motion numeric transition mismatch")
    recorded_direction = record["motion_sign_delta_population_by_timestep"]
    for direction in ("-2", "-1", "1", "2"):
        require(direction_sources[direction] == recorded_direction[direction],
                "Motion sign direction mismatch")
    return {
        "sample_id": record["sample_id"], "operator": record["operator"],
        "all_values_integer": record["value_audit"]["all_values_integer"],
        "all_values_ternary": record["value_audit"]["all_values_ternary"],
        "noninteger_values": record["value_audit"]["noninteger_count"],
        "values_bit_exact_mitered": record["elements"],
        "value_bit_mismatches": 0,
        "amplitude_float32_bits_hex": nonzero_bits,
        "amplitude_uq0p24_raw": amplitude_uq0p24_raw,
        "Local": {
            "active_sources": sum(local_sources),
            "source_destination_pairs": sum(local_pairs),
            "active_products": sum(local_pairs) * 768,
            "exact_96_lane_product_lower_bound_cycles": sum(local_pairs) * 8,
        },
        "Motion": {
            "active_support_transitions": sum(motion_sources),
            "source_destination_pairs": sum(motion_pairs),
            "active_products": sum(motion_pairs) * 768,
            "exact_96_lane_product_lower_bound_cycles": sum(motion_pairs) * 8,
            "direction": {
                direction: {
                    "support_transitions": sum(direction_sources[direction]),
                    "source_destination_pairs": sum(direction_pairs[direction]),
                    "active_products": sum(direction_pairs[direction]) * 768,
                } for direction in ("-2", "-1", "1", "2")
            },
        },
    }


def audit_m22(path):
    fields_required_for_products = {
        "source_index", "destination_index", "input_channel", "output_channel",
        "kernel_y", "kernel_x", "activation_value", "weight_value",
        "accumulator_value", "input_physical_address", "weight_physical_address",
        "accumulator_physical_address",
    }
    counts = {name: 0 for name in TARGETS}
    evidence = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(fields_required_for_products - set(reader.fieldnames or []))
        for row in reader:
            if row["name"] in counts:
                counts[row["name"]] += 1
                evidence.add((row["phase"], row["address_pattern"], row["evidence_class"]))
    require(all(value == 540 for value in counts.values()),
            "M22 four-bottleneck aggregate row population drift")
    return {
        "aggregate_rows_by_operator": counts,
        "aggregate_evidence_classes": [list(item) for item in sorted(evidence)],
        "required_product_event_fields_missing": missing,
        "has_product_event_coordinates_and_operands": not missing,
        "conclusion": "M22_M23_ARE_CALL_LEVEL_BYTE_ENVELOPES_NOT_PRODUCT_EVENT_TRACES",
    }


def verify_upstream(contract, payloads):
    validator_path = resolve(contract["inputs"]["m39_review_validator"]["path"])
    validator = load_module(validator_path, "m40_m39_review_validator")
    review = validator.validate_review(
        resolve(contract["inputs"]["m39_review"]["path"]))
    require(mismatch(review, payloads["m39_review"]) is None,
            "M39 review rebuild drift")
    require(review["review"]["decision"] == "GO_MODEL_ONLY_CONDITIONAL_DSE",
            "M39 model-only review not GO")
    receipt = payloads["m35_receipt"]
    require(receipt["status"] ==
            "PASS_M35_R2_R7_AND_STRICT_FLAT_M33_STANDALONE_COMPARISON_NO_SYSTEM_OR_PAPER_PPA_CLAIM",
            "M35 receipt status drift")
    return {
        "m39_review_rebuilt": True,
        "m39_review_sha256": contract["inputs"]["m39_review"]["sha256"],
        "m35_receipt_sha256": contract["inputs"]["m35_receipt"]["sha256"],
        "m35_general_independent_admission": False,
    }


def build(contract_path=DEFAULT_CONTRACT):
    contract, payloads = load_contract(contract_path)
    upstream = verify_upstream(contract, payloads)
    trace = payloads["packed_source_manifest"]
    require(trace["schema"] == "m40_bottleneck_packed_source_trace_v1",
            "M40 packed source schema drift")
    require(trace["cohort"]["samples"] == 10 and
            trace["cohort"]["operators"] == list(TARGETS) and
            trace["cohort"]["records"] == 40,
            "M40 packed source cohort drift")
    trace_dir = resolve(contract["inputs"]["packed_source_manifest"]["path"]).parent
    rows = [audit_packed_record(trace_dir, record) for record in trace["records"]]
    require(len(rows) == 40, "M40 audit row population drift")
    noninteger = sum(row["noninteger_values"] for row in rows)
    require(noninteger > 0 and any(not row["all_values_integer"] for row in rows),
            "M40 expected noninteger bottleneck evidence disappeared")
    m35_by_producer = {row["producer"]: row
                       for row in payloads["m35_csd_result"]["thresholds"]}
    amplitude_by_operator = {}
    for operator in TARGETS:
        operator_rows = [row for row in rows if row["operator"] == operator]
        require(len(operator_rows) == 10, "M40 operator cohort population drift")
        bit_codes = {row["amplitude_float32_bits_hex"] for row in operator_rows}
        raw_codes = {row["amplitude_uq0p24_raw"] for row in operator_rows}
        require(len(bit_codes) == 1 and len(raw_codes) == 1,
                "M40 amplitude code changed across ten samples")
        producer = TARGET_TO_M35_PRODUCER[operator]
        threshold = m35_by_producer[producer]
        raw = next(iter(raw_codes))
        require(raw == threshold["threshold_uq0p24_raw"],
                "M40 amplitude does not equal M35 checkpoint threshold")
        amplitude_by_operator[operator] = {
            "float32_bits_hex": next(iter(bit_codes)),
            "uq0p24_raw": raw,
            "m35_producer": producer,
            "m35_delta": threshold["delta"],
            "m35_csd_terms": threshold["csd_terms"],
            "records_mitered": 10,
            "values_mitered": sum(row["values_bit_exact_mitered"]
                                   for row in operator_rows),
            "bit_mismatches": 0,
            "constant_across_ten_samples": True,
        }
    total_values_mitered = sum(row["values_bit_exact_mitered"] for row in rows)
    require(total_values_mitered == 92160000,
            "M40 exact value miter population drift")
    per_sample = {line: [] for line in ("Local", "Motion")}
    for sample_id in range(10):
        sample_rows = [row for row in rows if row["sample_id"] == sample_id]
        require(len(sample_rows) == 4, "M40 sample operator population drift")
        per_sample["Local"].append(sum(
            row["Local"]["exact_96_lane_product_lower_bound_cycles"]
            for row in sample_rows))
        per_sample["Motion"].append(sum(
            row["Motion"]["exact_96_lane_product_lower_bound_cycles"]
            for row in sample_rows))
    m22 = audit_m22(payloads["m22_transactions"])
    scheduler_contract = contract["scheduler_contract"]
    synthetic = contract["synthetic_oracle"]
    synthetic_result = schedule_events(synthetic["events"], synthetic["config"])
    require(mismatch(synthetic_result, synthetic["expected_result"]) is None,
            "M40 synthetic scheduler oracle drift")
    missing = list(contract["required_next_trace"]["missing_fields"])
    require(missing and not m22["has_product_event_coordinates_and_operands"],
            "M40 fail-closed missing trace gate drift")
    admission = {
        "exact_h67_ep35_s10_source_support_sign_trace_admitted": True,
        "exact_40_record_two_code_amplitude_trace_admitted": True,
        "exact_92160000_value_bitmap_codebook_bit_miter_admitted": True,
        "exact_four_layer_m35_uq0p24_amplitude_mapping_admitted": True,
        "exact_bias_free_real_algebra_amplitude_factorization_admitted": True,
        "exact_one_bit_plus_static_amplitude_32x_dense_representation_admitted": True,
        "actual_structured_local_motion_support_distribution_admitted": True,
        "padding_valid_product_count_expansion_admitted": True,
        "small_trace_conflict_scheduler_reference_admitted": True,
        "small_trace_exact_event_conservation_admitted": True,
        "small_trace_m35_integer_miter_admitted": True,
    }
    admission.update({key: False for key in FORBIDDEN})
    return {
        "schema": "m40_conflict_aware_event_schedule_audit_v1",
        "status": (
            "PASS_M40A_EXACT_AMPLITUDE_CODEBOOK_SOURCE_TRACE_BUT_REAL_PRODUCT_"
            "SCHEDULE_BLOCKED_ON_WEIGHT_QUANTIZATION_PHYSICAL_LAYOUT_AND_ACCUMULATORS"),
        "identity": {
            "contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "packed_source_manifest_sha256":
                contract["inputs"]["packed_source_manifest"]["sha256"],
        },
        "upstream": upstream,
        "m22_m23_trace_audit": m22,
        "real_source_trace": {
            "records": rows,
            "noninteger_value_population": noninteger,
            "numeric_values_reconstructable_from_bitmap_plus_layer_amplitude": True,
            "amplitude_codebook_m35_reconciliation": amplitude_by_operator,
            "float32_values_bit_exact_mitered": total_values_mitered,
            "float32_value_bit_mismatches": 0,
            "amplitude_codebook_status": (
                "PASS_40_OF_40_EXACT_TWO_CODE_ZERO_PLUS_LAYER_STATIC_M35_UQ0P24_"
                "THRESHOLD_ALL_92160000_VALUES_BIT_EXACT"),
            "amplitude_carry_architecture_candidate": {
                "name": "LAYER_STATIC_AMPLITUDE_CARRY_EVENT_ACCUMULATION",
                "exact_identity": (
                    "bias-free sum_i((bitmap_i*theta_layer)*weight_i) = "
                    "theta_layer*sum_i(bitmap_i*weight_i) over exact arithmetic"),
                "all_four_convolutions_bias_free": all(
                    not record["module_geometry"]["bias_present"]
                    for record in trace["records"]),
                "dense_float32_activation_bytes": total_values_mitered * 4,
                "one_bit_activity_bytes": (total_values_mitered + 7) // 8,
                "dense_representation_reduction_exact": {
                    "numerator": 32, "denominator": 1},
                "layer_static_uq0p24_amplitude_words": 4,
                "event_activation_multiplier_required": False,
                "one_m35_complement_csd_scale_per_output_required": True,
                "integer_weight_quantization_and_float_output_equivalence_admitted": False,
                "integrated_rtl_admitted": False,
            },
            "structured_trace_not_uniform_density_sweep": True,
            "exact_work_lower_bound_distribution_by_line": {
                line: distribution(per_sample[line]) for line in ("Local", "Motion")
            },
            "cycle_metric_qualification": (
                "EXACT_96_LANE_PRODUCT_COUNT_LOWER_BOUND_ONLY_NOT_EXECUTABLE_"
                "SCHEDULE_MEAN_P95_P99"),
            "executable_cycle_mean_p95_p99": {
                "Local": None, "Motion": None,
            },
        },
        "executable_small_trace_reference": {
            "scheduler_contract": scheduler_contract,
            "coalescing_key": contract["coalescing_and_flush"]["coalescing_key"],
            "flush_boundary": contract["coalescing_and_flush"]["flush_boundary"],
            "address_mapping": contract["address_mapping"],
            "synthetic_result": synthetic_result,
            "qualification": "UNIT_ORACLE_ONLY_NOT_REAL_H67_SCHEDULE",
        },
        "m39_p2_disposition": {
            "bank_service": "EXECUTABLE_UNIT_MODEL_ONLY_REAL_MACRO_AND_TRACE_BLOCKED",
            "uniform_density": "CLOSED_FOR_SOURCE_SUPPORT_BY_REAL_S10_BITMAP_REPLAY",
            "m35_general_independent_admission": "OPEN_REAL_MITER_BLOCKED",
            "compulsory_bytes_physical_boundary": "OPEN_REAL_LAYOUT_AND_TILING_BLOCKED",
        },
        "required_next_trace": contract["required_next_trace"],
        "admission": admission,
        "claim_boundary": contract["claim_boundary"],
    }


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite M40 result")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_output(args.output, build(args.contract.resolve()))
    print(args.output)


if __name__ == "__main__":
    main()
