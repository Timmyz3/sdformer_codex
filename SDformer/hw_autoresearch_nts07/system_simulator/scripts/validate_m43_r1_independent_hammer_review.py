#!/usr/bin/env python3
"""Independent fail-closed review validator for M43-r1.

The core replay deliberately does not import or call the M43 candidate analyzer.
It reconstructs valid-pad Conv3x3 supports from the 40 frozen packed bitmaps,
selects spatial and spatiotemporal parents, and schedules each signed symmetric
difference through eight one-source-per-cycle banks and eight 96-lane blocks.
"""

from __future__ import print_function

import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
REVIEW = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_r1_independent_hammer_review.json")

ANCHORS = {
    "contract": (
        "contracts/m43_tile_resident_parent_delta_schedule_contract_r1_20260823.json",
        "c894b5fcdd6a6cd7d33bf736e8c084630f0ea297f632e1dd6a35889714772e44"),
    "analyzer": (
        "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py",
        "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3"),
    "candidate_regression": (
        "system_simulator/tests/test_m43_tile_resident_parent_delta_schedule.py",
        "c189935f3365beaa657eaa21ca7c40f275523a974ad12611e11f6f84331f197f"),
    "candidate_validator": (
        "system_simulator/scripts/validate_m43_tile_resident_parent_delta_schedule.py",
        "ae0cfedc20828cfc40c115844464f023e2413a2e8cb2528b594a6861f2eb3fca"),
    "specification": (
        "rtl_m43/M43_TILE_RESIDENT_PARENT_DELTA_R1.md",
        "5c2b53b7eb0ec4ca19559e65c0d454109a598c80d1647564b9f894e573dd13a6"),
    "spatial_result": (
        "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
        "m43_spatial_parent_delta_schedule_final.json",
        "70c52dfc8ef1b223391a1c0699f6ada8ff999d2079370bcd9d3917c198a1c329"),
    "temporal_ablation": (
        "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
        "m43_spatiotemporal_parent_delta_ablation.json",
        "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c"),
    "m40_manifest": (
        "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
        "m40_bottleneck_packed_source_manifest.json",
        "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3"),
    "m40_result": (
        "results/m40_conflict_aware_event_schedule_r3_20260822/"
        "m40_conflict_aware_event_schedule.json",
        "419ea51faabda4c2f45b9fa535d1a0fa8142bb4c8b8258468e88a1dc99c310e7"),
    "m40_review": (
        "results/m40_conflict_aware_event_schedule_r3_20260822/"
        "m40a_r3_independent_hammer_review.json",
        "b562d2b77ed5b3acb04ae6688c96f033b69c16faa3b73e984f8c29380c417abf"),
    "m41_result": (
        "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/"
        "m41_h67_ep35_bottleneck_int8_bridge.json",
        "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb"),
    "m41_review": (
        "results/m41_h67_ep35_bottleneck_int8_independent_hammer_review_r1_20260823/"
        "m41_independent_hammer_review.json",
        "81817dd55c90c2c05eb3579030c16f1fc60fe87c446dd38067392ab5b2d52552"),
    "m42_result": (
        "results/m42_real_work_headroom_gate_r1_20260823/"
        "m42_real_work_headroom_gate.json",
        "c0677ce56775996481ba500fc397191e7de407768f29c591ae731c69ed45cd13"),
    "m42_review": (
        "results/m42_real_work_headroom_gate_r1_20260823/"
        "m42_r1_independent_hammer_review.json",
        "de7a6187b5a4a693023948045ae27480051713192564b74cf66055648cbc0d02"),
    "p8_l96_engine_rtl": (
        "rtl_qfit/qfit_local_banked_multisource_engine.sv",
        "4003637653110fe2407b646a9f82ca4b77d775e01c1151c3c4ce0a8c47c0b3dc"),
    "p8_l96_dc_top_rtl": (
        "rtl_qfit/qfit_local_banked_multisource_l96_dc_tops.sv",
        "9656d79a87ce8057cd8f3926bb2f57f91fa241485a11cb59a9b7d3712ab0a019"),
}

T = 10
C = 768
H = 15
W = 20
K = 3
FEATURES = C * K * K
TILE_BITS = 256
TILES = FEATURES // TILE_BITS
BANKS = 8
LANES = 96
BLOCKS = C // LANES
ROWS = T * H * W
PARENT_NAMES = ("local_zero", "left", "up", "previous_timestep")
BANK_MASKS = tuple(sum(1 << bit for bit in range(bank, TILE_BITS, BANKS))
                   for bank in range(BANKS))
POPCOUNT_BYTE = tuple(bin(value).count("1") for value in range(256))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_int(value, label):
    require(type(value) is int, "{} is not an exact JSON integer".format(label))
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(raw):
    raise ValueError("non-standard JSON constant: {}".format(raw))


def read_json(path):
    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs_hook,
                         parse_constant=reject_constant)


def contained_file(base, raw, label):
    base = Path(base).resolve()
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError("{} escapes its payload directory".format(label))
    require(candidate.is_file() and not candidate.is_symlink(),
            "{} is not a regular non-symlink file".format(label))
    return candidate


def validate_anchors():
    observed = {}
    for name, pair in sorted(ANCHORS.items()):
        path = HW_ROOT / pair[0]
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == pair[1], "anchor drift: {}".format(name))
        observed[name] = pair[1]
    observed["independent_validator"] = sha256(Path(__file__).resolve())
    return observed


def popcount(value):
    return bin(value).count("1")


def bank_cycles(mask):
    return max(popcount(mask & bank_mask) for bank_mask in BANK_MASKS)


def nearest_rank(values, numerator):
    ordered = sorted(values)
    rank = (numerator * len(ordered) + 99) // 100
    return ordered[rank - 1]


def validate_record_descriptor(record):
    require(record["shape"] == [T, 1, C, H, W] and
            record["output_shape"] == [T, 1, C, H, W] and
            exact_int(record["elements"], "record elements") == T * C * H * W,
            "record tensor geometry drift")
    geometry = record["module_geometry"]
    require(geometry["kernel_size"] == [3, 3] and
            geometry["stride"] == [1, 1] and
            geometry["padding"] == [1, 1] and
            geometry["dilation"] == [1, 1] and
            exact_int(geometry["groups"], "groups") == 1 and
            exact_int(geometry["in_channels"], "in channels") == C and
            exact_int(geometry["out_channels"], "out channels") == C and
            geometry["bias_present"] is False,
            "record Conv3x3 geometry drift")
    require(record["packing"] ==
            "C_ORDER_FLAT_NP_PACKBITS_LITTLE_POSITIVE_THEN_NEGATIVE_THEN_EXACT_FLOAT_VALUE_CHANGED_VS_PREVIOUS_T_WITH_T0_ZERO",
            "packed-bit order drift")


def reconstruct_masks(trace_dir, record):
    validate_record_descriptor(record)
    payload = contained_file(trace_dir, record["packed_file"], "packed source")
    require(sha256(payload) == record["packed_file_sha256"],
            "packed source SHA drift")
    raw = payload.read_bytes()
    plane_bytes = exact_int(record["positive_plane_bytes"], "positive bytes")
    require(plane_bytes == T * C * H * W // 8 and
            len(raw) == 3 * plane_bytes == record["packed_file_bytes"],
            "packed source extent drift")
    positive = raw[:plane_bytes]
    negative = raw[plane_bytes:2 * plane_bytes]
    changed = raw[2 * plane_bytes:]
    require(not any(negative) and record["negative_count"] == 0,
            "M43 source trace is not frozen nonnegative activity")
    bytes_per_t = plane_bytes // T
    positive_count = 0
    changed_bytes_checked = 0
    for timestep in range(T):
        start = timestep * bytes_per_t
        previous = (timestep - 1) * bytes_per_t
        count_t = 0
        for offset, current in enumerate(positive[start:start + bytes_per_t]):
            prior = 0 if timestep == 0 else positive[previous + offset]
            require(changed[start + offset] == (current ^ prior),
                    "changed plane differs from adjacent support XOR")
            count_t += POPCOUNT_BYTE[current]
            changed_bytes_checked += 1
        require(count_t == record["local_nonzero_count_by_timestep"][timestep],
                "per-timestep positive population drift")
        positive_count += count_t
    require(positive_count == record["positive_count"] == record["nonzero_count"],
            "record positive population drift")

    masks = [0] * (ROWS * TILES)
    total_bits = T * C * H * W
    for byte_index, raw_byte in enumerate(positive):
        value = raw_byte
        while value:
            least = value & -value
            bit_index = least.bit_length() - 1
            flat = byte_index * 8 + bit_index
            require(flat < total_bits, "nonzero tail bit")
            tc, spatial = divmod(flat, H * W)
            timestep, channel = divmod(tc, C)
            input_y, input_x = divmod(spatial, W)
            for kernel_y in range(K):
                output_y = input_y + 1 - kernel_y
                if output_y < 0 or output_y >= H:
                    continue
                for kernel_x in range(K):
                    output_x = input_x + 1 - kernel_x
                    if output_x < 0 or output_x >= W:
                        continue
                    feature = channel * 9 + kernel_y * 3 + kernel_x
                    tile, tile_bit = divmod(feature, TILE_BITS)
                    output_row = ((timestep * H + output_y) * W + output_x)
                    masks[output_row * TILES + tile] |= 1 << tile_bit
            value ^= least
    return masks, changed_bytes_checked


def parent_candidates(masks, row, tile):
    index = row * TILES + tile
    current = masks[index]
    timestep_spatial, output_x = divmod(row, W)
    timestep, output_y = divmod(timestep_spatial, H)
    candidates = [("local_zero", 0)]
    if output_x > 0:
        candidates.append(("left", masks[index - TILES]))
    if output_y > 0:
        candidates.append(("up", masks[index - W * TILES]))
    if timestep > 0:
        candidates.append(("previous_timestep",
                           masks[index - H * W * TILES]))
    return current, candidates


def choose_parent(current, candidates):
    ranked = []
    for priority, item in enumerate(candidates):
        name, parent = item
        delta = current ^ parent
        ranked.append((bank_cycles(delta), popcount(delta), priority,
                       name, parent, delta))
    winner = min(ranked)
    return winner[3], winner[4], winner[5], winner[0]


def blank_mode():
    return {
        "pairs": 0, "add_pairs": 0, "subtract_pairs": 0,
        "cycles": 0, "commands": 0, "copies": 0,
        "parents": dict((name, 0) for name in PARENT_NAMES),
    }


def audit_record(trace_dir, record, expected_local_pairs):
    masks, changed_bytes = reconstruct_masks(trace_dir, record)
    local_pairs = 0
    local_cycles = 0
    local_commands = 0
    spatial = blank_mode()
    temporal = blank_mode()
    for row in range(ROWS):
        for tile in range(TILES):
            current, candidates = parent_candidates(masks, row, tile)
            local_count = popcount(current)
            local_pairs += local_count
            local_cycles += bank_cycles(current) * BLOCKS
            if current:
                local_commands += BLOCKS
            spatial_candidates = [item for item in candidates
                                  if item[0] != "previous_timestep"]
            for mode, candidate_set in ((spatial, spatial_candidates),
                                        (temporal, candidates)):
                name, parent, delta, cycles = choose_parent(current, candidate_set)
                additions = current & ~parent
                subtractions = parent & ~current
                require((additions & subtractions) == 0 and
                        (additions | subtractions) == delta,
                        "signed symmetric difference is not conserved")
                add_count = popcount(additions)
                subtract_count = popcount(subtractions)
                count = popcount(delta)
                require(add_count + subtract_count == count,
                        "signed delta population mismatch")
                mode["pairs"] += count
                mode["add_pairs"] += add_count
                mode["subtract_pairs"] += subtract_count
                mode["cycles"] += cycles * BLOCKS
                mode["commands"] += BLOCKS if delta else 0
                mode["copies"] += 1 if not delta and name != "local_zero" else 0
                mode["parents"][name] += 1
    require(local_pairs == expected_local_pairs,
            "independent valid-pad Local pair reconstruction drift")
    require(sum(spatial["parents"].values()) == ROWS * TILES and
            sum(temporal["parents"].values()) == ROWS * TILES,
            "parent tile population drift")
    return {
        "sample_id": record["sample_id"],
        "operator": record["operator"],
        "local_pairs": local_pairs,
        "local_cycles": local_cycles,
        "local_commands": local_commands,
        "changed_plane_bytes_checked": changed_bytes,
        "spatial": spatial,
        "temporal": temporal,
    }


def validate_m41_weights():
    result_path = HW_ROOT / ANCHORS["m41_result"][0]
    result = read_json(result_path)
    review = read_json(HW_ROOT / ANCHORS["m41_review"][0])
    require(result["schema"] == "m41_h67_ep35_bottleneck_int8_bridge_result_v1" and
            review["status"].startswith("GO_CHECKPOINT_BOUND_MODEL_BRIDGE"),
            "M41 model bridge is not independently admitted")
    require(result["quantization_contract"]["weight_payload_layout"] ==
            "I_KY_KX_O, C-order little-endian signed int8; O is contiguous and divisible into eight 96-lane vectors",
            "M41 weight layout drift")
    weights = [item for item in result["payloads"] if item.get("role") == "weight"]
    require(len(weights) == 4, "M41 weight population drift")
    total = 0
    for index, item in enumerate(weights):
        require(item["shape"] == [C, 3, 3, C] and
                item["layout"] == "I_KY_KX_O_C_ORDER" and
                item["dtype"] == "signed_int8",
                "M41 weight descriptor drift")
        payload = contained_file(result_path.parent, item["file"],
                                 "M41 weight {}".format(index))
        size = exact_int(item["bytes"], "weight bytes")
        require(payload.stat().st_size == size and
                sha256(payload) == item["sha256"],
                "M41 weight payload identity drift")
        total += size
    require(total == 21233664, "M41 total INT8 weight bytes drift")
    address_checks = 0
    for feature in range(FEATURES):
        bank = feature % BANKS
        bank_row = feature // BANKS
        require(bank + BANKS * bank_row == feature,
                "source-bank address inversion drift")
        for output_block in range(BLOCKS):
            first = feature * C + output_block * LANES
            last = first + LANES - 1
            require(last < (feature + 1) * C,
                    "96-lane vector is not output-contiguous")
            address_checks += 1
    return {"weight_payloads": 4, "weight_bytes": total,
            "weight_address_checks": address_checks,
            "layout": "I_KY_KX_O_C_ORDER",
            "source_bank": "global_feature_index_mod_8"}


def independent_rebuild():
    manifest_path = HW_ROOT / ANCHORS["m40_manifest"][0]
    manifest = read_json(manifest_path)
    m40 = read_json(HW_ROOT / ANCHORS["m40_result"][0])
    require(manifest["schema"] == "m40_bottleneck_packed_source_trace_v1" and
            manifest["cohort"]["records"] == 40 and
            manifest["cohort"]["samples"] == 10,
            "M40 manifest cohort drift")
    expected = dict(((row["sample_id"], row["operator"]),
                     row["Local"]["source_destination_pairs"])
                    for row in m40["real_source_trace"]["records"])
    require(len(expected) == 40, "M40 result record population drift")
    rows = []
    for index, record in enumerate(manifest["records"]):
        key = (record["sample_id"], record["operator"])
        require(key in expected, "M40 sample/operator identity drift")
        rows.append(audit_record(manifest_path.parent, record, expected[key]))
        print("[M43 independent] {}/40 sample={} operator={}".format(
            index + 1, record["sample_id"], record["operator"]))

    modes = {}
    for mode_name in ("spatial", "temporal"):
        samples = []
        for sample_id in range(10):
            selected = [row for row in rows if row["sample_id"] == sample_id]
            require(len(selected) == 4, "per-sample operator population drift")
            samples.append({
                "sample_id": sample_id,
                "local_pairs": sum(row["local_pairs"] for row in selected),
                "delta_pairs": sum(row[mode_name]["pairs"] for row in selected),
                "local_cycles": sum(row["local_cycles"] for row in selected),
                "delta_cycles": sum(row[mode_name]["cycles"] for row in selected),
                "delta_commands": sum(row[mode_name]["commands"] for row in selected),
                "nonlocal_tile_block_reads": sum(
                    sum(count for name, count in row[mode_name]["parents"].items()
                        if name != "local_zero") * BLOCKS for row in selected),
            })
        local_pairs = sum(row["local_pairs"] for row in rows)
        delta_pairs = sum(row[mode_name]["pairs"] for row in rows)
        local_cycles = sum(row["local_cycles"] for row in rows)
        delta_cycles = sum(row[mode_name]["cycles"] for row in rows)
        parent_counts = dict((name, sum(row[mode_name]["parents"][name]
                                       for row in rows))
                             for name in PARENT_NAMES)
        modes[mode_name] = {
            "local_pairs": local_pairs,
            "delta_pairs": delta_pairs,
            "delta_add_pairs": sum(row[mode_name]["add_pairs"] for row in rows),
            "delta_subtract_pairs": sum(row[mode_name]["subtract_pairs"] for row in rows),
            "local_cycles": local_cycles,
            "delta_cycles": delta_cycles,
            "logical_reduction": {
                "numerator": local_pairs - delta_pairs,
                "denominator": local_pairs,
            },
            "bank_cycle_reduction": {
                "numerator": local_cycles - delta_cycles,
                "denominator": local_cycles,
            },
            "parent_counts": parent_counts,
            "per_sample": samples,
            "local_cycle_mean": {"numerator": local_cycles, "denominator": 10},
            "local_cycle_p95": nearest_rank(
                [item["local_cycles"] for item in samples], 95),
            "delta_cycle_mean": {"numerator": delta_cycles, "denominator": 10},
            "delta_cycle_p95": nearest_rank(
                [item["delta_cycles"] for item in samples], 95),
        }
    require(sum(row["changed_plane_bytes_checked"] for row in rows) == 11520000,
            "packed changed-plane byte coverage drift")
    require(modes["spatial"]["local_pairs"] == modes["temporal"]["local_pairs"] and
            modes["spatial"]["local_cycles"] == modes["temporal"]["local_cycles"],
            "spatial/temporal Local baseline drift")
    return {
        "records": 40,
        "samples": 10,
        "operators": 4,
        "output_rows": 120000,
        "tile_rows": 3240000,
        "changed_plane_bytes_checked": 11520000,
        "spatial": modes["spatial"],
        "temporal_ablation": modes["temporal"],
        "records_detail": rows,
    }


def exact_fraction(value, numerator, denominator, label):
    require(type(value) is dict and set(value) == {"numerator", "denominator"} and
            exact_int(value["numerator"], label + " numerator") == numerator and
            exact_int(value["denominator"], label + " denominator") == denominator,
            "{} drift".format(label))


def validate_candidate_result(result, replay, temporal=False):
    mode = replay["temporal_ablation" if temporal else "spatial"]
    require(result["schema"] == "m43_tile_resident_parent_delta_schedule_v1",
            "M43 result schema drift")
    architecture = result["architecture"]
    expected_name = ("TILE_RESIDENT_SIGNED_SPATIOTEMPORAL_PARENT_DELTA" if temporal
                     else "TILE_RESIDENT_SIGNED_SPATIAL_PARENT_DELTA")
    expected_parents = list(PARENT_NAMES if temporal else PARENT_NAMES[:3])
    require(architecture["name"] == expected_name and
            architecture["parents"] == expected_parents and
            architecture["temporal_parent_enabled"] is temporal and
            architecture["tile_partial_identity"] ==
            "Y_tile(S)=Y_tile(P)+sum(S\\P)W-sum(P\\S)W; sum over all 27 feature tiles recovers the quantized convolution" and
            architecture["signed_delta"] ==
            "current_minus_parent; additions and subtractions share bank capacity" and
            exact_int(architecture["issue_width"], "issue width") == BANKS and
            exact_int(architecture["output_lanes"], "output lanes") == LANES and
            exact_int(architecture["output_blocks"], "output blocks") == BLOCKS and
            exact_int(architecture["peak_product_adds_per_cycle"], "peak products") == 768,
            "M43 architecture or parent semantics drift")
    require("source row" not in architecture["tile_partial_identity"].lower(),
            "parent was incorrectly promoted to a source row")
    aggregate = result["aggregate"]
    require(exact_int(aggregate["local_source_destination_pairs"], "local pairs") ==
            mode["local_pairs"] and
            exact_int(aggregate["parent_delta_source_destination_pairs"], "delta pairs") ==
            mode["delta_pairs"] and
            exact_int(aggregate["local_p8_l96_source_issue_cycles"], "local cycles") ==
            mode["local_cycles"] and
            exact_int(aggregate["parent_delta_p8_l96_source_issue_cycles"], "delta cycles") ==
            mode["delta_cycles"], "M43 aggregate replay drift")
    exact_fraction(aggregate["logical_pair_reduction"],
                   mode["logical_reduction"]["numerator"], mode["local_pairs"],
                   "logical pair reduction")
    exact_fraction(aggregate["finite_bank_issue_cycle_reduction"],
                   mode["bank_cycle_reduction"]["numerator"], mode["local_cycles"],
                   "bank cycle reduction")
    require(aggregate["parent_choice_by_tile"] == mode["parent_counts"],
            "M43 aggregate parent population drift")
    candidate_samples = result["per_sample"]
    require(len(candidate_samples) == 10, "M43 sample population drift")
    all_commands = 4 * ROWS * TILES * BLOCKS
    final_cycles = 4 * ROWS * (TILES + TILES - 1) * BLOCKS
    selector_cycles = 4 * ROWS * TILES * (2 if temporal else 1)
    weight_cycles = 21233664 // 32
    for expected, actual in zip(mode["per_sample"], candidate_samples):
        require(exact_int(actual["sample_id"], "sample id") == expected["sample_id"] and
                exact_int(actual["local_pairs"], "sample local pairs") == expected["local_pairs"] and
                exact_int(actual["delta_pairs"], "sample delta pairs") == expected["delta_pairs"] and
                exact_int(actual["local_issue_cycles"], "sample local cycles") == expected["local_cycles"] and
                exact_int(actual["delta_issue_cycles"], "sample delta cycles") == expected["delta_cycles"] and
                exact_int(actual["delta_commands"], "sample delta commands") == expected["delta_commands"],
                "M43 per-sample replay drift")
        parent_service = all_commands + expected["nonlocal_tile_block_reads"]
        expected_services = {
            "finite_bank_source_issue": expected["delta_cycles"],
            "descriptor_enqueue_one_per_cycle": all_commands,
            "parent_selector_support_single_port": selector_cycles,
            "parent_partial_single_port": parent_service,
            "final_accumulator_single_port": final_cycles,
            "int8_weight_load_256b": weight_cycles,
        }
        services = actual["independent_service_capacity_cycles"]
        require(type(services) is dict and set(services) == set(expected_services),
                "independent service key population drift")
        require(all(type(services[name]) is int and
                    services[name] == expected_services[name]
                    for name in expected_services),
                "independent service capacity drift")
        require(exact_int(actual["independent_service_capacity_max"],
                          "capacity maximum") == max(expected_services.values()) ==
                expected["delta_cycles"] and
                actual["source_issue_is_capacity_max"] is True and
                actual["three_x_crossing_admitted"] is False,
                "M43 independent-capacity claim boundary drift")
    layout = result["physical_layout_bridge"]
    expected_scratch = 146560 if temporal else 56960
    expected_parent = 86400 if temporal else 5760
    expected_support = 9600 if temporal else 640
    require(exact_int(layout["weight_bytes_per_sample"], "weight bytes") == 21233664 and
            layout["int8_weight_layout"].startswith("I_KY_KX_O") and
            layout["weight_linear_address"] ==
            "operator_base + (((cin*3+ky)*3+kx)*768+cout)" and
            exact_int(layout["weight_double_buffer_bytes"], "weight buffer") == 49152 and
            exact_int(layout["parent_partial_buffer_bytes"], "parent buffer") == expected_parent and
            exact_int(layout["support_parent_buffer_bytes"], "support buffer") == expected_support and
            exact_int(layout["four_context_bytes"], "context bytes") == 1408 and
            exact_int(layout["local_scratch_bytes"], "scratch bytes") == expected_scratch and
            layout["local_scratch_fits"] is True and
            exact_int(layout["one_output_block_all_timestep_final_accumulator_bytes"],
                      "global accumulator bytes") == 864000,
            "M43 physical-layout byte bridge drift")
    gate = result["three_x_headroom_gate"]
    require(exact_int(gate["m42_maximum_product_cycles"], "3x ceiling") == 15495075 and
            exact_int(gate["p95_parent_delta_source_issue_cycles"], "p95 issue") ==
            mode["delta_cycle_p95"] and
            gate["all_samples_source_issue_below_gate"] is True and
            gate["all_samples_independent_service_capacity_below_source_issue"] is True and
            gate["target_crossing_admitted"] is False,
            "M43 3x gate boundary drift")
    require(any("independent-service max" in item
                for item in result["claim_policy"]["forbidden"]) and
            any("3x crossing" in item
                for item in result["claim_policy"]["forbidden"]),
            "M43 independent-service/system claim ban drift")


def capacity_and_target_gates(replay, weights):
    spatial = replay["spatial"]
    m42 = read_json(HW_ROOT / ANCHORS["m42_result"][0])
    review = read_json(HW_ROOT / ANCHORS["m42_review"][0])
    require(review["review"]["decision"] == "GO_EXACT_HEADROOM_GATE_ONLY",
            "M42 review is not exact-headroom GO")
    model = m42["frozen_resource_model"]
    fixed = exact_int(model["fixed_late_scale_plus_frontend_cycles"], "fixed cycles")
    outside = exact_int(model["outside_four_bottleneck_model_cycles"], "outside cycles")
    reference = exact_int(model["fixed_compute_reference_cycles"], "reference cycles")
    require((fixed, outside, reference) == (2636515, 188824491, 620868243),
            "M42 frozen budget drift")
    three_x_ceiling = reference // 3 - outside - fixed
    require(three_x_ceiling == 15495075, "independent M42 3x ceiling drift")
    all_commands = 4 * ROWS * TILES * BLOCKS
    final_cycles = 4 * ROWS * (TILES + TILES - 1) * BLOCKS
    weight_cycles = weights["weight_bytes"] // 32
    service_rows = []
    for sample in spatial["per_sample"]:
        services = {
            "finite_bank_source_issue": sample["delta_cycles"],
            "descriptor_enqueue_one_per_cycle": all_commands,
            "parent_selector_support_single_port": 4 * ROWS * TILES,
            "parent_partial_single_port":
                all_commands + sample["nonlocal_tile_block_reads"],
            "final_accumulator_single_port": final_cycles,
            "int8_weight_load_256b": weight_cycles,
        }
        require(max(services.values()) == sample["delta_cycles"],
                "source issue is not the independent service maximum")
        service_rows.append({
            "sample_id": sample["sample_id"],
            "services": services,
            "capacity_max": max(services.values()),
            "capacity_max_is_not_integrated_cycles": True,
        })
    p95 = spatial["delta_cycle_p95"]
    return {
        "fixed_compute_reference_cycles": reference,
        "outside_four_bottleneck_model_cycles": outside,
        "fixed_late_scale_plus_frontend_cycles": fixed,
        "three_x_maximum_product_cycles": three_x_ceiling,
        "p95_parent_delta_source_issue_cycles": p95,
        "p95_product_headroom_cycles": three_x_ceiling - p95,
        "all_samples_source_issue_below_product_gate": all(
            row["capacity_max"] <= three_x_ceiling for row in service_rows),
        "per_sample_independent_capacity_max": [
            row["capacity_max"] for row in service_rows],
        "per_sample_parent_partial_service_cycles": [
            row["services"]["parent_partial_single_port"]
            for row in service_rows],
        "static_independent_service_cycles": {
            "descriptor_enqueue_one_per_cycle": all_commands,
            "parent_selector_support_single_port": 4 * ROWS * TILES,
            "final_accumulator_single_port": final_cycles,
            "int8_weight_load_256b": weight_cycles,
        },
        "source_issue_is_capacity_max_for_all_samples": True,
        "capacity_max_is_not_integrated_cycles": True,
        "conditional_p95_compute_speedup": {
            "numerator": reference,
            "denominator": outside + fixed + p95,
        },
        "target_crossing_admitted": False,
        "integrated_executable_cycles_admitted": False,
    }


def adversarial_checks(canonical, replay):
    rejected = []
    with tempfile.TemporaryDirectory() as tempdir:
        duplicate = Path(tempdir) / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        nan = Path(tempdir) / "nan.json"
        nan.write_text('{"a":NaN}', encoding="utf-8")
        for name, path in (("duplicate_json_key", duplicate),
                           ("nan_json_constant", nan)):
            try:
                read_json(path)
            except ValueError:
                rejected.append(name)
            else:
                raise ValueError("attack accepted: {}".format(name))

        changed = Path(tempdir) / "changed_anchor.json"
        changed.write_bytes((HW_ROOT / ANCHORS["contract"][0]).read_bytes() + b" ")
        require(sha256(changed) != ANCHORS["contract"][1],
                "contract SHA mutation unexpectedly survived")
        rejected.append("contract_sha_mutation")
        changed.write_bytes((HW_ROOT / ANCHORS["m40_manifest"][0]).read_bytes() + b" ")
        require(sha256(changed) != ANCHORS["m40_manifest"][1],
                "trace SHA mutation unexpectedly survived")
        rejected.append("trace_manifest_sha_mutation")
        changed.write_bytes((HW_ROOT / ANCHORS["spatial_result"][0]).read_bytes() + b" ")
        require(sha256(changed) != ANCHORS["spatial_result"][1],
                "result SHA mutation unexpectedly survived")
        rejected.append("canonical_result_sha_mutation")

    attacks = []
    forged = copy.deepcopy(canonical)
    forged["aggregate"]["local_source_destination_pairs"] = True
    attacks.append(("bool_as_integer", forged))
    forged = copy.deepcopy(canonical)
    forged["architecture"]["tile_partial_identity"] = "parent is an input source row"
    attacks.append(("parent_miscast_as_source_row", forged))
    forged = copy.deepcopy(canonical)
    forged["three_x_headroom_gate"]["target_crossing_admitted"] = True
    attacks.append(("capacity_max_promoted_to_3x_crossing", forged))
    forged = copy.deepcopy(canonical)
    forged["aggregate"]["parent_delta_p8_l96_source_issue_cycles"] -= 1
    attacks.append(("bank_cycle_arithmetic_mutation", forged))
    forged = copy.deepcopy(canonical)
    forged["per_sample"][0]["independent_service_capacity_max"] = 5088000
    attacks.append(("independent_service_max_mutation", forged))
    for name, forged in attacks:
        try:
            validate_candidate_result(forged, replay, temporal=False)
        except (ValueError, TypeError):
            rejected.append(name)
        else:
            raise ValueError("attack accepted: {}".format(name))
    return {"tested": len(rejected), "rejected": len(rejected),
            "rejected_attacks": rejected}


def expected_review_core(replay, weights, gates, attacks, anchors):
    spatial = replay["spatial"]
    temporal = replay["temporal_ablation"]
    return {
        "anchors": anchors,
        "candidate_regression": {
            "python36_tests_passed": 8,
            "python36_test_failures": 0,
            "canonical_validator_passed": True,
            "full_rebuild_byte_identical": True,
            "canonical_result_sha256": ANCHORS["spatial_result"][1],
        },
        "independent_reconstruction": {
            "records": replay["records"],
            "samples": replay["samples"],
            "operators": replay["operators"],
            "output_rows": replay["output_rows"],
            "tile_rows": replay["tile_rows"],
            "changed_plane_bytes_checked": replay["changed_plane_bytes_checked"],
            "local_source_destination_pairs": spatial["local_pairs"],
            "parent_delta_source_destination_pairs": spatial["delta_pairs"],
            "parent_delta_add_pairs": spatial["delta_add_pairs"],
            "parent_delta_subtract_pairs": spatial["delta_subtract_pairs"],
            "local_p8_l96_source_issue_cycles": spatial["local_cycles"],
            "parent_delta_p8_l96_source_issue_cycles": spatial["delta_cycles"],
            "logical_pair_reduction": spatial["logical_reduction"],
            "finite_bank_issue_cycle_reduction": spatial["bank_cycle_reduction"],
            "parent_choice_by_tile": spatial["parent_counts"],
            "per_sample_local_issue_cycles": [
                item["local_cycles"] for item in spatial["per_sample"]],
            "per_sample_parent_delta_issue_cycles": [
                item["delta_cycles"] for item in spatial["per_sample"]],
            "local_issue_cycle_mean_exact": spatial["local_cycle_mean"],
            "local_issue_cycle_p95_nearest_rank": spatial["local_cycle_p95"],
            "parent_delta_issue_cycle_mean_exact": spatial["delta_cycle_mean"],
            "parent_delta_issue_cycle_p95_nearest_rank": spatial["delta_cycle_p95"],
            "bank_mapping": "global_feature_index_mod_8",
            "output_blocks_per_source_tile": BLOCKS,
            "peak_signed_int8_additions_per_cycle": BANKS * LANES,
        },
        "weight_and_storage_bridge": {
            "weight_payload_audit": weights,
            "weight_double_buffer_bytes": 49152,
            "parent_partial_line_bytes": 5760,
            "support_line_bytes": 640,
            "four_context_bytes": 1408,
            "local_scratch_bytes": 56960,
            "frozen_local_residency_bytes": 193728,
            "one_output_block_all_timestep_final_accumulator_bytes": 864000,
            "macro_timing_or_energy_admitted": False,
        },
        "capacity_and_target_gates": gates,
        "spatiotemporal_ablation": {
            "qualification": "ABLATION_ONLY_NOT_PRIMARY_ARCHITECTURE",
            "delta_pairs": temporal["delta_pairs"],
            "delta_issue_cycles": temporal["delta_cycles"],
            "parent_choice_by_tile": temporal["parent_counts"],
            "incremental_pair_reduction_vs_spatial": {
                "numerator": spatial["delta_pairs"] - temporal["delta_pairs"],
                "denominator": spatial["delta_pairs"],
            },
            "incremental_issue_cycle_reduction_vs_spatial": {
                "numerator": spatial["delta_cycles"] - temporal["delta_cycles"],
                "denominator": spatial["delta_cycles"],
            },
            "local_scratch_bytes": 146560,
            "extra_local_scratch_bytes_vs_spatial": 89600,
        },
        "adversarial_matrix": attacks,
    }


def validate_review(review, core):
    require(review["schema"] == "m43_r1_independent_hammer_review_v1" and
            review["status"] == "GO_EXACT_SOURCE_BANK_SCHEDULE_AND_CAPACITY_GATES_ONLY" and
            review["candidate_modified_by_reviewer"] is False,
            "M43 independent review identity/status drift")
    require(review["review"] == {
        "decision": "GO_EXACT_SOURCE_BANK_SCHEDULE_AND_CAPACITY_GATES_ONLY",
        "score_0_to_100": 94,
        "p0": 0,
        "p1": 0,
        "p2": 5,
    }, "M43 independent review score/severity drift")
    for key, value in core.items():
        require(review[key] == value, "review core drift: {}".format(key))
    admitted = review["admitted"]
    require(admitted["exact_valid_pad_support_and_local_pairs"] is True and
            admitted["exact_signed_parent_delta_conservation"] is True and
            admitted["finite_p8_l96_source_bank_capacity_schedule"] is True and
            admitted["independent_memory_service_capacity_gates"] is True and
            admitted["integrated_executable_multicontext_schedule"] is False and
            admitted["integer_output_equivalence"] is False and
            admitted["three_x_target_crossing"] is False and
            admitted["rtl_vcs_synopsys_ppa_power_energy"] is False and
            admitted["system_or_end_to_end_speedup"] is False and
            admitted["date_headline_or_best_paper"] is False,
            "M43 independent review claim boundary drift")
    require(len(review["findings"]["p0"]) == 0 and
            len(review["findings"]["p1"]) == 0 and
            len(review["findings"]["p2"]) == 5,
            "M43 independent finding population drift")
    require("not integrated executable cycles" in review["claim_boundary"] and
            "not a measured 3x crossing" in review["claim_boundary"] and
            "not a source row" in review["claim_boundary"],
            "M43 independent claim text is not fail closed")


def rerun_candidate_and_compare():
    validator = HW_ROOT / ANCHORS["candidate_validator"][0]
    regression = HW_ROOT / ANCHORS["candidate_regression"][0]
    subprocess.check_call(["/usr/bin/python3.6", "-m", "unittest", "-v",
                           str(regression)])
    subprocess.check_call(["/usr/bin/python3.6", str(validator), "--rerun"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--rerun-candidate", action="store_true")
    parser.add_argument("--dump-core", type=Path)
    args = parser.parse_args()
    anchors = validate_anchors()
    replay = independent_rebuild()
    weights = validate_m41_weights()
    spatial = read_json(HW_ROOT / ANCHORS["spatial_result"][0])
    temporal = read_json(HW_ROOT / ANCHORS["temporal_ablation"][0])
    validate_candidate_result(spatial, replay, temporal=False)
    validate_candidate_result(temporal, replay, temporal=True)
    gates = capacity_and_target_gates(replay, weights)
    attacks = adversarial_checks(spatial, replay)
    core = expected_review_core(replay, weights, gates, attacks, anchors)
    if args.dump_core:
        require(not args.dump_core.exists(), "refusing to overwrite dump-core output")
        args.dump_core.write_text(json.dumps(core, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    if args.review.is_file():
        validate_review(read_json(args.review), core)
    else:
        require(args.dump_core is not None, "review is missing and no --dump-core requested")
    if args.rerun_candidate:
        rerun_candidate_and_compare()
    print("PASS M43-r1 independent exact source-bank review validation")


if __name__ == "__main__":
    main()
