#!/usr/bin/env python3
"""M43 exact tile-resident spatial/temporal parent-delta source scheduler.

This analyzer deliberately reports source-bank issue work and state traffic.  It
does not turn those quantities into an end-to-end or PPA claim.  A later RTL
milestone must implement the signed add/subtract datapath and prove its timing.
"""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = HW_ROOT / (
    "contracts/m43_tile_resident_parent_delta_schedule_contract_r1_20260823.json")
EXPECTED_CONTRACT_SHA256 = (
    "c894b5fcdd6a6cd7d33bf736e8c084630f0ea297f632e1dd6a35889714772e44")
DEFAULT_MANIFEST = HW_ROOT / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json")
DEFAULT_M40_RESULT = HW_ROOT / (
    "results/m40_conflict_aware_event_schedule_r3_20260822/"
    "m40_conflict_aware_event_schedule.json")
DEFAULT_M41_RESULT = HW_ROOT / (
    "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/"
    "m41_h67_ep35_bottleneck_int8_bridge.json")
DEFAULT_M41_REVIEW = HW_ROOT / (
    "results/m41_h67_ep35_bottleneck_int8_independent_hammer_review_r1_20260823/"
    "m41_independent_hammer_review.json")
DEFAULT_M42_RESULT = HW_ROOT / (
    "results/m42_real_work_headroom_gate_r1_20260823/"
    "m42_real_work_headroom_gate.json")
DEFAULT_M42_REVIEW = HW_ROOT / (
    "results/m42_real_work_headroom_gate_r1_20260823/"
    "m42_r1_independent_hammer_review.json")
EXPECTED_SHA256 = {
    "manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m40_result": "419ea51faabda4c2f45b9fa535d1a0fa8142bb4c8b8258468e88a1dc99c310e7",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m41_review": "81817dd55c90c2c05eb3579030c16f1fc60fe87c446dd38067392ab5b2d52552",
    "m42_result": "c0677ce56775996481ba500fc397191e7de407768f29c591ae731c69ed45cd13",
    "m42_review": "de7a6187b5a4a693023948045ae27480051713192564b74cf66055648cbc0d02",
}

TIMESTEPS = 10
CHANNELS = 768
HEIGHT = 15
WIDTH = 20
FEATURES = CHANNELS * 3 * 3
TILE_BITS = 256
TILES = (FEATURES + TILE_BITS - 1) // TILE_BITS
ISSUE_WIDTH = 8
OUTPUT_LANES = 96
OUTPUT_BLOCKS = CHANNELS // OUTPUT_LANES
ROWS = TIMESTEPS * HEIGHT * WIDTH
ACCUMULATOR_BITS = 19
ACCUMULATOR_STORAGE_BYTES = 3
WEIGHT_LOAD_BYTES_PER_CYCLE = 32
LOCAL_RESIDENCY_BYTES = 193728
CONTEXTS = 4
PARENT_PRIORITY = ("local_zero", "left", "up", "previous_timestep")
ALLOW_TEMPORAL_PARENT = False
BANK_MASKS = tuple(sum(1 << position
                       for position in range(bank, TILE_BITS, ISSUE_WIDTH))
                   for bank in range(ISSUE_WIDTH))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook,
                      parse_constant=reject_constant)


def population(value):
    method = getattr(value, "bit_count", None)
    if method is not None:
        return method()
    return bin(value).count("1")


def distribution(values):
    require(values, "empty distribution")
    ordered = sorted(values)

    def nearest_rank(numerator, denominator):
        rank = (numerator * len(ordered) + denominator - 1) // denominator
        return ordered[rank - 1]

    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean_exact": {"numerator": sum(values), "denominator": len(values)},
        "p50_nearest_rank": nearest_rank(50, 100),
        "p95_nearest_rank": nearest_rank(95, 100),
        "p99_nearest_rank": nearest_rank(99, 100),
    }


def unpack_record_masks(trace_dir, record):
    require(record["shape"] == [TIMESTEPS, 1, CHANNELS, HEIGHT, WIDTH],
            "record shape drift")
    path = trace_dir / record["packed_file"]
    require(path.is_file() and sha256(path) == record["packed_file_sha256"],
            "packed source identity drift")
    raw = path.read_bytes()
    plane_bytes = record["positive_plane_bytes"]
    require(len(raw) == 3 * plane_bytes, "packed source extent drift")
    positive = raw[:plane_bytes]
    negative = raw[plane_bytes:2 * plane_bytes]
    require(not any(negative), "M43 expects the frozen nonnegative two-code trace")

    masks = [0] * (ROWS * TILES)
    total_bits = TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    for byte_index, byte in enumerate(positive):
        if byte == 0:
            continue
        bit_base = byte_index * 8
        while byte:
            low = byte & -byte
            bit = low.bit_length() - 1
            flat = bit_base + bit
            require(flat < total_bits, "nonzero packed tail bit")
            tc, spatial = divmod(flat, HEIGHT * WIDTH)
            timestep, channel = divmod(tc, CHANNELS)
            input_y, input_x = divmod(spatial, WIDTH)
            feature_base = channel * 9
            for kernel_y in range(3):
                output_y = input_y - kernel_y + 1
                if output_y < 0 or output_y >= HEIGHT:
                    continue
                for kernel_x in range(3):
                    output_x = input_x - kernel_x + 1
                    if output_x < 0 or output_x >= WIDTH:
                        continue
                    feature = feature_base + kernel_y * 3 + kernel_x
                    tile, tile_bit = divmod(feature, TILE_BITS)
                    row = (timestep * HEIGHT + output_y) * WIDTH + output_x
                    index = row * TILES + tile
                    masks[index] |= 1 << tile_bit
            byte ^= low
    return masks


def select_parent(masks, row, tile):
    index = row * TILES + tile
    current = masks[index]
    timestep_spatial, output_x = divmod(row, WIDTH)
    timestep, output_y = divmod(timestep_spatial, HEIGHT)
    candidates = [("local_zero", 0)]
    if output_x > 0:
        candidates.append(("left", masks[index - TILES]))
    if output_y > 0:
        candidates.append(("up", masks[index - WIDTH * TILES]))
    if ALLOW_TEMPORAL_PARENT and timestep > 0:
        candidates.append(("previous_timestep",
                           masks[index - HEIGHT * WIDTH * TILES]))
    rank = dict((name, order) for order, name in enumerate(PARENT_PRIORITY))
    # The physical objective is finite-bank issue time, not only the number of
    # logical source terms.  Count is the deterministic secondary objective.
    best = min(candidates,
               key=lambda item: (bank_issue_cycles(current ^ item[1]),
                                 population(current ^ item[1]),
                                 rank[item[0]]))
    parent_name, parent = best
    add_mask = current & ~parent
    subtract_mask = parent & ~current
    require((add_mask & subtract_mask) == 0 and
            (add_mask | subtract_mask) == (current ^ parent),
            "signed symmetric-difference partition mismatch")
    return parent_name, parent, add_mask, subtract_mask


def bank_issue_cycles(mask):
    return max(population(mask & bank_mask) for bank_mask in BANK_MASKS)


def analyze_record(trace_dir, record, expected_local_pairs):
    masks = unpack_record_masks(trace_dir, record)
    local_pairs = 0
    delta_pairs = 0
    add_pairs = 0
    subtract_pairs = 0
    local_issue_cycles = 0
    delta_issue_cycles = 0
    local_nonempty_commands = 0
    delta_nonempty_commands = 0
    exact_copy_tiles = 0
    parent_tiles = dict((name, 0) for name in PARENT_PRIORITY)
    parent_rows = dict((name, 0) for name in PARENT_PRIORITY)
    row_delta_pairs = []
    row_local_pairs = []
    row_parent = []

    for row in range(ROWS):
        row_local = 0
        row_delta = 0
        row_parent_counts = dict((name, 0) for name in PARENT_PRIORITY)
        for tile in range(TILES):
            current = masks[row * TILES + tile]
            current_count = population(current)
            row_local += current_count
            local_pairs += current_count
            local_cycles = bank_issue_cycles(current)
            local_issue_cycles += local_cycles * OUTPUT_BLOCKS
            if current:
                local_nonempty_commands += OUTPUT_BLOCKS

            parent_name, parent, add_mask, subtract_mask = select_parent(
                masks, row, tile)
            delta_mask = add_mask | subtract_mask
            count = population(delta_mask)
            row_delta += count
            delta_pairs += count
            add_pairs += population(add_mask)
            subtract_pairs += population(subtract_mask)
            parent_tiles[parent_name] += 1
            row_parent_counts[parent_name] += 1
            cycles = bank_issue_cycles(delta_mask)
            delta_issue_cycles += cycles * OUTPUT_BLOCKS
            if delta_mask:
                delta_nonempty_commands += OUTPUT_BLOCKS
            elif parent_name != "local_zero":
                exact_copy_tiles += 1
            if parent_name != "local_zero":
                require(parent != 0 or current == 0,
                        "nonlocal empty parent selected against nonempty current")
        local_winner = min(PARENT_PRIORITY,
                           key=lambda name: (-row_parent_counts[name],
                                             PARENT_PRIORITY.index(name)))
        parent_rows[local_winner] += 1
        row_parent.append(local_winner)
        row_local_pairs.append(row_local)
        row_delta_pairs.append(row_delta)

    require(local_pairs == expected_local_pairs,
            "M40 Local pair reconciliation mismatch")
    require(add_pairs + subtract_pairs == delta_pairs,
            "delta polarity conservation mismatch")
    require(sum(parent_tiles.values()) == ROWS * TILES,
            "tile parent population mismatch")
    require(sum(parent_rows.values()) == ROWS,
            "row parent population mismatch")
    return {
        "sample_id": record["sample_id"],
        "operator": record["operator"],
        "rows": ROWS,
        "feature_tiles": TILES,
        "local_source_destination_pairs": local_pairs,
        "parent_delta_source_destination_pairs": delta_pairs,
        "parent_delta_add_pairs": add_pairs,
        "parent_delta_subtract_pairs": subtract_pairs,
        "logical_pair_reduction": {
            "numerator": local_pairs - delta_pairs,
            "denominator": local_pairs,
        },
        "local_p8_l96_source_issue_cycles": local_issue_cycles,
        "parent_delta_p8_l96_source_issue_cycles": delta_issue_cycles,
        "local_effective_issue_width": {
            "numerator": local_pairs * OUTPUT_BLOCKS,
            "denominator": local_issue_cycles,
        },
        "parent_delta_effective_issue_width": {
            "numerator": delta_pairs * OUTPUT_BLOCKS,
            "denominator": delta_issue_cycles,
        },
        "local_nonempty_tile_block_commands": local_nonempty_commands,
        "parent_delta_nonempty_tile_block_commands": delta_nonempty_commands,
        "exact_nonlocal_zero_delta_tile_copies": exact_copy_tiles,
        "parent_choice_by_tile": parent_tiles,
        "dominant_parent_by_row": parent_rows,
        "row_local_pair_distribution": distribution(row_local_pairs),
        "row_parent_delta_pair_distribution": distribution(row_delta_pairs),
    }


def validate_int8_bridge(result_path, review_path):
    result = read_json(result_path)
    review = read_json(review_path)
    require(result["schema"] == "m41_h67_ep35_bottleneck_int8_bridge_result_v1",
            "M41 result schema drift")
    require(review["status"] == (
        "GO_CHECKPOINT_BOUND_MODEL_BRIDGE_WITH_ONE_P1_REPRODUCIBILITY_REPAIR_"
        "NO_GO_SYSTEM_OR_EXPLICIT_MASK_CLAIMS"), "M41 review is not model GO")
    require(result["m40_schedule_bridge"][
        "checkpoint_tight_accumulator_signed_bits"] == ACCUMULATOR_BITS,
        "M41 accumulator width drift")
    result_dir = Path(result_path).resolve().parent
    weight_payloads = [item for item in result["payloads"]
                       if item["role"] == "weight"]
    require(len(weight_payloads) == 4, "M41 weight payload population drift")
    total_bytes = 0
    for index, item in enumerate(weight_payloads):
        require(item["shape"] == [CHANNELS, 3, 3, CHANNELS] and
                item["layout"] == "I_KY_KX_O_C_ORDER" and
                item["dtype"] == "signed_int8",
                "M41 weight layout drift")
        path = result_dir / item["file"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"],
                "M41 weight payload identity drift: {}".format(index))
        total_bytes += item["bytes"]
    require(total_bytes == 21233664, "M41 total INT8 weight bytes drift")
    return result, review, total_bytes


def validate_headroom_gate(result_path, review_path):
    result = read_json(result_path)
    review = read_json(review_path)
    require(result["schema"] == "m42_real_work_headroom_gate_result_v1",
            "M42 result schema drift")
    require(review["review"]["decision"] == "GO_EXACT_HEADROOM_GATE_ONLY" and
            review["review"]["p0"] == 0 and review["review"]["p1"] == 0,
            "M42 independent headroom review is not GO")
    model = result["frozen_resource_model"]
    require(model["event_engine_issue_width"] == ISSUE_WIDTH and
            model["event_engine_output_lanes"] == OUTPUT_LANES and
            model["event_engine_peak_product_adds_per_cycle"] ==
            ISSUE_WIDTH * OUTPUT_LANES,
            "M42 P8-L96 geometry drift")
    return result, review


def validate_contract(contract_path):
    require(Path(contract_path).resolve() == DEFAULT_CONTRACT.resolve() and
            sha256(contract_path) == EXPECTED_CONTRACT_SHA256,
            "M43 canonical contract identity drift")
    contract = read_json(contract_path)
    require(contract["schema"] ==
            "m43_tile_resident_parent_delta_schedule_contract_v1",
            "M43 contract schema drift")
    geometry = contract["geometry"]
    require(geometry["timesteps"] == TIMESTEPS and
            geometry["channels"] == CHANNELS and
            geometry["height"] == HEIGHT and geometry["width"] == WIDTH and
            geometry["features"] == FEATURES and
            geometry["feature_tile_bits"] == TILE_BITS and
            geometry["feature_tiles"] == TILES and
            geometry["issue_width"] == ISSUE_WIDTH and
            geometry["output_lanes"] == OUTPUT_LANES and
            geometry["output_blocks"] == OUTPUT_BLOCKS and
            geometry["peak_product_adds_per_cycle"] ==
            ISSUE_WIDTH * OUTPUT_LANES,
            "M43 contract geometry drift")
    capacity = contract["capacity_model"]
    require(capacity["accumulator_signed_bits_checkpoint_tight"] ==
            ACCUMULATOR_BITS and
            capacity["accumulator_storage_bytes"] ==
            ACCUMULATOR_STORAGE_BYTES and
            capacity["weight_load_bytes_per_cycle"] ==
            WEIGHT_LOAD_BYTES_PER_CYCLE and
            capacity["local_residency_bytes"] == LOCAL_RESIDENCY_BYTES and
            capacity["candidate_contexts"] == CONTEXTS,
            "M43 contract capacity drift")
    for name, item in contract["inputs"].items():
        require(type(item) is dict and set(item) == {"path", "sha256"},
                "M43 contract input descriptor drift: {}".format(name))
        path = HW_ROOT / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                "M43 contract input identity drift: {}".format(name))
    return contract


def build(manifest_path=DEFAULT_MANIFEST, m40_result_path=DEFAULT_M40_RESULT,
          m41_result_path=DEFAULT_M41_RESULT,
          m41_review_path=DEFAULT_M41_REVIEW,
          m42_result_path=DEFAULT_M42_RESULT,
          m42_review_path=DEFAULT_M42_REVIEW,
          contract_path=DEFAULT_CONTRACT):
    contract = validate_contract(contract_path)
    exact_paths = {
        "manifest": manifest_path, "m40_result": m40_result_path,
        "m41_result": m41_result_path, "m41_review": m41_review_path,
        "m42_result": m42_result_path, "m42_review": m42_review_path,
    }
    for name, path in exact_paths.items():
        require(Path(path).is_file() and sha256(path) == EXPECTED_SHA256[name],
                "M43 upstream identity drift: {}".format(name))
    manifest = read_json(manifest_path)
    m40 = read_json(m40_result_path)
    m41, m41_review, int8_weight_bytes = validate_int8_bridge(
        m41_result_path, m41_review_path)
    m42, m42_review = validate_headroom_gate(m42_result_path, m42_review_path)
    require(manifest["schema"] == "m40_bottleneck_packed_source_trace_v1",
            "M40 manifest schema drift")
    require(manifest["cohort"]["records"] == 40 and
            manifest["cohort"]["samples"] == 10,
            "M40 cohort drift")
    expected = dict(((row["sample_id"], row["operator"]),
                     row["Local"]["source_destination_pairs"])
                    for row in m40["real_source_trace"]["records"])
    trace_dir = Path(manifest_path).resolve().parent
    rows = []
    for index, record in enumerate(manifest["records"]):
        key = (record["sample_id"], record["operator"])
        require(key in expected, "M40 record identity mismatch")
        rows.append(analyze_record(trace_dir, record, expected[key]))
        print("[M43] {}/40 sample={} operator={}".format(
            index + 1, record["sample_id"], record["operator"]))

    per_sample = []
    for sample_id in range(10):
        selected = [row for row in rows if row["sample_id"] == sample_id]
        require(len(selected) == 4, "sample operator population drift")
        per_sample.append({
            "sample_id": sample_id,
            "local_pairs": sum(row["local_source_destination_pairs"]
                               for row in selected),
            "delta_pairs": sum(row["parent_delta_source_destination_pairs"]
                               for row in selected),
            "local_issue_cycles": sum(row["local_p8_l96_source_issue_cycles"]
                                      for row in selected),
            "delta_issue_cycles": sum(
                row["parent_delta_p8_l96_source_issue_cycles"]
                for row in selected),
            "local_commands": sum(row["local_nonempty_tile_block_commands"]
                                  for row in selected),
            "delta_commands": sum(
                row["parent_delta_nonempty_tile_block_commands"]
                for row in selected),
            "parent_nonlocal_tile_block_reads": sum(
                sum(value for name, value in row["parent_choice_by_tile"].items()
                    if name != "local_zero") * OUTPUT_BLOCKS
                for row in selected),
        })

    total_local = sum(row["local_source_destination_pairs"] for row in rows)
    total_delta = sum(row["parent_delta_source_destination_pairs"] for row in rows)
    total_local_cycles = sum(row["local_p8_l96_source_issue_cycles"]
                             for row in rows)
    total_delta_cycles = sum(row["parent_delta_p8_l96_source_issue_cycles"]
                             for row in rows)
    parent_tiles = dict((name, sum(row["parent_choice_by_tile"][name]
                                  for row in rows))
                        for name in PARENT_PRIORITY)
    frozen = m42["frozen_resource_model"]
    fixed_cycles = frozen["fixed_late_scale_plus_frontend_cycles"]
    outside_cycles = frozen["outside_four_bottleneck_model_cycles"]
    fixed_reference = frozen["fixed_compute_reference_cycles"]
    three_x_product_ceiling = next(
        item["maximum_executable_product_cycles_required"]["numerator"]
        // item["maximum_executable_product_cycles_required"]["denominator"]
        for item in m42["target_gates"]
        if item["target_compute_speedup"] == {"numerator": 3, "denominator": 1})
    require(three_x_product_ceiling == 15495075,
            "M42 integer 3x product ceiling drift")
    tile_rows_per_sample = 4 * ROWS * TILES
    tile_block_commands = tile_rows_per_sample * OUTPUT_BLOCKS
    final_accumulator_service_cycles = (
        4 * ROWS * (TILES + TILES - 1) * OUTPUT_BLOCKS)
    parent_selector_service_cycles = tile_rows_per_sample * (
        2 if ALLOW_TEMPORAL_PARENT else 1)
    weight_load_cycles = ((int8_weight_bytes + WEIGHT_LOAD_BYTES_PER_CYCLE - 1)
                          // WEIGHT_LOAD_BYTES_PER_CYCLE)
    for sample in per_sample:
        sample["all_tile_block_commands"] = tile_block_commands
        sample["parent_partial_single_port_service_cycles"] = (
            tile_block_commands + sample["parent_nonlocal_tile_block_reads"])
        sample["final_accumulator_single_port_service_cycles"] = (
            final_accumulator_service_cycles)
        sample["int8_weight_load_256b_cycles"] = weight_load_cycles
        services = {
            "finite_bank_source_issue": sample["delta_issue_cycles"],
            "descriptor_enqueue_one_per_cycle": tile_block_commands,
            "parent_selector_support_single_port":
                parent_selector_service_cycles,
            "parent_partial_single_port":
                sample["parent_partial_single_port_service_cycles"],
            "final_accumulator_single_port": final_accumulator_service_cycles,
            "int8_weight_load_256b": weight_load_cycles,
        }
        sample["independent_service_capacity_cycles"] = services
        sample["independent_service_capacity_max"] = max(services.values())
        sample["source_issue_is_capacity_max"] = (
            sample["independent_service_capacity_max"] ==
            sample["delta_issue_cycles"])
        sample["conditional_zero_visible_overhead_replacement_cycles"] = (
            sample["delta_issue_cycles"] + fixed_cycles)
        sample["conditional_zero_visible_overhead_compute_total"] = (
            outside_cycles + sample[
                "conditional_zero_visible_overhead_replacement_cycles"])
        sample["conditional_zero_visible_overhead_compute_speedup"] = {
            "numerator": fixed_reference,
            "denominator": sample[
                "conditional_zero_visible_overhead_compute_total"],
        }
        sample["visible_product_overhead_headroom_to_3x"] = (
            three_x_product_ceiling - sample["delta_issue_cycles"])
        sample["three_x_crossing_admitted"] = False

    spatial_parent_bytes = WIDTH * OUTPUT_LANES * ACCUMULATOR_STORAGE_BYTES
    temporal_parent_bytes = (HEIGHT * WIDTH * OUTPUT_LANES *
                             ACCUMULATOR_STORAGE_BYTES)
    support_parent_bytes = ((HEIGHT * WIDTH if ALLOW_TEMPORAL_PARENT else WIDTH)
                            * (TILE_BITS // 8))
    weight_double_buffer_bytes = 2 * TILE_BITS * OUTPUT_LANES
    context_bytes = CONTEXTS * (OUTPUT_LANES * ACCUMULATOR_STORAGE_BYTES + 64)
    local_scratch_bytes = (weight_double_buffer_bytes +
                           (temporal_parent_bytes if ALLOW_TEMPORAL_PARENT
                            else spatial_parent_bytes) +
                           support_parent_bytes + context_bytes)
    require(local_scratch_bytes <= LOCAL_RESIDENCY_BYTES,
            "M43 candidate local scratch exceeds frozen residency")
    issue_values = [sample["delta_issue_cycles"] for sample in per_sample]
    local_issue_values = [sample["local_issue_cycles"] for sample in per_sample]
    p95_issue = distribution(issue_values)["p95_nearest_rank"]
    visible_headroom = three_x_product_ceiling - p95_issue
    require(visible_headroom > 0, "M43 source issue misses the 3x product gate")
    return {
        "schema": "m43_tile_resident_parent_delta_schedule_v1",
        "status": (
            "PASS_M43A_EXACT_FINITE_SOURCE_BANK_SCHEDULE_AND_CAPACITY_GATES_"
            "BUT_MULTICONTEXT_RTL_MEMORY_TIMING_AND_SYSTEM_SPEEDUP_UNADMITTED"),
        "identity": {
            "contract_sha256": sha256(contract_path),
            "upstream_sha256": dict((name, sha256(path))
                                    for name, path in exact_paths.items()),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
        },
        "architecture": {
            "name": ("TILE_RESIDENT_SIGNED_SPATIOTEMPORAL_PARENT_DELTA"
                     if ALLOW_TEMPORAL_PARENT else
                     "TILE_RESIDENT_SIGNED_SPATIAL_PARENT_DELTA"),
            "feature_order": "cin_then_ky_then_kx",
            "features": FEATURES,
            "tile_bits": TILE_BITS,
            "feature_tiles": TILES,
            "issue_width": ISSUE_WIDTH,
            "output_lanes": OUTPUT_LANES,
            "output_blocks": OUTPUT_BLOCKS,
            "peak_product_adds_per_cycle": ISSUE_WIDTH * OUTPUT_LANES,
            "source_bank": "global_feature_index_mod_8",
            "parents": (["local_zero", "left", "up"] if
                        not ALLOW_TEMPORAL_PARENT else list(PARENT_PRIORITY)),
            "temporal_parent_enabled": ALLOW_TEMPORAL_PARENT,
            "selection_objective": (
                "minimum finite-bank source-issue cycles, then minimum signed "
                "delta population, then frozen parent priority"),
            "tie_break_priority": (["local_zero", "left", "up"] if
                                   not ALLOW_TEMPORAL_PARENT else
                                   list(PARENT_PRIORITY)),
            "signed_delta": "current_minus_parent; additions and subtractions share bank capacity",
            "tile_partial_identity": (
                "Y_tile(S)=Y_tile(P)+sum(S\\P)W-sum(P\\S)W; "
                "sum over all 27 feature tiles recovers the quantized convolution"),
            "accumulator_bits": ACCUMULATOR_BITS,
            "accumulator_storage_bytes": ACCUMULATOR_STORAGE_BYTES,
            "contexts_required_by_candidate": CONTEXTS,
        },
        "physical_layout_bridge": {
            "int8_weight_layout": m41["quantization_contract"][
                "weight_payload_layout"],
            "weight_linear_address": (
                "operator_base + (((cin*3+ky)*3+kx)*768+cout)"),
            "source_bank": "global_feature_index_mod_8",
            "source_bank_row": "global_feature_index_div_8",
            "weight_bytes_per_sample": int8_weight_bytes,
            "weight_load_bus_bytes_per_cycle": WEIGHT_LOAD_BYTES_PER_CYCLE,
            "weight_double_buffer_bytes": weight_double_buffer_bytes,
            "parent_partial_buffer_bytes": (
                temporal_parent_bytes if ALLOW_TEMPORAL_PARENT
                else spatial_parent_bytes),
            "support_parent_buffer_bytes": support_parent_bytes,
            "four_context_bytes": context_bytes,
            "local_scratch_bytes": local_scratch_bytes,
            "frozen_local_residency_bytes": LOCAL_RESIDENCY_BYTES,
            "local_scratch_fits": True,
            "one_output_block_all_timestep_final_accumulator_bytes": (
                ROWS * OUTPUT_LANES * ACCUMULATOR_STORAGE_BYTES),
            "qualification": (
                "exact byte capacities and service counts; macro timing and energy "
                "are not admitted"),
        },
        "population": {
            "records": len(rows),
            "samples": 10,
            "operators": 4,
            "output_rows": len(rows) * ROWS,
            "tile_rows": len(rows) * ROWS * TILES,
        },
        "aggregate": {
            "local_source_destination_pairs": total_local,
            "parent_delta_source_destination_pairs": total_delta,
            "logical_pair_reduction": {
                "numerator": total_local - total_delta,
                "denominator": total_local,
            },
            "local_p8_l96_source_issue_cycles": total_local_cycles,
            "parent_delta_p8_l96_source_issue_cycles": total_delta_cycles,
            "local_effective_issue_width": {
                "numerator": total_local * OUTPUT_BLOCKS,
                "denominator": total_local_cycles,
            },
            "parent_delta_effective_issue_width": {
                "numerator": total_delta * OUTPUT_BLOCKS,
                "denominator": total_delta_cycles,
            },
            "parent_choice_by_tile": parent_tiles,
            "local_source_issue_cycle_distribution_per_sample":
                distribution(local_issue_values),
            "parent_delta_source_issue_cycle_distribution_per_sample":
                distribution(issue_values),
            "finite_bank_issue_cycle_reduction": {
                "numerator": total_local_cycles - total_delta_cycles,
                "denominator": total_local_cycles,
            },
        },
        "three_x_headroom_gate": {
            "m42_maximum_product_cycles": three_x_product_ceiling,
            "p95_parent_delta_source_issue_cycles": p95_issue,
            "p95_visible_product_overhead_headroom_cycles": visible_headroom,
            "p95_visible_overhead_cycles_per_all_tile_block_command": {
                "numerator": visible_headroom,
                "denominator": tile_block_commands,
            },
            "all_samples_source_issue_below_gate": all(
                sample["delta_issue_cycles"] <= three_x_product_ceiling
                for sample in per_sample),
            "all_samples_independent_service_capacity_below_source_issue": all(
                sample["source_issue_is_capacity_max"] for sample in per_sample),
            "target_crossing_admitted": False,
            "reason": (
                "multi-context dependency schedule, memory overlap, signed-delta RTL "
                "and same-resource integration remain unproved"),
        },
        "per_sample": per_sample,
        "records": rows,
        "qualification": {
            "exact": [
                "M40 bitmap to valid-padding Conv3x3 patch expansion",
                "zero/left/up/previous-timestep tile-parent selection",
                "signed symmetric-difference source conservation",
                "eight finite source banks with one source per bank per cycle",
                "eight 96-output blocks for every source issue",
                "M41 reviewed I_KY_KX_O INT8 payload identity and byte layout",
                "per-sample independent service-capacity counts",
            ],
            "not_yet_admitted": [
                "command/response/state-memory overlap and complete engine cycles",
                "signed parent-delta RTL and integer output miter",
                "SRAM macro timing/energy and DRAM schedule",
                "end-to-end/system speedup, PPA, energy, external comparison or headline",
            ],
        },
        "claim_policy": contract["claim_policy"],
    }


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite M43 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--m40-result", type=Path, default=DEFAULT_M40_RESULT)
    parser.add_argument("--m41-result", type=Path, default=DEFAULT_M41_RESULT)
    parser.add_argument("--m41-review", type=Path, default=DEFAULT_M41_REVIEW)
    parser.add_argument("--m42-result", type=Path, default=DEFAULT_M42_RESULT)
    parser.add_argument("--m42-review", type=Path, default=DEFAULT_M42_REVIEW)
    parser.add_argument("--enable-temporal-parent", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global ALLOW_TEMPORAL_PARENT
    ALLOW_TEMPORAL_PARENT = args.enable_temporal_parent
    payload = build(args.manifest.resolve(), args.m40_result.resolve(),
                    args.m41_result.resolve(), args.m41_review.resolve(),
                    args.m42_result.resolve(), args.m42_review.resolve(),
                    args.contract.resolve())
    write_output(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
