#!/usr/bin/env python3
"""Independent, read-only M103 correction/fallback reuse preflight.

This script does not import M43/M72/M78/M83/M88 producer code.  It SHA-pins
their frozen sources/results, independently decodes the twenty heldout M40
packed support binaries, reconstructs cap11 correction/fallback multisets, and
reports order-independent grouping facts plus explicitly non-admitted bounds.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
OUTPUT = HERE / "m103_correction_reuse_preflight_audit.json"

M40_DIR = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822"
M40_MANIFEST = M40_DIR / "m40_bottleneck_packed_source_manifest.json"
M43_ANALYZER = HW / "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py"
M72_ANALYZER = HW / "system_simulator/scripts/analyze_m72_phi_kmeans_k16q16_heldout.py"
M72_RESULT = HW / "results/m72_phi_kmeans_k16q16_valid825_internal_screen_dev_r1_20260823/m72_phi_kmeans_k16q16_valid825_internal_screen.json"
M78_ANALYZER = HW / "system_simulator/scripts/analyze_m78_precision_elastic_pwp.py"
M78_RESULT = HW / "results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/m78_precision_elastic_pwp.json"
M83_EXPORTER = HW / "system_simulator/scripts/export_m83_canonical_cap11_pwp_records.py"
M83_RECEIPT = HW / "results/m83_canonical_cap11_pwp_records_r1_20260823/m83_canonical_cap11_pwp_records_receipt.json"
M83_RECORDS = Path("/tmp/m85_inputs/m83_cap11_phase_records.bin")
M83_OFFSETS = Path("/tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin")
M88_ANALYZER = HW / "system_simulator/scripts/analyze_m88_bounded_sync_bank_double_buffer.py"
M88_RESULT = HW / "results/m88_bounded_sync_bank_double_buffer_valid825_internal_r1_20260823/m88_bounded_sync_bank_double_buffer.json"

EXPECTED_SHA256 = {
    "m40_manifest": "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3",
    "m43_analyzer": "a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3",
    "m72_analyzer": "eb31555b6be64a8a9376647b16a1cb039dc3b49b19f176abb759b522dc93dfa2",
    "m72_result": "e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133",
    "m78_analyzer": "9215c2eeff8ccbfa0ef7d27f48ed6100a56813c1881873013e9e23a2e149df6b",
    "m78_result": "00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc",
    "m83_exporter": "1e279f5b19d60dec52817e8f0f5bf066f14b2226f8a5646299597315419d3e45",
    "m83_receipt": "46893b0dc7499f3c163d4c3709560f5d208a2272bb49dd8ce709132062bb4303",
    "m83_records": "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d",
    "m83_offsets": "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c",
    "m88_analyzer": "5b62d1f23555fba4bc00f1e1b427ae5861089e0a8ea5f8ae98c062acb071dfae",
    "m88_result": "36e9b0603422ccff7afd23e6e5e2309bc5d53b3c7e9898538095d6baa23da483",
}

TIMESTEPS = 10
CHANNELS = 768
HEIGHT = 15
WIDTH = 20
ROWS = TIMESTEPS * HEIGHT * WIDTH
FEATURES = CHANNELS * 3 * 3
TILE_BITS = 256
TILES = (FEATURES + TILE_BITS - 1) // TILE_BITS
PARTITION_BITS = 16
PARTITIONS = FEATURES // PARTITION_BITS
OUTPUT_BLOCKS = 8
POPCOUNT = tuple(bin(value).count("1") for value in range(1 << 16))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError("nonstandard JSON " + token)))


def decode_support_masks(record):
    require(record["shape"] == [10, 1, 768, 15, 20], "M40 shape")
    path = M40_DIR / record["packed_file"]
    require(path.is_file() and path.stat().st_size == record["packed_file_bytes"]
            and sha256(path) == record["packed_file_sha256"],
            "M40 packed identity")
    raw = path.read_bytes()
    plane_bytes = record["positive_plane_bytes"]
    require(len(raw) == 3 * plane_bytes and
            not any(raw[plane_bytes:2 * plane_bytes]),
            "M40 plane geometry/nonnegative identity")
    positive = raw[:plane_bytes]
    masks = [0] * (ROWS * TILES)
    total_bits = TIMESTEPS * CHANNELS * HEIGHT * WIDTH
    for byte_index, byte in enumerate(positive):
        bit_base = byte_index * 8
        while byte:
            low = byte & -byte
            bit = low.bit_length() - 1
            flat = bit_base + bit
            require(flat < total_bits, "M40 nonzero tail")
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
                    masks[row * TILES + tile] |= 1 << tile_bit
            byte ^= low
    return masks


def nearest(value, centers):
    # Frozen M78 tie order: Hamming, numeric center, then center index.
    return min((POPCOUNT[value ^ center], center, index)
               for index, center in enumerate(centers))


def nearest_rank(histogram, numerator, denominator):
    count = sum(histogram.values())
    target = (numerator * count + denominator - 1) // denominator
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= target:
            return value
    raise ValueError("empty rank histogram")


def target_math(bit_sparse_service, current_candidate, correction, pwp,
                pwp_events):
    result = {}
    for label, target in (("2p0x", 2.0), ("2p5x", 2.5)):
        limit = int(math.floor(bit_sparse_service / target))
        result[label] = {
            "candidate_service_target_max": limit,
            "total_reduction_from_current": current_candidate - limit,
            "correction_reduction_if_pwp_unchanged":
                correction - (limit - pwp),
            "absolute_minimum_correction_reduction_even_if_pwp_deleted":
                correction - limit,
            "minimum_correction_reduction_if_pwp_one_token_per_event":
                correction - (limit - pwp_events),
        }
    return result


def main():
    paths = {
        "m40_manifest": M40_MANIFEST,
        "m43_analyzer": M43_ANALYZER,
        "m72_analyzer": M72_ANALYZER,
        "m72_result": M72_RESULT,
        "m78_analyzer": M78_ANALYZER,
        "m78_result": M78_RESULT,
        "m83_exporter": M83_EXPORTER,
        "m83_receipt": M83_RECEIPT,
        "m83_records": M83_RECORDS,
        "m83_offsets": M83_OFFSETS,
        "m88_analyzer": M88_ANALYZER,
        "m88_result": M88_RESULT,
    }
    observed = {}
    for name, path in paths.items():
        require(path.is_file(), "missing " + name)
        observed[name] = sha256(path)
        require(observed[name] == EXPECTED_SHA256[name], name + " SHA drift")

    manifest = strict_json(M40_MANIFEST)
    m72 = strict_json(M72_RESULT)
    m78 = strict_json(M78_RESULT)
    m83 = strict_json(M83_RECEIPT)
    m88 = strict_json(M88_RESULT)
    require(manifest["schema"] == "m40_bottleneck_packed_source_trace_v1" and
            len(manifest["records"]) == 40, "M40 manifest")
    require(m72["status"].startswith("PASS_M72_VALID825_INTERNAL_SCREEN"),
            "M72 status")
    require(m78["status"] ==
            "PASS_M78_EXACT_INT8_PWP_WIDTH_AND_BLOCK_ESCAPE_DSE_INTERNAL_ONLY",
            "M78 status")
    require(m83["status"] == "PASS_M83_CANONICAL_BINARY_ALL_ENTRIES_ROUNDTRIP",
            "M83 status")
    require(m88["status"] ==
            "PASS_M88_BOUNDED_MODULE_CYCLE_SIM_VALID825_INTERNAL_ONLY",
            "M88 status")
    require(m83["files"]["phase_records"]["sha256"] ==
            EXPECTED_SHA256["m83_records"] and
            m83["files"]["phase_offsets"]["sha256"] ==
            EXPECTED_SHA256["m83_offsets"], "M83 binary receipt")

    cap11 = next(row for row in m78["configurations"]
                 if row["signed_width_cap"] == 11)
    held = cap11["heldout"]
    shared32 = next(row for row in cap11["cycle_simulations"]
                    if row["port"] == "SHARED_32B")
    outliers = m78["pwp_precision"]["required_12bit_outliers"]
    require(len(outliers) == 1 and outliers[0]["operator_index"] == 2 and
            outliers[0]["partition"] == 378 and
            outliers[0]["pattern_index"] == 5 and
            outliers[0]["output_block"] == 5 and
            outliers[0]["center_hex"] == "ffff", "cap11 outlier")

    operator_index = dict((row["operator"], index)
                          for index, row in enumerate(m72["operators"]))
    heldout_records = [row for row in manifest["records"]
                       if row["sample_id"] >= 5]
    require(len(heldout_records) == 20, "heldout record count")

    total_events = 0
    total_weight_groups = 0
    total_signed_groups = 0
    weight_group_size_histogram = Counter()
    signed_group_size_histogram = Counter()
    phase_weight_group_count_histogram = Counter()
    phase_signed_group_count_histogram = Counter()
    phase_event_count_histogram = Counter()
    record_summaries = []
    profile_cache = {}

    for record_number, record in enumerate(heldout_records, 1):
        op = operator_index[record["operator"]]
        masks = decode_support_masks(record)
        record_events = 0
        record_weight_groups = 0
        record_signed_groups = 0
        for partition in range(PARTITIONS):
            tile, subtile = divmod(partition, TILE_BITS // PARTITION_BITS)
            shift = subtile * PARTITION_BITS
            histogram = Counter(
                (masks[row * TILES + tile] >> shift) & 0xffff
                for row in range(ROWS))
            centers = [int(value, 16) for value in
                       m72["operators"][op]["partitions"][partition][
                           "centers_hex"]]
            weight_counts = [0] * (16 * OUTPUT_BLOCKS)
            signed_counts = [0] * (16 * OUTPUT_BLOCKS * 2)
            cache = profile_cache.setdefault((op, partition), {})
            for value, count in histogram.items():
                profile = cache.get(value)
                if profile is None:
                    distance, center, center_index = nearest(value, centers)
                    beneficial = 1 + distance < POPCOUNT[value]
                    profile = []
                    for block in range(OUTPUT_BLOCKS):
                        eligible = not (op == 2 and partition == 378 and
                                        center_index == 5 and block == 5)
                        if beneficial and eligible:
                            bits = value ^ center
                            is_signed_correction = True
                        else:
                            bits = value
                            is_signed_correction = False
                        while bits:
                            low = bits & -bits
                            source = low.bit_length() - 1
                            subtract = bool(is_signed_correction and
                                            not (value & low))
                            weight_index = block * 16 + source
                            signed_index = weight_index * 2 + int(subtract)
                            profile.append((weight_index, signed_index))
                            bits ^= low
                    cache[value] = profile
                for weight_index, signed_index in profile:
                    weight_counts[weight_index] += count
                    signed_counts[signed_index] += count

            phase_events = sum(weight_counts)
            require(phase_events == sum(signed_counts), "phase conservation")
            active_weight_groups = [value for value in weight_counts if value]
            active_signed_groups = [value for value in signed_counts if value]
            total_events += phase_events
            total_weight_groups += len(active_weight_groups)
            total_signed_groups += len(active_signed_groups)
            record_events += phase_events
            record_weight_groups += len(active_weight_groups)
            record_signed_groups += len(active_signed_groups)
            weight_group_size_histogram.update(active_weight_groups)
            signed_group_size_histogram.update(active_signed_groups)
            phase_weight_group_count_histogram[len(active_weight_groups)] += 1
            phase_signed_group_count_histogram[len(active_signed_groups)] += 1
            phase_event_count_histogram[phase_events] += 1

        record_summaries.append({
            "sample_id": record["sample_id"],
            "operator_index": op,
            "operator": record["operator"],
            "correction_or_fallback_events": record_events,
            "phase_weight_groups": record_weight_groups,
            "phase_signed_groups": record_signed_groups,
        })
        print("[M103 RECONSTRUCT] {}/20 sample={} op={} events={}".format(
            record_number, record["sample_id"], op, record_events), flush=True)

    require(total_events == held["correction_ops_all_blocks"] == 188148490,
            "M78 correction/fallback event conservation")
    require(total_weight_groups == 8640 * 128 == 1105920 and
            phase_weight_group_count_histogram == {128: 8640},
            "all 128 weights used per phase")
    require(total_signed_groups == 1900560, "signed group total")
    require(shared32["candidate_cycles"] == 790689185 and
            m88["aggregate"]["bounded_candidate_cycles"] == 790706475,
            "M78/M88 candidate identity")

    def group_summary(histogram):
        groups = sum(histogram.values())
        events = sum(size * count for size, count in histogram.items())
        return {
            "groups": groups,
            "events": events,
            "singleton_groups": histogram[1],
            "maximum_group_size": max(histogram),
            "mean_group_size": events / float(groups),
            "p50_group_size": nearest_rank(histogram, 50, 100),
            "p95_group_size": nearest_rank(histogram, 95, 100),
            "p99_group_size": nearest_rank(histogram, 99, 100),
            "events_after_first_per_group": events - groups,
        }

    weight_summary = group_summary(weight_group_size_histogram)
    signed_summary = group_summary(signed_group_size_histogram)
    require(weight_summary == {
        "groups": 1105920, "events": 188148490, "singleton_groups": 0,
        "maximum_group_size": 695,
        "mean_group_size": 188148490 / float(1105920),
        "p50_group_size": 152, "p95_group_size": 344,
        "p99_group_size": 409, "events_after_first_per_group": 187042570,
    }, "weight group digest")
    require(signed_summary == {
        "groups": 1900560, "events": 188148490,
        "singleton_groups": 33528, "maximum_group_size": 681,
        "mean_group_size": 188148490 / float(1900560),
        "p50_group_size": 59, "p95_group_size": 291,
        "p99_group_size": 364, "events_after_first_per_group": 186247930,
    }, "signed group digest")

    bit_sparse_service = 1114383288
    pwp_service = 226222255
    pwp_events = held["pwp_ops_all_blocks"]
    correction_service = total_events * 3
    current_candidate_service = correction_service + pwp_service
    require(correction_service == 564445470 and
            current_candidate_service == 790667725 and
            pwp_events == 58969374, "M102 service ledger")

    weight_grouped_token_envelope = total_events + 2 * total_weight_groups
    signed_grouped_token_envelope = total_events + 2 * total_signed_groups
    one_token_per_event_envelope = total_events
    target = target_math(bit_sparse_service, current_candidate_service,
                         correction_service, pwp_service, pwp_events)

    output = {
        "schema": "m103_correction_service_reuse_preflight_audit_v1",
        "status": "ORDER_INDEPENDENT_REUSE_GROUPS_EXACT_ORDERED_CACHE_OR_MULTICAST_NOT_ADMITTED",
        "producer_modules_imported": False,
        "producer_or_simulator_executed": False,
        "sha256": observed,
        "evidence_coverage": {
            "m40_packed_sources": {
                "ordered_activity_masks": True,
                "sample_operator_timestep_y_x_identity": True,
                "partition_and_source_bits_derivable": True,
                "correction_issue_order": False,
                "output_block_service_order": False,
                "hardware_destination_tag_or_accumulator_bank": False,
            },
            "m72_json": {
                "per_operator_partition_centers": True,
                "aggregate_heldout_correction_counts": True,
                "event_sequence_source_block_tag": False,
            },
            "m78_json": {
                "aggregate_correction_fallback_and_pwp_counts": True,
                "pwp_width_use_counts": True,
                "event_sequence_source_block_tag": False,
                "reason": "phase_metrics consumes Counter histograms and destroys row order",
            },
            "m83_records": {
                "catalog_phase_pattern_block_pwp_payload": True,
                "heldout_use_or_correction_events": False,
                "source_or_destination_tags": False,
            },
            "m88_json": {
                "per_sample_phase_duration_and_prepare_schedule": True,
                "correction_issue_sequence_or_accumulator_mapping": False,
            },
        },
        "independent_reconstruction": {
            "heldout_records": 20,
            "phases": 8640,
            "partition_vectors": 25920000,
            "correction_or_fallback_events": total_events,
            "matches_m78": True,
            "record_summaries": record_summaries,
        },
        "order_independent_grouping": {
            "weight_identity": "operator,partition,source,output_block; phase/sample boundary invalidates values",
            "signed_identity": "weight identity plus add/subtract direction",
            "weight_groups": weight_summary,
            "signed_groups": signed_summary,
            "phase_weight_groups": {
                "minimum": min(phase_weight_group_count_histogram),
                "maximum": max(phase_weight_group_count_histogram),
                "all_8640_phases_exactly_128": True,
            },
            "phase_signed_groups": {
                "minimum": min(phase_signed_group_count_histogram),
                "maximum": max(phase_signed_group_count_histogram),
                "mean": total_signed_groups / 8640.0,
            },
            "phase_event_count": {
                "minimum": min(phase_event_count_histogram),
                "maximum": max(phase_event_count_histogram),
            },
        },
        "opportunity_boundaries_not_cycle_claims": {
            "last_vector_cache_actual_hit_count": None,
            "last_vector_cache_fail_closed_hit_interval": [
                0, weight_summary["events_after_first_per_group"]],
            "canonical_raster_block_then_source_adjacent_same_weight_hits": 0,
            "canonical_order_is_proposed_not_frozen": True,
            "all_weights_already_phase_resident_in_m88_narrow_buffer": True,
            "residency_alone_eliminates_32byte_service": False,
            "conditional_weight_grouped_one_line_hold_service_token_envelope":
                weight_grouped_token_envelope,
            "conditional_signed_grouped_one_line_hold_service_token_envelope":
                signed_grouped_token_envelope,
            "conditional_full_96byte_resident_one_token_per_event_envelope":
                one_token_per_event_envelope,
            "bank_multicast_actual_opportunity": None,
            "bank_multicast_fail_closed_eliminable_event_interval": [
                0, signed_summary["events_after_first_per_group"]],
            "why_not_admitted": [
                "no frozen correction issue order",
                "no destination tag to accumulator bank map",
                "no accumulator port/conflict model",
                "no proof that signed reordering preserves finite-width intermediate semantics",
                "no executable bounded reorder queue or combined candidate top"
            ],
        },
        "service_target_math_only": {
            "bit_sparse_service_denominator": bit_sparse_service,
            "current_correction_service": correction_service,
            "current_pwp_service": pwp_service,
            "current_candidate_service": current_candidate_service,
            "pwp_events_one_token_floor": pwp_events,
            "pwp_one_token_floor_candidate_with_correction_unchanged":
                correction_service + pwp_events,
            "pwp_deleted_candidate_with_correction_unchanged":
                correction_service,
            "targets": target,
        },
        "admission": {
            "exact_order_independent_correction_multiset": True,
            "exact_phase_source_block_group_population": True,
            "last_vector_cache_hit_rate": False,
            "run_length_or_broadcast_schedule": False,
            "bank_multicast": False,
            "cycle_reduction": False,
            "physical_or_system_speedup": False,
            "headline": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M103 independent correction/fallback reuse preflight")
    print("events={} weight_groups={} signed_groups={}".format(
        total_events, total_weight_groups, total_signed_groups))
    print("weight_group p50={} p95={} p99={} max={}".format(
        weight_summary["p50_group_size"], weight_summary["p95_group_size"],
        weight_summary["p99_group_size"], weight_summary["maximum_group_size"]))
    print("ordered_cache=false multicast=false cycle_reduction=false")
    print("target_2x={} target_2p5x={}".format(
        target["2p0x"]["candidate_service_target_max"],
        target["2p5x"]["candidate_service_target_max"]))
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
