#!/usr/bin/env python3
"""Independent, fail-closed validator for the M40a-r3 hammer review.

The core trace and arithmetic checks below do not call the M40 analyzer.  The
candidate module is loaded only after those checks to reproduce adversarial
accept/reject behavior recorded as review findings.
"""

from __future__ import print_function

import argparse
import copy
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_REVIEW = HW_ROOT / (
    "results/m40_conflict_aware_event_schedule_r3_20260822/"
    "m40a_r3_independent_hammer_review.json")

ANCHORS = {
    "contract": (
        "contracts/m40_conflict_aware_event_schedule_contract_r1_20260822.json",
        "1eeeea8f1778f45305226dbccf31a920586dff3eb14ee0bf684ef833728f9018"),
    "analyzer": (
        "system_simulator/scripts/analyze_m40_conflict_aware_event_schedule.py",
        "dd6dc32f773d8aa8c095173d51b4b182cb7cead3e8d0e8e3076ed7cb76fba372"),
    "regression": (
        "system_simulator/tests/test_m40_conflict_aware_event_schedule.py",
        "e85e877448641dca1cd1acde8e87f96c4e0a4688cb65e45affd1291bc02dcb0f"),
    "result": (
        "results/m40_conflict_aware_event_schedule_r3_20260822/"
        "m40_conflict_aware_event_schedule.json",
        "419ea51faabda4c2f45b9fa535d1a0fa8142bb4c8b8258468e88a1dc99c310e7"),
    "specification": (
        "rtl_m40/M40_AMPLITUDE_CODEBOOK_EVENT_MILESTONE_R1.md",
        "d93380d0105849c2736b63fb10f24b94eeeca9751f72d3dbcd77e174564d874c"),
    "packed_source_manifest": (
        "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
        "m40_bottleneck_packed_source_manifest.json",
        "e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3"),
    "tracer": (
        "system_simulator/scripts/trace_m40_bottleneck_packed_sources.py",
        "b02ac10fb95e68fa2871b74330d6f39d7d3d8cbfa6440990d43ec832e943bf19"),
    "m35_result": (
        "results/m35_complement_csd_r3_20260822/m35_complement_csd.json",
        "c47121f7d9b9fef15f4f1d770c4944d0bef9f640e5c9e4d522e6529742687869"),
}

TARGETS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
EXPECTED_AMPLITUDES = {
    TARGETS[0]: ("3f7fff87", 16777095, 121,
                 "sttmultires_unet.resblocks.0.sn1.spiking_neuron",
                 ((1, 0), (-1, 3), (1, 7))),
    TARGETS[1]: ("3f7fff70", 16777072, 144,
                 "sttmultires_unet.resblocks.0.sn2.spiking_neuron",
                 ((1, 4), (1, 7))),
    TARGETS[2]: ("3f7fff9f", 16777119, 97,
                 "sttmultires_unet.resblocks.1.sn1.spiking_neuron",
                 ((1, 0), (-1, 5), (1, 7))),
    TARGETS[3]: ("3f7ffdb4", 16776628, 588,
                 "sttmultires_unet.resblocks.1.sn2.spiking_neuron",
                 ((-1, 2), (1, 4), (1, 6), (1, 9))),
}
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
    raise ValueError("non-standard JSON constant: {}".format(raw))


def read_json(path):
    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook,
                      parse_constant=reject_constant)


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys),
            "{} key population drift".format(label))


def contained_file(base, raw, label):
    base = Path(base).resolve()
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError("{} escapes trace directory".format(label))
    require(candidate.is_file(), "{} missing".format(label))
    return candidate


def validate_anchors():
    observed = {}
    for name, pair in sorted(ANCHORS.items()):
        path = HW_ROOT / pair[0]
        require(path.is_file() and sha256(path) == pair[1],
                "M40a anchor drift: {}".format(name))
        observed[name] = pair[1]
    observed["independent_validator"] = sha256(Path(__file__).resolve())
    return observed


def minimum_signed_power_terms(value):
    for count in range(5):
        for shifts in itertools.combinations(range(13), count):
            for signs in itertools.product((-1, 1), repeat=count):
                if sum(sign * (1 << shift)
                       for sign, shift in zip(signs, shifts)) == value:
                    return count
    raise ValueError("no <=4-term CSD solution")


def independent_trace_rebuild():
    manifest_path = HW_ROOT / ANCHORS["packed_source_manifest"][0]
    trace_dir = manifest_path.parent.resolve()
    manifest = read_json(manifest_path)
    result = read_json(HW_ROOT / ANCHORS["result"][0])
    m35 = read_json(HW_ROOT / ANCHORS["m35_result"][0])
    require(manifest["schema"] == "m40_bottleneck_packed_source_trace_v1",
            "manifest schema drift")
    require(manifest["cohort"]["samples"] == 10 and
            manifest["cohort"]["records"] == 40 and
            manifest["cohort"]["operators"] == list(TARGETS),
            "manifest cohort drift")

    result_rows = {}
    for row in result["real_source_trace"]["records"]:
        key = (row["sample_id"], row["operator"])
        require(key not in result_rows, "duplicate result sample/operator")
        result_rows[key] = row
    require(len(result_rows) == 40, "result row population drift")

    popcount = [bin(value).count("1") for value in range(256)]
    spatial = 15 * 20
    period = spatial // math.gcd(8, spatial)
    weighted = []
    for phase in range(period):
        base = (phase * 8) % spatial
        phase_values = []
        for value in range(256):
            destinations = 0
            for bit in range(8):
                if value & (1 << bit):
                    y, x = divmod((base + bit) % spatial, 20)
                    destinations += ((2 if y in (0, 14) else 3) *
                                     (2 if x in (0, 19) else 3))
            phase_values.append(destinations)
        weighted.append(phase_values)

    seen = set()
    codes = {operator: set() for operator in TARGETS}
    per_sample = {line: [0] * 10 for line in ("Local", "Motion")}
    total_values = 0
    total_changed_bits = 0
    total_positive_bytes = 0
    total_three_plane_bytes = 0
    total_compressed_value_bytes = 0
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key not in seen and key[0] in range(10) and key[1] in TARGETS,
                "manifest sample/operator population drift")
        seen.add(key)
        require(record["shape"] == [10, 1, 768, 15, 20] and
                record["output_shape"] == [10, 1, 768, 15, 20] and
                record["elements"] == 2304000,
                "record shape/element drift")
        geometry = record["module_geometry"]
        require(geometry["kernel_size"] == [3, 3] and
                geometry["stride"] == [1, 1] and
                geometry["padding"] == [1, 1] and
                geometry["dilation"] == [1, 1] and
                geometry["groups"] == 1 and
                geometry["in_channels"] == 768 and
                geometry["out_channels"] == 768 and
                geometry["bias_present"] is False,
                "record geometry/bias drift")

        packed_path = contained_file(trace_dir, record["packed_file"],
                                     "packed file")
        packed = packed_path.read_bytes()
        require(sha256(packed_path) == record["packed_file_sha256"],
                "packed SHA drift")
        plane_bytes = record["positive_plane_bytes"]
        require(plane_bytes == 288000 and len(packed) == 3 * plane_bytes ==
                record["packed_file_bytes"], "packed extent drift")
        positive = packed[:plane_bytes]
        negative = packed[plane_bytes:2 * plane_bytes]
        changed = packed[2 * plane_bytes:]
        require(not any(negative), "negative population is nonzero")

        codebook = record["value_bit_pattern_population"]
        require(codebook["unique_float32_bit_patterns"] == 2 and
                codebook["full_codebook_in_manifest"] is True and
                type(codebook["codebook"]) is list and
                len(codebook["codebook"]) == 2,
                "two-code declaration drift")
        code_counts = {}
        for item in codebook["codebook"]:
            require(type(item["count"]) is int and item["count"] >= 0 and
                    item["float32_bits_hex"] not in code_counts,
                    "codebook type/duplicate drift")
            code_counts[item["float32_bits_hex"]] = item["count"]
        require("00000000" in code_counts and len(code_counts) == 2,
                "two-code population drift")
        nonzero_bits = next(value for value in code_counts
                            if value != "00000000")
        amplitude_word = struct.pack("<I", int(nonzero_bits, 16))
        amplitude = struct.unpack("<f", amplitude_word)[0]
        amplitude_raw = int(round(amplitude * (1 << 24)))
        require(amplitude == amplitude_raw / float(1 << 24),
                "amplitude is not exact UQ0.24")
        codes[key[1]].add((nonzero_bits, amplitude_raw))

        value_path = contained_file(trace_dir, record["value_payload_file"],
                                    "value payload")
        compressed = value_path.read_bytes()
        require(hashlib.sha256(compressed).hexdigest() ==
                record["value_payload_sha256"], "value payload SHA drift")
        inflater = zlib.decompressobj()
        value_raw = inflater.decompress(compressed) + inflater.flush()
        require(inflater.eof and not inflater.unused_data and
                not inflater.unconsumed_tail,
                "value payload is not one canonical zlib stream")
        require(len(value_raw) == record["input_content_bytes"] == 9216000 and
                hashlib.sha256(value_raw).hexdigest() ==
                record["input_content_sha256"], "decompressed value drift")
        decode = tuple(b"".join(
            amplitude_word if byte & (1 << bit) else b"\x00\x00\x00\x00"
            for bit in range(8)) for byte in range(256))
        reconstructed = b"".join(decode[byte] for byte in positive)
        require(reconstructed == value_raw,
                "independent 92.16M float32 bit miter mismatch")

        positive_count = sum(popcount[byte] for byte in positive)
        require(positive_count == record["positive_count"] ==
                record["nonzero_count"] == code_counts[nonzero_bits] and
                code_counts["00000000"] + positive_count == record["elements"],
                "bitmap/codebook count conservation mismatch")
        require(record["value_audit"]["noninteger_count"] == positive_count and
                record["value_audit"]["all_values_integer"] is False and
                record["value_audit"]["all_values_ternary"] is False,
                "noninteger audit drift")

        bytes_per_timestep = plane_bytes // 10
        local_sources = local_pairs = motion_sources = motion_pairs = 0
        direction_sources = {-1: 0, 1: 0}
        direction_pairs = {-1: 0, 1: 0}
        for timestep in range(10):
            start = timestep * bytes_per_timestep
            previous_start = (timestep - 1) * bytes_per_timestep
            local_timestep = motion_timestep = 0
            for offset, current in enumerate(
                    positive[start:start + bytes_per_timestep]):
                previous = (0 if timestep == 0 else
                            positive[previous_start + offset])
                expected_changed = current ^ previous
                require(changed[start + offset] == expected_changed,
                        "changed plane is not adjacent-timestep XOR")
                phase = offset % period
                local_sources += popcount[current]
                local_pairs += weighted[phase][current]
                local_timestep += popcount[current]
                motion_sources += popcount[expected_changed]
                motion_pairs += weighted[phase][expected_changed]
                motion_timestep += popcount[expected_changed]
                rise = current & (~previous & 0xff)
                fall = (~current & 0xff) & previous
                direction_sources[1] += popcount[rise]
                direction_sources[-1] += popcount[fall]
                direction_pairs[1] += weighted[phase][rise]
                direction_pairs[-1] += weighted[phase][fall]
                total_changed_bits += 8
            require(local_timestep ==
                    record["local_nonzero_count_by_timestep"][timestep] and
                    motion_timestep ==
                    record["motion_numeric_transition_count_by_timestep"][timestep],
                    "per-timestep support drift")
        require(local_sources == positive_count and
                direction_sources[-1] + direction_sources[1] == motion_sources and
                direction_pairs[-1] + direction_pairs[1] == motion_pairs,
                "Local/Motion conservation mismatch")

        candidate = result_rows[key]
        expected_local = (local_sources, local_pairs, local_pairs * 768,
                          local_pairs * 8)
        observed_local = (
            candidate["Local"]["active_sources"],
            candidate["Local"]["source_destination_pairs"],
            candidate["Local"]["active_products"],
            candidate["Local"]["exact_96_lane_product_lower_bound_cycles"])
        expected_motion = (motion_sources, motion_pairs, motion_pairs * 768,
                           motion_pairs * 8)
        observed_motion = (
            candidate["Motion"]["active_support_transitions"],
            candidate["Motion"]["source_destination_pairs"],
            candidate["Motion"]["active_products"],
            candidate["Motion"]["exact_96_lane_product_lower_bound_cycles"])
        require(expected_local == observed_local and
                expected_motion == observed_motion,
                "result product expansion drift")
        for direction in (-1, 1):
            direction_row = candidate["Motion"]["direction"][str(direction)]
            require((direction_sources[direction], direction_pairs[direction],
                     direction_pairs[direction] * 768) ==
                    (direction_row["support_transitions"],
                     direction_row["source_destination_pairs"],
                     direction_row["active_products"]),
                    "Motion direction expansion drift")
        for direction in (-2, 2):
            require(all(value == 0 for value in
                        candidate["Motion"]["direction"][str(direction)].values()),
                    "unexpected +/-2 direction population")
        per_sample["Local"][key[0]] += local_pairs * 8
        per_sample["Motion"][key[0]] += motion_pairs * 8
        total_values += record["elements"]
        total_positive_bytes += plane_bytes
        total_three_plane_bytes += len(packed)
        total_compressed_value_bytes += len(compressed)

    require(seen == set((sample, operator) for sample in range(10)
                        for operator in TARGETS),
            "40-record Cartesian cohort is incomplete")
    require(total_values == total_changed_bits == 92160000,
            "full bit-miter population drift")

    m35_rows = {row["producer"]: row for row in m35["thresholds"]}
    for operator, expected in EXPECTED_AMPLITUDES.items():
        bits, raw, delta, producer, terms = expected
        require(codes[operator] == {(bits, raw)} and (1 << 24) - raw == delta,
                "operator amplitude mapping drift")
        m35_row = m35_rows[producer]
        observed_terms = tuple((item["coefficient"], item["shift"])
                               for item in m35_row["csd_terms"])
        require(m35_row["threshold_uq0p24_raw"] == raw and
                m35_row["delta"] == delta and observed_terms == terms and
                sum(sign * (1 << shift) for sign, shift in terms) == delta and
                minimum_signed_power_terms(delta) == len(terms),
                "independent M35 amplitude/CSD reconciliation failed")

    expected_distributions = {
        "Local": ([74514032, 73731376, 74254256, 73745520, 73417496,
                   73847856, 74995872, 73721232, 74040896, 74855240],
                  741123776, 73417496, 74995872),
        "Motion": ([110550816, 109632536, 110434912, 109509648, 108971224,
                    109343896, 110962768, 109257448, 109652160, 110832144],
                   1099147552, 108971224, 110962768),
    }
    for line, expected in expected_distributions.items():
        values, total, minimum, maximum = expected
        require(per_sample[line] == values and sum(values) == total and
                min(values) == minimum and max(values) == maximum,
                "independent {} distribution drift".format(line))
        row = result["real_source_trace"][
            "exact_work_lower_bound_distribution_by_line"][line]
        require(row == {
            "count": 10, "minimum": minimum, "maximum": maximum,
            "mean_exact": {"numerator": total, "denominator": 10},
            "p95_nearest_rank": maximum, "p99_nearest_rank": maximum},
            "frozen {} distribution mismatch".format(line))

    require(result["real_source_trace"]["float32_values_bit_exact_mitered"] ==
            92160000 and
            result["real_source_trace"]["float32_value_bit_mismatches"] == 0,
            "frozen bit-miter summary drift")
    require(all(result["admission"][key] is False for key in FORBIDDEN) and
            result["real_source_trace"]["executable_cycle_mean_p95_p99"] ==
            {"Local": None, "Motion": None},
            "forbidden cycle/system claim opened")

    dense_bytes = total_values * 4
    bitmap_only = Fraction(dense_bytes, total_positive_bytes)
    bitmap_plus_words = Fraction(dense_bytes, total_positive_bytes + 16)
    three_planes_plus_words = Fraction(dense_bytes,
                                       total_three_plane_bytes + 16)
    require(bitmap_only == 32 and total_compressed_value_bytes == 7786309,
            "storage arithmetic drift")
    return {
        "records": 40,
        "unique_sample_operator_pairs": 40,
        "float32_values_bit_mitered": total_values,
        "float32_value_bit_mismatches": 0,
        "changed_plane_bits_recomputed": total_changed_bits,
        "changed_plane_bit_mismatches": 0,
        "operator_static_amplitudes": 4,
        "m35_csd_global_minimum_mappings": 4,
        "dense_float32_bytes": dense_bytes,
        "positive_bitmap_bytes": total_positive_bytes,
        "three_plane_trace_bytes": total_three_plane_bytes,
        "compressed_float_payload_bytes": total_compressed_value_bytes,
        "bitmap_only_reduction_exact": {
            "numerator": bitmap_only.numerator,
            "denominator": bitmap_only.denominator},
        "bitmap_plus_four_u32_words_reduction_exact": {
            "numerator": bitmap_plus_words.numerator,
            "denominator": bitmap_plus_words.denominator},
        "three_planes_plus_four_u32_words_reduction_exact": {
            "numerator": three_planes_plus_words.numerator,
            "denominator": three_planes_plus_words.denominator},
        "local_product_lower_bound_cycles_per_sample": per_sample["Local"],
        "motion_product_lower_bound_cycles_per_sample": per_sample["Motion"],
        "local_cycle_lower_bound_sum": sum(per_sample["Local"]),
        "motion_cycle_lower_bound_sum": sum(per_sample["Motion"]),
        "local_active_products_sum": sum(per_sample["Local"]) * 96,
        "motion_active_products_sum": sum(per_sample["Motion"]) * 96,
        "motion_over_local_product_lower_bound_exact": {
            "numerator": Fraction(sum(per_sample["Motion"]),
                                  sum(per_sample["Local"])).numerator,
            "denominator": Fraction(sum(per_sample["Motion"]),
                                    sum(per_sample["Local"])).denominator},
        "padding_geometry": "PASS_DIRECT_VALID_DESTINATION_MULTIPLICITY",
    }


def load_candidate():
    path = HW_ROOT / ANCHORS["analyzer"][0]
    spec = importlib.util.spec_from_file_location("m40a_hammer_candidate", str(path))
    require(spec is not None and spec.loader is not None,
            "candidate module import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adversarial_rebuild():
    module = load_candidate()
    canonical = module.read_json(module.DEFAULT_CONTRACT)
    rejected = 0
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        mutations = []
        item = copy.deepcopy(canonical)
        item["scheduler_contract"]["banks"] = 24.0
        mutations.append(item)
        item = copy.deepcopy(canonical)
        item["scheduler_contract"]["banks"] = True
        mutations.append(item)
        item = copy.deepcopy(canonical)
        item["inputs"]["packed_source_manifest"]["path"] = "../../etc/passwd"
        mutations.append(item)
        item = copy.deepcopy(canonical)
        item["inputs"]["packed_source_manifest"]["sha256"] = "0" * 64
        mutations.append(item)
        item = copy.deepcopy(canonical)
        item["extra"] = 0
        mutations.append(item)
        for index, forged in enumerate(mutations):
            path = directory / "forged_{}.json".format(index)
            path.write_text(json.dumps(forged), encoding="utf-8")
            try:
                module.build(path)
            except ValueError:
                rejected += 1
        require(rejected == len(mutations), "canonical forgery was accepted")

        duplicate = directory / "duplicate.json"
        raw_contract = module.DEFAULT_CONTRACT.read_text(encoding="utf-8")
        duplicate.write_text('{"schema":"forged",' + raw_contract.lstrip()[1:],
                             encoding="utf-8")
        duplicate_rejected = False
        try:
            module.build(duplicate)
        except ValueError as error:
            duplicate_rejected = "duplicate JSON key" in str(error)
        require(duplicate_rejected, "duplicate-key contract accepted")

        constants_rejected = 0
        for index, token in enumerate(("NaN", "Infinity", "-Infinity")):
            path = directory / "constant_{}.json".format(index)
            path.write_text(raw_contract.replace(
                '"banks": 24', '"banks": {}'.format(token), 1),
                encoding="utf-8")
            try:
                module.build(path)
            except ValueError as error:
                if "non-standard JSON" in str(error):
                    constants_rejected += 1
        require(constants_rejected == 3, "non-standard number was accepted")

        occupied = directory / "occupied.json"
        occupied.write_text("sentinel", encoding="utf-8")
        occupied_rejected = False
        try:
            module.write_output(occupied, {})
        except ValueError:
            occupied_rejected = occupied.read_text(encoding="utf-8") == "sentinel"
        require(occupied_rejected, "occupied output was overwritten")

        dangling_target = directory / "created_through_symlink.json"
        dangling = directory / "dangling.json"
        dangling.symlink_to(dangling_target)
        module.write_output(dangling, {"probe": 1})
        require(dangling_target.is_file(),
                "expected dangling-symlink output weakness disappeared")

        manifest = read_json(HW_ROOT / ANCHORS["packed_source_manifest"][0])
        source_dir = (HW_ROOT / ANCHORS["packed_source_manifest"][0]).parent
        record = copy.deepcopy(manifest["records"][0])
        packed = bytearray((source_dir / record["packed_file"]).read_bytes())
        changed_index = 2 * record["positive_plane_bytes"]
        old_bit = packed[changed_index] & 1
        packed[changed_index] ^= 1
        record["motion_numeric_transition_count_by_timestep"][0] += (
            -1 if old_bit else 1)
        record["packed_file"] = "forged_changed.bin"
        record["packed_file_sha256"] = hashlib.sha256(packed).hexdigest()
        value_bytes = (source_dir / record["value_payload_file"]).read_bytes()
        record["value_payload_file"] = "value.zlib"
        (directory / record["packed_file"]).write_bytes(bytes(packed))
        (directory / record["value_payload_file"]).write_bytes(value_bytes)
        module.audit_packed_record(directory, record)

        events = copy.deepcopy(canonical["synthetic_oracle"]["events"])
        events[1]["accumulator_before_s32"] += 10
        events[1]["accumulator_after_s32"] += 10
        events[1]["expected_scaled_s56"] = (
            events[1]["accumulator_after_s32"] *
            events[1]["threshold_raw_uq0p24"])
        module.schedule_events(events,
                               canonical["synthetic_oracle"]["config"])

    return {
        "canonical_type_path_sha_population_forgeries": {
            "tested": 5, "rejected": rejected},
        "duplicate_json_keys": {"tested": 1, "rejected": 1},
        "nonstandard_json_numbers": {"tested": 3, "rejected": 3},
        "occupied_output": {"tested": 1, "rejected": 1},
        "dangling_symlink_output": {"tested": 1, "accepted": 1},
        "self_consistent_changed_plane_forgery_direct_record_audit": {
            "tested": 1, "accepted": 1},
        "disconnected_accumulator_chain_direct_scheduler": {
            "tested": 1, "accepted": 1},
    }


def validate_review(review_path=DEFAULT_REVIEW):
    anchors = validate_anchors()
    independent = independent_trace_rebuild()
    attacks = adversarial_rebuild()
    review = read_json(review_path)
    exact_keys(review, {
        "schema", "status", "date", "exact_anchors", "validator",
        "mandatory_rereview_passes", "findings", "review", "admitted",
        "claim_boundary", "next_gate"}, "review")
    require(review["schema"] == "m40a_r3_independent_hammer_review_v1" and
            review["status"] == "GO_M40A_R3_EXACT_TRACE_AND_ALGEBRA_ONLY",
            "review schema/status drift")
    for name, pair in ANCHORS.items():
        require(review["exact_anchors"][name] == [pair[0], pair[1]],
                "review anchor drift: {}".format(name))
    require(review["validator"] == [
        str(Path(__file__).resolve().relative_to(ROOT)),
        anchors["independent_validator"]], "review validator identity drift")
    require(review["mandatory_rereview_passes"]["independent_raw_rebuild"] ==
            independent, "review independent rebuild drift")
    require(review["mandatory_rereview_passes"]["adversarial_matrix"] ==
            attacks, "review attack matrix drift")
    require(review["mandatory_rereview_passes"]["python36_regression"] ==
            {"passed": 13, "failed": 0, "errors": 0},
            "Python3.6 regression receipt drift")
    require(review["review"] == {
        "decision": "GO_EXACT_TRACE_AND_ALGEBRA_ONLY",
        "independent_of_m40a_implementation": True,
        "score_0_to_100": 92,
        "p0": 0, "p1": 0, "p2": 5,
        "pass_admission_may_be_generated": True},
        "review decision/score drift")
    require(len(review["findings"]["p0"]) == 0 and
            len(review["findings"]["p1"]) == 0 and
            len(review["findings"]["p2"]) == 5,
            "review finding population drift")
    admitted = review["admitted"]
    require(admitted["exact_92160000_float32_value_bit_miter"] is True and
            admitted["bitmap_only_exact_32x_representation"] is True and
            admitted["bitmap_plus_static_words_exact_32x"] is False and
            admitted["real_executable_cycles"] is False and
            admitted["system_speedup"] is False and
            admitted["integrated_rtl_vcs_synopsys"] is False and
            admitted["ppa_power_energy"] is False,
            "review claim boundary drift")
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    review = validate_review(args.review.resolve())
    print("PASS {} score={}".format(
        review["review"]["decision"], review["review"]["score_0_to_100"]))


if __name__ == "__main__":
    main()
