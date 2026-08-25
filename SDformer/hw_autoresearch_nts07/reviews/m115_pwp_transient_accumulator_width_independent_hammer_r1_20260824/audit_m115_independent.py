#!/usr/bin/env python3
"""Independent payload, transient-coefficient, storage and claim audit for M115."""

import argparse
import hashlib
import itertools
import json
import re
import struct
from collections import Counter
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
M41_RESULT = M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json"
M108_ANALYZER = HW / "system_simulator/scripts/analyze_m108_w64_fused_pwp_accumulator_schedule.py"
M108_RESULT = HW / (
    "results/m108_w64_fused_pwp_accumulator_schedule_r1_20260824/"
    "m108_w64_fused_pwp_accumulator_schedule.json")
M112_RECEIPT = HW / (
    "dc_handoff/runs/m112_w384_lane_sliced_accumulator_vcs_r1_sealed_20260824/"
    "RUN_COMPLETE.txt")
M114_RESULT = HW / (
    "results/m114_storage_valid_admission_correction_r1_20260824/"
    "m114_storage_valid_admission_correction.json")
M115_ANALYZER = HW / "system_simulator/scripts/analyze_m115_pwp_transient_accumulator_width.py"
M115_RESULT = HW / (
    "results/m115_pwp_transient_accumulator_width_r1_20260824/"
    "m115_pwp_transient_accumulator_width.json")
M115_CONTRACT = HW / "contracts/m115_pwp_transient_accumulator_width_contract_r1_20260824.json"
M115_MANIFEST = HW / "results/m115_pwp_transient_accumulator_width_r1_20260824/SHA256SUMS.txt"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m108_analyzer": "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m108_result": "358640e62c2e52f859b7143f0bac957d6988ed1bd7c56e5dd54d21bc01344318",
    "m112_receipt": "458dc8af156165bf726d36a57813d2d476ec25dded82ffdee077c186f63bba26",
    "m114_result": "1559c65779fbc15026b3d744e3f1463bba8effd13c2efaa04e8562d4dbfb2226",
    "m115_analyzer": "bafadcf53e5221d70ab86da0fb17dcbae8da661b0148007dbd537f4fa519aa27",
    "m115_result": "9f62d9cb3e56c293cc117bd92c21844e8bd10515ea418a51cbfae0ebab62b94b",
    "m115_contract": "ba730fcb6612fd8aa5c8e8c7d1aba976b759de54cbab05779ca409dadf9af9c8",
    "m115_manifest": "bb12196b1ed7e0c10cb6b41db85271db24bfefab62bf0058b194666353afc951",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
WEIGHT_SHA = [
    "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
]
FEATURES = 768 * 3 * 3
CHANNELS = 768
WINDOW_ROWS = 384
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def strict_loads(text):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=reject)


def strict_json(path):
    return strict_loads(Path(path).read_text(encoding="utf-8"))


def parse_manifest_text(text, base, allow_absolute=False):
    entries = []
    seen = set()
    for number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed manifest line {}".format(number))
        expected, raw_path = match.groups()
        path = Path(raw_path)
        require(allow_absolute or not path.is_absolute(), "absolute manifest path")
        require(".." not in path.parts, "manifest traversal")
        require(raw_path not in seen, "duplicate manifest path")
        seen.add(raw_path)
        entries.append((expected, raw_path, path if path.is_absolute() else Path(base) / path))
    require(entries, "empty manifest")
    return entries


def verify_manifest(path, base, allow_absolute=False):
    entries = parse_manifest_text(Path(path).read_text(encoding="utf-8"),
                                  base, allow_absolute)
    failed = [raw for expected, raw, resolved in entries
              if not resolved.is_file() or sha256(resolved) != expected]
    return {
        "sha256": sha256(path),
        "entries": len(entries),
        "failed": failed,
        "listed_paths": [raw for _, raw, _ in entries],
    }


def expect_rejected(callable_obj, label):
    try:
        callable_obj()
    except (ValueError, json.JSONDecodeError):
        return True
    raise ValueError(label + " not rejected")


def signed_bits(magnitude):
    require(magnitude >= 0, "negative magnitude")
    for bits in range(2, 33):
        if magnitude <= (1 << (bits - 1)) - 1:
            return bits
    raise ValueError("magnitude too large")


def storage_row(bits):
    descriptor_bits = 2 * 128 * WINDOW_ROWS * 2
    metadata_bits = 314
    valid_bits = WINDOW_ROWS * OUTPUT_BLOCKS
    accumulator_bits = WINDOW_ROWS * OUTPUT_BLOCKS * OUTPUT_LANES * bits
    combined_bits = descriptor_bits + metadata_bits + valid_bits + accumulator_bits
    return {
        "accumulator_signed_bits": bits,
        "accumulator_payload_bits": accumulator_bits,
        "accumulator_payload_bytes": accumulator_bits // 8,
        "fixed_descriptor_metadata_valid_bits": descriptor_bits + metadata_bits + valid_bits,
        "combined_bits": combined_bits,
        "combined_bytes_ceiling": (combined_bits + 7) // 8,
    }


def unique_permutations(values):
    return sorted(set(itertools.permutations(values)))


def term_case(eligible, center, target):
    if not eligible:
        operations = (1,) if target else ()
        route = "escape_raw_event"
    else:
        operations = (() if not center else (1,))
        if target and not center:
            operations += (1,)
        if center and not target:
            operations += (-1,)
        route = "pwp_anchor_plus_signed_correction"
    prefixes = []
    max_abs_prefix = 0
    for permutation in unique_permutations(operations):
        running = 0
        row = [0]
        for coefficient in permutation:
            running += coefficient
            row.append(running)
            max_abs_prefix = max(max_abs_prefix, abs(running))
        prefixes.append({"order": list(permutation), "prefix_coefficients": row})
    if not operations:
        prefixes = [{"order": [], "prefix_coefficients": [0]}]
    return {
        "eligible": bool(eligible),
        "center_bit": center,
        "target_bit": target,
        "route": route,
        "operations": list(operations),
        "absolute_operation_count": sum(abs(item) for item in operations),
        "maximum_absolute_prefix_coefficient_all_orders": max_abs_prefix,
        "orders": prefixes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing independent overwrite")

    identity_paths = {
        "m41_result": M41_RESULT,
        "m108_analyzer": M108_ANALYZER,
        "m108_result": M108_RESULT,
        "m112_receipt": M112_RECEIPT,
        "m114_result": M114_RESULT,
        "m115_analyzer": M115_ANALYZER,
        "m115_result": M115_RESULT,
        "m115_contract": M115_CONTRACT,
        "m115_manifest": M115_MANIFEST,
        "docs359": DOCS359,
    }
    identity = {key: sha256(path) for key, path in identity_paths.items()}
    require(identity == EXPECTED_SHA, "identity drift")

    m41 = strict_json(M41_RESULT)
    m108 = strict_json(M108_RESULT)
    m114 = strict_json(M114_RESULT)
    m115 = strict_json(M115_RESULT)
    contract = strict_json(M115_CONTRACT)
    require(m115["schema"] == "m115_pwp_transient_accumulator_width_result_v1",
            "M115 schema drift")

    strict_attacks = {
        "duplicate_json_key_rejected": expect_rejected(
            lambda: strict_loads('{"proof":{},"proof":{}}'), "duplicate JSON"),
        "nan_rejected": expect_rejected(
            lambda: strict_loads('{"bits":NaN}'), "NaN"),
        "infinity_rejected": expect_rejected(
            lambda: strict_loads('{"bits":Infinity}'), "Infinity"),
        "duplicate_manifest_path_rejected": expect_rejected(
            lambda: parse_manifest_text(
                "{}  a\n{}  a".format("0" * 64, "1" * 64), HW),
            "duplicate manifest path"),
        "malformed_manifest_hash_rejected": expect_rejected(
            lambda: parse_manifest_text("bad  a", HW), "bad manifest hash"),
        "manifest_traversal_rejected": expect_rejected(
            lambda: parse_manifest_text("{}  ../a".format("0" * 64), HW),
            "manifest traversal"),
        "payload_byte_mutation_changes_sha": True,
    }

    producer_manifest = verify_manifest(M115_MANIFEST, HW)
    require(not producer_manifest["failed"], "producer manifest failure")
    analyzer_inputs = {
        str(M41_RESULT.relative_to(HW)), str(M108_ANALYZER.relative_to(HW)),
        str(M108_RESULT.relative_to(HW)), str(M114_RESULT.relative_to(HW)),
        str(M112_RECEIPT.relative_to(HW)),
    }
    producer_manifest["missing_analyzer_inputs"] = sorted(
        analyzer_inputs - set(producer_manifest["listed_paths"]))
    producer_manifest["covers_all_analyzer_inputs"] = not producer_manifest[
        "missing_analyzer_inputs"]

    channel_ledgers = []
    summaries = []
    global_max = 0
    for operator, layer in enumerate(m41["layers"]):
        weight = next(row for row in layer["payloads"] if row["role"] == "weight")
        require(weight["shape"] == [768, 3, 3, 768], "weight shape drift")
        require(weight["layout"] == "I_KY_KX_O_C_ORDER", "weight layout drift")
        path = M41_DIR / weight["file"]
        data = path.read_bytes()
        require(len(data) == FEATURES * CHANNELS == 5308416,
                "weight extent drift op{}".format(operator))
        require(sha256(path) == WEIGHT_SHA[operator] == weight["sha256"],
                "weight SHA drift op{}".format(operator))
        strict_attacks["payload_byte_mutation_changes_sha"] &= (
            sha256_bytes(data[:-1] + bytes([data[-1] ^ 1])) != WEIGHT_SHA[operator])
        sums = [0] * CHANNELS
        value_min = 127
        value_max = -127
        negative_128 = 0
        for index, raw in enumerate(data):
            value = raw if raw < 128 else raw - 256
            value_min = min(value_min, value)
            value_max = max(value_max, value)
            negative_128 += int(value == -128)
            sums[index % CHANNELS] += abs(value)
        require(negative_128 == 0 and value_min >= -127 and value_max <= 127,
                "payload range drift op{}".format(operator))
        bound = layer["accumulator_bound"]
        require(min(sums) == bound["per_channel_sum_abs_q_minimum"]
                and max(sums) == bound["per_channel_sum_abs_q_maximum"],
                "M41 sumabs mismatch op{}".format(operator))
        producer_row = m115["operators"][operator]
        maximum = max(sums)
        maximum_channel = sums.index(maximum)
        require(producer_row["maximum_per_channel_sum_abs_q"] == maximum
                and producer_row["minimum_per_channel_sum_abs_q"] == min(sums)
                and producer_row["maximum_channel"] == maximum_channel,
                "M115 operator summary mismatch op{}".format(operator))
        ledger = []
        for channel, sumabs in enumerate(sums):
            ledger.append({
                "channel": channel,
                "sumabs": sumabs,
                "once_bound_required_signed_bits": signed_bits(sumabs),
                "twice_sumabs": 2 * sumabs,
                "loose_twice_bound_required_signed_bits": signed_bits(2 * sumabs),
            })
        ledger_bytes = b"".join(struct.pack("<I", value) for value in sums)
        once_distribution = Counter(row["once_bound_required_signed_bits"] for row in ledger)
        twice_distribution = Counter(row["loose_twice_bound_required_signed_bits"] for row in ledger)
        channel_ledgers.append({
            "operator_index": operator,
            "operator": layer["operator"],
            "sumabs_u32le_sha256": sha256_bytes(ledger_bytes),
            "channels": ledger,
        })
        summaries.append({
            "operator_index": operator,
            "operator": layer["operator"],
            "weight_payload_sha256": WEIGHT_SHA[operator],
            "minimum_sumabs": min(sums),
            "maximum_sumabs": maximum,
            "maximum_channel": maximum_channel,
            "mean_sumabs": sum(sums) / float(CHANNELS),
            "once_required_bits_distribution": {str(k): v for k, v in sorted(once_distribution.items())},
            "loose_twice_required_bits_distribution": {str(k): v for k, v in sorted(twice_distribution.items())},
            "negative_128_count": negative_128,
        })
        global_max = max(global_max, maximum)

    require(all(strict_attacks.values()), "strict attack failure")
    require(global_max == 218338 and signed_bits(global_max) == 19,
            "checkpoint once bound drift")
    require(2 * global_max == 436676 and signed_bits(2 * global_max) == 20,
            "checkpoint loose twice bound drift")

    init_payloads = []
    for operator, layer in enumerate(m41["layers"]):
        init = next(row for row in layer["payloads"] if row["role"] == "accumulator_init")
        path = M41_DIR / init["file"]
        data = path.read_bytes()
        require(len(data) == 3072 and set(data) <= {0},
                "nonzero accumulator init op{}".format(operator))
        require(sha256(path) == init["sha256"], "init SHA drift")
        init_payloads.append({"operator_index": operator, "sha256": sha256(path), "all_zero": True})

    truth_table = [term_case(eligible, center, target)
                   for eligible in (False, True)
                   for center in (0, 1)
                   for target in (0, 1)]
    maximum_absolute_operation_count = max(
        row["absolute_operation_count"] for row in truth_table)
    maximum_absolute_prefix_coefficient = max(
        row["maximum_absolute_prefix_coefficient_all_orders"] for row in truth_table)
    require(maximum_absolute_operation_count == 2,
            "abstract max operation count drift")
    require(maximum_absolute_prefix_coefficient == 1,
            "prefix coefficient should be bounded by one")

    dense_once = FEATURES * 127
    dense_loose_twice = 2 * dense_once
    require(dense_once == 877824 and signed_bits(dense_once) == 21,
            "dense once bound drift")
    require(dense_loose_twice == 1755648 and signed_bits(dense_loose_twice) == 22,
            "dense twice bound drift")

    storage = {bits: storage_row(bits) for bits in (19, 20, 21, 22, 24)}
    producer_storage = {row["accumulator_signed_bits"]: row
                        for row in m115["w384_storage_frontier"]}
    for bits, independent in storage.items():
        produced = producer_storage[bits]
        require(produced["accumulator_payload_bits"] == independent["accumulator_payload_bits"]
                and produced["accumulator_payload_bytes"] == independent["accumulator_payload_bytes"]
                and produced["combined_descriptor_valid_accumulator_bits"] == independent["combined_bits"]
                and produced["combined_bytes_ceiling_before_control_ecc_macro_rounding"]
                    == independent["combined_bytes_ceiling"],
                "storage mismatch signed{}".format(bits))
    require(storage[24]["combined_bytes_ceiling"] == 909736
            and storage[20]["combined_bytes_ceiling"] == 762280
            and storage[22]["combined_bytes_ceiling"] == 836008,
            "headline storage mismatch")

    signed20_saving = storage[24]["combined_bytes_ceiling"] - storage[20]["combined_bytes_ceiling"]
    signed22_saving = storage[24]["combined_bytes_ceiling"] - storage[22]["combined_bytes_ceiling"]
    signed19_saving = storage[24]["combined_bytes_ceiling"] - storage[19]["combined_bytes_ceiling"]
    signed21_saving = storage[24]["combined_bytes_ceiling"] - storage[21]["combined_bytes_ceiling"]
    require(signed20_saving == 147456 and signed22_saving == 73728,
            "producer saving mismatch")

    receipt = set(M112_RECEIPT.read_text(encoding="utf-8").splitlines())
    for line in ("status=PASS_M112_W384_LANE_SLICED_ACCUMULATOR_DIRECTED_VCS_SVA",
                 "signed_bits_per_lane=24", "foundry_sram_macro=false",
                 "exact_heldout_integrated_replay=false",
                 "physical_speedup=false", "system_speedup=false", "headline=false"):
        require(line in receipt, "M112 receipt missing " + line)
    require(m108["admission"]["source_coefficient_miter"] is True
            and m108["admission"]["full_lane_numeric_miter"] is False,
            "M108 admission drift")
    require(m108["work_conservation"]["source_coefficient_checks"] == 3317760000,
            "M108 coefficient extent drift")
    require(next(row for row in m114["frontier"] if row["window_rows"] == 384)
            ["storage_lower_bound_corrected"]
            ["combined_bytes_ceiling_before_control_ecc_macro_rounding"] == 909736,
            "M114 bridge drift")

    contract_claims = " ".join(contract["prohibited_claims"]).lower()
    for token in ("signed19", "rtl", "vcs", "cycle", "physical", "system",
                  "headline", "macro area", "energy"):
        require(token in contract_claims, "missing claim token " + token)
    for key in ("signed20_accumulator_rtl", "signed20_full_lane_numeric_vcs",
                "foundry_sram_macro", "macro_inclusive_ppa", "cycle_reduction",
                "physical_speedup", "system_speedup", "headline"):
        require(m115["admission"][key] is False, "M115 over-admits " + key)

    payload = {
        "schema": "m115_pwp_transient_accumulator_width_independent_audit_v1",
        "status": "P0_REQUIRED_WIDTH_AND_SIGNED19_REJECTION_NOT_PROVED_SAFE_LOOSE_BOUNDS_AND_STORAGE_REPRODUCED",
        "identity": identity,
        "strict_attacks": strict_attacks,
        "producer_manifest": producer_manifest,
        "payload_recomputation": {
            "operators": summaries,
            "full_per_channel_ledgers": channel_ledgers,
            "all_3072_channels_recomputed": True,
            "checkpoint_maximum_sumabs": global_max,
            "checkpoint_once_required_signed_bits": signed_bits(global_max),
            "checkpoint_loose_twice_magnitude": 2 * global_max,
            "checkpoint_loose_twice_required_signed_bits": signed_bits(2 * global_max),
            "accumulator_init_payloads": init_payloads,
        },
        "multiplicity_and_order_attack": {
            "truth_table": truth_table,
            "maximum_absolute_operation_count_per_term": maximum_absolute_operation_count,
            "maximum_absolute_prefix_coefficient_per_term_all_orders": maximum_absolute_prefix_coefficient,
            "pwp_positive_negative_relation": "positive only when center=0,target=1; negative only when center=1,target=0",
            "escape_relation": "no PWP anchor; raw target event occurs at most once",
            "service_reordering_result": "arbitrary permutations of each term's legal operations keep prefix coefficient in {-1,0,+1}",
            "duplicate_or_retry_attack": "if an accepted anchor/correction can be replayed, the exact-once premise fails; unbounded retries also invalidate the 2x bound",
            "two_sumabs_is_safe": True,
            "two_sumabs_is_tight_or_required": False,
            "one_sumabs_is_sufficient_under_same_exact_once_coefficient_contract": True,
            "signed19_direct_inheritance_rejected_by_current_evidence": False,
        },
        "dense_envelope": {
            "quantized_range": [-127, 127],
            "source_terms": FEATURES,
            "once_magnitude": dense_once,
            "once_required_signed_bits": signed_bits(dense_once),
            "loose_twice_magnitude": dense_loose_twice,
            "loose_twice_required_signed_bits": signed_bits(dense_loose_twice),
            "signed22_is_safe": True,
            "signed22_is_required_under_exact_once_contract": False,
        },
        "storage": {
            "rows": [storage[bits] for bits in sorted(storage)],
            "signed24_combined_bytes": storage[24]["combined_bytes_ceiling"],
            "producer_signed20_combined_bytes": storage[20]["combined_bytes_ceiling"],
            "producer_signed20_saved_bytes": signed20_saving,
            "producer_signed20_saved_fraction": signed20_saving / 909736.0,
            "producer_dense_signed22_combined_bytes": storage[22]["combined_bytes_ceiling"],
            "producer_dense_signed22_saved_bytes": signed22_saving,
            "producer_dense_signed22_saved_fraction": signed22_saving / 909736.0,
            "exact_once_checkpoint_signed19_combined_bytes": storage[19]["combined_bytes_ceiling"],
            "exact_once_checkpoint_signed19_saved_bytes": signed19_saving,
            "exact_once_checkpoint_signed19_saved_fraction": signed19_saving / 909736.0,
            "exact_once_dense_signed21_combined_bytes": storage[21]["combined_bytes_ceiling"],
            "exact_once_dense_signed21_saved_bytes": signed21_saving,
            "exact_once_dense_signed21_saved_fraction": signed21_saving / 909736.0,
            "all_producer_storage_and_saving_arithmetic_reproduced": True,
        },
        "claim_boundary": {
            "software_arithmetic_bound_only": True,
            "signed20_rtl": False,
            "signed20_commercial_vcs": False,
            "foundry_macro": False,
            "macro_inclusive_ppa": False,
            "cycle_reduction": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("P0 M115 independent channels=3072 max_sumabs={} prefix_coeff={} signed19={} signed20_safe={} W20={} W22={}".format(
        global_max, maximum_absolute_prefix_coefficient, signed_bits(global_max),
        signed_bits(2 * global_max), storage[20]["combined_bytes_ceiling"],
        storage[22]["combined_bytes_ceiling"]), flush=True)


if __name__ == "__main__":
    main()
