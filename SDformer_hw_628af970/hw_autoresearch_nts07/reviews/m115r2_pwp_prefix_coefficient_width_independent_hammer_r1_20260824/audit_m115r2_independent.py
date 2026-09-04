#!/usr/bin/env python3
"""Independent M115-r2 prefix, payload, storage, identity and boundary audit."""

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
PATHS = {
    "m41_result": M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json",
    "m108_analyzer": HW / "system_simulator/scripts/analyze_m108_w64_fused_pwp_accumulator_schedule.py",
    "m108_result": HW / "results/m108_w64_fused_pwp_accumulator_schedule_r1_20260824/m108_w64_fused_pwp_accumulator_schedule.json",
    "m114_result": HW / "results/m114_storage_valid_admission_correction_r1_20260824/m114_storage_valid_admission_correction.json",
    "m115_r1_analyzer": HW / "system_simulator/scripts/analyze_m115_pwp_transient_accumulator_width.py",
    "m115_r1_result": HW / "results/m115_pwp_transient_accumulator_width_r1_20260824/m115_pwp_transient_accumulator_width.json",
    "m115_r1_contract": HW / "contracts/m115_pwp_transient_accumulator_width_contract_r1_20260824.json",
    "m115_r1_revocation": HW / "contracts/m115_r1_transient_width_claim_revocation_r1_20260824.json",
    "trigger_review": HW / "reviews/m115_pwp_transient_accumulator_width_independent_hammer_r1_20260824/m115_pwp_transient_accumulator_width_independent_hammer_review.json",
    "trigger_manifest": HW / "reviews/m115_pwp_transient_accumulator_width_independent_hammer_r1_20260824/manifest.sha256",
    "m115r2_analyzer": HW / "system_simulator/scripts/analyze_m115r2_pwp_prefix_coefficient_width.py",
    "m115r2_result": HW / "results/m115r2_pwp_prefix_coefficient_width_r1_20260824/m115r2_pwp_prefix_coefficient_width.json",
    "m115r2_contract": HW / "contracts/m115r2_pwp_prefix_coefficient_width_contract_r1_20260824.json",
    "m115r2_manifest": HW / "results/m115r2_pwp_prefix_coefficient_width_r1_20260824/SHA256SUMS.complete_r1.txt",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED_SHA = {
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m108_analyzer": "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m108_result": "358640e62c2e52f859b7143f0bac957d6988ed1bd7c56e5dd54d21bc01344318",
    "m114_result": "1559c65779fbc15026b3d744e3f1463bba8effd13c2efaa04e8562d4dbfb2226",
    "m115_r1_analyzer": "bafadcf53e5221d70ab86da0fb17dcbae8da661b0148007dbd537f4fa519aa27",
    "m115_r1_result": "9f62d9cb3e56c293cc117bd92c21844e8bd10515ea418a51cbfae0ebab62b94b",
    "m115_r1_contract": "ba730fcb6612fd8aa5c8e8c7d1aba976b759de54cbab05779ca409dadf9af9c8",
    "m115_r1_revocation": "120d0faae82cec9d434fbdc66083bc87d1f6264840a79f142010ba39e2a536a9",
    "trigger_review": "461c7b78ca6f0b4132ba8ccd22833d42e5df85041c5429cceebafdafa6a18d64",
    "trigger_manifest": "14192c606f31e980016712a00c390dcc742a27dd5f0ed483fc31473009f5f287",
    "m115r2_analyzer": "2f3512f2c664daea6430c1360838c7496228b49ae2dd5a648db9af361fbf0f31",
    "m115r2_result": "b0e7fbb0573473ad854ca856d5eab3eaf15af1ba79ea2ce3a958810575bc6708",
    "m115r2_contract": "9edd6aac10186e24f21fffa5ce1b5a28da292258ad30df1d6934a7b1d1927eec",
    "m115r2_manifest": "6b9af5e9e7de61edc770e1d4d738d6c0b0070e7947f6aec12633da7181f96326",
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
W = 384
BLOCKS = 8
LANES = 96


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
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=reject)


def strict_json(path):
    return strict_loads(Path(path).read_text(encoding="utf-8"))


def parse_manifest_text(text, base):
    entries = []
    seen = set()
    for number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed manifest line {}".format(number))
        expected, raw = match.groups()
        path = Path(raw)
        require(not path.is_absolute(), "absolute manifest path")
        require(".." not in path.parts, "manifest traversal")
        require(raw not in seen, "duplicate manifest path")
        seen.add(raw)
        entries.append((expected, raw, Path(base) / path))
    require(entries, "empty manifest")
    return entries


def attack_raises(function):
    try:
        function()
    except (ValueError, json.JSONDecodeError):
        return True
    return False


def signed_bits(magnitude):
    require(magnitude >= 0, "negative magnitude")
    for bits in range(2, 64):
        if magnitude <= (1 << (bits - 1)) - 1:
            return bits
    raise ValueError("magnitude too large")


def named_permutations(operations):
    if not operations:
        return [()]
    return sorted(set(itertools.permutations(operations)))


def case_row(eligible, center, target):
    if eligible:
        operations = []
        if center:
            operations.append(("anchor", 1))
        if target and not center:
            operations.append(("positive_correction", 1))
        if center and not target:
            operations.append(("negative_correction", -1))
        route = "pwp"
    else:
        operations = [("raw_target", 1)] if target else []
        route = "escape"
    orders = []
    maximum = 0
    for permutation in named_permutations(tuple(operations)):
        prefixes = [0]
        for _, coefficient in permutation:
            prefixes.append(prefixes[-1] + coefficient)
        maximum = max(maximum, *(abs(value) for value in prefixes))
        orders.append({
            "operation_order": [name for name, _ in permutation],
            "prefix_coefficients": prefixes,
        })
        require(prefixes[-1] == target, "final coefficient mismatch")
    return {
        "eligible": eligible,
        "route": route,
        "center_bit": center,
        "target_bit": target,
        "operations": [{"name": name, "coefficient": coefficient}
                       for name, coefficient in operations],
        "all_service_orders": orders,
        "maximum_absolute_prefix_coefficient": maximum,
    }


def retry_attack(case):
    attacks = []
    operations = [(row["name"], row["coefficient"])
                  for row in case["operations"]]
    for index, operation in enumerate(operations):
        replayed = operations[:index + 1] + [operation] + operations[index + 1:]
        prefixes = [0]
        for _, coefficient in replayed:
            prefixes.append(prefixes[-1] + coefficient)
        attacks.append({
            "duplicated_operation": operation[0],
            "prefix_coefficients": prefixes,
            "maximum_absolute_prefix_coefficient": max(abs(x) for x in prefixes),
            "final_coefficient": prefixes[-1],
        })
    return attacks


def storage_row(bits):
    descriptor_bits = 2 * 128 * W * 2
    metadata_bits = 314
    valid_bits = W * BLOCKS
    accumulator_bits = W * BLOCKS * LANES * bits
    combined = descriptor_bits + metadata_bits + valid_bits + accumulator_bits
    without_valid = descriptor_bits + metadata_bits + accumulator_bits
    return {
        "accumulator_signed_bits": bits,
        "descriptor_bits": descriptor_bits,
        "metadata_bits": metadata_bits,
        "valid_bits": valid_bits,
        "valid_bytes": valid_bits // 8,
        "accumulator_bits": accumulator_bits,
        "combined_bits": combined,
        "combined_bytes_ceiling": (combined + 7) // 8,
        "combined_bytes_without_valid_correction": (without_valid + 7) // 8,
        "valid_correction_byte_delta": ((combined + 7) // 8) - ((without_valid + 7) // 8),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing audit overwrite")

    actual_sha = {label: sha256(path) for label, path in PATHS.items()}
    require(actual_sha == EXPECTED_SHA, "frozen identity drift")

    m41 = strict_json(PATHS["m41_result"])
    m108 = strict_json(PATHS["m108_result"])
    m114 = strict_json(PATHS["m114_result"])
    r1_result = strict_json(PATHS["m115_r1_result"])
    r1_contract = strict_json(PATHS["m115_r1_contract"])
    revocation = strict_json(PATHS["m115_r1_revocation"])
    trigger_review = strict_json(PATHS["trigger_review"])
    result = strict_json(PATHS["m115r2_result"])
    contract = strict_json(PATHS["m115r2_contract"])

    strict_attacks = {
        "duplicate_json_key_rejected": attack_raises(
            lambda: strict_loads('{"x":1,"x":2}')),
        "nan_rejected": attack_raises(lambda: strict_loads('{"x":NaN}')),
        "infinity_rejected": attack_raises(
            lambda: strict_loads('{"x":Infinity}')),
        "duplicate_manifest_path_rejected": attack_raises(
            lambda: parse_manifest_text(("0" * 64 + "  x\n") * 2, HW)),
        "malformed_manifest_hash_rejected": attack_raises(
            lambda: parse_manifest_text("xyz  x\n", HW)),
        "manifest_traversal_rejected": attack_raises(
            lambda: parse_manifest_text("0" * 64 + "  ../x\n", HW)),
    }
    require(all(strict_attacks.values()), "strict attack failure")

    entries = parse_manifest_text(PATHS["m115r2_manifest"].read_text(
        encoding="utf-8"), HW)
    failures = [raw for expected, raw, path in entries
                if not path.is_file() or sha256(path) != expected]
    require(not failures, "M115r2 manifest verification failed")
    listed = {raw for _, raw, _ in entries}
    direct_inputs = {
        str(PATHS[label].relative_to(HW)) for label in (
            "m41_result", "m108_analyzer", "m108_result", "m114_result",
            "m115_r1_revocation", "trigger_review")
    }

    require(revocation["status"] == "REVOKED_MINIMUM_WIDTH_CLAIMS_DO_NOT_CITE",
            "revocation status drift")
    revoked_actual = {
        "analyzer_sha256": actual_sha["m115_r1_analyzer"],
        "result_sha256": actual_sha["m115_r1_result"],
        "contract_sha256": actual_sha["m115_r1_contract"],
    }
    require(revocation["revoked_evidence"] == revoked_actual,
            "revoked evidence identity mismatch")
    require(revocation["trigger"]["independent_review_sha256"]
            == actual_sha["trigger_review"], "trigger review identity mismatch")
    require(revocation["trigger"]["independent_manifest_sha256"]
            == actual_sha["trigger_manifest"], "trigger manifest identity mismatch")
    require(trigger_review["severity_counts"]["P0"] == 1,
            "trigger review P0 drift")
    require(r1_result["proof"]["checkpoint_transient_required_signed_bits"] == 20
            and r1_contract["arithmetic_contract"]["checkpoint_required_signed_bits"] == 20,
            "r1 revoked claim drift")

    cases = [case_row(eligible, center, target)
             for eligible in (True, False)
             for center in (0, 1) for target in (0, 1)]
    pwp_cases = [row for row in cases if row["eligible"]]
    escape_cases = [row for row in cases if not row["eligible"]]
    max_prefix = max(row["maximum_absolute_prefix_coefficient"] for row in cases)
    require(max_prefix == 1, "prefix bound drift")
    require(result["prefix_coefficient_proof"]["legal_cases_all_service_orders"]
            == [{key: row[key] for key in (
                "center_bit", "target_bit", "operations", "all_service_orders",
                "maximum_absolute_prefix_coefficient")}
                for row in pwp_cases], "producer PWP case enumeration mismatch")

    retry_rows = []
    for case in cases:
        attacks = retry_attack(case)
        if attacks:
            retry_rows.append({
                "eligible": case["eligible"],
                "center_bit": case["center_bit"],
                "target_bit": case["target_bit"],
                "attacks": attacks,
            })
    require(any(attack["maximum_absolute_prefix_coefficient"] > 1
                for row in retry_rows for attack in row["attacks"]),
            "retry attack did not break prefix bound")

    operator_summaries = []
    ledgers = []
    init_rows = []
    global_max = 0
    for operator, layer in enumerate(m41["layers"]):
        require(layer["q_min"] >= -127 and layer["q_max"] <= 127
                and layer["reserved_negative_128_count"] == 0,
                "quantized range drift")
        require(layer["conv_bias_present"] is False, "bias drift")
        weight = next(row for row in layer["payloads"] if row["role"] == "weight")
        require(weight["shape"] == [768, 3, 3, 768]
                and weight["layout"] == "I_KY_KX_O_C_ORDER",
                "weight layout drift")
        path = M41_DIR / weight["file"]
        data = path.read_bytes()
        require(len(data) == FEATURES * CHANNELS, "weight extent drift")
        require(sha256(path) == WEIGHT_SHA[operator] == weight["sha256"],
                "weight SHA drift")
        sums = [0] * CHANNELS
        negative_128 = 0
        for index, raw in enumerate(data):
            value = raw if raw < 128 else raw - 256
            negative_128 += int(value == -128)
            sums[index % CHANNELS] += abs(value)
        require(negative_128 == 0, "-128 in payload")
        maximum = max(sums)
        minimum = min(sums)
        max_channel = sums.index(maximum)
        require(maximum == layer["accumulator_bound"]["per_channel_sum_abs_q_maximum"]
                and minimum == layer["accumulator_bound"]["per_channel_sum_abs_q_minimum"],
                "M41 sumabs drift")
        producer = result["operators"][operator]
        require(producer["maximum_per_channel_sum_abs_q"] == maximum
                and producer["minimum_per_channel_sum_abs_q"] == minimum
                and producer["maximum_channel"] == max_channel
                and producer["prefix_bound_required_signed_bits"]
                == signed_bits(maximum), "M115r2 summary drift")
        ledger_sha = sha256_bytes(b"".join(struct.pack("<I", x) for x in sums))
        distribution = Counter(signed_bits(x) for x in sums)
        operator_summaries.append({
            "operator_index": operator,
            "weight_sha256": WEIGHT_SHA[operator],
            "minimum_sumabs": minimum,
            "maximum_sumabs": maximum,
            "maximum_channel": max_channel,
            "required_bits_distribution": {str(k): value for k, value in sorted(distribution.items())},
            "sumabs_u32le_sha256": ledger_sha,
        })
        ledgers.append({
            "operator_index": operator,
            "sumabs_u32le_sha256": ledger_sha,
            "channels": [{"channel": channel, "sumabs": value,
                          "signed_bits_for_sumabs": signed_bits(value)}
                         for channel, value in enumerate(sums)],
        })
        global_max = max(global_max, maximum)

        init = next(row for row in layer["payloads"]
                    if row["role"] == "accumulator_init")
        init_path = M41_DIR / init["file"]
        init_data = init_path.read_bytes()
        require(len(init_data) == 3072 and set(init_data) <= {0},
                "nonzero accumulator init")
        require(sha256(init_path) == init["sha256"], "init SHA drift")
        init_rows.append({"operator_index": operator,
                          "sha256": sha256(init_path), "all_zero": True})

    require(global_max == 218338 and signed_bits(global_max) == 19,
            "checkpoint signed-width drift")
    dense_bound = FEATURES * 127
    require(dense_bound == 877824 and signed_bits(dense_bound) == 21,
            "dense signed-width drift")

    storage = [storage_row(bits) for bits in (19, 20, 21, 22, 24)]
    by_bits = {row["accumulator_signed_bits"]: row for row in storage}
    require(all(row["valid_bits"] == 3072 and row["valid_bytes"] == 384
                and row["valid_correction_byte_delta"] == 384 for row in storage),
            "valid correction drift")
    require(by_bits[19]["combined_bytes_ceiling"] == 725416
            and by_bits[21]["combined_bytes_ceiling"] == 799144
            and by_bits[24]["combined_bytes_ceiling"] == 909736,
            "storage drift")
    producer_storage = {row["accumulator_signed_bits"]: row
                        for row in result["w384_storage_frontier"]}
    for bits, row in by_bits.items():
        require(producer_storage[bits]["combined_bytes_ceiling_before_control_ecc_macro_rounding"]
                == row["combined_bytes_ceiling"], "producer storage mismatch")
    old_w384 = next(row for row in m114["frontier"] if row["window_rows"] == W)
    require(old_w384["storage_lower_bound_corrected"][
        "combined_bytes_ceiling_before_control_ecc_macro_rounding"] == 909736,
        "M114 corrected W384 drift")

    boundary_keys = (
        "integrated_accepted_transaction_exact_once_miter",
        "signed19_accumulator_rtl", "signed19_full_lane_numeric_commercial_vcs",
        "foundry_sram_macro", "macro_inclusive_ppa", "cycle_reduction",
        "physical_speedup", "system_speedup", "headline")
    require(all(result["admission"][key] is False for key in boundary_keys),
            "result admission overreach")
    require(all(contract["admission"][key] is False for key in boundary_keys),
            "contract admission overreach")
    require("mathematical candidate" in contract["paper_safe_statement"]
            and "remain pending" in contract["paper_safe_statement"],
            "paper-safe candidate boundary drift")

    payload = {
        "schema": "m115r2_pwp_prefix_coefficient_width_independent_audit_v1",
        "status": "PASS_CORRECTION_MATHEMATICAL_CANDIDATES_ONLY_HARDWARE_ADMISSION_PENDING",
        "identity": actual_sha,
        "strict_attacks": strict_attacks,
        "producer_manifest": {
            "sha256": actual_sha["m115r2_manifest"],
            "entries": len(entries),
            "failed": failures,
            "listed_paths": sorted(listed),
            "direct_analyzer_inputs": sorted(direct_inputs),
            "missing_direct_analyzer_inputs": sorted(direct_inputs - listed),
            "covers_all_direct_analyzer_inputs": direct_inputs <= listed,
            "weight_payloads_covered": all(str((M41_DIR / ("o{}_weight_i_ky_kx_o_s8.bin".format(i))).relative_to(HW)) in listed for i in range(4)),
        },
        "r1_revocation": {
            "status": revocation["status"],
            "revoked_evidence_identity_matches_current_bytes": True,
            "trigger_review_and_manifest_identity_match_current_bytes": True,
            "withdrawn_claims": revocation["withdrawn_claims"],
            "r2_supersedes_revocation_and_trigger_hashes_match": (
                contract["supersedes"]["m115_r1_revocation_sha256"]
                == actual_sha["m115_r1_revocation"]
                and contract["supersedes"]["triggering_independent_review_sha256"]
                == actual_sha["trigger_review"]),
        },
        "prefix_proof": {
            "eligible_pwp_cases": pwp_cases,
            "escape_cases_independently_enumerated": escape_cases,
            "maximum_absolute_prefix_coefficient_all_pwp_and_escape_orders": max_prefix,
            "global_interleaving_argument": "At any global service prefix each exact-once source term has coefficient in {-1,0,+1}; triangle inequality bounds the output-channel accumulator by sum(abs(weight)).",
            "retry_attacks": retry_rows,
            "retry_or_duplicate_can_exceed_coefficient_one": True,
            "unbounded_retry_has_no_finite_width_bound": True,
            "exact_once_is_explicitly_pending_integrated_miter": True,
            "producer_mechanically_enumerates_escape": False,
            "producer_states_escape_contract_in_text": True,
        },
        "payload_recomputation": {
            "all_3072_channels_recomputed": True,
            "operators": operator_summaries,
            "full_per_channel_ledgers": ledgers,
            "accumulator_init_payloads": init_rows,
            "bias_free": True,
            "checkpoint_maximum_sumabs": global_max,
            "checkpoint_mathematical_candidate_signed_bits": signed_bits(global_max),
            "dense_bound": dense_bound,
            "dense_mathematical_candidate_signed_bits": signed_bits(dense_bound),
        },
        "storage": {
            "formula": "ceil((2*128*W*2 + 314 + W*8 valid + W*8*96*bits)/8)",
            "rows": storage,
            "signed19_saved_bytes_vs_signed24": 909736 - 725416,
            "signed19_saved_fraction": (909736 - 725416) / 909736.0,
            "signed21_saved_bytes_vs_signed24": 909736 - 799144,
            "signed21_saved_fraction": (909736 - 799144) / 909736.0,
            "all_producer_storage_and_saving_arithmetic_reproduced": (
                result["w384_savings_vs_signed24"]["checkpoint_signed19_saved_bytes"] == 184320
                and result["w384_savings_vs_signed24"]["dense_signed21_saved_bytes"] == 110592),
        },
        "claim_boundary": {
            "mathematical_prefix_bound": True,
            **{key: False for key in boundary_keys},
            "mathematical_candidate_not_hardware_admission": True,
            "overclaim_found": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M115r2 independent prefix={} checkpoint_bits={} dense_bits={} W19={} W21={} manifest={}".format(
        max_prefix, signed_bits(global_max), signed_bits(dense_bound),
        by_bits[19]["combined_bytes_ceiling"], by_bits[21]["combined_bytes_ceiling"],
        len(entries)), flush=True)


if __name__ == "__main__":
    main()
