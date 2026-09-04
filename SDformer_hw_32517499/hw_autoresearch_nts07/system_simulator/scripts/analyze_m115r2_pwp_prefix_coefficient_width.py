#!/usr/bin/env python3
"""Correct M115 using an exact per-term prefix-coefficient proof.

For an eligible PWP source term, the only two-operation case is a +1 anchor
and a -1 correction.  Enumerating every legal center/target case and every
operation permutation proves that a term's prefix coefficient stays in
{-1, 0, +1}; it never reaches magnitude two.  This restores the original
one-sumabs checkpoint and dense INT8 width candidates while explicitly
leaving accepted-transaction replay and commercial RTL validation pending.
"""

import argparse
from array import array
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
INPUTS = {
    "m41_result": M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json",
    "m108_coefficient_analyzer": HW / (
        "system_simulator/scripts/analyze_m108_w64_fused_pwp_accumulator_schedule.py"),
    "m108_coefficient_result": HW / (
        "results/m108_w64_fused_pwp_accumulator_schedule_r1_20260824/"
        "m108_w64_fused_pwp_accumulator_schedule.json"),
    "m114_storage_result": HW / (
        "results/m114_storage_valid_admission_correction_r1_20260824/"
        "m114_storage_valid_admission_correction.json"),
    "m115_r1_revocation": HW / (
        "contracts/m115_r1_transient_width_claim_revocation_r1_20260824.json"),
    "m115_r1_independent_review": HW / (
        "reviews/m115_pwp_transient_accumulator_width_independent_hammer_r1_20260824/"
        "m115_pwp_transient_accumulator_width_independent_hammer_review.json"),
}
EXPECTED_SHA256 = {
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m108_coefficient_analyzer": "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m108_coefficient_result": "358640e62c2e52f859b7143f0bac957d6988ed1bd7c56e5dd54d21bc01344318",
    "m114_storage_result": "1559c65779fbc15026b3d744e3f1463bba8effd13c2efaa04e8562d4dbfb2226",
    "m115_r1_revocation": "120d0faae82cec9d434fbdc66083bc87d1f6264840a79f142010ba39e2a536a9",
    "m115_r1_independent_review": "461c7b78ca6f0b4132ba8ccd22833d42e5df85041c5429cceebafdafa6a18d64",
}
EXPECTED_WEIGHT_SHA256 = (
    "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "f0b8ed22f4fbefc7753e9eff12bec6880d7c199db6a78ccf7f2f6d1343e890d9",
    "c2a5f5b2489dadc7b46892d40e12fd960f6ca0bd595ef238cdf9915bcb5f5c8a",
    "f3d7f2587d2b72518d945dfb6e6b954d8b2d9627e491b74b879a36a5d031c6e1",
)
FEATURES = 768 * 3 * 3
OUTPUT_CHANNELS = 768
OUTPUT_BLOCKS = 8
OUTPUT_LANES = 96
WINDOW_ROWS = 384
DESCRIPTOR_BITS = 2 * 128 * WINDOW_ROWS * 2
DESCRIPTOR_METADATA_BITS_MIN = 314
VALID_BITS = WINDOW_ROWS * OUTPUT_BLOCKS


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
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def signed_bits_for_magnitude(magnitude):
    require(magnitude >= 0, "negative magnitude")
    for bits in range(2, 33):
        if magnitude <= (1 << (bits - 1)) - 1:
            return bits
    raise ValueError("magnitude exceeds signed32")


def storage_row(bits):
    accumulator_bits = WINDOW_ROWS * OUTPUT_BLOCKS * OUTPUT_LANES * bits
    combined_bits = (DESCRIPTOR_BITS + DESCRIPTOR_METADATA_BITS_MIN
                     + VALID_BITS + accumulator_bits)
    return {
        "accumulator_signed_bits": bits,
        "accumulator_payload_bits": accumulator_bits,
        "accumulator_payload_bytes": accumulator_bits // 8,
        "combined_descriptor_valid_accumulator_bits": combined_bits,
        "combined_bytes_ceiling_before_control_ecc_macro_rounding":
            (combined_bits + 7) // 8,
    }


def prefix_case(center, target):
    operations = []
    if center:
        operations.append({"name": "anchor", "coefficient": 1})
    if target and not center:
        operations.append({"name": "positive_correction", "coefficient": 1})
    if center and not target:
        operations.append({"name": "negative_correction", "coefficient": -1})
    order_rows = []
    maximum = 0
    for permutation in sorted(set(itertools.permutations(
            tuple((op["name"], op["coefficient"]) for op in operations)))):
        prefixes = [0]
        for _, coefficient in permutation:
            prefixes.append(prefixes[-1] + coefficient)
        maximum = max(maximum, *(abs(value) for value in prefixes))
        order_rows.append({
            "operation_order": [name for name, _ in permutation],
            "prefix_coefficients": prefixes,
        })
    if not operations:
        order_rows = [{"operation_order": [], "prefix_coefficients": [0]}]
    require(all(row["prefix_coefficients"][-1] == target for row in order_rows),
            "final coefficient mismatch")
    return {
        "center_bit": center,
        "target_bit": target,
        "operations": operations,
        "all_service_orders": order_rows,
        "maximum_absolute_prefix_coefficient": maximum,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M115r2 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    for label, path in INPUTS.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)

    m41 = strict_json(INPUTS["m41_result"])
    m108 = strict_json(INPUTS["m108_coefficient_result"])
    m114 = strict_json(INPUTS["m114_storage_result"])
    revocation = strict_json(INPUTS["m115_r1_revocation"])
    review = strict_json(INPUTS["m115_r1_independent_review"])
    require(revocation["status"] == "REVOKED_MINIMUM_WIDTH_CLAIMS_DO_NOT_CITE",
            "M115r1 revocation status drift")
    require(review["severity_counts"]["P0"] == 1,
            "M115r1 independent P0 drift")
    require(m41["m40_schedule_bridge"][
                "checkpoint_tight_accumulator_signed_bits"] == 19,
            "M41 checkpoint width drift")
    require(m41["m40_schedule_bridge"][
                "dense_envelope_accumulator_signed_bits"] == 21,
            "M41 dense width drift")
    require(m108["admission"]["source_coefficient_miter"] is True
            and m108["work_conservation"]["source_coefficient_checks"]
            == 3317760000,
            "M108 coefficient proof drift")

    cases = [prefix_case(center, target)
             for center in (0, 1) for target in (0, 1)]
    max_prefix = max(row["maximum_absolute_prefix_coefficient"] for row in cases)
    require(max_prefix == 1, "prefix coefficient bound drift")

    operators = []
    global_max_once = 0
    for operator in range(4):
        layer = m41["layers"][operator]
        payload = next(row for row in layer["payloads"] if row["role"] == "weight")
        path = M41_DIR / payload["file"]
        require(sha256(path) == EXPECTED_WEIGHT_SHA256[operator] == payload["sha256"],
                "weight identity drift op{}".format(operator))
        values = array("b")
        values.frombytes(path.read_bytes())
        require(len(values) == FEATURES * OUTPUT_CHANNELS,
                "weight extent drift op{}".format(operator))
        per_channel = [0] * OUTPUT_CHANNELS
        for index, value in enumerate(values):
            per_channel[index % OUTPUT_CHANNELS] += abs(value)
        maximum = max(per_channel)
        require(maximum == layer["accumulator_bound"]["per_channel_sum_abs_q_maximum"],
                "M41 sumabs mismatch op{}".format(operator))
        operators.append({
            "operator_index": operator,
            "operator": layer["operator"],
            "weight_payload_sha256": EXPECTED_WEIGHT_SHA256[operator],
            "minimum_per_channel_sum_abs_q": min(per_channel),
            "maximum_per_channel_sum_abs_q": maximum,
            "maximum_channel": per_channel.index(maximum),
            "prefix_bound_required_signed_bits": signed_bits_for_magnitude(maximum),
        })
        global_max_once = max(global_max_once, maximum)

    checkpoint_bits = signed_bits_for_magnitude(global_max_once)
    dense_bound = FEATURES * 127
    dense_bits = signed_bits_for_magnitude(dense_bound)
    require(global_max_once == 218338 and checkpoint_bits == 19,
            "checkpoint prefix bound drift")
    require(dense_bound == 877824 and dense_bits == 21,
            "dense prefix bound drift")

    storage = [storage_row(bits) for bits in (19, 20, 21, 22, 24)]
    by_bits = {row["accumulator_signed_bits"]: row for row in storage}
    old_w384 = next(row for row in m114["frontier"]
                    if row["window_rows"] == WINDOW_ROWS)
    old_bytes = old_w384["storage_lower_bound_corrected"][
        "combined_bytes_ceiling_before_control_ecc_macro_rounding"]
    require(old_bytes == by_bits[24][
                "combined_bytes_ceiling_before_control_ecc_macro_rounding"]
            == 909736, "M114 signed24 storage bridge drift")
    checkpoint_bytes = by_bits[19][
        "combined_bytes_ceiling_before_control_ecc_macro_rounding"]
    dense_bytes = by_bits[21][
        "combined_bytes_ceiling_before_control_ecc_macro_rounding"]
    require(checkpoint_bytes == 725416 and dense_bytes == 799144,
            "corrected storage drift")

    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M115r2 analyzer changed during execution")
    payload = {
        "schema": "m115r2_pwp_prefix_coefficient_width_result_v1",
        "status": "PASS_PREFIX_BOUND_SIGNED19_SIGNED21_CANDIDATES_RTL_PENDING",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
            "weight_payload_sha256": list(EXPECTED_WEIGHT_SHA256),
        },
        "prefix_coefficient_proof": {
            "coefficient_identity": "target = center + positive_correction - negative_correction",
            "legal_cases_all_service_orders": cases,
            "maximum_absolute_prefix_coefficient": max_prefix,
            "ordering_independent_partial_sum_bound":
                "sum(abs(INT8 weight)) per output channel under exact-once accepted operations",
            "escape_contract": "raw target bit is applied at most once and has coefficient 0 or +1",
            "retry_boundary": "duplicate or replayed accepted operations invalidate this proof",
        },
        "checkpoint": {
            "maximum_per_channel_sum_abs_q": global_max_once,
            "mathematical_candidate_signed_bits": checkpoint_bits,
        },
        "dense_int8": {
            "maximum_terms": FEATURES,
            "maximum_abs_weight": 127,
            "sum_abs_bound": dense_bound,
            "mathematical_candidate_signed_bits": dense_bits,
        },
        "operators": operators,
        "w384_storage_frontier": storage,
        "w384_savings_vs_signed24": {
            "checkpoint_signed19_combined_bytes": checkpoint_bytes,
            "checkpoint_signed19_saved_bytes": old_bytes - checkpoint_bytes,
            "checkpoint_signed19_saved_fraction": (old_bytes - checkpoint_bytes) / old_bytes,
            "dense_signed21_combined_bytes": dense_bytes,
            "dense_signed21_saved_bytes": old_bytes - dense_bytes,
            "dense_signed21_saved_fraction": (old_bytes - dense_bytes) / old_bytes,
        },
        "admission": {
            "all_four_int8_payloads_recomputed": True,
            "all_legal_term_cases_and_orders_enumerated": True,
            "mathematical_prefix_bound": True,
            "integrated_accepted_transaction_exact_once_miter": False,
            "signed19_accumulator_rtl": False,
            "signed19_full_lane_numeric_commercial_vcs": False,
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "cycle_reduction": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "m115r2_pwp_prefix_coefficient_width.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M115r2 checkpoint_signed19={}B save={}B dense_signed21={}B save={}B".format(
        checkpoint_bytes, old_bytes - checkpoint_bytes,
        dense_bytes, old_bytes - dense_bytes), flush=True)


if __name__ == "__main__":
    main()
