#!/usr/bin/env python3
"""Prove a tighter accumulator width for the H67 PWP+correction stream.

The original convolution bound counts every INT8 weight at most once.  A PWP
anchor followed by a negative correction can transiently count one weight and
then cancel it, so signed19 cannot be inherited without a stream proof.  The
frozen coefficient miter permits at most one anchor and one correction per
source term.  Therefore every partial sum is bounded by twice the exact
per-output-channel sum(abs(weight)).  This script recomputes that bound from
all four serialized INT8 payloads and publishes storage implications only.
"""

import argparse
from array import array
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M41_DIR = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823"
INPUTS = {
    "m41_result": M41_DIR / "m41_h67_ep35_bottleneck_int8_bridge.json",
    "m108_coefficient_analyzer": HW / (
        "system_simulator/scripts/"
        "analyze_m108_w64_fused_pwp_accumulator_schedule.py"),
    "m108_coefficient_result": HW / (
        "results/m108_w64_fused_pwp_accumulator_schedule_r1_20260824/"
        "m108_w64_fused_pwp_accumulator_schedule.json"),
    "m114_storage_result": HW / (
        "results/m114_storage_valid_admission_correction_r1_20260824/"
        "m114_storage_valid_admission_correction.json"),
    "m112_run_complete": HW / (
        "dc_handoff/runs/m112_w384_lane_sliced_accumulator_vcs_r1_sealed_20260824/"
        "RUN_COMPLETE.txt"),
}
EXPECTED_SHA256 = {
    "m41_result":
        "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "m108_coefficient_analyzer":
        "4404e5825ece95fbf0a28dd580c03c7e9f34bcfa9ec12fa3b66d226a9042cbe2",
    "m108_coefficient_result":
        "358640e62c2e52f859b7143f0bac957d6988ed1bd7c56e5dd54d21bc01344318",
    "m114_storage_result":
        "1559c65779fbc15026b3d744e3f1463bba8effd13c2efaa04e8562d4dbfb2226",
    "m112_run_complete":
        "458dc8af156165bf726d36a57813d2d476ec25dded82ffdee077c186f63bba26",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing M115 output overwrite")
    script_start_sha = sha256(Path(__file__).resolve())
    for label, path in INPUTS.items():
        require(sha256(path) == EXPECTED_SHA256[label],
                "frozen input identity drift: " + label)

    m41 = strict_json(INPUTS["m41_result"])
    m108 = strict_json(INPUTS["m108_coefficient_result"])
    m114 = strict_json(INPUTS["m114_storage_result"])
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
    require(m108["work_conservation"]["negative_events"] == 17557357,
            "M108 negative-event extent drift")
    receipt_lines = set(INPUTS["m112_run_complete"].read_text(
        encoding="utf-8").splitlines())
    require("signed_bits_per_lane=24" in receipt_lines
            and "foundry_sram_macro=false" in receipt_lines
            and "physical_speedup=false" in receipt_lines,
            "M112 receipt boundary drift")

    operators = []
    global_max_once = 0
    for operator in range(4):
        layer = m41["layers"][operator]
        payload = next(row for row in layer["payloads"]
                       if row["role"] == "weight")
        path = M41_DIR / payload["file"]
        require(sha256(path) == EXPECTED_WEIGHT_SHA256[operator]
                == payload["sha256"],
                "weight identity drift op{}".format(operator))
        values = array("b")
        values.frombytes(path.read_bytes())
        require(len(values) == FEATURES * OUTPUT_CHANNELS,
                "weight extent drift op{}".format(operator))
        per_channel = [0] * OUTPUT_CHANNELS
        for index, value in enumerate(values):
            per_channel[index % OUTPUT_CHANNELS] += abs(value)
        maximum = max(per_channel)
        maximum_channel = per_channel.index(maximum)
        require(maximum == layer["accumulator_bound"][
                    "per_channel_sum_abs_q_maximum"],
                "M41 sumabs mismatch op{}".format(operator))
        operators.append({
            "operator_index": operator,
            "operator": layer["operator"],
            "weight_payload_sha256": EXPECTED_WEIGHT_SHA256[operator],
            "minimum_per_channel_sum_abs_q": min(per_channel),
            "maximum_per_channel_sum_abs_q": maximum,
            "maximum_channel": maximum_channel,
            "once_bound_required_signed_bits":
                signed_bits_for_magnitude(maximum),
            "twice_bound_magnitude": 2 * maximum,
            "pwp_plus_correction_transient_required_signed_bits":
                signed_bits_for_magnitude(2 * maximum),
        })
        global_max_once = max(global_max_once, maximum)

    checkpoint_transient = 2 * global_max_once
    checkpoint_bits = signed_bits_for_magnitude(checkpoint_transient)
    dense_once = FEATURES * 127
    dense_transient = 2 * dense_once
    dense_bits = signed_bits_for_magnitude(dense_transient)
    require(global_max_once == 218338 and checkpoint_transient == 436676
            and checkpoint_bits == 20,
            "checkpoint decomposition bound drift")
    require(dense_once == 877824 and dense_transient == 1755648
            and dense_bits == 22,
            "dense decomposition bound drift")

    storage = [storage_row(bits) for bits in (19, 20, 21, 22, 24)]
    by_bits = {row["accumulator_signed_bits"]: row for row in storage}
    old_w384 = next(row for row in m114["frontier"]
                    if row["window_rows"] == WINDOW_ROWS)
    require(by_bits[24][
                "combined_bytes_ceiling_before_control_ecc_macro_rounding"]
            == old_w384["storage_lower_bound_corrected"][
                "combined_bytes_ceiling_before_control_ecc_macro_rounding"]
            == 909736,
            "M114 signed24 storage bridge drift")
    checkpoint_saving = 909736 - by_bits[20][
        "combined_bytes_ceiling_before_control_ecc_macro_rounding"]
    dense_saving = 909736 - by_bits[22][
        "combined_bytes_ceiling_before_control_ecc_macro_rounding"]
    require(checkpoint_saving == 147456 and dense_saving == 73728,
            "storage saving drift")

    require(sha256(Path(__file__).resolve()) == script_start_sha,
            "M115 analyzer changed during execution")
    payload = {
        "schema": "m115_pwp_transient_accumulator_width_result_v1",
        "status": "PASS_CHECKPOINT_SIGNED20_DENSE_SIGNED22_TRANSIENT_BOUND_RTL_PENDING",
        "identity": {
            "analyzer_start_end_sha256": script_start_sha,
            "frozen_inputs_sha256": EXPECTED_SHA256,
            "weight_payload_sha256": list(EXPECTED_WEIGHT_SHA256),
        },
        "proof": {
            "coefficient_contract": "for each source term, reconstructed coefficient is center + positive_correction - negative_correction",
            "maximum_absolute_term_multiplicity_during_stream": 2,
            "ordering_independent_partial_sum_bound":
                "2 * sum(abs(INT8 weight)) per output channel",
            "checkpoint_maximum_sum_abs_q": global_max_once,
            "checkpoint_transient_magnitude_bound": checkpoint_transient,
            "checkpoint_transient_required_signed_bits": checkpoint_bits,
            "dense_int8_once_magnitude_bound": dense_once,
            "dense_int8_decomposed_transient_magnitude_bound": dense_transient,
            "dense_int8_decomposed_transient_required_signed_bits": dense_bits,
            "signed19_direct_inheritance_rejected": True,
        },
        "operators": operators,
        "w384_storage_frontier": storage,
        "w384_savings_vs_current_signed24": {
            "checkpoint_specific_signed20_bytes": checkpoint_saving,
            "checkpoint_specific_signed20_fraction_of_combined_lower_bound":
                checkpoint_saving / 909736.0,
            "dense_safe_signed22_bytes": dense_saving,
            "dense_safe_signed22_fraction_of_combined_lower_bound":
                dense_saving / 909736.0,
        },
        "admission": {
            "all_four_int8_payloads_recomputed": True,
            "ordering_independent_checkpoint_width_proof": True,
            "signed20_accumulator_rtl": False,
            "signed20_full_lane_numeric_vcs": False,
            "foundry_sram_macro": False,
            "macro_inclusive_ppa": False,
            "cycle_reduction": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "m115_pwp_transient_accumulator_width.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M115 checkpoint_signed20={}B save={}B dense_signed22={}B save={}B".format(
        by_bits[20]["combined_bytes_ceiling_before_control_ecc_macro_rounding"],
        checkpoint_saving,
        by_bits[22]["combined_bytes_ceiling_before_control_ecc_macro_rounding"],
        dense_saving), flush=True)


if __name__ == "__main__":
    main()
