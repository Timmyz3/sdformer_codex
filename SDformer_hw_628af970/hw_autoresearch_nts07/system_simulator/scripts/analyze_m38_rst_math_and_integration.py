#!/usr/bin/env python3
"""Build fail-closed M38-RST arithmetic and integration-theory evidence."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    ROOT
    / "hw_autoresearch_nts07/contracts/"
    "m38_rst_math_input_contract_r1_20260822.json"
)

Q8_MIN = -(1 << 7)
Q8_MAX = (1 << 7) - 1
Q24_MIN = -(1 << 23)
Q24_MAX = (1 << 23) - 1
TERNARY_CODE_TO_VALUE = {0: 0, 1: 1, 2: -1}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def signed_bounds(bits):
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def minimum_signed_bits(minimum, maximum):
    if minimum > maximum:
        raise ValueError("invalid signed range")
    for bits in range(1, 65):
        lower, upper = signed_bounds(bits)
        if minimum >= lower and maximum <= upper:
            return bits
    raise ValueError("range exceeds supported audit width")


def decode_ternary(code):
    if code not in TERNARY_CODE_TO_VALUE:
        raise ValueError("illegal M38 ternary code: {}".format(code))
    return TERNARY_CODE_TO_VALUE[code]


def ternary_product(value, code):
    value = int(value)
    if value < Q8_MIN or value > Q8_MAX:
        raise ValueError("M38 scalar input is not signed q8")
    coefficient = decode_ternary(int(code))
    if coefficient == 0:
        return 0
    if coefficient == 1:
        return value
    return -value


def sat_signed(value, bits):
    lower, upper = signed_bounds(bits)
    return min(upper, max(lower, int(value)))


def event_bit(value, threshold):
    if threshold < Q24_MIN or threshold > Q24_MAX:
        raise ValueError("threshold is not signed Q24")
    return int(int(value) >= int(threshold))


def load_contract(path):
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema") != "m38_rst_math_input_contract_v1":
        raise ValueError("unexpected M38 contract schema")
    expected_inputs = {
        "m29_config_generator",
        "m31_vcs_contract",
        "m31_vcs_receipt",
        "m37_math_contract",
        "m37_math_result",
        "m37_vcs_contract",
        "m37_vcs_receipt",
    }
    if set(contract.get("inputs", {})) != expected_inputs:
        raise ValueError("M38 input population drift")
    payloads = {}
    hashes = {}
    for name, spec in sorted(contract["inputs"].items()):
        if set(spec) != {"path", "sha256"}:
            raise ValueError("M38 input specification drift for {}".format(name))
        source = resolve(spec["path"])
        if not source.is_file():
            raise ValueError("M38 input is missing for {}".format(name))
        actual = sha256(source)
        if actual != spec["sha256"]:
            raise ValueError("M38 input hash drift for {}".format(name))
        source_text = source.read_text(encoding="utf-8")
        if source.suffix == ".json":
            payloads[name] = json.loads(source_text)
        else:
            payloads[name] = source_text
        hashes[name] = actual
    return contract, payloads, hashes


def validate_frozen_sources(contract, payloads, hashes):
    arch = contract["frozen_architecture"]
    required_architecture = {
        "temporal_rows": 10,
        "rank": 3,
        "lanes": 16,
        "signed_input_bits": 8,
        "stage1_accumulator_bits": 24,
        "stage1_intermediate_bits": 8,
        "bias_bits": 24,
        "threshold_bits": 24,
        "shared_signed_int8_multiplier_lanes": 96,
        "rows_per_phase": 2,
        "phases_per_tile": 5,
        "result_beats_per_tile": 5,
        "result_fifo_entries": 16,
        "result_fifo_atomic_credit_per_t10_tile": 5,
        "intermediate_elastic_slots_target": 1,
        "intermediate_slot_bits": 384,
        "configuration_generation_bits": 16,
        "configuration_crc_bits": 32,
        "t10_factorized_modules_expected_from_m29_interface": 45,
        "t2_dense_fallback_modules_expected_from_m29_interface": 60,
    }
    for key, expected in required_architecture.items():
        if arch.get(key) != expected:
            raise ValueError("M38 frozen architecture drift for {}".format(key))
    if arch.get("ternary_codes") != {
            "0": 0, "1": 1, "2": -1, "3": "illegal"}:
        raise ValueError("M38 ternary codebook drift")
    if arch.get("configuration_crc_profile") != {
            "name": "CRC-32C_Castagnoli",
            "normal_polynomial": "0x1EDC6F41",
            "reflected_polynomial": "0x82F63B78",
            "initial_value": "0xFFFFFFFF",
            "final_xor": "0xFFFFFFFF",
            "reflect_input": True,
            "reflect_output": True,
            "protected_payload_order": (
                "right_factor_then_left_codes_then_bias_then_threshold_then_"
                "requant_shift_then_generation_lsb_first_with_zero_pad_to_next_byte"
            ),
    }:
        raise ValueError("M38 configuration CRC profile drift")
    required_theory_rules = {
        "conditional_t10_steady_ii_serialized": 10,
        "conditional_t10_steady_ii_parallel": 5,
        "conditional_t10_steady_throughput_ratio": 2.0,
        "integrated_parallel_cycles_for_n_tiles": "5 + 5*N",
        "serialized_cycles_for_n_tiles": "10*N",
        "configuration_load_cycles_included": False,
        "result_backpressure_included": False,
        "system_speedup_admitted": False,
        "area_admitted": False,
        "energy_admitted": False,
    }
    if contract.get("theory_rules") != required_theory_rules:
        raise ValueError("M38 theory rule drift")

    m29_generator = payloads["m29_config_generator"]
    if ('"m29_expected_t10_factorized_modules": 45' not in m29_generator
            or '"m29_expected_t2_dense_fallback_modules": 60'
            not in m29_generator
            or '"temporal_factor_rank": 3' not in m29_generator):
        raise ValueError("M38 M29 scope/rank identity drift")

    m31_contract = payloads["m31_vcs_contract"]
    m31_receipt = payloads["m31_vcs_receipt"]
    if m31_receipt["contract"]["sha256"] != hashes["m31_vcs_contract"]:
        raise ValueError("M38 M31 receipt-to-contract hash drift")
    if not m31_receipt["status"].startswith(
            "PASS_UNIFIED_T10_T2_EXACT_FIXED_POINT"):
        raise ValueError("M38 M31 receipt is not the exact fixed-point anchor")
    schedule = m31_contract["t10_schedule"]
    resources = m31_contract["rtl_resource_contract"]
    if (schedule["stage1_cycles_per_tile"] != 5
            or schedule["stage2_cycles_per_tile"] != 5
            or schedule["conditional_steady_ii_cycles"] != 10
            or schedule["lanes_per_tile"] != 16
            or resources["signed_int8_multiplier_assignments_in_pool"] != 96
            or resources["shared_result_fifo_entries"] != 16):
        raise ValueError("M38 M31 schedule/resource identity drift")

    m37_result = payloads["m37_math_result"]
    m37_math_contract = payloads["m37_math_contract"]
    m37_contract = payloads["m37_vcs_contract"]
    m37_receipt = payloads["m37_vcs_receipt"]
    if (m37_result["identity"]["contract_sha256"]
            != hashes["m37_math_contract"]):
        raise ValueError("M38 M37 math result-to-contract hash drift")
    if m37_math_contract["architecture"]["maximum_csd_terms_per_coefficient"] != 4:
        raise ValueError("M38 M37 math contract architecture drift")
    if m37_result["signed_int8_coefficient_audit"]["maximum_terms"] != 4:
        raise ValueError("M38 M37 CSD4 math identity drift")
    if m37_contract["architecture"]["maximum_canonical_naf_terms_per_coefficient"] != 4:
        raise ValueError("M38 M37 VCS architecture drift")
    if m37_receipt["contract"]["sha256"] != hashes["m37_vcs_contract"]:
        raise ValueError("M38 M37 receipt-to-contract hash drift")
    if not m37_receipt["status"].startswith(
            "PASS_STANDALONE_T10_CANONICAL_NAF_CSD_RECONSTRUCTION"):
        raise ValueError("M38 M37 VCS receipt is not the standalone exact anchor")
    if (m37_receipt["math_anchor"]["contract"][1]
            != hashes["m37_math_contract"]
            or m37_receipt["math_anchor"]["result"][1]
            != hashes["m37_math_result"]):
        raise ValueError("M38 M37 VCS receipt-to-math-anchor hash drift")
    if m37_receipt["observed"]["data_multiplier_operator_in_DUT"]:
        raise ValueError("M38 M37 source unexpectedly contains a data multiplier")


def build_scalar_audit():
    rows = []
    observed = []
    boundary_witnesses = {}
    for value in range(Q8_MIN, Q8_MAX + 1):
        for code in range(3):
            coefficient = decode_ternary(code)
            product = ternary_product(value, code)
            expected = value * coefficient
            if product != expected:
                raise ValueError("M38 scalar arithmetic mismatch")
            observed.append(product)
            if (value, coefficient) in {
                    (-128, -1), (-128, 0), (-128, 1),
                    (127, -1), (127, 0), (127, 1)}:
                boundary_witnesses["{}_{}".format(value, coefficient)] = product
            rows.append({
                "input_q8": value,
                "ternary_code": code,
                "coefficient": coefficient,
                "product": product,
            })
    product_minimum = min(observed)
    product_maximum = max(observed)
    if (len(rows) != 768 or product_minimum != -128
            or product_maximum != 128
            or boundary_witnesses["-128_-1"] != 128):
        raise ValueError("M38 exhaustive scalar audit drift")
    return {
        "input_domain": [Q8_MIN, Q8_MAX],
        "ternary_coefficient_domain": [-1, 0, 1],
        "legal_code_domain": [0, 1, 2],
        "illegal_code_domain": [3],
        "pairs_checked": len(rows),
        "all_products_exact": True,
        "product_range": [product_minimum, product_maximum],
        "minimum_signed_product_bits": minimum_signed_bits(
            product_minimum, product_maximum),
        "negative_minimum_negation_witness": {
            "expression": "-(-128)",
            "result": boundary_witnesses["-128_-1"],
            "requires_more_than_signed_q8": True,
        },
        "boundary_witnesses": boundary_witnesses,
        "rows": rows,
    }


def build_rank3_and_threshold_audit(scalar):
    product_minimum, product_maximum = scalar["product_range"]
    rank_minimum = 3 * product_minimum
    rank_maximum = 3 * product_maximum
    rank_bits = minimum_signed_bits(rank_minimum, rank_maximum)
    if (rank_minimum, rank_maximum, rank_bits) != (-384, 384, 10):
        raise ValueError("M38 rank-3 signed-width drift")

    pre_minimum = Q24_MIN + rank_minimum
    pre_maximum = Q24_MAX + rank_maximum
    pre_bits = minimum_signed_bits(pre_minimum, pre_maximum)
    if pre_bits != 25:
        raise ValueError("M38 bias-plus-rank-sum width drift")

    bias_vectors = [
        Q24_MIN,
        Q24_MIN + 383,
        Q24_MIN + 384,
        -384,
        -1,
        0,
        1,
        384,
        Q24_MAX - 384,
        Q24_MAX - 383,
        Q24_MAX,
    ]
    saturation_rows = []
    positive_saturations = 0
    negative_saturations = 0
    equality_checks = 0
    just_below_checks = 0
    for bias in bias_vectors:
        for rank_sum in range(rank_minimum, rank_maximum + 1):
            pre_saturation = bias + rank_sum
            saturated = sat_signed(pre_saturation, 24)
            if saturated == Q24_MAX and pre_saturation > Q24_MAX:
                positive_saturations += 1
            if saturated == Q24_MIN and pre_saturation < Q24_MIN:
                negative_saturations += 1
            if event_bit(saturated, saturated) != 1:
                raise ValueError("M38 threshold equality semantics drift")
            equality_checks += 1
            if saturated > Q24_MIN:
                if event_bit(saturated - 1, saturated) != 0:
                    raise ValueError("M38 just-below threshold semantics drift")
                just_below_checks += 1
            saturation_rows.append({
                "bias_q24": bias,
                "rank3_sum": rank_sum,
                "pre_saturation": pre_saturation,
                "saturated_q24": saturated,
            })
    if positive_saturations == 0 or negative_saturations == 0:
        raise ValueError("M38 saturation rails lack directed coverage")
    return {
        "rank": 3,
        "rank3_sum_range": [rank_minimum, rank_maximum],
        "minimum_signed_rank3_sum_bits": rank_bits,
        "implemented_pre_saturation_bits_target": 26,
        "mathematical_minimum_bias_plus_rank_sum_bits": pre_bits,
        "bias_q24_domain": [Q24_MIN, Q24_MAX],
        "pre_saturation_range": [pre_minimum, pre_maximum],
        "saturation_output_domain": [Q24_MIN, Q24_MAX],
        "saturation_vectors_checked": len(saturation_rows),
        "positive_saturation_witnesses": positive_saturations,
        "negative_saturation_witnesses": negative_saturations,
        "threshold_equality_checks": equality_checks,
        "threshold_just_below_checks": just_below_checks,
        "threshold_equality_event": 1,
        "threshold_just_below_event": 0,
        "saturation_rows": saturation_rows,
    }


def configuration_ledger():
    common = {
        "right_factor_bits": 30 * 8,
        "bias_bits": 10 * 24,
        "threshold_bits": 24,
        "stage1_requant_shift_bits": 5,
    }
    common_bits = sum(common.values())
    integrity = {"generation_bits": 16, "crc_bits": 32}
    rows = [
        {
            "name": "m31_serialized_shared96",
            "left_payload": {"signed_int8_coefficients": 30 * 8},
            "left_payload_bits": 240,
        },
        {
            "name": "direct_second96_parallel",
            "left_payload": {"signed_int8_coefficients": 30 * 8},
            "left_payload_bits": 240,
        },
        {
            "name": "m37_csd4_parallel_normalized_integration_target",
            "left_payload": {
                "redundant_signed_int8_coefficients": 30 * 8,
                "term_valid_bits": 30 * 4,
                "term_negative_bits": 30 * 4,
                "term_shift_bits": 30 * 4 * 3,
            },
            "left_payload_bits": 840,
        },
        {
            "name": "m38_rst_parallel",
            "left_payload": {"ternary_codes": 30 * 2},
            "left_payload_bits": 60,
        },
    ]
    for row in rows:
        row["common_payload"] = dict(common)
        row["t10_parameter_payload_bits"] = common_bits + row["left_payload_bits"]
        row["required_context_integrity_bits"] = dict(integrity)
        row["t10_context_bits_with_integrity"] = (
            row["t10_parameter_payload_bits"] + sum(integrity.values()))
        row["parameter_load_cycles_included_in_throughput"] = False
    return {
        "common_payload_bits_excluding_left_factor": common_bits,
        "required_context_integrity_bits": sum(integrity.values()),
        "rows": rows,
        "scope": "resident T10 context bits only; physical SRAM/compiler and load transactions are unmodeled",
    }


def integration_ledger():
    candidates = [
        {
            "name": "m31_serialized_shared96",
            "shared_int8_multiplier_lanes": 96,
            "additional_int8_multiplier_lanes": 0,
            "added_programmable_shift_term_sites_per_cycle": 0,
            "added_ternary_select_sites_per_cycle": 0,
            "stage1_cycles_per_tile": 5,
            "stage2_cycles_per_tile": 5,
            "phase_relation": "serialized",
            "conditional_t10_steady_ii_cycles": 10,
            "conditional_t10_steady_throughput_ratio_vs_m31": 1.0,
            "physical_stage2_operator": "reuse_shared_signed_int8_mul96",
        },
        {
            "name": "direct_second96_parallel",
            "shared_int8_multiplier_lanes": 96,
            "additional_int8_multiplier_lanes": 96,
            "added_programmable_shift_term_sites_per_cycle": 0,
            "added_ternary_select_sites_per_cycle": 0,
            "stage1_cycles_per_tile": 5,
            "stage2_cycles_per_tile": 5,
            "phase_relation": "parallel_after_five_cycle_fill",
            "conditional_t10_steady_ii_cycles": 5,
            "conditional_t10_steady_throughput_ratio_vs_m31": 2.0,
            "physical_stage2_operator": "second_signed_int8_mul96",
        },
        {
            "name": "m37_csd4_parallel_normalized_integration_target",
            "shared_int8_multiplier_lanes": 96,
            "additional_int8_multiplier_lanes": 0,
            "added_programmable_shift_term_sites_per_cycle": 384,
            "added_ternary_select_sites_per_cycle": 0,
            "stage1_cycles_per_tile": 5,
            "stage2_cycles_per_tile": 5,
            "phase_relation": "parallel_after_five_cycle_fill",
            "conditional_t10_steady_ii_cycles": 5,
            "conditional_t10_steady_throughput_ratio_vs_m31": 2.0,
            "physical_stage2_operator": "up_to_four_signed_power_terms_per_coefficient_product",
            "existing_standalone_vcs_input_slots": 2,
            "existing_standalone_vcs_private_result_fifo_entries": 16,
        },
        {
            "name": "m38_rst_parallel",
            "shared_int8_multiplier_lanes": 96,
            "additional_int8_multiplier_lanes": 0,
            "added_programmable_shift_term_sites_per_cycle": 0,
            "added_ternary_select_sites_per_cycle": 96,
            "stage1_cycles_per_tile": 5,
            "stage2_cycles_per_tile": 5,
            "phase_relation": "parallel_after_five_cycle_fill",
            "conditional_t10_steady_ii_cycles": 5,
            "conditional_t10_steady_throughput_ratio_vs_m31": 2.0,
            "physical_stage2_operator": "zero_or_sign_select_then_rank3_add",
        },
    ]
    for row in candidates:
        row.update({
            "normalized_integrated_intermediate_slots": 1,
            "normalized_integrated_intermediate_bits": 384,
            "normalized_shared_result_fifo_entries": 16,
            "atomic_result_fifo_credit_per_t10_tile": 5,
            "result_beats_per_t10_tile": 5,
            "configuration_load_cycles_included": False,
            "result_backpressure_included": False,
            "executable_integrated_cycles_admitted": False,
            "area_admitted": False,
            "energy_admitted": False,
            "system_speedup_admitted": False,
        })
        if row["phase_relation"] == "serialized":
            row["conditional_n_tile_cycle_equation"] = "10*N"
        else:
            row["conditional_n_tile_cycle_equation"] = "5 + 5*N"
    return {
        "normalization": (
            "all future integrated variants must share the M31 input banks, "
            "one 384-bit intermediate elastic slot, one 16-entry result FIFO, "
            "the same atomic five-entry T10 credit rule, configuration integrity, "
            "and the same scheduler; standalone block totals are not area-matched"
        ),
        "algorithmic_work_per_t10_tile": {
            "stage1_rank3_products": 480,
            "stage2_rank3_coefficient_products": 480,
            "factorized_products_total": 960,
            "dense_t10_equivalent_products": 1600,
            "warning": "algorithmic-equivalent products are not physical GOPS",
        },
        "candidates": candidates,
        "conditional_claim_only": True,
        "system_speedup_admitted": False,
    }


def build(contract_path=DEFAULT_CONTRACT):
    contract, payloads, hashes = load_contract(contract_path)
    validate_frozen_sources(contract, payloads, hashes)
    scalar = build_scalar_audit()
    rank3 = build_rank3_and_threshold_audit(scalar)
    configuration = configuration_ledger()
    integration = integration_ledger()
    return {
        "schema": "m38_rst_math_and_integration_audit_v1",
        "status": "PASS_M38_RST_EXHAUSTIVE_SCALAR_AND_INTEGRATION_THEORY_ONLY",
        "identity": {
            "contract": str(Path(contract_path).resolve()),
            "contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
        },
        "scalar_ternary_audit": scalar,
        "rank3_q24_threshold_audit": rank3,
        "configuration_bit_ledger": configuration,
        "integrated_theory_ledger": integration,
        "line_scope": {
            "motion_h67": (
                "structurally applicable to the 45 T10 modules expected by the "
                "M29 interface; H67 constrained training and valid825 are pending"
            ),
            "local5": (
                "RTL shape is reusable only after an ep44 module census and an "
                "independent constrained checkpoint/export/valid825 closure"
            ),
            "t2": "60 expected T2 modules remain dense on the sole M31 pool",
        },
        "admission": {
            "q8_times_ternary_scalar_math_admitted": True,
            "rank3_width_and_q24_threshold_reference_admitted": True,
            "configuration_bit_arithmetic_admitted": True,
            "conditional_t10_theory_ledger_admitted": True,
            "trained_codebook_admitted": False,
            "integrated_rtl_admitted": False,
            "executable_integrated_cycles_admitted": False,
            "area_timing_power_energy_admitted": False,
            "system_cycles_admitted": False,
            "system_speedup_admitted": False,
            "headline_admitted": False,
        },
        "unmeasured_nonzero_costs": [
            "ternary select and rank3 adder physical area/timing/power",
            "single-slot simultaneous retire/replace scheduler",
            "completed-pending stage1 state under result backpressure",
            "configuration generation/CRC logic and parameter load transactions",
            "trained H67 and Local5 codebook accuracy",
            "address-timed SRAM/DRAM transactions and full-network contention",
        ],
        "claim_boundary": contract["claim_boundary"],
    }


def write_output(output, result):
    output = Path(output)
    if output.exists():
        raise ValueError("refusing to overwrite M38 output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.contract.resolve())
    write_output(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
