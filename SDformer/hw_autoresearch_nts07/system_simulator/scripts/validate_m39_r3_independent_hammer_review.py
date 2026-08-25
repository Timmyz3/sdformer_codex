#!/usr/bin/env python3
"""Fail-closed independent validator for the M39-r3 model-only review."""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import tempfile
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / "contracts/m39_remaining_bottleneck_input_contract_r3_20260822.json"
ANALYZER = HW_ROOT / "system_simulator/scripts/analyze_m39_remaining_bottleneck_r3.py"
REGRESSION = HW_ROOT / "system_simulator/tests/test_m39_remaining_bottleneck_r3.py"
RESULT = HW_ROOT / "results/m39_remaining_bottleneck_r3_20260822/m39_remaining_bottleneck.json"
SPECIFICATION = HW_ROOT / "rtl_m39/M39_CURRENT_ANCHORED_BOTTLENECK_DSE_R3.md"
M38_ADMISSION = HW_ROOT / "contracts/m38_r5_independent_model_only_admission_r1_20260822.json"
M38_REVIEW = HW_ROOT / "results/m38_rst_math_protocol_reachable_r5_20260822/m38_r5_independent_hammer_go_review.json"
M38_VALIDATOR = HW_ROOT / "system_simulator/scripts/validate_m38_r5_independent_hammer_admission.py"
M35_RECEIPT = HW_ROOT / "contracts/m35_output_receipt_r3_20260822.json"
OPERATOR_CSV = Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/"
    "runs/h67_full_network_ledger_v2_multisample_vcs_20260821/"
    "operator_transactions.csv")

ANCHORS = {
    "contract": [
        "hw_autoresearch_nts07/contracts/m39_remaining_bottleneck_input_contract_r3_20260822.json",
        "bf58fbbc852f10a6f7695585ecbb2cc01e14ed06f59e2ec4a38f912e33ebc5e2"],
    "analyzer": [
        "hw_autoresearch_nts07/system_simulator/scripts/analyze_m39_remaining_bottleneck_r3.py",
        "9bbe19beecb55b1b3495352081309370d8909ad1001aa05efea147f7e645b470"],
    "regression": [
        "hw_autoresearch_nts07/system_simulator/tests/test_m39_remaining_bottleneck_r3.py",
        "53ec41f5a1babcf6bbe41b6fb4091a2fb53be48fcd1190f3efc0ddda9c867c34"],
    "result": [
        "hw_autoresearch_nts07/results/m39_remaining_bottleneck_r3_20260822/m39_remaining_bottleneck.json",
        "8923bbf5b1e630ad8e940ffa967f18ae9e59176c3f2dd6b29af2c1d696fbdcbb"],
    "specification": [
        "hw_autoresearch_nts07/rtl_m39/M39_CURRENT_ANCHORED_BOTTLENECK_DSE_R3.md",
        "9f84df8bc95424788eea902bad2cb558260e860ac412282c5a75ed50949d6649"],
}
UPSTREAM_ANCHORS = {
    "m38_admission": [str(M38_ADMISSION.relative_to(ROOT)),
                      "2d231c4a88d616158bcac0e867ec166a109fe8df55f10fc81182fc8ec01f08fe"],
    "m38_review": [str(M38_REVIEW.relative_to(ROOT)),
                   "36bb10294a209bd32ad4131d8b0171749aa50535083166dc38b5de5b28d2d529"],
    "m38_validator": [str(M38_VALIDATOR.relative_to(ROOT)),
                      "ce34da7dd759c0b43efc147a9b8f22f700414e17f7a8a9f1a3336c4afb64b445"],
    "m35_receipt_r3": [str(M35_RECEIPT.relative_to(ROOT)),
                       "d088daa8e51a40eb26ee07624f2c6a3b06f95bd0d1395c4bb91bdd1532195b84"],
    "operator_csv": [str(OPERATOR_CSV),
                     "15b9cf98dfe75f92d640c3d513f2837648a1064552d1c540b2a8499f97f280c2"],
}
BOTTLE_NAMES = [
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
]


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
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook,
                      parse_constant=reject_constant)


def mismatch(actual, expected, path="$"):
    if type(actual) is not type(expected):
        return "{} type {} != {}".format(
            path, type(actual).__name__, type(expected).__name__)
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return "{} key population differs".format(path)
        for key in sorted(actual):
            found = mismatch(actual[key], expected[key],
                             "{}.{}".format(path, key))
            if found is not None:
                return found
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return "{} list length differs".format(path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            found = mismatch(left, right, "{}[{}]".format(path, index))
            if found is not None:
                return found
        return None
    if actual != expected:
        return "{} value differs".format(path)
    return None


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "module load failed: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ceil_fraction(value):
    value = Fraction(value)
    return (value.numerator + value.denominator - 1) // value.denominator


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def validate_anchor_hashes():
    for name, pair in ANCHORS.items():
        path = ROOT / pair[0]
        require(path.is_file() and sha256(path) == pair[1],
                "M39-r3 anchor drift: {}".format(name))
    for name, pair in UPSTREAM_ANCHORS.items():
        path = Path(pair[0]) if Path(pair[0]).is_absolute() else ROOT / pair[0]
        require(path.is_file() and sha256(path) == pair[1],
                "M39-r3 upstream anchor drift: {}".format(name))


def validate_m38_rebuild():
    validator = load_module(M38_VALIDATOR, "m39_hammer_m38_validator")
    admission = validator.validate_admission(M38_ADMISSION, M38_REVIEW)
    review = validator.validate_review(M38_REVIEW)
    require(mismatch(admission, read_json(M38_ADMISSION)) is None,
            "M38 admission rebuild mismatch")
    require(mismatch(review, read_json(M38_REVIEW)) is None,
            "M38 review rebuild mismatch")
    require(admission["status"] ==
            "PASS_EXACT_M38_R5_PYTHON36_REFERENCE_MODEL_ONLY",
            "M38 scope drift")
    require(all(value is False for value in admission["forbidden"].values()),
            "M38 forbidden scope opened")


def validate_m35_boundary(result):
    receipt = read_json(M35_RECEIPT)
    require(receipt["independent_r3_review_required"] is True,
            "M35 independent-review boundary drift")
    require(receipt["paper_ppa_ready"] is False and
            receipt["headline_admitted"] is False and
            receipt["strict_fair_flat_standalone_comparison"][
                "integrated_density_admitted"] is False,
            "M35 forbidden scope opened")
    comparison = receipt["strict_fair_flat_standalone_comparison"]
    m33_area = Fraction(12997403898, 1000000)
    m35_area = Fraction(19633571938, 1000000)
    require(comparison["m35_over_m33_result_rate_per_area_exact"] == {
        "numerator": 12997403898, "denominator": 9816785969},
        "M35/M33 exact density drift")
    require(Fraction(2, 1) * m33_area / m35_area ==
            Fraction(12997403898, 9816785969),
            "M35/M33 independent density recomputation failed")
    audit = result["recursive_evidence_audit"]["m35_r3_standalone"]
    require(audit["independent_r3_review_required_by_receipt"] is True and
            audit["integrated_density_admitted"] is False,
            "M39 erased M35 review boundary")


def validate_independent_math(result):
    with OPERATOR_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["category"] == "bottleneck"]
    rows.sort(key=lambda row: row["name"])
    require([row["name"] for row in rows] == BOTTLE_NAMES,
            "independent bottleneck population drift")
    active = [int(row["activity_weighted_macs_per_frame"]) for row in rows]
    dense = [int(row["dense_macs_per_frame"]) for row in rows]
    baseline = [ceil_div(value, 96) for value in active]
    outputs = [int(row["output_elements_per_frame"]) for row in rows]
    weights = [int(row["weight_bytes_int8"]) for row in rows]
    inputs = [int(row["input_bytes_int8_per_frame"]) for row in rows]
    require(active == [2630357176, 947018995, 2898921692, 1168273912],
            "independent active-product ledger drift")
    require(sum(active) == 7644571775 and sum(dense) == 63700992000 and
            sum(baseline) == 79630957 and sum(outputs) == 9216000,
            "independent bottleneck aggregate drift")

    m4_population = 327131854
    projections = {}
    for line, accelerated in (("Local", 54565804), ("Motion", 52733277)):
        projections[line] = [ceil_fraction(Fraction(
            value * accelerated, m4_population * 96)) for value in active]
    require(projections["Local"] == [4570264, 1645452, 5036896, 2029884]
            and sum(projections["Local"]) == 13282496,
            "independent Local per-operator projection drift")
    require(projections["Motion"] == [4416777, 1590192, 4867738, 1961713]
            and sum(projections["Motion"]) == 12836420,
            "independent Motion per-operator projection drift")

    model = result["four_bottleneck_event_late_scale_model"]
    aggregate = model["aggregate"]
    require(aggregate["active_product_terms"] == sum(active) and
            aggregate["baseline_activity_cycles_96"] == sum(baseline),
            "result aggregate does not match independent source computation")
    require(len(model["resource_cycle_sensitivity"]) == 90,
            "resource sensitivity population drift")
    actual_grid = {}
    for row in model["resource_cycle_sensitivity"]:
        key = (row["line"], row["density_name"], row["lanes"], row["banks"])
        require(key not in actual_grid, "duplicate sensitivity row")
        actual_grid[key] = row
    require(len(actual_grid) == 90, "sensitivity key population drift")

    density_points = [
        ("observed_exact", None), ("density_5pct", Fraction(1, 20)),
        ("density_10pct", Fraction(1, 10)),
        ("density_20pct", Fraction(1, 5)),
        ("density_40pct", Fraction(2, 5)),
    ]
    for line, accelerated, frontend in (
            ("Local", 54565804, 6098531),
            ("Motion", 52733277, 6260784)):
        for density_name, ratio in density_points:
            point_active = (active if ratio is None else
                            [ceil_fraction(Fraction(value, 1) * ratio)
                             for value in dense])
            base96 = sum(ceil_div(value, 96) for value in point_active)
            control = ceil_fraction(Fraction(frontend * base96, m4_population))
            for lanes in (48, 96, 192):
                for banks in (12, 24, 48):
                    bank_bytes = banks * 4
                    service = min(lanes, bank_bytes)
                    expected_uncoalesced = sum(max(
                        ceil_div(value, lanes), ceil_div(value, bank_bytes))
                        for value in point_active)
                    expected_projected = sum(ceil_fraction(Fraction(
                        value * accelerated, m4_population * service))
                        for value in point_active)
                    expected_late = max(ceil_div(sum(outputs), 8),
                                        ceil_div(sum(outputs) * 3, bank_bytes))
                    row = actual_grid[(line, density_name, lanes, banks)]
                    require((row["active_product_terms"],
                             row["effective_event_service_width"],
                             row["uncoalesced_event_cycle_lower_bound"],
                             row["conditional_m4_projected_event_cycles"],
                             row["m35_late_scale_cycle_lower_bound"],
                             row["proportional_frontend_control_cycles"]) ==
                            (sum(point_active), service, expected_uncoalesced,
                             expected_projected, expected_late, control),
                            "independent sensitivity arithmetic drift: {}".format(
                                (line, density_name, lanes, banks)))
                    require(row["conditional_projection_only"] is True and
                            row["system_speedup_admitted"] is False,
                            "sensitivity scope opened")

    traffic = model["bandwidth_traffic_lower_bounds"]
    q24 = sum(outputs) * 3
    packed = ceil_div(sum(outputs), 8)
    fused = sum(weights) + sum(inputs) + packed
    materialized = sum(weights) + sum(inputs) + 2 * q24 + packed
    require((fused, materialized) == (31601664, 86897664),
            "independent traffic-byte recomputation drift")
    require((traffic["fused_compulsory_bytes_lower_bound"],
             traffic["materialized_compulsory_bytes_lower_bound"]) ==
            (fused, materialized) and
            traffic["address_timed_memory_admitted"] is False,
            "traffic result or boundary drift")
    storage = model["sram_capacity_lower_bounds"]
    preferred = 240 * 1024 - 52032
    hard = 408 * 1024 - 52032
    require((preferred, hard, ceil_div(sum(weights), preferred),
             ceil_div(sum(weights), hard)) == (193728, 365760, 110, 59),
            "independent SRAM lower-bound recomputation drift")
    require((storage["minimum_weight_tiles_preferred"],
             storage["minimum_weight_tiles_hard_cap"]) == (110, 59) and
            storage["macro_realization_admitted"] is False,
            "SRAM lower-bound result or boundary drift")

    dse_rows = {(row["line"], row["late_scale_implementation"]): row
                for row in result["conditional_dse"]["four_bottleneck_rows"]}
    expected = {
        ("Local", "M35_parallel_complement_CSD_sidecar"):
            (13282496, 1152000, 1484515, 15919011, 204743502),
        ("Motion", "M35_parallel_complement_CSD_sidecar"):
            (12836420, 1152000, 1524011, 15512431, 202666648),
    }
    for key, values in expected.items():
        row = dse_rows[key]
        require((row["replacement"]["conditional_m4_projected_event_cycles"],
                 row["replacement"]["late_scale_cycles"],
                 row["replacement"]["proportional_frontend_control_cycles"],
                 row["replacement"]["total_cycles"],
                 row["conditional_cycles_after_scope_substitution"]) == values,
                "conditional M35 row drift: {}".format(key))
        require(row["conditional_dse_only"] is True and
                row["system_speedup_admitted"] is False,
                "conditional DSE boundary opened")


def validate_adversarial_rejection(module):
    contract = read_json(CONTRACT)
    cases = []
    for path, value in (
            (("frozen_dse_rules", "bottleneck_operator_count"), True),
            (("sensitivity_rules", "nominal_lanes"), False),
            (("resource_and_admission_gates", "sram_banks"), True),
            (("external_comparison_boundary",
              "external_accelerator_normalized_comparison_admitted"), 0)):
        forged = copy.deepcopy(contract)
        forged[path[0]][path[1]] = value
        cases.append(forged)
    for path, value in (
            (("frozen_dse_rules", "bottleneck_operator_count"), 4.0),
            (("sensitivity_rules", "lane_points"), [48, 96.0, 192]),
            (("resource_and_admission_gates", "sram_banks"), 24.0)):
        forged = copy.deepcopy(contract)
        forged[path[0]][path[1]] = value
        cases.append(forged)
    recursive = copy.deepcopy(contract)
    recursive["sensitivity_rules"]["density_points"][0]["ratio"]["numerator"] = False
    cases.append(recursive)
    claim = copy.deepcopy(contract)
    claim["claim_boundary"] += " FORGED_SYSTEM_SPEEDUP"
    cases.append(claim)
    with tempfile.TemporaryDirectory(prefix="m39_r3_validator_") as directory:
        root = Path(directory)
        for index, forged in enumerate(cases):
            path = root / "forged_{}.json".format(index)
            path.write_text(json.dumps(forged, indent=2) + "\n", encoding="utf-8")
            try:
                module.build(path)
            except ValueError:
                pass
            else:
                raise ValueError("adversarial contract accepted: {}".format(index))
        canonical = CONTRACT.read_text(encoding="utf-8")
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"schema":"FORGED",' + canonical.lstrip()[1:],
                             encoding="utf-8")
        try:
            module.build(duplicate)
        except ValueError as error:
            require("duplicate JSON key" in str(error),
                    "duplicate key rejected for wrong reason")
        else:
            raise ValueError("duplicate JSON key accepted")
        for index, token in enumerate(("NaN", "Infinity", "-Infinity")):
            path = root / "constant_{}.json".format(index)
            path.write_text(canonical.replace(
                '"bottleneck_operator_count": 4',
                '"bottleneck_operator_count": {}'.format(token), 1),
                encoding="utf-8")
            try:
                module.build(path)
            except ValueError as error:
                require("non-standard JSON numeric constant" in str(error),
                        "non-standard constant rejected for wrong reason")
            else:
                raise ValueError("non-standard JSON constant accepted: {}".format(token))


def expected_review():
    return {
        "schema": "m39_r3_independent_hammer_review_v1",
        "status": "GO_M39_R3_MODEL_ONLY_CONDITIONAL_DSE",
        "date": "2026-08-22",
        "exact_anchors": ANCHORS,
        "upstream_anchors": UPSTREAM_ANCHORS,
        "validator": [
            "hw_autoresearch_nts07/system_simulator/scripts/validate_m39_r3_independent_hammer_review.py",
            sha256(Path(__file__).resolve()),
        ],
        "mandatory_rereview_passes": {
            "python36_regression": {"passed": 17, "failed": 0, "errors": 0},
            "repeat_build": {
                "runs": 2, "a_equals_b": True,
                "both_equal_frozen_result": True,
                "sha256": ANCHORS["result"][1],
            },
            "adversarial_matrix": {
                "duplicate_json_keys": {"tested": 1, "rejected": 1},
                "nonstandard_json_numeric_constants": {"tested": 3, "rejected": 3},
                "bool_integer_and_recursive_substitutions": {"tested": 5, "rejected": 5},
                "float_integer_substitutions": {"tested": 3, "rejected": 3},
                "claim_boundary_drift": {"tested": 1, "rejected": 1},
            },
            "independent_arithmetic": {
                "operators": 4,
                "active_product_terms": 7644571775,
                "dense_product_terms": 63700992000,
                "uncoalesced_96lane_24bank_cycles": 79630957,
                "local_per_operator_ceil_projection_cycles": 13282496,
                "motion_per_operator_ceil_projection_cycles": 12836420,
                "sensitivity_rows_recomputed": 90,
                "preferred_weight_tile_lower_bound": 110,
                "hard_cap_weight_tile_lower_bound": 59,
                "fused_compulsory_operand_bytes": 31601664,
                "materialized_compulsory_operand_bytes": 86897664,
            },
            "m38_r5_validator_rebuild": True,
            "m35_r3_relevant_standalone_math_recomputed": True,
            "m35_r3_general_independent_admission_generated": False,
        },
        "findings": {
            "p0": [],
            "p1": [],
            "p2": [
                {
                    "id": "P2_BANK_SERVICE_IS_CONTRACTUAL_NOT_MACRO_MEASURED",
                    "disposition": "nonblocking_model_only_scope_guard_preserved",
                    "detail": "Four bytes per bank per cycle and min(lanes,4*banks) are analytical assumptions; there is no target SRAM macro, conflict schedule, address trace, or timing proof.",
                },
                {
                    "id": "P2_DENSITY_SWEEP_IS_UNIFORM_HYPOTHETICAL_NOT_TRACE_REPLAY",
                    "disposition": "nonblocking_model_only_scope_guard_preserved",
                    "detail": "The four hypothetical densities scale every operator's dense products uniformly; they are sensitivity points, not measured Local/Motion distributions or structured reuse evidence.",
                },
                {
                    "id": "P2_M35_R3_NOT_GENERAL_INDEPENDENTLY_ADMITTED",
                    "disposition": "nonblocking_relevant_math_recomputed_no_scope_extension",
                    "detail": "The M35-r3 receipt still requires independent review. This review recomputes only the exact standalone fields consumed by M39 and does not create a general M35-r3 admission.",
                },
                {
                    "id": "P2_COMPULSORY_BYTES_HAVE_NO_PHYSICAL_MEMORY_BOUNDARY",
                    "disposition": "nonblocking_operand_volume_only",
                    "detail": "The 31,601,664/86,897,664-byte figures are abstract compulsory operand-volume formulas; they are not SRAM or DRAM transaction counts and cannot support energy or bandwidth claims.",
                },
            ],
        },
        "review": {
            "decision": "GO_MODEL_ONLY_CONDITIONAL_DSE",
            "independent_of_m39_r3_implementation": True,
            "score_0_to_100": 93,
            "p0": 0, "p1": 0, "p2": 4,
            "pass_admission_may_be_generated": True,
        },
        "admitted": {
            "exact_sha_bound_m39_r3_python36_model": True,
            "four_frozen_h67_bottleneck_source_rows_and_work_counts": True,
            "analytical_resource_and_operand_volume_lower_bounds": True,
            "conditional_dse_arithmetic": True,
            "m4_projection_executable_schedule": False,
            "bank_service_physical_realization": False,
            "density_sweep_trace_representative": False,
            "m35_r3_general_independent_admission": False,
            "local5_full_system": False,
            "integrated_rtl": False,
            "integrated_vcs_dc_sta_formality": False,
            "address_timed_memory": False,
            "sram_dram_realization": False,
            "ppa": False,
            "power_energy": False,
            "system_speedup": False,
            "external_accelerator_comparison": False,
            "headline": False,
            "best_paper": False,
        },
        "claim_boundary": "GO admits only the exact SHA-bound M39-r3 Python3.6 model, its four frozen H67 source-work rows, analytical lower-bound formulas, and conditional DSE arithmetic. The 4-byte/bank/cycle service, uniform density points, M4 effective-ratio projection, and M35 sidecar substitution are not an executable integrated schedule. Local5/full-system cycles, RTL, Synopsys closure, address-timed memory, macro realization, PPA, energy, system speedup, external comparison, DATE headline, and best-paper claims remain forbidden.",
        "next_gate": "Implement and independently verify a conflict-aware address-bearing four-bottleneck event-coalescing schedule with exact fixed-point M35 integration before any cycle or speedup admission.",
    }


def validate_review(path):
    validate_anchor_hashes()
    module = load_module(ANALYZER, "m39_r3_hammer_rebuild")
    rebuilt = module.build(CONTRACT)
    frozen = read_json(RESULT)
    require(mismatch(rebuilt, frozen) is None,
            "M39-r3 frozen result does not match rebuilt result")
    validate_m38_rebuild()
    validate_m35_boundary(rebuilt)
    validate_independent_math(rebuilt)
    validate_adversarial_rejection(module)
    review = read_json(path)
    expected = expected_review()
    found = mismatch(review, expected)
    require(found is None, "M39-r3 review drift: {}".format(found))
    require(review["review"]["p0"] == 0 and
            review["review"]["p1"] == 0 and
            review["review"]["decision"] == "GO_MODEL_ONLY_CONDITIONAL_DSE",
            "M39-r3 review is not model-only GO")
    require(all(review["admitted"][key] is False for key in (
        "m4_projection_executable_schedule", "bank_service_physical_realization",
        "density_sweep_trace_representative", "m35_r3_general_independent_admission",
        "local5_full_system", "integrated_rtl", "integrated_vcs_dc_sta_formality",
        "address_timed_memory", "sram_dram_realization", "ppa", "power_energy",
        "system_speedup", "external_accelerator_comparison", "headline",
        "best_paper")), "forbidden M39-r3 review scope opened")
    return review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    review = validate_review(args.review)
    print("M39_R3_INDEPENDENT_MODEL_ONLY_REVIEW_VALID=1")
    print("M39_R3_REVIEW_STATUS={}".format(review["status"]))


if __name__ == "__main__":
    main()
