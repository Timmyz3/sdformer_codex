#!/usr/bin/env python3
"""Build M39-r3 current-anchor, type-strict conditional bottleneck DSE."""

import argparse
import csv
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = HW_ROOT / (
    "contracts/m39_remaining_bottleneck_input_contract_r3_20260822.json")
EXPECTED_CONTRACT_SHA256 = (
    "bf58fbbc852f10a6f7695585ecbb2cc01e14ed06f59e2ec4a38f912e33ebc5e2")
R2_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m39_remaining_bottleneck_r2.py")
R2_ANALYZER_SHA256 = (
    "6c5efc9a7e5b74fbfe637c6952499bcd096d7c11884e680b968cb58d4790319d")

CSV_FIELDS = [
    "name", "operator", "scope", "category", "calls_per_frame",
    "input_elements_per_frame", "input_nonzero_per_frame", "input_activity",
    "output_elements_per_frame", "dense_macs_per_frame",
    "activity_weighted_macs_per_frame", "weight_elements",
    "weight_bytes_int8", "input_bytes_int8_per_frame",
    "output_bytes_int8_per_frame", "input_binary_packed_eligible",
    "input_bytes_binary_packed_per_frame", "dense_cycles_at_config_lanes",
    "activity_cycles_at_config_lanes", "replaced_by_attention_rtl_anchor",
    "input_shape_first", "output_shape_first",
]
BOTTLE_NAMES = [
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
]
FORBIDDEN_ADMISSION_KEYS = [
    "local5_full_system_admitted", "integrated_rtl_admitted",
    "integrated_rtl_vcs_admitted", "integrated_dc_sta_formality_admitted",
    "address_timed_memory_admitted", "sram_dram_realization_admitted",
    "trained_coverage_admitted", "ppa_admitted", "power_energy_admitted",
    "system_speedup_admitted", "external_accelerator_comparison_admitted",
    "headline_admitted", "best_paper_admitted",
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
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook, parse_constant=reject_constant)


def type_strict_mismatch(actual, expected, path="$"):
    if type(actual) is not type(expected):
        return "{} type {} != {}".format(
            path, type(actual).__name__, type(expected).__name__)
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return "{} key population differs".format(path)
        for key in sorted(actual):
            mismatch = type_strict_mismatch(
                actual[key], expected[key], "{}.{}".format(path, key))
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return "{} list length differs".format(path)
        for index, pair in enumerate(zip(actual, expected)):
            mismatch = type_strict_mismatch(
                pair[0], pair[1], "{}[{}]".format(path, index))
            if mismatch is not None:
                return mismatch
        return None
    if actual != expected:
        return "{} value differs".format(path)
    return None


def require_type_strict_equal(actual, expected, label):
    mismatch = type_strict_mismatch(actual, expected)
    require(mismatch is None,
            "{} recursive type-strict drift: {}".format(label, mismatch))


def exact_keys(payload, expected, label):
    require(isinstance(payload, dict) and set(payload) == set(expected),
            "{} population drift".format(label))


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "module import failed: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction_json(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def ceil_fraction(value):
    value = Fraction(value)
    return (value.numerator + value.denominator - 1) // value.denominator


def ceil_div(numerator, denominator):
    require(type(numerator) is int and type(denominator) is int
            and numerator >= 0 and denominator > 0,
            "ceil_div domain/type violation")
    return (numerator + denominator - 1) // denominator


def canonical_contract():
    require(sha256(DEFAULT_CONTRACT) == EXPECTED_CONTRACT_SHA256,
            "M39-r3 canonical contract identity drift")
    return read_json(DEFAULT_CONTRACT)


def load_contract(path=DEFAULT_CONTRACT):
    contract = read_json(path)
    require_type_strict_equal(contract, canonical_contract(), "M39-r3 contract")
    payloads, hashes, paths = {}, {}, {}
    for name, item in sorted(contract["inputs"].items()):
        exact_keys(item, {"path", "sha256"}, "M39-r3 input {}".format(name))
        source = resolve(item["path"])
        require(source.is_file(), "M39-r3 input missing: {}".format(name))
        actual = sha256(source)
        require(actual == item["sha256"],
                "M39-r3 input identity drift: {}".format(name))
        payloads[name] = (read_json(source) if source.suffix == ".json"
                          else source.read_text(encoding="utf-8"))
        hashes[name] = actual
        paths[name] = item["path"]
    for pair in contract["supersedes"]["artifacts"].values():
        target = resolve(pair[0])
        require(target.is_file() and sha256(target) == pair[1],
                "M39-r2 superseded artifact identity drift")
    return contract, payloads, hashes, paths


def verify_m38_current_anchor(contract, payloads):
    admission_path = resolve(contract["inputs"]["m38_model_only_admission"]["path"])
    review_path = resolve(contract["inputs"]["m38_independent_go_review"]["path"])
    validator_path = resolve(contract["inputs"]["m38_admission_validator"]["path"])
    validator = load_module(validator_path, "m39_r3_m38_admission_validator")
    rebuilt = validator.validate_admission(admission_path, review_path)
    require_type_strict_equal(
        payloads["m38_model_only_admission"], rebuilt,
        "M38-r5 model-only admission rebuild")
    rebuilt_review = validator.validate_review(review_path)
    require_type_strict_equal(
        payloads["m38_independent_go_review"], rebuilt_review,
        "M38-r5 independent review rebuild")
    admission = payloads["m38_model_only_admission"]
    review = payloads["m38_independent_go_review"]
    require(admission["status"] ==
            "PASS_EXACT_M38_R5_PYTHON36_REFERENCE_MODEL_ONLY",
            "M38-r5 admission status drift")
    require(review["status"] == "GO_M38_R5_TYPE_STRICT_REFERENCE_MODEL_ONLY"
            and review["review"]["p0"] == 0 and review["review"]["p1"] == 0,
            "M38-r5 independent GO drift")
    require(all(value is False for value in admission["forbidden"].values()),
            "M38-r5 forbidden scope opened")
    return {
        "admission_sha256": sha256(admission_path),
        "review_sha256": sha256(review_path),
        "validator_sha256": sha256(validator_path),
        "status": admission["status"],
        "review_decision": review["review"]["decision"],
        "review_score": review["review"]["score_0_to_100"],
        "python36_reference_model_admitted": True,
        "local_motion_system_cycles_admitted": False,
        "system_speedup_admitted": False,
    }


def m35_r2_expected_summary(receipt):
    vcs = receipt["vcs_r6"]
    dc = receipt["dc_sta_r7"]
    fm = receipt["formality_r7"]
    return {
        "schema": receipt["schema"], "status": receipt["status"],
        "vcs_r6": {
            "input_ledger_sha256": vcs["input_ledger_sha256"],
            "output_ledger_sha256": vcs["output_ledger_sha256"],
            "packets": 5120, "valid_products": 23680,
            "unstalled_functional_ii": 1, "outputs_per_packet": 8,
        },
        "dc_sta_r7": {
            "sealed_dc_evidence_sha256": dc["sealed_dc_evidence_sha256"],
            "admission_sha256": dc["admission_sha256"],
            "cell_area_um2": dc["cell_area_um2"],
            "clock_period_ns": dc["clock_period_ns"],
            "setup_wns_ns": dc["setup_wns_ns"],
            "hold_wns_ns": dc["hold_wns_ns"],
            "integer_multiplier_operators": dc["integer_multiplier_operators"],
        },
        "formality_r7": {
            "passing_compare_points": fm["passing_compare_points"],
            "failing_compare_points": fm["failing_compare_points"],
            "unmatched_compare_points": fm["unmatched_compare_points"],
            "self_contained_snapshot_ledger_sha256":
            fm["self_contained_snapshot"]["evidence_ledger_sha256"],
            "authority":
            "SELF_CONTAINED_SNAPSHOT_ONLY_LIVE_WRAPPER_DRIFT_IGNORED",
        },
    }


def verify_m35_r3(contract, receipt):
    exact_keys(receipt, {
        "schema", "status", "date", "supersedes", "m35_r2_recursive_anchor",
        "m33_final_recursive_anchor", "strict_fair_flat_standalone_comparison",
        "claim_boundary", "paper_ppa_ready", "independent_r3_review_required",
        "headline_admitted",
    }, "M35-r3 receipt")
    require(receipt["schema"] == "m35_output_receipt_v3" and
            receipt["status"] ==
            "PASS_M35_R2_R7_AND_STRICT_FLAT_M33_STANDALONE_COMPARISON_NO_SYSTEM_OR_PAPER_PPA_CLAIM",
            "M35-r3 receipt identity drift")
    require(receipt["paper_ppa_ready"] is False
            and receipt["headline_admitted"] is False
            and receipt["independent_r3_review_required"] is True,
            "M35-r3 receipt boundary drift")
    predecessor = resolve(receipt["supersedes"]["path"])
    require(sha256(predecessor) == receipt["supersedes"]["sha256"] ==
            "63b61a88213e3882a0ad3a67c3e74047291c920d80a237bcdd44a8e84dcb5d5e",
            "M35-r3 predecessor identity drift")
    r2_receipt = read_json(predecessor)
    require_type_strict_equal(
        receipt["m35_r2_recursive_anchor"], m35_r2_expected_summary(r2_receipt),
        "M35-r3 recursive r2 summary")

    require(sha256(R2_ANALYZER) == R2_ANALYZER_SHA256,
            "M39-r2 helper analyzer identity drift")
    r2 = load_module(R2_ANALYZER, "m39_r3_frozen_r2_helper")
    r2.load_json = read_json
    m35_audit = r2.verify_m35_receipt(r2_receipt)

    m33_spec = receipt["m33_final_recursive_anchor"]
    m33_path = resolve(m33_spec["path"])
    require(sha256(m33_path) == m33_spec["sha256"] ==
            "9d670a6e950c3d0a1d934004901b9380a021b6d2375d3c96cc139bac96aa766e",
            "M35-r3 M33 anchor identity drift")
    m33_receipt = read_json(m33_path)
    m33_audit = r2.verify_m33_receipt(m33_receipt)
    require(m33_spec["schema"] == m33_receipt["schema"]
            and m33_spec["status"] == m33_receipt["status"],
            "M35-r3 M33 summary drift")
    require_type_strict_equal(m33_spec["dc_sta_flat_r2"], {
        "sealed_dc_evidence_sha256":
        m33_receipt["dc_sta_flat_r2"]["sealed_dc_evidence_sha256"],
        "admission_sha256": m33_receipt["dc_sta_flat_r2"]["admission_sha256"],
        "cell_area_um2": m33_receipt["dc_sta_flat_r2"]["cell_area_um2"],
        "clock_period_ns": m33_receipt["dc_sta_flat_r2"]["clock_period_ns"],
        "setup_wns_ns": m33_receipt["dc_sta_flat_r2"]["setup_wns_ns"],
        "hold_wns_ns": m33_receipt["dc_sta_flat_r2"]["hold_wns_ns"],
    }, "M35-r3 M33 DC summary")
    require_type_strict_equal(m33_spec["formality_flat_r2"], {
        "passing_compare_points": 655, "failing_compare_points": 0,
        "unmatched_compare_points": 0,
        "self_contained_snapshot_ledger_sha256":
        m33_receipt["formality_flat_r2"]["self_contained_snapshot"][
            "evidence_ledger_sha256"],
        "authority": "SELF_CONTAINED_SNAPSHOT_ONLY",
    }, "M35-r3 M33 Formality summary")

    comparison = receipt["strict_fair_flat_standalone_comparison"]
    m33_area = Fraction(str(m33_audit["standalone_area_um2"]))
    m35_area = Fraction(str(m35_audit["standalone_area_um2"]))
    require_type_strict_equal(comparison, {
        "comparison_status":
        "ADMITTED_STANDALONE_ONLY_M33_AND_M35_FORMALITY_CLOSED",
        "common_technology_nm": 28, "common_clock_period_ns": 2.0,
        "common_qualification": "ZERO_WIRE_IDEAL_CLOCK_FLAT_DC_NOT_POST_LAYOUT",
        "m33_area_um2": float(m33_area), "m35_area_um2": float(m35_area),
        "m35_over_m33_area_exact": fraction_json(m35_area / m33_area),
        "m35_over_m33_area_decimal": float(m35_area / m33_area),
        "m33_results_per_cycle": 4, "m35_results_per_cycle": 8,
        "m35_over_m33_peak_result_rate_exact": fraction_json(Fraction(2, 1)),
        "m35_over_m33_result_rate_per_area_exact":
        fraction_json(Fraction(2, 1) * m33_area / m35_area),
        "m35_over_m33_result_rate_per_area_decimal":
        float(Fraction(2, 1) * m33_area / m35_area),
        "m35_area_per_result_reduction_exact":
        fraction_json(1 - m35_area / (2 * m33_area)),
        "m35_area_per_result_reduction_decimal":
        float(1 - m35_area / (2 * m33_area)),
        "m35_area_per_result_reduction_percent":
        (1.0 - float(m35_area) / (2.0 * float(m33_area))) * 100.0,
        "strict_fair_density_admitted": True,
        "integrated_density_admitted": False,
    }, "M35-r3 standalone comparison")

    math_result = read_json(resolve(r2_receipt["math_identity"]["result"][0]))
    threshold_rows = {row["producer"]: row for row in math_result["thresholds"]}
    return {
        "receipt_sha256": contract["inputs"]["m35_receipt_r3"]["sha256"],
        "receipt_status": receipt["status"],
        "m35_r2": m35_audit, "m33_final": m33_audit,
        "strict_fair_flat_density_admitted": True,
        "integrated_density_admitted": False,
        "independent_r3_review_required_by_receipt": True,
        "m33_area_um2": float(m33_area), "m35_area_um2": float(m35_area),
        "threshold_rows": threshold_rows,
    }


def verify_system_inputs(contract, payloads, hashes, m35_audit):
    rules = contract["frozen_dse_rules"]
    m22 = payloads["m22_summary"]
    require(m22["status"] ==
            "PASS_FROZEN_INPUT_PARTIAL_TRANSACTION_LEDGER_NOT_DRAMSIM_OR_SPEEDUP",
            "M22 status drift")
    require(m22["identities"]["local_ep44"]["attention_execution_records"] == 0
            and m22["identities"]["local_ep44"]["attention_coverage_status"] ==
            "MISSING_FROM_EXECUTION_TRACE_NOT_ZERO_COST",
            "Local5 fail-closed trace boundary drift")
    require(m22["identities"]["h67_ep35"]["attention_execution_records"] == 120,
            "H67 attention population drift")

    m25 = payloads["m25_cycle_ledger"]
    require(m25["status"] ==
            "PASS_FROZEN_C4_TILING_AND_CYCLE_ENVELOPE_HEADLINE_NO_GO",
            "M25 status drift")
    require(m25["attention_completeness"]["Local5"]["speedup"] == "UNKNOWN"
            and m25["attention_completeness"]["Local5"][
                "minimum_missing_module_calls"] == 120,
            "M25 Local5 completeness drift")
    local = m25["compute_envelopes"]["local"]["10"]
    motion = m25["compute_envelopes"]["hybrid"]["10"]
    require(local["accelerated_m4_cycles"] == rules["local_m4_accelerated_cycles"]
            and motion["accelerated_m4_cycles"] ==
            rules["motion_m4_accelerated_cycles"],
            "M25 M4 accelerated-cycle anchor drift")
    require(local["m21_fifo4_phase1_incremental_cycles"] ==
            rules["local_frontend_increment_cycles"]
            and motion["m21_fifo4_phase1_incremental_cycles"] ==
            rules["motion_frontend_increment_cycles"],
            "M25 frontend-cycle anchor drift")

    m26 = payloads["m26_factor_lower_bound"]
    m30 = payloads["m30_system_dse"]
    require(m26["schema"] == "m26_atlif_factor_arithmetic_lower_bound_v2"
            and m30["identity"]["m26_sha256"] == hashes["m26_factor_lower_bound"],
            "M26/M30 recursive identity drift")
    candidates = {row["name"]: row for row in m30["port_candidates"]}
    selected = candidates[rules["selected_m30_candidate"]]
    require((selected["local_cycles"], selected["motion_cycles"],
             selected["t10_cycles"]) ==
            (rules["selected_m30_local_cycles"],
             rules["selected_m30_motion_cycles"], rules["m30_t10_cycles"]),
            "M30 selected candidate drift")
    require(candidates["384b_independent_output_packed24"]["local_cycles"] ==
            selected["local_cycles"] + 24
            and candidates["384b_independent_output_packed24"]["motion_cycles"] ==
            selected["motion_cycles"] + 24,
            "M30 384-bit comparison drift")

    m32 = payloads["m32_threshold_carry"]
    require(m32["status"] ==
            "PASS_H67_EP35_S10_EXACT_RUNTIME_DATAFLOW_REAL_DOMAIN_SEMANTIC_ADMISSION_ONLY",
            "M32 status drift")
    require(m32["candidate_census"]["semantically_admitted_operators"] == 10
            and m32["candidate_census"][
                "semantically_admitted_cycles_candidate_population"] ==
            rules["consumer_population_cycles"]
            and m32["candidate_census"][
                "semantically_admitted_outputs_per_sample"] ==
            rules["consumer_outputs_per_sample"],
            "M32 population drift")
    require(m32["admission"]["semantic_admission"] is True
            and m32["admission"]["fixed_point_admitted"] is False
            and m32["admission"]["system_cycle_admitted"] is False,
            "M32 admission boundary drift")
    balanced = {row["line"]: row for row in
                m32["control_charged_cycle_sensitivity"]["rows"]
                if row["variant"] == "balanced_radix20_exact_product"}
    require((balanced["local"]["event_accumulation_cycles_borrowed"],
             balanced["local"]["late_scale_cycles_arithmetic"],
             balanced["local"]["proportional_frontend_control_cycles"]) ==
            (17662220, 7614000, 1974013), "M32 Local row drift")
    require((balanced["motion"]["event_accumulation_cycles_borrowed"],
             balanced["motion"]["late_scale_cycles_arithmetic"],
             balanced["motion"]["proportional_frontend_control_cycles"]) ==
            (17069055, 7614000, 2026532), "M32 Motion row drift")

    dual = payloads["h67_dual_line_contract"]
    categories = dual["coverage"]["categories"]
    expected_categories = {
        "bottleneck": (79630957, 0),
        "patch_embed": (199420620, 172321077),
        "ffn_expand": (118370114, 100895624),
        "downsample": (21012750, 12321697),
        "prediction": (271156, 179459),
        "attention_q_projection": (14536040, 14536040),
        "attention_k_projection": (14536040, 14536040),
    }
    for name, expected in expected_categories.items():
        require((categories[name]["cycles"], categories[name]["eligible_cycles"])
                == expected, "H67 category drift: {}".format(name))

    reader = csv.DictReader(payloads["h67_operator_transactions"].splitlines())
    require(reader.fieldnames == CSV_FIELDS, "operator CSV column population/order drift")
    rows = list(reader)
    bottlenecks = sorted(
        [row for row in rows if row["category"] == "bottleneck"],
        key=lambda row: row["name"])
    require([row["name"] for row in bottlenecks] == BOTTLE_NAMES,
            "four-bottleneck operator population drift")
    k = rules["bottleneck_input_channels"] * rules["bottleneck_kernel_positions"]
    n = rules["bottleneck_output_channels"]
    m = rules["bottleneck_spatiotemporal_rows"]
    for row in bottlenecks:
        require(row["operator"] == "Conv2d" and row["scope"] == "bottleneck"
                and row["calls_per_frame"] == "1"
                and json.loads(row["input_shape_first"]) == [10, 1, 768, 15, 20]
                and json.loads(row["output_shape_first"]) == [10, 1, 768, 15, 20],
                "bottleneck shape/operator drift: {}".format(row["name"]))
        products = int(row["activity_weighted_macs_per_frame"])
        require(int(row["dense_macs_per_frame"]) == m * k * n
                and int(row["weight_bytes_int8"]) == k * n
                and int(row["output_elements_per_frame"]) == m * n
                and int(row["activity_cycles_at_config_lanes"]) ==
                ceil_div(products, 96),
                "bottleneck arithmetic identity drift: {}".format(row["name"]))

    m32_bottlenecks = {row["name"]: row for row in
                       m32["candidate_census"]["candidates"]
                       if row["category"] == "bottleneck"}
    require(set(m32_bottlenecks) == set(BOTTLE_NAMES),
            "M32 bottleneck population drift")
    for row in bottlenecks:
        candidate = m32_bottlenecks[row["name"]]
        require(candidate["baseline_activity_cycles"] ==
                int(row["activity_cycles_at_config_lanes"])
                and candidate["output_elements_per_sample"] ==
                int(row["output_elements_per_frame"])
                and candidate["producer"] in m35_audit["threshold_rows"],
                "M32/M35 bottleneck cross-anchor drift: {}".format(row["name"]))
    require(sum(int(row["activity_cycles_at_config_lanes"])
                for row in bottlenecks) == rules["bottleneck_population_cycles"]
            and sum(int(row["output_elements_per_frame"])
                    for row in bottlenecks) == rules["bottleneck_outputs_per_sample"],
            "four-bottleneck aggregate drift")
    return {"local": local, "motion": motion, "selected": selected,
            "balanced": balanced, "categories": categories,
            "bottlenecks": bottlenecks, "m32_bottlenecks": m32_bottlenecks}


def target_gates(fixed, ideal, population, replacement):
    rows = []
    for target in (Fraction(27, 10), Fraction(3, 1)):
        ceiling = Fraction(fixed, 1) / target
        required = Fraction(ideal, 1) - ceiling
        maximum = Fraction(population, 1) - required
        rows.append({
            "target_conditional_compute_speedup": fraction_json(target),
            "target_cycle_ceiling": fraction_json(ceiling),
            "maximum_scope_replacement_cycles": fraction_json(maximum),
            "replacement_headroom_cycles":
            fraction_json(maximum - Fraction(replacement, 1)),
            "crosses_in_conditional_dse": Fraction(replacement, 1) <= maximum,
            "system_speedup_admitted": False,
        })
    return rows


def scope_row(scope, line, ideal, before, event, late, control,
              implementation, fixed):
    replacement = event + late + control
    after = ideal - before + replacement
    require(after + before == ideal + replacement,
            "M39-r3 scope conservation failure")
    return {
        "scope": scope, "line": line,
        "late_scale_implementation": implementation,
        "before_cycles": before,
        "replacement": {
            "conditional_m4_projected_event_cycles": event,
            "late_scale_cycles": late,
            "proportional_frontend_control_cycles": control,
            "overlap_credit_cycles": 0,
            "total_cycles": replacement,
        },
        "savings_cycles": before - replacement,
        "m38_model_substituted_ideal_before_scope_cycles": ideal,
        "conditional_cycles_after_scope_substitution": after,
        "conditional_compute_speedup_vs_fixed_exact":
        fraction_json(Fraction(fixed, after)),
        "target_gates": target_gates(fixed, ideal, before, replacement),
        "conditional_dse_only": True,
        "system_speedup_admitted": False,
    }


def build_bottleneck_model(contract, validated, m35_audit):
    rules = contract["frozen_dse_rules"]
    sensitivity = contract["sensitivity_rules"]
    resource = contract["resource_and_admission_gates"]
    rows = validated["bottlenecks"]
    m32_rows = validated["m32_bottlenecks"]
    total_dense = sum(int(row["dense_macs_per_frame"]) for row in rows)
    total_active = sum(int(row["activity_weighted_macs_per_frame"]) for row in rows)
    total_outputs = sum(int(row["output_elements_per_frame"]) for row in rows)
    total_weights = sum(int(row["weight_bytes_int8"]) for row in rows)
    total_inputs = sum(int(row["input_bytes_int8_per_frame"]) for row in rows)
    require(total_active == 7644571775 and total_dense == 63700992000,
            "four-bottleneck exact work drift")

    line_parameters = {
        "Local": (rules["local_m4_accelerated_cycles"],
                  rules["local_frontend_increment_cycles"]),
        "Motion": (rules["motion_m4_accelerated_cycles"],
                   rules["motion_frontend_increment_cycles"]),
    }
    operator_rows = []
    for row in rows:
        name = row["name"]
        producer = m32_rows[name]["producer"]
        threshold = m35_audit["threshold_rows"][producer]
        active = int(row["activity_weighted_macs_per_frame"])
        dense = int(row["dense_macs_per_frame"])
        outputs = int(row["output_elements_per_frame"])
        projections = {}
        for line, pair in line_parameters.items():
            projections[line] = ceil_fraction(
                Fraction(active * pair[0],
                         rules["m4_profiled_population_cycles"] * 96))
        operator_rows.append({
            "name": name, "producer": producer,
            "input_shape": json.loads(row["input_shape_first"]),
            "output_shape": json.loads(row["output_shape_first"]),
            "input_nonzero_elements": int(row["input_nonzero_per_frame"]),
            "input_elements": int(row["input_elements_per_frame"]),
            "input_density_exact": fraction_json(Fraction(
                int(row["input_nonzero_per_frame"]),
                int(row["input_elements_per_frame"]))),
            "dense_product_terms": dense, "active_product_terms": active,
            "product_density_exact": fraction_json(Fraction(active, dense)),
            "baseline_activity_cycles_96":
            int(row["activity_cycles_at_config_lanes"]),
            "nominal_conditional_m4_projection_cycles_by_line": projections,
            "output_elements": outputs,
            "m35_late_scale_cycles_at_8_per_cycle": ceil_div(outputs, 8),
            "unique_int8_weight_bytes": int(row["weight_bytes_int8"]),
            "q24_intermediate_bytes": outputs * 3,
            "packed_output_bytes": ceil_div(outputs, 8),
            "m35_threshold_uq0p24_raw": threshold["threshold_uq0p24_raw"],
            "m35_csd_nonzero_terms": threshold["csd_nonzero_terms"],
            "fixed_trace_source_row_admitted": True,
            "integrated_cycle_admitted": False,
        })

    observed_density = {
        "name": "observed_exact",
        "ratio": fraction_json(Fraction(total_active, total_dense)),
    }
    density_points = [observed_density] + sensitivity["density_points"]
    grid = []
    for line in ("Local", "Motion"):
        accelerated, frontend = line_parameters[line]
        for density in density_points:
            ratio = Fraction(density["ratio"]["numerator"],
                             density["ratio"]["denominator"])
            active_by_operator = [
                (int(row["activity_weighted_macs_per_frame"])
                 if density["name"] == "observed_exact"
                 else ceil_fraction(Fraction(
                     int(row["dense_macs_per_frame"]), 1) * ratio))
                for row in rows]
            active = sum(active_by_operator)
            # Operators are distinct schedule regions; never borrow a rounding
            # credit by merging their final partial cycles.
            baseline96 = sum(ceil_div(value, 96)
                             for value in active_by_operator)
            control = ceil_fraction(Fraction(
                frontend * baseline96, rules["m4_profiled_population_cycles"]))
            for lanes in sensitivity["lane_points"]:
                for banks in sensitivity["bank_points"]:
                    bank_bytes = banks * sensitivity[
                        "bytes_per_bank_read_or_write"]
                    service = min(lanes, bank_bytes)
                    uncoalesced = sum(
                        max(ceil_div(value, lanes),
                            ceil_div(value, bank_bytes))
                        for value in active_by_operator)
                    projected = sum(
                        ceil_fraction(Fraction(
                            value * accelerated,
                            rules["m4_profiled_population_cycles"] * service))
                        for value in active_by_operator)
                    late_compute = ceil_div(total_outputs,
                                            sensitivity["m35_outputs_per_cycle"])
                    late_read = ceil_div(
                        total_outputs * sensitivity[
                            "q24_intermediate_bytes_per_output"], bank_bytes)
                    late = max(late_compute, late_read)
                    grid.append({
                        "line": line, "density_name": density["name"],
                        "density_exact": density["ratio"],
                        "active_product_terms": active, "lanes": lanes,
                        "banks": banks, "bank_payload_bytes_per_cycle": bank_bytes,
                        "effective_event_service_width": service,
                        "uncoalesced_event_cycle_lower_bound": uncoalesced,
                        "conditional_m4_projected_event_cycles": projected,
                        "m35_late_scale_cycle_lower_bound": late,
                        "proportional_frontend_control_cycles": control,
                        "conditional_serial_replacement_cycles":
                        projected + late + control,
                        "minimum_effective_event_work_reduction_required_exact":
                        fraction_json(Fraction(uncoalesced, projected)),
                        "conditional_projection_only": True,
                        "system_speedup_admitted": False,
                    })
    require(len(grid) == 90, "sensitivity grid population drift")

    preferred_available = (resource["sram_preferred_kib"] * 1024
                           - resource["fixed_resident_bytes"])
    hard_available = (resource["sram_hard_cap_kib"] * 1024
                      - resource["fixed_resident_bytes"])
    q24_bytes = total_outputs * sensitivity["q24_intermediate_bytes_per_output"]
    packed_bytes = ceil_div(
        total_outputs * sensitivity["packed_output_bits_per_output"], 8)
    materialized_compulsory = total_weights + total_inputs + 2 * q24_bytes + packed_bytes
    fused_compulsory = total_weights + total_inputs + packed_bytes
    nominal_bank_bytes = (sensitivity["nominal_banks"]
                          * sensitivity["bytes_per_bank_read_or_write"])
    traffic = {
        "unique_int8_weight_bytes": total_weights,
        "input_activation_bytes": total_inputs,
        "q24_intermediate_bytes": q24_bytes,
        "q24_materialized_write_plus_read_bytes": 2 * q24_bytes,
        "packed_output_bytes": packed_bytes,
        "fused_compulsory_bytes_lower_bound": fused_compulsory,
        "materialized_compulsory_bytes_lower_bound": materialized_compulsory,
        "fused_compulsory_bank_cycles_lower_bound_nominal":
        ceil_div(fused_compulsory, nominal_bank_bytes),
        "materialized_compulsory_bank_cycles_lower_bound_nominal":
        ceil_div(materialized_compulsory, nominal_bank_bytes),
        "uncoalesced_event_weight_stream_bytes": total_active,
        "uncoalesced_event_weight_bank_cycles_nominal":
        sum(ceil_div(int(row["activity_weighted_macs_per_frame"]),
                     nominal_bank_bytes) for row in rows),
        "traffic_lower_bound_scope":
        "COMPULSORY_ONLY_EXCLUDES_PARTIAL_SUM_METADATA_INDEX_REPLAY_AND_DRAM_TIMING",
        "address_timed_memory_admitted": False,
    }
    storage = {
        "fixed_resident_bytes": resource["fixed_resident_bytes"],
        "preferred_available_bytes_after_fixed": preferred_available,
        "hard_available_bytes_after_fixed": hard_available,
        "four_operator_weight_bytes": total_weights,
        "minimum_weight_tiles_preferred": ceil_div(total_weights, preferred_available),
        "minimum_weight_tiles_hard_cap": ceil_div(total_weights, hard_available),
        "full_q24_intermediate_bytes": q24_bytes,
        "all_weights_fit_preferred": total_weights <= preferred_available,
        "all_weights_fit_hard_cap": total_weights <= hard_available,
        "full_q24_intermediate_fits_hard_cap": q24_bytes <= hard_available,
        "macro_realization_admitted": False,
    }
    return {
        "operator_census": operator_rows,
        "aggregate": {
            "operators": 4, "dense_product_terms": total_dense,
            "active_product_terms": total_active,
            "observed_product_density_exact": observed_density["ratio"],
            "baseline_activity_cycles_96":
            sum(int(row["activity_cycles_at_config_lanes"]) for row in rows),
            "outputs": total_outputs,
        },
        "resource_cycle_sensitivity": grid,
        "bandwidth_traffic_lower_bounds": traffic,
        "sram_capacity_lower_bounds": storage,
        "hard_lower_bound_interpretation":
        "WITHOUT_PROVEN_EVENT_COALESCING_THE_96_LANE_24_BANK_EVENT_BOUND_EQUALS_THE_FROZEN_ACTIVITY_BASELINE",
        "conditional_projection_interpretation":
        "M25_EFFECTIVE_M4_RATIO_PROJECTED_ONTO_BOTTLENECK_WORK_NOT_AN_EXECUTABLE_SCHEDULE",
    }


def build_conditional_dse(contract, validated, bottleneck):
    rules = contract["frozen_dse_rules"]
    fixed = rules["fixed_compute_cycles"]
    ideals = {
        "Local": rules["selected_m30_local_cycles"] - rules["m30_t10_cycles"]
        + rules["m38_r5_conditional_t10_cycles"],
        "Motion": rules["selected_m30_motion_cycles"] - rules["m30_t10_cycles"]
        + rules["m38_r5_conditional_t10_cycles"],
    }
    require(ideals == {"Local": 268455448, "Motion": 266785174},
            "M38-r5 conditional ideal reconciliation drift")
    nominal = {(row["line"], row["density_name"]): row for row in
               bottleneck["resource_cycle_sensitivity"]
               if row["lanes"] == 96 and row["banks"] == 24}
    four_rows, ten_rows = [], []
    for line in ("Local", "Motion"):
        row = nominal[(line, "observed_exact")]
        for implementation, outputs_per_cycle in (
                ("M33_shared96_generic_UQ0p24", 4),
                ("M35_parallel_complement_CSD_sidecar", 8)):
            late = ceil_div(rules["bottleneck_outputs_per_sample"],
                            outputs_per_cycle)
            four_rows.append(scope_row(
                "four_bottleneck_conv3x3", line, ideals[line],
                rules["bottleneck_population_cycles"],
                row["conditional_m4_projected_event_cycles"], late,
                row["proportional_frontend_control_cycles"],
                implementation, fixed))
        source = validated["balanced"][line.lower()]
        for implementation, outputs_per_cycle in (
                ("M33_shared96_generic_UQ0p24", 4),
                ("M35_parallel_complement_CSD_sidecar", 8)):
            ten_rows.append(scope_row(
                "ten_semantically_admitted_consumers", line, ideals[line],
                rules["consumer_population_cycles"],
                source["event_accumulation_cycles_borrowed"],
                ceil_div(rules["consumer_outputs_per_sample"], outputs_per_cycle),
                source["proportional_frontend_control_cycles"],
                implementation, fixed))
    return {
        "fixed_compute_reference_cycles": fixed,
        "selected_m30_anchor": {
            "name": rules["selected_m30_candidate"],
            "Local_cycles": rules["selected_m30_local_cycles"],
            "Motion_cycles": rules["selected_m30_motion_cycles"],
        },
        "m38_r5_model_only_t10_substitution": {
            "old_t10_cycles": rules["m30_t10_cycles"],
            "conditional_model_t10_cycles": rules["m38_r5_conditional_t10_cycles"],
            "conditional_model_t10_ii": rules["m38_r5_conditional_t10_ii"],
            "Local_ideal_cycles": ideals["Local"],
            "Motion_ideal_cycles": ideals["Motion"],
            "conditional_model_substitution_math_admitted": True,
            "integrated_cycle_or_system_speedup_admitted": False,
        },
        "four_bottleneck_rows": four_rows,
        "ten_consumer_legacy_reconciled_rows": ten_rows,
        "scope_alternatives_not_additive": True,
        "overlap_credit_cycles": 0,
        "conditional_dse_math_admitted": True,
        "system_speedup_admitted": False,
    }


def remaining_cycle_ledger(validated):
    categories = validated["categories"]
    rows = []
    for name in ("bottleneck", "patch_embed", "ffn_expand", "downsample",
                 "prediction"):
        source = categories[name]
        rows.append({
            "category": name, "total_cycles": source["cycles"],
            "already_m4_eligible_cycles": source["eligible_cycles"],
            "remaining_noneligible_cycles":
            source["cycles"] - source["eligible_cycles"],
        })
    noneligible = sum(row["remaining_noneligible_cycles"] for row in rows)
    qk = (categories["attention_q_projection"]["cycles"]
          + categories["attention_k_projection"]["cycles"])
    require(noneligible == 132987740 and qk == 29072080
            and noneligible + qk == 162059820,
            "remaining-cycle decomposition drift")
    return {
        "noneligible_operator_cycles": noneligible,
        "q_projection_cycles": 14536040, "k_projection_cycles": 14536040,
        "qk_cycles": qk, "noneligible_plus_qk_cycles": noneligible + qk,
        "categories": rows,
        "independent_reduction_ceilings": [
            {"scope": "four_bottleneck_conv3x3", "cycles": 79630957,
             "can_save_50m_alone_if_replacement_le_29630957": True},
            {"scope": "qk_plus_rqtb_attention", "cycles": 32162811,
             "can_save_50m_alone": False},
            {"scope": "patch_embed_remaining", "cycles": 27099543,
             "can_save_50m_alone": False},
            {"scope": "ffn_expand_remaining", "cycles": 17474490,
             "can_save_50m_alone": False},
            {"scope": "downsample_remaining", "cycles": 8691053,
             "can_save_50m_alone": False},
            {"scope": "prediction_remaining", "cycles": 91697,
             "can_save_50m_alone": False},
        ],
    }


def build(contract_path=DEFAULT_CONTRACT):
    contract, payloads, hashes, paths = load_contract(contract_path)
    m38_audit = verify_m38_current_anchor(contract, payloads)
    m35_audit = verify_m35_r3(contract, payloads["m35_receipt_r3"])
    validated = verify_system_inputs(contract, payloads, hashes, m35_audit)
    bottleneck = build_bottleneck_model(contract, validated, m35_audit)
    dse = build_conditional_dse(contract, validated, bottleneck)
    result = {
        "schema": "m39_remaining_bottleneck_v3",
        "status": "PASS_M39_R3_CURRENT_ANCHORS_CONDITIONAL_BOTTLENECK_DSE_ONLY",
        "identity": {
            "contract": "hw_autoresearch_nts07/contracts/{}".format(
                Path(contract_path).name),
            "contract_sha256": sha256(contract_path),
            "analyzer": "hw_autoresearch_nts07/system_simulator/scripts/{}".format(
                Path(__file__).name),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
            "verified_input_paths": paths,
        },
        "supersedes": contract["supersedes"],
        "scope_boundaries": {
            "Local": {"definition": contract["scope_definitions"]["Local"],
                      "conditional_dse_math_admitted": True,
                      "system_speedup_admitted": False},
            "Motion": {"definition": contract["scope_definitions"]["Motion"],
                       "conditional_dse_math_admitted": True,
                       "system_speedup_admitted": False},
            "four_bottleneck_conv3x3": {
                "definition": contract["scope_definitions"][
                    "four_bottleneck_conv3x3"],
                "source_work_and_lower_bounds_admitted": True,
                "conditional_replacement_only": True},
            "Local5_ep44": {"definition": contract["scope_definitions"][
                "Local5_ep44"], "full_system_admitted": False},
        },
        "recursive_evidence_audit": {
            "m38_r5_model_only": m38_audit,
            "m35_r3_standalone": {key: value for key, value in m35_audit.items()
                                  if key != "threshold_rows"},
        },
        "r2_formula_audit": {
            "stale_m38_anchor_removed": True,
            "m35_r2_replaced_by_r3_receipt": True,
            "duplicate_key_safe_json": True,
            "recursive_type_strict_contract_and_anchor_equality": True,
            "aggregate_event_constants_now_derived_from_four_rows": True,
            "uncoalesced_hard_bound_separated_from_m4_projection": True,
            "per_operator_and_resource_sensitivity_added": True,
            "absolute_external_trace_paths_remain_hash_bound_nonportable": True,
        },
        "attention_and_trace_completeness": {
            "H67": "120_ATTENTION_ROWS_ABSTRACT_COMPUTE_ANCHOR_NOT_PHYSICAL_TRAFFIC",
            "Local5_ep44": "MISSING_UNKNOWN_NONZERO_AT_LEAST_120_CALLS",
            "trained_distribution_coverage": "UNADMITTED",
        },
        "remaining_cycle_ledger": remaining_cycle_ledger(validated),
        "four_bottleneck_event_late_scale_model": bottleneck,
        "conditional_dse": dse,
        "standalone_late_scale_evidence": {
            "M33": {"results_per_cycle": 4,
                    "flat_area_um2_at_2ns": m35_audit["m33_area_um2"],
                    "integrated_claim": False},
            "M35": {"results_per_cycle": 8,
                    "flat_area_um2_at_2ns": m35_audit["m35_area_um2"],
                    "integer_multipliers": 0,
                    "maximum_bottleneck_threshold_csd_terms": 4,
                    "strict_flat_result_rate_per_area_vs_M33_exact":
                    payloads["m35_receipt_r3"][
                        "strict_fair_flat_standalone_comparison"][
                        "m35_over_m33_result_rate_per_area_exact"],
                    "integrated_claim": False},
        },
        "external_adapter_boundary": contract["external_comparison_boundary"],
        "admission": {
            "m38_r5_exact_model_only_admission_validated": True,
            "m35_r3_exact_standalone_receipt_validated": True,
            "m35_strict_flat_standalone_density_admitted": True,
            "h67_four_bottleneck_source_work_admitted": True,
            "h67_four_bottleneck_resource_lower_bounds_admitted": True,
            "conditional_dse_math_admitted": True,
            "conditional_m4_projection_is_executable_cycle_evidence": False,
            "m35_r3_integrated_density_admitted": False,
            "local5_full_system_admitted": False,
            "integrated_rtl_admitted": False,
            "integrated_rtl_vcs_admitted": False,
            "integrated_dc_sta_formality_admitted": False,
            "address_timed_memory_admitted": False,
            "sram_dram_realization_admitted": False,
            "trained_coverage_admitted": False,
            "ppa_admitted": False, "power_energy_admitted": False,
            "system_speedup_admitted": False,
            "external_accelerator_comparison_admitted": False,
            "headline_admitted": False, "best_paper_admitted": False,
        },
        "next_gates": [
            "prove an executable four-bottleneck event-coalescing schedule and exact operand/address trace",
            "integrate M35 late scale with the sole shared resource pool and full fixed-point miter",
            "run Synopsys VCS DC STA Formality SAIF PTPX on the integrated top",
            "close address-timed SRAM/DRAM and Local5 ep44 attention/full-system coverage",
        ],
        "claim_boundary": contract["claim_boundary"],
    }
    require(set(result["admission"]).issuperset(FORBIDDEN_ADMISSION_KEYS)
            and all(result["admission"][key] is False
                    for key in FORBIDDEN_ADMISSION_KEYS),
            "M39-r3 forbidden admission opened")
    return result


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite existing M39-r3 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.contract)
    write_output(args.output, result)
    print(json.dumps({
        "status": result["status"], "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
