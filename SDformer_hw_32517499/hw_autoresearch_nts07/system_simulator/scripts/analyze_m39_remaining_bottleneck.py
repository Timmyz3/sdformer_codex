#!/usr/bin/env python3
"""Build fail-closed M39 remaining-bottleneck and sidecar DSE evidence."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    ROOT
    / "hw_autoresearch_nts07/contracts/"
    "m39_remaining_bottleneck_input_contract_r1_20260822.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_contract(path):
    path = Path(path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(
        contract.get("schema") == "m39_remaining_bottleneck_input_contract_v1",
        "unexpected M39 contract schema",
    )
    expected_inputs = {
        "m22_summary",
        "m25_cycle_ledger",
        "m26_factor_lower_bound",
        "m30_system_dse",
        "m32_threshold_carry",
        "m33_math",
        "m33_rtl",
        "m33_vcs_log",
        "m33_dc_admission",
        "m33_dc_area",
        "m35_math",
        "m35_receipt",
        "m35_vcs_log",
        "m35_dc_admission",
        "m35_dc_area",
        "m38_theory",
        "h67_dual_line_contract",
        "h67_operator_transactions",
    }
    require(
        set(contract.get("inputs", {})) == expected_inputs,
        "M39 input population drift",
    )
    payloads = {}
    hashes = {}
    paths = {}
    for name, spec in sorted(contract["inputs"].items()):
        require(
            set(spec) == {"path", "sha256"},
            "M39 input specification drift for {}".format(name),
        )
        source = resolve(spec["path"])
        require(source.is_file(), "M39 input is missing for {}".format(name))
        actual = sha256(source)
        require(
            actual == spec["sha256"],
            "M39 input hash drift for {}".format(name),
        )
        text = source.read_text(encoding="utf-8")
        payloads[name] = json.loads(text) if source.suffix == ".json" else text
        hashes[name] = actual
        paths[name] = str(source)
    return contract, payloads, hashes, paths


def parse_area(text, label):
    match = re.search(r"Total cell area:\s+([0-9.]+)", text)
    require(match is not None, "{} area report is not parseable".format(label))
    return float(match.group(1))


def parse_operator_rows(text):
    return list(csv.DictReader(text.splitlines()))


def validate_sources(contract, payloads, hashes):
    rules = contract["frozen_dse_rules"]
    gates = contract["resource_and_admission_gates"]
    required_rules = {
        "fixed_compute_cycles": 620868243,
        "selected_m30_candidate": "dual256b_independent_output_packed24",
        "selected_m30_local_cycles": 305047198,
        "selected_m30_motion_cycles": 303376924,
        "m30_t10_cycles": 73183500,
        "m38_conditional_t10_cycles": 36591750,
        "m38_conditional_t10_ii": 5,
        "m33_outputs_per_cycle": 4,
        "m33_products_per_output": 20,
        "m33_multiplier_lanes_used": 80,
        "m35_outputs_per_cycle": 8,
        "m35_integer_multipliers": 0,
        "consumer_population_cycles": 105888197,
        "consumer_outputs_per_sample": 30456000,
        "bottleneck_population_cycles": 79630957,
        "bottleneck_outputs_per_sample": 9216000,
        "m4_profiled_population_cycles": 327131854,
        "minimum_saved_cycles_for_candidate": 50000000,
    }
    require(rules == required_rules, "M39 frozen DSE rule drift")
    require(
        gates == {
            "clock_period_ns": 3.0,
            "signed_int8_multiplier_lanes": 96,
            "sram_preferred_kib": 240,
            "sram_hard_cap_kib": 408,
            "sram_row_bytes": 96,
            "sram_banks": 24,
            "sram_read_ports_per_bank": 1,
            "sram_write_ports_per_bank": 1,
            "minimum_system_speedup": 2.7,
            "stretch_system_speedup": 3.0,
            "maximum_integrated_area_delta_fraction": 0.15,
            "accuracy_primary_gate": "BIT_EXACT_TO_FROZEN_INTEGER_REFERENCE",
            "accuracy_fallback_delta_aee_max": 0.02,
            "energy_gate": (
                "SAME_TRACE_PTPX_PLUS_MACRO_ENERGY_NOT_WORSE_THAN_"
                "M38_BASELINE"
            ),
            "m35_r7_formality_status": "PENDING",
        },
        "M39 resource/admission gate drift",
    )

    m22 = payloads["m22_summary"]
    require(
        m22["status"]
        == "PASS_FROZEN_INPUT_PARTIAL_TRANSACTION_LEDGER_NOT_DRAMSIM_OR_SPEEDUP",
        "M39 M22 status drift",
    )
    h67_identity = m22["identities"]["h67_ep35"]
    local_identity = m22["identities"]["local_ep44"]
    require(h67_identity["attention_execution_records"] == 120, "M39 H67 attention drift")
    require(
        h67_identity["attention_coverage_status"]
        == "ABSTRACT_PACKED1_COUNTER_SUMMARY_NOT_PHYSICAL_TRAFFIC",
        "M39 H67 attention qualification drift",
    )
    require(local_identity["attention_execution_records"] == 0, "M39 Local trace drift")
    require(
        local_identity["attention_coverage_status"]
        == "MISSING_FROM_EXECUTION_TRACE_NOT_ZERO_COST",
        "M39 Local missing-attention fail-close drift",
    )
    require(local_identity["profile_identity"]["samples"] == 10, "M39 Local sample drift")

    m25 = payloads["m25_cycle_ledger"]
    require(
        m25["status"] == "PASS_FROZEN_C4_TILING_AND_CYCLE_ENVELOPE_HEADLINE_NO_GO",
        "M39 M25 status drift",
    )
    require(
        m25["attention_completeness"]["Local5"]["speedup"] == "UNKNOWN"
        and m25["attention_completeness"]["Local5"]["minimum_missing_module_calls"] == 120,
        "M39 M25 Local attention fail-close drift",
    )
    require(
        m25["uniform_resource_contract"]["logic_budget"]["int8_multipliers"] == 96
        and m25["uniform_resource_contract"]["sram_budgets_kib"] == [96, 128, 240, 408]
        and m25["fixed_resident_footprint"]["total_bytes"] == 52032,
        "M39 M25 resource contract drift",
    )
    local = m25["compute_envelopes"]["local"]["10"]
    motion = m25["compute_envelopes"]["hybrid"]["10"]
    expected_local = {
        "accelerated_m4_cycles": 54565804,
        "noneligible_plus_qk_cycles": 162059820,
        "m21_fifo4_phase1_incremental_cycles": 6098531,
        "m21_registered_result_bubble_cycles": 738,
        "rqtb_attention_cycles": 3090731,
    }
    expected_motion = {
        "accelerated_m4_cycles": 52733277,
        "noneligible_plus_qk_cycles": 162059820,
        "m21_fifo4_phase1_incremental_cycles": 6260784,
        "m21_registered_result_bubble_cycles": 738,
        "rqtb_attention_cycles": 3090731,
    }
    for key, value in expected_local.items():
        require(local[key] == value, "M39 M25 Local {} drift".format(key))
    for key, value in expected_motion.items():
        require(motion[key] == value, "M39 M25 Motion {} drift".format(key))
    require(
        local["m4_profiled_eligible_cycles"] == rules["m4_profiled_population_cycles"]
        and motion["m4_profiled_eligible_cycles"] == rules["m4_profiled_population_cycles"],
        "M39 M4 population drift",
    )

    m30 = payloads["m30_system_dse"]
    require(m30["schema"] == "m30_resident_stream_system_dse_v2", "M39 M30 schema drift")
    require(
        m30["frozen_resources"]["fixed_compute_cycles"] == rules["fixed_compute_cycles"]
        and m30["frozen_resources"]["shared_non_atlif_local_cycles"] == 225815624
        and m30["frozen_resources"]["shared_non_atlif_motion_cycles"] == 224145350,
        "M39 M30 frozen resource drift",
    )
    m26 = payloads["m26_factor_lower_bound"]
    require(
        m26["schema"] == "m26_atlif_factor_arithmetic_lower_bound_v2"
        and m26["status"]
        == "PASS_CHECKPOINT_BOUND_FACTOR_ARITHMETIC_LOWER_BOUND_TRAINING_REQUIRED_NO_SPEEDUP_CLAIM"
        and m26["identity"]["m25_sha256"] == hashes["m25_cycle_ledger"]
        and m30["identity"]["m26_sha256"] == hashes["m26_factor_lower_bound"],
        "M39 M26/M30 recursive identity drift",
    )
    candidates = {row["name"]: row for row in m30["port_candidates"]}
    require(rules["selected_m30_candidate"] in candidates, "M39 M30 selected candidate missing")
    selected = candidates[rules["selected_m30_candidate"]]
    require(
        selected["local_cycles"] == rules["selected_m30_local_cycles"]
        and selected["motion_cycles"] == rules["selected_m30_motion_cycles"]
        and selected["t10_cycles"] == rules["m30_t10_cycles"]
        and selected["parameter_cold_fill_cycles"] == 74
        and selected["t2_cycles"] == 6048000,
        "M39 M30 selected candidate drift",
    )

    m38 = payloads["m38_theory"]
    require(m38["admission"]["conditional_t10_theory_ledger_admitted"], "M39 M38 theory missing")
    require(not m38["admission"]["system_speedup_admitted"], "M39 M38 system claim opened")
    m38_rows = {row["name"]: row for row in m38["integrated_theory_ledger"]["candidates"]}
    require(
        m38_rows["m38_rst_parallel"]["conditional_t10_steady_ii_cycles"]
        == rules["m38_conditional_t10_ii"]
        and not m38_rows["m38_rst_parallel"]["executable_integrated_cycles_admitted"],
        "M39 M38 conditional-II boundary drift",
    )

    dual = payloads["h67_dual_line_contract"]
    require(dual["schema"] == "h67_dual_line_full_system_contract_v0", "M39 dual-line schema drift")
    require(dual["envelopes"]["fixed_total_cycles_model"] == rules["fixed_compute_cycles"], "M39 fixed cycle drift")
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
    for name, pair in expected_categories.items():
        require(
            (categories[name]["cycles"], categories[name]["eligible_cycles"]) == pair,
            "M39 category ledger drift for {}".format(name),
        )
    require(
        not dual["artifacts"]["selected_binary_tile_vectors"]["available"],
        "M39 product-pattern observability unexpectedly changed",
    )

    operator_rows = parse_operator_rows(payloads["h67_operator_transactions"])
    bottlenecks = [row for row in operator_rows if row["category"] == "bottleneck"]
    require(len(bottlenecks) == 4, "M39 bottleneck operator population drift")
    bottleneck_expected = {
        "sttmultires_unet.resblocks.0.conv1.0": (27399554, 0.16516899305555555),
        "sttmultires_unet.resblocks.0.conv2.0": (9864782, 0.05946651475694445),
        "sttmultires_unet.resblocks.1.conv1.0": (30197101, 0.1820330642361111),
        "sttmultires_unet.resblocks.1.conv2.0": (12169520, 0.07335985677083333),
    }
    for row in bottlenecks:
        require(row["name"] in bottleneck_expected, "M39 unknown bottleneck operator")
        expected_cycles, expected_activity = bottleneck_expected[row["name"]]
        require(
            int(row["activity_cycles_at_config_lanes"]) == expected_cycles
            and abs(float(row["input_activity"]) - expected_activity) < 1e-15
            and row["input_shape_first"] == "[10, 1, 768, 15, 20]"
            and row["output_shape_first"] == "[10, 1, 768, 15, 20]",
            "M39 bottleneck operator identity drift for {}".format(row["name"]),
        )
    require(
        sum(int(row["activity_cycles_at_config_lanes"]) for row in bottlenecks)
        == rules["bottleneck_population_cycles"],
        "M39 bottleneck cycle sum drift",
    )

    m32 = payloads["m32_threshold_carry"]
    require(
        m32["status"]
        == "PASS_H67_EP35_S10_EXACT_RUNTIME_DATAFLOW_REAL_DOMAIN_SEMANTIC_ADMISSION_ONLY",
        "M39 M32 semantic status drift",
    )
    census = m32["candidate_census"]
    require(
        census["semantically_admitted_operators"] == 10
        and census["semantically_admitted_cycles_candidate_population"]
        == rules["consumer_population_cycles"]
        and census["semantically_admitted_outputs_per_sample"]
        == rules["consumer_outputs_per_sample"],
        "M39 M32 candidate census drift",
    )
    require(
        not m32["admission"]["fixed_point_admitted"]
        and not m32["admission"]["system_cycle_admitted"],
        "M39 M32 claim boundary opened",
    )
    balanced_rows = {
        row["line"]: row
        for row in m32["control_charged_cycle_sensitivity"]["rows"]
        if row["variant"] == "balanced_radix20_exact_product"
    }
    require(
        balanced_rows["local"]["event_accumulation_cycles_borrowed"] == 17662220
        and balanced_rows["local"]["late_scale_cycles_arithmetic"] == 7614000
        and balanced_rows["local"]["proportional_frontend_control_cycles"] == 1974013
        and balanced_rows["motion"]["event_accumulation_cycles_borrowed"] == 17069055
        and balanced_rows["motion"]["late_scale_cycles_arithmetic"] == 7614000
        and balanced_rows["motion"]["proportional_frontend_control_cycles"] == 2026532,
        "M39 M32 balanced-radix sensitivity drift",
    )

    m33 = payloads["m33_math"]
    require(
        m33["status"] == "PASS_EXACT_UQ0P24_AND_SIGNED_DIGIT_CROSS_PRODUCT_IDENTITY"
        and m33["admission"]["integer_cross_product_identity_admitted"]
        and not m33["admission"]["rtl_admitted"],
        "M39 M33 math boundary drift",
    )
    rtl = payloads["m33_rtl"]
    require(
        "20 cross" in rtl
        and "use 80 lanes of one 96-lane pool" in rtl
        and "must route it into the M31 pool rather than instantiate a second pool" in rtl,
        "M39 M33 RTL resource contract drift",
    )
    require(
        "M33_UQ_PASS packets=2048" in payloads["m33_vcs_log"]
        and "valid_scalar_products=4608" in payloads["m33_vcs_log"]
        and "digit_reconstruction_checks=8192" in payloads["m33_vcs_log"],
        "M39 M33 VCS evidence drift",
    )
    require(
        "status=PASS_EXPLORATORY_FLAT_FAIR_AREA_DC" in payloads["m33_dc_admission"]
        and "timing_status=MET" in payloads["m33_dc_admission"]
        and "paper_ppa_ready=false" in payloads["m33_dc_admission"],
        "M39 M33 DC admission drift",
    )
    m33_area = parse_area(payloads["m33_dc_area"], "M33")
    require(abs(m33_area - 12997.403898) < 1e-6, "M39 M33 area drift")

    m35 = payloads["m35_math"]
    require(
        m35["status"]
        == "PASS_TEN_CHECKPOINT_THRESHOLDS_EXACT_UP_TO_FOUR_TERM_COMPLEMENT_CSD_SIGNED42",
        "M39 M35 math status drift",
    )
    require(
        len(m35["thresholds"]) == 10
        and min(row["delta"] for row in m35["thresholds"]) == 1
        and max(row["delta"] for row in m35["thresholds"]) == 588
        and max(row["csd_nonzero_terms"] for row in m35["thresholds"]) == 4,
        "M39 M35 threshold/CSD population drift",
    )
    receipt = payloads["m35_receipt"]
    require(
        receipt["math_identity"]["result"][1] == hashes["m35_math"]
        and receipt["vcs_r6"]["sim_log_sha256"] == hashes["m35_vcs_log"]
        and receipt["vcs_r6"]["unstalled_functional_ii"] == 1,
        "M39 M35 receipt identity drift",
    )
    require(
        "M35_PASS packets=5120" in payloads["m35_vcs_log"]
        and "valid_products=23680" in payloads["m35_vcs_log"]
        and "consecutive_full_rate=630" in payloads["m35_vcs_log"],
        "M39 M35 VCS evidence drift",
    )
    require(
        "status=PASS_STANDALONE_COMPLEMENT_CSD8_DC" in payloads["m35_dc_admission"]
        and "integer_multiplier_count=0" in payloads["m35_dc_admission"]
        and "timing_status=MET" in payloads["m35_dc_admission"]
        and "paper_ppa_ready=false" in payloads["m35_dc_admission"],
        "M39 M35 r7 DC admission drift",
    )
    m35_area = parse_area(payloads["m35_dc_area"], "M35")
    require(abs(m35_area - 19633.571938) < 1e-6, "M39 M35 area drift")
    return {
        "local_m25": local,
        "motion_m25": motion,
        "m30_selected": selected,
        "m32_balanced_rows": balanced_rows,
        "bottleneck_rows": bottlenecks,
        "categories": categories,
        "m33_area": m33_area,
        "m35_area": m35_area,
    }


def category_ledger(categories):
    residuals = []
    for name in ("bottleneck", "patch_embed", "ffn_expand", "downsample", "prediction"):
        source = categories[name]
        residuals.append({
            "category": name,
            "total_cycles": source["cycles"],
            "already_m4_eligible_cycles": source["eligible_cycles"],
            "remaining_noneligible_cycles": source["cycles"] - source["eligible_cycles"],
        })
    require(
        sum(row["remaining_noneligible_cycles"] for row in residuals) == 132987740,
        "M39 noneligible category reconciliation failed",
    )
    return residuals


def target_gates(fixed_cycles, ideal_cycles, population_cycles, replacement_cycles):
    rows = []
    for target in (2.7, 3.0):
        target_cycles = fixed_cycles / target
        saving_required = ideal_cycles - target_cycles
        maximum_replacement = population_cycles - saving_required
        rows.append({
            "target_speedup": target,
            "target_cycle_ceiling": target_cycles,
            "saving_required_from_scope": saving_required,
            "maximum_scope_replacement_cycles": maximum_replacement,
            "modeled_replacement_overhead_headroom_cycles": (
                maximum_replacement - replacement_cycles
            ),
            "crosses_target_in_conditional_dse": (
                replacement_cycles <= maximum_replacement
            ),
        })
    return rows


def scope_row(name, line, ideal_cycles, before, event_cycles, late_cycles,
              control_cycles, implementation, fixed_cycles):
    # No overlap credit is admitted.  This conservation equation is the only
    # system-row substitution performed by M39.
    overlap_credit = 0
    replacement = event_cycles + late_cycles + control_cycles - overlap_credit
    savings = before - replacement
    after = ideal_cycles - before + replacement
    require(after + before == ideal_cycles + replacement, "M39 scope conservation failed")
    return {
        "scope": name,
        "line": line,
        "late_scale_implementation": implementation,
        "before_cycles": before,
        "replacement": {
            "event_accumulation_cycles": event_cycles,
            "late_scale_cycles": late_cycles,
            "frontend_control_cycles": control_cycles,
            "overlap_credit_cycles": overlap_credit,
            "overlap_policy": (
                "SERIAL_SUM_CONSERVATIVE; max(event,late) or any sidecar overlap is "
                "forbidden until independent ports, accumulator buffering, VCS, and "
                "the integrated scheduler prove it"
            ),
            "total_cycles": replacement,
        },
        "savings_cycles": savings,
        "minimum_50m_saving_pass": savings >= 50000000,
        "m38_ideal_before_scope_substitution_cycles": ideal_cycles,
        "conditional_cycles_after_substitution": after,
        "conditional_speedup_vs_fixed": fixed_cycles / after,
        "conservation_equation": (
            "after = m38_ideal - before + event + late + control - overlap_credit"
        ),
        "bucket_disjointness": (
            "M38 changes only the 73,183,500-cycle T10 ATLIF bucket; this scope "
            "is drawn from the M7/M25 noneligible operator bucket, so it is not "
            "subtracted from T10 again"
        ),
        "target_gates": target_gates(fixed_cycles, ideal_cycles, before, replacement),
        "claim": "CONDITIONAL_COMPUTE_DSE_NOT_EXECUTABLE_OR_MEASURED_SYSTEM_CYCLES",
    }


def build_dse(contract, validated):
    rules = contract["frozen_dse_rules"]
    fixed_cycles = rules["fixed_compute_cycles"]
    selected = validated["m30_selected"]
    ideals = {
        "Local": selected["local_cycles"] - rules["m30_t10_cycles"]
        + rules["m38_conditional_t10_cycles"],
        "Motion": selected["motion_cycles"] - rules["m30_t10_cycles"]
        + rules["m38_conditional_t10_cycles"],
    }
    require(ideals == {"Local": 268455448, "Motion": 266785174}, "M39 M38 ideal drift")

    m25_rows = {
        "Local": validated["local_m25"],
        "Motion": validated["motion_m25"],
    }
    m32_rows = {
        "Local": validated["m32_balanced_rows"]["local"],
        "Motion": validated["m32_balanced_rows"]["motion"],
    }
    full_rows = []
    bottleneck_rows = []
    for line in ("Local", "Motion"):
        m32_row = m32_rows[line]
        full_event = m32_row["event_accumulation_cycles_borrowed"]
        full_control = m32_row["proportional_frontend_control_cycles"]
        for implementation, outputs_per_cycle in (("M33_shared96", 4), ("M35_zero_mul_sidecar", 8)):
            full_late = int(math.ceil(
                rules["consumer_outputs_per_sample"] / float(outputs_per_cycle)
            ))
            full_rows.append(scope_row(
                "ten_semantically_admitted_consumers",
                line,
                ideals[line],
                rules["consumer_population_cycles"],
                full_event,
                full_late,
                full_control,
                implementation,
                fixed_cycles,
            ))

        m25_row = m25_rows[line]
        speed = m25_row["effective_m4_speed"]
        bottleneck_event = int(math.ceil(
            rules["bottleneck_population_cycles"] / float(speed)
        ))
        bottleneck_control = int(math.ceil(
            m25_row["m21_fifo4_phase1_incremental_cycles"]
            * rules["bottleneck_population_cycles"]
            / float(rules["m4_profiled_population_cycles"])
        ))
        for implementation, outputs_per_cycle in (("M33_shared96", 4), ("M35_zero_mul_sidecar", 8)):
            bottleneck_late = int(math.ceil(
                rules["bottleneck_outputs_per_sample"] / float(outputs_per_cycle)
            ))
            bottleneck_rows.append(scope_row(
                "four_bottleneck_conv3x3",
                line,
                ideals[line],
                rules["bottleneck_population_cycles"],
                bottleneck_event,
                bottleneck_late,
                bottleneck_control,
                implementation,
                fixed_cycles,
            ))
    return {
        "selected_m30_anchor": {
            "name": rules["selected_m30_candidate"],
            "local_cycles": selected["local_cycles"],
            "motion_cycles": selected["motion_cycles"],
            "qualification": (
                "best M30 r3 dual256b candidate; the 384b candidate is 24 cycles "
                "slower on each line and is not the M39 primary anchor"
            ),
        },
        "m38_conditional_ideal": {
            "substitution": "M30 cycles - 73,183,500 T10 + 36,591,750 T10",
            "local_cycles": ideals["Local"],
            "local_speedup_vs_fixed": fixed_cycles / ideals["Local"],
            "motion_cycles": ideals["Motion"],
            "motion_speedup_vs_fixed": fixed_cycles / ideals["Motion"],
            "claim": "THEORY_ONLY_NOT_INTEGRATED_RTL_OR_MEASURED_SYSTEM_SPEEDUP",
        },
        "scope_alternatives_not_additive": True,
        "four_bottleneck_rows": bottleneck_rows,
        "ten_consumer_rows": full_rows,
    }


def build(contract_path=DEFAULT_CONTRACT):
    contract, payloads, hashes, paths = load_contract(contract_path)
    validated = validate_sources(contract, payloads, hashes)
    categories = category_ledger(validated["categories"])
    noneligible = sum(row["remaining_noneligible_cycles"] for row in categories)
    qk_cycles = (
        validated["categories"]["attention_q_projection"]["cycles"]
        + validated["categories"]["attention_k_projection"]["cycles"]
    )
    require(noneligible + qk_cycles == 162059820, "M39 noneligible+QK reconciliation failed")
    dse = build_dse(contract, validated)

    line_shared = []
    for line, row, expected in (
        ("Local", validated["local_m25"], 225815624),
        ("Motion", validated["motion_m25"], 224145350),
    ):
        parts = {
            "accelerated_m4_cycles": row["accelerated_m4_cycles"],
            "noneligible_plus_qk_cycles": row["noneligible_plus_qk_cycles"],
            "m21_frontend_control_cycles": row["m21_fifo4_phase1_incremental_cycles"],
            "registered_bubble_cycles": row["m21_registered_result_bubble_cycles"],
            "h67_rqtb_attention_cycles": row["rqtb_attention_cycles"],
        }
        require(sum(parts.values()) == expected, "M39 {} shared ledger mismatch".format(line))
        line_shared.append({
            "line": line,
            "shared_non_atlif_cycles": expected,
            "parts": parts,
            "qualification": (
                "H67 profile100 compute-model ledger under Local/Motion mechanism; "
                "not a Local5 ep44 full-system ledger"
            ),
        })

    bottleneck_operator_rows = []
    for row in sorted(validated["bottleneck_rows"], key=lambda item: item["name"]):
        bottleneck_operator_rows.append({
            "name": row["name"],
            "operator": row["operator"],
            "input_shape": json.loads(row["input_shape_first"]),
            "output_shape": json.loads(row["output_shape_first"]),
            "input_activity": float(row["input_activity"]),
            "baseline_activity_cycles": int(row["activity_cycles_at_config_lanes"]),
            "im2col_per_invocation": {"M": 3000, "K": 6912, "N": 768},
        })

    m33_area = validated["m33_area"]
    m35_area = validated["m35_area"]
    m35_density = (8.0 / m35_area) / (4.0 / m33_area)
    result = {
        "schema": "m39_remaining_bottleneck_v1",
        "status": "PASS_FAIL_CLOSED_REMAINING_LEDGER_AND_CONDITIONAL_SIDECAR_DSE",
        "identity": {
            "contract": str(Path(contract_path).resolve()),
            "contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
            "verified_input_paths": paths,
        },
        "attention_and_trace_completeness": {
            "h67": (
                "120 attention rows exist only as abstract packed1 summaries; H67 "
                "RQTB has a 3,090,731-cycle compute anchor, not closed physical traffic"
            ),
            "local5_ep44": (
                "0 attention execution rows means MISSING UNKNOWN NONZERO, not zero "
                "cost; at least 120 calls and Local5 system cycles/speedup remain unknown"
            ),
            "local_motion_name_boundary": (
                "Local and Motion below are mechanisms evaluated on the same frozen H67 "
                "profile100 ledger; Local is not Local5 ep44"
            ),
        },
        "remaining_cycle_ledger": {
            "fixed_compute_cycles": contract["frozen_dse_rules"]["fixed_compute_cycles"],
            "shared_non_atlif_by_line": line_shared,
            "noneligible_plus_qk_decomposition": {
                "noneligible_operator_cycles": noneligible,
                "q_projection_cycles": 14536040,
                "k_projection_cycles": 14536040,
                "qk_cycles": qk_cycles,
                "total_cycles": noneligible + qk_cycles,
                "noneligible_categories": categories,
            },
            "independent_cycle_reduction_ceilings": [
                {"scope": "four_bottleneck_conv3x3", "cycles": 79630957, "can_save_50m_alone": True},
                {"scope": "qk_plus_rqtb_attention", "cycles": 32162811, "can_save_50m_alone": False},
                {"scope": "patch_embed_remaining", "cycles": 27099543, "can_save_50m_alone": False},
                {"scope": "ffn_expand_remaining", "cycles": 17474490, "can_save_50m_alone": False},
                {"scope": "downsample_remaining", "cycles": 8691053, "can_save_50m_alone": False},
                {"scope": "prediction_remaining", "cycles": 91697, "can_save_50m_alone": False},
            ],
            "bn_and_control_observability": (
                "only the line-level M21 incremental control totals 6,098,531/6,260,784 "
                "are cycle-defined; an independent BN-only split is not observable"
            ),
            "bitplane_materialization": {
                "q24_output_bytes": 4383720000,
                "bitpack_output_bytes": 182655000,
                "output_payload_reduction": 24.0,
                "boundary_payload_reduction": 3.5555555555555554,
                "cycle_credit_admitted": False,
                "reason": (
                    "M22/M23 transport and bank ticks are not system cycles; address-timed "
                    "overlap and semantic consumer integration are still missing"
                ),
            },
            "bottleneck_operator_census": bottleneck_operator_rows,
        },
        "conditional_dse": dse,
        "late_scale_architecture_alternatives": [
            {
                "name": "M33_shared96_generic_UQ0p24",
                "datapath": (
                    "event-accumulate W*b into Acc32; decompose four Acc32 and one UQ0.24 "
                    "threshold into balanced radix-128 digits; use 80 lanes of the sole "
                    "96-lane INT8 pool; recombine signed56, then bias/RNE/saturate"
                ),
                "outputs_per_cycle": 4,
                "additional_int8_multipliers": 0,
                "pool_contention": (
                    "nonzero: late scale occupies 80/96 existing lanes and must arbitrate "
                    "against M38 stage1, T2, and other operator clients"
                ),
                "standalone_flat_area_um2_at_2ns": m33_area,
                "evidence": (
                    "standalone VCS 2048 packets and exploratory flat DC timing MET; no "
                    "same-top integration, Formality, PTPX, or system schedule"
                ),
            },
            {
                "name": "M35_parallel_complement_CSD_sidecar",
                "datapath": (
                    "for each frozen threshold q=2^24-delta, form (Acc<<24) minus up to "
                    "four signed shifted Acc terms; eight independent outputs per cycle"
                ),
                "outputs_per_cycle": 8,
                "additional_int8_multipliers": 0,
                "pool_contention": (
                    "none arithmetically, but overlap credit stays zero until input/output "
                    "ports and accumulator buffers are independent and integrated VCS/system "
                    "scheduling proves simultaneous service"
                ),
                "frozen_h67_threshold_delta_range": [1, 588],
                "maximum_csd_terms": 4,
                "standalone_area_um2_at_2ns": m35_area,
                "standalone_throughput_density_vs_flat_m33": m35_density,
                "latest_r7_formality": "PENDING",
                "evidence": (
                    "M35 math covers the ten H67 thresholds; VCS r6 II=1x8; fair DC r7 "
                    "timing MET and zero multipliers; r7 Formality, integration, and Local5 "
                    "threshold coverage remain open"
                ),
            },
        ],
        "resource_bandwidth_sram_contract": {
            "common": {
                "sole_signed_int8_multiplier_lanes": 96,
                "frequency_mhz": 333.333333,
                "sram_banks": 24,
                "sram_row_bytes": 96,
                "ports_per_bank": "1R1W",
                "fixed_resident_bytes": 52032,
                "preferred_total_sram_kib": 240,
                "hard_total_sram_kib": 408,
                "weight_stream_requirement_bytes_per_cycle": 96,
            },
            "bottleneck_conv_window": {
                "one_timestep_bitplane_bytes": 28800,
                "three_row_line_buffer_bytes": 5760,
                "local_single_buffer_plus_fixed_bytes": 86592,
                "double_or_motion_buffer_plus_fixed_bytes": 115392,
            },
            "prosperity_probe_tile_m256_k16_n96": {
                "activation_bitmap_bytes": 512,
                "weight_tile_bytes": 1536,
                "acc32_tile_bytes": 98304,
                "prefix_index_bytes": 256,
                "residual_bitmap_bytes": 512,
                "conv_line_buffer_bytes": 5760,
                "incremental_bytes": 106880,
                "with_fixed_resident_bytes": 158912,
                "fits_240kib": True,
                "claim": "PROPOSED_RTL_SHAPE_ONLY_PRODUCT_DENSITY_AND_CYCLES_UNOBSERVED",
            },
        },
        "prosperity_phi_adapter_assessment": {
            "Prosperity": {
                "primary_source": "https://arxiv.org/abs/2503.03379",
                "official_code": "https://github.com/dubcyfor3/Prosperity",
                "mechanism": (
                    "runtime subset/prefix reuse between binary activation rows; official "
                    "simulator defaults use M=256, K=16, N=128 and eight popcount units"
                ),
                "fit": (
                    "the 3x3 bottleneck im2col M=3000,K=6912,N=768 is large enough, and "
                    "threshold carry makes its amplitude input binary without accuracy loss"
                ),
                "blocking_observability": (
                    "only aggregate activity 5.95%-18.20% is frozen; exact binary im2col "
                    "rows, subset forest, product density, detector cycles, and metadata are "
                    "absent because selected_binary_tile_vectors.npz is unavailable"
                ),
                "go_gate": (
                    "after detector/issue/memory overhead, four-bottleneck replacement must "
                    "be <=29,630,957 cycles to save 50M; do not infer product density from "
                    "bit density"
                ),
            },
            "Phi": {
                "primary_source": "https://arxiv.org/abs/2505.10909",
                "mechanism": (
                    "128 predefined 16-bit patterns, offline pattern-weight products, and a "
                    "signed sparse residual; pattern selection and pattern-aware fine-tuning"
                ),
                "fit": (
                    "K=16 partitions match the bottleneck dimensions, but the current H67 "
                    "trace has no pattern histogram and Local5 has no independent calibration"
                ),
                "risk": (
                    "precomputed products trade computation for SRAM/DRAM traffic; the Phi "
                    "paper reports this as a central issue, while our address-timed memory "
                    "model is not closed and fine-tuning changes the accuracy contract"
                ),
                "go_gate": (
                    "require calibration/test pattern coverage, residual density, full PWP "
                    "traffic, and valid825 accuracy before RTL; otherwise NO-GO"
                ),
            },
            "qk_attention_conclusion": (
                "Q/K plus the H67 attention anchor totals only 32,162,811 cycles, so even "
                "perfect Prosperity/Phi elimination cannot independently save 50M"
            ),
        },
        "go_no_go_matrix": {
            "VCS": [
                "full threshold-carry consumer pipeline miter including Conv padding/window order, bias, RNE, and saturation",
                "Local and Motion state/tag identity, signed 0->1/+W and 1->0/-W updates, reset and sequence boundaries",
                "single-pool M33 arbitration against M38/T2 and M35 independent-port simultaneous-service coverage",
                "long ready/valid stalls, accumulator hazards, FIFO full/pop/push, and zero double-retirement",
            ],
            "DC_STA_Formality": [
                "one integrated top with exactly one 96-lane INT8 pool at 3.000ns and identical hierarchy/constraints",
                "M33 and M35 same-top A/B; M35 r7 Formality must pass before its DC number is admitted",
                "GO only if setup/hold MET, zero unintended multipliers, Formality all compare points pass, and integrated area delta <=15%",
            ],
            "PTPX_memory": [
                "SAIF from the same H67 and Local ordered traces, not random-only activity",
                "include SRAM macro/CACTI and address-timed DRAM traffic; GO only if same-trace energy is not worse than M38 baseline",
                "preferred SRAM <=240KiB and hard NO-GO above 408KiB under the frozen 24-bank 96B-row organization",
            ],
            "accuracy": [
                "primary GO is bit-exact frozen integer inference including threshold quantization, bias, RNE, and saturation",
                "if checkpoint re-quantization is unavoidable, valid825 delta AEE must be <=0.02 under the identical evaluator",
                "Local5 requires ep44 threshold census, attention trace, independent fixed-point miter, and valid825; H67 evidence cannot substitute",
            ],
            "cycle_gates": [
                "candidate must save at least 50,000,000 cycles in the conserved ledger",
                "use each row's pre-registered 2.7x/3.0x replacement ceiling; no overlap credit before integrated proof",
                "NO-GO if full ten-consumer replacement exceeds 67,383,950/69,054,224 cycles for Local/Motion 2.7x",
                "NO-GO for 3x if full ten-consumer replacement exceeds 44,388,830/46,059,104 cycles",
            ],
        },
        "admission": {
            "remaining_cycle_decomposition_admitted": True,
            "conditional_h67_compute_dse_admitted": True,
            "m32_h67_real_domain_semantics_admitted": True,
            "m33_standalone_arithmetic_vcs_dc_observed": True,
            "m35_h67_threshold_math_vcs_r6_dc_r7_observed": True,
            "m35_r7_formality_admitted": False,
            "integrated_rtl_admitted": False,
            "executable_integrated_cycles_admitted": False,
            "address_timed_memory_admitted": False,
            "accuracy_admitted": False,
            "power_energy_admitted": False,
            "local5_full_system_admitted": False,
            "system_speedup_admitted": False,
            "headline_admitted": False,
        },
        "claim_boundary": {
            "permitted": (
                "frozen H67 profile100 cycle decomposition; disjoint-bucket conservation; "
                "conditional M38 plus threshold-carry DSE; standalone M33/M35 evidence; "
                "resource and admission gates"
            ),
            "forbidden": (
                "measured Local/Motion system speedup, Local5 zero attention, Local5 full-system "
                "cycles, overlap credit, double-counted M38/consumer savings, executable "
                "integrated cycles, FPS, accuracy, paper PPA, power, energy, or headline"
            ),
        },
    }
    return result


def write_output(path, payload):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite existing M39 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.contract)
    write_output(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
